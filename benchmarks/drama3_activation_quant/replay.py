import json, statistics, contextlib
from pathlib import Path
import torch
from safetensors import safe_open
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts, override_config
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

work = Path("/data/drama3-actq-1127")
model = Path("/models/drama3")


@torch.inference_mode()
def main():
    from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder

    Drama3EngineBuilder(context_length=16384, max_running_requests=16).pre_infra_setup(
        str(model)
    )
    publish(
        ServerArgs(model_path=str(model), device="cuda", moe_runner_backend="triton"),
        role="scheduler",
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29787",
        backend="nccl",
    )
    initialize_model_parallel()
    torch.manual_seed(71)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with contextlib.ExitStack() as stack:
        fs = {}

        def weight(e, proj, suffix):
            k = f"model.language_model.layers.0.mlp.experts.{e}.{proj}_proj.{suffix}"
            fn = mapping[k]
            if fn not in fs:
                fs[fn] = stack.enter_context(
                    safe_open(model / fn, framework="pt", device="cpu")
                )
            return fs[fn].get_tensor(k)

        def load(projs, suffix):
            return torch.stack(
                [
                    torch.cat([weight(e, p, suffix) for p in projs], dim=0)
                    for e in range(256)
                ]
            ).cuda()

        up = load(["gate", "up"], "weight")
        down = load(["down"], "weight")
        us = load(["gate", "up"], "weight_scale_inv")
        ds = load(["down"], "weight_scale_inv")
    config = MoeRunnerConfig(
        num_experts=256,
        num_local_experts=256,
        hidden_size=2048,
        intermediate_size_per_partition=512,
        top_k=8,
        inplace=False,
        gate_up_interleaved=False,
    )
    import importlib
    from unittest.mock import patch

    mod = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )
    activation = mod.silu_and_mul
    quant = mod.per_token_group_quant
    invoke = mod.invoke_fused_moe_kernel
    result = {}
    for m in [1, 4, 8, 16]:
        pairs = []
        for _ in range(6):
            h = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
            l = torch.randn(m, 256, device="cuda")
            v, i = l.topk(8)
            pairs.append((h, StandardTopKOutput(v.softmax(-1), i.int(), l)))

        def expert(pair):
            return fused_experts(
                pair[0],
                up,
                down,
                pair[1],
                config,
                use_fp8_w8a8=True,
                w1_scale=us,
                w2_scale=ds,
                block_shape=[128, 128],
            )

        xs = []

        def record(x, out, **kwargs):
            xs.append(x.clone())
            return activation(x, out, **kwargs)

        with patch.object(mod, "silu_and_mul", record):
            ref = [expert(p).clone() for p in pairs]
        assert len(xs) == 6 and xs[0].shape == (m * 8, 1024)

        def act(x):
            return activation(
                x, torch.empty_like(x[:, :512], memory_format=torch.contiguous_format)
            )

        def split(x):
            return quant(act(x), group_size=128)

        def fused(x):
            return quant(
                x, group_size=128, fuse_silu_and_mul=True, silu_and_mul_mode="dense"
            )

        for x in xs:
            a, b = split(x)
            c, d = fused(x)
            assert torch.equal(
                a.view(torch.uint8), c.view(torch.uint8)
            ) and torch.equal(b, d)
        pending = []

        def capture(x, out, **kwargs):
            assert not any(v is not None for v in kwargs.values()), kwargs
            pending.append(fused(x))

        def project(*args, **kwargs):
            if pending:
                assert args[0].shape[-1] == 512 and args[4] is None
                args = list(args)
                args[0], args[4] = pending.pop()
            return invoke(*args, **kwargs)

        def candidate(pair):
            with (
                patch.object(mod, "silu_and_mul", capture),
                patch.object(mod, "invoke_fused_moe_kernel", project),
            ):
                out = expert(pair)
            assert not pending
            return out

        for pair, r in zip(pairs, ref):
            assert torch.equal(candidate(pair), r)
        graphs = {}
        for key, fn, inputs in [
            ("activation", act, xs),
            ("split", split, xs),
            ("fused", fused, xs),
            ("experts_base", expert, pairs),
            ("experts_fused", candidate, pairs),
        ]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = [fn(x) for x in inputs]
            graphs[key] = (graph, outputs)
        for graph, _ in graphs.values():
            for _ in range(30):
                graph.replay()
        torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        for rep in range(12):
            for k in list(graphs) if rep % 2 == 0 else reversed(graphs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(64):
                    graphs[k][0].replay()
                end.record()
                end.synchronize()
                samples[k].append(start.elapsed_time(end) * 1000 / (64 * 6))
        result[m] = {
            "samples_us": samples,
            "medians_us": {k: statistics.median(t) for k, t in samples.items()},
        }
        print(m, result[m]["medians_us"], flush=True)
        (work / "result.json").write_text(json.dumps(result, indent=2))
        if m == 4:
            for k in ["experts_base", "experts_fused"]:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(8):
                        graphs[k][0].replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(work / (k + ".json.gz")))


if __name__ == "__main__":
    main()
