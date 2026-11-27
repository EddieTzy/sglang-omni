import argparse
import contextlib
import json
import statistics
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


@torch.inference_mode()
def main(model, output):
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
        distributed_init_method="tcp://127.0.0.1:29761",
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
    report = []
    for m in [1, 4, 8, 16, 32]:
        samples = []
        for rep in range(6):
            h = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
            l = torch.randn(m, 256, device="cuda")
            v, ii = l.topk(8)
            samples.append((h, StandardTopKOutput(v.softmax(-1), ii.int(), l)))

        def call(pair):
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

        ref = [call(p).clone() for p in samples]
        graphs = []
        for bm, bn, g, st in [
            (64, 128, 32, 3),
            (16, 32, 1, 2),
            (16, 64, 1, 2),
            (16, 128, 1, 2),
            (32, 64, 1, 3),
            (16, 64, 16, 3),
        ]:
            tc = dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=128,
                GROUP_SIZE_M=g,
                num_warps=4,
                num_stages=st,
            )
            with override_config(tc):
                test = [call(p) for p in samples]
                err = max(
                    (
                        (t.float() - r.float()).norm(dim=-1)
                        / r.float().norm(dim=-1).clamp_min(1e-8)
                    )
                    .max()
                    .item()
                    for t, r in zip(test, ref)
                )
                assert all(torch.isfinite(t).all() for t in test)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = [call(p) for p in samples]
                graph.replay()
                torch.cuda.synchronize()
                graphs.append(
                    (graph, outputs, dict(m=m, config=tc, error=err, times=[]))
                )
        for rep in range(8):
            for graph, _, row in graphs if rep % 2 == 0 else reversed(graphs):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                events[0].record()
                for _ in range(12):
                    graph.replay()
                events[1].record()
                events[1].synchronize()
                row["times"].append(events[0].elapsed_time(events[1]) * 1000 / 72)
        for _, __, row in graphs:
            row["median_us"] = statistics.median(row["times"])
            report.append(row)
        output.write_text(json.dumps(report, indent=2))
        print(
            m,
            [
                (
                    r["config"]["BLOCK_SIZE_N"],
                    round(r["median_us"], 1),
                    round(r["error"], 5),
                )
                for _, __, r in graphs
            ],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.model, args.output)
