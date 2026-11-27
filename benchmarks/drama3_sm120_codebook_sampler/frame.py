"""Fixed-input, real-weight, full-frame sampler screen with dense 64-position attention."""

import contextlib
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.sampling_kernels import CodebookSampler
from sglang_omni.models.fishaudio_drama3.weights import (
    load_audio_decoder_weights,
    AUDIO_DECODER_PREFIXES,
)

root = Path("/models/drama3")
work = Path("/data/drama3-drain")


@torch.inference_mode()
def main():
    torch.manual_seed(490)
    Drama3EngineBuilder(context_length=16384, max_running_requests=4).pre_infra_setup(
        str(root)
    )
    publish(ServerArgs(model_path=str(root), device="cuda"), role="scheduler")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29792",
        backend="nccl",
    )
    initialize_model_parallel()
    cfg = FishOmniConfig(**json.loads((root / "config.json").read_text()))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        decoder = AudioDecoder(cfg).eval()
    weights = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with contextlib.ExitStack() as stack:
        files = {}

        def parameters():
            for key, file in weights.items():
                if key.startswith(AUDIO_DECODER_PREFIXES):
                    if file not in files:
                        files[file] = stack.enter_context(
                            safe_open(root / file, framework="pt", device="cpu")
                        )
                    yield key, files[file].get_tensor(key)

        load_audio_decoder_weights(decoder, parameters())
    report = {}
    for n in [1, 4]:
        heads, kv_heads, dim = 16, 2, 256
        cache = [torch.randn(13, n, kv_heads, 64, dim, device="cuda") for _ in range(2)]

        def attention(layer, q, k, v):
            key, value = cache[0][layer], cache[1][layer]
            key[:, :, -1].copy_(k.reshape(n, kv_heads, dim))
            value[:, :, -1].copy_(v.reshape(n, kv_heads, dim))
            return F.scaled_dot_product_attention(
                q.reshape(n, heads, 1, dim), key, value, enable_gqa=True
            ).reshape(n, -1)

        logits = torch.randn(
            n,
            decoder.main_token_candidate_ids.numel(),
            dtype=torch.float32,
            device="cuda",
        )
        hidden = torch.randn(n, 2048, device="cuda")
        frames = torch.randint(0, 4096, (n, 11), device="cuda")
        frames[:, 0] = cfg.semantic_start_token_id + frames[:, 1]
        pos = torch.full((n,), 63, device="cuda", dtype=torch.int64)
        recent = torch.full((n, 16), -1, device="cuda", dtype=torch.int64)
        temp = torch.full((n,), 0.8, device="cuda", dtype=torch.float32)
        topk = torch.full((n,), 1, device="cuda", dtype=torch.int64)
        topp = torch.full((n,), 0.9, device="cuda", dtype=torch.float32)

        def call():
            return decoder.decode_step(
                logits, hidden, pos, frames, recent, temp, topk, topp, attention
            )

        sampler = CodebookSampler(step=decoder.sampling_step, seed=490)
        decoder.sampler = None
        ref = call().clone()
        decoder.sampler = sampler
        check = call().clone()
        equal = (ref == check).sum().item()
        print("top1", n, equal, ref.numel(), flush=True)
        topk.fill_(30)
        graphs = []
        for impl in [None, sampler]:
            decoder.sampler = impl
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = call()
            graphs.append((g, out))
        samples = [[], []]
        for rep in range(12):
            for j in [0, 1] if rep % 2 == 0 else [1, 0]:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(30):
                    graphs[j][0].replay()
                b.record()
                b.synchronize()
                samples[j].append(a.elapsed_time(b) / 30)
        report[n] = dict(
            top1_equal=equal,
            top1_elements=ref.numel(),
            samples_ms=samples,
            medians_ms=list(map(statistics.median, samples)),
        )
        (work / "frame_results.json").write_text(json.dumps(report, indent=2))
        print(n, report[n]["medians_ms"], flush=True)
        if n == 4:
            for j, label in enumerate(["torch", "triton"]):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(5):
                        graphs[j][0].replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(work / (label + ".json.gz")))


if __name__ == "__main__":
    main()
