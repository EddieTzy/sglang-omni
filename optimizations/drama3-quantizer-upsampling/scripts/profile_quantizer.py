import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.stages import load_codec


@torch.inference_mode()
def main():
    output = Path(os.environ["PROFILE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; preserve the existing service")
    torch.cuda.set_per_process_memory_fraction(0.035)
    torch.manual_seed(42)
    codec = load_codec("/models/drama3", "cuda")
    codec.fold_weight_norm()
    decoder = ChannelsLastDecoder(codec, dtype=torch.float16)
    optimized = decoder.quantizer_upsample
    decoder.quantizer_upsample = codec.quantizer.upsample
    captured = {}
    handles = []
    for index, block in enumerate(codec.quantizer.upsample):

        def capture(module, arguments, block_index=index):
            captured[block_index] = arguments[0].detach().clone()

        handles.append(block[0].register_forward_pre_hook(capture))
    codes_path = Path("/validation/datasets/codec-codes/en-0.npy")
    codes = (
        torch.from_numpy(np.load(codes_path, allow_pickle=False))
        .to("cuda")
        .unsqueeze(0)[..., :32]
    )
    decoder(codes.clone())
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    report = {
        "scope": "Shared-GPU operator-path diagnosis, not an exclusive performance benchmark or service scheduling trace",
        "service_stopped": False,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "free_bytes_at_start": free_bytes,
        "total_bytes": total_bytes,
        "source_sha256": hashlib.sha256(
            Path(
                "/validation/quantizer/sglang_omni/models/fishaudio_drama3/channels_last_decoder.py"
            ).read_bytes()
        ).hexdigest(),
        "codes_sha256": hashlib.sha256(codes_path.read_bytes()).hexdigest(),
        "codes_shape": list(codes.shape),
        "profile_iterations": 2,
        "warmup_iterations": 3,
        "cases": [],
    }
    for index, block in enumerate(codec.quantizer.upsample):
        reference = block[0]
        candidate = optimized[index][0]
        for batch_size in (1, 16, 64):
            hidden = captured[index].repeat(batch_size, 1, 1)
            expected = reference(hidden)
            actual = candidate(hidden)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            case = {
                "block": index,
                "input_shape": list(hidden.shape),
                "output_shape": list(actual.shape),
                "maximum_error": float((actual - expected).abs().max()),
                "variants": {},
            }
            for variant, module in (("baseline", reference), ("candidate", candidate)):
                for _ in range(3):
                    module(hidden)
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profiler:
                    for iteration in range(2):
                        with record_function(
                            f"{variant}/block{index}/batch{batch_size}/iteration{iteration}"
                        ):
                            module(hidden)
                    torch.cuda.synchronize()
                trace_path = output / f"block{index}-b{batch_size}-{variant}.json"
                profiler.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
                kernels = [
                    event
                    for event in trace["traceEvents"]
                    if event.get("cat") == "kernel" and event.get("ph") == "X"
                ]
                kernel_counts = Counter(event["name"] for event in kernels)
                operations = [
                    {
                        "name": item.key,
                        "count": item.count,
                        "self_cpu_us": item.self_cpu_time_total,
                        "self_device_us": item.self_device_time_total,
                        "input_shapes": item.input_shapes,
                    }
                    for item in profiler.key_averages(group_by_input_shape=True)
                ]
                with gzip.open(str(trace_path) + ".gz", "wb") as compressed:
                    compressed.write(trace_path.read_bytes())
                trace_path.unlink()
                case["variants"][variant] = {
                    "trace": trace_path.name + ".gz",
                    "trace_sha256": hashlib.sha256(
                        Path(str(trace_path) + ".gz").read_bytes()
                    ).hexdigest(),
                    "kernels_total": len(kernels),
                    "kernel_counts": dict(kernel_counts),
                    "kernel_time_sum_us_shared_gpu": sum(
                        event["dur"] for event in kernels
                    ),
                    "operations": operations,
                }
            report["cases"].append(case)
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(report["cases"]),
                "peak_allocated_mib": report["peak_allocated_mib"],
            }
        )
    )


if __name__ == "__main__":
    main()
