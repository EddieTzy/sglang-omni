from pathlib import Path
import hashlib
import json
import shutil
import sys

root = Path(__file__).resolve().parent
public = root / "public"
public.mkdir(exist_ok=True)
arms = sys.argv[1:] or ["a1", "b1", "fixed-a1", "b2", "a2", "default2"]
keys = [
    "total_requests",
    "completed_requests",
    "failed_requests",
    "max_token_hits",
    "latency_mean_s",
    "audio_duration_mean_s",
    "rtf_mean",
    "audio_throughput_s_per_s",
    "audio_ttfp_mean_s",
    "audio_ttfp_median_s",
    "audio_ttfp_p95_s",
    "audio_ttfp_p99_s",
]
metrics = {}
for arm in arms:
    folder = root / arm
    result = json.loads((folder / "timed/speed_results.json").read_text())
    metrics[arm] = {k: result["summary"].get(k) for k in keys}
    metrics[arm]["max_ttfa_s"] = max(
        r.get("audio_ttfp_s") or 0 for r in result["per_request"]
    )
    out = public / arm
    out.mkdir(exist_ok=True)
    metrics[arm]["sample_set_sha256"] = hashlib.sha256(
        json.dumps(
            sorted((r["id"], r["text"]) for r in result["per_request"]),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    selected = [
        "id",
        "completion_tokens",
        "prompt_tokens",
        "audio_ttfp_s",
        "is_success",
        "audio_duration_s",
        "latency_s",
        "rtf",
        "finish_reason",
        "audio_chunk_count",
        "first_audio_payload_bytes",
        "output_tokens",
    ]
    (out / "requests.json").write_text(
        json.dumps(
            [{k: r[k] for k in selected if k in r} for r in result["per_request"]],
            indent=2,
        )
    )
    for name in [
        "launch.json",
        "bench.json",
        "request-timeline.json.gz",
        "phases.json",
        "http-events.json",
    ]:
        if (folder / name).is_file():
            shutil.copyfile(folder / name, out / name)
    logs = (folder / "server.log").read_text(errors="replace")
    (out / "activation.txt").write_text(
        "\n".join(
            line
            for line in logs.splitlines()
            if any(
                k in line
                for k in [
                    "CPU thread fallback plan",
                    "Drama 3 codec selection:",
                    "Drama 3 codec activation:",
                    "Drama 3 state arena:",
                    "Detected fp8 checkpoint",
                    "window fallback",
                    "fallback to window",
                ]
            )
        )
        + "\n"
    )
(public / "serving.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
