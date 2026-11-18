import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    comparison = json.loads(arguments.input.read_text())
    report = {
        "run": comparison["run"],
        "scope": comparison["scope"],
        "base_commit": comparison["base_commit"],
        "candidate_commit": comparison["candidate_commit"],
        "units": {
            "cost": "seconds per request, client send-to-complete",
            "rtf": "request elapsed seconds / generated audio seconds",
            "ttfa": "seconds, send-to-first-nonempty-audio-payload",
        },
        "profiler_enabled": False,
        "points": [],
    }
    for point in comparison["points"]:
        item = {
            "concurrency": point["concurrency"],
            "requests_per_variant": point["requests_per_variant"],
            "text_multiset_sha256": point["text_multiset_sha256"],
            "variants": {},
        }
        for variant, key in [
            ("baseline", "baseline_summary"),
            ("candidate", "candidate_summary"),
        ]:
            summary = point[key]
            if (
                summary["failed_requests"]
                or summary["completed_requests"] != point["requests_per_variant"]
            ):
                raise ValueError("Incomplete serving measurement")
            item["variants"][variant] = {
                "cost_s": {
                    percentile: summary["latency_" + suffix + "_s"]
                    for percentile, suffix in [
                        ("mean", "mean"),
                        ("p50", "median"),
                        ("p95", "p95"),
                        ("p99", "p99"),
                    ]
                },
                "rtf": {
                    percentile: summary["rtf_" + suffix]
                    for percentile, suffix in [
                        ("mean", "mean"),
                        ("p50", "median"),
                        ("p95", "p95"),
                        ("p99", "p99"),
                    ]
                },
                "ttfa_s": {
                    percentile: summary["audio_ttfp_" + suffix + "_s"]
                    for percentile, suffix in [
                        ("mean", "mean"),
                        ("p50", "median"),
                        ("p95", "p95"),
                        ("p99", "p99"),
                    ]
                },
                "original_summary": summary,
            }
        report["points"].append(item)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
