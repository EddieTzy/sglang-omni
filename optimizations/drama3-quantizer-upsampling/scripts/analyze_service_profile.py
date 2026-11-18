import argparse
import gzip
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import ijson


def distribution(values):
    if not values:
        return None
    quantiles = (
        statistics.quantiles(values, n=100, method="inclusive")
        if len(values) > 1
        else values * 99
    )
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": quantiles[94],
        "p99": quantiles[98],
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    directory = arguments.directory
    report = {
        "scope": "Exclusive GPU service diagnosis, profiler timings are not uninstrumented performance claims",
        "points": [],
    }
    for point in sorted(directory.glob("*/profile-c*/speed_results.json")):
        folder = point.parent
        data = json.loads(point.read_text())
        if data["summary"]["failed_requests"] or data["summary"].get(
            "max_token_hits", 0
        ):
            raise RuntimeError(
                "Failed or truncated diagnostic requests: " + str(folder)
            )
        events = []
        statuses = []
        for path in (folder / "profile/events").glob("*.jsonl"):
            events.extend(
                json.loads(line) for line in path.read_text().splitlines() if line
            )
        for path in (folder / "profile/events").glob("run_status_*.json"):
            statuses.append(json.loads(path.read_text()))
        if not statuses or not all(
            item["complete"] and item["dropped_events"] == 0 for item in statuses
        ):
            raise RuntimeError("Incomplete event storage: " + str(folder))
        requests = defaultdict(list)
        for event in events:
            requests[event["request_id"]].append(event)
        admissions = [
            event["timestamp_ns"] / 1000
            for event in events
            if event["event_name"] == "request_admission"
        ]
        lower = min(admissions)
        upper = max(event["timestamp_ns"] / 1000 for event in events)
        event_names = Counter(
            (event["stage"] + "/" + event["event_name"]) for event in events
        )
        per_request = []
        timing_values = defaultdict(list)
        for item in data["per_request"]:
            identity = item.get("speech_outcome_id")
            matching = requests.get(identity, [])
            details = {
                "dataset_id": item["id"],
                "speech_outcome_id": identity,
                "client_cost_s": item["latency_s"],
                "client_ttfa_s": item["audio_ttfp_s"],
                "rtf": item["rtf"],
                "completion_tokens": item["completion_tokens"],
                "event_count": len(matching),
                "boundaries_ms": {},
            }
            boundaries = {}
            for event in matching:
                key = event["stage"] + "/" + event["event_name"]
                boundaries[key] = min(
                    boundaries.get(key, event["timestamp_ns"]), event["timestamp_ns"]
                )
            pairs = {
                "admission_to_first_audio": (
                    "coordinator/request_admission",
                    "coordinator/coordinator_stream_received",
                ),
                "preprocessing_queue": (
                    "preprocessing/stage_input_received",
                    "preprocessing/stage_dispatch",
                ),
                "preprocessing_dispatch_to_complete": (
                    "preprocessing/stage_dispatch",
                    "preprocessing/stage_complete",
                ),
                "engine_stage_queue": (
                    "tts_engine/stage_input_received",
                    "tts_engine/stage_dispatch",
                ),
                "engine_dispatch_to_first_stream": (
                    "tts_engine/stage_dispatch",
                    "tts_engine/stage_first_stream_chunk_sent",
                ),
                "engine_input_to_first_stream": (
                    "tts_engine/stage_input_received",
                    "tts_engine/stage_first_stream_chunk_sent",
                ),
                "vocoder_first_input_to_first_stream": (
                    "vocoder/stage_stream_chunk_received",
                    "vocoder/stage_first_stream_chunk_sent",
                ),
            }
            for name, (start, end) in pairs.items():
                if start in boundaries and end in boundaries:
                    value = (boundaries[end] - boundaries[start]) / 1e6
                    details["boundaries_ms"][name] = value
                    timing_values[name].append(value)
            per_request.append(details)
        kernel_time = Counter()
        kernel_count = Counter()
        categories = Counter()
        traces = []
        intervals = []
        for path in (folder / "profile").glob("*.trace.json.gz"):
            with gzip.open(path, "rb") as stream:
                header = stream.read(4096).decode("utf-8", errors="replace")
            base_match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
            if not base_match:
                raise ValueError("No clock base: " + str(path))
            base_us = int(base_match.group(1)) / 1000
            counts = Counter()
            with gzip.open(path, "rb") as stream:
                for event in ijson.items(stream, "traceEvents.item", use_float=True):
                    counts[event.get("cat", "metadata")] += 1
                    category = event.get("cat", "")
                    if event.get("ph") != "X" or category not in (
                        "kernel",
                        "gpu_memcpy",
                        "gpu_memset",
                    ):
                        continue
                    timestamp = base_us + event["ts"]
                    end = timestamp + event.get("dur", 0)
                    if end <= lower or timestamp >= upper:
                        continue
                    intervals.append((max(lower, timestamp), min(upper, end)))
                    if category == "kernel":
                        name = event["name"]
                        kernel_count[name] += 1
                        kernel_time[name] += event.get("dur", 0)
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            traces.append(
                {
                    "name": str(path.relative_to(directory)),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "clock_base_ns": int(base_match.group(1)),
                    "event_categories": dict(counts),
                }
            )
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        gaps = []
        cursor = lower
        for start, end in merged:
            if start > cursor:
                gaps.append((start - cursor) / 1000)
            cursor = max(cursor, end)
        if cursor < upper:
            gaps.append((upper - cursor) / 1000)
        active = sum(end - start for start, end in merged)
        bin_width = (upper - lower) / 200
        activity_bins = []
        for bin_index in range(200):
            bin_start = lower + bin_index * bin_width
            bin_end = bin_start + bin_width
            overlap = sum(
                max(0, min(end, bin_end) - max(start, bin_start))
                for start, end in merged
                if start < bin_end and end > bin_start
            )
            activity_bins.append(overlap / bin_width)
        point_report = {
            "variant": folder.parent.name,
            "point": folder.name,
            "profiler_enabled": True,
            "summary": data["summary"],
            "event_storage_complete": True,
            "matched_requests": sum(item["event_count"] > 0 for item in per_request),
            "requests": per_request,
            "stage_boundary_ms": {
                key: distribution(value) for key, value in timing_values.items()
            },
            "event_names": dict(event_names),
            "trace_files": traces,
            "gpu_activity": {
                "scope": "Union of captured kernel/memcpy/memset intervals across both worker processes within event admission envelope; includes host/IPC gaps, not SM utilization",
                "envelope_ms": (upper - lower) / 1000,
                "active_union_ms": active / 1000,
                "active_fraction": active / (upper - lower),
                "equal_width_activity_bins": activity_bins,
                "gaps_ms": distribution(gaps),
                "gaps_over_1ms": sum(gap > 1 for gap in gaps),
                "gaps_over_10ms": sum(gap > 10 for gap in gaps),
            },
            "layout_and_conv_kernels": [
                {"name": name, "count": kernel_count[name], "summed_duration_us": time}
                for name, time in kernel_time.items()
                if any(
                    part in name.lower()
                    for part in ("dgrad", "nchw", "nhwc", "copy_kernel")
                )
            ],
            "top_kernels_by_time_sum": [
                {"name": name, "count": kernel_count[name], "summed_duration_us": time}
                for name, time in kernel_time.most_common(25)
            ],
        }
        report["points"].append(point_report)
        (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            folder.parent.name,
            folder.name,
            "matched",
            point_report["matched_requests"],
            "/",
            len(per_request),
            "kernels",
            sum(kernel_count.values()),
            flush=True,
        )
    (directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
