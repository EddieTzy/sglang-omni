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
    percentiles = (
        statistics.quantiles(values, n=100, method="inclusive")
        if len(values) > 1
        else values * 99
    )
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentiles[94],
        "p99": percentiles[98],
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    output = {
        "scope": "Exclusive-GPU instrumented service profiles, not uninstrumented performance; preprocessing CPU+CUDA with snake launch scopes, shared engine/vocoder CUDA only",
        "points": [],
    }
    for speed_path in sorted(options.directory.glob("*/profile-c*/speed_results.json")):
        folder = speed_path.parent
        speed = json.loads(speed_path.read_text())
        summary = speed["summary"]
        assert summary["failed_requests"] == 0 and summary["max_token_hits"] == 0
        events = []
        statuses = []
        for path in (folder / "profile/events").glob("*.jsonl"):
            events.extend(
                json.loads(line) for line in path.read_text().splitlines() if line
            )
        for path in (folder / "profile/events").glob("run_status*.json"):
            statuses.append(json.loads(path.read_text()))
        assert statuses and all(
            row["complete"] and row["dropped_events"] == 0 for row in statuses
        )
        request_events = defaultdict(list)
        for event in events:
            request_events[event["request_id"]].append(event)
        first_ns = min(
            event["timestamp_ns"]
            for event in events
            if event["event_name"] == "request_admission"
        )
        last_ns = max(event["timestamp_ns"] for event in events)
        end_us = (last_ns - first_ns) / 1000
        intervals = []
        kernel_counts = Counter()
        kernel_times = Counter()
        snake_scopes = []
        slow_cpu = []
        trace_metadata = []
        for path in sorted((folder / "profile").glob("*.trace.json.gz")):
            with gzip.open(path, "rb") as stream:
                header = stream.read(4096).decode(errors="replace")
            match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
            assert match, path
            base_ns = int(match.group(1))
            offset_us = (base_ns - first_ns) / 1000
            categories = Counter()
            with gzip.open(path, "rb") as stream:
                for event in ijson.items(stream, "traceEvents.item", use_float=True):
                    category = event.get("cat", "metadata")
                    categories[category] += 1
                    if event.get("ph") != "X":
                        continue
                    name = event.get("name", "")
                    start_us = offset_us + event["ts"]
                    duration_us = event.get("dur", 0)
                    stop_us = start_us + duration_us
                    if stop_us <= 0 or start_us >= end_us:
                        continue
                    if category in ("kernel", "gpu_memcpy", "gpu_memset"):
                        intervals.append((max(0, start_us), min(end_us, stop_us)))
                        if category == "kernel":
                            kernel_counts[name] += 1
                            kernel_times[name] += duration_us
                    if name.startswith("reference.snake"):
                        snake_scopes.append(
                            {
                                "name": name,
                                "start_us": start_us,
                                "stop_us": stop_us,
                                "duration_ms": duration_us / 1000,
                            }
                        )
                    if duration_us >= 10000 and name.startswith(
                        (
                            "aten::",
                            "cuda",
                            "cuModule",
                            "cuLink",
                            "cuLibrary",
                            "reference.snake",
                        )
                    ):
                        slow_cpu.append(
                            {
                                "name": name,
                                "start_us": start_us,
                                "duration_ms": duration_us / 1000,
                            }
                        )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            trace_metadata.append(
                {
                    "name": str(path.relative_to(options.directory)),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "base_time_ns": base_ns,
                    "event_categories": dict(categories),
                }
            )
        intervals.sort()
        merged = []
        for start_us, stop_us in intervals:
            if merged and start_us <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop_us)
            else:
                merged.append([start_us, stop_us])
        gaps = []
        cursor = 0.0
        for start_us, stop_us in merged:
            if start_us > cursor:
                gaps.append((start_us - cursor) / 1000)
            cursor = max(cursor, stop_us)
        if cursor < end_us:
            gaps.append((end_us - cursor) / 1000)
        active_us = sum(stop - start for start, stop in merged)
        bins = []
        width = end_us / 200
        for position in range(200):
            start = position * width
            stop = start + width
            overlap = sum(
                max(0, min(right, stop) - max(left, start))
                for left, right in merged
                if left < stop and right > start
            )
            bins.append(overlap / width)
        details = []
        boundary_values = defaultdict(list)
        for index, row in enumerate(speed["per_request"]):
            related = request_events.get(row.get("speech_outcome_id"), [])
            assert related, ("No matching stage events", row["id"])
            boundaries = {}
            for event in related:
                key = event["stage"] + "/" + event["event_name"]
                boundaries[key] = min(
                    boundaries.get(key, event["timestamp_ns"]), event["timestamp_ns"]
                )
            measured = {}
            pairs = {
                "preprocessing_queue": (
                    "preprocessing/stage_input_received",
                    "preprocessing/stage_dispatch",
                ),
                "preprocessing_dispatch_to_complete": (
                    "preprocessing/stage_dispatch",
                    "preprocessing/stage_complete",
                ),
                "engine_input_to_first_stream": (
                    "tts_engine/stage_input_received",
                    "tts_engine/stage_first_stream_chunk_sent",
                ),
                "vocoder_first_input_to_first_stream": (
                    "vocoder/stage_stream_chunk_received",
                    "vocoder/stage_first_stream_chunk_sent",
                ),
                "admission_to_coordinator_first_audio": (
                    "coordinator/request_admission",
                    "coordinator/coordinator_stream_received",
                ),
            }
            for key, (start, stop) in pairs.items():
                if start in boundaries and stop in boundaries:
                    value = (boundaries[stop] - boundaries[start]) / 1e6
                    assert value >= 0, (row["id"], key, value)
                    measured[key] = value
                    boundary_values[key].append(value)
            detail = {
                "index": index,
                "id": row["id"],
                "speech_outcome_id": row.get("speech_outcome_id"),
                "cost_s": row["latency_s"],
                "rtf": row["rtf"],
                "ttfa_s": row["audio_ttfp_s"],
                "completion_tokens": row["completion_tokens"],
                "boundaries_ms": measured,
            }
            if folder.name == "profile-c1":
                start = (boundaries["preprocessing/stage_dispatch"] - first_ns) / 1000
                stop = (boundaries["preprocessing/stage_complete"] - first_ns) / 1000
                matching = [
                    scope
                    for scope in snake_scopes
                    if scope["start_us"] >= start - 1 and scope["stop_us"] <= stop + 1
                ]
                detail["reference_snake_scope_count"] = len(matching)
                detail["reference_snake_total_cpu_scope_ms"] = sum(
                    scope["duration_ms"] for scope in matching
                )
                detail["reference_snake_slowest_cpu_scope"] = (
                    max(matching, key=lambda scope: scope["duration_ms"])
                    if matching
                    else None
                )
                detail["reference_slow_cpu_regions"] = [
                    scope
                    for scope in slow_cpu
                    if scope["start_us"] >= start - 1 and scope["start_us"] <= stop + 1
                ]
            details.append(detail)
        result = {
            "variant": folder.parent.name,
            "point": folder.name,
            "summary": summary,
            "requests": details,
            "event_storage_complete": True,
            "stage_boundary_ms": {
                key: distribution(values) for key, values in boundary_values.items()
            },
            "trace_files": trace_metadata,
            "first_admission_epoch_ns": first_ns,
            "gpu_activity": {
                "scope": "Union of captured kernel/memcpy/memset intervals across both processes; relative clock conversion before floating-point epoch addition; not SM utilization",
                "envelope_ms": end_us / 1000,
                "active_union_ms": active_us / 1000,
                "active_fraction": active_us / end_us,
                "gaps_ms": distribution(gaps),
                "gaps_over_1ms": sum(value > 1 for value in gaps),
                "equal_width_activity_bins": bins,
            },
            "reference_snake_scopes": snake_scopes,
            "slow_cpu_regions": sorted(
                slow_cpu, key=lambda scope: scope["duration_ms"], reverse=True
            )[:100],
            "kernel_count": sum(kernel_counts.values()),
            "top_kernels_by_time_sum": [
                {
                    "name": name,
                    "count": kernel_counts[name],
                    "sum_duration_ms": duration / 1000,
                }
                for name, duration in kernel_times.most_common(25)
            ],
            "layout_conv_kernels": [
                {
                    "name": name,
                    "count": kernel_counts[name],
                    "sum_duration_ms": duration / 1000,
                }
                for name, duration in kernel_times.items()
                if any(
                    part in name.lower()
                    for part in ("dgrad", "nchw", "nhwc", "copy_kernel")
                )
            ],
        }
        output["points"].append(result)
        (options.directory / "analysis.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        print(
            result["variant"],
            result["point"],
            "requests",
            len(details),
            "kernels",
            result["kernel_count"],
            "snake_scopes",
            len(snake_scopes),
            flush=True,
        )
    assert len(output["points"]) == 8
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
