"""Analyze four-request bursts using unprofiled client times and separate stage markers."""

import collections
import gzip
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent


def quantile(xs, p):
    a = sorted(xs)
    i = (len(a) - 1) * p
    j = int(i)
    return a[j] + (a[min(j + 1, len(a) - 1)] - a[j]) * (i - j)


outputs = {}
for arm in ["A1", "B1", "A2", "B3", "A3", "C1", "C2", "A4", "A5"]:
    folder = root / arm
    if not folder.exists():
        continue
    rows = []
    walls = []
    for wave in json.loads((folder / "clean.json").read_text()):
        requests = wave["rows"]
        walls.append(
            (max(r["end"] for r in requests) - min(r["start"] for r in requests)) / 1e9
        )
        for r in requests:
            first = r["chunks"][0][0]
            audio, missing = 0, 0
            for t, size in r["chunks"]:
                missing = max(missing, (t - first) / 1e9 - audio)
                audio += size / (int(r["headers"]["x-sample-rate"]) * 2)
            elapsed = (r["end"] - r["start"]) / 1e9
            rows.append(
                dict(
                    ttfa=(first - r["start"]) / 1e6,
                    rtf=elapsed / audio,
                    latency=elapsed,
                    audio=audio,
                    underrun=missing * 1000,
                )
            )
    remaining_waits = []
    count = collections.Counter()
    firsts, others, first_audio, other_audio = [], [], [], []
    trace = []
    markers = collections.defaultdict(dict)
    for e in json.loads((folder / "events.json").read_text()):
        markers[e["request_id"]][e["event_name"]] = e
    for wave in json.loads((folder / "request_profile.json").read_text()):
        group = sorted(
            wave["rows"],
            key=lambda r: markers[r["rid"]]["stage_dispatch"]["timestamp_ns"],
        )
        start = markers[group[0]["rid"]]["stage_dispatch"]["timestamp_ns"]
        for i, r in enumerate(group):
            ev = markers[r["rid"]]
            t = [
                ev[k]["timestamp_ns"]
                for k in [
                    "stage_dispatch",
                    "scheduler_request_build_start",
                    "scheduler_request_build_end",
                    "scheduler_queue_enter",
                    "scheduler_prefill_start",
                    "scheduler_prefill_end",
                ]
            ]
            (firsts if i == 0 else others).append(
                [(t[1] - t[0]) / 1e6, (t[4] - start) / 1e6]
            )
            (first_audio if i == 0 else other_audio).append(
                (r["chunks"][0][0] - r["start"]) / 1e6
            )
            if i:
                remaining_waits.append([(t[3] - t[2]) / 1e6, (t[4] - t[3]) / 1e6])
            count[ev["scheduler_prefill_end"]["metadata"]["batch_size"]] += 1
            # Markers are independent instants; early building and queue admission may overlap.
            for e in ev.values():
                trace.append(
                    dict(
                        name=e["event_name"],
                        ph="i",
                        s="t",
                        pid=1,
                        tid=wave["rep"] * 4 + i,
                        ts=e["timestamp_ns"] / 1000,
                        args=e["metadata"],
                    )
                )
    with gzip.open(folder / "scheduler-trace.json.gz", "wt") as f:
        json.dump({"traceEvents": trace}, f)
    assert len(rows) == 48 and len(others) == 12
    outputs[arm] = dict(
        requests=len(rows),
        throughput=sum(r["audio"] for r in rows) / sum(walls),
        rtf=statistics.mean(r["rtf"] for r in rows),
        latency=statistics.mean(r["latency"] for r in rows),
        ttfa50=quantile([r["ttfa"] for r in rows], 0.5),
        ttfa95=quantile([r["ttfa"] for r in rows], 0.95),
        underrun95_ms=quantile([r["underrun"] for r in rows], 0.95),
        batch_requests=count,
        later_built_to_queue_and_queue_to_prefill=list(
            map(statistics.median, zip(*remaining_waits))
        ),
        first_build_prefill=list(map(statistics.median, zip(*firsts))),
        later_build_prefill=list(map(statistics.median, zip(*others))),
        diagnostic_first_ttfa=statistics.median(first_audio),
        diagnostic_later_ttfa=statistics.median(other_audio),
    )
(root / "summary.json").write_text(json.dumps(outputs, indent=2) + "\n")
for k, v in outputs.items():
    print(k, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
