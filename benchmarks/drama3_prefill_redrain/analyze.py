import collections
import gzip
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent


def pct(x, p):
    v = sorted(x)
    i = (len(v) - 1) * p
    return v[int(i)] + (v[min(int(i) + 1, len(v) - 1)] - v[int(i)]) * (i % 1)


clean = json.loads((root / "clean.json").read_text())
waves = json.loads((root / "trace.json").read_text())
events = collections.defaultdict(list)
for e in json.loads((root / "markers.json").read_text()):
    events[e["request_id"]].append(e)
result = {}
for n in (1, 4):
    for arm in "AB":
        timed = [w for w in clean if (w["arm"], w["n"]) == (arm, n)]
        metrics = collections.defaultdict(list)
        walls = []
        first = []
        others = []
        paired = []
        for w in timed:
            rows = w["rows"]
            walls.append(
                (max(r["end"] for r in rows) - min(r["start"] for r in rows)) / 1e9
            )
            wave_ttfa = []
            for r in rows:
                assert r["rid"] not in events
                t = r["chunks"][0][0]
                audio = 0
                stall = 0
                for stamp, size in r["chunks"]:
                    stall = max(stall, (stamp - t) / 1e9 - audio)
                    audio += size / (2 * int(r["headers"]["x-sample-rate"]))
                latency = (r["end"] - r["start"]) / 1e9
                ttfa = (t - r["start"]) / 1e6
                wave_ttfa.append(ttfa)
                for k, v in dict(
                    audio=audio,
                    ttfa=ttfa,
                    rtf=latency / audio,
                    latency=latency,
                    stall=stall * 1000,
                ).items():
                    metrics[k].append(v)
            first.append(min(wave_ttfa))
            paired.append(statistics.median(wave_ttfa))
        z = dict(
            requests=len(metrics["audio"]),
            first_audio_p50=pct(metrics["ttfa"], 0.5),
            first_audio_p95=pct(metrics["ttfa"], 0.95),
            fastest_request_per_wave_p50=pct(first, 0.5),
            throughput=sum(metrics["audio"]) / sum(walls),
            mean_rtf=statistics.mean(metrics["rtf"]),
            mean_latency=statistics.mean(metrics["latency"]),
            underrun_p95_ms=pct(metrics["stall"], 0.95),
            wave_medians=paired,
        )
        splits = []
        reads = []
        late = []
        trace = []
        leader_ttfa = []
        last_dispatch_after_decision = []
        for w in (w for w in waves if (w["arm"], w["n"]) == (arm, n)):
            group = []
            for i, r in enumerate(w["rows"]):
                ev = sorted(events[r["rid"]], key=lambda e: e["timestamp_ns"])

                def find(k):
                    return next(e for e in ev if e["event_name"] == k)

                group.append(
                    (
                        find("stage_dispatch")["timestamp_ns"],
                        find("study_batch_selected")["timestamp_ns"],
                        find("scheduler_prefill_start")["timestamp_ns"],
                        find("scheduler_prefill_end")["args"]["batch_size"],
                        (r["chunks"][0][0] - r["start"]) / 1e6,
                    )
                )
                reads.extend(
                    e["args"]["added"]
                    for e in ev
                    if e["event_name"] == "study_second_read"
                )
                for e in ev:
                    trace.append(
                        dict(
                            name=e["event_name"],
                            ph="i",
                            s="t",
                            pid=1,
                            tid=w["rep"] * n + i,
                            ts=e["timestamp_ns"] / 1000,
                            args=e["args"],
                        )
                    )
            group.sort()
            splits.append([r[3] for r in group])
            leader_ttfa.append(group[0][4])
            late.extend((r[2] - group[0][0]) / 1e6 for r in group[1:])
            last_dispatch_after_decision.append(
                (max(r[0] for r in group) - min(r[1] for r in group)) / 1e6
            )
        z.update(
            prefill_sizes_by_request=splits,
            extra_reads=reads,
            later_prefill_ms=statistics.median(late) if late else None,
            first_dispatched_ttfa=statistics.median(leader_ttfa),
            last_dispatch_relative_to_first_batch_selected_ms=last_dispatch_after_decision,
        )
        with gzip.open(root / f"{arm}-c{n}.json.gz", "wt") as f:
            json.dump({"traceEvents": trace}, f)
        result[f"{arm}-C{n}"] = z
        print(
            arm,
            n,
            {k: round(v, 3) for k, v in z.items() if isinstance(v, float)},
            "reads",
            reads,
            "batches",
            splits,
        )
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
