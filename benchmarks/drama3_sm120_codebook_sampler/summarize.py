"""Compare independent service runs and paired English quality measurements."""

import collections
import json
import random
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent


def load(file):
    return json.loads((root / file).read_text())


metrics = [
    "audio_throughput_s_per_s",
    "rtf_mean",
    "latency_mean_s",
    "audio_ttfp_median_s",
    "audio_ttfp_p95_s",
    "max_playback_underrun_p95_s",
    "audio_duration_mean_s",
]


def speed_report():
    output = {}
    for c, n in [(4, 48), (16, 128)]:
        records = {
            arm: load(f"{arm}-c{c}.json")["summary"] for arm in ["A1", "B1", "B2", "A2"]
        }
        assert all(
            x["completed_requests"] == n and x["failed_requests"] == 0
            for x in records.values()
        )
        summary = {}
        for key in metrics:
            a = statistics.mean(records[label][key] for label in ["A1", "A2"])
            b = statistics.mean(records[label][key] for label in ["B1", "B2"])
            summary[key] = dict(
                baseline=a, candidate=b, percent=(100 * (b / a - 1) if a else None)
            )
        output[c] = summary
    return output


def quality_report():
    data = load("quality-scores.json")
    groups = data["groups"]
    rng = random.Random(91023)
    results = {}
    for c in [4, 16]:
        by_sample = collections.defaultdict(list)
        for r in data["rows"]:
            if r["concurrency"] == c:
                by_sample[r["sample"]].append(r)
        diffs = []
        for rs in by_sample.values():
            assert sorted(r["arm"] for r in rs) == ["A1", "A2", "B1", "B2"]

            def aggregate(rs):
                return (
                    sum(r["sub"] + r["ins"] + r["deleted"] for r in rs),
                    sum(r["hits"] + r["sub"] + r["deleted"] for r in rs),
                    statistics.mean(r["sim"] for r in rs),
                )

            a = aggregate([r for r in rs if r["arm"][0] == "A"])
            b = aggregate([r for r in rs if r["arm"][0] == "B"])
            assert a[1] == b[1]
            diffs.append((b[0] - a[0], a[1], b[2] - a[2]))
        trials = []
        for _ in range(4000):
            sample = rng.choices(diffs, k=len(diffs))
            col = list(zip(*sample))
            trials.append((100 * sum(col[0]) / sum(col[1]), statistics.mean(col[2])))
        results[c] = dict(
            A=groups["A"][str(c)],
            B=groups["B"][str(c)],
            paired_bootstrap_95=[
                (sorted(col)[100], sorted(col)[3899]) for col in zip(*trials)
            ],
        )
    return results


if __name__ == "__main__":
    r = dict(speed=speed_report(), quality=quality_report())
    (root / "summary.json").write_text(json.dumps(r, indent=2) + "\n")
    for c in (4, 16):
        print(c, r["speed"][c])
        print(c, r["quality"][c])
