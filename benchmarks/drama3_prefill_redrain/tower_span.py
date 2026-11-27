"""Bound the residual tower timing on identical decode graph replays in the archived trace."""

import collections
import gzip
import json
import statistics
import sys

with gzip.open(sys.argv[1]) as f:
    events = json.load(f)["traceEvents"]
groups = collections.defaultdict(list)
for e in events:
    if e.get("cat") == "kernel" and e["args"].get("graph id") == 3098:
        groups[e["args"]["correlation"]].append(e)
rows = []
for group in groups.values():
    group.sort(key=lambda e: e["ts"])
    last_expert = max(i for i, e in enumerate(group) if e["name"] == "fused_moe_kernel")
    rope = [
        e for e in group[last_expert + 1 :] if "sglang::fused_rope_kernel" in e["name"]
    ]
    if len(rope) != 13:
        continue
    total = max(e["ts"] + e["dur"] for e in group) - group[0]["ts"]
    residual_span = rope[-1]["ts"] - rope[4]["ts"]
    whole_tail = (
        max(e["ts"] + e["dur"] for e in group)
        - group[last_expert]["ts"]
        - group[last_expert]["dur"]
    )
    rows.append(
        [
            total / 1000,
            residual_span / 1000,
            whole_tail / 1000,
            100 * residual_span / total,
        ]
    )
print(len(rows), [round(statistics.median(col), 3) for col in zip(*rows)])
