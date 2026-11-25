"""Compare the two overlap-only process layouts within the same experiment."""

import json
from statistics import mean
from pathlib import Path
import sys

root = Path(sys.argv[1])
s = json.loads((root / "serving.json").read_text())
assert len({r["sample_set_sha256"] for r in s.values()}) == 1
keys = [
    ("audio_throughput_s_per_s", "吞吐（音频秒/秒）", 1),
    ("audio_ttfp_median_s", "首包 P50（ms）", 1000),
    ("audio_ttfp_p95_s", "首包 P95（ms）", 1000),
    ("audio_ttfp_p99_s", "首包 P99（ms）", 1000),
    ("rtf_mean", "平均 RTF", 1),
    ("latency_mean_s", "平均请求耗时（秒）", 1),
]
lines = [
    "## 保持 overlap，仅改变 vocoder 进程布局",
    "",
    "同进程两次运行位于实验开始和结束。独立进程两次运行穿插在 stateful 对照之间。C16、权重、客户端、输入和所有服务参数相同，单独指定 vocoder 的进程名称会同时改变通信方式和主机调度。",
    "",
    "| 指标 | 同进程 1 | 同进程 2 | 独立 1 | 独立 2 | 同进程均值 | 独立均值 | 变化 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for key, title, m in keys:
    d = [s[k][key] * m for k in ["a1", "default2", "fixed-a1", "a2"]]
    a, b = mean(d[:2]), mean(d[2:])
    lines.append(
        "| "
        + title
        + " | "
        + " | ".join(f"{v:.3f}" for v in d + [a, b])
        + f" | {100 * (b / a - 1):+.2f}% |"
    )
with (root / "RESULTS.md").open("a") as w:
    w.write("\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
