from pathlib import Path
import json, statistics, sys

root = Path(sys.argv[1])
s = json.loads((root / "serving.json").read_text())
rows = []


def load_phases(arm):
    p = json.loads((root / arm / "phases.json").read_text())
    assert p["matched"] == len(p["rows"]) == 128
    selected = p["rows"][:13]
    v = [statistics.mean(r["phases_ms"][i] for r in selected) for i in range(4)]
    return [arm, statistics.mean(r["ttfa_ms"] for r in selected), *v]


for title, key, mul in [
    ("吞吐（音频秒/秒）", "audio_throughput_s_per_s", 1),
    ("首包 P50（ms）", "audio_ttfp_median_s", 1000),
    ("首包 P95（ms）", "audio_ttfp_p95_s", 1000),
    ("首包 P99（ms）", "audio_ttfp_p99_s", 1000),
    ("平均请求耗时（秒）", "latency_mean_s", 1),
    ("平均请求 RTF", "rtf_mean", 1),
]:
    v = [s[arm][key] * mul for arm in ["a1", "b1", "fixed-a1", "b2", "a2"]]
    a = statistics.mean(v[i] for i in [2, 4])
    b = statistics.mean(v[i] for i in [1, 3])
    rows.append(
        "| "
        + title
        + " | "
        + " | ".join(f"{x:.3f}" for x in v + [a, b])
        + f" | {100 * (b / a - 1):+.2f}% |"
    )
phases = [load_phases(a) for a in ["a2", "b1"]]
table = (
    "| 指标 | 默认同进程 overlap | stateful 1 | 独立 overlap 1 | stateful 2 | 独立 overlap 2 | 独立 overlap 均值 | stateful 均值 | 变化 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    + "\n".join(rows)
)
ph = (
    "| 诊断启动 | 首包均值 | 客户端发出至引擎送出首窗口所需最后一块码 | 引擎送出至 vocoder 开始 | vocoder 窗口执行 | 执行结束至客户端首 PCM |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    + "\n".join(
        "| " + x[0] + " | " + " | ".join(f"{v:.1f}" for v in x[1:]) + " |"
        for x in phases
    )
)
(root / "RESULTS.md").write_text(
    "# C16 测量\n\n"
    + table
    + "\n\n## 独立诊断中首包最慢的 13/128 个请求\n\n单位为毫秒。各段按同一条请求配对，按该组首包最慢约 10% 取均值。\n\n"
    + ph
    + "\n"
)
print(table, ph, sep="\n")
