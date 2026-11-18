# Fish S2-Pro：调度与 GPU 反馈状态

状态：已合并，上游 PR #3283。主分类：merged。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode)

## 为什么优化

下一步解码依赖上一轮 GPU 反馈状态，CPU 结果收集可能使 GPU 等待。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：CPU/GPU 调度重叠。

把下一步依赖的反馈更新提前留在 GPU，并延后 CPU 收集；用独立快照保证前后两轮不会相互覆盖。预期减少串行等待，必须用 CPU/CUDA 时间线确认重叠和关键路径。

实现说明：将下一步依赖的码本、历史与采样计数更新提前到 GPU，并为延后结果收集保留独立快照，使下一步 GPU 推理可与上一步 CPU 收集重叠；prefill 和请求行变化后重新同步。

代码位置：

- sglang_omni/models/fishaudio_s2_pro/sglang_model.py：decode_codebooks
- sglang_omni/models/fishaudio_s2_pro/model_runner.py：FishS2ProModelRunner.post_decode_launch, FishS2ProModelRunner.post_decode_resolve, lookahead_eligible
- sglang_omni/models/fishaudio_s2_pro/engine_builder.py：FishS2ProEngineBuilder.setup_model

## 基线与复现条件

基线：历史运行基线见下方原始记录和报告；未补造缺失版本。

依赖、设备、语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：固定 128-token 非流式小基准：并发 1/4/8 请求吞吐 +7.7%/+8.8%/+9.7%。不外推为所有自然长度流式请求的收益。

## 正确性、质量与既有详细记录

状态：已合并。2026-11-18 核对 GitHub 页面与上游提交记录，PR #3283 于 2026-11-04 11:07（北京时间）合并；提交 8da1c6c0d20ce96f85b2e0a92a2be61b25a8e7e2。

将下一步依赖的码本、历史与采样计数更新提前到 GPU，并为延后结果收集保留独立快照，使下一步 GPU 推理可与上一步 CPU 收集重叠；prefill 和请求行变化后重新同步。

固定 128-token 非流式小基准：并发 1/4/8 请求吞吐 +7.7%/+8.8%/+9.7%。不外推为所有自然长度流式请求的收益。

实现：[ perf/fish-async-decode ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，decode_codebooks。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，FishS2ProModelRunner.post_decode_launch, FishS2ProModelRunner.post_decode_resolve, lookahead_eligible。

代码位置：[engine_builder.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/engine_builder.py)，FishS2ProEngineBuilder.setup_model。

上游：[PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283)。

历史 GitHub 快照（已非当前状态）：2026-11-03T11:07:41+00:00，当前提交 eca3f90e022fa451cddc01cf73cdee92d90573f4，合并状态 blocked。

当时最新一轮 CI 标记为 action_required；以下保留历史记录，不表示目前仍在等待批准。当前状态以本节顶部的已合并核对为准。

- [Lint](https://github.com/sgl-project/sglang-omni/actions/runs/41620325016)：action_required（2026-11-03T10:34:00Z）
- [Test Layout](https://github.com/sgl-project/sglang-omni/actions/runs/41620324998)：action_required（2026-11-03T10:34:00Z）
- [Release Docker Images Nightly (NPU)](https://github.com/sgl-project/sglang-omni/actions/runs/41620325111)：action_required（2026-11-03T10:34:00Z）
- [XPU CI](https://github.com/sgl-project/sglang-omni/actions/runs/41620324794)：action_required（2026-11-03T10:33:59Z）
- [Omni CI](https://github.com/sgl-project/sglang-omni/actions/runs/41620324758)：action_required（2026-11-03T10:33:59Z）
- [Docs Check](https://github.com/sgl-project/sglang-omni/actions/runs/41620324825)：action_required（2026-11-03T10:33:59Z）
- [Rust Router](https://github.com/sgl-project/sglang-omni/actions/runs/41620324602)：action_required（2026-11-03T10:33:59Z）
- [CPU CI](https://github.com/sgl-project/sglang-omni/actions/runs/41620324806)：action_required（2026-11-03T10:33:59Z）

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

保持历史记录中的作者与实现归属；本目录只是整理证据，不把引用实现改记为原创。

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
