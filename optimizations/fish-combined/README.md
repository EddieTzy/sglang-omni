# Fish S2-Pro：语义头与 FastAR 清空及结果收集组合

状态：仅完成短测。主分类：in_progress。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined)

## 为什么优化

单项优化可能作用于同一瓶颈或彼此抵消，需要检查组合效果。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：组合优化。

在明确基线上组合三个改动，直接测量组合收益；不能把各自百分比相加。后续拆项归因必须做逐项对照。

实现说明：在独立基线上组合语义输出头、FastAR 清零移除和 batch 结果快照，检查收益是否叠加；没有把单项百分比直接相加。

代码位置：

- sglang_omni/models/fishaudio_s2_pro/sglang_model.py：forward, decode_codebooks
- sglang_omni/models/fishaudio_s2_pro/model_runner.py：collect_s2pro_step_outputs

## 基线与复现条件

基线：历史运行基线见下方原始记录和报告；未补造缺失版本。

依赖、设备、语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。

## 正确性、质量与既有详细记录

状态：仅完成短测。

在独立基线上组合语义输出头、FastAR 清零移除和 batch 结果快照，检查收益是否叠加；没有把单项百分比直接相加。

短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。

实现：[ perf/fish-decode-combined ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-combined/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，forward, decode_codebooks。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-combined/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，collect_s2pro_step_outputs。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

保持历史记录中的作者与实现归属；本目录只是整理证据，不把引用实现改记为原创。

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
