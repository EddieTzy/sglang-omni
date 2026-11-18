# Fish S2-Pro：SlowAR 输出投影

状态：PR 已主动关闭，保留实验。主分类：archived。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head)

## 为什么优化

SlowAR 只需语义 token 和 EOS，却可能执行完整词表的输出投影。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：输出投影算子。

只计算活跃输出权重视图，再恢复完整词表布局和非法位置负无穷；不改变输入 embedding 和 token ID。矩阵变小不保证采样与质量完全不变。

实现说明：将 tied 输出投影缩小到语义 token 与 EOS 的活跃权重视图，再恢复原词表形状及非法位置的负无穷值；保留完整输入 embedding、原 token ID 和采样器。

代码位置：

- sglang_omni/models/fishaudio_s2_pro/sglang_model.py：S2ProSGLangTextModel.forward

## 基线与复现条件

基线：历史运行基线见下方原始记录和报告；未补造缺失版本。

依赖、设备、语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。

## 正确性、质量与既有详细记录

状态：PR 已主动关闭，保留实验。

将 tied 输出投影缩小到语义 token 与 EOS 的活跃权重视图，再恢复原词表形状及非法位置的负无穷值；保留完整输入 embedding、原 token ID 和采样器。

帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。

实现：[ perf/fish-semantic-head ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-semantic-head/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，S2ProSGLangTextModel.forward。

上游：[PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408)。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

保持历史记录中的作者与实现归属；本目录只是整理证据，不把引用实现改记为原创。

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
