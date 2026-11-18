# Fish Audio Drama 3：SM100 kernel 支持、quantizer 上采样与 MoE 调优

状态：历史组合移植已拆分；分支仅作备份，未提 PR。主分类：archived。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe)

## 为什么优化

历史分支同时调整 SM100 支持、上采样和 MoE，无法把组合收益归给单项。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：历史组合，已拆分。

该条只保存来源和恢复信息；独立原理、验证与归因分别进入三个候选目录，不继续在组合条目累计收益。

实现说明：移植 SM100 sampler / fused tower 支持、FP16 quantizer phase projection 和可选 MoE up/down 配置，保留最新主线采样与张量并行限制。

代码位置：

- sglang_omni/models/fishaudio_drama3/channels_last_decoder.py：PhaseLinearUpsample, ChannelsLastDecoder
- sglang_omni/models/fishaudio_drama3/sampling_kernels.py：untested_gpu_reason
- examples/configs/drama3_b200_moe

## 基线与复现条件

基线：历史运行基线见下方原始记录和报告；未补造缺失版本。

依赖、设备、语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：历史组合记录，不代表当前三个独立候选的验证状态。17 项隔离 CPU 检查与原始 23.9% 组合收益保留为历史；当前结果见拆分条目。

## 正确性、质量与既有详细记录

[perf/drama3-quantizer-moe](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe) 基于 a338eb0e，曾记录 17 项隔离 CPU 检查。它已被上述三个独立候选替代，保留作来源与恢复备份，不再作为当前验证入口。外部 buffered GDN 原型没有合入这些候选。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

保持历史记录中的作者与实现归属；本目录只是整理证据，不把引用实现改记为原创。

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
