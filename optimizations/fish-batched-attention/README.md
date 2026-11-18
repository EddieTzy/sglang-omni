# Fish S2-Pro：FastAR 短序列 attention 批处理

状态：全量性能与质量评测完成，保留实验分支。主分类：measured_tradeoff。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention)

## 为什么优化

短序列逐请求调用 attention，可能使高并发下的 kernel 派发和小计算重复。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：attention 批处理与派发。

把逐请求 single_prefill 改为批量 FlashInfer decode，把现有 KV 缓冲解释为每请求一页，缓存 graph 兼容 plan，并在 GPU 更新有效页长度；保留单请求和 FA3 分支。

实现说明：将逐请求 single_prefill 调用改为一次批量 FlashInfer decode，把既有 KV 缓冲直接解释为每请求一页，缓存图兼容 plan，并在 GPU 更新有效页长度。保留单请求和 FA3 计算分支。

代码位置：

- sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/audio_decoder.py：flashinfer_kvcache_attention, fast_ar_batch_decode_plan

## 基线与复现条件

基线：历史运行基线见下方原始记录和报告；未补造缺失版本。

依赖、设备、语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：完整扫描并发1/16/64 帧吞吐变化 -0.16%/+3.39%/+10.27%；详细首包延迟和质量变化见实验报告，不据此自动提 PR。

## 正确性、质量与既有详细记录

状态：全量性能与质量评测完成，保留实验分支。

将逐请求 single_prefill 调用改为一次批量 FlashInfer decode，把既有 KV 缓冲直接解释为每请求一页，缓存图兼容 plan，并在 GPU 更新有效页长度。保留单请求和 FA3 计算分支。

完整扫描并发1/16/64 帧吞吐变化 -0.16%/+3.39%/+10.27%；详细首包延迟和质量变化见实验报告，不据此自动提 PR。

实现：[ perf/fish-batched-fast-ar-attention ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention)。

代码位置：[audio_decoder.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-batched-fast-ar-attention/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/audio_decoder.py)，flashinfer_kvcache_attention, fast_ar_batch_decode_plan。

完整评测：[实验报告](https://github.com/EddieTzy/sglang-omni/blob/experiment/fish-batched-fast-ar-attention/benchmarks/fish_batched_attention_experiment/standard_eval_20261103/RESULTS.txt)。

当前最值得继续推进的独立优化：高并发服务吞吐有收益，单请求基本持平。全量质量波动的配对区间跨零，未证明质量改善或统计意义上的不退化；尚无 H100 验证，按当前要求只保存分支。

6528 条性能请求与 12432 条质量音频全部完成；无生成失败、截断或评分跳过。质量为每臂两轮全集，以下是两轮均值。

| 指标 | 基线 | 批量 attention | 变化 | 配对 95% 描述区间 |
| --- | --- | --- | --- | --- |
| 英文 WER | 1.001% | 1.038% | +0.038 个百分点 | [-0.033, +0.104] 个百分点 |
| EN 音色相似度 | 64.759 | 64.770 | +0.011 | [-0.194, +0.212] |
| 中文 CER | 0.665% | 0.639% | -0.026 个百分点 | [-0.060, +0.012] 个百分点 |
| ZH 音色相似度 | 74.052 | 74.157 | +0.104 | [-0.003, +0.206] |

区间使用 2000 次按语句 ID 的配对 bootstrap，每个 ID 保留两轮生成；这是描述性评估，不是预先指定容忍阈值的非劣效检验。

评测基线为 3b394b4193a8b76a6b5f52bf99da2a2729d6161e，本地被测提交为 5d52af12b1ac8b26bf82e2b2501ff94e81386cb5。GitHub 备份提交 55d5fc62e70dd4954caea65cf5e24b93a88f6ab8 基于 fork 已有基线 7537f89baca54d19de2b7afe5465bb614c10b625；Fish 模型子树逐字相同，但整个运行时基线不同。该备份方式用于避免提交与本次实现无关、且当前令牌无权更新的上游 workflow 历史。原始补丁 SHA256 为 00da284bf296508e0469c20626474dc64aab574a1d31f7023031256c6d42d87e，完整实测版本可按报告中的补丁或 bundle 恢复。

恢复校验：备份补丁与实测补丁完全相同；将其应用到实测基线后，整个源码树 SHA 为 74223650ec61f67b3a1e759a25556ce02e8c42ff，与被测提交一致。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

保持历史记录中的作者与实现归属；本目录只是整理证据，不把引用实现改记为原创。

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
