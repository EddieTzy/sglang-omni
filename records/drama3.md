# Fish Audio Drama 3 优化记录

框架：SGLang / sglang-omni。目标硬件：NVIDIA B200。当前三项是独立候选分支，均未创建 PR；代码推送不等于 PR 已提交或已合并。

最后整理：2026-11-18。测试基线固定为上游 f17a438cd753248d70916700aa4dccd9e7feb098，不将它称为此后任意时刻的“最新 main”。

## 当前分类

| 候选 | 分类 | 当前结论 | 分支 / 提交 |
| --- | --- | --- | --- |
| Quantizer 上采样 | 已测有收益或取舍 | 单组初筛在并发64 音频吞吐 +11.10%，并发1 为 -5.75%；首音长尾需定位，未决定提 PR | [perf/drama3-b200-quantizer-upsample](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-quantizer-upsample)，57c747e0 |
| SM100 sampler / fused tower | 进行中或待验证 | 相关 kernel 检查通过，但严格 graph/eager 等问题未完全解释，没有独立性能结论 | [feat/drama3-sm100-kernels](https://github.com/EddieTzy/sglang-omni/tree/feat/drama3-sm100-kernels)，e34702b2 |
| B200 MoE 配置 | 进行中或待验证 | 可选 up/down 配置已分离；尚无当前版本的独立数值、性能和完整质量结论 | [perf/drama3-b200-moe-configs](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-moe-configs)，aa7aae97 |

原始思路和实现来自 lamianlbe 的 [1f7e3317](https://github.com/lamianlbe/sglang-omni/commit/1f7e33177c17fb243262a3296b69147fd1450c60) 与 [b82e2ad9](https://github.com/lamianlbe/sglang-omni/commit/b82e2ad974fefa068e65583ba537e5f59062b4b7)。本人的工作包括主线适配、分离候选、避免修改共享 codec、修正同名包装类判断、补充回归与配对验证；不把原始优化计为本人原创，提交保留原作者归属。

## Quantizer 上采样

将满足条件的 FP16 quantizer 转置卷积表达为 phase-ordered dense projection：仅适用于 SM100、无分组、无 dilation、无 padding、无 output padding，且 kernel size 等于 stride。优化只作用于 channels-last 解码器的路径，不替换共享 codec 内的模块，避免影响 incremental / stateful 解码器。

最终实际生效路径通过 30 项正确性检查，其中 13 项为 CUDA 检查；覆盖输出采样位置、bias、非连续输入、适用条件、投影 CUDA Graph 与 eager 的逐位一致性，以及共享 codec 的 stateful 使用。

真实 checkpoint 的 16 组 EN/ZH 固定码本对照确认选中 2 个投影，输出长度保持一致，均通过既有 FP16 codec 容差（rtol=0.02、atol=0.02）；最大波形绝对差为 0.00323486328125，最低波形 SNR 为 53.2953 dB。这是数值与重构检查，不是听感、ASR、音色相似度或质量非劣效结论。

一组独占 B200 的服务性能对照已完成，顺序为 base→优化：并发1/16/64 分别测 64/512/1088 个请求，每臂先预热，未追加重复。全部 3328 个正式请求成功、观察到正常 stop、无生成预算截断。音频吞吐变化分别为 -5.75%/+3.77%/+11.10%，首音 p95 在并发1/16 变慢、64 变快。完整条件和数据见[单组实测报告](drama3-quantizer-b200-20261118.md)。

离线配对发现：并发1 的 64 条请求逐条 completion token 数都一致，首音中位数从 72.5 ms 变为 74.2 ms，但 3 条早期请求分别多等待约 0.40/0.81/1.18 s；它们贡献了约 93.5% 的首音增加总量。尚不能将退化归因为 GEMM 每一步更慢，也不能剔除这些样本美化结果。下一步是定位首次路径和调度等因素，再决定是否限制启用条件；后续用户已授权独占 GPU 诊断，见下节。

## 独占 profile 与后续诊断

[上采样完整详细目录](../optimizations/drama3-quantizer-upsampling/README.md)包含为什么优化、相位映射原理、代码位置和前后 profile。[服务归因报告](../optimizations/drama3-quantizer-upsampling/profiles/service-20261118/README.md)确认目标 cuDNN direct-dgrad 和相关布局转换在优化版消失，未发现 GPU 空隙比例下降。144条 profile 请求全部匹配阶段事件，384条无 profiler 控制请求全部成功，原服务已恢复且健康。

后续短控制 C1 cost均值0.555→0.551s、RTF均值0.1324→0.1316，首包p95约0.165s；C64 cost均值2.457→2.234s、RTF均值0.6376→0.5945，但首包p95 1.5960→2.0174s、RTF p95 0.9676→1.0443。语料前缀、样本数和预热顺序不同，不替换前面的正式初筛；保留首包和尾部回退。旧三条首包尖峰在热路径未复现，根因未确认。

## SM100 支持

开放已有 Triton sampler 和 fused tower 的 B200 路径，保留模型、形状、确定性推理和张量并行的回退条件。测试辅助代码适配 SGLang 0.5.23 的逐层 attention hook，没有降低数值或状态断言。

Sampler 模块 38 项通过，tower 模块 31 项通过。完整 Drama3 套件包含 benchmark 标记的 graph 检查：833 项通过、4 项跳过、21 项失败、0 项准备错误。未修改上游另加 channels-last 卷积套件的结果为 767 项通过、67 项跳过、19 项失败、28 项准备错误；测试集合和架构跳过条件不同，不能直接相减作为修复数量。

上游的准备错误来自旧 attention hook。适配后仍有两个严格 graph/eager 检查失败；保存的禁用 sampler 和 tower 控制也存在 semantic-logit 差异，分析的主场景中调度、输入帧、生成帧一致。这缩小了问题范围，但没有完全解释或豁免严格失败。

B200 的默认 FA3 增量 codec 路径报不支持，另有 stage/memory、CPU-without-Triton 和严格 vocoder 检查的基线问题；一次 stateful capture 失败还需隔离复现。因此不宣称全套通过，也没有本候选的独立吞吐或完整音质结论。

## MoE 配置

提供 Triton 3.8.0、B200、block-FP8 专家形状对应的 up/down TMA 配置，通过 SGLANG_MOE_CONFIG_DIR 显式选择。目录中的 baseline 是历史 non-TMA 调优，不是当前上游默认；对比当前默认时应取消环境变量并重启服务。

目前仅完成文档与静态检查，当前运行时的专家输出对照、单项服务收益和完整质量评测均未完成。原分支 23.9% 属于旧基线的上采样与 MoE 组合结果，不能拆算给本项，也不能当作本次移植的实测结果。后续独占 base profile 发现 fused_moe_kernel 为最大单类热点，可作为[MoE详细目录](../optimizations/drama3-b200-moe-configs/README.md)的动机证据；本项优化后验证仍缺失。

## 历史组合分支

[perf/drama3-quantizer-moe](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe) 基于 a338eb0e，曾记录 17 项隔离 CPU 检查。它已被上述三个独立候选替代，保留作来源与恢复备份，不再作为当前验证入口。外部 buffered GDN 原型没有合入这些候选。

## 证据与下一步

运行环境为 PyTorch 2.14.1+cu130、SGLang 0.5.23、Triton 3.8.0、FlashInfer 0.7.2、驱动 580.173.02。相关分支的全仓库 pre-commit 已通过，后续 graph 回归测试的文件检查也通过。原始 JUnit、逐请求记录、源码指纹和音频已保留，但不将权重、音频或凭据加入公开记录。

上采样保留为“有初步收益但有取舍”；SM100 与 MoE 保留为“进行中”，不能归成无收益。完整配对 EN/ZH ASR、音色相似度、听感和长生成质量评测尚未完成，也未创建新的 PR。后续性能测量必须等用户明确允许，并暂停其他 GPU 操作后进行。

## RTX PRO 6000，C16，2026-11-25

RTX PRO 6000，FP8，C16。保持 overlap，把 vocoder 移到独立进程后吞吐 26.50 → 28.22 音频秒/秒（+6.48%），首包 P99 508.4 → 458.1 ms。相同独立进程布局下 stateful 吞吐 −1.05%，首包 P95 +12.37%。 [服务对照](https://github.com/EddieTzy/sglang-omni/tree/baeb84306054aca240062996bdfc38d82786e327/benchmarks/drama3_sm120_first_audio/RESULTS.md) · [原始 trace](https://github.com/EddieTzy/sglang-omni/tree/baeb84306054aca240062996bdfc38d82786e327/benchmarks/drama3_sm120_first_audio/traces)

此项作为已有部署配置的实测记录保留，暂缓提交上游 PR。CPU 竞争对首包的影响尚未单独量化。

## RTX PRO 6000 的 MoE 与线性注意力投影（2026-11-27）

线上 C4 首窗，预处理完成到 TTS 引擎交付音频码的中位时间为 162.2 ms，声码器计算为 3.3 ms。引擎累计 GPU kernel 时间中，两种 BF16 GEMM 共占 37.1%，FP8 专家 GEMM 占 21.4%。

- 小批量专家计算从默认 64 行分块切到 16 行，单独比较 A/B/B/A 四次服务启动。C4 吞吐 13.054 → 13.313 音频秒/秒，平均 RTF 0.2980 → 0.2905。C16 吞吐 27.363 → 27.716，平均 RTF 0.5651 → 0.5587。C16 首包 P95 波动较大，英文 WER 均为 0.44%，音色相似度 60.70 → 60.14。端到端吞吐提升较小，实验保留归档。[已关闭的 MoE PR](https://github.com/sgl-project/sglang-omni/pull/4715) · [全部请求、质量与九份 trace](https://github.com/EddieTzy/sglang-omni/tree/f49fbbea8e2aa09c07f495c476b8d6455f12fcd6/benchmarks/drama3_sm120_moe)
- 线性注意力的两个 BF16 输入投影实际在两个 CUDA 流上重叠。合并后，4 行输入每层 36.62 → 36.49 µs。四个行数的改善都小于 0.40 µs/层，单独归档。[微基准及两份 trace](https://github.com/EddieTzy/sglang-omni/tree/d78def4264b9adad95bbb2d3fb6c8e8c47ab63f7/benchmarks/drama3_sm120_gdn)
