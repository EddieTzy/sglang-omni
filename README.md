# EddieTzy 推理优化与贡献记录

按状态查看进展，再按框架、模型和优化位置定位证据。代码分支已推送、PR 已创建、PR 已合并是三个不同状态；实现完成也不等于性能收益成立。

最后整理：2026-11-18（北京时间）。本次核实 #3283 已合并、#3408 已关闭，并更新 Drama3 拆分验证记录；其他历史实验不表示本次重新测试。

最新进展：三项 Drama3 独立候选已完成两轮流式性能和全量 EN/ZH 自动质量评分，38,176 条正式请求全部成功。C64 音频吞吐分别 +10.54% / +4.03% / +8.07%；SM100 中文错误率有上升风险。GPU 已释放。分支已推送，上游 PR 尚未创建。[最终验证总报告](records/drama3-final-20261118/README.md)。

当前 Drama3 验证结论及完整限制见最终报告；其他模型条目保留其历史条件。

## Drama 3 在 RTX PRO 6000 上的流式首包

RTX PRO 6000，FP8，C16。保持 overlap，把 vocoder 移到独立进程后吞吐 26.50 → 28.22 音频秒/秒（+6.48%），首包 P99 508.4 → 458.1 ms。相同独立进程布局下 stateful 吞吐 −1.05%，首包 P95 +12.37%。 两种模式各保留逐请求时间线，并另存 6 份 torch trace。此项为已有部署选项的对照。[详细数据与归档](https://github.com/EddieTzy/sglang-omni/tree/baeb84306054aca240062996bdfc38d82786e327/benchmarks/drama3_sm120_first_audio)

## 状态总览

| 分类 | 数量 | 归类规则 |
| --- | ---: | --- |
| [已合并](#已合并) | 1 | 已核实进入上游，不再列为待审核 |
| [已提PR](#已提pr) | 0 | PR 已创建但未合并；草稿、审核和 CI 状态在条目中说明 |
| [进行中或待验证](#进行中或待验证) | 4 | 尚缺正确性、单项性能、质量或问题定位证据 |
| [已测有收益或取舍](#已测有收益或取舍) | 2 | 已有实测信号，仍需决定是否推进；注明初筛与完整评测的区别 |
| [无稳定收益或暂不采用](#无稳定收益或暂不采用) | 3 | 无稳定端到端收益，或收益伴随不可接受的回退；保留具体原因 |
| [已关闭或归档](#已关闭或归档) | 2 | 已关闭的 PR 或被替代的历史组合分支 |

## 已合并

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | 调度与 GPU 反馈状态 | 已于 2026-11-04 合并，上游提交 8da1c6c0；历史性能与质量口径见 PR。 | [PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283) · [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode) · [详情](optimizations/fish-async-decode/README.md) · [记录](records/fish-s2-pro.md) |

## 已提PR

当前已登记条目中没有核实仍开放的 PR。Drama3 三个候选只是代码分支，尚未创建 PR。

## 进行中或待验证

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | FastAR KV 缓存清空 | 并发1/16/64 帧吞吐 +1.49%/+0.16%/+1.68%；复用历史基线并做短校准，仍需新基线配对复测。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fast-ar-cache-reset) · [详情](optimizations/fish-fast-ar-cache-reset/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | 语义头与 FastAR 清空及结果收集组合 | 短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined) · [详情](optimizations/fish-combined/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | SM100 sampler / fused tower 支持 | 两轮并发1/16/64吞吐 +14.67%/+12.33%/+4.03%；中文错误率上升，暂不建议直接合并；保留完整套件失败。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/feat/drama3-sm100-kernels-pr) · [详情](optimizations/drama3-sm100-kernels/README.md) · [记录](records/drama3.md) |
| SGLang / Fish Audio Drama 3 | B200 MoE up/down TMA 可选配置 | 真实专家形状数值检查通过；两轮并发1/16/64吞吐 +2.02%/+8.76%/+8.07%；全量自动质量评分完成，可准备可选配置 Draft。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-moe-configs-pr) · [详情](optimizations/drama3-b200-moe-configs/README.md) · [记录](records/drama3.md) |

## 已测有收益或取舍

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | FastAR 短序列 attention 批处理 | 完整扫描并发1/16/64 帧吞吐变化 -0.16%/+3.39%/+10.27%；详细首包延迟和质量变化见实验报告，不据此自动提 PR。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention) · [详情](optimizations/fish-batched-attention/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | Quantizer phase projection 上采样 | 两轮并发1/16/64吞吐 +1.53%/+5.61%/+10.54%；正确性与全量自动质量评分完成，可准备 Draft。早期初筛另存历史。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-quantizer-upsample-pr) · [详情](optimizations/drama3-quantizer-upsampling/README.md) · [记录](records/drama3.md) · [实测报告](records/drama3-quantizer-b200-20261118.md) |

## 无稳定收益或暂不采用

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | 解码整数状态 Triton 融合 | 全量服务并发 1/16/64 帧吞吐 -0.15%/+0.02%/-0.36%；单段加速未形成端到端收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-state-fusion) · [详情](optimizations/fish-state-fusion/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | 固定请求状态槽位 | 局部吞吐 +2.75%，但并发64 首音中位延迟 +19.95%，且音色相似度下降；暂不采用。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fixed-slot-overlap) · [详情](optimizations/fish-fixed-slots/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | CPU 结果收集与 GPU 快照 | 并发1/16/64 帧吞吐 +0.17%/-0.67%/+0.31%；无稳定服务收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-step-outputs) · [详情](optimizations/fish-batched-outputs/README.md) · [记录](records/fish-s2-pro.md) |

## 已关闭或归档

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | SlowAR 输出投影 | 帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。 | [PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408) · [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head) · [详情](optimizations/fish-semantic-head/README.md) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | SM100 kernel 支持、quantizer 上采样与 MoE 调优 | 历史组合记录，不代表当前三个独立候选的验证状态。17 项隔离 CPU 检查与原始 23.9% 组合收益保留为历史；当前结果见拆分条目。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe) · [详情](optimizations/drama3-quantizer-moe/README.md) · [记录](records/drama3.md) |

## 每项优化的详细目录

- [全部优化目录](optimizations/README.md)：每项独立记录为什么优化、原理、代码位置、前后 profile、性能和最终取舍。
- [记录模板](optimizations/TEMPLATE.md) · [RTF、cost、首包与 profile 口径](records/performance-methodology.md)。cost 指每请求推理耗时。
- [Drama3 上采样](optimizations/drama3-quantizer-upsampling/README.md)：当前重点，已有服务指标和局部算子轨迹；完整服务归因已补齐，保留尚未定位的旧首包长尾。

## 按模型查看详细记录

- [Fish S2-Pro](records/fish-s2-pro.md)：历史数据来自 RTX 5090，保留全部实验，包括关闭和不采用的方案。
- [Fish Audio Drama 3](records/drama3.md)：B200 上的独立上采样、SM100 支持和 MoE 配置，不将组合收益拆算给单项。
- [上采样单组性能与长尾分析](records/drama3-quantizer-b200-20261118.md)。
- [结构化索引](records/entries.json)：workflow_category 表示主分类；具体 PR 状态、验证强度和结论分别保留。

## vLLM

vLLM 的既有工作将在确认对应模型、本人实现的组件和仓库记录后补入同一索引。项目内集成与 vllm-project/vllm 上游贡献分别记录；借鉴的实现不计为本人原创。

## 维护规则

每项详细目录记录为什么优化、优化原理、base 与优化后的 profile、RTF、每请求 cost、首包、代码来源与本人工作、独立分支、精确基线、硬件和依赖、正确性、性能、质量、最终决定和下一步。每次更新只移动有新证据的条目；历史结果不冒充当前版本实测。

只把真实服务评测支持的结果写成收益；注明单组初筛、重复实验和完整质量评测。无显著收益、存在回退和未测试是不同结论。不同基线、硬件或采样配置的提升不能直接相加，慢请求也不能为美化结果而删除。

性能测试须在用户明确允许并暂停其他 GPU 操作后运行。公开记录不包含模型权重、数据集音频、生成音频、凭据或内部源码。
