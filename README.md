# EddieTzy 推理优化与贡献记录

按状态查看进展，再按框架、模型和优化位置定位证据。代码分支已推送、PR 已创建、PR 已合并是三个不同状态；实现完成也不等于性能收益成立。

最后整理：2026-11-18 11:40（北京时间）。本次核实 #3283 已合并、#3408 已关闭，并更新 Drama3 拆分验证记录；其他历史实验不表示本次重新测试。

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
| SGLang / Fish S2-Pro | 调度与 GPU 反馈状态 | 已于 2026-11-04 合并，上游提交 8da1c6c0；历史性能与质量口径见 PR。 | [PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283) · [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode) · [记录](records/fish-s2-pro.md) |

## 已提PR

当前已登记条目中没有核实仍开放的 PR。Drama3 三个候选只是代码分支，尚未创建 PR。

## 进行中或待验证

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | FastAR KV 缓存清空 | 并发1/16/64 帧吞吐 +1.49%/+0.16%/+1.68%；复用历史基线并做短校准，仍需新基线配对复测。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fast-ar-cache-reset) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | 语义头与 FastAR 清空及结果收集组合 | 短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | SM100 sampler / fused tower 支持 | Sampler 38 项、tower 31 项通过；完整 Drama3 套件 833 通过、4 跳过、21 失败。无独立性能结论。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/feat/drama3-sm100-kernels) · [记录](records/drama3.md) |
| SGLang / Fish Audio Drama 3 | B200 MoE up/down TMA 可选配置 | 文档和静态检查通过；尚无当前版本的专家数值、独立吞吐或完整音质结论。旧 23.9% 是组合结果，不能计为本项收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-moe-configs) · [记录](records/drama3.md) |

## 已测有收益或取舍

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | FastAR 短序列 attention 批处理 | 完整扫描并发1/16/64 帧吞吐变化 -0.16%/+3.39%/+10.27%；详细首包延迟和质量变化见实验报告，不据此自动提 PR。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | Quantizer phase projection 上采样 | B200 单组 base→优化初筛：并发1/16/64 音频吞吐 -5.75%/+3.77%/+11.10%；首音 p95 在并发1/16 变慢、64 变快。不是全场景提升，也不是无收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-quantizer-upsample) · [记录](records/drama3.md) · [实测报告](records/drama3-quantizer-b200-20261118.md) |

## 无稳定收益或暂不采用

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | 解码整数状态 Triton 融合 | 全量服务并发 1/16/64 帧吞吐 -0.15%/+0.02%/-0.36%；单段加速未形成端到端收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-state-fusion) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | 固定请求状态槽位 | 局部吞吐 +2.75%，但并发64 首音中位延迟 +19.95%，且音色相似度下降；暂不采用。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fixed-slot-overlap) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish S2-Pro | CPU 结果收集与 GPU 快照 | 并发1/16/64 帧吞吐 +0.17%/-0.67%/+0.31%；无稳定服务收益。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-step-outputs) · [记录](records/fish-s2-pro.md) |

## 已关闭或归档

| 框架 / 模型 | 改动 | 当前结果与决定 | 入口 |
| --- | --- | --- | --- |
| SGLang / Fish S2-Pro | SlowAR 输出投影 | 帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。 | [PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408) · [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head) · [记录](records/fish-s2-pro.md) |
| SGLang / Fish Audio Drama 3 | SM100 kernel 支持、quantizer 上采样与 MoE 调优 | 历史组合记录，不代表当前三个独立候选的验证状态。17 项隔离 CPU 检查与原始 23.9% 组合收益保留为历史；当前结果见拆分条目。 | [分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe) · [记录](records/drama3.md) |

## 按模型查看详细记录

- [Fish S2-Pro](records/fish-s2-pro.md)：历史数据来自 RTX 5090，保留全部实验，包括关闭和不采用的方案。
- [Fish Audio Drama 3](records/drama3.md)：B200 上的独立上采样、SM100 支持和 MoE 配置，不将组合收益拆算给单项。
- [上采样单组性能与长尾分析](records/drama3-quantizer-b200-20261118.md)。
- [结构化索引](records/entries.json)：workflow_category 表示主分类；具体 PR 状态、验证强度和结论分别保留。

## vLLM

vLLM 的既有工作将在确认对应模型、本人实现的组件和仓库记录后补入同一索引。项目内集成与 vllm-project/vllm 上游贡献分别记录；借鉴的实现不计为本人原创。

## 维护规则

每项记录问题、代码来源与本人工作、独立分支、精确基线、硬件和依赖、正确性、性能、质量、最终决定和下一步。每次更新只移动有新证据的条目；历史结果不冒充当前版本实测。

只把真实服务评测支持的结果写成收益；注明单组初筛、重复实验和完整质量评测。无显著收益、存在回退和未测试是不同结论。不同基线、硬件或采样配置的提升不能直接相加，慢请求也不能为美化结果而删除。

性能测试须在用户明确允许并暂停其他 GPU 操作后运行。公开记录不包含模型权重、数据集音频、生成音频、凭据或内部源码。
