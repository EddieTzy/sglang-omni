# EddieTzy 推理优化与贡献记录

按框架、模型和优化位置记录实现、评测及上游贡献，保留未获得收益的实验。PR、已合并贡献与实验分支分别标注；实现完成不等于性能收益成立。

最后核对时间：2026-11-03T04:55:19+00:00。

## 按框架和模型查找

| 框架 | 模型 | 优化位置 | 当前结论 | 记录 |
| --- | --- | --- | --- | --- |
| SGLang / sglang-omni | Fish S2-Pro | 调度与 GPU 反馈状态 | 开放，已获批准，等待合并 | [PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283) · [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode) |
| SGLang / sglang-omni | Fish S2-Pro | 解码整数状态 Triton 融合 | 实现并测试，未采用 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-state-fusion) |
| SGLang / sglang-omni | Fish S2-Pro | 固定请求状态槽位 | 实现并测试，暂不采用 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fixed-slot-overlap) |
| SGLang / sglang-omni | Fish S2-Pro | SlowAR 输出投影 | PR 已主动关闭，保留实验 | [PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408) · [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head) |
| SGLang / sglang-omni | Fish S2-Pro | FastAR KV 缓存清空 | 实现并测试，待复现收益 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fast-ar-cache-reset) |
| SGLang / sglang-omni | Fish S2-Pro | CPU 结果收集与 GPU 快照 | 实现并测试，未采用 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-step-outputs) |
| SGLang / sglang-omni | Fish S2-Pro | 语义头与 FastAR 清空及结果收集组合 | 仅完成短测 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined) |
| SGLang / sglang-omni | Fish S2-Pro | FastAR 短序列 attention 批处理 | 完整性能扫描完成，全集质量评测进行中 | [实现分支](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention) |

详细过程与评测见 [Fish S2-Pro 优化记录](records/fish-s2-pro.md)。结构化索引见 [entries.json](records/entries.json)。

## vLLM

vLLM 的既有工作将在确认对应模型、本人实现的组件和仓库记录后补入同一索引。项目内集成与 vllm-project/vllm 上游贡献分别记录；借鉴的实现不计为本人贡献。

## 维护方式

每次实验更新框架与项目、模型、优化位置、触发的性能问题、实现分支、正确性验证、性能与质量结果、最终决定。PR 状态单独核对，区分开放、批准、合并与关闭。

只将真实服务评测支持的结果写成收益。小基准、短测、完整服务评测分别注明；不同基线、硬件或采样配置的提升不直接相加。公开记录不包含模型权重、数据集音频、生成音频或内部源码。
