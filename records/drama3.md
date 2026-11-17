# Fish Audio Drama 3 优化移植记录

项目：sgl-project/sglang-omni。框架：SGLang。

状态：已移植到最新官方 main，待开发机完整 GPU 测试；尚未创建 PR。

实现分支：[perf/drama3-quantizer-moe](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-quantizer-moe)。

实现提交：[a9d0aee8](https://github.com/EddieTzy/sglang-omni/commit/a9d0aee81a71b65203ece956e92114fb9d1e4a41)。

基线：[官方 main a338eb0e](https://github.com/sgl-project/sglang-omni/commit/a338eb0e6ac39c43bf1d08aad42dd8da9970198f)，于 2026-11-17 核对；依赖沿用该版本锁定的 SGLang 0.5.23。分支保留主线新的 16-warp sampler、strided output 和 fused tower 的张量并行回退限制。

## 修改与来源

- 为 SM100 开放已有 Triton sampling 和 fused codebook tower 路径，更新架构覆盖测试，并将 graph 测试适配至 SGLang 0.5.23 的逐层 attention hook。
- 将满足条件的 quantizer 转置卷积转换为 phase-ordered dense projection，保持输出采样位置。自动替换仅适用于 B200 / SM100、FP16、无分组、无 dilation、无 padding、无 output padding，且 kernel size 等于 stride 的情况。
- 保存可选的 B200 MoE up/down 独立 TMA 配置及原始对照配置。配置针对 Triton 3.8.0，通过 SGLANG_MOE_CONFIG_DIR 显式启用；其他版本需重新调优。
- 补充卷积适用条件测试，以及最新基线上的 GPU 回归和服务评测说明。未包含外部 buffered GDN 原型。

原始实现作者为 lamianlbe，来源提交为 [1f7e3317](https://github.com/lamianlbe/sglang-omni/commit/1f7e33177c17fb243262a3296b69147fd1450c60) 和 [b82e2ad9](https://github.com/lamianlbe/sglang-omni/commit/b82e2ad974fefa068e65583ba537e5f59062b4b7)。本条记录的是移植、主线适配与验证准备，不将原始优化计为本人原创；实现提交保留共同作者归属。

## 当前验证

- 直接从源码提取实际 PhaseLinearUpsample 类及其测试函数，隔离可选依赖后运行，17 项 CPU 数值与条件检查通过。CPU PyTorch 为 2.14.0；不替代目标 CUDA 依赖上的验证。
- 最终修改文件的 pre-commit 检查通过。全仓库检查中的 Rust 格式检查因堡垒机缺少 cargo 未运行，其余检查通过或在自动格式化后通过。
- 标准 vocoder / sampler 测试模块在堡垒机收集受缺少 Fish codec 依赖及本地 CPU SGLang 环境限制；没有将这些测试记作通过，也未启动推理服务。
- 远端实现源码树与本地最终版本一致，tree SHA 为 db016d49c93a029b2e8f1740c2956338ecc9492b。

## 待测与 PR 状态

目前无新 GPU 性能或音质结论。源分支报告的 23.9% 吞吐提升属于原始基线上的 quantizer 与 MoE 组合结果，不能当作本次移植的实测收益。源记录中两项严格 graph/eager 一致性失败尚未解释，应在最新基线和候选分支上配对复现，不降低断言或标记为预期失败。

开发机需运行 sampler、tower、vocoder、channels-last convolution 和完整 Drama3 测试，再进行真实 checkpoint 的固定码本数值检查、吞吐、首音延迟、播放卡顿、长生成及音质评测。具体命令、启动配置与历史实验边界见[分支内说明](https://github.com/EddieTzy/sglang-omni/blob/perf/drama3-quantizer-moe/docs/cookbook/fishaudio_drama3_b200.md)。

PR：未创建；等待完整测试后决定是否提交、是否拆分 kernel enablement 与 quantizer / MoE 两部分。
