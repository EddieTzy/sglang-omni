# Fish Audio Drama 3：B200 MoE up/down TMA 可选配置

状态：独立配置分支已保存，待当前依赖版本上的单项验证，未提 PR。主分类：in_progress。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/perf/drama3-b200-moe-configs)

## 为什么优化

block-FP8 专家矩阵在 B200 上可能需要不同 tile、pipeline 和数据搬运配置。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：MoE kernel 调优配置。

通过显式 SGLANG_MOE_CONFIG_DIR 选择 Triton 3.8.0 的 up/down TMA 配置；未设变量的当前默认才是独立性能对照，历史 non-TMA 配置不能冒充当前默认。

实现说明：分别提供 up/down 投影配置，显式选择；历史 non-TMA 配置明确标为历史对照，不作为当前上游默认。

代码位置：

- examples/configs/drama3_b200_moe/README.md：SGLANG_MOE_CONFIG_DIR, up/down TMA configuration

## 基线与复现条件

基线：f17a438cd753248d70916700aa4dccd9e7feb098

当前验证运行时：torch 2.14.1+cu130、sglang 0.5.23、triton 3.8.0、flashinfer-python 0.7.2、driver 580.173.02。设备为 B200；语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

已在相同 f17a438c 基线的独占 C64 服务 profile 发现 fused_moe_kernel 为最大单类热点，累计794.471ms（kernel 时长和，不是关键路径占比）。[基线证据](../drama3-quantizer-upsampling/profiles/service-20261118/README.md)。这支持优先检查本项，但本候选尚无优化后 profile、数值或独立性能结论，不能把热点大写成配置有收益。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：文档和静态检查通过；尚无当前版本的专家数值、独立吞吐或完整音质结论。旧 23.9% 是组合结果，不能计为本项收益。

## 正确性、质量与既有详细记录

提供 Triton 3.8.0、B200、block-FP8 专家形状对应的 up/down TMA 配置，通过 SGLANG_MOE_CONFIG_DIR 显式选择。目录中的 baseline 是历史 non-TMA 调优，不是当前上游默认；对比当前默认时应取消环境变量并重启服务。

目前仅完成文档与静态检查，当前运行时的专家输出对照、单项服务收益和完整质量评测均未完成。原分支 23.9% 属于旧基线的上采样与 MoE 组合结果，不能拆算给本项，也不能当作本次移植的实测结果。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

移植、主线适配与验证；原始优化来自 lamianlbe，保留归属，不计为本人原创

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
