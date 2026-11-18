# Fish Audio Drama 3：SM100 sampler / fused tower 支持

状态：GPU 相关检查已完成，严格一致性问题和独立收益待确认，未提 PR。主分类：in_progress。

[总索引](../../README.md) · [指标与归因口径](../../records/performance-methodology.md) · [代码分支](https://github.com/EddieTzy/sglang-omni/tree/feat/drama3-sm100-kernels)

## 为什么优化

既有 sampler 与 fused tower 的架构门控未开放 SM100，B200 可能落入通用路径。

这是优化动机；只有以下实测和 profile 支持的部分才作为已确认瓶颈或收益。

## 优化位置与原理

优化层面：架构分派与 sampler/tower kernel。

开放已有兼容 kernel 的架构分派，保持形状、张量并行和推理模式限制；这首先是硬件支持适配，必须独立验证数值与端到端收益。

实现说明：开放既有 SM100 sampler 与 fused tower 路径，并适配 SGLang 0.5.23 的逐层 attention 测试 hook；未降低断言。

代码位置：

- sglang_omni/models/fishaudio_drama3/sampling_kernels.py：architecture dispatch
- tests/unit_test/fishaudio_drama3/graph_recording_model.py：attention instrumentation for SGLang 0.5.23

## 基线与复现条件

基线：f17a438cd753248d70916700aa4dccd9e7feb098

当前验证运行时：torch 2.14.1+cu130、sglang 0.5.23、triton 3.8.0、flashinfer-python 0.7.2、driver 580.173.02。设备为 B200；语料、采样、预热和重复次数见下方已有记录。原始资料缺少的字段标为待补，不跨设备、分支或运行版本合并收益。代码已推送不表示 PR 已提交。

## Base 与优化后 profile

当前本目录没有核实可恢复的前后服务 trace；保留现有测量结论，调度/气泡/算子归因仍需独立证据，不用设计原理代替 profile。

后续逐项对照服务阶段关键路径、CPU/CUDA 同步、动态 batch/形状、kernel、内存拷贝和空闲区间。明确 cold/warm 与 profiler 开销，并保留反例。

## 性能：RTF、cost 与首包

现有数字与历史报告见下文；未提取或未测的 RTF、cost、首包不填 0，也不从吞吐倒算。后续补齐统一的逐请求统计。

已有结论：Sampler 38 项、tower 31 项通过；完整 Drama3 套件 833 通过、4 跳过、21 失败。无独立性能结论。

## 正确性、质量与既有详细记录

开放已有 Triton sampler 和 fused tower 的 B200 路径，保留模型、形状、确定性推理和张量并行的回退条件。测试辅助代码适配 SGLang 0.5.23 的逐层 attention hook，没有降低数值或状态断言。

Sampler 模块 38 项通过，tower 模块 31 项通过。完整 Drama3 套件包含 benchmark 标记的 graph 检查：833 项通过、4 项跳过、21 项失败、0 项准备错误。未修改上游另加 channels-last 卷积套件的结果为 767 项通过、67 项跳过、19 项失败、28 项准备错误；测试集合和架构跳过条件不同，不能直接相减作为修复数量。

上游的准备错误来自旧 attention hook。适配后仍有两个严格 graph/eager 检查失败；保存的禁用 sampler 和 tower 控制也存在 semantic-logit 差异，分析的主场景中调度、输入帧、生成帧一致。这缩小了问题范围，但没有完全解释或豁免严格失败。

B200 的默认 FA3 增量 codec 路径报不支持，另有 stage/memory、CPU-without-Triton 和严格 vocoder 检查的基线问题；一次 stateful capture 失败还需隔离复现。因此不宣称全套通过，也没有本候选的独立吞吐或完整音质结论。

## 归因边界与决定

已观测指标、代码机制和未证实的因果分别记录。当前分类和已有记录中的取舍继续有效；没有 profile 或质量证据时不写成已解决。

## 来源与后续工作

移植、主线适配与验证；原始优化来自 lamianlbe，保留归属，不计为本人原创

补齐尚缺的精确复现字段、前后 profile、逐请求 RTF/cost/首包及质量证据后，再判断是否推进 PR；无收益和回退实验同样保留。
