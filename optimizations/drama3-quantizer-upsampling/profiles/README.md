# 上采样 profile 与归因

## 采集批次与边界

2026-11-18 使用真实 Drama3 codec 权重、既有 en-0 固定码本，截取前32帧，先通过完整基线 decoder 的前置 hook 抓取两个上采样层真实输入。再分别将相同输入复制为 batch1/16/64；批量项是受控形状实验，不是服务运行时 batch 分布，也不是客户端并发。每个模块先预热3次，再用 torch.profiler 采2次；base 与优化共享相同权重、输入和 FP16。

这是保留现有服务的共享 GPU 诊断，峰值 allocated 1907.9 MiB；不是无竞争吞吐或延迟评测。摘要保留 CPU/device 原始时间，但不据此计算可发布的加速比。只采到热算子路径，不能定位首次调用的初始化成本。

## 优化了哪部分

数学上：kernel=stride、无分组/填充/dilation 的转置卷积可按相位映射为等价 dense projection。代码把每帧输入映射到相位×输出通道，再恢复连续 NCT 布局；这是 codec 上采样算子与数据布局优化，没有直接修改请求调度。

profile 上：多数采样形状的基线使用 cuDNN implicit-GEMM dgrad 和3次布局转换，再单独加 bias；优化为1次 GEMM、1次 bias add、1次 contiguous copy。batch1 的线性路径出现 aten::mm，batch16/64 为 aten::bmm；没有证明跨 batch 已压成一个大矩阵。

## 形状与每次调用 kernel 数

计数由每段2次采样的 kernel 数除以2；不是 CPU operator 嵌套节点的计数。kernel 减少不直接等于耗时减少。

| 层 | 输入 NCT | 输出 NCT | Base kernel/次 | 优化 kernel/次 | 直接观测 |
| ---: | --- | --- | ---: | ---: | --- |
| 0 | [1, 1024, 32] | [1, 1024, 64] | 5 | 3 | 布局转换由3次变1次；保留 bias add |
| 0 | [16, 1024, 32] | [16, 1024, 64] | 5 | 3 | 布局转换由3次变1次；保留 bias add |
| 0 | [64, 1024, 32] | [64, 1024, 64] | 5 | 3 | 布局转换由3次变1次；保留 bias add |
| 1 | [1, 1024, 64] | [1, 1024, 128] | 2 | 3 | cuDNN 直接卷积；优化反而多一次 copy |
| 1 | [16, 1024, 64] | [16, 1024, 128] | 5 | 3 | 布局转换由3次变1次；保留 bias add |
| 1 | [64, 1024, 64] | [64, 1024, 128] | 5 | 3 | 布局转换由3次变1次；保留 bias add |

第二层 batch1 是明确反例：cuDNN 直接卷积加 bias 只需2个 kernel，优化后为3个。因此不写“全部形状减少 kernel”，也不能仅凭此认定它造成了那3个首包尖峰。6组输入的逐层输出最大误差均为0；这只描述本次输入，不替代完整 codec 的 FP16 误差边界。

## 证据层级

| 问题 | 当前证据 | 结论 |
| --- | --- | --- |
| 替换了什么 | 代码与局部 CUDA trace 一致 | 满足条件的 quantizer 转置卷积变为 phase projection |
| 少了哪些操作 | 5组形状 kernel 5→3；1组 2→3 | 算法与布局开销随形状变化 |
| 服务收益 | 独占单组 base→优化：C64 +11.10%，C1 -5.75% | 仅一组关联信号，未建立稳定收益 |
| 首包尖峰根因 | 3条请求占首音增加约93.5%；尚无旧请求的同步 profile | 不能断言编译、prefill 或调度中的任何一项已证实 |
| 调度和 GPU 气泡 | 完整服务独占 profile 已完成，见[服务归因报告](service-20261118/README.md) | 不从局部 kernel 计数推断 |

## 复现与原始资料

[operator-summary.json](operator-summary.json) 包含每个形状的 CPU/CUDA operator、kernel 名、计数、原始时间、trace SHA256 和源码指纹。每条 .json.gz 是压缩 Chrome trace，先解压再用 Perfetto 查看；trace 只有受控输入形状和算子信息，不含权重或音频。[采集脚本](../scripts/profile_quantizer.py) 需要原始目录结构、真实 codec 权重和既有固定码本；PROFILE_OUTPUT 指向新目录，不覆盖旧证据。

完整服务最终采用 C1预热16→profile8→控制64（预热16），C64预热64→profile64→控制128（预热64）；预热在 profile 窗口外。两臂独占采集已完成，原服务恢复且健康。完整的前后热点、GPU 活动、阶段边界、短控制和两次失败采样见[服务归因报告](service-20261118/README.md)。
