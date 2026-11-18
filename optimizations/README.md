# 每项优化的详细记录

每个目录记录动机、优化位置与原理、基线、前后 profile、RTF/cost/首包、正确性、质量和取舍。[统一口径](../records/performance-methodology.md)。

- [Fish S2-Pro / 调度与 GPU 反馈状态](fish-async-decode/README.md)：merged。
- [Fish S2-Pro / 解码整数状态 Triton 融合](fish-state-fusion/README.md)：not_adopted。
- [Fish S2-Pro / 固定请求状态槽位](fish-fixed-slots/README.md)：not_adopted。
- [Fish S2-Pro / SlowAR 输出投影](fish-semantic-head/README.md)：archived。
- [Fish S2-Pro / FastAR KV 缓存清空](fish-fast-ar-cache-reset/README.md)：in_progress。
- [Fish S2-Pro / CPU 结果收集与 GPU 快照](fish-batched-outputs/README.md)：not_adopted。
- [Fish S2-Pro / 语义头与 FastAR 清空及结果收集组合](fish-combined/README.md)：in_progress。
- [Fish S2-Pro / FastAR 短序列 attention 批处理](fish-batched-attention/README.md)：measured_tradeoff。
- [Fish Audio Drama 3 / SM100 kernel 支持、quantizer 上采样与 MoE 调优](drama3-quantizer-moe/README.md)：archived。
- [Fish Audio Drama 3 / Quantizer phase projection 上采样](drama3-quantizer-upsampling/README.md)：measured_tradeoff。
- [Fish Audio Drama 3 / SM100 sampler / fused tower 支持](drama3-sm100-kernels/README.md)：in_progress。
- [Fish Audio Drama 3 / B200 MoE up/down TMA 可选配置](drama3-b200-moe-configs/README.md)：in_progress。
