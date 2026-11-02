# Fish S2-Pro 优化记录

项目：sgl-project/sglang-omni。框架：SGLang。

以下数据来自已保留的 RTX 5090 实验，不代表 H100 性能。常用完整评测口径为流式参考音频合成、自然 EOS、并发 1/16/64 性能扫描和 EN 1088 / ZH 2020 全集质量评测；小基准或短测在对应条目注明。

## 调度与 GPU 反馈状态

状态：开放，已获批准，等待合并。

固定 128-token 非流式小基准：并发 1/4/8 请求吞吐 +7.7%/+8.8%/+9.7%。不外推为所有自然长度流式请求的收益。

实现：[ perf/fish-async-decode ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode)。

上游：[PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283)。

## 解码整数状态 Triton 融合

状态：实现并测试，未采用。

全量服务并发 1/16/64 帧吞吐 -0.15%/+0.02%/-0.36%；单段加速未形成端到端收益。

实现：[ perf/fish-decode-state-fusion ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-state-fusion)。

## 固定请求状态槽位

状态：实现并测试，暂不采用。

相对现有 async：并发16 帧吞吐 +2.75%，并发64 首包中位延迟 +19.95%；英文/中文音色相似度下降 0.244/0.122 分。

实现：[ perf/fish-fixed-slot-overlap ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fixed-slot-overlap)。

## SlowAR 输出投影

状态：PR 已主动关闭，保留实验。

帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。

实现：[ perf/fish-semantic-head ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head)。

上游：[PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408)。

## FastAR KV 缓存清空

状态：实现并测试，待复现收益。

并发1/16/64 帧吞吐 +1.49%/+0.16%/+1.68%；复用历史基线并做短校准，仍需新基线配对复测。

实现：[ perf/fish-fast-ar-cache-reset ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fast-ar-cache-reset)。

## CPU 结果收集与 GPU 快照

状态：实现并测试，未采用。

并发1/16/64 帧吞吐 +0.17%/-0.67%/+0.31%；无稳定服务收益。

实现：[ perf/fish-batched-step-outputs ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-step-outputs)。

## 语义头与 FastAR 清空及结果收集组合

状态：仅完成短测。

短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。

实现：[ perf/fish-decode-combined ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined)。

## FastAR 短序列 attention 批处理

状态：原型验证中。

对比逐请求 FlashInfer、批量 SDPA 和批量 FlashInfer；尚无服务收益结论。

## 已保存的完整实验报告

- [整数状态融合](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-state-fusion/benchmarks/fish_decode_state_fusion_experiment/standard_eval_20261031/RESULTS.txt)
- [固定请求槽位](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-fixed-slot-overlap/benchmarks/fish_fixed_slots_experiment/standard_eval_20261031/RESULTS.txt)
- [语义输出头](https://github.com/EddieTzy/sglang-omni/blob/experiment/fish-semantic-head/benchmarks/fish_semantic_head_experiment/standard_eval_20261031/RESULTS.txt)
- [批量结果收集](https://github.com/EddieTzy/sglang-omni/blob/experiment/fish-batched-step-outputs/benchmarks/fish_collect_experiment/standard_eval_20261101/RESULTS.txt)

报告中的音色相似度为 WavLM cosine ×100；英文 WER 与中文 CER 的变化用百分点表示。动态批处理和随机采样下的质量比较不等价于逐帧一致性检查。
