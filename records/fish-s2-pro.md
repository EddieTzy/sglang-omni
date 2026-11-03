# Fish S2-Pro 优化记录

项目：sgl-project/sglang-omni。框架：SGLang。

以下数据来自已保留的 RTX 5090 实验，不代表 H100 性能。常用完整评测口径为流式参考音频合成、自然 EOS、并发 1/16/64 性能扫描和 EN 1088 / ZH 2020 全集质量评测；小基准或短测在对应条目注明。

## 调度与 GPU 反馈状态

状态：开放，已获批准，等待合并。

将下一步依赖的码本、历史与采样计数更新提前到 GPU，并为延后结果收集保留独立快照，使下一步 GPU 推理可与上一步 CPU 收集重叠；prefill 和请求行变化后重新同步。

固定 128-token 非流式小基准：并发 1/4/8 请求吞吐 +7.7%/+8.8%/+9.7%。不外推为所有自然长度流式请求的收益。

实现：[ perf/fish-async-decode ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-async-decode)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，decode_codebooks。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，async decode launch and resolve。

代码位置：[engine_builder.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-async-decode/sglang_omni/models/fishaudio_s2_pro/engine_builder.py)，Fish engine setup。

上游：[PR #3283](https://github.com/sgl-project/sglang-omni/pull/3283)。

## 解码整数状态 Triton 融合

状态：实现并测试，未采用。

将窗口历史写入、计数推进和码本反馈合并为一个整数 Triton kernel，保持采样与随机数调用不变。原路径已被 CUDA Graph 捕获，因此单段 kernel 减少不能直接等同于 CPU 派发收益。

全量服务并发 1/16/64 帧吞吐 -0.15%/+0.02%/-0.36%；单段加速未形成端到端收益。

实现：[ perf/fish-decode-state-fusion ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-state-fusion)。

代码位置：[decode_state.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-state-fusion/sglang_omni/models/fishaudio_s2_pro/decode_state.py)，advance_decode_state。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-state-fusion/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，decode_codebooks。

## 固定请求状态槽位

状态：实现并测试，暂不采用。

按请求 KV 池槽位保存 GPU 反馈状态，解耦 packed batch 行号与请求身份，并为图填充行分配独立槽位，以减少请求换位导致的 lookahead 退出。

相对现有 async：并发16 帧吞吐 +2.75%，并发64 首包中位延迟 +19.95%；英文/中文音色相似度下降 0.244/0.122 分。

实现：[ perf/fish-fixed-slot-overlap ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fixed-slot-overlap)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-fixed-slot-overlap/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，select_decode_state。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-fixed-slot-overlap/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，request-slot binding and prefill restore。

## SlowAR 输出投影

状态：PR 已主动关闭，保留实验。

将 tied 输出投影缩小到语义 token 与 EOS 的活跃权重视图，再恢复原词表形状及非法位置的负无穷值；保留完整输入 embedding、原 token ID 和采样器。

帧吞吐并发1 +2.98%、16 +1.10%、64 +0.29%；英文 WER +0.0795 个百分点，质量不退化尚未证明。

实现：[ perf/fish-semantic-head ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-semantic-head)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-semantic-head/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，S2ProSGLangTextModel.forward。

上游：[PR #3408](https://github.com/sgl-project/sglang-omni/pull/3408)。

## FastAR KV 缓存清空

状态：实现并测试，待复现收益。

FastAR 每帧按顺序覆盖当前有效 KV 前缀后，只读该前缀，因而移除每帧的全缓存清零；保留初始化与预热所需的 reset 接口。

并发1/16/64 帧吞吐 +1.49%/+0.16%/+1.68%；复用历史基线并做短校准，仍需新基线配对复测。

实现：[ perf/fish-fast-ar-cache-reset ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-fast-ar-cache-reset)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-fast-ar-cache-reset/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，decode_codebooks。

## CPU 结果收集与 GPU 快照

状态：实现并测试，未采用。

将逐请求的输出码本及反馈复制改为一次 batch 快照，并让每个请求持有该快照的行视图，保持后续模型缓冲覆盖时已收集结果不变。

并发1/16/64 帧吞吐 +0.17%/-0.67%/+0.31%；无稳定服务收益。

实现：[ perf/fish-batched-step-outputs ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-step-outputs)。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-batched-step-outputs/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，collect_s2pro_step_outputs。

## 语义头与 FastAR 清空及结果收集组合

状态：仅完成短测。

在独立基线上组合语义输出头、FastAR 清零移除和 batch 结果快照，检查收益是否叠加；没有把单项百分比直接相加。

短测并发1/16/64 帧吞吐 +2.49%/+0.17%/+1.67%；未完成全量质量评测，不作为可合并结论。

实现：[ perf/fish-decode-combined ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-decode-combined)。

代码位置：[sglang_model.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-combined/sglang_omni/models/fishaudio_s2_pro/sglang_model.py)，forward, decode_codebooks。

代码位置：[model_runner.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-combined/sglang_omni/models/fishaudio_s2_pro/model_runner.py)，collect_s2pro_step_outputs。

## FastAR 短序列 attention 批处理

状态：完整性能扫描完成，全集质量评测进行中。

将逐请求 single_prefill 调用改为一次批量 FlashInfer decode，把既有 KV 缓冲直接解释为每请求一页，缓存图兼容 plan，并在 GPU 更新有效页长度。保留单请求和 FA3 计算分支。

完整扫描并发1/16/64 帧吞吐变化 -0.16%/+3.39%/+10.27%；首包 P95 变化 +0.10%/-8.20%/-5.24%。质量结论待全集评分。

实现：[ perf/fish-batched-fast-ar-attention ](https://github.com/EddieTzy/sglang-omni/tree/perf/fish-batched-fast-ar-attention)。

代码位置：[audio_decoder.py](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-batched-fast-ar-attention/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/audio_decoder.py)，flashinfer_kvcache_attention, fast_ar_batch_decode_plan。

评测基线为 3b394b4193a8b76a6b5f52bf99da2a2729d6161e，本地被测提交为 5d52af12b1ac8b26bf82e2b2501ff94e81386cb5。GitHub 备份提交 55d5fc62e70dd4954caea65cf5e24b93a88f6ab8 基于 fork 已有基线 7537f89baca54d19de2b7afe5465bb614c10b625；Fish 模型子树逐字相同，但整个运行时基线不同。该备份方式用于避免提交与本次实现无关、且当前令牌无权更新的上游 workflow 历史。原始补丁 SHA256 为 00da284bf296508e0469c20626474dc64aab574a1d31f7023031256c6d42d87e，完整实测版本可按报告中的补丁或 bundle 恢复。

## 已保存的完整实验报告

- [整数状态融合](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-decode-state-fusion/benchmarks/fish_decode_state_fusion_experiment/standard_eval_20261031/RESULTS.txt)
- [固定请求槽位](https://github.com/EddieTzy/sglang-omni/blob/perf/fish-fixed-slot-overlap/benchmarks/fish_fixed_slots_experiment/standard_eval_20261031/RESULTS.txt)
- [语义输出头](https://github.com/EddieTzy/sglang-omni/blob/experiment/fish-semantic-head/benchmarks/fish_semantic_head_experiment/standard_eval_20261031/RESULTS.txt)
- [批量结果收集](https://github.com/EddieTzy/sglang-omni/blob/experiment/fish-batched-step-outputs/benchmarks/fish_collect_experiment/standard_eval_20261101/RESULTS.txt)

报告中的音色相似度为 WavLM cosine ×100；英文 WER 与中文 CER 的变化用百分点表示。动态批处理和随机采样下的质量比较不等价于逐帧一致性检查。
