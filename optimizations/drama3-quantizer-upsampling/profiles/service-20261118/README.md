# Drama3 上采样：独占服务 profile 与诊断控制

## 结论

已确认优化位置是 quantizer 上采样的算子选择与布局转换。完整服务的目标 cuDNN direct-dgrad 路径在优化版中消失，局部固定输入轨迹同时证明了替换机制。没有证据把收益归因为减少调度气泡；本次 GPU 活跃区间比例反而略降。原先3个首包尖峰没有在当前热路径复现，具体根因仍未确定。

本轮无 profiler 的短控制在 C1 基本持平、C64 吞吐有小幅正向变化，但 C64 首包 p95 和 RTF 尾部回退。保持“有收益或取舍”的分类，不据此宣称普遍更快或已经适合合并。完整音质非劣效仍未完成。

## 实验与版本

运行编号 quantizer-service-profile-20261118T040421Z。base 为 f17a438cd753248d70916700aa4dccd9e7feb098；优化生产实现为 57c747e0。单张 B200、同一权重和运行时、相同 serve 参数：容量64、async decode、overlap codec、seed42、max_new_tokens2048、模型默认采样、自然 EOS。基线和优化各重启一次。

两臂模型源码指纹仅 channels_last_decoder.py 不同，启动命令逐字相同，详见 [manifest](manifests.json)。为控制工具开销，两臂施加相同诊断覆盖：Torch profiler 只启用 CUDA、关闭全线程和 shapes；benchmark 额外序列化已有 speech_outcome_id，用于客户端与服务阶段事件直接关联。这些诊断改动不在生产 PR 内，指纹见 [profiler 覆盖](light-profiler-overlay.json) 与 [请求身份覆盖](request-identity-overlay.json)。

每臂顺序：C1预热16 → profile C1正式8 → 无 profiler C1控制64（先预热16）→ C64预热64 → profile C64正式64 → 无 profiler C64控制128（先预热64）。四个 profile 点144条请求、两个并发控制合计384条请求，均成功、正常 stop、无预算截断；预热单列，不计正式性能样本。所有 profile 请求144/144匹配原始服务事件，各事件存储 complete=true、dropped_events=0。

这里的预热顺序、请求数和语料前缀与之前 C1/16/64 正式初筛不同，因此是诊断控制，不能替换旧表或组成跨轮加速比。C64只有两波请求，填充与排空比例较高，不代表长时间稳态容量。

## Base profile 看到了什么

C64 基线主要 kernel 按累计执行时长排序如下；这是 profiler 下的 kernel duration 之和，不是关键路径墙钟占比，存在并发重叠。

| Kernel | Base 累计时长 ms | 解释 |
| --- | ---: | --- |
| fused_moe_kernel | 794.471 | 本次最大单类热点，适合下一步独立检查 MoE 配置；热点大不证明配置有收益 |
| cuDNN dgrad2d_grouped_direct_kernel | 314.442 | 与局部上采样直接卷积签名匹配，优化版不再出现 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 144.119 | 自回归模型的递归状态计算 |
| act_and_mul_kernel | 106.841 | 激活与逐元素计算 |
| cublasLt splitKreduce_kernel | 91.235 | 矩阵计算的归约步骤 |

## 优化后 profile：哪些变了，哪些没证实

C1 的该 direct-dgrad kernel 为42次、53.148 ms，C64 为188次、314.442 ms；优化版两个点均为0次。基线的 NCHW↔NHWC 转换也在优化版相应签名中消失。其他 waveform decoder 的 dgrad kernel 仍存在，不能写成“模型完全不再用转置卷积”。

固定输入的[局部轨迹](../README.md)证明：多数形状 kernel 5→3，第二层 batch1 为2→3；线性层还保留 bias add 和 contiguous copy。这说明改变的是特定卷积算法及布局路径，不能简单写成全部算子融合或全部 kernel 变少。新增 GEMM 的全部耗时不能仅凭通用 kernel 名从完整模型中准确拆出，因此不捏造纯上采样的服务加速比。

C1 profile 逐条 completion tokens 一致，总数均785、平均音频均4.557s；C64 为5773→5860 tokens、平均音频4.189→4.252s，输出工作量不完全相同。原始 kernel 计数和分布见 [analysis.json](analysis.json)。

## GPU 活跃与空隙

以事件中的首个请求 admission 到最后请求事件为区间，按 trace 的 baseTimeNanoseconds 将两个 worker 进程时间转换为同机 epoch；对全部 kernel/memcpy/memset 区间取并集。不是相加各 kernel 的时长，也不是 SM 利用率。profile 期间无其他 GPU 工作；不把导出阶段计入请求区间。

| Profile 并发 | Base 请求区间 ms | 优化请求区间 ms | Base GPU 活跃比例 | 优化 GPU 活跃比例 | Base 最大无活动空隙 ms | 优化最大空隙 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4638.979 | 4639.692 | 92.02% | 91.69% | 10.614 | 8.654 |
| 64 | 2659.314 | 2532.883 | 90.90% | 90.07% | 12.813 | 22.607 |

这些数据不支持“优化压缩调度气泡”的结论。空隙可能来自 CPU、IPC、请求边界、依赖或未捕获活动；CUDA-only 轨迹没有完整 CPU 调用栈，不能仅凭空白定责。C64 GPU 活跃区间缩短与减少算子工作相符，但单轮且生成长度不同，不能把全部服务变化精确拆算到某一项。

## 首包前的阶段边界

下表为按同一请求 ID 计算的事件边界均值，单位ms。它们包含墙钟等待和阶段重叠，不是纯 GPU 计算时间；尤其 engine dispatch→first stream 可能含内部排队和生成。

| 并发 | 阶段边界 | Base ms | 优化 ms |
| ---: | --- | ---: | ---: |
| 1 | 预处理 dispatch→complete | 1.874 | 1.928 |
| 1 | 引擎 input→first stream | 24.816 | 24.477 |
| 1 | vocoder first input→first stream | 22.347 | 22.242 |
| 1 | 服务 admission→首音频到 coordinator | 52.435 | 51.949 |
| 64 | 预处理 dispatch→complete | 57.703 | 68.646 |
| 64 | 引擎 input→first stream | 231.140 | 238.000 |
| 64 | vocoder first input→first stream | 74.944 | 85.007 |
| 64 | 服务 admission→首音频到 coordinator | 367.817 | 397.230 |

热路径 C1 没有重现旧的0.4–1.2秒额外首包等待。本次增强预热和执行历史与原实验不同，因此只能说当前未复现，不能断言原问题已修复或一定是编译/初始化。C64 的首包阶段没有整体变快，服务 admission→首包均值反而增加约29ms；完成耗时改善不能替代首包改善。

流式 vocoder 的 stage_dispatch 在终态处理路径出现，可能晚于 first stream；因此没有使用 dispatch→first stream 作为 vocoder 首包耗时，而使用首个 stream input→首个 stream output。不能把不同生命周期事件硬拼成负耗时。

## 无 profiler 的诊断控制：cost、RTF、首包

每格 base → 优化。cost 为每请求 send→complete，单位秒；首包单位秒；RTF 为逐请求比值。完整 mean/p50/p95/p99、逐请求记录见 [base](controls-baseline.json) 与 [优化](controls-quantizer.json)。

| 并发 | 每臂请求 | cost mean | cost p95 | RTF mean | RTF p95 | 首包 p50 | 首包 p95 | 音频秒/秒 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.5550 → 0.5510 | 0.8790 → 0.8950 | 0.1324 → 0.1316 | 0.1533 → 0.1554 | 0.0529 → 0.0529 | 0.1653 → 0.1659 | 7.6650 → 7.7130 |
| 64 | 128 | 2.4570 → 2.2340 | 3.5040 → 3.2640 | 0.6376 → 0.5945 | 0.9676 → 1.0443 | 0.4490 → 0.3626 | 1.5960 → 2.0174 | 84.7700 → 86.9090 |

C1 音频吞吐约+0.63%，基本持平；C64 约+2.52%，但首包 p95 1.5960→2.0174s、RTF p95 0.9676→1.0443。保留回退，不用当前较小收益覆盖旧 C64 +11.10% 初筛，也不拿带 profiler 的数字替代这些控制。

## 失败采样与恢复

[失败记录](failed-attempts.json)原样保留。首次全线程 Torch trace 产生约6GB的引擎中间文件，导出期间紧接的控制请求触发阶段通信超时；这反映采样/导出干扰，不计为模型性能结果。第二次 CUDA-only 尝试成功导出 C1 的两个 worker trace，但控制脚本误按“三阶段=三文件”验收而超时；实际 engine/vocoder 共进程。修正为两个进程、缩短窗口、预热移出采样，并等待压缩完成后，第三次两臂全部完成。

每次失败均自动恢复原服务。最终原 tts-drama3 已恢复并确认 GET /health=200；验证容器已停止。[控制器最终状态](controller.json)保存原始完成与恢复记录。

## 证据、原始 trace 与复现

公开目录包含完整请求级事件压缩包、事件存储状态、脱敏逐请求控制指标、源码指纹、分析摘要和脚本。大型 CUDA Chrome trace 保留在 GPU 主机的 /home/tzy/drama3-pr-validation/results/serving/quantizer-service-profile-20261118T040421Z-{baseline,quantizer}/profile-c{1,64}/profile/；analysis.json 对每个 .trace.json.gz 记录准确路径、字节数、SHA256 和时钟基准，可解压后用 Perfetto 查看。没有将大型 trace、权重、参考音频、生成音频或凭据加入 Git。

[采集服务脚本](../../scripts/serve_profile.py)使用相同请求配置并调用现有 profiler API；[控制器](../../scripts/run_service_profile.py)负责临时停服和 finally 恢复；[分析脚本](../../scripts/analyze_service_profile.py)使用 ijson 流式解析，避免一次载入数GB文件。脚本依赖原有 /validation 数据与权重挂载、诊断源码覆盖层（[准备脚本](../../scripts/prepare_diagnostic_sources.py)记录相同的源码覆盖步骤）。分析使用 ijson 3.6.0；运行前必须安排独占窗口并确认服务可临时中断，不能只复制脚本就默认为无扰动操作。

## 下一步

暂不根据 kernel 数直接增加 batch 门控；先用真实流式窗口形状与固定输入复核小 batch 的取舍。若要定位旧首包尖峰，需要相同执行历史下的冷/热路径对照和更细的 CPU/编译事件；当前证据不够。MoE 已是本次 base 的主要热点，可先做独立数值验证和配置选择核对，再决定是否安排独占性能对照。SM100 路径仍需先解释正确性失败。完整生成质量验证继续作为 PR 前缺口。
