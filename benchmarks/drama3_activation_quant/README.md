# Drama 3 expert activation and block-FP8 quantization

The live C4 trace repeatedly showed about 43 µs for the SiLU-and-multiply kernel after the first expert GEMM. This experiment checks whether eliminating the intermediate BF16 tensor also reduces the duration of a complete expert layer.

On one RTX PRO 6000 Blackwell Server Edition, the replay loads Drama 3's layer-0 FP8 expert weights, uses the default SGLang 0.5.23 Triton 3.8.0 tile, and draws six activation and top-8 routing samples per token count. The isolated activation tests use actual outputs of the first expert GEMM on those draws. The candidate uses the existing dense-mode fused group quantizer with group size 128 and supplies its FP8 values and FP32 scales to the down projection. The override applies only inside the benchmark process. Both paths use CUDA Graph replay; timings are medians of twelve blocks of 64 replays of six inputs. Each successive block reverses the timing order.

| Input token rows | Activation alone (µs) | Activation + quantization (µs) | Fused (µs) | Complete experts, baseline → fused (µs) | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.38 | 2.06 | 1.38 | 28.45 → 28.00 | 1.60% |
| 4 | 1.38 | 2.40 | 1.38 | 76.64 → 75.88 | 0.98% |
| 8 | 1.39 | 2.42 | 1.65 | 143.93 → 143.35 | 0.40% |
| 16 | 1.39 | 2.43 | 1.72 | 245.39 → 244.60 | 0.32% |

All 24 draws had byte-identical quantized activations, equal scales and equal final expert outputs. The separate diagnostic C4-shaped traces each replayed 48 complete expert calls. In the baseline trace the activation's median duration was 44.38 µs; in the candidate the fused quantization kernel at the same position measured 44.66 µs. The apparent long event persisted in that position even after eliminating the separate activation. Its origin remains unresolved. Timings with profiling off show that this fusion recovers only 0.75 µs per layer at four rows. We archived the result at the operator stage. Serving RTF, throughput, first-audio latency and audio quality were not evaluated for this change.

[Raw timings](result.json) · [Baseline Perfetto trace](experts_base.json.gz) · [Fused Perfetto trace](experts_fused.json.gz) · [Replay script](replay.py)

Run in the deployment Python environment with the model mounted at `/models/drama3` and a writable output directory at `/data/drama3-actq-1127`. Use `python replay.py`.
