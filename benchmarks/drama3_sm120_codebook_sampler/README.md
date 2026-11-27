# Drama 3 codebook sampler on RTX PRO 6000 Blackwell Server Edition

The shipped serving path uses PyTorch codebook sampling on SM120. This experiment permits the existing Triton semantic and residual samplers in an isolated serving container on SM120. The shared tower kernels and the fused head projection retain their original hardware guards. The candidate is an experimental device-selection change. The text backbone, checkpoint and MoE configuration are identical across arms.

## Complete frame replay

The standalone test loads all audio decoder weights from the real FP8 Drama 3 checkpoint (decoder linears are BF16). Inputs are fixed synthetic hidden states, semantic logits, prior frame, history, and 64-position dense attention caches. Both arms use the same dense grouped-query attention stand-in for the production paged attention. They run the semantic selection, four shared layers, nine residual layers and nine subsequent codebook selections, with the same checkpoint geometry, layer inputs and sampling parameters at entry. Normal sampling can generate different subsequent codes. The measured unit is one complete codebook frame for the whole input batch. Graph replay alternates A/B order over twelve blocks of thirty frames after warmup. The standalone experiment runs outside the production model process.

| Token rows | PyTorch frame (ms) | Triton frame (ms) | Time reduction |
| ---: | ---: | ---: | ---: |
| 1 | 2.655 | 2.164 | 18.5% |
| 4 | 2.939 | 2.325 | 20.9% |

With top-k one, all 11 output fields for one row and 44 for four rows matched the baseline on the tested inputs. Normal-sampling performance uses top-k 30, top-p 0.9 and temperature 0.8. A separate four-row diagnostic trace for each arm captured five graph replays; the baseline contained 3,450 GPU kernel events and the candidate 2,100. [Measurement code](frame.py) · [All twelve blocks](frame_results.json) · [Baseline trace](torch.json.gz) · [Candidate trace](triton.json.gz)

## Service protocol

One RTX PRO 6000 Blackwell Server Edition (96 GB), TP1, SGLang 0.5.23, Triton 3.8.0, serving checkout f1c94f0d8772127aca6a93992c3240ded10a74af. Four independent warm server launches, A1, B1, B2, A2. The same image and checkpoint served FP8 Drama 3 under normal sampling, with separate vocoder, overlap enabled and engine capacity sixteen on both variants. Each arm uses the first 48 English SeedTTS requests at C4 and first 128 at C16 after four or sixteen warmup requests. The formal comparisons exclude profiling. The client uses the first audible payload as reported by benchmark_tts_seedtts. Each variant generates 352 scored waveforms if all requests complete. The shipped four-request engine limit is restored with the original container.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --stream --meta "$SEEDTTS_EN_META" --lang en --model fishaudio/drama-3 \
  --host 127.0.0.1 --port "$PORT" --concurrency "$C" --warmup "$C" \
  --max-samples "$N" --max-new-tokens 1000 --output-dir "$OUT"
```

## Interpreting the service and quality results

The B1 launch logged Triton codebook sampling, with both fused tower layers and fused codebook head sampling disabled on SM120. Normal TTS sampling uses different random streams on the two paths. PCM hashes are not an equivalence test for this mode. The arithmetic tests compare kernels with a reference using matched noise and cover top-k/top-p, repetition handling, ties and CUDA Graph replay. For that separate test run the arithmetic-test hardware precondition includes the actual (12,0) GPU; capability-routing tests are excluded. No numerical assertions or tolerances are modified.

The C16 B2 launch had 15 requests above 500 ms first audio, all among the first sixteen entries in dataset order. Those 16 ranged from 171 to 793 ms. The subsequent 112 were all at or below 266 ms. All are included in the reported percentiles. This localization alone does not establish the cause of the spike.

For English WER, sum substitutions, deletions and insertions over each variant and divide by its total reference word count. Speaker similarity uses the benchmark's native WavLM scale. The summary script also resamples prompt IDs in pairs across both launches to describe uncertainty on this English subset. It does not model hardware-to-hardware or launch-to-launch variation.

## Service measurements

| Load and launch | Audio s/s | Mean RTF | Mean latency (s) | First audio P50/P95 (ms) | Playback underrun P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| C4 A1 | 13.706 | 0.2830 | 1.204 | 116.4 / 184.8 | 0.0 |
| C4 B1 | 14.330 | 0.2719 | 1.145 | 115.3 / 176.4 | 5.2 |
| C4 B2 | 14.371 | 0.2686 | 1.155 | 111.9 / 165.4 | 0.0 |
| C4 A2 | 13.434 | 0.2886 | 1.251 | 115.1 / 178.5 | 0.0 |
| C16 A1 | 27.670 | 0.5582 | 2.196 | 158.8 / 321.2 | 264.3 |
| C16 B1 | 29.423 | 0.5261 | 2.090 | 146.5 / 395.9 | 132.3 |
| C16 B2 | 28.556 | 0.5429 | 2.162 | 160.4 / 780.3 | 175.8 |
| C16 A2 | 28.714 | 0.5384 | 2.139 | 156.0 / 333.5 | 157.4 |

All 704 formal requests completed with zero generation-cap hits. Averaged across two independent launches, C4 throughput improved 5.75%, mean RTF decreased 5.44%, mean request latency decreased 6.31% and first-audio P95 decreased 5.92%. The two matched throughput comparisons were +4.55% and +6.98%. C16 mean throughput improved 2.83%. Its two-launch mean of first-audio P95 increased from 327 to 588 ms, while playback underrun P95 decreased from 211 to 154 ms. Run-level percentiles are averaged, not pooled across requests. The data do not establish a general high-concurrency latency improvement.

## Quality and arithmetic

| Concurrency | Waveforms per variant | Original WER | Candidate WER | Original speaker similarity | Candidate speaker similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 96 | 4/1,084 (0.37%) | 4/1,084 (0.37%) | 62.67 | 61.62 |
| 16 | 256 | 12/2,732 (0.44%) | 11/2,732 (0.40%) | 60.65 | 59.96 |

Qwen3-ASR and the SeedTTS WavLM scorer covered all 704 files. The paired bootstrap 95% intervals for the candidate-minus-original speaker score include zero at both loads (C4 −2.40 to +0.25; C16 −1.42 to +0.05). These are exploratory results on the specific repeated English prompt set. The operator suite passed 31 arithmetic tests, with six hardware-selection tests excluded. It ran on the real SM120 GPU. Only the test fixture's allowed device condition was widened; reference draws, numerical assertions and tolerances were retained.

Given the C16 first-packet regression and lower observed similarity, the opt-in experiment stays in the fork. The original deployment was restored and its model endpoint verified. [Numerical results](summary.json) · [Per-sample quality](quality-scores.json) · [Recompute](summarize.py) · [Candidate diff](candidate.diff)
