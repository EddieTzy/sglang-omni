## Motivation

On NVIDIA B200, Drama3's non-overlapping FP16 quantizer transposed convolutions use a cuDNN dgrad/layout-conversion path. Represent eligible operations as phase-ordered dense projections to reduce that cost.

## Modifications

- Restrict the fast path to SM100 FP16, kernel_size == stride, groups 1, dilation 1, and zero padding/output padding.
- Keep a separate quantizer upsampling sequence in ChannelsLastDecoder so incremental/stateful consumers can reuse the original shared codec.
- Cover phase/sample placement, bias, noncontiguous inputs, fallback, shared-codec state, and CUDA Graph replay.

Adapted from lamianlbe’s b82e2ad9; original attribution is retained. No sampler/tower enablement or MoE tuning is included.

## Related Issues

None.

## Accuracy Test

30 targeted tests passed, including 13 CUDA cases, actual optimized-path selection, shared-codec stateful reuse, and bitwise graph/eager replay. The same production implementation passed 16 EN/ZH fixed-codebook real-checkpoint reconstructions: max absolute waveform error 0.00323486328125 and min SNR 53.2953 dB, within the existing FP16 rtol=0.02/atol=0.02 tolerance. Reconstruction tolerances do not establish generated-audio equivalence.

Two full generation passes per arm: EN 1088 and ZH 2020 utterances per pass, C64, scored with Qwen3-ASR 1.7B and WavLM. No skipped or high-error filtered samples. Raw corpus error rates and cosine similarity ×100 (two-pass means):

| Language | Base error % | Candidate error % | Base SIM | Candidate SIM |
|---|---:|---:|---:|---:|
| en | 0.9001 | 0.9001 | 62.3678 | 62.6668 |
| zh | 0.6104 | 0.6412 | 73.4980 | 73.5497 |

Paired utterance bootstrap, 2000 resamples retaining both passes:
- en: error change +0.0000 percentage points, descriptive 95% interval [-0.0590, +0.0627].
- zh: error change +0.0308 percentage points, descriptive 95% interval [-0.0047, +0.0709].

These intervals are conditional on two passes and do not prove noninferiority; no acceptance margin was predefined and no full human listening study was performed. Both languages and both rounds are retained in the linked evidence. Changed-file pre-commit and git diff --check passed.

## Benchmark & Profiling

Base `8848e5003ba79923af1e175364cef6a7cdb20109`; candidate `054d0b1c77f0e78d24d6aeb8667dbb031c31d009`. B200, PyTorch 2.14.1+cu130, SGLang 0.5.23, Triton 3.8.0, FlashInfer 0.7.2, driver 580.173.02.

Streaming PCM, natural EOS; max_new_tokens=2048 is a safety ceiling and no formal request hit it. C1/C16/C64 use 64/512/1088 performance requests per arm per pass. Two passes per arm, with a shared installed-default baseline at both ends of the campaign; all candidate changes are independent. Warmup and profiler runs are excluded. All formal requests succeeded with observed stop. MoE baseline leaves SGLANG_MOE_CONFIG_DIR unset.

| Concurrency | Base audio s/s | Candidate audio s/s | Change | Base/candidate mean cost (s) | Base/candidate mean RTF | Base/candidate TTFA p95 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.288 | 7.399 | +1.53% | 0.5866/0.5779 | 0.1364/0.1343 | 0.1754/0.1749 |
| 16 | 51.467 | 54.352 | +5.61% | 1.3733/1.3028 | 0.3114/0.2949 | 0.3391/0.3782 |
| 64 | 98.648 | 109.046 | +10.54% | 2.8387/2.5683 | 0.6449/0.5829 | 0.9426/0.8811 |

Throughput is the mean of the two per-pass audio rates; latency quantiles pool per-request samples. Cost is client send through full stream completion; RTF divides cost by generated audio duration. TTFA measures the first nonempty PCM HTTP payload. Two passes do not establish device-independent gains; low-concurrency small improvements should be treated cautiously.

The C64 service trace removes the targeted direct-dgrad/layout path. Controlled-shape profiles showed 5→3 kernels for five shapes and 2→3 for one shape; kernel reduction is not universal. There is no established scheduling-bubble reduction.

C1 first-audio outliers occur in the baseline too. Per-request diagnostic boundaries locate the dominant delay in reference preprocessing, inside host-side cuDNN attention calls; they do not establish which internal initialization/planning/synchronization step is responsible. Do not attribute this tail to the candidate.

[Full experiment report, exact manifests, both rounds, p50/p95/p99, continuity, quality intervals, and profile analysis](https://github.com/EddieTzy/sglang-omni/blob/docs/contribution-log/records/drama3-final-20261118/README.md). Earlier screening and invalid no-op selection measurements are retained as history, excluded from this result. No combined-optimization gain is claimed.

## Checklist

- [x] Format changed files with pre-commit.
- [x] Record targeted numerical/correctness evidence.
- [x] Complete two-pass serving and EN/ZH automatic quality scoring.
- [x] Document opt-in conditions, limitations, and profiling evidence.
- [ ] Maintainer review and broader portability/quality acceptance.
