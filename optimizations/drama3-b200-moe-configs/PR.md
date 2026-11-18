## Motivation

Provide opt-in B200 block-FP8 Drama3 expert configurations with separate up/down projection tuning, rather than requiring users to reuse tuning from another architecture.

## Modifications

- Add optional Triton 3.8.0 B200 configurations selected with SGLANG_MOE_CONFIG_DIR.
- Tune projection-specific tiling and TMA choices; leave installed defaults unchanged unless explicitly selected.
- Document selection, fallback, and controlled comparison. The historical baseline subdirectory is earlier non-TMA tuning, not the current installed baseline.

Adapted from lamianlbe’s original B200 tuning work; original attribution is retained. No quantizer or sampler/tower change is included.

## Related Issues

None.

## Accuracy Test

Independent CPU FP8-emulation comparisons passed for E=256, H=2048, I=512, top-k=8 at M=1/8/16/32/64/128/256/512 against both installed defaults and the candidate. Maximum reference row error was approximately 0.00317146; maximum candidate/default row-relative difference was approximately 0.00069151. Existing tolerances were retained. Initialization/harness failures preceding valid comparisons are recorded separately in the experiment history.

Two full generation passes per arm: EN 1088 and ZH 2020 utterances per pass, C64, scored with Qwen3-ASR 1.7B and WavLM. No skipped or high-error filtered samples. Raw corpus error rates and cosine similarity ×100 (two-pass means):

| Language | Base error % | Candidate error % | Base SIM | Candidate SIM |
|---|---:|---:|---:|---:|
| en | 0.9001 | 0.9043 | 62.3678 | 62.4423 |
| zh | 0.6104 | 0.6317 | 73.4980 | 73.5809 |

Paired utterance bootstrap, 2000 resamples retaining both passes:
- en: error change +0.0042 percentage points, descriptive 95% interval [-0.0464, +0.0554].
- zh: error change +0.0213 percentage points, descriptive 95% interval [-0.0166, +0.0642].

These intervals are conditional on two passes and do not prove noninferiority; no acceptance margin was predefined and no full human listening study was performed. Both languages and both rounds are retained in the linked evidence. Changed-file pre-commit and git diff --check passed.

## Benchmark & Profiling

Base `8848e5003ba79923af1e175364cef6a7cdb20109`; candidate `dc5e1dc490801dd4a9d1cc3c922067249e99f9a6`. B200, PyTorch 2.14.1+cu130, SGLang 0.5.23, Triton 3.8.0, FlashInfer 0.7.2, driver 580.173.02.

Streaming PCM, natural EOS; max_new_tokens=2048 is a safety ceiling and no formal request hit it. C1/C16/C64 use 64/512/1088 performance requests per arm per pass. Two passes per arm, with a shared installed-default baseline at both ends of the campaign; all candidate changes are independent. Warmup and profiler runs are excluded. All formal requests succeeded with observed stop. MoE baseline leaves SGLANG_MOE_CONFIG_DIR unset.

| Concurrency | Base audio s/s | Candidate audio s/s | Change | Base/candidate mean cost (s) | Base/candidate mean RTF | Base/candidate TTFA p95 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.288 | 7.435 | +2.02% | 0.5866/0.5711 | 0.1364/0.1341 | 0.1754/0.1736 |
| 16 | 51.467 | 55.974 | +8.76% | 1.3733/1.2657 | 0.3114/0.2868 | 0.3391/0.3767 |
| 64 | 98.648 | 106.606 | +8.07% | 2.8387/2.6297 | 0.6449/0.5967 | 0.9426/0.9778 |

Throughput is the mean of the two per-pass audio rates; latency quantiles pool per-request samples. Cost is client send through full stream completion; RTF divides cost by generated audio duration. TTFA measures the first nonempty PCM HTTP payload. Two passes do not establish device-independent gains; low-concurrency small improvements should be treated cautiously.

In the C64 diagnostic service trace, cumulative fused_moe_kernel GPU duration changed from 825.63 ms to 574.49 ms. Natural output length and scheduling affect aggregate counts, so this is mechanism evidence, not a controlled per-kernel speedup ratio. Improvements belong to the configuration bundle, including tiling; they cannot be attributed solely to TMA.

C1 first-audio outliers occur in the baseline too. Per-request diagnostic boundaries locate the dominant delay in reference preprocessing, inside host-side cuDNN attention calls; they do not establish which internal initialization/planning/synchronization step is responsible. Do not attribute this tail to the candidate.

[Full experiment report, exact manifests, both rounds, p50/p95/p99, continuity, quality intervals, and profile analysis](https://github.com/EddieTzy/sglang-omni/blob/docs/contribution-log/records/drama3-final-20261118/README.md). Earlier screening and invalid no-op selection measurements are retained as history, excluded from this result. No combined-optimization gain is claimed.

## Checklist

- [x] Format changed files with pre-commit.
- [x] Record targeted numerical/correctness evidence.
- [x] Complete two-pass serving and EN/ZH automatic quality scoring.
- [x] Document opt-in conditions, limitations, and profiling evidence.
- [ ] Maintainer review and broader portability/quality acceptance.
