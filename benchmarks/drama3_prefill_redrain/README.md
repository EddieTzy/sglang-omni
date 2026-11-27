# Nonblocking second inbox read before the first Drama 3 prefill

This experiment tests whether another request has already reached the engine inbox while the scheduler prepares to select the first prefill batch. For a single GPU, an empty running batch and fewer than four queued requests, variant B does one more regular nonblocking inbox drain and processes any returned requests before batch selection. Variant A retains the original scheduling path. Both run in the same temporary process using the same per-wave flag lookup. The patch and the markers are diagnostic scaffolding. [Experiment diff](experiment.diff)

Original RTX PRO 6000 Blackwell Server Edition deployment checkout f1c94f0d8772127aca6a93992c3240ded10a74af, TP1, FP8 Drama 3, separate vocoder, overlap, four-request engine limit. One warmup burst, twelve measured waves per condition and six additional marker-enabled waves per condition. The four prompt texts rotate. Adjacent A and B waves use identical prompts and the same client, reversing order on alternate repetitions. C1 has twelve timed samples per variant and C4 has 48. Generation uses normal sampling. The experiment compares client time to first PCM, audio seconds per wall second, per-request latency and per-request RTF. The response rate and PCM bytes determine generated duration. Percentiles interpolate sorted request measurements. No quality study was run.

| Load and variant | First audio P50/P95 (ms) | Audio s/s | Mean RTF | Mean latency (s) | Underrun P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 original | 76.87 / 90.25 | 5.497 | 0.182 | 1.138 | 0 |
| C1 second read | 78.96 / 85.14 | 5.565 | 0.180 | 1.091 | 0 |
| C4 original | 180.11 / 196.08 | 14.141 | 0.252 | 1.579 | 0 |
| C4 second read | 184.50 / 220.67 | 14.564 | 0.252 | 1.534 | 0 |

In six original C4 diagnostic bursts all prefill splits were 1+3. With the second read three bursts split 2+2, two split 1+3, and one split 3+1. The earliest client response in each timed C4 burst had a median of 90.94 ms on A and 103.48 ms on B. C1 diagnostic second reads returned zero additional requests in all six bursts; its P95 improvement on twelve timed samples needs independent replication. At C4 the 2.99% throughput gain came with a 12.54% increase in first-audio P95. This result did not justify adoption.

Each Perfetto timeline contains scheduler markers on one track per request. Markers cover dispatch, the scheduler's inbox read, the B-side second read, batch selection and scheduler-reported prefill start/end. These are host timestamps. [C4 original](A-c4.json.gz) · [C4 second read](B-c4.json.gz) · [C1 original](A-c1.json.gz) · [C1 second read](B-c1.json.gz) · [Individual waves and summaries](summary.json)

An independent offline check of archived decode graph 3098 found 130 matching replays with thirteen post-expert rotary kernels (four shared layers and nine residual codebooks). The span from the first residual rotary kernel to the ninth was 1.918 ms median, within a median 9.478 ms full profiled graph span. Kernel timestamps include any intervening work. This interval excludes the outer parts of the first and last residual layers. [Calculation script](tower_span.py) · [Source trace](https://github.com/EddieTzy/sglang-omni/blob/f49fbbea8e2aa09c07f495c476b8d6455f12fcd6/benchmarks/drama3_sm120_moe/live-tts_engine.json.gz)
