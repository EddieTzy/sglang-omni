# Drama 3 early request construction in a four-request burst

The deployed service on an RTX PRO 6000 Blackwell Server Edition typically began its first prefill on one request and admitted the remaining three later. This experiment changes when the Drama 3 scheduler can construct incoming requests. The candidate starts builds on arrival using the existing scheduler pool with two workers and four pending builds. The workload, prompts and C4 engine limit are fixed. The process layout, model snapshot, SGLang installation and graph configuration match the existing FP8 service. The candidate source change applies to the deployment checkout `f1c94f0d8772127aca6a93992c3240ded10a74af`. [Candidate settings](candidate.diff)

Each group starts with a four-request warmup. Twelve timed bursts contain the same four reference-conditioned English prompts rotated through arrival slots. Requests start simultaneously. Four subsequent diagnostic bursts have stage markers enabled. Timed and diagnostic samples are disjoint. Duration and playback shortfall use streamed PCM byte counts and the response's sample rate. Audio throughput includes the time from the earliest send to completion of the final request in each burst. The source uses normal sampling.

The formal comparison uses C1/C2 for the two-worker candidate and A4/A5 for the restored original service. Each group name denotes one measurement run; all four groups use concurrency four. Clients in both cases connect over loopback. Two sets of measurements share one server startup per variant, with the original measured after the candidate. These conditions support a targeted test of the scheduling hypothesis. Other arrival patterns, audio quality and independently restarted replications were outside this test.

A1/A2 are additional baseline controls. B1 enabled early builds with a single worker, which leaves the executor disabled. B2 was discarded after the profiler cleanup used a mismatched run ID. A3 also overlapped an earlier marker session for nine timed waves and is excluded from the primary result. The collector included here closes each session with its actual run ID. Clean request IDs in the four formal groups are checked against recorded scheduler events.

The timing and stage logs accompany this report. `python analyze.py` recreates `summary.json` and the Perfetto-compatible scheduler event timelines. Each diagnostic event is an instant marker. In particular, the return of the background request builder and admission into the runnable queue are different events.

## Results

The table averages the two run-level measurements for each variant. Each variant contributed 96 unprofiled requests and 32 different requests with stage markers.

| Metric | Original | Early build |
| --- | ---: | ---: |
| First PCM P50 (ms) | 208.31 | 211.80 |
| First PCM P95 (ms) | 225.53 | 229.99 |
| Audio seconds per wall second | 13.654 | 13.464 |
| Mean request RTF | 0.2682 | 0.2688 |
| Mean request latency (s) | 1.6304 | 1.6364 |
| Playback underrun P95 (ms) | 0 | 0 |

For the last three requests by engine dispatch order in each burst, median dispatch-to-build fell from 39.73 to 1.08 ms. Median build-end-to-queue rose from 0.024 to 38.21 ms. From first dispatch in a burst to the later requests' prefill starts, the median moved from 85.85 to 90.28 ms. Eight of eight original diagnostic bursts used a 1+3 prefill split. Seven of eight candidate bursts used 1+3; one used 2+2. Earlier host-side construction left completed requests waiting for the scheduler's admission pass. This candidate did not meet the 5% serving target.

[Original timeline, run 1](A4/scheduler-trace.json.gz) · [Original, run 2](A5/scheduler-trace.json.gz) · [Early build, run 1](C1/scheduler-trace.json.gz) · [Early build, run 2](C2/scheduler-trace.json.gz) · [Per-run summaries](summary.json)
