# Drama 3 arrival spacing and prefill batching

Single RTX PRO 6000, deployed streaming Drama 3 service, four fixed English prompts with the same reference voice. Four requests per wave, offsets 0/10/30 ms between neighboring requests. Four waves per condition, rotating prompt order with reversed spacing order on alternating repetitions. After a warmup, 48 timed requests ran without profiling and 48 separate requests with scheduler markers. All completed. No server-side change was applied. Throughput includes the arrival offsets and wave tails. RTF is the mean per-request wall time divided by generated audio duration. The underrun estimate uses cumulative PCM bytes received since the first packet.

| Spacing (ms) | First PCM P50/P95 (ms) | Audio s/s | Mean RTF | Underrun P95 (ms) | Dispatch to build P50 (ms) | Queue to prefill P50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 215.2 / 229.9 | 13.55 | 0.2715 | 0.0 | 37.3 | 41.0 |
| 10 | 167.4 / 180.0 | 13.63 | 0.2647 | 0.0 | 18.4 | 40.6 |
| 30 | 146.2 / 190.9 | 13.11 | 0.2683 | 25.0 | 9.4 | 30.0 |

At simultaneous arrival each wave split into prefill batches of one and three. The remaining three requests reached engine dispatch within 10.9 ms of the first dispatch, started building around 43–46 ms and entered prefill around 85–88 ms. The first prefill finished at 42–45 ms. At 10 ms spacing all waves still split one plus three. Spacing changes the offered load and starts the individual latency clock later. These differences characterize the workload and do not establish a serving speedup.

Run analyze.py to regenerate the request table and Perfetto-compatible scheduler timeline. This timeline contains request markers; it measures host-observed intervals. There are 16 timed requests per condition, so tail estimates are exploratory.
