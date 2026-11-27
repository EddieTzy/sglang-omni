# Dual-stream versus packed Drama 3 GDN input projections

The deployed RTX PRO 6000 Blackwell Server Edition FP8 service uses BF16 projections for the linear-attention layers. On the live C4 trace, the QKVZ and BA input projections appeared in 4,108 adjacent pairs with different CUDA streams. Their median start times were 0.064 µs apart. Median durations were 39.71 µs and 39.58 µs. Both consume the same input. One packed GEMM can compute both results, but the baseline already overlaps the two operations.

This isolated screen uses six checkpoint layers (0, 1, 2, 4, 5, 6), synthetic fixed activations, and one CUDA Graph per variant. The layer weights rotate to exceed L2 capacity. The baseline uses a second CUDA stream and waits for both results. The candidate concatenates the weights once and splits views of the output. Ten timing blocks alternate variant order; each block replays the six layers fifty times. Reported times include both projections. The standalone BF16 linear screen omits subsequent consumers and any materialization they may require.

| Rows | Dual stream (µs) | Packed (µs) | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 36.20 | 35.85 | 0.97% |
| 4 | 36.62 | 36.49 | 0.36% |
| 8 | 37.05 | 36.65 | 1.07% |
| 16 | 37.35 | 37.23 | 0.32% |

The largest measured difference was 0.40 µs per layer. The relative L2 error against the baseline was zero through eight rows and at most 0.000116 at sixteen rows. This screen did not justify changing the model or running a service A/B.

The separate diagnostic graphs each contain 12 replays at four rows. Download the compressed traces and open them in Perfetto. [Dual stream](dual.json.gz) · [Packed](packed.json.gz) · [All samples and numerical differences](projection-results.json) · [Original service capture](https://github.com/EddieTzy/sglang-omni/blob/f49fbbea8e2aa09c07f495c476b8d6455f12fcd6/benchmarks/drama3_sm120_moe/live-tts_engine.json.gz)

Run the screen and the diagnostic from the same serving environment:

```bash
python projections.py --model /models/drama3 --output /tmp/drama3-gdn
python projections.py --model /models/drama3 --output /tmp/drama3-gdn --trace-only
```
