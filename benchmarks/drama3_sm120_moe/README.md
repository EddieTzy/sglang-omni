# Drama 3 small-batch FP8 experts on RTX PRO 6000

The live C4 service spent a median 162.2 ms between preprocessing completion and handing off the first window of codes. The handoff to the vocoder took 2.3 ms and computing that window took 3.3 ms (8 matched requests). In a separate unprofiled sample of 16 requests, median and P95 time to first audio were 195.0 and 200.4 ms.

## GPU profile

The TTS engine recorded 2,053 ms summed GPU kernel time during the live C4 capture. Kernels on different streams can overlap. The times below measure the relative frequency and cost of work during that capture.

| Kernel in the engine | Calls | Summed ms | Share |
| --- | ---: | ---: | ---: |
| CUTLASS BF16 16×16 matrix multiplication | 25,802 | 595.7 | 29.0% |
| Triton routed expert multiplication | 11,600 | 438.7 | 21.4% |
| Fused activation and multiplication | 13,511 | 267.8 | 13.0% |
| CUTLASS BF16 32×32 matrix multiplication (64×1) | 14,548 | 166.1 | 8.1% |

In 4,108 adjacent pairs of Gated DeltaNet input projections (grids 8×96×1 and 4×1×1), the projections ran on different CUDA streams. Median start separation was 0.064 µs. The median duration of each was about 40 µs. Count this pair once when estimating potential critical-path savings. The checkpoint uses linear attention in 30 of its 40 backbone layers.

## Experiment

The candidate installs one optional SGLang Triton config for this GPU and expert shape. Small decode batches dispatch few tokens to each of 256 experts. Both projections use a 16×64×128 tile for 1–16 token rows. At 17 and above the selection returns to the baseline 64×128×128 tile. This bounds the change to the small batches tested.

On real layer-0 weights with six synthetic activation/routing draws per size, median CUDA Graph time for the complete routed expert calculation fell from 29.1 to 19.8 µs (1 row), 77.1 to 69.0 µs (4), 143.3 to 128.9 µs (8) and 245.5 to 231.7 µs (16). Max row-relative difference was zero in these comparisons. [Measurement script](measure_experts.py) · [All configurations and timings](kernel-measurements.json)

One RTX PRO 6000 Blackwell Server Edition, 96 GB, TP1, FP8 Drama 3, SGLang 0.5.23, Triton 3.8.0. Serving checkout `f1c94f0d8772127aca6a93992c3240ded10a74af`, checkpoint revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`. The original deployed service allowed four running requests. Both A and B used a cap of sixteen, separate vocoder process, overlap enabled, and the same model and image. Baseline A used SGLang's default tile. B set `SGLANG_MOE_CONFIG_DIR` and logged that it loaded the candidate. Start a fresh server per arm in A1/B1/B2/A2 order. Use the first 48 (C4) or 128 (C16) English SeedTTS requests after a warmup of 4 or 16, respectively. Benchmark in deployment checkout:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream --model fishaudio/drama-3 \
  --meta "$SEEDTTS_EN_META" --lang en --host 127.0.0.1 --port "$PORT" \
  --concurrency "$C" --warmup "$C" --max-samples "$N" --max-new-tokens 1000 \
  --output-dir "$OUT"
```

| Concurrency & launch | Audio s/s | Mean RTF | Mean latency (s) | First audio P50 (ms) | First audio P95 (ms) | Playback underrun P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 · A1 | 13.027 | 0.2992 | 1.269 | 134.2 | 187.4 | 7.8 |
| 4 · B1 | 13.459 | 0.2893 | 1.217 | 123.8 | 200.2 | 0.0 |
| 4 · B2 | 13.166 | 0.2916 | 1.245 | 123.3 | 169.7 | 0.0 |
| 4 · A2 | 13.081 | 0.2967 | 1.266 | 124.3 | 182.1 | 0.0 |
| 16 · A1 | 27.414 | 0.5655 | 2.257 | 164.9 | 483.4 | 222.2 |
| 16 · B1 | 27.532 | 0.5634 | 2.234 | 161.5 | 301.3 | 213.2 |
| 16 · B2 | 27.900 | 0.5540 | 2.208 | 161.5 | 323.1 | 193.5 |
| 16 · A2 | 27.312 | 0.5646 | 2.233 | 164.6 | 384.5 | 219.1 |

All 704 timed requests completed with zero truncations. Variant means give +1.98%/ +1.29% audio throughput at C4/C16. First-audio P95 varied between launches. At C4 its two-launch mean stayed around 185 ms; at C16 it moved from 434 to 312 ms. These tail values average two run-level percentiles. Normal sampling generated different waveform lengths. In a separate C1 check with temperature zero and top-k one, one of eight PCM hashes matched, and three lengths matched. This check alone cannot establish quality parity or the cause of the differences. [Per-launch metrics](serving.json)

## Perfetto files

Download and open the compressed Chrome traces in Perfetto. The original service and the first launch of each A/B variant have separate captures for the engine, preprocessing and vocoder. The C16 captures are short diagnostic windows. Each stage has its own capture interval. No profiling took place during the timed runs.

| Capture | TTS engine | Preprocessing | Vocoder | Manifest |
| --- | --- | --- | --- | --- |
| Live C4 | [trace](live-tts_engine.json.gz) | [trace](live-preprocessing.json.gz) | [trace](live-vocoder.json.gz) | [events and checksums](live-profile.json) |
| A1 C16 | [trace](A1-tts_engine.json.gz) | [trace](A1-preprocessing.json.gz) | [trace](A1-vocoder.json.gz) | [checksums](A1-profile.json) |
| B1 C16 | [trace](B1-tts_engine.json.gz) | [trace](B1-preprocessing.json.gz) | [trace](B1-vocoder.json.gz) | [checksums](B1-profile.json) |

## SeedTTS English quality

All 704 timed waveforms were scored using Qwen3-ASR and the SeedTTS WavLM speaker similarity scorer. Each variant used the same prompts twice. English word error rate is computed from total substitutions, insertions and deletions divided by reference words. Similarity uses the benchmark's native scale.

| Concurrency | Audio files per variant | A WER | B WER | A speaker similarity | B speaker similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 96 | 6/1,084 (0.55%) | 4/1,084 (0.37%) | 61.12 | 61.38 |
| 16 | 256 | 12/2,732 (0.44%) | 12/2,732 (0.44%) | 60.70 | 60.14 |

C16 speaker similarity decreased by 0.56 points. Natural sampling, the small English prompt set and two launches per variant limit generalization. [Individual edit counts and similarity scores](quality-scores.json)
