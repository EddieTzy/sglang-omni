# Drama 3 on RTX PRO 6000 Blackwell: C16 time to first audio

[Metrics, repeats and slow-request breakdown](RESULTS.md) · [raw traces and checksums](traces/manifest.json)

## Result

All six measured launches completed all 256 requests with zero generation-length-cap hits. Matching the vocoder process layout, stateful averaged 27.92 audio s/s and overlap 28.22 audio s/s. P95 first audio was 330.1 ms and 293.8 ms respectively. Across the two default-layout overlap runs, P99 was 494.2 and 522.5 ms. This workload did not reproduce the seconds-long P99 reported on the 5090.

Holding the overlap codec fixed, running the vocoder in a dedicated process changed mean throughput from 26.50 to 28.22 audio s/s (+6.48%), median first audio from 191.6 to 155.6 ms, and P99 first audio from 508.4 to 458.1 ms. Both dedicated-process repeats exceeded both default-layout repeats on throughput. This is evidence for an optional deployment configuration on this GPU and workload. Two starts per layout cannot establish generality or partition the effects of IPC, host scheduling and GPU overlap.

Both diagnostics paired the entire first-window path for 128/128 requests. Among the 13 slowest first-audio requests per arm, roughly 305–307 ms had already elapsed when the engine sent the last codes needed for the first window. Window execution averaged 13.2 ms with overlap and 4.5 ms with stateful. The send-to-start interval averaged 2.4 and 18.2 ms, respectively. In subsequent windows, real trace metadata showed re-decoding of previous frames in 505/505 overlap windows and 0/507 stateful windows. [Grouped timings](RESULTS.md)

## Serving configuration

One RTX PRO 6000 Blackwell Server Edition (96 GB), SGLang-Omni revision `f1c94f0d8772127aca6a93992c3240ded10a74af`, locally cached Fish Audio Drama 3 checkpoint revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216` and its bundled codec. Runtime detected an FP8 checkpoint. SGLang 0.5.23, PyTorch 2.14.1, NVIDIA driver 580.173.02. The source tree was mounted over the existing SGLang environment. The upstream comparison used int4 weights and a 32 GB RTX 5090. [Prior RTX 5090 results](https://github.com/sgl-project/sglang-omni/pull/4150)

Each fresh server used one GPU, C16, 16 engine slots, 1,000 max new tokens, engine seed 42, engine/vocoder memory shares 0.70/0.18, 3 GiB preprocessing reserve, and streaming window graphs. Non-streaming CUDA graph maximum frames was zero. A1 and default2 bracket the matched experiment and record the shipped overlap process layout with the engine and vocoder together. All four matched runs explicitly placed the vocoder in a dedicated process, as required for stateful decoding. The server logs in each arm's `activation.txt` identify the selected path and process layout. The four matched launches alternated stateful, overlap, stateful, overlap.

## Workload and timing

Each timed launch used the upstream SeedTTS benchmark with its first 256 English voice-clone targets and per-sample reference clip/transcript, with 16 warmups, closed-loop concurrency 16, natural generation, and the same sample sequence. Individual dispatch times depend on completion of earlier requests. The client sends no per-request seed. Audio duration and E2E can therefore vary across launches. No profiler was active during timed calls. Metrics in RESULTS.md are arithmetic means of two process-level results per mode; P99 is calculated per process and depends on only a few requests. TTFA ends at the client's first audio payload. RTF is the mean of individual request latency divided by generated audio duration. No separate listening or ASR evaluation was performed.

## Diagnostics

A separate round after timed measurement in each selected arm issued 128 requests from a C16 client to the warmed server. The client's HTTP request ID joins its monotonic timestamps to native stage events on the same machine using an epoch anchor per request. For the first vocoder window, the diagnostic matches its input chunk ID to the engine's send event. That boundary includes accumulation of all codes required by the window. The next span includes transfer, queueing and dispatch. The execution span covers the vocoder's host-side window call. The final span covers everything until first PCM at the client. These are request-wall-time measurements. They cannot partition GPU-exclusive work or establish an optimization's prospective speedup. All samples, stage counts and clock landmarks are included in the request timelines and phase JSONs. The timed client and diagnostic client are different, so compare stage attribution within the diagnostic runs.

Open `a2/request-timeline.json.gz` and `b1/request-timeline.json.gz` in Perfetto for request-level landmarks. Raw per-process request events and their storage-completeness files are under `traces/`. Separate brief torch-profiler captures cover a C16 wave of 16 requests. They are designed for inspection of CPU/CUDA work; the profiler changes service timing.

## Reproduction

The host-side scripts in `reproduction/` document the original mounts and launch command. Place the pinned source at `source/` under the experiment directory, mount the existing runtime and model, and supply the SeedTTS test set at the location in the per-arm benchmark commands. Each arm writes to a fresh directory. Use `trial.py a1 overlap` for the default-process control, then `trial.py b1 stateful`, `trial.py b2 stateful` (inserts `fixed-a1 overlap` first), and `trial.py a2 overlap`. Dedicated request and GPU diagnostics follow the timed calls. The original stateful torch capture used a separately restarted service warmed with 16 requests; `gpu-only.py` records that procedure. The published procedure can also capture in the same already-warmed server after the b2 measurement. Repeat the default-process control with `trial.py default2 overlap`. Regenerate the result tables with `write-report.py` followed by `placement.py`, passing the published experiment directory to each. Audio, model files and reference transcripts are excluded from the archive.
