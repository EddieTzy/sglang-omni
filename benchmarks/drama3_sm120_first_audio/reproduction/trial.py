import json, os, subprocess as sp, sys, time, urllib.request
from pathlib import Path

work = Path(__file__).resolve().parent
root = work.parent
port = 18948
label, mode = sys.argv[1:3]
if label == "b2" and not (work / "fixed-a1").exists():
    sp.run([sys.executable, str(Path(__file__)), "fixed-a1", "overlap"], check=True)
out = work / label
out.mkdir()
name = "drama3-c16-" + label
stateful = mode == "stateful"
assert mode in ["overlap", "stateful"]
assert not sp.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip(), "GPU busy"
docker = ["sudo", "-n", "docker"]
serve = [
    "serve",
    "--model-path",
    "/models/drama3",
    "--vocoder.factory.codec_path",
    "/models/drama3",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--allowed-local-media-path",
    "/validation/datasets",
    "--preprocessing.total_reserve_bytes",
    "3GiB",
    "--tts_engine.gpu_memory_fraction",
    "0.70",
    "--tts_engine.factory.max_running_requests",
    "16",
    "--tts_engine.factory.max_new_tokens",
    "1000",
    "--tts_engine.engine.random_seed",
    "42",
    "--vocoder.gpu_memory_fraction",
    "0.18",
    "--vocoder.factory.stream_decode",
    "overlap",
    "--vocoder.factory.stateful_decode",
    str(stateful).lower(),
    "--vocoder.factory.non_streaming_cuda_graph_max_frames",
    "0",
]
if label not in ["a1", "default2"]:
    serve += ["--vocoder.process", "vocoder"]
run = docker + [
    "run",
    "-d",
    "--name",
    name,
    "--gpus",
    "all",
    "--network",
    "host",
    "--ipc",
    "private",
    "--shm-size",
    "8g",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "-v",
    f"{root}:/data",
    "-v",
    f"{work}/source:/opt/sglang-omni:ro",
    "-v",
    "/home/tzy/tts-containers/models/drama3:/models/drama3:ro",
    "-v",
    "/home/tzy/drama3-pr-validation/datasets:/validation/datasets:ro",
    "-w",
    "/opt/sglang-omni",
]
for k, v in {
    "HOME": "/data",
    "HF_HOME": "/data/cache/huggingface",
    "HF_HUB_OFFLINE": "1",
    "PYTHONPATH": "/opt/sglang-omni",
    "OMP_NUM_THREADS": "4",
    "TRITON_CACHE_DIR": "/data/cache/triton",
    "FLASHINFER_WORKSPACE_BASE": "/data/cache/flashinfer",
    "SGLANG_TORCH_PROFILER_DIR": "/data/" + work.name + "/" + label + "/traces",
}.items():
    run += ["-e", f"{k}={v}"]
run += [
    "--entrypoint",
    "/data/venv/bin/sgl-omni",
    "tts-runtime:20261117-pinned",
    *serve,
]
(out / "launch.json").write_text(json.dumps(serve, indent=2))
sp.run(run, check=True)
try:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if (
            sp.check_output(
                docker + ["inspect", "-f", "{{.State.Running}}", name], text=True
            ).strip()
            != "true"
        ):
            raise RuntimeError("server exited")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as r:
                model = json.load(r)["data"][0]["id"]
            break
        except OSError:
            time.sleep(2)
    else:
        raise TimeoutError("startup")
    logs = sp.check_output(docker + ["logs", name], stderr=sp.STDOUT, text=True)
    assert f"codec selection: stateful={stateful}" in logs, (
        "codec selection not confirmed"
    )
    if stateful:
        assert "Drama 3 codec activation:" in logs, "activation not confirmed"
    bench = [
        "/data/venv/bin/python",
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--generate-only",
        "--use-existing-server",
        "--stream",
        "--meta",
        "/validation/datasets/seedtts/en/meta.lst",
        "--lang",
        "en",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--concurrency",
        "16",
        "--warmup",
        "16",
        "--max-samples",
        "256",
        "--max-new-tokens",
        "1000",
        "--output-dir",
        f"/data/{work.name}/{label}/timed",
    ]
    (out / "bench.json").write_text(json.dumps(bench, indent=2))
    with (out / "client.log").open("w") as f:
        sp.run(docker + ["exec", name] + bench, stdout=f, stderr=sp.STDOUT, check=True)
    stats = json.loads((out / "timed/speed_results.json").read_text())["summary"]
    print(json.dumps(stats), flush=True)
    if stats["completed_requests"] == 0:
        raise RuntimeError("no valid streaming outputs; see per-request errors")
    if label in ["a2", "b1", "fixed-a", "fixed-b"]:
        with (out / "diag.log").open("w") as log:
            sp.run(
                docker
                + [
                    "exec",
                    name,
                    "/data/venv/bin/python",
                    f"/data/{work.name}/diag.py",
                    label,
                    model,
                    str(port),
                ],
                stdout=log,
                stderr=sp.STDOUT,
                check=True,
            )
    if label in ["b2", "a2"]:
        with (out / "gpu-diag.log").open("w") as log:
            sp.run(
                docker
                + [
                    "exec",
                    name,
                    "/data/venv/bin/python",
                    f"/data/{work.name}/diag.py",
                    label,
                    model,
                    str(port),
                    "--gpu",
                ],
                stdout=log,
                stderr=sp.STDOUT,
                check=True,
            )
finally:
    with (out / "server.log").open("w") as f:
        sp.run(docker + ["logs", name], stdout=f, stderr=sp.STDOUT)
    sp.run(docker + ["stop", "-t", "8", name])
