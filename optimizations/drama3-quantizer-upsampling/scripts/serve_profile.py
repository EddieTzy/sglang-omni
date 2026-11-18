import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["baseline", "quantizer", "sm100", "moe"])
    parser.add_argument("label")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    options = parser.parse_args()
    plan_path = Path("/validation/results/performance-plan.json")
    if options.performance_only and plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if options.label.startswith(plan["run"] + "-arm"):
            arm_index = int(options.label.split("-arm", 1)[1].split("-", 1)[0])
            if arm_index > plan["max_arms"]:
                cancelled = Path("/validation/results") / (
                    options.label + "-cancelled.json"
                )
                cancelled.write_text(
                    json.dumps(
                        {
                            "state": "cancelled-before-server-launch",
                            "reason": "User reduced the experiment to one baseline/candidate pair",
                        },
                        indent=2,
                    )
                )
                raise SystemExit(125)
    source = Path("/validation") / ("profile-" + options.variant)
    output = Path("/validation/results/serving") / options.label
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        SGLANG_TORCH_PROFILER_RECORD_SHAPES="0",
    )
    if options.variant == "moe":
        environment["SGLANG_MOE_CONFIG_DIR"] = "/validation/moe-configs"
    command = [
        "/opt/sglang/bin/sgl-omni",
        "serve",
        "--model-path",
        "/models/drama3",
        "--host",
        "127.0.0.1",
        "--port",
        "18100",
        "--allowed-local-media-path",
        "/validation/datasets",
        "--vocoder.factory.codec_path",
        "/models/drama3",
        "--vocoder.factory.stream_decode",
        "overlap",
        "--tts_engine.factory.max_running_requests",
        "64",
        "--tts_engine.factory.enable_async_decode",
        "true",
        "--tts_engine.engine.random_seed",
        "42",
        "--preprocessing.factory.reference_encode_graph_max_seconds",
        "0",
    ]
    manifest = {
        "variant": options.variant,
        "command": command,
        "base_commit": "f17a438cd753248d70916700aa4dccd9e7feb098",
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source / "sglang_omni/models/fishaudio_drama3").glob("*.py")
        },
        "state": "starting",
        "started_at": time.time(),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18100/health", timeout=3
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server readiness exceeded 900 seconds")
                time.sleep(2)
            with urllib.request.urlopen("http://127.0.0.1:18100/v1/models") as response:
                model = json.load(response)["data"][0]["id"]
            manifest.update(state="ready", ready_at=time.time(), model=model)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            cases = [
                ("warmup-c1", "en", 16, 1, 0),
                ("profile-c1", "en", 8, 1, 0),
                ("control-c1", "en", 64, 1, 16),
                ("warmup-c64", "en", 64, 64, 0),
                ("profile-c64", "en", 64, 64, 0),
                ("control-c64", "en", 128, 64, 64),
            ]
            for name, language, count, concurrency, warmup in cases:
                profile_run = options.label + "-" + name
                if name.startswith("profile-"):
                    profile_output = output / name / "profile"
                    payload = {
                        "run_id": profile_run,
                        "trace_path_template": str(profile_output / "{stage}"),
                        "event_dir": str(profile_output / "events"),
                        "event_buffer_capacity": 1000000,
                    }
                    request = urllib.request.Request(
                        "http://127.0.0.1:18100/start_profile",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        started = json.load(response)
                    (output / (name + "-profile-start.json")).write_text(
                        json.dumps(started, indent=2)
                    )
                    time.sleep(2)
                benchmark = [
                    sys.executable,
                    "-m",
                    "benchmarks.eval.benchmark_tts_seedtts",
                    "--generate-only",
                    "--use-existing-server",
                    "--stream",
                    "--meta",
                    f"/validation/datasets/seedtts/{language}/meta.lst",
                    "--lang",
                    language,
                    "--model",
                    model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18100",
                    "--concurrency",
                    str(concurrency),
                    "--warmup",
                    str(warmup),
                    "--max-samples",
                    str(count),
                    "--max-new-tokens",
                    "2048",
                    "--output-dir",
                    str(output / name),
                    "--disable-tqdm",
                    "--fingerprint",
                    "--measure-gpu-memory",
                ]
                with (output / f"{name}.log").open("w") as benchmark_log:
                    subprocess.run(
                        benchmark,
                        cwd=source,
                        env=environment,
                        stdout=benchmark_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if name.startswith("profile-"):
                    request = urllib.request.Request(
                        "http://127.0.0.1:18100/stop_profile",
                        data=json.dumps({"run_id": profile_run}).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        json.load(response)
                    export_deadline = time.monotonic() + 120
                    while True:
                        exported = list(profile_output.glob("*.trace.json.gz"))
                        if (
                            len(exported) == 2
                            and not list(profile_output.glob("*.tmp"))
                            and not list(profile_output.glob("*.trace.json"))
                        ):
                            break
                        if time.monotonic() > export_deadline:
                            raise TimeoutError(
                                "Not both process CUDA traces exported; engine and vocoder share a process"
                            )
                        time.sleep(1)
                    time.sleep(2)

            manifest.update(state="complete", finished_at=time.time())
        except Exception as error:
            manifest.update(
                state="failed",
                error=f"{type(error).__name__}: {error}",
                finished_at=time.time(),
            )
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
