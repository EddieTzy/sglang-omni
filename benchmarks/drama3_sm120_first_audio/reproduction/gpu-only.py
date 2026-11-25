"""Collect a short raw torch trace on a warmed stateful service."""

import json, subprocess as sp, sys, time, urllib.request
from pathlib import Path

w = Path(__file__).resolve().parent
folder = w / "b2"
name = "drama3-c16-b2"
docker = ["sudo", "-n", "docker"]
if list((folder / "traces").rglob("*.gz")):
    sys.exit()
assert not sp.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
sp.run(docker + ["start", name], check=True)
try:
    for _ in range(400):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:18948/v1/models", timeout=2
            ) as r:
                model = json.load(r)["data"][0]["id"]
            break
        except OSError:
            if (
                sp.check_output(
                    docker + ["inspect", "-f", "{{.State.Running}}", name], text=True
                ).strip()
                != "true"
            ):
                raise RuntimeError("container exited")
            time.sleep(2)
    else:
        raise TimeoutError("service readiness")
    bench = json.loads((folder / "bench.json").read_text())
    for arg, value in [
        ("--warmup", "0"),
        ("--max-samples", "16"),
        ("--output-dir", f"/data/{w.name}/b2/diagnostic-warmup"),
    ]:
        bench[bench.index(arg) + 1] = value
    with (folder / "trace-warmup.log").open("w") as log:
        sp.run(
            docker + ["exec", name, *bench], stdout=log, stderr=sp.STDOUT, check=True
        )
    with (folder / "gpu-diag.log").open("w") as log:
        sp.run(
            docker
            + [
                "exec",
                name,
                "/data/venv/bin/python",
                f"/data/{w.name}/diag.py",
                "b2",
                model,
                "18948",
                "--gpu",
            ],
            stdout=log,
            stderr=sp.STDOUT,
            check=True,
        )
finally:
    with (folder / "profile-startup.log").open("w") as log:
        sp.run(docker + ["logs", "--tail", "550", name], stdout=log, stderr=sp.STDOUT)
    sp.run(docker + ["stop", "-t", "8", name])
