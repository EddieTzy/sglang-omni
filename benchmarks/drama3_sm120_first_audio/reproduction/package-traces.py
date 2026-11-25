"""Keep original Chrome traces and request-event records from the measured arms."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path(__file__).resolve().parent
public = root / "public"
trace_dir = public / "traces"
trace_dir.mkdir(exist_ok=True)
manifest = []

for arm in ["b1", "a2"]:
    files = sorted((root / arm / "events").glob("*"))
    assert len([f for f in files if f.suffix == ".jsonl"]) >= 4
    for f in files:
        if f.name.startswith("run_status"):
            status = json.loads(f.read_text())
            assert (
                status["complete"]
                and status["dropped_events"] == 0
                and status["unwritten_events"] == 0
            )
    filename = arm + "-raw-request-events.tar.gz"
    with tarfile.open(trace_dir / filename, "w:gz") as targz:
        for f in files:
            targz.add(f, arcname=f.name)
    manifest.append(
        dict(arm=arm, kind="raw request events and recorder status", file=filename)
    )
for arm in ["a2", "b2"]:
    paths = sorted((root / arm / "traces").rglob("*.gz"))
    for path in paths:
        data = json.load(gzip.open(path))
        events = data.get("traceEvents", [])
        assert events
        gpu = sum(
            e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"] for e in events
        )
        dst = arm + "-" + path.name
        if (trace_dir / dst).exists() and (
            trace_dir / dst
        ).stat().st_size != path.stat().st_size:
            dst = arm + "-" + path.parent.name + "-" + path.name
        shutil.copyfile(path, trace_dir / dst)
        manifest.append(
            dict(
                arm=arm,
                kind="torch CPU/CUDA Chrome trace",
                file=dst,
                events=len(events),
                device_events=gpu,
            )
        )
for e in manifest:
    p = trace_dir / e["file"]
    e.update(
        bytes=p.stat().st_size,
        sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
    )
(trace_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print([(m["file"], m["bytes"], m.get("device_events")) for m in manifest])
