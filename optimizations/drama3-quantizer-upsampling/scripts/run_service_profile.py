import datetime
import json
import subprocess
from pathlib import Path

root = Path("/home/tzy/drama3-pr-validation")
run = "quantizer-service-profile-" + datetime.datetime.now(
    datetime.timezone.utc
).strftime("%Y%m%dT%H%M%SZ")
output = root / "results" / run
output.mkdir()
state = {
    "run": run,
    "state": "preparing",
    "arms": [],
    "original_service_restored": False,
}


def save():
    (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")


def docker(*arguments, **kwargs):
    return subprocess.run(["sudo", "-n", "docker", *arguments], check=True, **kwargs)


was_running = (
    subprocess.check_output(
        [
            "sudo",
            "-n",
            "docker",
            "inspect",
            "tts-drama3",
            "--format",
            "{{.State.Running}}",
        ],
        text=True,
    ).strip()
    == "true"
)
state["original_running"] = was_running
(root / "results" / "current-profile-run.txt").write_text(run + "\n")
save()
try:
    if was_running:
        docker("stop", "-t", "30", "tts-drama3")
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if applications:
        raise RuntimeError("Other GPU processes remain: " + applications)
    docker("start", "codex-drama3-pr-validation")
    for variant in ("baseline", "quantizer"):
        state["state"] = "profiling-" + variant
        arm = {"variant": variant, "label": run + "-" + variant, "state": "running"}
        state["arms"].append(arm)
        save()
        with (output / (variant + ".log")).open("w") as log:
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "docker",
                    "exec",
                    "codex-drama3-pr-validation",
                    "/opt/sglang/bin/python3",
                    "/validation/serve_profile.py",
                    variant,
                    arm["label"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        arm["exit_code"] = result.returncode
        arm["state"] = "complete" if result.returncode == 0 else "failed"
        save()
        if result.returncode:
            raise RuntimeError(variant + " profile failed")
    state["state"] = "complete"
except BaseException as error:
    state.update(state="failed", error=str(error))
    raise
finally:
    subprocess.run(
        ["sudo", "-n", "docker", "stop", "-t", "15", "codex-drama3-pr-validation"],
        check=False,
    )
    if was_running:
        docker("start", "tts-drama3")
        state["original_service_restored"] = True
    save()
