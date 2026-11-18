import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    changes = []
    for variant in ("baseline", "quantizer"):
        source = arguments.root / variant
        destination = arguments.root / ("profile-" + variant)
        if destination.exists():
            raise FileExistsError(f"Refuse to overwrite {destination}")
        shutil.copytree(
            source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__")
        )
        replacements = {
            "sglang_omni/profiler/torch_profiler.py": [
                (
                    "activities=profiler_activities(),",
                    "activities=[ProfilerActivity.CUDA],",
                ),
                ("profile_all_threads=not profile_memory", "profile_all_threads=False"),
            ],
            "benchmarks/metrics/performance.py": [
                (
                    '"id": output.request_id,',
                    '"id": output.request_id,\n        "speech_outcome_id": getattr(output, "speech_outcome_id", None),',
                ),
            ],
        }
        for relative, substitutions in replacements.items():
            path = destination / relative
            original = path.read_text()
            modified = original
            for before, after in substitutions:
                if before not in modified:
                    raise ValueError(
                        f"Diagnostic overlay does not match {path}: {before}"
                    )
                modified = modified.replace(before, after)
            path.write_text(modified)
            changes.append(
                {
                    "variant": variant,
                    "file": relative,
                    "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "diagnostic_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                }
            )
    (arguments.root / "results" / "diagnostic-source-overlay.json").write_text(
        json.dumps(changes, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
