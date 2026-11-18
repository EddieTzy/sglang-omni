import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False)
    else:
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--post")
    options = parser.parse_args()
    root = Path("/home/tzy/drama3-pr-validation")
    matrix = json.loads((root / "results" / options.matrix / "status.json").read_text())
    output = root / "results" / options.matrix / "public-evidence"
    output.mkdir(exist_ok=True)
    package = {
        "matrix": matrix,
        "heads": {
            "baseline": "8848e5003ba79923af1e175364cef6a7cdb20109",
            "quantizer": "054d0b1c77f0e78d24d6aeb8667dbb031c31d009",
            "sm100": "ff3103fcddf9c9cbe8d7a674bff25a9622c589d6",
            "moe": "dc5e1dc490801dd4a9d1cc3c922067249e99f9a6",
        },
        "arms": [],
    }
    dataset = root / "datasets/seedtts-manifest.json"
    package["dataset"] = json.loads(dataset.read_text())
    package["runtime"] = {
        "torch": "2.14.1+cu130",
        "sglang": "0.5.23",
        "triton": "3.8.0",
        "flashinfer": "0.7.2",
        "driver": "580.173.02",
        "gpu": "NVIDIA B200",
        "flashinfer_cubin": "0.7.1 ignored due version mismatch",
        "flashinfer_jit_cache": "0.7.1+cu130 ignored due version mismatch",
    }
    for arm in matrix["arms"]:
        if arm["state"] != "complete":
            continue
        source = root / "results/serving" / arm["label"]
        metadata = {
            "variant": arm["variant"],
            "round": arm["round"],
            "label": arm["label"],
            "manifest": json.loads((source / "manifest.json").read_text()),
            "performance": {},
            "quality": {},
        }
        for point in source.glob("speed-*/speed_results.json"):
            results = json.loads(point.read_text())
            payload = {
                "summary": results["summary"],
                "per_request": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("text", "wav_path")
                    }
                    for row in results["per_request"]
                ],
            }
            relative = Path("performance") / (
                arm["variant"]
                + "-r"
                + str(arm["round"])
                + "-"
                + point.parent.name
                + ".json.gz"
            )
            write_json(output / relative, payload)
            metadata["performance"][point.parent.name] = {
                "summary": results["summary"],
                "file": str(relative),
            }
        for language in ("en", "zh"):
            folder = source / ("quality-" + language)
            speed = json.loads((folder / "speed_results.json").read_text())
            generated = json.loads((folder / "generated.json").read_text())
            quality = {
                "generation_summary": speed["summary"],
                "generated_count": len(generated),
                "generated_ids_sha256": hashlib.sha256(
                    json.dumps(sorted(row["sample_id"] for row in generated)).encode()
                ).hexdigest(),
                "target_text_multiset_sha256": hashlib.sha256(
                    json.dumps(
                        sorted(
                            (row["sample_id"], row["target_text"]) for row in generated
                        ),
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
            }
            generation_relative = Path("quality") / (
                arm["variant"]
                + "-r"
                + str(arm["round"])
                + "-"
                + language
                + "-generation.json.gz"
            )
            timing_by_id = {row["id"]: row for row in speed["per_request"]}
            generation_rows = [
                {
                    "id": row["sample_id"],
                    "target_text_sha256": hashlib.sha256(
                        row["target_text"].encode()
                    ).hexdigest(),
                    "audio_duration_s": row["audio_duration_s"],
                    "is_success": row["is_success"],
                    "finish_reason": timing_by_id[row["sample_id"]]["finish_reason"],
                    "completion_tokens": timing_by_id[row["sample_id"]][
                        "completion_tokens"
                    ],
                }
                for row in generated
            ]
            write_json(output / generation_relative, {"per_sample": generation_rows})
            quality["generation_manifest"] = str(generation_relative)
            for kind, name in (
                ("asr", "wer_results.json"),
                ("similarity", "similarity_results.json"),
            ):
                path = folder / name
                if path.exists():
                    results = json.loads(path.read_text())
                    selected = (
                        (
                            "id",
                            "wer",
                            "substitutions",
                            "deletions",
                            "insertions",
                            "hits",
                            "audio_duration_s",
                            "is_success",
                            "is_transcribed",
                            "error",
                        )
                        if kind == "asr"
                        else ("id", "speaker_similarity", "is_success", "error")
                    )
                    payload = {
                        "summary": results["summary"],
                        "per_sample": [
                            {key: row.get(key) for key in selected}
                            for row in results["per_sample"]
                        ],
                    }
                    relative = Path("quality") / (
                        arm["variant"]
                        + "-r"
                        + str(arm["round"])
                        + "-"
                        + language
                        + "-"
                        + kind
                        + ".json.gz"
                    )
                    write_json(output / relative, payload)
                    quality[kind] = {
                        "summary": results["summary"],
                        "file": str(relative),
                    }
            metadata["quality"][language] = quality
        package["arms"].append(metadata)
    model = root / "pr-baseline"
    original = root.parent / "tts-containers/models/drama3"
    package["checkpoint_files"] = {
        name: {
            "bytes": (original / name).stat().st_size,
            "sha256": hashlib.file_digest(
                (original / name).open("rb"), "sha256"
            ).hexdigest(),
        }
        for name in ("config.json", "model.safetensors.index.json")
        if (original / name).is_file()
    }
    for stem in ("config.json", "model.safetensors.index.json"):
        metadata = original / ".cache/huggingface/download" / (stem + ".metadata")
        if metadata.exists():
            package.setdefault("checkpoint_download_metadata", {})[stem] = (
                metadata.read_text().splitlines()[:2]
            )
    numeric = root / "results/moe-current-runtime-numerics.json"
    if numeric.exists():
        write_json(output / "moe-numerics.json", json.loads(numeric.read_text()))
    if options.post:
        post = root / "results" / options.post
        for name in ("status.json", "work-status.json"):
            if (post / name).exists():
                write_json(
                    output / ("post-" + name), json.loads((post / name).read_text())
                )
        overlay = root / "results/final-profile-overlays.json"
        if overlay.exists():
            write_json(
                output / "profile-overlays.json", json.loads(overlay.read_text())
            )
        package["profile_labels"] = [
            options.post + "-profile-" + variant
            for variant in ("baseline", "quantizer", "sm100", "moe")
        ]
    write_json(output / "INDEX.json", package)
    with tarfile.open(output.parent / "public-evidence.tar", "w") as archive:
        archive.add(output, arcname="final-pr-evidence")
    print(output.parent / "public-evidence.tar")


if __name__ == "__main__":
    main()
