import argparse
import gzip
import json
import statistics
from pathlib import Path

import numpy as np


def load(path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    assert array.size and np.isfinite(array).all()
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    index = load(options.input / "INDEX.json")
    assert len(index["arms"]) == 8
    arms = {(arm["variant"], arm["round"]): arm for arm in index["arms"]}
    report = {
        "matrix": index["matrix"]["run"],
        "heads": index["heads"],
        "units": {
            "cost_s": "client send to full stream completion, seconds",
            "rtf": "per-request elapsed / actual generated audio duration",
            "ttfa_s": "client send to first nonempty PCM HTTP audio payload, seconds",
            "similarity": "WavLM cosine x100",
            "error_rate": "raw corpus errors/reference units, all samples retained",
        },
        "performance": {},
        "quality": {},
        "quality_comparisons": [],
    }
    total = failed = capped = unknown = 0
    for arm in index["arms"]:
        summaries = [point["summary"] for point in arm["performance"].values()] + [
            quality["generation_summary"] for quality in arm["quality"].values()
        ]
        for summary in summaries:
            total += summary["total_requests"]
            failed += summary["failed_requests"]
            capped += summary.get("max_token_hits", 0)
            unknown += summary["total_requests"] - summary.get(
                "finish_reason_observed", 0
            )
    report["generation_counts"] = {
        "formal_requests": total,
        "failed": failed,
        "budget_hits": capped,
        "unknown_terminal_state": unknown,
    }
    assert total == 38176 and failed == capped == unknown == 0
    for variant in ("baseline", "quantizer", "sm100", "moe"):
        report["performance"][variant] = {}
        for concurrency in (1, 16, 64):
            point_name = "speed-c" + str(concurrency)
            rounds = [
                arms[variant, number]["performance"][point_name] for number in (1, 2)
            ]
            requests = [
                row
                for point in rounds
                for row in load(options.input / point["file"])["per_request"]
            ]
            assert all(
                row["is_success"] and row["finish_reason"] == "stop" for row in requests
            )
            collected = {
                "round_summaries": [point["summary"] for point in rounds],
                "audio_throughput_mean": statistics.mean(
                    point["summary"]["audio_throughput_s_per_s"] for point in rounds
                ),
                "request_throughput_mean": statistics.mean(
                    point["summary"]["throughput_qps"] for point in rounds
                ),
                "cost_s": distribution([row["latency_s"] for row in requests]),
                "rtf": distribution([row["rtf"] for row in requests]),
                "ttfa_s": distribution([row["audio_ttfp_s"] for row in requests]),
                "audio_duration_s": distribution(
                    [row["audio_duration_s"] for row in requests]
                ),
                "gpu_memory_peak_mib": max(
                    point["summary"]["gpu_memory_peak_mib"] for point in rounds
                ),
            }
            continuity = [
                row["max_playback_underrun_s"]
                for row in requests
                if (row.get("audio_chunk_count") or 0) >= 2
                and row.get("max_playback_underrun_s") is not None
            ]
            collected["playback_max_underrun_s"] = (
                distribution(continuity) if continuity else None
            )
            collected["playback_continuity_scored"] = len(continuity)
            collected["playback_continuity_na"] = len(requests) - len(continuity)
            report["performance"][variant][str(concurrency)] = collected
    for variant in ("quantizer", "sm100", "moe"):
        for concurrency in ("1", "16", "64"):
            base = report["performance"]["baseline"][concurrency]
            candidate = report["performance"][variant][concurrency]
            candidate["audio_throughput_change_percent"] = (
                candidate["audio_throughput_mean"] / base["audio_throughput_mean"] - 1
            ) * 100
            candidate["cost_mean_change_percent"] = (
                candidate["cost_s"]["mean"] / base["cost_s"]["mean"] - 1
            ) * 100
    all_scores = all(
        all(kind in quality for kind in ("asr", "similarity"))
        for arm in index["arms"]
        for quality in arm["quality"].values()
    )
    report["quality_status"] = "complete" if all_scores else "pending"
    if all_scores:
        grouped = {}
        for variant in ("baseline", "quantizer", "sm100", "moe"):
            report["quality"][variant] = {}
            for language, count in (("en", 1088), ("zh", 2020)):
                rounds = []
                grouped[variant, language] = []
                for number in (1, 2):
                    quality = arms[variant, number]["quality"][language]
                    asr = load(options.input / quality["asr"]["file"])
                    similarity = load(options.input / quality["similarity"]["file"])
                    generation = load(options.input / quality["generation_manifest"])[
                        "per_sample"
                    ]
                    reference = {row["id"]: row for row in asr["per_sample"]}
                    voices = {row["id"]: row for row in similarity["per_sample"]}
                    assert len(generation) == len(reference) == len(voices) == count
                    assert (
                        {row["id"] for row in generation}
                        == set(reference)
                        == set(voices)
                    )
                    assert all(
                        row["is_success"] and row["is_transcribed"] and not row["error"]
                        for row in reference.values()
                    )
                    assert all(
                        row["is_success"]
                        and row["speaker_similarity"] is not None
                        and not row["error"]
                        for row in voices.values()
                    )
                    ids = sorted(reference)
                    errors = np.array(
                        [
                            reference[identity]["substitutions"]
                            + reference[identity]["deletions"]
                            + reference[identity]["insertions"]
                            for identity in ids
                        ],
                        dtype=np.float64,
                    )
                    units = np.array(
                        [
                            reference[identity]["substitutions"]
                            + reference[identity]["deletions"]
                            + reference[identity]["hits"]
                            for identity in ids
                        ],
                        dtype=np.float64,
                    )
                    scores = np.array(
                        [voices[identity]["speaker_similarity"] for identity in ids],
                        dtype=np.float64,
                    )
                    assert np.isfinite(scores).all() and units.sum() > 0
                    rate = float(errors.sum() / units.sum())
                    assert abs(rate - asr["summary"]["wer_corpus"]) < 1e-10
                    rounds.append(
                        {
                            "round": number,
                            "samples": count,
                            "raw_error_rate_percent": rate * 100,
                            "similarity_mean": float(scores.mean()),
                            "errors": int(errors.sum()),
                            "reference_units": int(units.sum()),
                            "asr_skipped": asr["summary"]["skipped"],
                            "above_50_percent_error_samples": sum(
                                (row["wer"] or 0) > 0.5 for row in reference.values()
                            ),
                        }
                    )
                    grouped[variant, language].append((ids, errors, units, scores))
                assert (
                    grouped[variant, language][0][0] == grouped[variant, language][1][0]
                )
                report["quality"][variant][language] = {
                    "rounds": rounds,
                    "error_rate_mean_percent": statistics.mean(
                        row["raw_error_rate_percent"] for row in rounds
                    ),
                    "similarity_mean": statistics.mean(
                        row["similarity_mean"] for row in rounds
                    ),
                }
        for language, count in (("en", 1088), ("zh", 2020)):
            rng = np.random.default_rng(20261118 + (language == "zh"))
            draws = rng.integers(0, count, size=(2000, count), dtype=np.int32)
            base = grouped["baseline", language]
            base_errors = base[0][1] + base[1][1]
            base_units = base[0][2] + base[1][2]
            base_scores = (base[0][3] + base[1][3]) / 2
            denominators = base_units[draws].sum(axis=1)
            assert (denominators > 0).all()
            for variant in ("quantizer", "sm100", "moe"):
                candidate = grouped[variant, language]
                assert candidate[0][0] == base[0][0]
                assert np.array_equal(candidate[0][2] + candidate[1][2], base_units)
                error_difference = candidate[0][1] + candidate[1][1] - base_errors
                voice_difference = (candidate[0][3] + candidate[1][3]) / 2 - base_scores
                error_bootstrap = (
                    error_difference[draws].sum(axis=1) / denominators * 100
                )
                voice_bootstrap = voice_difference[draws].mean(axis=1)
                report["quality_comparisons"].append(
                    {
                        "variant": variant,
                        "language": language,
                        "paired_utterances": count,
                        "passes_retained_per_utterance": 2,
                        "bootstrap_resamples": 2000,
                        "error_delta_percentage_points": float(
                            error_difference.sum() / base_units.sum() * 100
                        ),
                        "error_delta_95_percent_interval_pp": np.quantile(
                            error_bootstrap, [0.025, 0.975]
                        ).tolist(),
                        "similarity_delta": float(voice_difference.mean()),
                        "similarity_delta_95_percent_interval": np.quantile(
                            voice_bootstrap, [0.025, 0.975]
                        ).tolist(),
                        "interpretation": "Descriptive paired-utterance intervals conditional on two generation passes; not a predefined-margin noninferiority test",
                    }
                )
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "quality_status": report["quality_status"],
                "generation_counts": report["generation_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
