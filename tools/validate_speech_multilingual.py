#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_multilingual_tool"
SUPPORTED_GENERATION_MODES = frozenset({"described", "clone"})


def _load_metrics():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    if PACKAGE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE_NAME,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE_NAME] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    module = __import__(f"{PACKAGE_NAME}.speech_verification", fromlist=["transcript_metrics"])
    return module.transcript_metrics


def _transcribe(model, audio_path: Path, language: str | None, beam_size: int) -> dict:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
    )
    segments = list(segments)
    return {
        "text": " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip(),
        "detected_language": str(info.language or ""),
        "language_probability": float(info.language_probability or 0.0),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(case: dict, key: str, index: int) -> str:
    value = str(case.get(key) or "").strip()
    if not value:
        raise ValueError(f"case {index} requires non-empty {key}")
    return value


def _parse_required_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(modes) - SUPPORTED_GENERATION_MODES)
    if unknown:
        raise ValueError(f"unsupported required generation modes: {', '.join(unknown)}")
    if not modes:
        raise ValueError("at least one required generation mode is required")
    return modes


def validate_manifest_design(
    payload,
    manifest_path: Path,
    metric_fn,
    *,
    strict_design: bool,
    minimum_samples_per_language: int,
    minimum_utterances_per_language: int,
    minimum_seeds_per_utterance_mode: int,
    required_generation_modes: tuple[str, ...],
) -> tuple[list[dict], dict]:
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain at least one case")

    normalized_cases = []
    case_ids: set[str] = set()
    audio_hashes: dict[str, str] = {}
    expected_by_utterance: dict[tuple[str, str], str] = {}
    experiment_cells: set[tuple[str, str, str, str, int]] = set()
    grouped: dict[str, list[dict]] = defaultdict(list)
    global_findings: list[str] = []

    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = _required_text(raw_case, "case_id", index) if strict_design else str(raw_case.get("case_id") or index)
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

        language_code = _required_text(raw_case, "language_code", index)
        if language_code.casefold() == "auto":
            raise ValueError(f"case {case_id} must use an explicit language_code, not auto")
        expected_text = _required_text(raw_case, "expected_text", index)
        # Fail closed before ASR loading when the expected transcript contains only punctuation.
        metric_fn(expected_text, expected_text)

        raw_audio_path = Path(_required_text(raw_case, "audio_path", index))
        audio_path = raw_audio_path if raw_audio_path.is_absolute() else manifest_path.parent / raw_audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        audio_sha256 = _sha256_file(audio_path)

        if strict_design:
            generation_mode = _required_text(raw_case, "generation_mode", index)
            utterance_id = _required_text(raw_case, "utterance_id", index)
            if generation_mode not in SUPPORTED_GENERATION_MODES:
                raise ValueError(
                    f"case {case_id} generation_mode must be one of "
                    f"{sorted(SUPPORTED_GENERATION_MODES)}"
                )
            seed = raw_case.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"case {case_id} requires an integer seed")
            speaker_id = str(raw_case.get("speaker_id") or "").strip()
            if generation_mode == "clone" and not speaker_id:
                raise ValueError(f"clone case {case_id} requires speaker_id")
            voice_profile_id = str(raw_case.get("voice_profile_id") or "").strip()
            if generation_mode == "described" and not voice_profile_id:
                raise ValueError(f"described case {case_id} requires voice_profile_id")
        else:
            generation_mode = str(raw_case.get("generation_mode") or "unspecified").strip()
            utterance_id = str(raw_case.get("utterance_id") or case_id).strip()
            seed = raw_case.get("seed", index)
            speaker_id = str(raw_case.get("speaker_id") or "").strip()
            voice_profile_id = str(raw_case.get("voice_profile_id") or "").strip()

        condition_id = speaker_id if generation_mode == "clone" else voice_profile_id
        utterance_key = (language_code, utterance_id)
        previous_expected = expected_by_utterance.setdefault(utterance_key, expected_text)
        if previous_expected != expected_text:
            raise ValueError(
                f"utterance {language_code}/{utterance_id} has inconsistent expected_text"
            )
        if strict_design:
            experiment_cell = (
                language_code,
                utterance_id,
                generation_mode,
                condition_id,
                seed,
            )
            if experiment_cell in experiment_cells:
                raise ValueError(
                    "duplicate language/utterance/mode/condition/seed cell: "
                    f"{experiment_cell}"
                )
            experiment_cells.add(experiment_cell)

        if audio_sha256 in audio_hashes:
            global_findings.append(
                f"audio content reused by {audio_hashes[audio_sha256]} and {case_id}"
            )
        else:
            audio_hashes[audio_sha256] = case_id

        normalized = {
            "index": index,
            "case_id": case_id,
            "language_code": language_code,
            "generation_mode": generation_mode,
            "utterance_id": utterance_id,
            "seed": seed,
            "speaker_id": speaker_id,
            "voice_profile_id": voice_profile_id,
            "condition_id": condition_id,
            "audio_path": str(audio_path),
            "audio_bytes": audio_path.stat().st_size,
            "audio_sha256": audio_sha256,
            "expected_text": expected_text,
        }
        normalized_cases.append(normalized)
        grouped[language_code].append(normalized)

    language_design = {}
    design_pass = strict_design and not global_findings
    for language_code, language_cases in sorted(grouped.items()):
        findings = []
        modes = sorted({case["generation_mode"] for case in language_cases})
        utterances = sorted({case["utterance_id"] for case in language_cases})
        if len(language_cases) < minimum_samples_per_language:
            findings.append(
                f"requires at least {minimum_samples_per_language} samples; found {len(language_cases)}"
            )
        if len(utterances) < minimum_utterances_per_language:
            findings.append(
                f"requires at least {minimum_utterances_per_language} utterances; found {len(utterances)}"
            )
        for mode in required_generation_modes:
            if mode not in modes:
                findings.append(f"required generation mode missing: {mode}")

        replicate_groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)
        for case in language_cases:
            if isinstance(case["seed"], int) and not isinstance(case["seed"], bool):
                replicate_groups[
                    (case["utterance_id"], case["generation_mode"], case["condition_id"])
                ].add(case["seed"])
        missing_utterance_modes = [
            {"utterance_id": utterance_id, "generation_mode": mode}
            for utterance_id in utterances
            for mode in required_generation_modes
            if not any(
                key[0] == utterance_id and key[1] == mode for key in replicate_groups
            )
        ]
        insufficient_replicates = [
            {
                "utterance_id": utterance_id,
                "generation_mode": mode,
                "condition_id": condition_id,
                "distinct_seed_count": len(seeds),
            }
            for (utterance_id, mode, condition_id), seeds in sorted(replicate_groups.items())
            if len(seeds) < minimum_seeds_per_utterance_mode
        ]
        if missing_utterance_modes:
            findings.append(
                f"{len(missing_utterance_modes)} utterance/mode groups are missing"
            )
        if insufficient_replicates:
            findings.append(
                f"{len(insufficient_replicates)} utterance/mode groups have fewer than "
                f"{minimum_seeds_per_utterance_mode} distinct seeds"
            )

        language_pass = strict_design and not findings
        design_pass = design_pass and language_pass
        language_design[language_code] = {
            "sample_count": len(language_cases),
            "unique_audio_count": len({case["audio_sha256"] for case in language_cases}),
            "utterance_count": len(utterances),
            "generation_modes": modes,
            "minimum_required_samples": minimum_samples_per_language,
            "minimum_required_utterances": minimum_utterances_per_language,
            "minimum_required_distinct_seeds_per_utterance_mode": minimum_seeds_per_utterance_mode,
            "required_generation_modes": list(required_generation_modes),
            "missing_utterance_modes": missing_utterance_modes,
            "insufficient_replicates": insufficient_replicates,
            "design_gate_pass": language_pass,
            "denial_reasons": findings,
        }

    if not strict_design:
        global_findings.append(
            "strict experimental design was disabled; stable multilingual claims are not eligible"
        )
        design_pass = False
    return normalized_cases, {
        "strict_design": strict_design,
        "manifest_sha256": _sha256_file(manifest_path),
        "case_count": len(normalized_cases),
        "unique_case_count": len(case_ids),
        "unique_audio_count": len(audio_hashes),
        "language_design": language_design,
        "global_findings": global_findings,
        "design_gate_pass": design_pass,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1))
    return ordered[index]


def _error_rate_summary(values: list[float], threshold: float) -> dict:
    if not values:
        raise ValueError("error-rate summary requires at least one value")
    return {
        "sample_count": len(values),
        "mean_primary_error_rate": statistics.fmean(values),
        "median_primary_error_rate": statistics.median(values),
        "p90_primary_error_rate": _percentile(values, 0.90),
        "maximum_primary_error_rate": max(values),
        "case_pass_rate_at_threshold": sum(value <= threshold for value in values) / len(values),
    }


def summarize_result_breakdowns(results: list[dict], threshold: float) -> tuple[dict, list[dict], list[dict]]:
    """Expose mode/condition failure structure without changing the release gate."""
    mode_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    condition_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for result in results:
        language = str(result["language_code"])
        generation_mode = str(result["generation_mode"])
        condition_id = str(result.get("condition_id") or "<unspecified>")
        mode_groups[(language, generation_mode)].append(result)
        condition_groups[(language, generation_mode, condition_id)].append(result)

    mode_summary: dict[str, dict[str, dict]] = defaultdict(dict)
    for (language, generation_mode), items in sorted(mode_groups.items()):
        summary = _error_rate_summary(
            [float(item["metrics"]["primary_error_rate"]) for item in items],
            threshold,
        )
        summary["case_ids"] = [str(item["case_id"]) for item in items]
        summary["cases_over_threshold"] = [
            str(item["case_id"])
            for item in items
            if float(item["metrics"]["primary_error_rate"]) > threshold
        ]
        mode_summary[language][generation_mode] = summary

    condition_summary = []
    for (language, generation_mode, condition_id), items in sorted(condition_groups.items()):
        summary = {
            "language_code": language,
            "generation_mode": generation_mode,
            "condition_id": condition_id,
            **_error_rate_summary(
                [float(item["metrics"]["primary_error_rate"]) for item in items],
                threshold,
            ),
            "case_ids": [str(item["case_id"]) for item in items],
        }
        condition_summary.append(summary)

    outlier_cases = [
        {
            "case_id": str(result["case_id"]),
            "language_code": str(result["language_code"]),
            "generation_mode": str(result["generation_mode"]),
            "condition_id": str(result.get("condition_id") or "<unspecified>"),
            "utterance_id": str(result["utterance_id"]),
            "primary_metric": str(result["metrics"]["primary_metric"]),
            "primary_error_rate": float(result["metrics"]["primary_error_rate"]),
        }
        for result in results
        if float(result["metrics"]["primary_error_rate"]) > threshold
    ]
    outlier_cases.sort(key=lambda item: (-item["primary_error_rate"], item["case_id"]))
    return dict(mode_summary), condition_summary, outlier_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible multilingual WER/CER evaluation on generated speech files."
    )
    parser.add_argument("manifest", type=Path, help="JSON list or {cases:[...]} with audio_path, expected_text and language_code")
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--minimum-samples-per-language", type=int, default=30)
    parser.add_argument("--minimum-utterances-per-language", type=int, default=10)
    parser.add_argument("--minimum-seeds-per-utterance-mode", type=int, default=3)
    parser.add_argument("--required-generation-modes", default="described,clone")
    parser.add_argument("--maximum-primary-error-rate", type=float, default=0.15)
    parser.add_argument("--minimum-case-pass-rate", type=float, default=0.90)
    parser.add_argument(
        "--strict-design",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require unique case/audio evidence, explicit mode/utterance/seed and balanced replicates.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit the manifest design and file hashes without importing or running ASR.",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    metric_fn = _load_metrics()
    required_modes = _parse_required_modes(args.required_generation_modes)
    cases, design = validate_manifest_design(
        payload,
        args.manifest.resolve(),
        metric_fn,
        strict_design=args.strict_design,
        minimum_samples_per_language=args.minimum_samples_per_language,
        minimum_utterances_per_language=args.minimum_utterances_per_language,
        minimum_seeds_per_utterance_mode=args.minimum_seeds_per_utterance_mode,
        required_generation_modes=required_modes,
    )
    if args.validate_only:
        report = {
            "schema": "minimax_h3_t8_multilingual_speech_manifest_audit_v2",
            "manifest": str(args.manifest.resolve()),
            "evaluation_executed": False,
            "experimental_design": design,
            "stable_multilingual_gate_pass": False,
            "scientific_boundary": (
                "Manifest design and file identity were audited without ASR or H3 generation. "
                "A valid design is necessary but cannot prove speech quality."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "design_gate_pass": design["design_gate_pass"],
                    "evaluation_executed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if design["design_gate_pass"] else 2

    if args.asr_model is None:
        parser.error("--asr-model is required unless --validate-only is used")

    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(args.asr_model.resolve()),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.cpu_threads,
    )
    results = []
    grouped = defaultdict(list)
    for index, case in enumerate(cases):
        audio_path = Path(case["audio_path"])
        expected = case["expected_text"]
        language = case["language_code"]
        transcription = _transcribe(model, audio_path, language, args.beam_size)
        metrics = metric_fn(expected, transcription["text"])
        result = {
            "index": index,
            "case_id": case["case_id"],
            "language_code": language,
            "generation_mode": case["generation_mode"],
            "utterance_id": case["utterance_id"],
            "seed": case["seed"],
            "speaker_id": case["speaker_id"],
            "voice_profile_id": case["voice_profile_id"],
            "condition_id": case["condition_id"],
            "audio_path": str(audio_path),
            "audio_bytes": case["audio_bytes"],
            "audio_sha256": case["audio_sha256"],
            "expected_text": expected,
            "transcript": transcription["text"],
            "detected_language": transcription["detected_language"],
            "language_probability": transcription["language_probability"],
            "metrics": metrics,
        }
        results.append(result)
        grouped[result["language_code"]].append(metrics["primary_error_rate"])

    language_summary = {}
    all_gates_pass = design["design_gate_pass"]
    for language, values in sorted(grouped.items()):
        enough = len(values) >= args.minimum_samples_per_language
        rate_summary = _error_rate_summary(values, args.maximum_primary_error_rate)
        mean_error = rate_summary["mean_primary_error_rate"]
        rate_pass = mean_error <= args.maximum_primary_error_rate
        case_pass_rate = rate_summary["case_pass_rate_at_threshold"]
        case_pass_rate_ok = case_pass_rate >= args.minimum_case_pass_rate
        design_pass = bool(
            design["language_design"].get(language, {}).get("design_gate_pass", False)
        )
        gate_pass = design_pass and enough and rate_pass and case_pass_rate_ok
        all_gates_pass = all_gates_pass and gate_pass
        language_summary[language] = {
            **rate_summary,
            "minimum_required_samples": args.minimum_samples_per_language,
            "maximum_allowed_mean_error_rate": args.maximum_primary_error_rate,
            "maximum_allowed_case_error_rate": args.maximum_primary_error_rate,
            "minimum_required_case_pass_rate": args.minimum_case_pass_rate,
            "design_gate_pass": design_pass,
            "gate_pass": gate_pass,
            "denial_reasons": [
                reason
                for reason, failed in (
                    ("experimental design gate failed", not design_pass),
                    ("insufficient sample count", not enough),
                    ("mean error rate exceeds threshold", not rate_pass),
                    ("case pass rate is below threshold", not case_pass_rate_ok),
                )
                if failed
            ],
        }
    generation_mode_summary, condition_summary, outlier_cases = summarize_result_breakdowns(
        results,
        args.maximum_primary_error_rate,
    )
    report = {
        "schema": "minimax_h3_t8_multilingual_speech_validation_v2",
        "metric_normalization": (
            "Unicode NFKC + casefold; CJK/kana/hangul use character units, while accented "
            "Latin, Cyrillic, Arabic and other scripts use Unicode word units"
        ),
        "manifest": str(args.manifest.resolve()),
        "asr_model": str(args.asr_model.resolve()),
        "case_count": len(results),
        "evaluation_executed": True,
        "experimental_design": design,
        "language_summary": language_summary,
        "generation_mode_summary": generation_mode_summary,
        "condition_summary": condition_summary,
        "outlier_cases": outlier_cases,
        "stable_multilingual_gate_pass": all_gates_pass,
        "results": results,
        "scientific_boundary": (
            "ASR WER/CER measures text accuracy only. It does not establish speaker identity, "
            "naturalness, acting quality, or high-fidelity cloning."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate_pass": all_gates_pass, "languages": language_summary}, ensure_ascii=False, indent=2))
    return 0 if all_gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
