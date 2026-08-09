#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_multilingual_tool"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible multilingual WER/CER evaluation on generated speech files."
    )
    parser.add_argument("manifest", type=Path, help="JSON list or {cases:[...]} with audio_path, expected_text and language_code")
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--minimum-samples-per-language", type=int, default=30)
    parser.add_argument("--maximum-primary-error-rate", type=float, default=0.15)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain at least one case")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(args.asr_model.resolve()),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.cpu_threads,
    )
    metric_fn = _load_metrics()
    results = []
    grouped = defaultdict(list)
    for index, case in enumerate(cases):
        audio_path = Path(case["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (args.manifest.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        expected = str(case["expected_text"])
        language = str(case.get("language_code") or "").strip() or None
        transcription = _transcribe(model, audio_path, language, args.beam_size)
        metrics = metric_fn(expected, transcription["text"])
        result = {
            "index": index,
            "case_id": str(case.get("case_id") or index),
            "language_code": language or "auto",
            "audio_path": str(audio_path),
            "expected_text": expected,
            "transcript": transcription["text"],
            "detected_language": transcription["detected_language"],
            "language_probability": transcription["language_probability"],
            "metrics": metrics,
        }
        results.append(result)
        grouped[result["language_code"]].append(metrics["primary_error_rate"])

    language_summary = {}
    all_gates_pass = True
    for language, values in sorted(grouped.items()):
        enough = len(values) >= args.minimum_samples_per_language
        mean_error = statistics.fmean(values)
        rate_pass = mean_error <= args.maximum_primary_error_rate
        gate_pass = enough and rate_pass
        all_gates_pass = all_gates_pass and gate_pass
        language_summary[language] = {
            "sample_count": len(values),
            "mean_primary_error_rate": mean_error,
            "median_primary_error_rate": statistics.median(values),
            "maximum_primary_error_rate": max(values),
            "minimum_required_samples": args.minimum_samples_per_language,
            "maximum_allowed_mean_error_rate": args.maximum_primary_error_rate,
            "gate_pass": gate_pass,
            "denial_reasons": [
                reason
                for reason, failed in (
                    ("insufficient sample count", not enough),
                    ("mean error rate exceeds threshold", not rate_pass),
                )
                if failed
            ],
        }
    report = {
        "schema": "minimax_h3_t8_multilingual_speech_validation_v1",
        "manifest": str(args.manifest.resolve()),
        "asr_model": str(args.asr_model.resolve()),
        "case_count": len(results),
        "language_summary": language_summary,
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
