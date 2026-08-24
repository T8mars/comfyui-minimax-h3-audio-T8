#!/usr/bin/env python3
"""Build and collect a pre-registered MiniMax H3 multilingual speech matrix.

This tool never submits a ComfyUI prompt. It deterministically expands reviewed language text,
three seeds, described voices and licensed clone references into separate API prompt JSON files.
The collector creates the strict ``validate_speech_multilingual.py`` manifest only after every
planned output resolves to exactly one unique audio file. Missing, ambiguous or duplicate audio
therefore cannot be converted into a formal validation claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping


SCHEMA = "minimax_h3_t8_multilingual_speech_formal_plan_v1"
SPEC_SCHEMA = "minimax_h3_t8_multilingual_speech_formal_spec_v1"
SOURCE_SCHEMA = "minimax_h3_t8_licensed_voice_sources_v1"
COLLECTION_SCHEMA = "minimax_h3_t8_multilingual_speech_collection_v1"
DEFAULT_SEEDS = (2608232101, 2608232102, 2608232103)
SUPPORTED_AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = str(value.get(key) or "").strip()
    if not result:
        raise ValueError(f"{context} requires non-empty {key}")
    return result


def _safe_token(value: str, context: str) -> str:
    token = str(value).strip()
    if not token or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in token
    ):
        raise ValueError(f"{context} must use only ASCII letters, digits, underscore or hyphen")
    return token


def _find_one(prompt: Mapping[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise ValueError(f"template requires exactly one {class_type}; found {len(matches)}")
    return matches[0]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _audio_contract(
    path: Path,
    *,
    context: str = "audio",
    minimum_duration_seconds: float = 2.0,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import soundfile
    except ImportError as error:
        raise RuntimeError("soundfile is required to audit speech audio") from error
    try:
        info = soundfile.info(str(path))
    except Exception as error:
        raise ValueError(f"{context} is not decodable audio: {path}") from error
    duration_seconds = float(info.duration)
    if not math.isfinite(duration_seconds) or duration_seconds < minimum_duration_seconds:
        raise ValueError(
            f"{context} is shorter than {minimum_duration_seconds:g} seconds: {path}"
        )
    if int(info.samplerate) <= 0 or int(info.channels) <= 0:
        raise ValueError(f"{context} has an invalid sample rate or channel count: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "duration_seconds": duration_seconds,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _finite_number(value: Any, key: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{key} must be within [{minimum:g}, {maximum:g}]")
    return result


def _safe_input_name(value: str, context: str) -> str:
    normalized = str(value).replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES
    ):
        raise ValueError(f"{context} must be a relative ComfyUI input audio path")
    return str(path)


def _normalize_spec(payload: Any, *, minimum_utterances: int) -> dict[str, Any]:
    if isinstance(minimum_utterances, bool) or minimum_utterances < 1:
        raise ValueError("minimum_utterances must be a positive integer")
    if not isinstance(payload, dict) or payload.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"spec schema must be {SPEC_SCHEMA}")
    plan_id = _safe_token(_required_text(payload, "plan_id", "spec"), "plan_id")
    seeds = payload.get("seeds", list(DEFAULT_SEEDS))
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(seed < 0 or seed > (2**64 - 1) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("spec requires at least three distinct integer seeds")
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ValueError("spec requires at least one language")
    normalized_languages = []
    seen_languages: set[str] = set()
    for language_index, raw in enumerate(languages):
        context = f"language {language_index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        language_code = _safe_token(_required_text(raw, "language_code", context), "language_code")
        language_key = language_code.casefold()
        if language_key == "auto" or language_key in seen_languages:
            raise ValueError(f"duplicate or automatic language_code: {language_code}")
        seen_languages.add(language_key)
        studio_language = _required_text(raw, "studio_language", context)
        profile_id = _safe_token(
            _required_text(raw, "described_voice_profile_id", context),
            "described_voice_profile_id",
        )
        description = _required_text(raw, "described_voice_description", context)
        utterances = raw.get("utterances")
        if not isinstance(utterances, list) or len(utterances) < minimum_utterances:
            raise ValueError(
                f"{language_code} requires at least {minimum_utterances} reviewed utterances"
            )
        normalized_utterances = []
        seen_utterances: set[str] = set()
        for utterance_index, utterance in enumerate(utterances):
            utterance_context = f"{language_code} utterance {utterance_index}"
            if not isinstance(utterance, dict):
                raise ValueError(f"{utterance_context} must be an object")
            utterance_id = _safe_token(
                _required_text(utterance, "utterance_id", utterance_context),
                "utterance_id",
            )
            text = _required_text(utterance, "text", utterance_context)
            utterance_key = utterance_id.casefold()
            if utterance_key in seen_utterances:
                raise ValueError(f"duplicate utterance_id in {language_code}: {utterance_id}")
            seen_utterances.add(utterance_key)
            normalized_utterances.append({"utterance_id": utterance_id, "text": text})
        normalized_languages.append(
            {
                "language_code": language_code,
                "studio_language": studio_language,
                "described_voice_profile_id": profile_id,
                "described_voice_description": description,
                "utterances": normalized_utterances,
            }
        )
    steps = payload.get("steps", 20)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer within [1, 100]")
    sampler_name = _required_text(payload, "sampler_name", "spec")
    scheduler = _required_text(payload, "scheduler", "spec")
    return {
        "schema": SPEC_SCHEMA,
        "plan_id": plan_id,
        "seeds": seeds,
        "render_seconds": _finite_number(
            payload.get("render_seconds", 8.0),
            "render_seconds",
            minimum=2.0,
            maximum=15.0,
        ),
        "steps": steps,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "shift_video": _finite_number(
            payload.get("shift_video", 12.0),
            "shift_video",
            minimum=0.000001,
            maximum=1000.0,
        ),
        "shift_audio": _finite_number(
            payload.get("shift_audio", 3.0),
            "shift_audio",
            minimum=0.000001,
            maximum=1000.0,
        ),
        "text_set_scope": _required_text(payload, "text_set_scope", "spec"),
        "languages": normalized_languages,
    }


def _normalize_sources(
    payload: Any,
    *,
    minimum_speakers: int,
    source_base: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(minimum_speakers, bool) or minimum_speakers < 1:
        raise ValueError("minimum_speakers must be a positive integer")
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"clone source schema must be {SOURCE_SCHEMA}")
    license_name = _required_text(payload, "license", "clone sources")
    license_source = _required_text(payload, "license_source", "clone sources")
    speakers = payload.get("speakers")
    if not isinstance(speakers, list) or len(speakers) < minimum_speakers:
        raise ValueError(f"clone sources require at least {minimum_speakers} speakers")
    normalized = []
    ids: set[str] = set()
    source_ids: set[str] = set()
    hashes: set[str] = set()
    for index, raw in enumerate(speakers):
        context = f"clone speaker {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        speaker_id = _safe_token(_required_text(raw, "speaker_id", context), "speaker_id")
        source_id = _required_text(raw, "source_id", context)
        input_name = _safe_input_name(_required_text(raw, "input_name", context), "input_name")
        reference = Path(_required_text(raw, "reference_audio", context))
        if not reference.is_absolute():
            reference = source_base / reference
        reference = reference.resolve()
        speaker_key = speaker_id.casefold()
        if speaker_key in ids:
            raise ValueError(f"duplicate clone speaker_id: {speaker_id}")
        ids.add(speaker_key)
        source_key = source_id.casefold()
        if source_key in source_ids:
            raise ValueError(f"duplicate clone source_id: {source_id}")
        source_ids.add(source_key)
        contract = _audio_contract(reference, context="clone reference")
        if contract["sha256"] in hashes:
            raise ValueError("clone speaker references must have unique audio content")
        hashes.add(contract["sha256"])
        normalized.append(
            {
                "speaker_id": speaker_id,
                "source_id": source_id,
                "input_name": input_name,
                "reference": contract,
                "source_language_code": _safe_token(
                    str(raw.get("source_language_code") or "en"),
                    "source_language_code",
                ),
            }
        )
    provenance = {
        "dataset": _required_text(payload, "dataset", "clone sources"),
        "dataset_revision": _required_text(payload, "dataset_revision", "clone sources"),
        "license": license_name,
        "license_source": license_source,
        "speaker_count": len(normalized),
    }
    return provenance, normalized


def _prepare_prompt(
    template: Mapping[str, Any],
    *,
    mode: str,
    language: Mapping[str, Any],
    utterance: Mapping[str, Any],
    seed: int,
    output_prefix: str,
    spec: Mapping[str, Any],
    clone_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prompt = deepcopy(dict(template))
    _, profile = _find_one(prompt, "MiniMaxH3VoiceProfileT8")
    _, speech_plan = _find_one(prompt, "MiniMaxH3SpeechPlanT8")
    _, studio = _find_one(prompt, "MiniMaxH3SpeechStudioT8")
    _, save = _find_one(prompt, "SaveAudio")
    profile_inputs = profile.setdefault("inputs", {})
    plan_inputs = speech_plan.setdefault("inputs", {})
    studio_inputs = studio.setdefault("inputs", {})
    save_inputs = save.setdefault("inputs", {})

    if mode == "described":
        profile_inputs.update(
            {
                "voice_mode": "described_voice",
                "speaker_id": language["described_voice_profile_id"],
                "voice_description": language["described_voice_description"],
                "language": language["studio_language"],
                "rights_confirmed": False,
                "reference_start_seconds": 0.0,
                "reference_duration_seconds": 0.0,
            }
        )
    elif mode == "clone":
        if clone_source is None:
            raise ValueError("clone prompt requires a clone source")
        _, load_audio = _find_one(prompt, "LoadAudio")
        load_audio.setdefault("inputs", {})["audio"] = clone_source["input_name"]
        profile_inputs.update(
            {
                "voice_mode": "reference_voice",
                "speaker_id": clone_source["speaker_id"],
                "voice_description": (
                    "the same licensed adult speaker as the connected reference, with natural "
                    "clear diction"
                ),
                "language": language["studio_language"],
                "rights_confirmed": True,
                "reference_start_seconds": 0.0,
                "reference_duration_seconds": min(
                    15.0, float(clone_source["reference"]["duration_seconds"])
                ),
            }
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    plan_inputs.update(
        {
            "text": utterance["text"],
            "language": language["studio_language"],
            "acting_direction": "neutral, clear, conversational, and naturally paced",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "space": "studio",
            "chunking": "single_segment",
        }
    )
    studio_inputs.update(
        {
            "segment_index": 0,
            "seed": int(seed),
            "render_seconds": float(spec["render_seconds"]),
            "steps": int(spec["steps"]),
            "sampler_name": spec["sampler_name"],
            "scheduler": spec["scheduler"],
            "shift_video": float(spec["shift_video"]),
            "shift_audio": float(spec["shift_audio"]),
            "trim_mode": "none",
            "verify_mode": "off",
            "asr_language": language["studio_language"],
            "speaker_check_mode": "off",
            "release_policy": "unload_all_models",
        }
    )
    save_inputs["filename_prefix"] = output_prefix
    return prompt


def build_plan(
    *,
    spec_payload: Any,
    source_payload: Any,
    described_template: Mapping[str, Any],
    clone_template: Mapping[str, Any],
    spec_path: Path,
    sources_path: Path,
    described_template_path: Path,
    clone_template_path: Path,
    minimum_utterances: int = 10,
    minimum_clone_speakers: int = 10,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    spec = _normalize_spec(spec_payload, minimum_utterances=minimum_utterances)
    source_provenance, sources = _normalize_sources(
        source_payload,
        minimum_speakers=minimum_clone_speakers,
        source_base=sources_path.resolve().parent,
    )
    cases = []
    prompts: dict[str, dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    seen_output_prefixes: set[str] = set()
    for language in spec["languages"]:
        for utterance_index, utterance in enumerate(language["utterances"]):
            clone_source = sources[utterance_index % len(sources)]
            for mode in ("described", "clone"):
                condition_id = (
                    language["described_voice_profile_id"]
                    if mode == "described"
                    else clone_source["speaker_id"]
                )
                for seed in spec["seeds"]:
                    case_id = _safe_token(
                        f"{language['language_code']}-{utterance['utterance_id']}-{mode}-{condition_id}-s{seed}",
                        "case_id",
                    )
                    case_key = case_id.casefold()
                    if case_key in seen_case_ids:
                        raise ValueError(f"duplicate generated case_id: {case_id}")
                    seen_case_ids.add(case_key)
                    output_prefix = str(
                        PurePosixPath("MiniMaxH3_T8_Speech")
                        / "multilingual_formal_v1"
                        / spec["plan_id"]
                        / case_id
                    )
                    output_key = output_prefix.casefold()
                    if output_key in seen_output_prefixes:
                        raise ValueError(f"duplicate generated output prefix: {output_prefix}")
                    seen_output_prefixes.add(output_key)
                    prompt = _prepare_prompt(
                        described_template if mode == "described" else clone_template,
                        mode=mode,
                        language=language,
                        utterance=utterance,
                        seed=seed,
                        output_prefix=output_prefix,
                        spec=spec,
                        clone_source=clone_source if mode == "clone" else None,
                    )
                    prompt_bytes = _json_bytes(prompt)
                    prompt_name = f"api_prompts/{case_id}.json"
                    prompts[prompt_name] = prompt
                    cases.append(
                        {
                            "case_id": case_id,
                            "language_code": language["language_code"],
                            "studio_language": language["studio_language"],
                            "generation_mode": mode,
                            "utterance_id": utterance["utterance_id"],
                            "seed": seed,
                            "speaker_id": clone_source["speaker_id"] if mode == "clone" else "",
                            "voice_profile_id": condition_id if mode == "described" else "",
                            "condition_id": condition_id,
                            "expected_text": utterance["text"],
                            "prompt_path": prompt_name,
                            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest().upper(),
                            "output_prefix": output_prefix,
                            "reference_audio": clone_source["reference"] if mode == "clone" else None,
                            "reference_source_id": clone_source["source_id"] if mode == "clone" else None,
                            "reference_source_language_code": (
                                clone_source["source_language_code"] if mode == "clone" else None
                            ),
                            "status": "PENDING_NOT_RUN",
                        }
                    )
    plan = {
        "schema": SCHEMA,
        "plan_id": spec["plan_id"],
        "execution_started": False,
        "evaluation_executed": False,
        "stable_multilingual_gate_pass": False,
        "spec": spec,
        "source_provenance": source_provenance,
        "source_files": {
            "spec": {"path": str(spec_path.resolve()), "sha256": _sha256_file(spec_path)},
            "clone_sources": {
                "path": str(sources_path.resolve()),
                "sha256": _sha256_file(sources_path),
            },
            "described_template": {
                "path": str(described_template_path.resolve()),
                "sha256": _sha256_file(described_template_path),
            },
            "clone_template": {
                "path": str(clone_template_path.resolve()),
                "sha256": _sha256_file(clone_template_path),
            },
        },
        "case_count": len(cases),
        "expected_case_count_formula": (
            f"{sum(len(language['utterances']) for language in spec['languages'])} total "
            f"language-utterances * 2 modes * {len(spec['seeds'])} seeds"
        ),
        "cases": cases,
        "scientific_boundary": (
            "This is a pre-registered execution plan only. It proves no H3 generation, text "
            "accuracy, speaker identity, naturalness, acting quality or clone fidelity."
        ),
    }
    return plan, prompts


def _write_new_atomic(target: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_plan(
    output: Path,
    plan: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
) -> Path:
    output = output.resolve()
    plan_bytes = _json_bytes(plan)
    plan_path = output / "plan.json"
    if output.exists() and any(output.iterdir()):
        if not plan_path.is_file() or plan_path.read_bytes() != plan_bytes:
            raise FileExistsError("non-empty output has no byte-identical authoritative plan.json")
    output.mkdir(parents=True, exist_ok=True)
    for relative, prompt in sorted(prompts.items()):
        target = output / PurePosixPath(relative)
        prompt_bytes = _json_bytes(prompt)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != prompt_bytes:
            raise ValueError(f"existing prompt differs from deterministic plan: {target}")
        if not target.exists():
            _write_new_atomic(target, prompt_bytes)
    if not plan_path.exists():
        _write_new_atomic(plan_path, plan_bytes)
    return plan_path


def _case_candidates(case: Mapping[str, Any], comfy_output: Path) -> list[Path]:
    prefix = PurePosixPath(str(case["output_prefix"]))
    if prefix.is_absolute() or any(part in {"", ".", ".."} for part in prefix.parts):
        raise ValueError(f"unsafe output_prefix in case {case.get('case_id')}: {prefix}")
    parent = comfy_output.joinpath(*prefix.parent.parts)
    if not parent.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in parent.glob(prefix.name + "*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
    )


def collect_outputs(plan: Mapping[str, Any], comfy_output: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"plan schema must be {SCHEMA}")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("plan requires a non-empty cases list")
    if plan.get("case_count") != len(cases):
        raise ValueError("plan case_count does not match cases")
    case_ids = [str(case.get("case_id") or "") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not case_id for case_id in case_ids):
        raise ValueError("every plan case requires a case_id")
    if len({case_id.casefold() for case_id in case_ids}) != len(case_ids):
        raise ValueError("plan case_id values must be unique")
    rows = []
    seen_hashes: dict[str, str] = {}
    complete_cases = []
    for case in cases:
        candidates = _case_candidates(case, comfy_output.resolve())
        row = {"case_id": case["case_id"], "candidates": [str(path) for path in candidates]}
        if not candidates:
            row["status"] = "PENDING_MISSING_OUTPUT"
        elif len(candidates) != 1:
            row["status"] = "ABSTAIN_AMBIGUOUS_OUTPUT"
        else:
            path = candidates[0]
            try:
                audio = _audio_contract(path, context="generated output")
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                row["status"] = "ABSTAIN_INVALID_AUDIO"
                row["error"] = str(error)
                rows.append(row)
                continue
            digest = audio["sha256"]
            if digest in seen_hashes:
                row["status"] = "ABSTAIN_DUPLICATE_AUDIO_CONTENT"
                row["duplicate_of"] = seen_hashes[digest]
            else:
                seen_hashes[digest] = str(case["case_id"])
                row.update(
                    {
                        "status": "COLLECTED_UNEVALUATED",
                        "audio_path": str(path),
                        "audio_bytes": path.stat().st_size,
                        "audio_sha256": digest,
                        "audio_contract": audio,
                    }
                )
                complete_cases.append(
                    {
                        "case_id": case["case_id"],
                        "language_code": case["language_code"],
                        "generation_mode": case["generation_mode"],
                        "utterance_id": case["utterance_id"],
                        "seed": case["seed"],
                        "speaker_id": case["speaker_id"],
                        "voice_profile_id": case["voice_profile_id"],
                        "audio_path": str(path),
                        "audio_bytes": path.stat().st_size,
                        "audio_sha256": digest,
                        "audio_contract": audio,
                        "expected_text": case["expected_text"],
                    }
                )
        rows.append(row)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    all_complete = len(complete_cases) == len(cases)
    report = {
        "schema": COLLECTION_SCHEMA,
        "created_at": _utc_now(),
        "plan_id": plan.get("plan_id"),
        "planned_case_count": len(cases),
        "collected_unique_case_count": len(complete_cases),
        "status_counts": status_counts,
        "all_outputs_collected": all_complete,
        "evaluation_executed": False,
        "stable_multilingual_gate_pass": False,
        "rows": rows,
        "scientific_boundary": (
            "Collection verifies file identity only. The strict ASR validator and human review "
            "have not run."
        ),
    }
    manifest = {"cases": complete_cases} if all_complete else None
    return report, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=root / "docs" / "specs" / "2026-08-23_speech_multilingual_en_zh_v1.json",
    )
    parser.add_argument("--clone-sources", type=Path, required=True)
    parser.add_argument(
        "--described-template",
        type=Path,
        default=root / "tests" / "fixtures" / "api" / "speech_described_api.json",
    )
    parser.add_argument(
        "--clone-template",
        type=Path,
        default=root / "tests" / "fixtures" / "api" / "speech_reference_clone_api.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-utterances", type=int, default=10)
    parser.add_argument("--minimum-clone-speakers", type=int, default=10)
    parser.add_argument(
        "--collect-from",
        type=Path,
        help="Collect existing SaveAudio outputs instead of rebuilding the plan.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.collect_from is not None:
        plan_path = args.output.resolve() / "plan.json"
        plan = _read_json(plan_path)
        report, manifest = collect_outputs(plan, args.collect_from)
        report_path = args.output.resolve() / "collection_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_bytes = _json_bytes(report)
        _write_new_atomic(report_path, report_bytes)
        manifest_path = args.output.resolve() / "multilingual_manifest.json"
        stale_manifest_removed = False
        if manifest is not None:
            _write_new_atomic(manifest_path, _json_bytes(manifest))
        elif manifest_path.exists():
            manifest_path.unlink()
            stale_manifest_removed = True
        print(
            json.dumps(
                {
                    "all_outputs_collected": report["all_outputs_collected"],
                    "report": str(report_path),
                    "manifest_written": manifest is not None,
                    "stale_manifest_removed": stale_manifest_removed,
                },
                ensure_ascii=False,
            )
        )
        return 0 if manifest is not None else 3

    plan, prompts = build_plan(
        spec_payload=_read_json(args.spec),
        source_payload=_read_json(args.clone_sources),
        described_template=_read_json(args.described_template),
        clone_template=_read_json(args.clone_template),
        spec_path=args.spec,
        sources_path=args.clone_sources,
        described_template_path=args.described_template,
        clone_template_path=args.clone_template,
        minimum_utterances=args.minimum_utterances,
        minimum_clone_speakers=args.minimum_clone_speakers,
    )
    plan_path = write_plan(args.output, plan, prompts)
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "case_count": plan["case_count"],
                "execution_started": False,
                "stable_multilingual_gate_pass": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
