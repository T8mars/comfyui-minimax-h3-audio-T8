#!/usr/bin/env python3
"""Build a pre-registered MiniMax H3 voice-clone identity and ABX matrix.

The generation plan contains exactly ten licensed target speakers, ten reviewed English
utterances per target and three fixed seeds (300 clone outputs).  A separate 90-case ABX
schedule assigns three same-metadata-label impostors and three known seeds to every target.

This tool never submits a ComfyUI prompt.  Collection only inventories existing outputs and,
when all 300 unique files exist, writes a standardization-job contract.  It deliberately does
not create a blind ABX package until A/B/X have been normalized to the same codec, sample rate,
channel count and container by a later explicit step.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

try:
    from . import build_speech_multilingual_formal_matrix as matrix
except ImportError:  # pragma: no cover - direct script execution
    import build_speech_multilingual_formal_matrix as matrix


SPEC_SCHEMA = "minimax_h3_t8_voice_clone_identity_formal_spec_v1"
PLAN_KIND = "voice_clone_identity_formal"
DESIGN_SCHEMA = "minimax_h3_t8_voice_clone_identity_formal_design_v1"
COLLECTION_SCHEMA = "minimax_h3_t8_voice_clone_identity_collection_v1"
GENERATION_MANIFEST_SCHEMA = "minimax_h3_t8_voice_clone_identity_generation_manifest_v1"
STANDARDIZATION_JOBS_SCHEMA = "minimax_h3_t8_voice_clone_abx_standardization_jobs_v1"
DEFAULT_REQUIRED_TARGETS = 10
DEFAULT_REQUIRED_UTTERANCES = 10
DEFAULT_IMPOSTORS_PER_TARGET = 3
DEFAULT_REVIEWERS = 3
GENDER_LABELS = {"F", "M"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fixed_sha256(value: Any, context: str) -> str:
    token = str(value or "").strip().upper()
    if len(token) != 64 or any(char not in "0123456789ABCDEF" for char in token):
        raise ValueError(f"{context} must be a fixed 64-character SHA-256")
    return token


def _normalize_spec(
    payload: Any,
    *,
    required_targets: int = DEFAULT_REQUIRED_TARGETS,
    required_utterances: int = DEFAULT_REQUIRED_UTTERANCES,
    impostors_per_target: int = DEFAULT_IMPOSTORS_PER_TARGET,
) -> dict[str, Any]:
    for name, value in (
        ("required_targets", required_targets),
        ("required_utterances", required_utterances),
        ("impostors_per_target", impostors_per_target),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(payload, dict) or payload.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"spec schema must be {SPEC_SCHEMA}")

    plan_id = matrix._safe_token(matrix._required_text(payload, "plan_id", "spec"), "plan_id")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(seed < 0 or seed > (2**64 - 1) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("identity spec requires exactly three distinct integer seeds")

    language_code = matrix._safe_token(
        matrix._required_text(payload, "language_code", "spec"), "language_code"
    )
    if language_code.casefold() != "en":
        raise ValueError("the v1 identity plan is restricted to reviewed English text")
    studio_language = matrix._required_text(payload, "studio_language", "spec")

    utterances = payload.get("utterances")
    if not isinstance(utterances, list) or len(utterances) != required_utterances:
        raise ValueError(f"identity spec requires exactly {required_utterances} utterances")
    normalized_utterances = []
    seen_utterance_ids: set[str] = set()
    seen_text: set[str] = set()
    for index, raw in enumerate(utterances):
        context = f"utterance {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        utterance_id = matrix._safe_token(
            matrix._required_text(raw, "utterance_id", context), "utterance_id"
        )
        text = matrix._required_text(raw, "text", context)
        if utterance_id.casefold() in seen_utterance_ids:
            raise ValueError(f"duplicate utterance_id: {utterance_id}")
        if text.casefold() in seen_text:
            raise ValueError("identity utterance text must be unique")
        seen_utterance_ids.add(utterance_id.casefold())
        seen_text.add(text.casefold())
        normalized_utterances.append({"utterance_id": utterance_id, "text": text})

    metadata = payload.get("speaker_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("speaker_metadata must be an object")
    normalized_metadata = {
        "dataset": matrix._required_text(metadata, "dataset", "speaker_metadata"),
        "revision": matrix._required_text(metadata, "revision", "speaker_metadata"),
        "relative_path": matrix._required_text(metadata, "relative_path", "speaker_metadata"),
        "source_url": matrix._required_text(metadata, "source_url", "speaker_metadata"),
        "sha256": _fixed_sha256(metadata.get("sha256"), "speaker_metadata.sha256"),
        "label_semantics": matrix._required_text(
            metadata, "label_semantics", "speaker_metadata"
        ),
    }

    targets = payload.get("target_speakers")
    if not isinstance(targets, list) or len(targets) != required_targets:
        raise ValueError(f"identity spec requires exactly {required_targets} target speakers")
    normalized_targets = []
    seen_targets: set[str] = set()
    for index, raw in enumerate(targets):
        context = f"target speaker {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        speaker_id = matrix._safe_token(
            matrix._required_text(raw, "speaker_id", context), "speaker_id"
        )
        gender_label = matrix._required_text(raw, "gender_label", context).upper()
        if gender_label not in GENDER_LABELS:
            raise ValueError(f"{context}.gender_label must be F or M")
        split = matrix._required_text(raw, "split", context)
        if split != "test-clean":
            raise ValueError(f"{context}.split must be test-clean")
        if speaker_id.casefold() in seen_targets:
            raise ValueError(f"duplicate target speaker_id: {speaker_id}")
        seen_targets.add(speaker_id.casefold())
        normalized_targets.append(
            {"speaker_id": speaker_id, "gender_label": gender_label, "split": split}
        )

    group_counts = Counter(row["gender_label"] for row in normalized_targets)
    insufficient_groups = sorted(
        label for label, count in group_counts.items() if count < impostors_per_target + 1
    )
    if insufficient_groups:
        raise ValueError(
            "same-label impostor groups require target plus at least "
            f"{impostors_per_target} alternatives; insufficient={insufficient_groups}"
        )
    required_abx_utterances = impostors_per_target * len(seeds)
    if required_abx_utterances > len(normalized_utterances):
        raise ValueError("not enough unique utterances for the ABX impostor/seed grid")

    steps = payload.get("steps", 20)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer within [1, 100]")
    minimum_reviewers = payload.get("minimum_independent_reviewers", DEFAULT_REVIEWERS)
    if (
        isinstance(minimum_reviewers, bool)
        or not isinstance(minimum_reviewers, int)
        or minimum_reviewers < DEFAULT_REVIEWERS
    ):
        raise ValueError(f"minimum_independent_reviewers must be at least {DEFAULT_REVIEWERS}")

    standardization = payload.get("abx_standardization")
    if not isinstance(standardization, dict):
        raise ValueError("abx_standardization must be an object")
    expected_standardization = {
        "sample_rate": 32000,
        "channels": 1,
        "codec": "flac",
        "container": "flac",
        "loudness_normalization": False,
        "duration_policy": "preserve_each_source",
    }
    if standardization != expected_standardization:
        raise ValueError(
            "abx_standardization must use 32kHz mono FLAC, preserve duration and disable "
            "loudness normalization"
        )

    return {
        "schema": SPEC_SCHEMA,
        "plan_id": plan_id,
        "seeds": seeds,
        "render_seconds": matrix._finite_number(
            payload.get("render_seconds", 8.0),
            "render_seconds",
            minimum=2.0,
            maximum=15.0,
        ),
        "steps": steps,
        "sampler_name": matrix._required_text(payload, "sampler_name", "spec"),
        "scheduler": matrix._required_text(payload, "scheduler", "spec"),
        "shift_video": matrix._finite_number(
            payload.get("shift_video", 12.0),
            "shift_video",
            minimum=0.000001,
            maximum=1000.0,
        ),
        "shift_audio": matrix._finite_number(
            payload.get("shift_audio", 3.0),
            "shift_audio",
            minimum=0.000001,
            maximum=1000.0,
        ),
        "language_code": language_code,
        "studio_language": studio_language,
        "text_set_scope": matrix._required_text(payload, "text_set_scope", "spec"),
        "utterances": normalized_utterances,
        "speaker_metadata": normalized_metadata,
        "target_speakers": normalized_targets,
        "impostors_per_target": impostors_per_target,
        "minimum_independent_reviewers": minimum_reviewers,
        "abx_standardization": expected_standardization,
    }


def _impostor_assignments(
    targets: list[Mapping[str, Any]], impostors_per_target: int
) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in targets:
        by_label[str(row["gender_label"])].append(str(row["speaker_id"]))
    assignments: dict[str, list[str]] = {}
    for members in by_label.values():
        for index, speaker_id in enumerate(members):
            assignments[speaker_id] = [
                members[(index + offset) % len(members)]
                for offset in range(1, impostors_per_target + 1)
            ]
    return assignments


def build_identity_plan(
    *,
    spec_payload: Any,
    source_payload: Any,
    clone_template: Mapping[str, Any],
    spec_path: Path,
    sources_path: Path,
    clone_template_path: Path,
    required_targets: int = DEFAULT_REQUIRED_TARGETS,
    required_utterances: int = DEFAULT_REQUIRED_UTTERANCES,
    impostors_per_target: int = DEFAULT_IMPOSTORS_PER_TARGET,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    spec = _normalize_spec(
        spec_payload,
        required_targets=required_targets,
        required_utterances=required_utterances,
        impostors_per_target=impostors_per_target,
    )
    source_provenance, all_sources = matrix._normalize_sources(
        source_payload,
        minimum_speakers=required_targets,
        source_base=sources_path.resolve().parent,
    )
    sources_by_id = {str(row["speaker_id"]): row for row in all_sources}
    target_ids = [str(row["speaker_id"]) for row in spec["target_speakers"]]
    missing = sorted(set(target_ids) - set(sources_by_id))
    if missing:
        raise ValueError(f"target speakers missing from licensed source manifest: {missing}")
    selected_sources = {speaker_id: sources_by_id[speaker_id] for speaker_id in target_ids}
    assignments = _impostor_assignments(
        spec["target_speakers"], spec["impostors_per_target"]
    )
    gender_by_speaker = {
        str(row["speaker_id"]): str(row["gender_label"])
        for row in spec["target_speakers"]
    }

    language = {
        "language_code": spec["language_code"],
        "studio_language": spec["studio_language"],
        "described_voice_profile_id": "unused_identity_clone_plan",
        "described_voice_description": "unused identity clone plan",
    }
    cases: list[dict[str, Any]] = []
    prompts: dict[str, dict[str, Any]] = {}
    case_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    seen_output_prefixes: set[str] = set()
    for speaker_id in target_ids:
        source = selected_sources[speaker_id]
        for utterance in spec["utterances"]:
            for seed in spec["seeds"]:
                case_id = matrix._safe_token(
                    f"en-{utterance['utterance_id']}-clone-{speaker_id}-s{seed}", "case_id"
                )
                if case_id.casefold() in seen_case_ids:
                    raise ValueError(f"duplicate generated case_id: {case_id}")
                seen_case_ids.add(case_id.casefold())
                output_prefix = str(
                    PurePosixPath("MiniMaxH3_T8_Speech")
                    / "voice_clone_identity_formal_v1"
                    / spec["plan_id"]
                    / case_id
                )
                if output_prefix.casefold() in seen_output_prefixes:
                    raise ValueError(f"duplicate generated output prefix: {output_prefix}")
                seen_output_prefixes.add(output_prefix.casefold())
                prompt = matrix._prepare_prompt(
                    clone_template,
                    mode="clone",
                    language=language,
                    utterance=utterance,
                    seed=seed,
                    output_prefix=output_prefix,
                    spec=spec,
                    clone_source=source,
                )
                prompt_name = f"api_prompts/{case_id}.json"
                prompt_bytes = matrix._json_bytes(prompt)
                prompts[prompt_name] = prompt
                case = {
                    "case_id": case_id,
                    "language_code": spec["language_code"],
                    "studio_language": spec["studio_language"],
                    "generation_mode": "clone",
                    "utterance_id": utterance["utterance_id"],
                    "seed": seed,
                    "speaker_id": speaker_id,
                    "voice_profile_id": "",
                    "condition_id": speaker_id,
                    "expected_text": utterance["text"],
                    "prompt_path": prompt_name,
                    "prompt_sha256": _sha256_bytes(prompt_bytes),
                    "output_prefix": output_prefix,
                    "reference_audio": source["reference"],
                    "reference_source_id": source["source_id"],
                    "reference_source_language_code": source["source_language_code"],
                    "speaker_metadata_label": gender_by_speaker[speaker_id],
                    "status": "PENDING_NOT_RUN",
                }
                cases.append(case)
                case_index[(speaker_id, str(utterance["utterance_id"]), int(seed))] = case

    abx_schedule = []
    for target_id in target_ids:
        for impostor_index, impostor_id in enumerate(assignments[target_id]):
            for seed_index, seed in enumerate(spec["seeds"]):
                utterance = spec["utterances"][
                    impostor_index * len(spec["seeds"]) + seed_index
                ]
                candidate = case_index[(target_id, str(utterance["utterance_id"]), int(seed))]
                schedule_case_id = matrix._safe_token(
                    f"abx-{target_id}-vs-{impostor_id}-{utterance['utterance_id']}-s{seed}",
                    "abx case_id",
                )
                abx_schedule.append(
                    {
                        "case_id": schedule_case_id,
                        "target_speaker_id": target_id,
                        "impostor_speaker_id": impostor_id,
                        "metadata_block_label": gender_by_speaker[target_id],
                        "condition_id": target_id,
                        "utterance_id": utterance["utterance_id"],
                        "language_code": spec["language_code"],
                        "seed": seed,
                        "seed_known": True,
                        "candidate_generation_case_id": candidate["case_id"],
                        "candidate_output_prefix": candidate["output_prefix"],
                        "target_reference": selected_sources[target_id]["reference"],
                        "impostor_reference": selected_sources[impostor_id]["reference"],
                    }
                )

    plan = {
        "schema": matrix.SCHEMA,
        "plan_kind": PLAN_KIND,
        "plan_id": spec["plan_id"],
        "execution_started": False,
        "evaluation_executed": False,
        "stable_multilingual_gate_pass": False,
        "identity_discrimination_panel_gate": "NOT_RUN",
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        "spec": spec,
        "source_provenance": source_provenance,
        "source_files": {
            "spec": {"path": str(spec_path.resolve()), "sha256": matrix._sha256_file(spec_path)},
            "clone_sources": {
                "path": str(sources_path.resolve()),
                "sha256": matrix._sha256_file(sources_path),
            },
            "clone_template": {
                "path": str(clone_template_path.resolve()),
                "sha256": matrix._sha256_file(clone_template_path),
            },
        },
        "case_count": len(cases),
        "expected_case_count_formula": (
            f"{len(target_ids)} target speakers * {len(spec['utterances'])} utterances * "
            f"{len(spec['seeds'])} seeds"
        ),
        "identity_design": {
            "schema": DESIGN_SCHEMA,
            "target_speaker_count": len(target_ids),
            "utterances_per_target": len(spec["utterances"]),
            "seeds_per_target_utterance": len(spec["seeds"]),
            "impostors_per_target": spec["impostors_per_target"],
            "abx_cases_per_target": spec["impostors_per_target"] * len(spec["seeds"]),
            "minimum_independent_reviewers": spec["minimum_independent_reviewers"],
            "same_metadata_label_blocking": True,
        },
        "cases": cases,
        "scientific_boundary": (
            "This pre-registered plan proves no generation or clone fidelity. Corpus F/M labels "
            "are used only as a coarse blocking variable to avoid trivial cross-label ABX cues; "
            "they are not inferred biological sex or personal identity. A future passing ABX panel "
            "will support only fixed-set identity discrimination, never a high-fidelity claim."
        ),
    }
    plan_sha256 = _sha256_bytes(matrix._json_bytes(plan))
    design = {
        "schema": DESIGN_SCHEMA,
        "plan_id": spec["plan_id"],
        "generation_plan_sha256": plan_sha256,
        "generation_case_count": len(cases),
        "abx_case_count": len(abx_schedule),
        "target_speakers": [
            {
                **row,
                "licensed_source_id": selected_sources[str(row["speaker_id"])]["source_id"],
                "impostor_speaker_ids": assignments[str(row["speaker_id"])],
            }
            for row in spec["target_speakers"]
        ],
        "speaker_metadata": spec["speaker_metadata"],
        "utterances": spec["utterances"],
        "seeds": spec["seeds"],
        "abx_schedule": abx_schedule,
        "abx_standardization": spec["abx_standardization"],
        "analysis_thresholds": {
            "minimum_target_speakers": len(target_ids),
            "minimum_impostors_per_target": spec["impostors_per_target"],
            "minimum_seeds_per_target": len(spec["seeds"]),
            "minimum_independent_reviewers": spec["minimum_independent_reviewers"],
            "minimum_accuracy": 0.80,
            "minimum_wilson_95_lower": 0.65,
            "maximum_abstain_rate": 0.20,
            "maximum_invalid_rate": 0.05,
        },
        "decision_state": {
            "generation": "NOT_RUN",
            "abx_package": "NOT_MATERIALIZED",
            "panel": "NOT_RUN",
            "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        },
        "scientific_boundary": plan["scientific_boundary"],
    }
    return plan, prompts, design


def write_identity_plan(
    output: Path,
    plan: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
    design: Mapping[str, Any],
) -> tuple[Path, Path]:
    output = output.resolve()
    design_path = output / "identity_design.json"
    design_bytes = matrix._json_bytes(design)
    if design_path.exists() and design_path.read_bytes() != design_bytes:
        raise ValueError("existing identity design differs from deterministic plan")
    plan_path = matrix.write_plan(output, plan, prompts)
    if not design_path.exists():
        _write_atomic(design_path, design_bytes)
    return plan_path, design_path


def collect_identity_outputs(
    *,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    comfy_output: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    if plan.get("schema") != matrix.SCHEMA or plan.get("plan_kind") != PLAN_KIND:
        raise ValueError("identity collection requires an authoritative identity generation plan")
    if design.get("schema") != DESIGN_SCHEMA or design.get("plan_id") != plan.get("plan_id"):
        raise ValueError("identity design does not match the generation plan")
    plan_sha256 = _sha256_bytes(matrix._json_bytes(plan))
    if str(design.get("generation_plan_sha256") or "").upper() != plan_sha256:
        raise ValueError("identity design generation_plan_sha256 differs")
    collection, generated = matrix.collect_outputs(plan, comfy_output.resolve())
    wrapped = {
        "schema": COLLECTION_SCHEMA,
        "plan_id": plan["plan_id"],
        "generation_plan_sha256": plan_sha256,
        "planned_case_count": collection["planned_case_count"],
        "collected_unique_case_count": collection["collected_unique_case_count"],
        "status_counts": collection["status_counts"],
        "all_outputs_collected": collection["all_outputs_collected"],
        "identity_discrimination_panel_gate": "NOT_RUN",
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        "rows": collection["rows"],
        "scientific_boundary": (
            "Collection verifies generated file identity only. No A/B/X standardization, blind "
            "package, listening panel or clone-fidelity evaluation has run."
        ),
    }
    if generated is None:
        return wrapped, None, None

    generated_by_case = {str(row["case_id"]): row for row in generated["cases"]}
    manifest = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "plan_id": plan["plan_id"],
        "generation_plan_sha256": plan_sha256,
        "cases": generated["cases"],
        "evaluation_executed": False,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
    }
    jobs = []
    for row in design["abx_schedule"]:
        candidate = generated_by_case.get(str(row["candidate_generation_case_id"]))
        if candidate is None:
            raise ValueError("complete collection omitted a scheduled ABX candidate")
        case_id = str(row["case_id"])
        jobs.append(
            {
                "case_id": case_id,
                "target_speaker_id": row["target_speaker_id"],
                "impostor_speaker_id": row["impostor_speaker_id"],
                "condition_id": row["condition_id"],
                "utterance_id": row["utterance_id"],
                "language_code": row["language_code"],
                "seed": row["seed"],
                "seed_known": True,
                "inputs": {
                    "target_reference": {
                        "path": row["target_reference"]["path"],
                        "sha256": row["target_reference"]["sha256"],
                    },
                    "impostor_reference": {
                        "path": row["impostor_reference"]["path"],
                        "sha256": row["impostor_reference"]["sha256"],
                    },
                    "candidate": {
                        "path": candidate["audio_path"],
                        "sha256": candidate["audio_sha256"],
                    },
                },
                "outputs": {
                    "target_reference": f"standardized/references/{row['target_speaker_id']}.flac",
                    "impostor_reference": (
                        f"standardized/references/{row['impostor_speaker_id']}.flac"
                    ),
                    "candidate": f"standardized/candidates/{case_id}.flac",
                },
            }
        )
    standardization_jobs = {
        "schema": STANDARDIZATION_JOBS_SCHEMA,
        "plan_id": plan["plan_id"],
        "generation_plan_sha256": plan_sha256,
        "identity_design_sha256": _sha256_bytes(matrix._json_bytes(design)),
        "review_id": f"{plan['plan_id']}-abx-v1",
        "contract": design["abx_standardization"],
        "job_count": len(jobs),
        "jobs": jobs,
        "execution_started": False,
        "abx_manifest_written": False,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        "scientific_boundary": (
            "These are pre-registered normalization jobs, not normalized files and not an ABX "
            "manifest. The existing blind-package builder must still verify identical A/B/X media "
            "contracts and distinct content before review."
        ),
    }
    return wrapped, manifest, standardization_jobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=root / "docs" / "specs" / "2026-08-23_voice_clone_identity_en_v1.json",
    )
    parser.add_argument("--clone-sources", type=Path, required=True)
    parser.add_argument(
        "--clone-template",
        type=Path,
        default=root / "tests" / "fixtures" / "api" / "speech_reference_clone_api.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--collect-from",
        type=Path,
        help="Collect existing SaveAudio outputs; never submits prompts or normalizes audio.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if args.collect_from is not None:
        plan_path = output / "plan.json"
        design_path = output / "identity_design.json"
        plan = _read_json(plan_path)
        design = _read_json(design_path)
        report, manifest, jobs = collect_identity_outputs(
            plan=plan, design=design, comfy_output=args.collect_from
        )
        _write_atomic(output / "identity_collection_report.json", matrix._json_bytes(report))
        materialized = {
            "identity_generation_manifest.json": manifest,
            "abx_standardization_jobs.json": jobs,
        }
        for name, value in materialized.items():
            path = output / name
            if value is None:
                if path.exists():
                    path.unlink()
            else:
                _write_atomic(path, matrix._json_bytes(value))
        print(
            json.dumps(
                {
                    "all_outputs_collected": report["all_outputs_collected"],
                    "collected_unique_case_count": report["collected_unique_case_count"],
                    "generation_manifest_written": manifest is not None,
                    "standardization_jobs_written": jobs is not None,
                    "normalization_executed": False,
                    "panel_executed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0 if manifest is not None else 3

    plan, prompts, design = build_identity_plan(
        spec_payload=_read_json(args.spec),
        source_payload=_read_json(args.clone_sources),
        clone_template=_read_json(args.clone_template),
        spec_path=args.spec,
        sources_path=args.clone_sources,
        clone_template_path=args.clone_template,
    )
    plan_path, design_path = write_identity_plan(output, plan, prompts, design)
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "identity_design": str(design_path),
                "generation_case_count": plan["case_count"],
                "abx_case_count": design["abx_case_count"],
                "execution_started": False,
                "panel_executed": False,
                "high_fidelity_clone_claim": "NOT_ESTABLISHED",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
