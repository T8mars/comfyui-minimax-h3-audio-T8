#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


KEY_SCHEMA = "t8.external_bridge_blind_package.v1"
REVIEW_SCHEMA = "t8.external_bridge_blind_review.v1"
OUTPUT_SCHEMA = "t8.external_bridge_blind_review_analysis.v1"
DEFAULT_GENERALIZATION = (
    "Each result applies only to its fixed method pair, material, prompt, seed and "
    "settings. ClipProj and Sol-Attn are different treatments; their votes must not "
    "be pooled into one universal quality, speed, memory or audio claim."
)
ARMS = ("control", "candidate")
PREFERENCE_FIELDS = (
    "overall",
    "motion",
    "audio",
    "prompt_adherence",
    "stability",
)
REFERENCE_METRICS = {"first_frame", "last_frame", "identity"}
ASSESSABILITY_VALUES = {
    "assessable",
    "source_material_insufficient",
    "playback_problem",
    "unsure",
}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _pair_map(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"Every {label} row must be an object")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Every {label} row must have a non-empty pair_id")
        if pair_id in result:
            raise ValueError(f"Duplicate {label} pair_id: {pair_id}")
        result[pair_id] = row
    return result


def _side_mapping(key_row: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    pair_id = key_row["pair_id"]
    sides = key_row.get("sides")
    if not isinstance(sides, list) or len(sides) != 2:
        raise ValueError(f"{pair_id} must contain exactly two key sides")
    mapping: dict[str, dict[str, str]] = {}
    seen_arms: set[str] = set()
    for side in sides:
        if not isinstance(side, dict):
            raise ValueError(f"{pair_id} has a malformed key side")
        code = side.get("code")
        arm = side.get("arm")
        method = side.get("method")
        digest = side.get("sha256")
        if code not in {"A", "B"} or arm not in ARMS:
            raise ValueError(f"{pair_id} has an invalid code or arm")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"{pair_id}.{code} method must be non-empty text")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(f"{pair_id}.{code} sha256 must be 64 hexadecimal characters")
        if code in mapping or arm in seen_arms:
            raise ValueError(f"{pair_id} repeats a code or arm")
        mapping[code] = {
            "arm": arm,
            "method": method.strip(),
            "sha256": digest.upper(),
        }
        seen_arms.add(arm)
    if set(mapping) != {"A", "B"} or seen_arms != set(ARMS):
        raise ValueError(f"{pair_id} must map A/B to control/candidate exactly once")
    return mapping


def _reference_metrics(key_row: Mapping[str, Any]) -> tuple[str, ...]:
    pair_id = key_row["pair_id"]
    metrics = key_row.get("reference_metrics", [])
    if not isinstance(metrics, list) or any(not isinstance(value, str) for value in metrics):
        raise ValueError(f"{pair_id} reference_metrics must be a list of strings")
    if len(metrics) != len(set(metrics)):
        raise ValueError(f"{pair_id} reference_metrics must not contain duplicates")
    unknown = sorted(set(metrics) - REFERENCE_METRICS)
    if unknown:
        raise ValueError(f"{pair_id} has unsupported reference_metrics: {unknown}")
    return tuple(metrics)


def _map_preference(
    value: Any, mapping: Mapping[str, Mapping[str, str]], *, field: str
) -> tuple[str, bool]:
    if value in (None, ""):
        return "tie", True
    if value == "tie":
        return "tie", False
    if value not in mapping:
        raise ValueError(f"{field} must be A, B, tie or omitted")
    return mapping[str(value)]["arm"], False


def _map_blocking_failure(
    value: Any, mapping: Mapping[str, Mapping[str, str]], *, field: str
) -> tuple[str, bool]:
    if value in (None, ""):
        return "none", True
    if value in {"none", "both"}:
        return str(value), False
    if value not in mapping:
        raise ValueError(f"{field} must be none, A, B, both or omitted")
    return mapping[str(value)]["arm"], False


def _map_assessability(value: Any, *, field: str) -> tuple[str, bool]:
    # Version-1 exports did not contain this field. Treat them exactly as the old
    # analyzer did, while reporting that the assumption came from a legacy export.
    if value in (None, ""):
        return "assessable", True
    if value not in ASSESSABILITY_VALUES:
        allowed = ", ".join(sorted(ASSESSABILITY_VALUES))
        raise ValueError(f"{field} must be one of: {allowed}, or omitted")
    return str(value), False


def _pair_decision(
    overall: str, blocking_failure: str, assessability: str = "assessable"
) -> str:
    if assessability == "source_material_insufficient":
        return "abstain_source_material_insufficient"
    if assessability == "playback_problem":
        return "abstain_playback_problem"
    if assessability == "unsure":
        return "abstain_reviewer_unsure"
    if blocking_failure == "candidate":
        return "candidate_rejected_explicit_failure"
    if blocking_failure == "control":
        return "control_failed_fixed_pair_candidate_not_promoted"
    if blocking_failure == "both":
        return "comparison_invalid_both_failed"
    if overall == "control":
        return "candidate_not_preferred_fixed_pair"
    if overall == "candidate":
        return "candidate_preferred_fixed_pair_single_reviewer_only"
    return "inconclusive_tie"


def _analysis_generalization(key: Mapping[str, Any]) -> str:
    contract = key.get("analysis_contract")
    if contract is None:
        return DEFAULT_GENERALIZATION
    if not isinstance(contract, Mapping):
        raise ValueError("Blind key analysis_contract must be an object")
    value = contract.get("generalization")
    if not isinstance(value, str) or not value.strip() or len(value) > 800:
        raise ValueError(
            "Blind key analysis_contract.generalization must be 1..800 characters"
        )
    return value.strip()


def analyze_review(
    review: Mapping[str, Any],
    key: Mapping[str, Any],
    *,
    required_independent_reviewers: int = 3,
) -> dict[str, Any]:
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"Review schema must be {REVIEW_SCHEMA}")
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError(f"Blind key schema must be {KEY_SCHEMA}")
    if required_independent_reviewers < 1:
        raise ValueError("required_independent_reviewers must be positive")
    review_id = review.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("Review review_id must be non-empty text")
    if review_id != key.get("review_id"):
        raise ValueError("Review/key review_id differs")
    generalization = _analysis_generalization(key)

    reviews = _pair_map(review.get("reviews"), "review")
    keys = _pair_map(key.get("pairs"), "key")
    if set(reviews) != set(keys):
        missing = sorted(set(keys) - set(reviews))
        extra = sorted(set(reviews) - set(keys))
        raise ValueError(f"Review/key pair IDs differ; missing={missing}, extra={extra}")

    preference_counts = {
        field: {"control": 0, "candidate": 0, "tie": 0}
        for field in PREFERENCE_FIELDS
    }
    reference_preference_counts = {
        metric: {"control": 0, "candidate": 0, "tie": 0, "applicable_pairs": 0}
        for metric in sorted(REFERENCE_METRICS)
    }
    blocking_failure_counts = {"none": 0, "control": 0, "candidate": 0, "both": 0}
    assessability_counts = {
        value: 0 for value in sorted(ASSESSABILITY_VALUES)
    }
    legacy_assessability_defaulted_pairs = 0
    defaulted_values = 0
    pair_results = []

    for pair_id, key_row in keys.items():
        review_row = reviews[pair_id]
        mapping = _side_mapping(key_row)
        assessability, legacy_assessability_defaulted = _map_assessability(
            review_row.get("assessability"), field=f"{pair_id}.assessability"
        )
        assessability_counts[assessability] += 1
        legacy_assessability_defaulted_pairs += int(
            legacy_assessability_defaulted
        )
        preferences_counted = assessability == "assessable"
        preferences: dict[str, str] = {}
        for field in PREFERENCE_FIELDS:
            mapped, defaulted = _map_preference(
                review_row.get(field), mapping, field=f"{pair_id}.{field}"
            )
            preferences[field] = mapped
            if preferences_counted:
                preference_counts[field][mapped] += 1
            defaulted_values += int(defaulted)

        expected_reference_metrics = _reference_metrics(key_row)
        raw_reference_preferences = review_row.get("reference_metrics", {})
        if raw_reference_preferences is None:
            raw_reference_preferences = {}
        if not isinstance(raw_reference_preferences, dict):
            raise ValueError(f"{pair_id}.reference_metrics must be an object")
        extra_metrics = sorted(
            set(raw_reference_preferences) - set(expected_reference_metrics)
        )
        if extra_metrics:
            raise ValueError(
                f"{pair_id}.reference_metrics contains unexpected metrics: {extra_metrics}"
            )
        mapped_reference_preferences: dict[str, str] = {}
        for metric in expected_reference_metrics:
            mapped, defaulted = _map_preference(
                raw_reference_preferences.get(metric),
                mapping,
                field=f"{pair_id}.reference_metrics.{metric}",
            )
            mapped_reference_preferences[metric] = mapped
            if preferences_counted:
                reference_preference_counts[metric][mapped] += 1
                reference_preference_counts[metric]["applicable_pairs"] += 1
            defaulted_values += int(defaulted)

        blocking_failure, defaulted = _map_blocking_failure(
            review_row.get("blocking_failure"),
            mapping,
            field=f"{pair_id}.blocking_failure",
        )
        if preferences_counted:
            blocking_failure_counts[blocking_failure] += 1
        defaulted_values += int(defaulted)
        notes = review_row.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise ValueError(f"{pair_id}.notes must be text")

        revealed_sides = {
            code: {
                "arm": side["arm"],
                "method": side["method"],
                "sha256": side["sha256"],
            }
            for code, side in mapping.items()
        }
        pair_results.append(
            {
                "pair_id": pair_id,
                "revealed_sides": revealed_sides,
                "assessability": assessability,
                "assessability_defaulted_from_legacy_export": (
                    legacy_assessability_defaulted
                ),
                "preferences_counted": preferences_counted,
                "preferences_by_arm": preferences,
                "reference_preferences_by_arm": mapped_reference_preferences,
                "blocking_failure_by_arm": blocking_failure,
                "notes": notes,
                "decision": _pair_decision(
                    preferences["overall"], blocking_failure, assessability
                ),
            }
        )

    any_candidate_rejection = any(
        row["decision"]
        in {
            "candidate_rejected_explicit_failure",
            "candidate_not_preferred_fixed_pair",
        }
        for row in pair_results
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "source_review_schema": REVIEW_SCHEMA,
        "source_key_schema": KEY_SCHEMA,
        "review_id": review_id,
        "exported_at": review.get("exported_at"),
        "reviewer_exports_analyzed": 1,
        "pair_count": len(keys),
        "assessable_pair_count": assessability_counts["assessable"],
        "excluded_unassessable_pair_count": len(keys)
        - assessability_counts["assessable"],
        "assessability_counts": assessability_counts,
        "legacy_assessability_defaulted_pairs": (
            legacy_assessability_defaulted_pairs
        ),
        "defaulted_omitted_values": defaulted_values,
        "preference_counts": preference_counts,
        "reference_preference_counts": reference_preference_counts,
        "blocking_failure_counts": blocking_failure_counts,
        "pair_results": pair_results,
        "decision": {
            "candidate_quality_noninferiority": (
                "failed_for_at_least_one_fixed_pair"
                if any_candidate_rejection
                else "not_established"
            ),
            "stable_default_or_auto_enable": "denied",
            "completed_independent_reviewers": 1,
            "required_independent_reviewers_for_promotion": required_independent_reviewers,
            "promotion_panel_gate": (
                "not_met_unassessable_pairs"
                if assessability_counts["assessable"] != len(keys)
                else (
                    "met"
                    if required_independent_reviewers == 1
                    else "not_met"
                )
            ),
            "generalization": generalization,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reveal and summarize one external-bridge blind review export."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-reviewers", type=int, default=3)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    result = analyze_review(
        review,
        key,
        required_independent_reviewers=args.required_reviewers,
    )
    result["source_files"] = {
        "review": str(args.review.resolve()),
        "review_sha256": _sha256(args.review),
        "blind_key": str(args.blind_key.resolve()),
        "blind_key_sha256": _sha256(args.blind_key),
    }
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "pairs": result["pair_count"],
                "overall": result["preference_counts"]["overall"],
                "blocking_failures": result["blocking_failure_counts"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
