#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


KEY_SCHEMA = "t8.face_refine_blind_package.v1"
REVIEW_SCHEMA = "t8.face_refine_blind_review.v1"
OUTPUT_SCHEMA = "t8.face_refine_blind_review_analysis.v1"
DIMENSIONS = (
    "identity",
    "expression_mouth",
    "temporal",
    "seam",
    "naturalness",
    "motion",
)
PREFERENCES = ("overall", "identity", "motion")
ARMS = ("source", "candidate")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def _pair_map(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Every {label} row must be an object")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Every {label} row must have a non-empty pair_id")
        if pair_id in result:
            raise ValueError(f"Duplicate {label} pair_id: {pair_id}")
        result[pair_id] = row
    return result


def _arm_codes(key_row: dict[str, Any]) -> dict[str, str]:
    sides = key_row.get("sides")
    if not isinstance(sides, list) or len(sides) != 2:
        raise ValueError(f"{key_row['pair_id']} must contain exactly two key sides")
    codes: dict[str, str] = {}
    seen_arms: set[str] = set()
    for side in sides:
        if not isinstance(side, dict):
            raise ValueError(f"{key_row['pair_id']} has a malformed key side")
        code = side.get("code")
        arm = side.get("arm")
        if code not in {"A", "B"} or arm not in ARMS:
            raise ValueError(f"{key_row['pair_id']} has an invalid code or arm")
        if code in codes or arm in seen_arms:
            raise ValueError(f"{key_row['pair_id']} repeats a code or arm")
        codes[code] = arm
        seen_arms.add(arm)
    if set(codes) != {"A", "B"} or seen_arms != set(ARMS):
        raise ValueError(f"{key_row['pair_id']} must map A/B to source/candidate")
    return codes


def _score(value: Any, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be blank or an integer from 1 to 5") from exc
    if parsed < 1 or parsed > 5 or str(value).strip() != str(parsed):
        raise ValueError(f"{field} must be blank or an integer from 1 to 5")
    return parsed


def _preference(value: Any, field: str, codes: dict[str, str]) -> str:
    if value in (None, ""):
        return "missing"
    if value == "tie":
        return "tie"
    if value not in codes:
        raise ValueError(f"{field} must be blank, A, B or tie")
    return codes[value]


def _score_summary(values: list[int], total: int) -> dict[str, Any]:
    return {
        "count": len(values),
        "missing": total - len(values),
        "mean": round(sum(values) / len(values), 6) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def analyze_review(
    review: dict[str, Any],
    key: dict[str, Any],
    *,
    required_independent_reviewers: int = 5,
) -> dict[str, Any]:
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"Review schema must be {REVIEW_SCHEMA}")
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError(f"Blind key schema must be {KEY_SCHEMA}")
    if required_independent_reviewers < 1:
        raise ValueError("required_independent_reviewers must be positive")

    reviews = _pair_map(review.get("reviews"), "review")
    keys = _pair_map(key.get("pairs"), "key")
    if set(reviews) != set(keys):
        missing = sorted(set(keys) - set(reviews))
        extra = sorted(set(reviews) - set(keys))
        raise ValueError(f"Review/key pair IDs differ; missing={missing}, extra={extra}")

    aggregate_scores: dict[str, dict[str, list[int]]] = {
        dimension: {arm: [] for arm in ARMS} for dimension in DIMENSIONS
    }
    preference_counts = {
        preference: {"source": 0, "candidate": 0, "tie": 0, "missing": 0}
        for preference in PREFERENCES
    }
    pair_results = []
    all_scores_complete = True
    all_preferences_complete = True

    for pair_id in keys:
        key_row = keys[pair_id]
        review_row = reviews[pair_id]
        codes = _arm_codes(key_row)
        arm_scores = {arm: {} for arm in ARMS}
        for code, arm in codes.items():
            for dimension in DIMENSIONS:
                field = f"{code}_{dimension}_1_to_5"
                parsed = _score(review_row.get(field), field)
                arm_scores[arm][dimension] = parsed
                if parsed is None:
                    all_scores_complete = False
                else:
                    aggregate_scores[dimension][arm].append(parsed)

        mapped_preferences = {}
        for preference in PREFERENCES:
            field = f"{preference}_preference"
            winner = _preference(review_row.get(field), field, codes)
            mapped_preferences[preference] = winner
            preference_counts[preference][winner] += 1
            if winner == "missing":
                all_preferences_complete = False

        notes = review_row.get("failure_notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise ValueError(f"{pair_id} failure_notes must be text")
        pair_results.append(
            {
                "pair_id": pair_id,
                "scores_by_arm": arm_scores,
                "preferences_by_arm": mapped_preferences,
                "failure_notes": notes,
            }
        )

    pair_count = len(keys)
    score_summaries = {}
    for dimension in DIMENSIONS:
        source = _score_summary(aggregate_scores[dimension]["source"], pair_count)
        candidate = _score_summary(
            aggregate_scores[dimension]["candidate"], pair_count
        )
        delta = None
        if source["mean"] is not None and candidate["mean"] is not None:
            delta = round(candidate["mean"] - source["mean"], 6)
        score_summaries[dimension] = {
            "source": source,
            "candidate": candidate,
            "candidate_minus_source_mean": delta,
        }

    overall = preference_counts["overall"]
    current_candidate_screen = "inconclusive_single_reviewer"
    if (
        overall["missing"] == 0
        and overall["candidate"] == 0
        and overall["source"] == pair_count
    ):
        current_candidate_screen = "rejected_for_this_fixed_source_and_settings"

    return {
        "schema": OUTPUT_SCHEMA,
        "source_review_schema": REVIEW_SCHEMA,
        "source_key_schema": KEY_SCHEMA,
        "exported_at": review.get("exported_at"),
        "reviewer_exports_analyzed": 1,
        "pair_count": pair_count,
        "export_declared_complete": bool(review.get("review_completed")),
        "all_scores_complete": all_scores_complete,
        "all_preferences_complete": all_preferences_complete,
        "scores_by_dimension": score_summaries,
        "preference_counts": preference_counts,
        "pair_results": pair_results,
        "decision": {
            "current_candidate_screen": current_candidate_screen,
            "quality_identity_promotion": "denied",
            "stable_default_or_auto_accept": "denied",
            "required_independent_reviewers_for_promotion": required_independent_reviewers,
            "completed_independent_reviewers": 1,
            "promotion_panel_gate": "not_met",
            "generalization": "not_granted_one_reviewer_one_source_repeated_control",
            "interpretation": (
                "This review can reject the six current candidates for the fixed source and "
                "settings, but cannot establish a cross-material or cross-reviewer universal claim."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reveal and summarize one exported Face Refine blind review."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-reviewers", type=int, default=5)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    result = analyze_review(
        review,
        key,
        required_independent_reviewers=args.required_reviewers,
    )
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "pairs": result["pair_count"],
                "overall": result["preference_counts"]["overall"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
