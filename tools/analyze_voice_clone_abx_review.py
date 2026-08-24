from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

try:
    from .build_voice_clone_abx_review import KEY_SCHEMA, REVIEW_SCHEMA
except ImportError:  # pragma: no cover - direct script execution
    from build_voice_clone_abx_review import KEY_SCHEMA, REVIEW_SCHEMA


OUTPUT_SCHEMA = "minimax_h3_t8_voice_clone_abx_analysis_v1"
VALID_CHOICES = {"A", "B", "unclear", "invalid", ""}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rows_by_id(rows: Any, field: str, *, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context} must be a non-empty list")
    mapped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{context}[{index}] must be an object")
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context}[{index}].{field} must be non-empty text")
        if value in mapped:
            raise ValueError(f"duplicate {field}: {value}")
        mapped[value] = row
    return mapped


def _rating(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
        raise ValueError(f"{field} must be an integer from 0 to 5")
    return value


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def _mean_or_none(values: Iterable[int]) -> float | None:
    material = list(values)
    return statistics.fmean(material) if material else None


def analyze_reviews(
    reviews: list[Mapping[str, Any]],
    key: Mapping[str, Any],
    *,
    minimum_target_speakers: int = 10,
    minimum_impostors_per_target: int = 3,
    minimum_seeds_per_target: int = 3,
    minimum_independent_reviewers: int = 3,
    minimum_accuracy: float = 0.80,
    minimum_wilson_lower: float = 0.65,
    maximum_abstain_rate: float = 0.20,
    maximum_invalid_rate: float = 0.05,
) -> dict[str, Any]:
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError(f"Blind key schema must be {KEY_SCHEMA}")
    review_id = key.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("Blind key review_id must be non-empty text")
    key_cases = _rows_by_id(
        key.get("cases"), "blind_case_id", context="key.cases"
    )
    for name, value in (
        ("minimum_target_speakers", minimum_target_speakers),
        ("minimum_impostors_per_target", minimum_impostors_per_target),
        ("minimum_seeds_per_target", minimum_seeds_per_target),
        ("minimum_independent_reviewers", minimum_independent_reviewers),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("minimum_accuracy", minimum_accuracy),
        ("minimum_wilson_lower", minimum_wilson_lower),
        ("maximum_abstain_rate", maximum_abstain_rate),
        ("maximum_invalid_rate", maximum_invalid_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    target_speakers = {str(row["target_speaker_id"]) for row in key_cases.values()}
    impostors_by_target: dict[str, set[str]] = defaultdict(set)
    seeds_by_target: dict[str, set[int]] = defaultdict(set)
    cases_by_target: dict[str, int] = defaultdict(int)
    for blind_case_id, row in key_cases.items():
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"key case {blind_case_id} has invalid private case_id")
        target = row.get("target_speaker_id")
        impostor = row.get("impostor_speaker_id")
        target_code = row.get("target_code")
        seed = row.get("seed")
        if not isinstance(target, str) or not target:
            raise ValueError(f"key case {case_id} has invalid target_speaker_id")
        if not isinstance(impostor, str) or not impostor or impostor == target:
            raise ValueError(f"key case {case_id} has invalid impostor_speaker_id")
        if target_code not in {"A", "B"}:
            raise ValueError(f"key case {case_id} target_code must be A or B")
        seed_known = row.get("seed_known", True)
        if not isinstance(seed_known, bool):
            raise ValueError(f"key case {case_id} seed_known must be boolean")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"key case {case_id} seed must be an integer")
        impostors_by_target[target].add(impostor)
        if seed_known:
            seeds_by_target[target].add(seed)
        cases_by_target[target] += 1

    design_findings = []
    if len(target_speakers) < minimum_target_speakers:
        design_findings.append("insufficient_target_speaker_count")
    insufficient_impostors = sorted(
        target
        for target in target_speakers
        if len(impostors_by_target[target]) < minimum_impostors_per_target
    )
    if insufficient_impostors:
        design_findings.append("insufficient_impostor_distribution")
    insufficient_seeds = sorted(
        target
        for target in target_speakers
        if len(seeds_by_target[target]) < minimum_seeds_per_target
    )
    if insufficient_seeds:
        design_findings.append("insufficient_seed_replication")
    design_gate = not design_findings

    reviewer_ids: set[str] = set()
    outcome_counts = {
        "correct": 0,
        "incorrect": 0,
        "abstain": 0,
        "invalid": 0,
        "unanswered": 0,
    }
    per_target: dict[str, dict[str, Any]] = {
        target: {
            "target_speaker_id": target,
            "case_count": cases_by_target[target],
            "unique_impostor_count": len(impostors_by_target[target]),
            "unique_seed_count": len(seeds_by_target[target]),
            "correct": 0,
            "incorrect": 0,
            "abstain": 0,
            "invalid": 0,
            "unanswered": 0,
        }
        for target in sorted(target_speakers)
    }
    naturalness: list[int] = []
    articulation: list[int] = []
    completed_reviewers = 0
    reviewer_summaries = []
    response_rows = []
    for review_index, review in enumerate(reviews):
        if review.get("schema") != REVIEW_SCHEMA:
            raise ValueError(f"Review schema must be {REVIEW_SCHEMA}")
        if review.get("review_id") != review_id:
            raise ValueError("Review/key review_id differs")
        reviewer_id = review.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError(f"review[{review_index}].reviewer_id must be non-empty text")
        reviewer_id = reviewer_id.strip()
        if reviewer_id in reviewer_ids:
            raise ValueError(f"duplicate reviewer_id: {reviewer_id}")
        reviewer_ids.add(reviewer_id)
        review_cases = _rows_by_id(
            review.get("reviews"),
            "case_id",
            context=f"review[{review_index}].reviews",
        )
        if set(review_cases) != set(key_cases):
            missing = sorted(set(key_cases) - set(review_cases))
            extra = sorted(set(review_cases) - set(key_cases))
            raise ValueError(f"Review/key case IDs differ; missing={missing}, extra={extra}")
        reviewer_counts = {name: 0 for name in outcome_counts}
        for blind_case_id, key_row in key_cases.items():
            row = review_cases[blind_case_id]
            case_id = str(key_row["case_id"])
            choice = row.get("identity_choice", "")
            if choice not in VALID_CHOICES:
                raise ValueError(
                    f"{reviewer_id}.{blind_case_id}.identity_choice must be A, B, unclear, invalid or empty"
                )
            confidence = _rating(
                row.get("confidence", 0),
                field=f"{reviewer_id}.{blind_case_id}.confidence",
            )
            natural = _rating(
                row.get("candidate_naturalness", 0),
                field=f"{reviewer_id}.{blind_case_id}.candidate_naturalness",
            )
            articulate = _rating(
                row.get("candidate_articulation", 0),
                field=f"{reviewer_id}.{blind_case_id}.candidate_articulation",
            )
            notes = row.get("notes", "")
            if not isinstance(notes, str):
                raise ValueError(f"{reviewer_id}.{blind_case_id}.notes must be text")
            if choice in {"A", "B", "unclear"} and confidence == 0:
                raise ValueError(
                    f"{reviewer_id}.{blind_case_id} answered identity but omitted confidence"
                )
            if choice == "":
                outcome = "unanswered"
            elif choice == "unclear":
                outcome = "abstain"
            elif choice == "invalid":
                outcome = "invalid"
            elif choice == key_row["target_code"]:
                outcome = "correct"
            else:
                outcome = "incorrect"
            outcome_counts[outcome] += 1
            reviewer_counts[outcome] += 1
            target = str(key_row["target_speaker_id"])
            per_target[target][outcome] += 1
            if natural:
                naturalness.append(natural)
            if articulate:
                articulation.append(articulate)
            response_rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "blind_case_id": blind_case_id,
                    "case_id": case_id,
                    "target_speaker_id": target,
                    "impostor_speaker_id": key_row["impostor_speaker_id"],
                    "condition_id": key_row["condition_id"],
                    "utterance_id": key_row["utterance_id"],
                    "language_code": key_row["language_code"],
                    "seed": key_row["seed"],
                    "seed_known": key_row.get("seed_known", True),
                    "identity_outcome": outcome,
                    "confidence": confidence,
                    "candidate_naturalness": natural,
                    "candidate_articulation": articulate,
                    "notes": notes,
                }
            )
        complete = reviewer_counts["unanswered"] == 0
        completed_reviewers += int(complete)
        reviewer_summaries.append(
            {
                "reviewer_id": reviewer_id,
                "complete": complete,
                "outcomes": reviewer_counts,
            }
        )

    total_responses = len(key_cases) * len(reviews)
    analyzable = outcome_counts["correct"] + outcome_counts["incorrect"]
    accuracy = outcome_counts["correct"] / analyzable if analyzable else 0.0
    wilson_lower = _wilson_lower(outcome_counts["correct"], analyzable)
    abstain_rate = (
        outcome_counts["abstain"] / total_responses if total_responses else 0.0
    )
    invalid_rate = (
        outcome_counts["invalid"] / total_responses if total_responses else 0.0
    )
    complete_response_rate = (
        1.0 - outcome_counts["unanswered"] / total_responses
        if total_responses
        else 0.0
    )
    panel_findings = list(design_findings)
    if completed_reviewers < minimum_independent_reviewers:
        panel_findings.append("insufficient_independent_reviewers")
    if complete_response_rate < 1.0:
        panel_findings.append("incomplete_reviews")
    if accuracy < minimum_accuracy:
        panel_findings.append("accuracy_below_threshold")
    if wilson_lower < minimum_wilson_lower:
        panel_findings.append("wilson_lower_bound_below_threshold")
    if abstain_rate > maximum_abstain_rate:
        panel_findings.append("abstain_rate_above_threshold")
    if invalid_rate > maximum_invalid_rate:
        panel_findings.append("invalid_rate_above_threshold")
    panel_gate_pass = not panel_findings

    for row in per_target.values():
        total = row["correct"] + row["incorrect"]
        row["identity_accuracy"] = row["correct"] / total if total else None
        row["wilson_95_lower"] = (
            _wilson_lower(row["correct"], total) if total else None
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "source_key_schema": KEY_SCHEMA,
        "source_review_schema": REVIEW_SCHEMA,
        "review_id": review_id,
        "case_count": len(key_cases),
        "reviewer_exports_analyzed": len(reviews),
        "completed_independent_reviewers": completed_reviewers,
        "experimental_design": {
            "unique_target_speakers": len(target_speakers),
            "minimum_target_speakers": minimum_target_speakers,
            "minimum_impostors_per_target": minimum_impostors_per_target,
            "minimum_seeds_per_target": minimum_seeds_per_target,
            "insufficient_impostor_targets": insufficient_impostors,
            "insufficient_seed_targets": insufficient_seeds,
            "design_gate_pass": design_gate,
            "findings": design_findings,
        },
        "identity_results": {
            "outcomes": outcome_counts,
            "analyzable_choice_count": analyzable,
            "accuracy": accuracy,
            "wilson_95_lower": wilson_lower,
            "abstain_rate": abstain_rate,
            "invalid_rate": invalid_rate,
            "complete_response_rate": complete_response_rate,
            "mean_candidate_naturalness": _mean_or_none(naturalness),
            "mean_candidate_articulation": _mean_or_none(articulation),
        },
        "thresholds": {
            "minimum_independent_reviewers": minimum_independent_reviewers,
            "minimum_accuracy": minimum_accuracy,
            "minimum_wilson_95_lower": minimum_wilson_lower,
            "maximum_abstain_rate": maximum_abstain_rate,
            "maximum_invalid_rate": maximum_invalid_rate,
        },
        "per_target_speaker": list(per_target.values()),
        "reviewer_summaries": reviewer_summaries,
        "responses": response_rows,
        "decision": {
            "identity_discrimination_panel_gate": (
                "PASS" if panel_gate_pass else "ABSTAIN"
            ),
            "findings": panel_findings,
            "high_fidelity_clone_claim": "NOT_ESTABLISHED",
            "generalization": (
                "A passing ABX panel supports only identity discrimination for the fixed "
                "speakers, impostors, utterances, seeds, generation conditions and reviewers. "
                "It does not prove high-fidelity cloning, naturalness, acting control, consent, "
                "safety or out-of-set generalization."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reveal and analyze one or more MiniMax H3 voice-clone ABX exports."
    )
    parser.add_argument(
        "--review",
        type=Path,
        action="append",
        default=[],
        help="Repeat for each independent export; omit to audit design only.",
    )
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-target-speakers", type=int, default=10)
    parser.add_argument("--minimum-impostors-per-target", type=int, default=3)
    parser.add_argument("--minimum-seeds-per-target", type=int, default=3)
    parser.add_argument("--minimum-reviewers", type=int, default=3)
    parser.add_argument("--minimum-accuracy", type=float, default=0.80)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.65)
    parser.add_argument("--maximum-abstain-rate", type=float, default=0.20)
    parser.add_argument("--maximum-invalid-rate", type=float, default=0.05)
    args = parser.parse_args()
    review_paths = [path.resolve(strict=True) for path in args.review]
    key_path = args.blind_key.resolve(strict=True)
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in review_paths]
    key = json.loads(key_path.read_text(encoding="utf-8"))
    result = analyze_reviews(
        reviews,
        key,
        minimum_target_speakers=args.minimum_target_speakers,
        minimum_impostors_per_target=args.minimum_impostors_per_target,
        minimum_seeds_per_target=args.minimum_seeds_per_target,
        minimum_independent_reviewers=args.minimum_reviewers,
        minimum_accuracy=args.minimum_accuracy,
        minimum_wilson_lower=args.minimum_wilson_lower,
        maximum_abstain_rate=args.maximum_abstain_rate,
        maximum_invalid_rate=args.maximum_invalid_rate,
    )
    result["source_files"] = {
        "reviews": [
            {"path": str(path), "sha256": _sha256(path)} for path in review_paths
        ],
        "blind_key": {"path": str(key_path), "sha256": _sha256(key_path)},
    }
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "cases": result["case_count"],
                "reviewers": result["reviewer_exports_analyzed"],
                "identity_results": result["identity_results"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["decision"]["identity_discrimination_panel_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
