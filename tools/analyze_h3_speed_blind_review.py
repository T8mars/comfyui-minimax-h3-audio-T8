#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


REVIEW_SCHEMA = "minimax_h3_speed_blind_review_v1"
REVEAL_SCHEMA = "minimax_h3_speed_blind_reveal_v1"
OUTPUT_SCHEMA = "minimax_h3_speed_blind_review_analysis_v1"
PAIR_NAMES = ("t2va", "fl2va", "ref2va")
PAIR_METRICS = {
    "t2va": ("overall", "motion_detail", "audio"),
    "fl2va": ("overall", "motion_detail", "audio"),
    "ref2va": ("overall", "motion_detail", "audio", "reference_adherence"),
}
TREATMENTS = {"baseline", "speed"}


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


def _review_rows(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("reviews must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Every review row must be an object")
        name = row.get("name")
        if name not in PAIR_NAMES:
            raise ValueError(f"Unknown review pair: {name!r}")
        if name in result:
            raise ValueError(f"Duplicate review pair: {name}")
        result[name] = row
    if set(result) != set(PAIR_NAMES):
        raise ValueError(
            f"Review pairs must be exactly {list(PAIR_NAMES)}; got {sorted(result)}"
        )
    return result


def _reveal_rows(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(PAIR_NAMES):
        raise ValueError(f"Reveal pairs must be exactly {list(PAIR_NAMES)}")
    result: dict[str, dict[str, str]] = {}
    for name in PAIR_NAMES:
        mapping = value[name]
        if not isinstance(mapping, dict) or set(mapping) != {"A", "B"}:
            raise ValueError(f"{name} reveal must map exactly A and B")
        if set(mapping.values()) != TREATMENTS:
            raise ValueError(f"{name} reveal must contain baseline and speed exactly once")
        result[name] = {"A": mapping["A"], "B": mapping["B"]}
    return result


def _map_vote(value: Any, mapping: Mapping[str, str], *, field: str) -> str:
    # The exported HTML states that an omitted answer is a tie.
    if value in (None, "", "tie"):
        return "tie"
    if value not in {"A", "B"}:
        raise ValueError(f"{field} must be A, B, tie or omitted")
    return mapping[str(value)]


def analyze_review(
    review: Mapping[str, Any],
    reveal: Mapping[str, Any],
    *,
    explicit_speed_failures: Sequence[str] = (),
    reviewer_comment: str = "",
) -> dict[str, Any]:
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"Review schema must be {REVIEW_SCHEMA}")
    if reveal.get("schema") != REVEAL_SCHEMA:
        raise ValueError(f"Reveal schema must be {REVEAL_SCHEMA}")
    rows = _review_rows(review.get("reviews"))
    mappings = _reveal_rows(reveal.get("pairs"))
    failures = set(explicit_speed_failures)
    if not failures.issubset(PAIR_NAMES):
        raise ValueError(f"Unknown explicit SPEED failure routes: {sorted(failures)}")
    if not isinstance(reviewer_comment, str):
        raise ValueError("reviewer_comment must be text")

    preference_counts = {
        metric: {"baseline": 0, "speed": 0, "tie": 0}
        for metric in ("overall", "motion_detail", "audio", "reference_adherence")
    }
    pair_results = []
    for name in PAIR_NAMES:
        mapped = {
            metric: _map_vote(
                rows[name].get(metric),
                mappings[name],
                field=f"{name}.{metric}",
            )
            for metric in PAIR_METRICS[name]
        }
        for metric, treatment in mapped.items():
            preference_counts[metric][treatment] += 1
        overall = mapped["overall"]
        if name in failures:
            route_decision = "speed_rejected_explicit_visible_failure"
        elif overall == "baseline":
            route_decision = "speed_rejected_fixed_profile_single_reviewer"
        elif overall == "speed":
            route_decision = "speed_preferred_fixed_profile_single_reviewer_only"
        else:
            route_decision = "inconclusive_tie"
        pair_results.append(
            {
                "name": name,
                "blind_mapping": mappings[name],
                "preferences_by_treatment": mapped,
                "explicit_visible_speed_failure": name in failures,
                "route_decision": route_decision,
            }
        )

    all_overall_baseline = all(
        row["preferences_by_treatment"]["overall"] == "baseline"
        for row in pair_results
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "source_review_schema": REVIEW_SCHEMA,
        "source_reveal_schema": REVEAL_SCHEMA,
        "reviewer_exports_analyzed": 1,
        "pair_count": len(PAIR_NAMES),
        "preference_counts": preference_counts,
        "pair_results": pair_results,
        "reviewer_comment": reviewer_comment,
        "decision": {
            "all_three_overall_preferred_baseline": all_overall_baseline,
            "speed_quality_noninferiority": (
                "failed_for_all_three_fixed_profiles"
                if all_overall_baseline
                else "not_established"
            ),
            "explicit_visible_speed_failure_routes": sorted(failures),
            "stable_default_or_auto_enable": "denied",
            "performance_speedup_evidence": "retained_exact_profiles_only",
            "memory_safe_claim": "denied",
            "generalization": (
                "The result rejects these exact SPEED schedules for this fixed material, seed, "
                "model and sampler. One reviewer and one case per route cannot establish a "
                "universal baseline-quality claim or identify the causal mechanism."
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reveal and summarize one MiniMax H3 SPEED blind review export."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--explicit-speed-failure",
        action="append",
        choices=PAIR_NAMES,
        default=[],
        help="Route where the reviewer explicitly observed a visibly broken SPEED result.",
    )
    parser.add_argument("--reviewer-comment", default="")
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    reveal = json.loads(args.reveal.read_text(encoding="utf-8"))
    result = analyze_review(
        review,
        reveal,
        explicit_speed_failures=args.explicit_speed_failure,
        reviewer_comment=args.reviewer_comment,
    )
    result["source_files"] = {
        "review": str(args.review.resolve()),
        "review_sha256": _sha256(args.review),
        "reveal": str(args.reveal.resolve()),
        "reveal_sha256": _sha256(args.reveal),
    }
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
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
