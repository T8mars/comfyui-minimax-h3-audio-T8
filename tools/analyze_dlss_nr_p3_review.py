#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from . import run_dlss_nr_validation as validation_tool
except ImportError:  # pragma: no cover - direct script execution
    import run_dlss_nr_validation as validation_tool


REVIEW_SCHEMA = validation_tool.P3_REVIEW_SCHEMA
VALIDATION_SCHEMA = validation_tool.REPORT_SCHEMA
OUTPUT_SCHEMA = "t8.dlss_nr.p3_human_review_analysis.v1"
FIELDS = (
    "overall",
    "mouth_lipsync",
    "face_identity_skin",
    "text_fine_texture",
    "color",
    "temporal_stability",
    "cut_history",
    "audio",
)
REQUIRED_BY_CLIP = {
    "speech": {
        "overall",
        "mouth_lipsync",
        "face_identity_skin",
        "color",
        "temporal_stability",
        "audio",
    },
    "hard_cut": {"overall", "color", "temporal_stability", "cut_history"},
    "fine_texture": {"overall", "text_fine_texture", "color", "temporal_stability"},
}
VALUES = {"pending", "pass", "fail", "not_applicable", "unsure"}


def _unique_rows(rows: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"every {label} row must be an object")
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in result:
            raise ValueError(f"every {label} row needs a unique non-empty clip_id")
        result[clip_id] = row
    return result


def analyze_review(review: dict[str, Any], validation_report: dict[str, Any]) -> dict[str, Any]:
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"P3 review schema must be {REVIEW_SCHEMA}")
    if validation_report.get("schema") != VALIDATION_SCHEMA:
        raise ValueError(f"validation report schema must be {VALIDATION_SCHEMA}")
    p3 = validation_report.get("p3")
    if not isinstance(p3, dict) or p3.get("status") != "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED":
        raise ValueError("validation report does not contain a mechanically passed P3 run")
    if p3.get("speech_phrase_operator_confirmation") != validation_tool.SPEECH_PHRASE:
        raise ValueError("P3 validation report lacks the exact speech phrase confirmation")
    if p3.get("runs_are_strictly_serial") is not True or p3.get(
        "stress_or_parallel_generation"
    ) is not False:
        raise ValueError("P3 validation report does not prove strict serial execution")
    runs = _unique_rows(p3.get("runs"), label="P3 validation runs")
    reviews = _unique_rows(review.get("reviews"), label="P3 reviews")
    if set(runs) != set(REQUIRED_BY_CLIP) or set(reviews) != set(REQUIRED_BY_CLIP):
        raise ValueError("P3 review and validation must contain speech, hard_cut and fine_texture")

    clip_results = []
    review_complete = True
    all_required_pass = True
    all_mechanical_pass = True
    for clip_id in ("speech", "hard_cut", "fine_texture"):
        run = runs[clip_id]
        mechanical_pass = run.get("mechanical_pass") is True
        all_mechanical_pass &= mechanical_pass
        row = reviews[clip_id]
        expected_keys = {"clip_id", *FIELDS, "notes"}
        if set(row) != expected_keys:
            raise ValueError(
                f"P3 review {clip_id} must contain exactly {sorted(expected_keys)}"
            )
        values = {}
        for field in FIELDS:
            value = row[field]
            if value not in VALUES:
                raise ValueError(f"P3 review {clip_id}.{field} is invalid")
            values[field] = value
        notes = row["notes"]
        if not isinstance(notes, str):
            raise ValueError(f"P3 review {clip_id}.notes must be text")
        complete = all(value not in {"pending", "unsure"} for value in values.values())
        required_pass = all(values[field] == "pass" for field in REQUIRED_BY_CLIP[clip_id])
        any_failure = any(value == "fail" for value in values.values())
        row_pass = complete and required_pass and not any_failure and mechanical_pass
        review_complete &= complete
        all_required_pass &= row_pass
        clip_results.append(
            {
                "clip_id": clip_id,
                "source_sha256": run["source"]["sha256"],
                "candidate_sha256": run["candidate"]["sha256"],
                "mechanical_pass": mechanical_pass,
                "review_complete": complete,
                "required_fields": sorted(REQUIRED_BY_CLIP[clip_id]),
                "values": values,
                "notes": notes,
                "fixed_clip_pass": row_pass,
            }
        )
    p3_passed = review_complete and all_required_pass and all_mechanical_pass
    return {
        "schema": OUTPUT_SCHEMA,
        "source_review_schema": REVIEW_SCHEMA,
        "source_validation_schema": VALIDATION_SCHEMA,
        "clip_count": len(clip_results),
        "review_complete": review_complete,
        "all_mechanical_pass": all_mechanical_pass,
        "all_required_human_checks_pass": all_required_pass,
        "clips": clip_results,
        "decision": {
            "p3_fixed_material_gate": "PASS" if p3_passed else "NOT_MET",
            "eligible_to_build_p4_comparison": p3_passed,
            "automatic_quality_claim": False,
            "automatic_promotion": False,
            "generalization": (
                "Only the three hash-bound P3 clips are covered. No universal lip-sync, identity, "
                "quality, speed, memory or safety claim follows."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce the normal-speed human portion of the DLSS-NR P3 gate."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    result = analyze_review(review, validation_report)
    result["source_files"] = {
        "review_sha256": validation_tool._sha256_file(args.review),
        "validation_report_sha256": validation_tool._sha256_file(
            args.validation_report
        ),
    }
    validation_tool._write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "clips": result["clip_count"],
                "review_complete": result["review_complete"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
