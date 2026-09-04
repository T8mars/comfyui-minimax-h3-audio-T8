#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from . import build_dlss_nr_blind_review as blind_tool
except ImportError:  # pragma: no cover - direct script execution
    import build_dlss_nr_blind_review as blind_tool


KEY_SCHEMA = blind_tool.PACKAGE_SCHEMA
REVIEW_SCHEMA = blind_tool.REVIEW_SCHEMA
SCREENING_SCHEMA = blind_tool.SCREENING_SCHEMA
OUTPUT_SCHEMA = "t8.dlss_nr.blind_review_analysis.v1"
METRICS = {
    "overall",
    "face_identity_skin",
    "mouth_lipsync",
    "text_fine_texture",
    "color",
    "temporal_stability",
}
REGRESSION_FIELDS = {
    "face_identity_skin",
    "mouth_lipsync",
    "text_fine_texture",
    "color",
    "temporal_stability",
    "blocking_failure",
}
ASSESSABILITY = {
    "pending",
    "assessable",
    "source_insufficient",
    "playback_problem",
    "unsure",
}
CHOICES = {"pending", "tie", "A", "B", "C", "D", "unsure"}


def _rows_by_id(rows: Any, *, field: str, id_field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{field} must be a non-empty list")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"every {field} row must be an object")
        identity = row.get(id_field)
        if not isinstance(identity, str) or not identity or identity in mapped:
            raise ValueError(f"every {field} row needs a unique non-empty {id_field}")
        mapped[identity] = row
    return mapped


def _side_map(clip: dict[str, Any]) -> dict[str, dict[str, str]]:
    sides = clip.get("sides")
    if not isinstance(sides, list) or len(sides) != 4:
        raise ValueError(f"{clip.get('public_clip_id')} key must have four sides")
    mapped = {}
    methods = set()
    for side in sides:
        if not isinstance(side, dict):
            raise ValueError("blind key side must be an object")
        code = side.get("code")
        method = side.get("method")
        profile = side.get("profile")
        digest = side.get("normalized_sha256")
        if code not in blind_tool.CODES or method not in blind_tool.METHODS:
            raise ValueError("blind key side has an invalid code or method")
        if profile != blind_tool.METHOD_PROFILES[method]:
            raise ValueError(f"blind key profile differs for {method}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError("blind key normalized_sha256 must be 64 hexadecimal characters")
        if code in mapped or method in methods:
            raise ValueError("blind key repeats a code or method")
        mapped[code] = {
            "method": method,
            "profile": profile,
            "normalized_sha256": digest.lower(),
        }
        methods.add(method)
    if set(mapped) != set(blind_tool.CODES) or methods != set(blind_tool.METHODS):
        raise ValueError("blind key must map A-D to all four methods exactly once")
    return mapped


def _screening_map(screening: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if screening.get("schema") != SCREENING_SCHEMA:
        raise ValueError(f"mechanical screening schema must be {SCREENING_SCHEMA}")
    if screening.get("quality_ranking") is not None:
        raise ValueError("mechanical screening must not contain a quality ranking")
    return _rows_by_id(screening.get("clips"), field="screening clips", id_field="clip_id")


def _mechanical_failures(
    key_clip: dict[str, Any], screening_clip: dict[str, Any], mapping: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    if screening_clip.get("clip_type") != key_clip.get("clip_type"):
        raise ValueError("mechanical screening/key clip type differs")
    sides = screening_clip.get("sides")
    if not isinstance(sides, list) or len(sides) != 4:
        raise ValueError("mechanical screening clip must have four sides")
    failures = {method: [] for method in blind_tool.METHODS}
    seen_codes = set()
    for side in sides:
        if not isinstance(side, dict):
            raise ValueError("mechanical screening side must be an object")
        code = side.get("code")
        method = side.get("method")
        if code not in mapping or method != mapping[code]["method"] or code in seen_codes:
            raise ValueError("mechanical screening mapping differs from blind key")
        seen_codes.add(code)
        normalized = side.get("normalized")
        screen = side.get("screen")
        if not isinstance(normalized, dict) or not isinstance(screen, dict):
            raise ValueError("mechanical screening side lacks normalized/screen reports")
        if normalized.get("sha256", "").lower() != mapping[code]["normalized_sha256"]:
            raise ValueError("mechanical screening normalized hash differs from blind key")
        if screen.get("quality_ranking") is not None:
            raise ValueError("per-side mechanical screen must not rank quality")
        if screen.get("black_regression_frames"):
            failures[method].append("black_regression")
        if screen.get("freeze_regression_frames"):
            failures[method].append("freeze_regression")
        cut = screen.get("hard_cut")
        if key_clip.get("clip_type") == "hard_cut":
            if not isinstance(cut, dict):
                failures[method].append("hard_cut_not_screened")
            else:
                required = (
                    "source_has_mechanical_hard_cut",
                    "candidate_preserves_cut_transition",
                    "post_cut_closer_to_current_source_than_previous_source",
                )
                if not all(bool(cut.get(field)) for field in required):
                    failures[method].append("hard_cut_history_failure")
    if seen_codes != set(blind_tool.CODES):
        raise ValueError("mechanical screening does not cover A-D exactly once")
    return failures


def _review_regressions(
    value: Any, mapping: dict[str, dict[str, str]], *, public_clip_id: str
) -> dict[str, list[str]]:
    if value is None:
        value = {}
    if not isinstance(value, dict) or not set(value).issubset(set(blind_tool.CODES)):
        raise ValueError(f"{public_clip_id}.regressions must be an A-D object")
    result = {method: [] for method in blind_tool.METHODS}
    for code in blind_tool.CODES:
        row = value.get(code, {})
        if not isinstance(row, dict) or not set(row).issubset(REGRESSION_FIELDS):
            raise ValueError(f"{public_clip_id}.regressions.{code} has unknown fields")
        for field, enabled in row.items():
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"{public_clip_id}.regressions.{code}.{field} must be boolean"
                )
            if enabled:
                result[mapping[code]["method"]].append(field)
    return result


def analyze_review(
    review: dict[str, Any], key: dict[str, Any], screening: dict[str, Any]
) -> dict[str, Any]:
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"review schema must be {REVIEW_SCHEMA}")
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError(f"blind key schema must be {KEY_SCHEMA}")
    review_id = key.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("blind key review_id must be non-empty")
    if review.get("review_id") != review_id or screening.get("review_id") != review_id:
        raise ValueError("review/key/screening review_id differs")
    key_by_public_id = _rows_by_id(
        key.get("clips"), field="key clips", id_field="public_clip_id"
    )
    review_by_id = _rows_by_id(
        review.get("reviews"), field="review clips", id_field="clip_id"
    )
    if set(review_by_id) != set(key_by_public_id):
        raise ValueError("review/key public clip IDs differ")
    screening_by_private_id = _screening_map(screening)
    if set(screening_by_private_id) != {
        clip.get("clip_id") for clip in key_by_public_id.values()
    }:
        raise ValueError("screening/key private clip IDs differ")

    method_summary = {
        method: {
            "best_votes": {metric: 0 for metric in METRICS},
            "human_regressions": [],
            "mechanical_failures": [],
        }
        for method in blind_tool.METHODS
    }
    clip_results = []
    complete = True
    all_assessable = True
    for public_id, key_clip in key_by_public_id.items():
        review_clip = review_by_id[public_id]
        mapping = _side_map(key_clip)
        assessability = review_clip.get("assessability", "pending")
        if assessability not in ASSESSABILITY:
            raise ValueError(f"{public_id}.assessability is invalid")
        assessable = assessability == "assessable"
        all_assessable &= assessable
        watched = review_clip.get("watched", {})
        if not isinstance(watched, dict) or not set(watched).issubset(
            set(blind_tool.CODES)
        ):
            raise ValueError(f"{public_id}.watched must be an A-D object")
        watched_all = all(watched.get(code) is True for code in blind_tool.CODES)
        metrics = review_clip.get("metrics", {})
        if not isinstance(metrics, dict) or set(metrics) != METRICS:
            raise ValueError(f"{public_id}.metrics must contain every required metric")
        metric_values = {}
        metrics_complete = True
        for field, value in metrics.items():
            if value not in CHOICES:
                raise ValueError(f"{public_id}.metrics.{field} is invalid")
            metrics_complete &= value not in {"pending", "unsure"}
            mapped = mapping[value]["method"] if value in mapping else value
            metric_values[field] = mapped
            if value in mapping and assessable:
                method_summary[mapped]["best_votes"][field] += 1
        human_regressions = _review_regressions(
            review_clip.get("regressions"), mapping, public_clip_id=public_id
        )
        mechanical_failures = _mechanical_failures(
            key_clip,
            screening_by_private_id[key_clip["clip_id"]],
            mapping,
        )
        for method in blind_tool.METHODS:
            method_summary[method]["human_regressions"].extend(
                f"{public_id}:{field}" for field in human_regressions[method]
            )
            method_summary[method]["mechanical_failures"].extend(
                f"{public_id}:{field}" for field in mechanical_failures[method]
            )
        row_complete = assessable and watched_all and metrics_complete
        complete &= row_complete
        notes = review_clip.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError(f"{public_id}.notes must be text")
        clip_results.append(
            {
                "public_clip_id": public_id,
                "private_clip_id": key_clip["clip_id"],
                "clip_type": key_clip["clip_type"],
                "revealed_mapping": {
                    code: {
                        "method": side["method"],
                        "profile": side["profile"],
                        "normalized_sha256": side["normalized_sha256"],
                    }
                    for code, side in mapping.items()
                },
                "assessability": assessability,
                "watched_all_four": watched_all,
                "metrics_complete": metrics_complete,
                "best_by_method": metric_values,
                "human_regressions_by_method": human_regressions,
                "mechanical_failures_by_method": mechanical_failures,
                "notes": notes,
                "review_complete": row_complete,
            }
        )
    all_mechanical_clean = all(
        not summary["mechanical_failures"] for summary in method_summary.values()
    )
    dlss_clean = not method_summary["dlss_nr"]["human_regressions"] and not method_summary[
        "dlss_nr"
    ]["mechanical_failures"]
    p4_passed = complete and all_assessable and all_mechanical_clean and dlss_clean
    return {
        "schema": OUTPUT_SCHEMA,
        "review_id": review_id,
        "source_review_schema": REVIEW_SCHEMA,
        "source_key_schema": KEY_SCHEMA,
        "source_screening_schema": SCREENING_SCHEMA,
        "clip_count": len(clip_results),
        "review_complete": complete,
        "all_clips_assessable": all_assessable,
        "all_methods_mechanically_clean": all_mechanical_clean,
        "method_summary": method_summary,
        "clip_results": clip_results,
        "decision": {
            "p4_fixed_material_gate": "PASS" if p4_passed else "NOT_MET",
            "dlss_human_and_mechanical_nonregression": dlss_clean,
            "remain_experimental": not p4_passed,
            "eligible_for_p5_release_decision": p4_passed,
            "automatic_promotion": False,
            "generalization": (
                "Only these hash-bound sources, four fixed profiles, normalized files and this "
                "review are covered. No universal quality, speed, memory or lip-sync claim follows."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reveal and enforce the DLSS-NR P4 full-view non-regression gate."
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--mechanical-screening", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    screening = json.loads(args.mechanical_screening.read_text(encoding="utf-8"))
    result = analyze_review(review, key, screening)
    result["source_files"] = {
        "review_sha256": blind_tool.validation_tool._sha256_file(args.review),
        "blind_key_sha256": blind_tool.validation_tool._sha256_file(args.blind_key),
        "mechanical_screening_sha256": blind_tool.validation_tool._sha256_file(
            args.mechanical_screening
        ),
    }
    blind_tool.validation_tool._write_json_atomic(args.output, result)
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
