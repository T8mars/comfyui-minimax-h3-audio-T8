from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import analyze_dlss_nr_p3_review as analysis


def _validation_report() -> dict:
    runs = []
    for index, clip_id in enumerate(("speech", "hard_cut", "fine_texture"), 1):
        runs.append(
            {
                "clip_id": clip_id,
                "mechanical_pass": True,
                "source": {"sha256": f"{index:064x}"},
                "candidate": {"sha256": f"{index + 10:064x}"},
            }
        )
    return {
        "schema": analysis.VALIDATION_SCHEMA,
        "p3": {
            "status": "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED",
            "speech_phrase_operator_confirmation": analysis.validation_tool.SPEECH_PHRASE,
            "runs_are_strictly_serial": True,
            "stress_or_parallel_generation": False,
            "runs": runs,
        },
    }


def _review() -> dict:
    rows = []
    for clip_id in ("speech", "hard_cut", "fine_texture"):
        row = {
            "clip_id": clip_id,
            **{field: "not_applicable" for field in analysis.FIELDS},
            "notes": "",
        }
        for field in analysis.REQUIRED_BY_CLIP[clip_id]:
            row[field] = "pass"
        rows.append(row)
    return {"schema": analysis.REVIEW_SCHEMA, "reviews": rows}


def test_complete_required_normal_speed_review_passes_only_fixed_p3_gate():
    result = analysis.analyze_review(_review(), _validation_report())

    assert result["schema"] == analysis.OUTPUT_SCHEMA
    assert result["review_complete"] is True
    assert result["all_mechanical_pass"] is True
    assert result["all_required_human_checks_pass"] is True
    assert result["decision"]["p3_fixed_material_gate"] == "PASS"
    assert result["decision"]["eligible_to_build_p4_comparison"] is True
    assert result["decision"]["automatic_quality_claim"] is False
    assert result["decision"]["automatic_promotion"] is False


def test_mouth_failure_or_pending_value_keeps_p3_not_met():
    review = _review()
    review["reviews"][0]["mouth_lipsync"] = "fail"
    result = analysis.analyze_review(review, _validation_report())
    assert result["decision"]["p3_fixed_material_gate"] == "NOT_MET"
    assert result["clips"][0]["fixed_clip_pass"] is False

    review = _review()
    review["reviews"][2]["text_fine_texture"] = "pending"
    result = analysis.analyze_review(review, _validation_report())
    assert result["review_complete"] is False
    assert result["decision"]["eligible_to_build_p4_comparison"] is False


def test_mechanical_failure_cannot_be_overridden_by_human_votes():
    validation_report = _validation_report()
    validation_report["p3"]["runs"][1]["mechanical_pass"] = False
    result = analysis.analyze_review(_review(), validation_report)
    assert result["all_mechanical_pass"] is False
    assert result["decision"]["p3_fixed_material_gate"] == "NOT_MET"


def test_p3_analysis_rejects_missing_fields_or_wrong_clip_set():
    review = _review()
    del review["reviews"][0]["color"]
    with pytest.raises(ValueError, match="must contain exactly"):
        analysis.analyze_review(review, _validation_report())

    review = _review()
    review["reviews"][0]["clip_id"] = "other"
    with pytest.raises(ValueError, match="must contain speech"):
        analysis.analyze_review(review, _validation_report())


def test_p3_analysis_cli_hash_binds_review_and_validation(monkeypatch, tmp_path):
    review_path = tmp_path / "review.json"
    validation_path = tmp_path / "validation.json"
    output_path = tmp_path / "analysis.json"
    review_path.write_text(json.dumps(_review()), encoding="utf-8")
    validation_path.write_text(json.dumps(_validation_report()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_dlss_nr_p3_review.py",
            "--review",
            str(review_path),
            "--validation-report",
            str(validation_path),
            "--output",
            str(output_path),
        ],
    )
    assert analysis.main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["decision"]["p3_fixed_material_gate"] == "PASS"
    assert set(result["source_files"]) == {
        "review_sha256",
        "validation_report_sha256",
    }
