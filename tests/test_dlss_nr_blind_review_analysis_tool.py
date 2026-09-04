from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import analyze_dlss_nr_blind_review as analysis


def _digest(index: int) -> str:
    return f"{index:064x}"


def _key() -> dict:
    clips = []
    types = ("speech", "hard_cut", "fine_texture")
    methods = list(analysis.blind_tool.METHODS)
    for clip_index, clip_type in enumerate(types):
        order = methods[clip_index:] + methods[:clip_index]
        clips.append(
            {
                "clip_id": f"private-{clip_index}",
                "public_clip_id": f"clip-{clip_index + 1:02d}",
                "clip_type": clip_type,
                "sides": [
                    {
                        "code": code,
                        "method": method,
                        "profile": analysis.blind_tool.METHOD_PROFILES[method],
                        "normalized_sha256": _digest(clip_index * 4 + side_index + 1),
                    }
                    for side_index, (code, method) in enumerate(
                        zip(analysis.blind_tool.CODES, order, strict=True)
                    )
                ],
            }
        )
    return {
        "schema": analysis.KEY_SCHEMA,
        "review_id": "p4-review",
        "clips": clips,
    }


def _review(key: dict) -> dict:
    return {
        "schema": analysis.REVIEW_SCHEMA,
        "review_id": key["review_id"],
        "exported_at": "2026-09-03T00:00:00Z",
        "reviews": [
            {
                "clip_id": clip["public_clip_id"],
                "assessability": "assessable",
                "watched": {code: True for code in analysis.blind_tool.CODES},
                "metrics": {metric: "tie" for metric in analysis.METRICS},
                "regressions": {},
                "notes": "",
            }
            for clip in key["clips"]
        ],
    }


def _screening(key: dict) -> dict:
    clips = []
    for clip in key["clips"]:
        sides = []
        for side in clip["sides"]:
            cut = None
            if clip["clip_type"] == "hard_cut":
                cut = {
                    "source_has_mechanical_hard_cut": True,
                    "candidate_preserves_cut_transition": True,
                    "post_cut_closer_to_current_source_than_previous_source": True,
                }
            sides.append(
                {
                    "code": side["code"],
                    "method": side["method"],
                    "normalized": {"sha256": side["normalized_sha256"]},
                    "screen": {
                        "quality_ranking": None,
                        "black_regression_frames": [],
                        "freeze_regression_frames": [],
                        "hard_cut": cut,
                    },
                }
            )
        clips.append(
            {
                "clip_id": clip["clip_id"],
                "clip_type": clip["clip_type"],
                "sides": sides,
            }
        )
    return {
        "schema": analysis.SCREENING_SCHEMA,
        "review_id": key["review_id"],
        "quality_ranking": None,
        "clips": clips,
    }


def _code_for(key: dict, public_id: str, method: str) -> str:
    clip = next(row for row in key["clips"] if row["public_clip_id"] == public_id)
    return next(side["code"] for side in clip["sides"] if side["method"] == method)


def test_complete_clean_review_closes_only_fixed_material_p4_gate():
    key = _key()
    result = analysis.analyze_review(_review(key), key, _screening(key))

    assert result["schema"] == analysis.OUTPUT_SCHEMA
    assert result["review_complete"] is True
    assert result["all_methods_mechanically_clean"] is True
    assert result["decision"]["p4_fixed_material_gate"] == "PASS"
    assert result["decision"]["eligible_for_p5_release_decision"] is True
    assert result["decision"]["automatic_promotion"] is False
    assert "No universal" in result["decision"]["generalization"]
    serialized = json.dumps(result)
    assert "source_candidate_path" not in serialized
    assert "normalized_path" not in serialized


def test_any_dlss_human_regression_keeps_feature_experimental():
    key = _key()
    review = _review(key)
    public_id = "clip-01"
    code = _code_for(key, public_id, "dlss_nr")
    review["reviews"][0]["regressions"] = {
        code: {"mouth_lipsync": True}
    }
    result = analysis.analyze_review(review, key, _screening(key))

    assert result["decision"]["p4_fixed_material_gate"] == "NOT_MET"
    assert result["decision"]["remain_experimental"] is True
    assert result["decision"]["dlss_human_and_mechanical_nonregression"] is False
    assert result["method_summary"]["dlss_nr"]["human_regressions"] == [
        "clip-01:mouth_lipsync"
    ]


def test_unwatched_or_unassessable_clip_cannot_pass_p4():
    key = _key()
    review = _review(key)
    review["reviews"][0]["watched"]["D"] = False
    review["reviews"][1]["assessability"] = "source_insufficient"
    result = analysis.analyze_review(review, key, _screening(key))

    assert result["review_complete"] is False
    assert result["all_clips_assessable"] is False
    assert result["decision"]["p4_fixed_material_gate"] == "NOT_MET"
    assert result["decision"]["eligible_for_p5_release_decision"] is False


def test_mechanical_failure_or_hash_mismatch_fails_closed():
    key = _key()
    screening = _screening(key)
    dlss_code = _code_for(key, "clip-01", "dlss_nr")
    side = next(
        row
        for row in screening["clips"][0]["sides"]
        if row["code"] == dlss_code
    )
    side["screen"]["freeze_regression_frames"] = [2]
    result = analysis.analyze_review(_review(key), key, screening)
    assert result["decision"]["p4_fixed_material_gate"] == "NOT_MET"
    assert result["method_summary"]["dlss_nr"]["mechanical_failures"] == [
        "clip-01:freeze_regression"
    ]

    screening = _screening(key)
    screening["clips"][0]["sides"][0]["normalized"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="normalized hash differs"):
        analysis.analyze_review(_review(key), key, screening)


def test_review_requires_every_metric_and_valid_boolean_regressions():
    key = _key()
    review = _review(key)
    del review["reviews"][0]["metrics"]["color"]
    with pytest.raises(ValueError, match="every required metric"):
        analysis.analyze_review(review, key, _screening(key))

    review = _review(key)
    review["reviews"][0]["regressions"] = {"A": {"color": "yes"}}
    with pytest.raises(ValueError, match="must be boolean"):
        analysis.analyze_review(review, key, _screening(key))


def test_analysis_cli_hash_binds_all_three_inputs(monkeypatch, tmp_path):
    key = _key()
    review = _review(key)
    screening = _screening(key)
    key_path = tmp_path / "blind_key.json"
    review_path = tmp_path / "review.json"
    screening_path = tmp_path / "mechanical_screening.json"
    output_path = tmp_path / "analysis.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    screening_path.write_text(json.dumps(screening), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_dlss_nr_blind_review.py",
            "--review",
            str(review_path),
            "--blind-key",
            str(key_path),
            "--mechanical-screening",
            str(screening_path),
            "--output",
            str(output_path),
        ],
    )

    assert analysis.main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["decision"]["p4_fixed_material_gate"] == "PASS"
    assert set(result["source_files"]) == {
        "review_sha256",
        "blind_key_sha256",
        "mechanical_screening_sha256",
    }
