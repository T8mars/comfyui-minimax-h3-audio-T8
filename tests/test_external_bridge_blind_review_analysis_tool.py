from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools.analyze_external_bridge_blind_review import (
    KEY_SCHEMA,
    OUTPUT_SCHEMA,
    REVIEW_SCHEMA,
    analyze_review,
    main,
)


def _digest(character: str) -> str:
    return character * 64


def _key() -> dict:
    return {
        "schema": KEY_SCHEMA,
        "review_id": "review-final",
        "pairs": [
            {
                "pair_id": "clipproj-i2va",
                "reference_metrics": ["first_frame", "identity"],
                "sides": [
                    {
                        "code": "A",
                        "arm": "candidate",
                        "method": "ClipProj 8B",
                        "source": "private-candidate.mp4",
                        "sha256": _digest("a"),
                    },
                    {
                        "code": "B",
                        "arm": "control",
                        "method": "Native 32B",
                        "source": "private-control.mp4",
                        "sha256": _digest("b"),
                    },
                ],
            },
            {
                "pair_id": "sol-t2va",
                "reference_metrics": [],
                "sides": [
                    {
                        "code": "A",
                        "arm": "control",
                        "method": "Dense",
                        "source": "private-dense.mp4",
                        "sha256": _digest("c"),
                    },
                    {
                        "code": "B",
                        "arm": "candidate",
                        "method": "Sol",
                        "source": "private-sol.mp4",
                        "sha256": _digest("d"),
                    },
                ],
            },
        ],
    }


def _review() -> dict:
    return {
        "schema": REVIEW_SCHEMA,
        "review_id": "review-final",
        "exported_at": "2026-08-23T00:00:00Z",
        "reviews": [
            {
                "pair_id": "clipproj-i2va",
                "overall": "A",
                "motion": "B",
                "audio": "tie",
                "prompt_adherence": "A",
                "stability": "A",
                "blocking_failure": "B",
                "notes": "control broke",
                "reference_metrics": {"first_frame": "A", "identity": "B"},
            },
            {
                "pair_id": "sol-t2va",
                "overall": "A",
                "motion": "A",
                "audio": "tie",
                "prompt_adherence": "A",
                "stability": "A",
                "blocking_failure": "none",
                "notes": "",
                "reference_metrics": {},
            },
        ],
    }


def test_external_bridge_analysis_reveals_exact_arms_without_private_paths():
    result = analyze_review(_review(), _key())

    assert result["schema"] == OUTPUT_SCHEMA
    assert result["preference_counts"]["overall"] == {
        "control": 1,
        "candidate": 1,
        "tie": 0,
    }
    assert result["blocking_failure_counts"]["control"] == 1
    assert result["reference_preference_counts"]["identity"]["control"] == 1
    assert result["pair_results"][0]["decision"].startswith("control_failed")
    assert result["pair_results"][1]["decision"].startswith("candidate_not_preferred")
    assert result["decision"]["candidate_quality_noninferiority"].startswith("failed")
    assert result["assessability_counts"]["assessable"] == 2
    assert result["legacy_assessability_defaulted_pairs"] == 2
    serialized = json.dumps(result)
    assert "private-candidate.mp4" not in serialized
    assert "private-control.mp4" not in serialized


def test_external_bridge_analysis_rejects_mismatched_or_invalid_exports():
    wrong_id = _review()
    wrong_id["review_id"] = "other"
    with pytest.raises(ValueError, match="review_id differs"):
        analyze_review(wrong_id, _key())

    missing_pair = _review()
    missing_pair["reviews"] = missing_pair["reviews"][:1]
    with pytest.raises(ValueError, match="pair IDs differ"):
        analyze_review(missing_pair, _key())

    bad_vote = _review()
    bad_vote["reviews"][0]["overall"] = "candidate"
    with pytest.raises(ValueError, match="must be A, B, tie"):
        analyze_review(bad_vote, _key())

    bad_assessability = _review()
    bad_assessability["reviews"][0]["assessability"] = "maybe"
    with pytest.raises(ValueError, match="assessability must be one of"):
        analyze_review(bad_assessability, _key())

    extra_metric = _review()
    extra_metric["reviews"][1]["reference_metrics"] = {"identity": "A"}
    with pytest.raises(ValueError, match="unexpected metrics"):
        analyze_review(extra_metric, _key())


def test_external_bridge_analysis_excludes_unassessable_pairs_from_votes():
    review = _review()
    review["reviews"][0]["assessability"] = "source_material_insufficient"
    review["reviews"][1]["assessability"] = "assessable"

    result = analyze_review(review, _key(), required_independent_reviewers=1)

    assert result["assessability_counts"] == {
        "assessable": 1,
        "playback_problem": 0,
        "source_material_insufficient": 1,
        "unsure": 0,
    }
    assert result["assessable_pair_count"] == 1
    assert result["excluded_unassessable_pair_count"] == 1
    assert result["preference_counts"]["overall"] == {
        "control": 1,
        "candidate": 0,
        "tie": 0,
    }
    assert result["pair_results"][0]["decision"] == (
        "abstain_source_material_insufficient"
    )
    assert result["pair_results"][0]["preferences_counted"] is False
    assert result["decision"]["promotion_panel_gate"] == (
        "not_met_unassessable_pairs"
    )


def test_external_bridge_analysis_uses_hash_bound_generalization_contract():
    key = _key()
    key["analysis_contract"] = {
        "generalization": "This result applies only to one fixed Creator AV pair."
    }
    result = analyze_review(_review(), key)
    assert result["decision"]["generalization"] == (
        "This result applies only to one fixed Creator AV pair."
    )

    key["analysis_contract"] = {"generalization": ""}
    with pytest.raises(ValueError, match="analysis_contract.generalization"):
        analyze_review(_review(), key)


def test_external_bridge_analysis_cli_preserves_input_hashes(
    monkeypatch, tmp_path
):
    review_path = tmp_path / "review.json"
    key_path = tmp_path / "key.json"
    output_path = tmp_path / "analysis.json"
    review_path.write_text(json.dumps(_review()), encoding="utf-8")
    key_path.write_text(json.dumps(_key()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_external_bridge_blind_review.py",
            "--review",
            str(review_path),
            "--blind-key",
            str(key_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["source_files"]["review_sha256"]
    assert result["source_files"]["blind_key_sha256"]
    assert result["pair_count"] == 2
