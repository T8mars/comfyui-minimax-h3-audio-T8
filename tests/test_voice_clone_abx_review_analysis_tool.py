from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools.analyze_voice_clone_abx_review import (
    OUTPUT_SCHEMA,
    analyze_reviews,
    main,
)
from h3_audio_t8_pkg.tools.build_voice_clone_abx_review import (
    KEY_SCHEMA,
    REVIEW_SCHEMA,
)


def _key(target_count=10, impostors_per_target=3):
    cases = []
    for target_index in range(target_count):
        target = f"speaker-{target_index:02d}"
        for replicate in range(impostors_per_target):
            impostor = f"speaker-{(target_index + replicate + 1) % target_count:02d}"
            cases.append(
                {
                    "blind_case_id": f"case-{len(cases) + 1:03d}",
                    "case_id": f"{target}-case-{replicate}",
                    "target_speaker_id": target,
                    "impostor_speaker_id": impostor,
                    "condition_id": "clone-clean-zh",
                    "utterance_id": f"utterance-{replicate}",
                    "language_code": "zh",
                    "seed": 100 + replicate,
                    "seed_known": True,
                    "target_code": "A" if (target_index + replicate) % 2 == 0 else "B",
                    "impostor_code": "B" if (target_index + replicate) % 2 == 0 else "A",
                    "media": {},
                }
            )
    return {
        "schema": KEY_SCHEMA,
        "review_schema": REVIEW_SCHEMA,
        "review_id": "voice-abx-final",
        "cases": cases,
    }


def _review(key, reviewer_id, *, correct=True):
    rows = []
    for case in key["cases"]:
        choice = case["target_code"]
        if not correct:
            choice = case["impostor_code"]
        rows.append(
            {
                "case_id": case["blind_case_id"],
                "identity_choice": choice,
                "confidence": 4,
                "candidate_naturalness": 4,
                "candidate_articulation": 4,
                "notes": "",
            }
        )
    return {
        "schema": REVIEW_SCHEMA,
        "review_id": key["review_id"],
        "reviewer_id": reviewer_id,
        "exported_at": "2026-08-23T00:00:00Z",
        "reviews": rows,
    }


def test_abx_analysis_can_pass_identity_panel_without_claiming_high_fidelity():
    key = _key()
    reviews = [_review(key, f"reviewer-{index}") for index in range(3)]
    result = analyze_reviews(reviews, key)
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["experimental_design"]["design_gate_pass"] is True
    assert result["identity_results"]["accuracy"] == 1.0
    assert result["identity_results"]["complete_response_rate"] == 1.0
    assert result["decision"]["identity_discrimination_panel_gate"] == "PASS"
    assert result["decision"]["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    assert len(result["per_target_speaker"]) == 10


def test_abx_analysis_abstains_for_small_panel_or_bad_accuracy():
    key = _key(target_count=2, impostors_per_target=1)
    result = analyze_reviews(
        [_review(key, "reviewer-1", correct=False)],
        key,
        minimum_target_speakers=10,
        minimum_impostors_per_target=3,
        minimum_seeds_per_target=3,
        minimum_independent_reviewers=3,
    )
    assert result["decision"]["identity_discrimination_panel_gate"] == "ABSTAIN"
    assert "insufficient_target_speaker_count" in result["decision"]["findings"]
    assert "accuracy_below_threshold" in result["decision"]["findings"]
    assert result["decision"]["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"


def test_abx_analysis_rejects_legacy_one_impostor_unknown_seed_design():
    key = _key(target_count=10, impostors_per_target=1)
    for case in key["cases"]:
        case["seed_known"] = False
    reviews = [_review(key, f"reviewer-{index}") for index in range(3)]
    result = analyze_reviews(reviews, key)
    assert result["experimental_design"]["unique_target_speakers"] == 10
    assert result["experimental_design"]["design_gate_pass"] is False
    assert "insufficient_impostor_distribution" in result["decision"]["findings"]
    assert "insufficient_seed_replication" in result["decision"]["findings"]
    assert result["decision"]["identity_discrimination_panel_gate"] == "ABSTAIN"


def test_abx_analysis_supports_design_only_audit_without_fake_reviewers():
    key = _key()
    result = analyze_reviews([], key)
    assert result["experimental_design"]["design_gate_pass"] is True
    assert result["reviewer_exports_analyzed"] == 0
    assert result["identity_results"]["complete_response_rate"] == 0.0
    assert "insufficient_independent_reviewers" in result["decision"]["findings"]
    assert "incomplete_reviews" in result["decision"]["findings"]


def test_abx_analysis_preserves_unanswered_and_rejects_invalid_exports():
    key = _key()
    review = _review(key, "reviewer-1")
    review["reviews"][0]["identity_choice"] = ""
    review["reviews"][0]["confidence"] = 0
    result = analyze_reviews([review], key)
    assert result["identity_results"]["outcomes"]["unanswered"] == 1
    assert "incomplete_reviews" in result["decision"]["findings"]

    duplicate = _review(key, "reviewer-1")
    with pytest.raises(ValueError, match="duplicate reviewer_id"):
        analyze_reviews([duplicate, duplicate], key)

    wrong = _review(key, "reviewer-2")
    wrong["review_id"] = "other"
    with pytest.raises(ValueError, match="review_id differs"):
        analyze_reviews([wrong], key)

    bad_choice = _review(key, "reviewer-3")
    bad_choice["reviews"][0]["identity_choice"] = "target"
    with pytest.raises(ValueError, match="must be A, B, unclear"):
        analyze_reviews([bad_choice], key)


def test_abx_analysis_cli_hashes_every_input(monkeypatch, tmp_path):
    key = _key()
    key_path = tmp_path / "blind_key.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    review_paths = []
    for index in range(3):
        path = tmp_path / f"review-{index}.json"
        path.write_text(json.dumps(_review(key, f"reviewer-{index}")), encoding="utf-8")
        review_paths.append(path)
    output = tmp_path / "analysis.json"
    argv = ["analyze_voice_clone_abx_review.py"]
    for path in review_paths:
        argv.extend(["--review", str(path)])
    argv.extend(["--blind-key", str(key_path), "--output", str(output)])
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["source_files"]["reviews"]) == 3
    assert all(row["sha256"] for row in result["source_files"]["reviews"])
    assert result["source_files"]["blind_key"]["sha256"]
