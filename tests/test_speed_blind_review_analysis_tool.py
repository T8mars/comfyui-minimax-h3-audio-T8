from __future__ import annotations

import pytest

from h3_audio_t8_pkg.tools.analyze_h3_speed_blind_review import analyze_review


def _review():
    return {
        "schema": "minimax_h3_speed_blind_review_v1",
        "reviews": [
            {"name": "t2va", "overall": "B", "motion_detail": "B", "audio": "B"},
            {"name": "fl2va", "overall": "B", "motion_detail": "B", "audio": "B"},
            {
                "name": "ref2va",
                "overall": "A",
                "motion_detail": "A",
                "audio": "A",
                "reference_adherence": "A",
            },
        ],
    }


def _reveal():
    return {
        "schema": "minimax_h3_speed_blind_reveal_v1",
        "pairs": {
            "t2va": {"A": "speed", "B": "baseline"},
            "fl2va": {"A": "speed", "B": "baseline"},
            "ref2va": {"A": "baseline", "B": "speed"},
        },
    }


def test_speed_blind_review_reveals_all_three_baseline_wins_and_visible_failures():
    result = analyze_review(
        _review(),
        _reveal(),
        explicit_speed_failures=("fl2va", "ref2va"),
        reviewer_comment="FL A and Ref B are visibly broken",
    )
    assert result["preference_counts"]["overall"] == {
        "baseline": 3,
        "speed": 0,
        "tie": 0,
    }
    assert result["decision"]["all_three_overall_preferred_baseline"] is True
    assert result["decision"]["speed_quality_noninferiority"] == (
        "failed_for_all_three_fixed_profiles"
    )
    assert result["decision"]["explicit_visible_speed_failure_routes"] == [
        "fl2va",
        "ref2va",
    ]
    assert result["decision"]["stable_default_or_auto_enable"] == "denied"
    assert result["pair_results"][1]["blind_mapping"]["A"] == "speed"
    assert result["pair_results"][2]["blind_mapping"]["B"] == "speed"


def test_missing_vote_is_tie_and_does_not_promote_speed():
    review = _review()
    review["reviews"][0].pop("audio")
    result = analyze_review(review, _reveal())
    assert result["preference_counts"]["audio"]["tie"] == 1
    assert result["decision"]["stable_default_or_auto_enable"] == "denied"


@pytest.mark.parametrize(
    ("review_mutator", "reveal_mutator", "message"),
    [
        (lambda value: value.update(schema="bad"), lambda _value: None, "Review schema"),
        (
            lambda _value: None,
            lambda value: value["pairs"]["t2va"].update(A="baseline"),
            "baseline and speed exactly once",
        ),
        (
            lambda value: value["reviews"][0].update(overall="C"),
            lambda _value: None,
            "must be A, B, tie",
        ),
    ],
)
def test_speed_blind_review_fails_closed_on_contract_errors(
    review_mutator, reveal_mutator, message
):
    review = _review()
    reveal = _reveal()
    review_mutator(review)
    reveal_mutator(reveal)
    with pytest.raises(ValueError, match=message):
        analyze_review(review, reveal)
