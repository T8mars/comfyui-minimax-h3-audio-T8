from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools.audit_face_refine_tracker_crossing import (
    build_crossing_scenario,
    summarize_crossing,
)
from h3_audio_t8_pkg.tools.analyze_face_refine_blind_review import analyze_review
from h3_audio_t8_pkg.tools import build_face_refine_blind_review as blind_tool
from h3_audio_t8_pkg.tools.evaluate_face_refine_yunet_wider import (
    _ratio_bin,
    _size_bin,
    match_detections,
    parse_wider_annotations,
)
from h3_audio_t8_pkg.tools.probe_face_refine_plan import _track_metrics
from h3_audio_t8_pkg.tools.summarize_face_refine_candidates import (
    expanded_face_roi,
    metric_summary,
)


def test_face_refine_track_probe_resets_motion_across_lost_frames():
    plan = {
        "source": {"width": 80, "height": 60},
        "frames": [
            {"state": "detected", "source_face_box_xyxy": [0, 0, 10, 10]},
            {"state": "detected", "source_face_box_xyxy": [3, 4, 13, 14]},
            {"state": "lost", "source_face_box_xyxy": [70, 50, 80, 60]},
            {"state": "reacquired", "source_face_box_xyxy": [60, 40, 80, 60]},
        ],
    }

    metrics = _track_metrics(plan)

    assert metrics["max_adjacent_center_jump_fraction_of_source_diagonal"] == pytest.approx(
        0.05
    )
    assert metrics["max_adjacent_log_area_change"] == pytest.approx(0.0)


def test_face_refine_blind_package_hides_mapping_and_preserves_media_hashes(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.mp4"
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source.write_bytes(b"source-video")
    first.write_bytes(b"candidate-one")
    second.write_bytes(b"candidate-two")
    contract = {
        "width": 736,
        "height": 416,
        "fps": 24.0,
        "frame_count": 124,
        "duration_seconds": 124 / 24,
    }
    monkeypatch.setattr(blind_tool, "_video_contract", lambda _path: dict(contract))

    output = tmp_path / "blind"
    result = blind_tool.build_package(source, [first, second], output, 260816)

    assert len(result["pairs"]) == 2
    assert {side["arm"] for pair in result["pairs"] for side in pair["sides"]} == {
        "source",
        "candidate",
    }
    html = (output / "blind_review.html").read_text(encoding="utf-8")
    assert str(source) not in html
    assert str(first) not in html
    key = json.loads((output / "blind_key.json").read_text(encoding="utf-8"))
    for pair in key["pairs"]:
        for side in pair["sides"]:
            copied = output / "media" / f"{pair['pair_id']}-{side['code']}.mp4"
            assert blind_tool._sha256_file(copied) == side["sha256"]


def test_face_refine_blind_package_rejects_contract_mismatch(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")

    def contract(path):
        return {
            "width": 736 if path.name == "source.mp4" else 704,
            "height": 416,
            "fps": 24.0,
            "frame_count": 124,
            "duration_seconds": 124 / 24,
        }

    monkeypatch.setattr(blind_tool, "_video_contract", contract)

    with pytest.raises(ValueError, match="differs from source contract at width"):
        blind_tool.build_package(source, [candidate], tmp_path / "blind", 1)


def test_face_refine_blind_analysis_reveals_arms_and_preserves_missing_values():
    key = {
        "schema": "t8.face_refine_blind_package.v1",
        "pairs": [
            {
                "pair_id": "pair-01",
                "sides": [
                    {"code": "A", "arm": "source", "source": "private-source"},
                    {"code": "B", "arm": "candidate", "source": "private-candidate"},
                ],
            },
            {
                "pair_id": "pair-02",
                "sides": [
                    {"code": "A", "arm": "candidate", "source": "other-private"},
                    {"code": "B", "arm": "source", "source": "private-source"},
                ],
            },
        ],
    }
    reviews = []
    for pair_id, source_code in (("pair-01", "A"), ("pair-02", "B")):
        candidate_code = "B" if source_code == "A" else "A"
        row = {
            "pair_id": pair_id,
            "overall_preference": source_code,
            "identity_preference": source_code,
            "motion_preference": "tie",
            "failure_notes": "candidate unstable",
        }
        for dimension in (
            "identity",
            "expression_mouth",
            "temporal",
            "seam",
            "naturalness",
            "motion",
        ):
            row[f"{source_code}_{dimension}_1_to_5"] = "5"
            row[f"{candidate_code}_{dimension}_1_to_5"] = (
                "" if pair_id == "pair-02" and dimension == "seam" else "1"
            )
        reviews.append(row)
    review = {
        "schema": "t8.face_refine_blind_review.v1",
        "review_completed": True,
        "reviews": reviews,
    }

    result = analyze_review(review, key)

    assert result["preference_counts"]["overall"] == {
        "source": 2,
        "candidate": 0,
        "tie": 0,
        "missing": 0,
    }
    assert result["scores_by_dimension"]["identity"]["source"]["mean"] == 5
    assert result["scores_by_dimension"]["identity"]["candidate"]["mean"] == 1
    assert result["scores_by_dimension"]["seam"]["candidate"]["missing"] == 1
    assert result["all_scores_complete"] is False
    assert result["decision"]["current_candidate_screen"].startswith("rejected")
    assert "private-source" not in json.dumps(result)


def test_face_refine_blind_analysis_rejects_pair_or_score_contract_errors():
    key = {
        "schema": "t8.face_refine_blind_package.v1",
        "pairs": [
            {
                "pair_id": "pair-01",
                "sides": [
                    {"code": "A", "arm": "source"},
                    {"code": "B", "arm": "candidate"},
                ],
            }
        ],
    }
    mismatched = {
        "schema": "t8.face_refine_blind_review.v1",
        "reviews": [{"pair_id": "pair-02"}],
    }
    with pytest.raises(ValueError, match="pair IDs differ"):
        analyze_review(mismatched, key)

    bad_score = {
        "schema": "t8.face_refine_blind_review.v1",
        "reviews": [{"pair_id": "pair-01", "A_identity_1_to_5": "6"}],
    }
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        analyze_review(bad_score, key)


def test_wider_annotation_parser_handles_valid_invalid_and_zero_face_rows(tmp_path):
    annotation = tmp_path / "wider.txt"
    annotation.write_text(
        "event/one.jpg\n"
        "2\n"
        "1 2 10 20 0 0 0 0 0 0\n"
        "30 40 5 6 2 1 1 1 2 1\n"
        "event/empty.jpg\n"
        "0\n"
        "0 0 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )

    records = parse_wider_annotations(annotation)

    assert [record["relative_path"] for record in records] == [
        "event/one.jpg",
        "event/empty.jpg",
    ]
    assert records[0]["boxes"][0]["box"] == [1.0, 2.0, 11.0, 22.0]
    assert records[0]["boxes"][1]["invalid"] is True
    assert records[1]["boxes"] == []


def test_wider_matching_ignores_invalid_boxes_and_counts_unmatched_detection():
    annotations = [
        {"box": [0.0, 0.0, 10.0, 10.0], "invalid": False},
        {"box": [20.0, 0.0, 30.0, 10.0], "invalid": True},
    ]
    detections = [
        {"box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9},
        {"box": [20.0, 0.0, 30.0, 10.0], "confidence": 0.8},
        {"box": [40.0, 0.0, 50.0, 10.0], "confidence": 0.7},
    ]

    result = match_detections(detections, annotations, 0.5)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["ignored_detections"] == 1


def test_wider_face_size_bins_have_explicit_pixel_and_relative_boundaries():
    assert _size_bin(15, 100) == "tiny_lt16px"
    assert _size_bin(16, 100) == "small_16_31px"
    assert _size_bin(32, 100) == "medium_32_63px"
    assert _size_bin(64, 100) == "large_ge64px"
    assert _ratio_bin(19, 100, 1000, 1000) == "tiny_lt2pct"
    assert _ratio_bin(20, 100, 1000, 1000) == "small_2_to_5pct"
    assert _ratio_bin(50, 100, 1000, 1000) == "large_ge5pct"


def test_crossing_scenario_is_deterministic_and_identity_summary_detects_swap():
    nine_frames = 9
    detections, truth = build_crossing_scenario(
        "confidence_order_flips_after_overlap", frame_count=nine_frames
    )
    assert len(detections) == len(truth) == nine_frames
    selected = [frame[0]["box"] for frame in detections]
    summary = summarize_crossing(selected, ["detected"] * nine_frames, truth)

    assert summary["first_identity"] == "A"
    assert summary["final_identity"] == "B"
    assert summary["swap_detected"] is True


def test_crossing_scenario_rejects_unknown_or_even_frame_contract():
    with pytest.raises(ValueError, match="odd integer"):
        build_crossing_scenario("stable_a_first", frame_count=10)
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_crossing_scenario("not-a-scenario")


def test_candidate_proxy_summary_ignores_nonfinite_values_and_reports_percentiles():
    result = metric_summary([1.0, 2.0, float("nan"), float("inf"), 3.0])

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["median"] == pytest.approx(2.0)
    assert result["p05"] == pytest.approx(1.1)
    assert result["p95"] == pytest.approx(2.9)


def test_candidate_face_roi_expands_and_clamps_without_silent_tiny_crop():
    assert expanded_face_roi([0.0, 0.0, 20.0, 20.0], 100, 80, 1.5) == (
        0,
        0,
        25,
        25,
    )
    with pytest.raises(ValueError, match="too small"):
        expanded_face_roi([1.0, 1.0, 2.0, 2.0], 100, 80, 1.0)
