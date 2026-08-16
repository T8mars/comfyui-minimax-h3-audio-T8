from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import build_face_refine_blind_review as blind_tool
from h3_audio_t8_pkg.tools.probe_face_refine_plan import _track_metrics


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
