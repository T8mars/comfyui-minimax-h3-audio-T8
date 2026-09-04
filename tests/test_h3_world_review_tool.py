from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_h3_world_pair_review.py"
SPEC = importlib.util.spec_from_file_location("build_h3_world_pair_review", TOOL)
assert SPEC is not None and SPEC.loader is not None
review_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_tool)

ANALYZER = Path(__file__).resolve().parents[1] / "tools" / "analyze_h3_world_pair_review.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "analyze_h3_world_pair_review", ANALYZER
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
review_analyzer = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(review_analyzer)


def test_build_review_randomizes_labels_without_leaking_mapping(monkeypatch, tmp_path):
    forward = tmp_path / "forward.mp4"
    still = tmp_path / "still.mp4"
    forward.write_bytes(b"forward")
    still.write_bytes(b"still")
    calls = []

    monkeypatch.setattr(
        review_tool,
        "_strict_decode",
        lambda path, ffmpeg: calls.append((path.name, ffmpeg)),
    )
    monkeypatch.setattr(
        review_tool,
        "_probe",
        lambda path, ffprobe: {
            "width": 832,
            "height": 480,
            "frame_count": 124,
            "fps": 24.0,
            "sample_rate": 32000,
            "channels": 2,
        },
    )
    monkeypatch.setattr(
        review_tool,
        "_motion_metrics",
        lambda path: {"decoded_frames": 124, "flow_magnitude_mean": 1.0},
    )
    monkeypatch.setattr(
        review_tool,
        "_audio_metrics",
        lambda path, ffmpeg: {"finite": True, "rms": 0.1},
    )
    output = tmp_path / "review"

    reveal = review_tool.build_review(
        forward=forward,
        still=still,
        output_dir=output,
        seed=7,
        ffmpeg="ffmpeg-test",
        ffprobe="ffprobe-test",
    )

    assert set(reveal["mapping"].values()) == {"forward", "still"}
    assert len(calls) == 2
    assert (output / "candidate_A.mp4").is_file()
    assert (output / "candidate_B.mp4").is_file()
    page = (output / "blind_review.html").read_text(encoding="utf-8")
    assert "candidate_A.mp4" in page and "candidate_B.mp4" in page
    assert str(reveal["mapping"]) not in page
    assert "reveal.json" in page
    assert reveal["screening_sha256"] in page
    assert 'id="watched"' in page
    assert "请完成四项评分后再导出" in page


def test_strict_decode_rejects_reported_decoder_errors(monkeypatch, tmp_path):
    path = tmp_path / "candidate.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr(
        review_tool,
        "_run",
        lambda args: SimpleNamespace(returncode=0, stdout="", stderr="decoder error"),
    )
    with pytest.raises(RuntimeError, match="strict decode reported errors"):
        review_tool._strict_decode(path, "ffmpeg")


def test_cli_status_does_not_reveal_candidate_mapping(monkeypatch, tmp_path, capsys):
    output = tmp_path / "review"
    monkeypatch.setattr(
        review_tool,
        "build_review",
        lambda **kwargs: {
            "mapping": {"A": "forward", "B": "still"},
            "screening_sha256": "A" * 64,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--forward",
            str(tmp_path / "forward.mp4"),
            "--still",
            str(tmp_path / "still.mp4"),
            "--output-dir",
            str(output),
        ],
    )

    assert review_tool.main() == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "status": "READY_FOR_BLIND_REVIEW",
        "screening_sha256": "A" * 64,
        "output_dir": str(output.resolve()),
    }
    assert "mapping" not in status


def _write_review_contract(tmp_path: Path):
    clip = {
        "strict_av_decode": True,
        "motion": {
            "decoded_frames": 124,
            "black_frame_count": 0,
            "frozen_pair_count_at_1e-5": 0,
        },
        "audio": {"finite": True, "clipped_sample_values": 0},
    }
    screening = {
        "schema": review_analyzer.SCREENING_SCHEMA,
        "clips": {"forward": clip, "still": clip},
        "mechanical_gate": "PASS",
    }
    screening_path = tmp_path / "screening.json"
    screening_path.write_text(json.dumps(screening) + "\n", encoding="utf-8")
    screening_sha256 = review_analyzer._sha256(screening_path)
    reveal = {
        "schema": review_analyzer.REVEAL_SCHEMA,
        "mapping": {"A": "forward", "B": "still"},
        "screening_sha256": screening_sha256,
    }
    reveal_path = tmp_path / "reveal.json"
    reveal_path.write_text(json.dumps(reveal) + "\n", encoding="utf-8")
    review = {
        "schema": review_analyzer.REVIEW_SCHEMA,
        "screening_sha256": screening_sha256,
        "watched_full_length": True,
        "votes": {
            "forward": "A",
            "stable": "yes",
            "visual": "tie",
            "audio": "both_ok",
        },
        "notes": "动作差异清楚",
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    return review_path, screening_path, reveal_path


def test_analyzer_hash_binds_complete_human_pass(tmp_path):
    review_path, screening_path, reveal_path = _write_review_contract(tmp_path)

    result = review_analyzer.analyze_review(
        review_path=review_path,
        screening_path=screening_path,
        reveal_path=reveal_path,
    )

    assert result["p3_fixed_material_gate"] == "PASS"
    assert result["promotion_allowed"] is True
    assert result["reasons"] == []
    assert result["forward_label_after_reveal"] == "A"


def test_analyzer_rejects_wrong_hash_and_fails_unusable_action(tmp_path):
    review_path, screening_path, reveal_path = _write_review_contract(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["screening_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        review_analyzer.analyze_review(
            review_path=review_path,
            screening_path=screening_path,
            reveal_path=reveal_path,
        )

    review["screening_sha256"] = review_analyzer._sha256(screening_path)
    review["votes"]["forward"] = "tie"
    review["votes"]["stable"] = "no"
    review["votes"]["audio"] = "problem"
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    result = review_analyzer.analyze_review(
        review_path=review_path,
        screening_path=screening_path,
        reveal_path=reveal_path,
    )
    assert result["p3_fixed_material_gate"] == "FAIL"
    assert result["promotion_allowed"] is False
    assert len(result["reasons"]) == 3
