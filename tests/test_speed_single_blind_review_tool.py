from __future__ import annotations

from pathlib import Path

from h3_audio_t8_pkg.tools.build_h3_speed_single_blind_review import (
    REVEAL_SCHEMA,
    REVIEW_SCHEMA,
    build_single_blind_review,
)


def test_single_speed_blind_review_randomizes_and_preserves_both_sources(tmp_path: Path):
    baseline = tmp_path / "baseline.mp4"
    speed = tmp_path / "speed.mp4"
    baseline.write_bytes(b"baseline-media")
    speed.write_bytes(b"speed-media")
    output = tmp_path / "blind"

    reveal = build_single_blind_review(
        baseline=baseline,
        speed=speed,
        output_dir=output,
        seed=7,
        title="calibrated test",
    )

    assert reveal["schema"] == REVEAL_SCHEMA
    assert set(reveal["mapping"].values()) == {"baseline", "speed"}
    for label, treatment in reveal["mapping"].items():
        assert (output / f"calibrated_t2va_{label}.mp4").read_bytes() == {
            "baseline": b"baseline-media",
            "speed": b"speed-media",
        }[treatment]
    page = (output / "blind_review.html").read_text(encoding="utf-8")
    assert REVIEW_SCHEMA in page
    assert "calibrated_t2va_A.mp4" in page
    assert "calibrated_t2va_B.mp4" in page
