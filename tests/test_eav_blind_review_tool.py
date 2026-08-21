from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import build_eav_blind_review as blind_tool


def _contract():
    return {
        "width": 1152,
        "height": 640,
        "fps": 24.0,
        "frame_count": 124,
        "video_duration_seconds": 124 / 24,
        "sample_rate": 32000,
        "channels": 2,
        "audio_duration_seconds": 5.152,
    }


def test_eav_blind_package_is_deterministic_private_and_hash_traceable(
    monkeypatch, tmp_path
):
    baseline = tmp_path / "reveals-baseline.mp4"
    apply = tmp_path / "reveals-apply-exp.mp4"
    baseline.write_bytes(b"baseline-media")
    apply.write_bytes(b"apply-media")
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "pairs": [
            {
                "pair_id": "i2va-stock20",
                "label": "第1组",
                "baseline": baseline.name,
                "apply_exp": apply.name,
                "reference_metric": True,
            }
        ],
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_key = blind_tool.build_package(manifest, tmp_path, first, 1234)
    second_key = blind_tool.build_package(manifest, tmp_path, second, 1234)
    assert first_key["pairs"][0]["sides"] == second_key["pairs"][0]["sides"]
    html = (first / "blind_review.html").read_text(encoding="utf-8")
    assert str(baseline) not in html
    assert str(apply) not in html
    assert "baseline" not in html
    assert "apply_exp" not in html
    assert blind_tool.REVIEW_SCHEMA in html
    assert "漏填按“平”导出" in html
    key = json.loads((first / "blind_key.json").read_text(encoding="utf-8"))
    for side in key["pairs"][0]["sides"]:
        copied = first / "media" / f"pair-01-{side['code']}.mp4"
        assert blind_tool._sha256_file(copied) == side["sha256"]


def test_eav_blind_package_rejects_media_contract_mismatch(monkeypatch, tmp_path):
    baseline = tmp_path / "baseline.mp4"
    apply = tmp_path / "apply.mp4"
    baseline.write_bytes(b"baseline")
    apply.write_bytes(b"apply")

    def contract(path):
        value = _contract()
        if path == apply:
            value["sample_rate"] = 44100
        return value

    monkeypatch.setattr(blind_tool, "_media_contract", contract)
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "pairs": [
            {
                "pair_id": "pair",
                "baseline": str(baseline),
                "apply_exp": str(apply),
            }
        ],
    }
    with pytest.raises(ValueError, match="differs at sample_rate"):
        blind_tool.build_package(manifest, tmp_path, tmp_path / "blind", 1)
