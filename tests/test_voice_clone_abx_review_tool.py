from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import build_voice_clone_abx_review as abx_tool


def _contract():
    return {
        "codec_name": "pcm_s16le",
        "sample_rate": 32000,
        "channels": 1,
        "format_name": "wav",
        "duration_seconds": 2.0,
        "bytes": 100,
    }


def _manifest(tmp_path):
    paths = {}
    for role, payload in (
        ("target_reference", b"target-reference"),
        ("impostor_reference", b"impostor-reference"),
        ("candidate", b"generated-candidate"),
    ):
        path = tmp_path / f"{role}.wav"
        path.write_bytes(payload)
        paths[role] = path.name
    return {
        "schema": abx_tool.MANIFEST_SCHEMA,
        "review_id": "voice-abx-final",
        "cases": [
            {
                "case_id": "speaker-01-seed-11",
                "target_speaker_id": "speaker-01",
                "impostor_speaker_id": "speaker-02",
                "condition_id": "clone-clean-zh",
                "utterance_id": "utterance-01",
                "language_code": "zh",
                "seed": 11,
                **paths,
            }
        ],
    }


def test_abx_builder_is_deterministic_blind_and_hash_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(abx_tool, "_audio_contract", lambda _path: _contract())
    manifest = _manifest(tmp_path)
    first = tmp_path / "blind-first"
    second = tmp_path / "blind-second"
    key = abx_tool.build_package(manifest, tmp_path, first, 260823)
    other = abx_tool.build_package(manifest, tmp_path, second, 260823)
    assert key["cases"][0]["target_code"] == other["cases"][0]["target_code"]
    html = (first / "blind_review.html").read_text(encoding="utf-8")
    assert abx_tool.REVIEW_SCHEMA in html
    assert "speaker-01" not in html
    assert "speaker-02" not in html
    assert str(tmp_path) not in html
    assert "高保真克隆" in html
    assert "未回答" in html
    case = key["cases"][0]
    assert {case["target_code"], case["impostor_code"]} == {"A", "B"}
    for code, row in case["media"].items():
        copied = first / row["blind_path"]
        assert copied.is_file()
        assert abx_tool._sha256_file(copied) == row["sha256"]
        assert code in {"A", "B", "X"}


def test_abx_builder_rejects_unfair_or_ambiguous_cases(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    def mismatch(path):
        contract = _contract()
        if path.name == "candidate.wav":
            contract["sample_rate"] = 24000
        return contract

    monkeypatch.setattr(abx_tool, "_audio_contract", mismatch)
    with pytest.raises(ValueError, match="contracts differ"):
        abx_tool.build_package(manifest, tmp_path, tmp_path / "mismatch", 1)

    monkeypatch.setattr(abx_tool, "_audio_contract", lambda _path: _contract())
    manifest["cases"][0]["impostor_speaker_id"] = "speaker-01"
    with pytest.raises(ValueError, match="must differ"):
        abx_tool.build_package(manifest, tmp_path, tmp_path / "same-speaker", 1)

    manifest = _manifest(tmp_path)
    manifest["cases"][0]["candidate"] = manifest["cases"][0]["target_reference"]
    with pytest.raises(ValueError, match="must be distinct"):
        abx_tool.build_package(manifest, tmp_path, tmp_path / "same-audio", 1)


def test_abx_builder_preserves_existing_private_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(abx_tool, "_audio_contract", lambda _path: _contract())
    manifest = _manifest(tmp_path)
    output = tmp_path / "blind"
    first = abx_tool.build_package(manifest, tmp_path, output, 7)
    original = (output / "blind_key.json").read_bytes()
    assert abx_tool.build_package(manifest, tmp_path, output, 7) == first
    assert (output / "blind_key.json").read_bytes() == original
    with pytest.raises(ValueError, match="different immutable key"):
        abx_tool.build_package(manifest, tmp_path, output, 8)
    assert (output / "blind_key.json").read_bytes() == original


def test_abx_builder_rejects_nonempty_unkeyed_output(monkeypatch, tmp_path):
    monkeypatch.setattr(abx_tool, "_audio_contract", lambda _path: _contract())
    output = tmp_path / "blind"
    output.mkdir()
    (output / "stale.html").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty but has no blind_key"):
        abx_tool.build_package(_manifest(tmp_path), tmp_path, output, 1)


def test_abx_builder_cli_writes_private_key(monkeypatch, tmp_path):
    monkeypatch.setattr(abx_tool, "_audio_contract", lambda _path: _contract())
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "blind"
    manifest_path.write_text(
        json.dumps(_manifest(tmp_path), ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_voice_clone_abx_review.py",
            str(manifest_path),
            "--output",
            str(output),
            "--random-seed",
            "42",
        ],
    )
    assert abx_tool.main() == 0
    assert json.loads((output / "blind_key.json").read_text(encoding="utf-8"))[
        "review_id"
    ] == "voice-abx-final"
