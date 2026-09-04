from __future__ import annotations

from pathlib import Path

import pytest

from h3_audio_t8_pkg.tools import prepare_dlss_nr_p4_candidates as candidates


def _gate_documents(tmp_path: Path):
    runs = []
    clips = []
    for clip_id in ("speech", "hard_cut", "fine_texture"):
        source = tmp_path / f"{clip_id}-source.mp4"
        candidate = tmp_path / f"{clip_id}-candidate.mp4"
        source.write_bytes(f"source-{clip_id}".encode())
        candidate.write_bytes(f"candidate-{clip_id}".encode())
        source_sha = candidates._sha256(source)
        candidate_sha = candidates._sha256(candidate)
        runs.append(
            {
                "clip_id": clip_id,
                "mechanical_pass": True,
                "source": {"path": str(source), "sha256": source_sha},
                "candidate": {"path": str(candidate), "sha256": candidate_sha},
            }
        )
        clips.append(
            {
                "clip_id": clip_id,
                "source_sha256": source_sha,
                "candidate_sha256": candidate_sha,
                "fixed_clip_pass": True,
            }
        )
    validation = {
        "schema": candidates.VALIDATION_SCHEMA,
        "p3": {"runs": runs},
    }
    analysis = {
        "schema": candidates.P3_ANALYSIS_SCHEMA,
        "decision": {
            "p3_fixed_material_gate": "PASS",
            "eligible_to_build_p4_comparison": True,
        },
        "clips": clips,
    }
    return validation, analysis


def test_validated_inputs_require_and_preserve_all_three_passed_clips(tmp_path):
    validation, analysis = _gate_documents(tmp_path)

    result = candidates._validated_inputs(validation, analysis)

    assert [row["clip_id"] for row in result] == ["speech", "hard_cut", "fine_texture"]
    assert all(row["source"].is_absolute() for row in result)
    assert all(row["dlss_nr"].is_absolute() for row in result)


def test_validated_inputs_reject_hash_drift(tmp_path):
    validation, analysis = _gate_documents(tmp_path)
    validation["p3"]["runs"][0]["source"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source hash differs"):
        candidates._validated_inputs(validation, analysis)


def test_validated_inputs_reject_unpassed_human_gate(tmp_path):
    validation, analysis = _gate_documents(tmp_path)
    analysis["decision"]["p3_fixed_material_gate"] = "NOT_MET"

    with pytest.raises(ValueError, match="P3 human gate has not passed"):
        candidates._validated_inputs(validation, analysis)


def test_bind_method_records_relative_path_and_hash(tmp_path):
    media = tmp_path / "speech" / "candidate.mp4"
    media.parent.mkdir()
    media.write_bytes(b"candidate")

    result = candidates._bind_method("a" * 64, media, "quality_locked", tmp_path)

    assert result == {
        "path": "speech/candidate.mp4",
        "source_sha256": "a" * 64,
        "candidate_sha256": candidates._sha256(media),
        "profile": "quality_locked",
    }


def test_resume_validation_fails_closed_on_bad_media(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(
        candidates,
        "_validate_2x",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad frame")),
    )

    assert candidates._is_valid_2x(source, candidate, ffmpeg="ffmpeg") is False
