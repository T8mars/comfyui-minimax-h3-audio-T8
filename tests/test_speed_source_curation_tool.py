from __future__ import annotations

import json

import numpy as np
import pytest

from h3_audio_t8_pkg.tools import curate_h3_speed_spectrum_sources as curator


def _probe_payload(**overrides):
    stream = {
        "codec_name": "h264",
        "width": 736,
        "height": 416,
        "avg_frame_rate": "24/1",
        "r_frame_rate": "24/1",
        "nb_frames": "124",
        "duration": "5.166667",
    }
    stream.update(overrides)
    return {"streams": [stream], "format": {"duration": "5.166667"}}


def _h3_prompt_payload(*, task="T2VA", seed=42):
    payload = _probe_payload()
    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "minimax_h3_fl2va_int8_convrot.safetensors"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "4": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {"task_type": task, "prompt": "private prompt text"},
        },
        "5": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {"steps": 20, "shift_video": 12.0, "shift_audio": 3.0},
        },
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    payload["format"]["tags"] = {
        "prompt": json.dumps(prompt),
        "workflow": json.dumps({"nodes": []}),
    }
    return payload


def test_probe_record_uses_fractional_rate_and_format_duration_fallback():
    payload = _probe_payload(
        avg_frame_rate="24000/1001", nb_frames="N/A", duration="N/A"
    )
    record = curator.probe_record_from_payload(payload)
    assert record["fps"] == pytest.approx(24000 / 1001)
    assert record["duration_seconds"] == pytest.approx(5.166667)
    assert record["reported_frame_count"] is None
    assert record["width"] == 736


def test_probe_record_fails_closed_without_video_contract():
    with pytest.raises(curator.CurationError, match="No video stream"):
        curator.probe_record_from_payload({"streams": []})
    with pytest.raises(curator.CurationError, match="dimensions"):
        curator.probe_record_from_payload(_probe_payload(width=0))
    with pytest.raises(curator.CurationError, match="frame rate"):
        curator.probe_record_from_payload(
            _probe_payload(avg_frame_rate="0/0", r_frame_rate="0/0")
        )


def test_center_cover_reports_crop_without_anisotropic_stretch():
    landscape = curator.center_cover_geometry(736, 416, 736, 416)
    portrait = curator.center_cover_geometry(416, 736, 736, 416)
    tiny = curator.center_cover_geometry(368, 208, 736, 416)
    assert landscape["retained_source_fraction"] == pytest.approx(1.0)
    assert portrait["retained_source_fraction"] == pytest.approx(0.31947, rel=1e-3)
    assert portrait["anisotropic_stretch"] is False
    assert tiny["requires_upscale"] is True


def test_embedded_h3_provenance_extracts_contract_without_prompt_text():
    result = curator.extract_embedded_h3_provenance(
        _h3_prompt_payload(task="Ref2VA", seed=123)
    )
    assert result["status"] == "parsed_h3_contract"
    assert result["contract"]["task_families"] == ["REF2VA"]
    assert result["contract"]["checkpoint_names"] == [
        "minimax_h3_fl2va_int8_convrot.safetensors"
    ]
    assert result["contract"]["video_vae_names"] == [
        "minimax_h3_video_vae_fp16.safetensors"
    ]
    assert result["seed_values"] == [123]
    assert len(result["conditioning_text_sha256"]) == 1
    assert result["source_asset_identifier_sha256"] == []
    assert len(result["content_signature_id"]) == 64
    serialized = json.dumps(result)
    assert "private prompt text" not in serialized
    assert len(result["contract_id"]) == 64


def test_embedded_provenance_separates_multikeyframe_generation_contract():
    payload = _h3_prompt_payload(task="FL2VA", seed=17)
    prompt = json.loads(payload["format"]["tags"]["prompt"])
    prompt["7"] = {
        "class_type": "MiniMaxH3MultiKeyframeT8Advanced",
        "inputs": {},
    }
    payload["format"]["tags"]["prompt"] = json.dumps(prompt)
    result = curator.extract_embedded_h3_provenance(payload)
    assert "MiniMaxH3MultiKeyframeT8Advanced" in result["contract"][
        "generation_modifier_classes"
    ]


def test_embedded_provenance_missing_and_malformed_are_not_claimed():
    assert curator.extract_embedded_h3_provenance(_probe_payload())["status"] == (
        "missing_prompt_tag"
    )
    malformed = _probe_payload()
    malformed["format"]["tags"] = {"prompt": "{bad"}
    assert curator.extract_embedded_h3_provenance(malformed)["status"] == (
        "malformed_prompt_tag"
    )


def test_provenance_contract_grouping_counts_statuses_without_raw_metadata():
    provenance = curator.extract_embedded_h3_provenance(_h3_prompt_payload())
    groups = curator.provenance_contract_groups(
        [
            {"path": "a.mp4", "status": "provisional_candidate", "embedded_provenance": provenance},
            {"path": "b.mp4", "status": "manual_review_required", "embedded_provenance": provenance},
        ]
    )
    assert len(groups) == 1
    assert groups[0]["total_files"] == 2
    assert groups[0]["provisional_candidate"] == 1
    assert groups[0]["manual_review_required"] == 1
    assert groups[0]["unique_content_signatures"] == 1
    assert groups[0]["unique_seed_values"] == 1


def test_suspect_name_is_flagged_but_not_automatically_rejected(tmp_path):
    path = tmp_path / "hanfu_three-way_blind_compare.mp4"
    flags = curator.suspect_name_flags(path)
    assert "blind" in flags
    assert "compare" in flags
    assert "three_way" in flags


def test_suspect_name_ignores_caller_selected_root_directory_name(tmp_path):
    root = tmp_path / "h3_speed_calibration"
    root.mkdir()
    path = root / "neutral_source.mp4"
    assert curator.suspect_name_flags(path, roots=[root]) == []


def test_temporal_hash_similarity_is_exact_and_monotonic():
    all_zero = (0, 0)
    one_bit = (0, 1)
    all_one = ((1 << 256) - 1, (1 << 256) - 1)
    assert curator.temporal_hash_similarity(all_zero, all_zero) == 1.0
    assert curator.temporal_hash_similarity(all_zero, one_bit) == pytest.approx(
        1.0 - 1.0 / 512.0
    )
    assert curator.temporal_hash_similarity(all_zero, all_one) == 0.0
    assert curator.temporal_hash_similarity((), ()) == 0.0


def test_average_hash_is_deterministic_for_decoded_grayscale_frame():
    frame = np.arange(256, dtype=np.uint8).reshape(16, 16)
    first = curator._average_hash(frame)
    second = curator._average_hash(frame.copy())
    assert first == second
    assert first.bit_count() == 128


def test_curator_rejects_exact_file_duplicate_and_requires_manual_review_for_name(
    tmp_path, monkeypatch
):
    first = tmp_path / "a.mp4"
    duplicate = tmp_path / "b.mp4"
    suspect = tmp_path / "blind_grid.mp4"
    first.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    suspect.write_bytes(b"different")

    monkeypatch.setattr(curator, "_probe_payload", lambda _path, _ffprobe: _probe_payload())
    report = curator.curate_sources(
        roots=[tmp_path],
        mode="metadata",
        ffprobe="ffprobe",
        ffmpeg="ffmpeg",
        target_width=736,
        target_height=416,
        required_frames=124,
        target_fps=24.0,
        decode_attempts=1,
        near_duplicate_threshold=0.98,
        reject_upscale=True,
    )
    assert report["counts"] == {
        "discovered": 3,
        "provisional_candidate": 1,
        "manual_review_required": 1,
        "rejected": 1,
    }
    rows = {item["path"]: item for item in report["items"]}
    assert rows[str(duplicate.resolve())]["rejection_reasons"] == [
        "exact_file_duplicate"
    ]
    assert rows[str(suspect.resolve())]["status"] == "manual_review_required"
    assert report["decision"]["formal_dataset_authorized"] is False


def test_require_embedded_provenance_demotes_metadata_free_candidate(
    tmp_path, monkeypatch
):
    path = tmp_path / "clean.mp4"
    path.write_bytes(b"unique")
    monkeypatch.setattr(curator, "_probe_payload", lambda _path, _ffprobe: _probe_payload())
    report = curator.curate_sources(
        roots=[tmp_path],
        mode="metadata",
        ffprobe="ffprobe",
        ffmpeg="ffmpeg",
        target_width=736,
        target_height=416,
        required_frames=124,
        target_fps=24.0,
        decode_attempts=1,
        near_duplicate_threshold=0.98,
        reject_upscale=True,
        require_embedded_provenance=True,
    )
    assert report["counts"]["manual_review_required"] == 1
    assert report["items"][0]["embedded_provenance"]["status"] == "missing_prompt_tag"


def test_atomic_json_writer_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "report.json"
    curator._write_json_atomic(output, {"finite": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_signature_mode_only_flags_hash_near_duplicates_for_manual_review(
    tmp_path, monkeypatch
):
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4", "c.mp4")]
    for index, path in enumerate(paths):
        path.write_bytes(f"file-{index}".encode())
    signatures = {
        "a.mp4": curator.DecodeSignature("RAW-A", (0, 0), 2),
        "b.mp4": curator.DecodeSignature("RAW-B", (0, 1), 2),
        "c.mp4": curator.DecodeSignature(
            "RAW-C", ((1 << 256) - 1, (1 << 256) - 1), 2
        ),
    }
    monkeypatch.setattr(curator, "_probe_payload", lambda _path, _ffprobe: _probe_payload())
    monkeypatch.setattr(
        curator,
        "strict_decode_window",
        lambda *_args, **_kwargs: {"passed": True, "trials": []},
    )
    monkeypatch.setattr(
        curator,
        "decode_signature",
        lambda path, **_kwargs: signatures[path.name],
    )
    report = curator.curate_sources(
        roots=[tmp_path],
        mode="signature",
        ffprobe="ffprobe",
        ffmpeg="ffmpeg",
        target_width=736,
        target_height=416,
        required_frames=124,
        target_fps=24.0,
        decode_attempts=1,
        near_duplicate_threshold=0.98,
        reject_upscale=True,
    )
    assert report["counts"] == {
        "discovered": 3,
        "provisional_candidate": 1,
        "manual_review_required": 2,
        "rejected": 0,
    }
    flagged = [row for row in report["items"] if row.get("near_duplicate_group")]
    assert len(flagged) == 2
    assert {row["near_duplicate_group"] for row in flagged} == {"near-0001"}
    assert report["decision"]["independence_validated"] is False
