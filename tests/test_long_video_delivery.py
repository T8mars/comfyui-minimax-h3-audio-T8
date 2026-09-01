from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.long_video_delivery import (
    _manifest_lock,
    MANIFEST_FORMAT,
    MANIFEST_SCHEMA,
    UnsupportedManifestSchemaError,
    accept_long_video_candidate,
    apply_cosine_bridge,
    compose_accepted_long_video,
    load_accepted_context,
    load_delivery_manifest,
    save_long_video_candidate,
)
from helpers import make_audio


def _candidate(
    chain_id,
    index,
    start_frame,
    candidate_id,
    *,
    parent_id="",
    parent_revision=0,
    final=False,
    value=0.25,
):
    frame_count = 5
    frames = torch.full((frame_count, 128, 128, 3), value)
    # The candidate writer applies absolute sample boundaries. Supplying the local rounded
    # duration intentionally exercises its one-sample trim/pad accounting on later segments.
    audio = make_audio(frame_count / 24, sample_rate=32000, value=value, channels=2)
    latent, _ = empty_av_latent(128, 128, 124)
    return save_long_video_candidate(
        frames,
        audio,
        latent,
        chain_id,
        index,
        start_frame / 24,
        not final,
        parent_id,
        parent_revision,
        candidate_id,
        "test-model",
        "4-step-test",
        f"segment {index}",
        index + 100,
        24,
        8,
        28,
    )


def test_candidate_accept_context_and_streaming_compose(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))

    candidate0, video0, report0 = _candidate("delivery-chain", 0, 0, "take-a")
    assert Path(candidate0).is_file()
    assert Path(video0).is_file()
    assert json.loads(report0)["accepted"] is False
    assert json.loads(report0)["strict_decode_validated"] is True
    assert json.loads(report0)["strict_decode_policy"] == (
        "ffmpeg_xerror_threads1_before_atomic_publish_v1"
    )
    assert json.loads(report0)["video_encoder_process"] == "isolated_ffmpeg_subprocess"
    assert json.loads(report0)["video_encoder_policy"] == (
        "ffmpeg_rawvideo_pipe_libx264_all_intra_baseline_threads1_subprocess_v3"
    )
    assert json.loads(report0)["video_only_strict_decode_validated"] is True

    preview, accepted, manifest_path, accept_report = accept_long_video_candidate(
        candidate0, True
    )
    assert accepted is True
    assert Path(preview).is_file()
    assert Path(manifest_path).is_file()
    first_accept = json.loads(accept_report)
    assert first_accept["manifest_revision"] == 1

    context, has_context, parent_id, parent_revision, context_report = load_accepted_context(
        "delivery-chain", 1
    )
    assert has_context is True
    assert parent_id == "take-a"
    assert parent_revision == 1
    assert context["metadata"]["accepted_candidate_id"] == "take-a"
    assert json.loads(context_report)["checksums_valid"] is True

    candidate1, _, report1 = _candidate(
        "delivery-chain",
        1,
        5,
        "take-b",
        parent_id=parent_id,
        parent_revision=parent_revision,
        final=True,
        value=-0.25,
    )
    # round(5/24*32000) is 6667, while the absolute [5,10) boundary is 6666.
    assert json.loads(report1)["audio_adjustment"]["trimmed_samples"] == 1
    accept_long_video_candidate(candidate1, True)

    output_path, compose_report = compose_accepted_long_video(
        "delivery-chain", "test", True, "cosine_bridge", 5.0, 28
    )
    report = json.loads(compose_report)
    assert Path(output_path).is_file()
    assert report["segment_count"] == 2
    assert report["frame_count"] == 10
    assert report["audio_samples"] == round(10 / 24 * 32000)
    assert report["absolute_sample_accounting"] is True
    assert report["audio_encoder_process"] == "isolated_ffmpeg_subprocess"
    assert report["video_encoder_process"] == "isolated_ffmpeg_subprocess"
    assert report["strict_decode_validated"] is True
    assert report["seams"][0]["jump_after"] <= report["seams"][0]["jump_before"]

    with av.open(output_path) as container:
        assert sum(1 for _ in container.decode(video=0)) == 10
    with av.open(output_path) as container:
        assert container.streams.audio
        decoded_audio_samples = sum(frame.samples for frame in container.decode(audio=0))
        assert decoded_audio_samples >= round(10 / 24 * 32000)


def test_isolated_video_encoder_uses_decoder_safe_all_intra_baseline(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    captured = {}

    def capture(args, _log_path, _chunks_factory, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(delivery.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(delivery, "_run_isolated_ffmpeg_with_input", capture)
    delivery._encode_rgb_frames_isolated(
        tmp_path / "probe.mp4",
        lambda: iter([bytes(32 * 32 * 3)]),
        frame_count=1,
        width=32,
        height=32,
        fps=24,
        bit_depth=8,
        crf=18,
    )

    args = captured["args"]
    assert args[args.index("-profile:v") + 1] == "baseline"
    x264_params = args[args.index("-x264-params") + 1]
    for required in ("threads=1", "ref=1", "bframes=0", "keyint=1", "cabac=0"):
        assert required in x264_params
    assert captured["kwargs"]["expected_chunks"] == 1


def test_candidate_ffmpeg_failure_is_atomic_and_cleans_temporaries(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(delivery.shutil, "which", lambda _name: "ffmpeg")

    def fail_child(_args, _log_path, **_kwargs):
        raise RuntimeError("simulated isolated AAC encoder failure")

    monkeypatch.setattr(delivery, "_run_isolated_ffmpeg", fail_child)
    with pytest.raises(RuntimeError, match="simulated isolated AAC encoder failure"):
        _candidate("ffmpeg-failure-chain", 0, 0, "failed-take")

    candidate_dir = tmp_path / "minimax_h3_t8_long_video" / "ffmpeg-failure-chain" / "candidates"
    assert not list(candidate_dir.glob("*.mp4"))
    assert not list(candidate_dir.glob(".*.tmp"))


def test_candidate_strict_decode_failure_is_atomic_and_never_becomes_accepted(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))

    def fail_decode(_path, **_kwargs):
        raise RuntimeError("simulated damaged H.264 packet")

    monkeypatch.setattr(delivery, "_strict_validate_mp4", fail_decode)
    with pytest.raises(RuntimeError, match="damaged H.264"):
        _candidate("strict-decode-failure-chain", 0, 0, "failed-take")

    root = tmp_path / "minimax_h3_t8_long_video" / "strict-decode-failure-chain"
    assert not list(root.rglob("candidate.mp4"))
    assert not list(root.rglob("candidate.json"))
    assert not list(root.rglob(".*.tmp"))
    with pytest.raises(FileNotFoundError):
        load_delivery_manifest("strict-decode-failure-chain")


def test_composed_video_strict_decode_failure_is_atomic(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate0, _, _ = _candidate("compose-decode-failure", 0, 0, "take-a")
    accept_long_video_candidate(candidate0, True)
    candidate1, _, _ = _candidate(
        "compose-decode-failure",
        1,
        5,
        "take-b",
        parent_id="take-a",
        parent_revision=1,
        final=True,
    )
    accept_long_video_candidate(candidate1, True)

    def fail_decode(_path, **_kwargs):
        raise RuntimeError("simulated damaged composed H.264 packet")

    monkeypatch.setattr(delivery, "_strict_validate_mp4", fail_decode)
    with pytest.raises(RuntimeError, match="damaged composed H.264"):
        compose_accepted_long_video(
            "compose-decode-failure", "test", True, "none", 0.0, 28
        )

    assembled = (
        tmp_path / "minimax_h3_t8_long_video" / "compose-decode-failure" / "assembled"
    )
    assert not list(assembled.glob("*.mp4"))
    assert not list(assembled.glob(".*.tmp"))


def test_isolated_ffmpeg_retries_one_observed_native_crash(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    returncodes = iter([0xC0000005, 0])
    processes = []

    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def fake_popen(*_args, **_kwargs):
        process = FakeProcess(next(returncodes))
        processes.append(process)
        return process

    monkeypatch.setattr(delivery.subprocess, "Popen", fake_popen)
    delivery._run_isolated_ffmpeg(["ffmpeg", "-version"], tmp_path / "ffmpeg.log")
    assert [process.returncode for process in processes] == [0xC0000005, 0]


def test_isolated_ffmpeg_does_not_retry_regular_cli_failure(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    calls = 0

    class FakeProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    def fake_popen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FakeProcess()

    monkeypatch.setattr(delivery.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match=r"exit code 1 after 1 attempt"):
        delivery._run_isolated_ffmpeg(
            ["ffmpeg", "-invalid"], tmp_path / "ffmpeg.log"
        )
    assert calls == 1


def test_planar_audio_raw_is_interleaved_little_endian(tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    path = tmp_path / "audio.f32"
    delivery._write_planar_audio_raw(
        path,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert np.fromfile(path, dtype="<f4").tolist() == [1.0, 3.0, 2.0, 4.0]


def test_review_only_is_non_mutating_and_accept_is_idempotent(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, candidate_video, _ = _candidate("review-chain", 0, 0, "preview")
    preview, accepted, manifest, report = accept_long_video_candidate(candidate, False)
    assert preview == candidate_video
    assert accepted is False
    assert manifest == ""
    assert json.loads(report)["reason"].startswith("accept_candidate is false")

    first = accept_long_video_candidate(candidate, True)
    second = accept_long_video_candidate(candidate, True)
    assert first[1] is True and second[1] is True
    assert json.loads(second[3])["idempotent"] is True
    manifest_data, _ = load_delivery_manifest("review-chain")
    assert manifest_data["revision"] == 1
    assert len(manifest_data["segments"]) == 1


def test_idempotent_reaccept_repairs_missing_or_corrupt_accepted_assets(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, _, _ = _candidate("repair-chain", 0, 0, "repairable")
    accept_long_video_candidate(candidate, True)
    manifest, _ = load_delivery_manifest("repair-chain")
    root = tmp_path / "minimax_h3_t8_long_video" / "repair-chain"
    entry = manifest["segments"][0]
    accepted_video = root / entry["video_path"]
    accepted_context = root / entry["context_path"]
    accepted_video.unlink()
    accepted_context.write_bytes(b"corrupt accepted context")

    _, accepted, _, report_json = accept_long_video_candidate(candidate, True)

    report = json.loads(report_json)
    assert accepted is True
    assert report["idempotent"] is True
    assert report["repaired_assets"] == ["video", "context"]
    assert delivery._sha256_file(accepted_video) == entry["video_sha256"]
    assert delivery._sha256_file(accepted_context) == entry["context_sha256"]
    repaired_manifest, _ = load_delivery_manifest("repair-chain")
    assert repaired_manifest["revision"] == 1


def test_reusing_candidate_id_for_different_assets_cannot_overwrite_active_accept(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    original_candidate, _, _ = _candidate("identity-reuse", 0, 0, "same-id", value=0.1)
    accept_long_video_candidate(original_candidate, True)
    manifest, _ = load_delivery_manifest("identity-reuse")
    root = tmp_path / "minimax_h3_t8_long_video" / "identity-reuse"
    accepted_video = root / manifest["segments"][0]["video_path"]
    accepted_hash_before = delivery._sha256_file(accepted_video)

    # Candidate ids are normally immutable. Simulate a user deleting the reviewed candidate
    # folder and then accidentally reusing the id for different bytes.
    candidate_dir = Path(original_candidate).parent
    for child in candidate_dir.iterdir():
        child.unlink()
    candidate_dir.rmdir()
    replacement, _, _ = _candidate(
        "identity-reuse", 0, 0, "same-id", parent_revision=1, value=0.9
    )

    with pytest.raises(ValueError, match="already bound to different accepted assets"):
        accept_long_video_candidate(
            replacement, True, "replace_and_invalidate_following"
        )

    unchanged, _ = load_delivery_manifest("identity-reuse")
    assert unchanged["revision"] == 1
    assert unchanged["segments"][0]["candidate_id"] == "same-id"
    assert delivery._sha256_file(accepted_video) == accepted_hash_before


def test_candidate_id_is_normalized_before_identity_reuse_check(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    original_candidate, _, _ = _candidate(
        "token-collision", 0, 0, "same id", value=0.1
    )
    accept_long_video_candidate(original_candidate, True)
    manifest, _ = load_delivery_manifest("token-collision")
    root = tmp_path / "minimax_h3_t8_long_video" / "token-collision"
    accepted_video = root / manifest["segments"][0]["video_path"]
    accepted_hash_before = delivery._sha256_file(accepted_video)

    candidate_dir = Path(original_candidate).parent
    for child in candidate_dir.iterdir():
        child.unlink()
    candidate_dir.rmdir()
    # "same id" and "same_id" normalize to the same accepted filename token.
    replacement, _, _ = _candidate(
        "token-collision", 0, 0, "same_id", parent_revision=1, value=0.9
    )

    with pytest.raises(
        ValueError, match="already bound to different accepted assets for segment 0"
    ):
        accept_long_video_candidate(
            replacement, True, "replace_and_invalidate_following"
        )

    unchanged, _ = load_delivery_manifest("token-collision")
    assert unchanged["revision"] == 1
    assert unchanged["segments"][0]["candidate_id"] == "same_id"
    assert delivery._sha256_file(accepted_video) == accepted_hash_before


def test_reusing_invalidated_id_with_new_bytes_cannot_overwrite_archived_asset(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first_candidate, _, _ = _candidate("archived-collision", 0, 0, "take-a", value=0.1)
    accept_long_video_candidate(first_candidate, True)
    second_candidate, _, _ = _candidate(
        "archived-collision", 0, 0, "take-b", parent_revision=1, value=0.5
    )
    accept_long_video_candidate(
        second_candidate, True, "replace_and_invalidate_following"
    )
    root = tmp_path / "minimax_h3_t8_long_video" / "archived-collision"
    archived_video = root / "accepted" / "segment_00000_take-a.mp4"
    archived_hash = delivery._sha256_file(archived_video)

    candidate_dir = Path(first_candidate).parent
    for child in candidate_dir.iterdir():
        child.unlink()
    candidate_dir.rmdir()
    reused, _, _ = _candidate(
        "archived-collision", 0, 0, "take-a", parent_revision=2, value=0.9
    )

    with pytest.raises(ValueError, match="Accepted destination collision"):
        accept_long_video_candidate(reused, True, "replace_and_invalidate_following")

    unchanged, _ = load_delivery_manifest("archived-collision")
    assert unchanged["revision"] == 2
    assert unchanged["segments"][0]["candidate_id"] == "take-b"
    assert delivery._sha256_file(archived_video) == archived_hash


def test_context_copy_failure_leaves_no_manifest_and_retry_commits_same_candidate(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, _, _ = _candidate("copy-failure", 0, 0, "retry-me")
    original_copy = delivery._copy_atomic
    calls = 0

    def fail_context_copy(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected context copy failure")
        return original_copy(source, target)

    monkeypatch.setattr(delivery, "_copy_atomic", fail_context_copy)
    with pytest.raises(OSError, match="injected context copy failure"):
        accept_long_video_candidate(candidate, True)
    with pytest.raises(FileNotFoundError):
        load_delivery_manifest("copy-failure")
    root = tmp_path / "minimax_h3_t8_long_video" / "copy-failure"
    assert len(list((root / "accepted").glob("*.mp4"))) == 1
    assert len(list((root / "accepted").glob("*.context.safetensors"))) == 0

    monkeypatch.setattr(delivery, "_copy_atomic", original_copy)
    accept_long_video_candidate(candidate, True)
    manifest, _ = load_delivery_manifest("copy-failure")
    assert manifest["revision"] == 1
    assert manifest["segments"][0]["candidate_id"] == "retry-me"


def test_manifest_primary_write_failure_preserves_revision_and_retry_recovers(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("manifest-write-failure", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("manifest-write-failure", 1)
    second, _, _ = _candidate(
        "manifest-write-failure",
        1,
        5,
        "second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    original_write_bytes = delivery._atomic_write_bytes

    def fail_primary_write(path, data):
        if path.name == "manifest.json":
            raise OSError("injected manifest primary write failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(delivery, "_atomic_write_bytes", fail_primary_write)
    with pytest.raises(OSError, match="injected manifest primary write failure"):
        accept_long_video_candidate(second, True)

    preserved, source = load_delivery_manifest("manifest-write-failure")
    assert source == "primary"
    assert preserved["revision"] == 1
    assert [item["candidate_id"] for item in preserved["segments"]] == ["first"]
    root = tmp_path / "minimax_h3_t8_long_video" / "manifest-write-failure"
    backup = json.loads((root / "manifest.json.bak").read_text(encoding="utf-8"))
    assert backup["revision"] == 1
    assert len(list((root / "accepted").glob("segment_00001_*.mp4"))) == 1

    monkeypatch.setattr(delivery, "_atomic_write_bytes", original_write_bytes)
    accept_long_video_candidate(second, True)
    recovered, _ = load_delivery_manifest("manifest-write-failure")
    assert recovered["revision"] == 2
    assert [item["candidate_id"] for item in recovered["segments"]] == ["first", "second"]


def test_replacing_an_accepted_segment_invalidates_all_dependent_segments(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("replace-chain", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("replace-chain", 1)
    second, _, _ = _candidate(
        "replace-chain", 1, 5, "second", parent_id=parent_id,
        parent_revision=revision, final=True,
    )
    accept_long_video_candidate(second, True)

    _, _, _, current_revision, _ = load_accepted_context("replace-chain", 0)
    replacement, _, _ = _candidate(
        "replace-chain", 0, 0, "replacement",
        parent_revision=current_revision, value=0.8,
    )
    _, accepted, _, report = accept_long_video_candidate(
        replacement, True, "replace_and_invalidate_following"
    )
    assert accepted is True
    assert json.loads(report)["invalidated_segment_count"] == 2
    manifest, _ = load_delivery_manifest("replace-chain")
    assert [segment["candidate_id"] for segment in manifest["segments"]] == ["replacement"]
    assert {segment["candidate_id"] for segment in manifest["invalidated"]} == {
        "first", "second",
    }


def test_stale_parent_and_gapped_accept_are_rejected(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("stale-chain", 0, 0, "first")
    accept_long_video_candidate(first, True)
    stale, _, _ = _candidate(
        "stale-chain", 1, 5, "stale", parent_id="not-first", parent_revision=1, final=True
    )
    with pytest.raises(ValueError, match="stale or different accepted parent"):
        accept_long_video_candidate(stale, True)

    gap, _, _ = _candidate(
        "stale-chain", 2, 10, "gap", parent_id="future", parent_revision=1, final=True
    )
    with pytest.raises(ValueError, match="next slot"):
        accept_long_video_candidate(gap, True)


def test_stale_manifest_revision_is_rejected_even_when_parent_id_matches(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("revision-chain", 0, 0, "first")
    accept_long_video_candidate(first, True)
    stale, _, _ = _candidate(
        "revision-chain", 1, 5, "stale-revision",
        parent_id="first", parent_revision=0, final=True,
    )
    with pytest.raises(ValueError, match="stale manifest revision"):
        accept_long_video_candidate(stale, True)


def test_manifest_primary_corruption_falls_back_to_last_valid_backup(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("backup-chain", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("backup-chain", 1)
    second, _, _ = _candidate(
        "backup-chain", 1, 5, "second", parent_id=parent_id,
        parent_revision=revision, final=True,
    )
    accept_long_video_candidate(second, True)
    manifest, _ = load_delivery_manifest("backup-chain")
    manifest_path = Path(tmp_path) / "minimax_h3_t8_long_video" / "backup-chain" / "manifest.json"
    manifest_path.write_text("{broken", encoding="utf-8")

    recovered, source = load_delivery_manifest("backup-chain")
    assert manifest["revision"] == 2
    assert source == "backup"
    assert recovered["revision"] == 1
    assert [segment["candidate_id"] for segment in recovered["segments"]] == ["first"]


def test_missing_manifest_primary_recovers_backup_instead_of_starting_empty(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("missing-primary", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("missing-primary", 1)
    second, _, _ = _candidate(
        "missing-primary",
        1,
        5,
        "second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    accept_long_video_candidate(second, True)
    root = tmp_path / "minimax_h3_t8_long_video" / "missing-primary"
    (root / "manifest.json").unlink()

    recovered, source = load_delivery_manifest("missing-primary", allow_new=True)
    assert source == "backup"
    assert recovered["revision"] == 1
    assert [item["candidate_id"] for item in recovered["segments"]] == ["first"]

    # The already-rendered second candidate can recommit against the recovered revision. Its
    # byte-identical accepted assets are reused, not treated as an empty-chain segment 1 gap.
    accept_long_video_candidate(second, True)
    restored, source = load_delivery_manifest("missing-primary")
    assert source == "primary"
    assert restored["revision"] == 2
    assert [item["candidate_id"] for item in restored["segments"]] == ["first", "second"]


def test_missing_primary_with_unknown_backup_schema_never_creates_empty_chain(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    root = tmp_path / "minimax_h3_t8_long_video" / "missing-primary-future"
    root.mkdir(parents=True)
    (root / "manifest.json.bak").write_text(
        json.dumps({"schema": 999, "chain_id": "missing-primary-future"}),
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedManifestSchemaError, match="schema 999"):
        load_delivery_manifest("missing-primary-future", allow_new=True)
    assert not (root / "manifest.json").exists()


def test_semantically_invalid_primary_never_replaces_valid_backup_during_repair(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("semantic-backup", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("semantic-backup", 1)
    root = tmp_path / "minimax_h3_t8_long_video" / "semantic-backup"
    manifest_path = root / "manifest.json"
    backup_path = root / "manifest.json.bak"
    valid_revision_1 = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_path.write_text(json.dumps(valid_revision_1), encoding="utf-8")

    invalid_primary = dict(valid_revision_1)
    invalid_primary["segments"] = [dict(valid_revision_1["segments"][0])]
    invalid_primary["segments"][0].pop("video_sha256")
    manifest_path.write_text(json.dumps(invalid_primary), encoding="utf-8")

    second, _, _ = _candidate(
        "semantic-backup",
        1,
        5,
        "second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    accept_long_video_candidate(second, True)
    repaired, source = load_delivery_manifest("semantic-backup")
    assert source == "primary"
    assert repaired["revision"] == 2

    manifest_path.write_text("{broken", encoding="utf-8")
    recovered, source = load_delivery_manifest("semantic-backup")
    assert source == "backup"
    assert recovered["revision"] == 1
    assert recovered["segments"][0]["video_sha256"] == valid_revision_1["segments"][0]["video_sha256"]


def _wait_for_files(paths, processes, timeout_seconds=30.0):
    deadline = time.monotonic() + timeout_seconds
    while not all(path.is_file() for path in paths):
        failed = [process for process in processes if process.poll() not in {None, 0}]
        if failed:
            details = []
            for process in failed:
                stdout, stderr = process.communicate()
                details.append(f"exit={process.returncode}\nstdout={stdout}\nstderr={stderr}")
            raise AssertionError("Manifest worker exited before signalling ready:\n" + "\n".join(details))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for worker files: {paths}")
        time.sleep(0.02)


def _worker_command(*arguments):
    worker = Path(__file__).with_name("multiprocess_manifest_worker.py")
    return [sys.executable, str(worker), *map(str, arguments)]


def test_same_chain_multiprocess_acceptance_serializes_one_winner(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate_a, _, _ = _candidate("process-race", 0, 0, "process-a", value=0.1)
    candidate_b, _, _ = _candidate("process-race", 0, 0, "process-b", value=0.9)

    sync = tmp_path / "process-sync"
    sync.mkdir()
    start = sync / "start"
    ready = [sync / "ready-a", sync / "ready-b"]
    results = [sync / "result-a.json", sync / "result-b.json"]
    candidates = [candidate_a, candidate_b]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    processes = [
        subprocess.Popen(
            _worker_command(
                "accept",
                "--output-dir", tmp_path,
                "--candidate", candidates[index],
                "--start", start,
                "--ready", ready[index],
                "--result", results[index],
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creation_flags,
        )
        for index in range(2)
    ]
    try:
        _wait_for_files(ready, processes)
        start.write_text("go", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)

    outcomes = [json.loads(path.read_text(encoding="utf-8")) for path in results]
    winners = [outcome for outcome in outcomes if outcome["ok"]]
    losers = [outcome for outcome in outcomes if not outcome["ok"]]
    assert len(winners) == 1
    assert len(losers) == 1
    assert losers[0]["error_type"] == "ValueError"
    assert "already accepted" in losers[0]["error"]

    manifest, source = load_delivery_manifest("process-race")
    assert source == "primary"
    assert manifest["revision"] == 1
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0]["candidate_id"] in {"process-a", "process-b"}
    accepted_dir = tmp_path / "minimax_h3_t8_long_video" / "process-race" / "accepted"
    assert len(list(accepted_dir.glob("*.mp4"))) == 1
    assert len(list(accepted_dir.glob("*.context.safetensors"))) == 1


def test_manifest_os_lock_is_released_when_owner_process_is_terminated(tmp_path):
    root = tmp_path / "minimax_h3_t8_long_video" / "terminated-owner"
    ready = tmp_path / "terminated-owner.ready"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        _worker_command("hold", "--root", root, "--ready", ready, "--hold-seconds", 60),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    try:
        _wait_for_files([ready], [process])
        process.kill()
        process.wait(timeout=10)
        started = time.monotonic()
        with _manifest_lock(root, timeout_seconds=2.0):
            pass
        assert time.monotonic() - started < 2.0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_process_kill_after_video_copy_keeps_chain_unaccepted_and_retry_recovers(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, _, _ = _candidate("kill-after-video", 0, 0, "recover-copy")
    ready = tmp_path / "kill-after-video.ready"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        _worker_command(
            "accept-killpoint",
            "--output-dir",
            tmp_path,
            "--candidate",
            candidate,
            "--ready",
            ready,
            "--break-at",
            "after-video-copy",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    try:
        _wait_for_files([ready], [process])
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    root = tmp_path / "minimax_h3_t8_long_video" / "kill-after-video"
    with pytest.raises(FileNotFoundError):
        load_delivery_manifest("kill-after-video")
    assert len(list((root / "accepted").glob("*.mp4"))) == 1
    assert len(list((root / "accepted").glob("*.context.safetensors"))) == 0

    started = time.monotonic()
    accept_long_video_candidate(candidate, True)
    assert time.monotonic() - started < 2.0
    manifest, _ = load_delivery_manifest("kill-after-video")
    assert manifest["revision"] == 1
    entry = manifest["segments"][0]
    assert delivery._sha256_file(root / entry["video_path"]) == entry["video_sha256"]
    assert delivery._sha256_file(root / entry["context_path"]) == entry["context_sha256"]


def test_process_kill_after_backup_write_preserves_revision_and_retry_recovers(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("kill-after-backup", 0, 0, "first")
    accept_long_video_candidate(first, True)
    _, _, parent_id, revision, _ = load_accepted_context("kill-after-backup", 1)
    second, _, _ = _candidate(
        "kill-after-backup",
        1,
        5,
        "second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    ready = tmp_path / "kill-after-backup.ready"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        _worker_command(
            "accept-killpoint",
            "--output-dir",
            tmp_path,
            "--candidate",
            second,
            "--ready",
            ready,
            "--break-at",
            "after-backup-write",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    try:
        _wait_for_files([ready], [process])
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    root = tmp_path / "minimax_h3_t8_long_video" / "kill-after-backup"
    preserved, source = load_delivery_manifest("kill-after-backup")
    assert source == "primary"
    assert preserved["revision"] == 1
    backup = json.loads((root / "manifest.json.bak").read_text(encoding="utf-8"))
    assert backup["revision"] == 1
    assert len(list((root / "accepted").glob("segment_00001_*.mp4"))) == 1

    started = time.monotonic()
    accept_long_video_candidate(second, True)
    assert time.monotonic() - started < 2.0
    recovered, _ = load_delivery_manifest("kill-after-backup")
    assert recovered["revision"] == 2
    assert [entry["candidate_id"] for entry in recovered["segments"]] == ["first", "second"]


def test_dead_legacy_lock_residue_does_not_block_v2_but_is_kept_for_rollback(tmp_path):
    root = tmp_path / "legacy-lock"
    root.mkdir()
    legacy = root / "manifest.lock"
    legacy.write_text(
        json.dumps({"pid": 2_147_483_647, "created_unix": time.time()}),
        encoding="utf-8",
    )
    with _manifest_lock(root, timeout_seconds=1.0):
        pass
    assert legacy.is_file()


def test_live_legacy_lock_is_respected_during_rolling_upgrade(tmp_path):
    root = tmp_path / "live-legacy-lock"
    root.mkdir()
    legacy = root / "manifest.lock"
    legacy.write_text(
        json.dumps({"pid": os.getpid(), "created_unix": time.time() - 3600}),
        encoding="utf-8",
    )
    with pytest.raises(TimeoutError, match="manifest is busy"):
        with _manifest_lock(root, timeout_seconds=0.1):
            pass
    legacy.unlink()
    with _manifest_lock(root, timeout_seconds=1.0):
        pass


def test_manifest_lock_serializes_four_processes_across_repeated_commits(tmp_path):
    root = tmp_path / "multiprocess-counter-root"
    sync = tmp_path / "multiprocess-counter-sync"
    sync.mkdir()
    counter = sync / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    start = sync / "start"
    process_count = 4
    iterations = 25
    ready = [sync / f"ready-{index}" for index in range(process_count)]
    results = [sync / f"result-{index}.json" for index in range(process_count)]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    processes = [
        subprocess.Popen(
            _worker_command(
                "counter",
                "--root", root,
                "--counter", counter,
                "--start", start,
                "--ready", ready[index],
                "--result", results[index],
                "--iterations", iterations,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creation_flags,
        )
        for index in range(process_count)
    ]
    try:
        _wait_for_files(ready, processes)
        start.write_text("go", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)

    assert int(counter.read_text(encoding="utf-8")) == process_count * iterations
    assert all(json.loads(path.read_text(encoding="utf-8"))["ok"] for path in results)


def test_newer_manifest_schema_never_falls_back_to_older_backup(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, _, _ = _candidate("schema-downgrade", 0, 0, "accepted-v1")
    accept_long_video_candidate(candidate, True)
    root = tmp_path / "minimax_h3_t8_long_video" / "schema-downgrade"
    manifest_path = root / "manifest.json"
    backup_path = root / "manifest.json.bak"
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_path.write_text(json.dumps(legacy), encoding="utf-8")
    newer = dict(legacy)
    newer["schema"] = 999
    newer["revision"] = 2
    manifest_path.write_text(json.dumps(newer), encoding="utf-8")

    with pytest.raises(UnsupportedManifestSchemaError, match="schema 999"):
        load_delivery_manifest("schema-downgrade")


def test_same_schema_additive_metadata_remains_forward_compatible(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    candidate, _, _ = _candidate("schema-additive", 0, 0, "accepted-v1")
    accept_long_video_candidate(candidate, True)
    manifest_path = tmp_path / "minimax_h3_t8_long_video" / "schema-additive" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer_version"] = "future-additive-build"
    manifest["segments"][0]["optional_future_note"] = "preserve-me"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, source = load_delivery_manifest("schema-additive")
    assert source == "primary"
    assert loaded["producer_version"] == "future-additive-build"
    assert loaded["segments"][0]["optional_future_note"] == "preserve-me"

    _, _, parent_id, revision, _ = load_accepted_context("schema-additive", 1)
    second, _, _ = _candidate(
        "schema-additive",
        1,
        5,
        "accepted-v1-second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    accept_long_video_candidate(second, True)
    rewritten, source = load_delivery_manifest("schema-additive")
    assert source == "primary"
    assert rewritten["revision"] == 2
    assert rewritten["producer_version"] == "future-additive-build"
    assert rewritten["segments"][0]["optional_future_note"] == "preserve-me"


def test_legacy_v1_manifest_migrates_in_memory_then_atomically_writes_v2(
    monkeypatch, tmp_path
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(tmp_path))
    first, _, _ = _candidate("schema-v1-migration", 0, 0, "legacy-first")
    accept_long_video_candidate(first, True)
    root = tmp_path / "minimax_h3_t8_long_video" / "schema-v1-migration"
    manifest_path = root / "manifest.json"
    backup_path = root / "manifest.json.bak"
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy["schema"] = 1
    legacy.pop("format", None)
    legacy.pop("created_unix", None)
    legacy.pop("migrated_from_schema", None)
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated, source = load_delivery_manifest("schema-v1-migration")
    assert source == "primary"
    assert migrated["schema"] == MANIFEST_SCHEMA
    assert migrated["format"] == MANIFEST_FORMAT
    assert migrated["migrated_from_schema"] == 1
    assert migrated["revision"] == 1
    # A read alone is non-mutating; upgrade happens at the next protected manifest commit.
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema"] == 1

    _, _, parent_id, revision, _ = load_accepted_context("schema-v1-migration", 1)
    second, _, _ = _candidate(
        "schema-v1-migration",
        1,
        5,
        "v2-second",
        parent_id=parent_id,
        parent_revision=revision,
        final=True,
    )
    accept_long_video_candidate(second, True)

    primary_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_raw = json.loads(backup_path.read_text(encoding="utf-8"))
    assert primary_raw["schema"] == MANIFEST_SCHEMA
    assert primary_raw["format"] == MANIFEST_FORMAT
    assert primary_raw["migrated_from_schema"] == 1
    assert primary_raw["revision"] == 2
    assert backup_raw["schema"] == 1
    assert "format" not in backup_raw
    assert backup_raw["revision"] == 1

    manifest_path.write_text("{broken", encoding="utf-8")
    recovered, source = load_delivery_manifest("schema-v1-migration")
    assert source == "backup"
    assert recovered["schema"] == MANIFEST_SCHEMA
    assert recovered["migrated_from_schema"] == 1
    assert recovered["revision"] == 1


def test_cosine_bridge_preserves_sample_count_and_removes_value_step():
    previous = np.array([0.75, -0.5], dtype=np.float32)
    current = np.stack(
        [np.full(100, -0.25, dtype=np.float32), np.full(100, 0.25, dtype=np.float32)]
    )
    bridged, report = apply_cosine_bridge(previous, current, 16)
    assert bridged.shape == current.shape
    assert np.allclose(bridged[:, 0], previous)
    assert np.allclose(bridged[:, 16:], current[:, 16:])
    assert report["jump_before"] > 0
    assert report["jump_after"] == pytest.approx(0.0)
