"""Minimal native CPU pixel/packet delivery; no model or human-quality claim."""
import hashlib
import importlib
import json
from pathlib import Path

import comfy.utils
from comfy_api.latest import InputImpl
from comfy_execution.graph_utils import ExecutionBlocker
import pytest
import torch

from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg import postprocess_delivery as post
from h3_audio_t8_pkg.nodes_postprocess import MiniMaxH3PostprocessSaveEXPT8
from h3_audio_t8_pkg.video_outpaint_delivery import read_delivery_report


@pytest.fixture
def source(tmp_path):
    frames = torch.linspace(0, 1, 24 * 32 * 64 * 3).reshape(24, 32, 64, 3)
    t = torch.arange(32000, dtype=torch.float32) / 32000
    audio = {"sample_rate": 32000, "waveform": torch.stack([
        .03 * torch.sin(2 * torch.pi * 440 * t), .04 * torch.sin(2 * torch.pi * 660 * t)])[None]}
    master = tmp_path / "master.mp4"
    saved = save_h3_av_safe(frames, audio, master)
    return InputImpl.VideoFromFile(str(master)), frames, saved["output_sha256"]


def test_cold_core_crop_keeps_audio_packets_pcm_pts_and_durable_receipts(source, tmp_path, monkeypatch):
    video, _, original_sha = source
    core = Path(comfy.utils.__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(core))
    native = importlib.import_module("comfy_extras.nodes_images")
    # Cold path: use this saved file's decoded pixels, not an old in-memory AUDIO.
    frames = video.get_components().images
    region = native.BoundingBox.execute(0, 4, 64, 24).result[0]
    crop = native.ImageCropV2.execute(frames, region).result[0]
    target = tmp_path / "crop.mp4"
    result = post.finalize_postprocess(video, crop, target, expected_master_sha256=original_sha)
    assert result["state"] == "postprocess_complete", result
    assert post.read_postprocess_state(post.state_path(target)) == result
    receipt = read_delivery_report(target, report_kind="postprocess")
    evidence = receipt["media"]
    assert all(evidence[key] is True for key in (
        "audio_packet_payload_exact", "audio_packet_timeline_exact",
        "audio_decoded_pcm_exact", "audio_decoded_timeline_exact"))
    assert evidence["decoded_video_frames"] == 24 and evidence["cfr"]["first_pts"] == 0
    assert receipt["master"]["audio_packets"] == evidence["audio_packets"]
    assert receipt["master"]["audio_pcm"] == evidence["audio_pcm"]
    assert InputImpl.VideoFromFile(str(target)).get_dimensions() == (64, 24)
    assert hashlib.sha256(Path(video.get_stream_source()).read_bytes()).hexdigest() == original_sha
    assert not list(tmp_path.glob(".h3-postprocess-*"))
    assert receipt["audio_regenerated"] is False and receipt["automatic_accept"] is False
    # State/receipt alone must not authorize a missing or mismatched movie.
    target.unlink()
    with pytest.raises(FileNotFoundError):
        post.read_postprocess_state(post.state_path(target))


def test_encoder_failure_persists_failed_status_and_never_presents_master_as_enhanced(source, tmp_path, monkeypatch):
    video, frames, original_sha = source
    def fail(*args, **kwargs):
        raise RuntimeError("injected encoder failure")
    monkeypatch.setattr(post.delivery, "_encode_rgb_frames_isolated", fail)
    target = tmp_path / "failed.mp4"
    result = post.finalize_postprocess(video, frames[:, 4:28], target)
    assert result["state"] == "postprocess_failed"
    assert result["master_available"] is True and result["output_published"] is False
    assert "injected encoder failure" in result["error"]
    assert post.read_postprocess_state(post.state_path(target)) == result
    assert not target.exists() and not list(tmp_path.glob(".h3-postprocess-*"))
    assert hashlib.sha256(Path(video.get_stream_source()).read_bytes()).hexdigest() == original_sha
    monkeypatch.setattr("h3_audio_t8_pkg.nodes_postprocess.finalize_postprocess", lambda *a, **k: result)
    returned = MiniMaxH3PostprocessSaveEXPT8.execute(video, frames, confirm_postprocess=True)
    assert isinstance(returned.result[0], ExecutionBlocker)
    assert returned.result[1] is video and returned.result[2] == "postprocess_failed"
    assert returned.result[3] == "" and "images" not in returned.ui


def test_wrong_master_sha_fails_before_any_postprocess_write(source, tmp_path):
    video, frames, _ = source
    target = tmp_path / "wrong-master.mp4"
    with pytest.raises(ValueError, match="expected SHA256"):
        post.finalize_postprocess(video, frames, target, expected_master_sha256="0" * 64)
    assert not post.state_path(target).exists() and not target.exists()


def test_time_trim_is_failed_not_silently_retained_or_padded(source, tmp_path):
    video, frames, _ = source
    target = tmp_path / "wrong-frames.mp4"
    result = post.finalize_postprocess(video, frames[:12], target)
    assert result["state"] == "postprocess_failed" and "cannot trim" in result["error"]
    assert result["master_available"] is True and not target.exists()
    with pytest.raises(FileExistsError):
        post.finalize_postprocess(video, frames, target)
    state = json.loads(post.state_path(target).read_text())
    state["state"] = "postprocess_complete"
    post.state_path(target).write_text(json.dumps(state))
    with pytest.raises(ValueError, match="integrity"):
        post.read_postprocess_state(post.state_path(target))


def test_unconfirmed_node_writes_nothing_and_keeps_master_in_its_own_output(source):
    video, frames, _ = source
    result = MiniMaxH3PostprocessSaveEXPT8.execute(video, frames)
    assert isinstance(result.result[0], ExecutionBlocker)
    assert result.result[1] is video and result.result[2] == "not_started"
    assert "images" not in result.ui.as_dict()


def test_cancel_after_job_reservation_is_durable_but_not_swallowed(source, tmp_path, monkeypatch):
    video, frames, original_sha = source
    class TestInterrupt(BaseException):
        pass
    def cancel(*args, **kwargs):
        raise TestInterrupt("explicit cancellation")
    monkeypatch.setattr(post.delivery, "_encode_rgb_frames_isolated", cancel)
    target = tmp_path / "cancelled.mp4"
    with pytest.raises(TestInterrupt):
        post.finalize_postprocess(video, frames, target)
    state = post.read_postprocess_state(post.state_path(target))
    assert state["state"] == "postprocess_cancelled" and state["output_published"] is False
    assert state["master_available"] is True and not target.exists()
    assert hashlib.sha256(Path(video.get_stream_source()).read_bytes()).hexdigest() == original_sha


def test_nonzero_external_video_origin_is_not_silently_retimed(source, tmp_path, monkeypatch):
    video, frames, _ = source
    real_inspection = post.inspect_outpaint_source
    def changed_clock(*args, **kwargs):
        inspected = real_inspection(*args, **kwargs)
        inspected["cfr"]["first_pts"] = 512
        return inspected
    monkeypatch.setattr(post, "inspect_outpaint_source", changed_clock)
    target = tmp_path / "shifted.mp4"
    with pytest.raises(ValueError, match="no implicit PTS reset"):
        post.finalize_postprocess(video, frames, target)
    assert not target.exists() and not post.state_path(target).exists()
