import json
import shutil

import numpy as np
import pytest
import torch

from h3_audio_t8_pkg import h3_av_delivery as delivery
from h3_audio_t8_pkg import nodes_h3_av_delivery as nodes
from tools.vdn_probe_extension.media import save_raw_av


def media():
    frames = torch.linspace(0, 1, 24 * 32 * 64 * 3).reshape(24, 32, 64, 3)
    audio = {"sample_rate": 32000, "waveform": torch.zeros(1, 2, 32000)}
    return frames, audio


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_production_saver_matches_probe_raw_av_contract_and_returns_reusable_video(tmp_path, monkeypatch):
    images, sound = media()
    probe_report = save_raw_av(images, sound, tmp_path / "probe.mp4")
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes.folder_paths, "get_save_image_path",
                        lambda *a: (str(tmp_path), "node", 1, "", "node"))
    result = nodes.MiniMaxH3SafeAVSaveT8Advanced.execute(images, sound, "node")
    video, path, raw_report = result.result
    report = json.loads(raw_report)
    assert report["output"] == path and report["status"] == "pass"
    assert report["source_rgb8_sha256"] == probe_report["source_rgb8_sha256"]
    assert report["audio"] == probe_report["audio"]
    assert report["output_sha256"] == probe_report["output_sha256"]
    assert video.get_dimensions() == (64, 32)
    with pytest.raises(ValueError, match="new MP4"):
        nodes.MiniMaxH3SafeAVSaveT8Advanced.execute(images, sound, "node")


def test_failure_never_publishes_or_leaves_private_temporary_files(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr(delivery.delivery, "_encode_rgb_frames_isolated", fail)
    images, sound = media()
    with pytest.raises(RuntimeError, match="encoder failed"):
        delivery.save_h3_av_safe(images, sound, tmp_path / "result.mp4")
    assert list(tmp_path.iterdir()) == []


def test_node_rejects_output_directory_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path / "allowed"))
    monkeypatch.setattr(nodes.folder_paths, "get_save_image_path", lambda *a: (str(tmp_path), "escape", 1, "", ""))
    with pytest.raises(ValueError, match="inside ComfyUI"):
        nodes.MiniMaxH3SafeAVSaveT8Advanced.execute(*media(), "escape")
    assert not (tmp_path / "escape_00001_.mp4").exists()


@pytest.mark.parametrize("crf", [-1, 52, True, 18.5])
def test_invalid_crf_rejected(tmp_path, crf):
    with pytest.raises(ValueError, match="crf"):
        delivery.save_h3_av_safe(*media(), tmp_path / "invalid.mp4", crf=crf)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_finite_float_overshoot_reaches_real_encoder_without_clamp_or_gain(tmp_path, monkeypatch):
    images, sound = media()
    waveform = .1 * torch.sin(torch.arange(32000, dtype=torch.float32) * .1)
    sound["waveform"] = waveform.expand(1, 2, -1).clone()
    sound["waveform"][0, 0, 1200] = 1.0471221  # Actual decoded AAC failure's peak.
    sound["waveform"][0, 1, 2000] = -1.02
    original = sound["waveform"].clone()
    write = delivery.delivery._write_planar_audio_raw
    observed = []

    def actual_write(path, array):
        np.testing.assert_array_equal(array, original[0].numpy())
        observed.append(array.copy())
        return write(path, array)

    monkeypatch.setattr(delivery.delivery, "_write_planar_audio_raw", actual_write)
    report = delivery.save_h3_av_safe(images, sound, tmp_path / "overshoot.mp4")
    assert len(observed) == 1 and torch.equal(original, sound["waveform"])
    assert report["delivery_verification"]["full_decode_verified"]
    assert report["audio"]["clipped_sample_values"] == 0
    assert report["audio_amplitude"]["above_full_scale_sample_values"] == 2
    assert report["audio_amplitude"]["source_peak_abs"] == float(original.abs().max())
    assert report["audio_amplitude"]["gain_applied"] == 1.0
    assert report["audio_amplitude"]["clipped_by_this_saver"] is False


def test_shared_world_audio_clipping_default_and_safe_float_opt_out_are_separate():
    sound = {"sample_rate": 32000, "waveform": torch.tensor([[[1.0471221, -.5, -1.02]]])}
    before = sound["waveform"].clone()
    legacy, _, old_report = delivery._normalize_output_audio(sound, expected_samples=3)
    preserved, _, report = delivery._normalize_output_audio(sound, expected_samples=3, clip_amplitude=False)
    np.testing.assert_array_equal(legacy, before[0].clamp(-1, 1).repeat(2, 1).numpy())
    np.testing.assert_array_equal(preserved, before[0].repeat(2, 1).numpy())
    assert old_report["clipped_sample_values"] == 4 and report["clipped_sample_values"] == 0
    assert torch.equal(sound["waveform"], before)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_audio_remains_a_real_error(tmp_path, value):
    images, sound = media()
    sound["waveform"][0, 0, 4] = value
    with pytest.raises(ValueError, match="NaN|infinity"):
        delivery.save_h3_av_safe(images, sound, tmp_path / "invalid.mp4")
    assert not list(tmp_path.iterdir())
