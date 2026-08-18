from __future__ import annotations

import subprocess

import numpy as np
import pytest

from h3_audio_t8_pkg.tools import validate_h3_speed_multimodal_outputs as output_tool
from h3_audio_t8_pkg.tools.validate_h3_speed_multimodal_outputs import (
    _audio_pair_metrics,
    _frame_metrics,
    _resize_like_conditioning,
)


def test_resize_like_conditioning_matches_requested_canvas_and_center_crop():
    source = np.zeros((100, 200, 3), dtype=np.uint8)
    source[:, :25] = 255
    stretched = _resize_like_conditioning(source, 128, 128, "disabled")
    centered = _resize_like_conditioning(source, 128, 128, "center")
    assert stretched.shape == (128, 128, 3)
    assert centered.shape == (128, 128, 3)
    assert centered[:, :8].mean() < stretched[:, :8].mean()


def test_frame_metrics_identical_input_is_exact():
    frame = np.full((16, 32, 3), 127, dtype=np.uint8)
    metrics = _frame_metrics(frame, frame.copy())
    assert metrics["normalized_mae"] == 0.0
    assert metrics["psnr_db"] is None


def test_audio_pair_metrics_reports_exact_match():
    waveform = np.linspace(-0.5, 0.5, 640, dtype=np.float32).reshape(-1, 2)
    metrics = _audio_pair_metrics(waveform, waveform.copy())
    assert metrics["sample_count_equal"] is True
    assert metrics["zero_lag_correlation"] == pytest.approx(1.0)
    assert metrics["generated_rms"] == metrics["reference_rms"]


def test_shared_strict_decode_makes_reported_decoder_errors_fatal(monkeypatch, tmp_path):
    commands = []

    def fake_run(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(output_tool, "_run", fake_run)
    result = output_tool._strict_decode(tmp_path / "sample.mp4", "ffmpeg", attempts=1)
    assert result["passed"] is True
    assert "-xerror" in commands[0]
    assert commands[0][commands[0].index("-err_detect") + 1] == "explode"
