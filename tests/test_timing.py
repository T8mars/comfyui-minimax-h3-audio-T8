from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.timing import make_timing_plan, window_audio
from helpers import make_audio


def test_short_scene_gets_trained_minimum_context_without_timeline_drift():
    plan = make_timing_plan(0.0, 1.0, ensure_minimum_context=True, source_duration_seconds=1.0)
    assert plan.frame_count == 124
    assert plan.render_duration_seconds == pytest.approx(124 / 24)
    assert plan.final_duration_seconds == 1.0
    assert plan.main_prompt_start_seconds == pytest.approx(plan.final_trim_start_seconds)
    assert "main requested action starts" in plan.prompt_note()


def test_warmup_and_alignment_are_reflected_in_final_trim():
    plan = make_timing_plan(10.0, 2.0, 1.0, 1.0, True, 20.0)
    assert plan.frame_count == 124
    assert plan.final_trim_start_seconds == pytest.approx(1 + ((124 / 24) - 4) / 2)
    assert json.loads(plan.report())["frame_count"] == 124


def test_audio_window_slices_and_zero_pads_at_source_start():
    audio = make_audio(seconds=1.0, sample_rate=24000, value=0.5)
    plan = make_timing_plan(0.0, 1.0, ensure_minimum_context=True, source_duration_seconds=1.0)
    output = window_audio(audio, plan)
    assert output["waveform"].shape[-1] == round((124 / 24) * 24000)
    first_nonzero = int(torch.nonzero(output["waveform"][0, 0], as_tuple=False)[0])
    assert first_nonzero == round(plan.left_pad_seconds * 24000)


def test_invalid_scene_duration_fails():
    with pytest.raises(ValueError):
        make_timing_plan(0, 0)
