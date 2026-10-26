"""Independent rational-clock/phase counterexamples for the H24 prototype."""
from fractions import Fraction

import pytest
import torch

from h3_audio_t8_pkg.av_loop_clock import LoopPlan, pcm_plan, periodic, repeat_pcm
from h3_audio_t8_pkg.av_loop_runtime import _view_payload


@pytest.mark.parametrize("clips", [1, 2, 3, 6, 9])
def test_actual_video_and_audio_clocks_are_not_falsely_equal(clips):
    plan = LoopPlan(17 * clips + 5)
    report = plan.report()
    error = Fraction(report["audio_period_error_ticks"]["numerator"],
                     report["audio_period_error_ticks"]["denominator"])
    assert Fraction(plan.audio_ticks, 40) - Fraction(plan.frames, 24) == error / 40
    assert report["latent_periods_exact"] is (clips % 3 == 0)
    if clips % 3:
        assert abs(error * 3 / 40) == Fraction(1, 40)  # 25ms after three raw tick loops.
    if clips == 9:
        second = report["views"][1]
        assert second["shift_frames"] == 68
        assert not second["is_exact_half_turn"]  # 153/2 is 76.5 frames, not 68.
        assert second["audio_rotation_error_ticks"] == {"numerator": -1, "denominator": 3}


def test_periodic_indices_keep_stereo_and_wrap_padding_larger_than_period():
    value = torch.tensor([[[1., 2., 3.], [11., 12., 13.]]])
    output = periodic(value, 7, 8, -1, shift=1)
    expected_indices = [(i - 1) % 3 for i in range(-7, 11)]
    assert torch.equal(output, value[..., expected_indices])
    assert torch.equal(output[:, 1] - output[:, 0], torch.full((1, 18), 10.))
    assert value.tolist() == [[[1., 2., 3.], [11., 12., 13.]]]


@pytest.mark.parametrize("length", [5, 21, 23, True])
def test_invalid_native_grid_is_real_input_failure(length):
    with pytest.raises(ValueError):
        LoopPlan(length)


def test_decoder_length_is_validated_without_floor_division():
    plan = LoopPlan(22)
    ticks = plan.audio_ticks + plan.audio_pad * 2
    crop = pcm_plan(plan, 32000, ticks, ticks * 800)
    assert crop["sample_count"] == 22667
    assert crop["latent_period_samples"] == {"numerator": 22400, "denominator": 1}
    assert crop["latent_seam_crossed"]
    assert not crop["phase_continuity_qualified"]
    with pytest.raises(ValueError, match="PCM samples"):
        pcm_plan(plan, 32000, ticks, ticks * 800 + 1)


@pytest.mark.parametrize("clips", [1, 2, 3, 9])
def test_three_cycles_use_cumulative_pcm_boundaries_without_tick_drift(clips):
    plan = LoopPlan(17 * clips + 5)
    count = round(Fraction(plan.frames * 32000, 24))
    left = torch.linspace(-.2, .2, count)
    sound = torch.stack((left, -left))[None]
    repeated, report = repeat_pcm(sound, 32000, plan.frames)
    assert repeated.shape == (1, 2, 68000 * clips)
    assert torch.equal(repeated[:, 0], -repeated[:, 1])
    assert report["sample_boundaries"][-1] == 68000 * clips
    assert report["total_rounding_error_samples"] == {"numerator": 0, "denominator": 1}
    assert not report["resampled"] and not report["crossfaded"]
    assert max(map(abs, report["cycle_sample_adjustments"])) <= 1
    if clips % 3:
        assert any(report["cycle_sample_adjustments"])


def test_timed_guides_wrap_actual_indices_and_never_mutate_reference_owners():
    plan = LoopPlan(56)
    reference = object()
    payload = {"layout": object(), "frame_count": 56, "refs": [reference],
               "keyframes": [{"resolved_frame_index": 0}, {"resolved_frame_index": 55}]}
    mapped = _view_payload(payload, plan, 1)
    # Native last frame is N+4, so modulo maps it to loop frame 4, not 0.
    assert [row["resolved_frame_index"] for row in mapped["keyframes"]] == [34, 38]
    assert "layout" not in mapped and mapped["frame_count"] == 90
    assert mapped["refs"] is payload["refs"] and mapped["refs"][0] is reference
    assert payload["keyframes"] == [{"resolved_frame_index": 0}, {"resolved_frame_index": 55}]


def test_nonfinite_pcm_is_never_a_clock_success():
    waveform = torch.zeros(1, 2, 22667)
    waveform[0, 1, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        repeat_pcm(waveform, 32000, 17)
