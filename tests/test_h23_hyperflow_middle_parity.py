"""Middle-image CPU parity diagnostic, not trained audio/visual quality proof.

Actual tiny 50+2-block Core H3, video encoder and HyperFlow samplers. Text,
adapter tensors and FakeAudioVAE remain explicitly synthetic test fixtures.
"""
import pytest
import torch

import test_h21_hyperflow_keyframe_runtime as cases

actual_base = cases.actual_base
actual_video_encoder = cases.actual_video_encoder


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_middle_image_continuous_split_does_not_mix_or_restart_native_audio(
        actual_base, actual_video_encoder, relay, nonuniform):
    base, weights, original = actual_base
    low = cases.prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    high = cases.prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    single, _ = cases.run_route(low, high, "single8")
    expected = tuple(part.clone() for part in single["samples"].unbind())
    for route in ("combined", "separate"):
        next_low = cases.prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
        next_high = cases.prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
        split, _ = cases.run_route(next_low, next_high, route)
        for actual, reference in zip(split["samples"].unbind(), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        cases.check_original(base, original)
    assert not torch.cuda.is_initialized()
