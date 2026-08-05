from __future__ import annotations

import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.core import (
    align_frame_count,
    fit_audio_latent,
    replace_audio_latent,
    temporal_shape,
)


def test_frame_grid_and_temporal_shapes():
    assert align_frame_count(5) == 5
    assert align_frame_count(6) == 22
    assert align_frame_count(123) == 124
    assert align_frame_count(124) == 124
    assert temporal_shape(124) == (124, 37, 207)


def test_fit_audio_latent_trims_and_pads():
    template = torch.zeros((1, 32, 2, 10))
    short = fit_audio_latent(torch.ones((1, 32, 2, 4)), template)
    assert short.shape == template.shape
    assert torch.all(short[..., :4] == 1)
    assert torch.all(short[..., 4:] == 0)
    long = fit_audio_latent(torch.ones((1, 32, 2, 20)), template)
    assert long.shape == template.shape
    assert torch.all(long == 1)


def test_audio_replacement_preserves_existing_video_mask():
    video = torch.zeros((1, 24, 2, 4, 4))
    audio = torch.zeros((1, 32, 2, 10))
    video_mask = torch.full_like(video, 0.42)
    av = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, torch.ones_like(audio))),
    }
    output = replace_audio_latent(av, torch.ones_like(audio), 0.25)
    out_video_mask, out_audio_mask = output["noise_mask"].unbind()
    assert torch.equal(out_video_mask, video_mask)
    assert torch.all(out_audio_mask == 0.25)
