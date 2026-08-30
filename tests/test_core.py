from __future__ import annotations

from collections.abc import Mapping

import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.core import (
    align_frame_count,
    encode_audio_once,
    ensure_h3_audio_vae_non_aligned_crop_compat,
    fit_audio_latent,
    replace_audio_latent,
    temporal_shape,
    validate_audio,
)


class LazyAudioMapping(Mapping):
    def __init__(self, audio):
        self.audio = audio
        self.loaded = False

    def __getitem__(self, key):
        self.loaded = True
        return self.audio[key]

    def __iter__(self):
        self.loaded = True
        return iter(self.audio)

    def __len__(self):
        self.loaded = True
        return len(self.audio)


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


def test_validate_audio_accepts_lazy_mapping_from_video_loaders():
    lazy_audio = LazyAudioMapping(
        {"waveform": torch.ones((1, 1, 8000)), "sample_rate": 16000}
    )
    waveform, sample_rate = validate_audio(lazy_audio)
    assert lazy_audio.loaded is True
    assert waveform.shape == (1, 1, 8000)
    assert sample_rate == 16000


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


class _H3AudioVAEProbe:
    latent_channels = 32
    latent_dim = 2
    audio_sample_rate = 32000

    def __init__(self, crop_input=True):
        self.crop_input = crop_input
        self.crop_seen_during_encode = None

    def encode(self, waveform):
        self.crop_seen_during_encode = self.crop_input
        # Preserve the non-aligned 13,667-sample contract as a distinct final step.
        latent_steps = (waveform.shape[1] + 799) // 800
        return torch.zeros((1, 32, 2, latent_steps))


def test_h3_audio_vae_non_aligned_crop_compat_runs_before_encode():
    vae = _H3AudioVAEProbe(crop_input=True)
    audio = {"waveform": torch.zeros((1, 2, 13_667)), "sample_rate": 32000}

    latent = encode_audio_once(vae, audio)

    assert vae.crop_input is False
    assert vae.crop_seen_during_encode is False
    assert latent.shape[-1] == 18


def test_h3_audio_vae_crop_compat_is_idempotent_and_scoped():
    vae = _H3AudioVAEProbe(crop_input=False)
    assert ensure_h3_audio_vae_non_aligned_crop_compat(vae) is False

    class NonH3VAE:
        crop_input = True
        latent_channels = 4
        latent_dim = 2

    other = NonH3VAE()
    assert ensure_h3_audio_vae_non_aligned_crop_compat(other) is False
    assert other.crop_input is True
