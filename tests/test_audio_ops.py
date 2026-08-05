from __future__ import annotations

import torch

from h3_audio_t8_pkg.audio_ops import decode_av_latent, inject_audio_latent, mix_audio, trim_av_output
from h3_audio_t8_pkg.core import empty_av_latent
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_standalone_audio_control_lock_and_remix():
    latent, _ = empty_av_latent(128, 128, 124)
    vae = FakeAudioVAE()
    locked, passthrough = inject_audio_latent(latent, make_audio(), vae, "lock", 0.9)
    assert torch.all(locked["noise_mask"].unbind()[1] == 0)
    remixed, _ = inject_audio_latent(latent, make_audio(), vae, "remix", 0.4)
    assert torch.all(remixed["noise_mask"].unbind()[1] == 0.4)
    assert len(vae.encode_calls) == 2


def test_mix_resamples_matches_channels_ducks_and_limits_peak():
    source = make_audio(1.0, 32000, 0.8, 1)
    generated = make_audio(0.5, 48000, 0.8, 2)
    output = mix_audio(source, generated, 0, 0, 0.5, "source", -1.0)
    assert output["sample_rate"] == 32000
    assert output["waveform"].shape == (1, 2, 32000)
    assert float(output["waveform"].abs().max()) <= 10 ** (-1 / 20) + 1e-5


def test_decode_returns_both_modalities_and_separate_latents():
    latent, _ = empty_av_latent(128, 128, 124)
    images, audio, video_latent, audio_latent = decode_av_latent(latent, FakeVideoVAE(), FakeAudioVAE())
    assert images.ndim == 4
    assert audio["sample_rate"] == 32000
    assert video_latent["samples"].ndim == 5
    assert audio_latent["samples"].ndim == 4


def test_output_trim_applies_same_time_window_to_frames_and_audio():
    frames = torch.arange(124).view(124, 1, 1, 1).float()
    audio = make_audio(124 / 24, sample_rate=24000, value=0.2)
    out_frames, out_audio, report = trim_av_output(frames, 1.0, 2.0, audio, 24.0)
    assert out_frames.shape[0] == 48
    assert out_frames[0].item() == 24
    assert out_audio["waveform"].shape[-1] == 48000
