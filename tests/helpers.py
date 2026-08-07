from __future__ import annotations

import math

import torch


class FakeClip:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"prompt": prompt, "kwargs": kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 4, 8)), {"tokens": tokens}]]


class FakeVideoVAE:
    # ComfyUI's generic VAE wrapper exposes audio_sample_rate even for video
    # VAEs. The H3 video/audio distinction must use the latent contract.
    audio_sample_rate = 44100
    latent_channels = 24
    latent_dim = 3
    output_channels = 3

    def __init__(self):
        self.encode_calls = []

    def encode(self, images):
        self.encode_calls.append(images)
        frames, height, width = images.shape[:3]
        latent_t = 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2
        return torch.zeros((1, 24, latent_t, max(1, height // 16), max(1, width // 16)))

    def decode(self, latent):
        latent_t = latent.shape[2]
        frames = 1 if latent_t == 1 else ((latent_t - 2) // 5) * 17 + 5
        return torch.zeros((1, frames, latent.shape[3] * 16, latent.shape[4] * 16, 3))


class FakeAudioVAE:
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000
    latent_channels = 32
    latent_dim = 2
    output_channels = 2

    def __init__(self):
        self.encode_calls = []

    def encode(self, waveform_last):
        self.encode_calls.append(waveform_last)
        latent_t = max(1, math.ceil(waveform_last.shape[1] / 800))
        return torch.full((1, 32, 2, latent_t), 0.25)

    def decode(self, latent):
        samples = latent.shape[-1] * 800
        return torch.full((latent.shape[0], samples, 2), 0.1)


def make_audio(seconds=5.0, sample_rate=32000, value=0.1, channels=1):
    samples = round(seconds * sample_rate)
    return {"waveform": torch.full((1, channels, samples), value), "sample_rate": sample_rate}
