"""Explicit paired NOISE providers for the existing LOW/HIGH samplers.

No sampling, model patch, latent upscale or audio resampling happens here.
Each request uses a local CPU RNG, so reusing a provider is deterministic and
does not change a stochastic sampler's global RNG. Large high noise tensors
are regenerated, not retained while the LOW model runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F


MODES = ("independent", "haar", "bilinear")


def _parts(latent):
    if not isinstance(latent, dict):
        raise ValueError("Connect an AV LATENT dictionary")
    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError("Paired noise preparation needs joint video/audio LATENT")
    parts = tuple(samples.unbind())
    if (len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4
            or any(not p.is_floating_point() or any(n < 1 for n in p.shape) for p in parts)):
        raise ValueError("Expected floating video [B,C,T,H,W] and audio [B,C,S,T] tensors")
    if parts[0].shape[0] != parts[1].shape[0]:
        raise ValueError("Video and audio noise batches must match")
    return parts


def _indices(latent, batch):
    raw = latent.get("batch_index")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != batch:
        raise ValueError("batch_index must contain one integer per latent batch")
    if any(type(i) is not int or i < 0 for i in raw):
        raise ValueError("batch_index entries must be nonnegative integers")
    return tuple(raw)


def haar_half(video, target_hw):
    """Orthonormal LL projection of disjoint 2x2 spatial noise blocks."""
    h, w = target_hw
    if tuple(video.shape[-2:]) != (2 * h, 2 * w):
        raise ValueError("Haar noise requires LOW height and width to be exactly half of HIGH; "
                         "use an exact half-size canvas or explicitly select another noise mode")
    blocks = video.float().reshape(*video.shape[:-2], h, 2, w, 2)
    return blocks.sum(dim=(-1, -3)) * 0.5


def _draw(shape, dtype, generator, indices):
    # Same float32 CPU draw order and dtype conversion as native Core NOISE,
    # but with a private generator instead of torch.manual_seed's global state.
    if indices is None:
        return torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu").to(dtype)
    selected = {}
    wanted = set(indices)
    for index in range(max(indices) + 1):
        value = torch.randn((1, *shape[1:]), dtype=torch.float32, generator=generator, device="cpu").to(dtype)
        if index in wanted:
            selected[index] = value
    return torch.cat([selected[index] for index in indices], dim=0)


@dataclass(frozen=True)
class PairedNoise:
    seed: int
    mode: str
    phase: str
    low_video_shape: tuple
    high_video_shape: tuple
    audio_shape: tuple
    low_dtype: torch.dtype
    high_dtype: torch.dtype
    audio_dtype: torch.dtype
    batch_indices: tuple | None

    def generate_noise(self, latent):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        expected_shape = self.low_video_shape if self.phase == "low" else self.high_video_shape
        if self.phase == "low":
            video, audio = _parts(latent)
            if tuple(audio.shape) != self.audio_shape or audio.dtype != self.audio_dtype:
                raise ValueError("LOW audio geometry/dtype differs from the paired noise preparation")
        else:
            video = samples
            if not isinstance(video, torch.Tensor) or getattr(video, "is_nested", False):
                raise ValueError("Connect HIGH video noise to the video-only handoff input")
        expected_dtype = self.low_dtype if self.phase == "low" else self.high_dtype
        if tuple(video.shape) != expected_shape or video.dtype != expected_dtype:
            raise ValueError("Latent geometry/dtype differs from the prepared paired noise")
        # HIGH handoff intentionally passes video only and omits batch_index;
        # the indices captured at preparation remain part of this provider.
        given_indices = _indices(latent, expected_shape[0])
        if ((self.phase == "low" or given_indices is not None)
                and given_indices != self.batch_indices):
            raise ValueError("Latent batch_index changed after paired noise preparation")
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.mode == "independent":
            value = _draw(expected_shape, expected_dtype, generator, self.batch_indices)
        else:
            master = _draw(self.high_video_shape, self.high_dtype, generator, self.batch_indices)
            if self.phase == "high":
                return master
            if self.mode == "haar":
                value = haar_half(master, self.low_video_shape[-2:]).to(self.low_dtype)
            else:
                b, c, t, h, w = master.shape
                frames = master.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                resized = F.interpolate(frames.float(), size=self.low_video_shape[-2:],
                                        mode="bilinear", align_corners=False)
                value = resized.reshape(b, t, c, *self.low_video_shape[-2:]).permute(0, 2, 1, 3, 4)
                value = value.contiguous().to(self.low_dtype)
        if self.phase == "high":
            return value
        from comfy.nested_tensor import NestedTensor
        sound = _draw(self.audio_shape, self.audio_dtype, generator, self.batch_indices)
        return NestedTensor((value, sound))


def prepare_paired_noise(low_source, high_source, seed, mode="independent"):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Noise seed must be an unsigned 64-bit integer")
    if mode not in MODES:
        raise ValueError(f"Unknown paired noise mode: {mode}")
    lv, la = _parts(low_source)
    hv, ha = _parts(high_source)
    if tuple(lv.shape[:-2]) != tuple(hv.shape[:-2]) or tuple(la.shape) != tuple(ha.shape):
        raise ValueError("LOW/HIGH must share video batch/channels/time and audio geometry")
    if la.dtype != ha.dtype:
        raise ValueError("LOW/HIGH audio dtype must match; paired noise never converts audio")
    indices = _indices(low_source, lv.shape[0])
    if indices != _indices(high_source, hv.shape[0]):
        raise ValueError("LOW/HIGH batch_index must select the same items")
    if mode == "haar" and tuple(hv.shape[-2:]) != tuple(2 * n for n in lv.shape[-2:]):
        raise ValueError("Haar noise requires LOW height and width to be exactly half of HIGH")
    high_seed = (seed + 1) % 2**64 if mode == "independent" else seed
    common = dict(mode=mode, low_video_shape=tuple(lv.shape), high_video_shape=tuple(hv.shape),
                  audio_shape=tuple(la.shape), low_dtype=lv.dtype, high_dtype=hv.dtype,
                  audio_dtype=la.dtype, batch_indices=indices)
    low = PairedNoise(seed=seed, phase="low", **common)
    high = PairedNoise(seed=high_seed, phase="high", **common)
    report = dict(schema="t8.progressive.paired_noise.v1", mode=mode, low_seed=seed,
                  high_seed=high_seed, low_video_shape=list(lv.shape), high_video_shape=list(hv.shape),
                  audio_shape=list(la.shape), batch_index=indices, cpu_rng="private_generator",
                  source_latents_modified=False, sampling_calls=0, model_patches=0,
                  audio_resampled=False, noise_materialized=False,
                  audio_rng_layout="low_video_then_audio" if mode == "independent" else "high_video_then_audio",
                  quality_verified=False,
                  note="Haar/Bilinear share the HIGH video draw. Audio is not projected; "
                       "its RNG offset follows that draw and differs from the independent route. "
                       "Equal seeds across different modes do not imply equal generated audio/video.")
    return low, high, json.dumps(report, ensure_ascii=False)
