"""Explicit native split-sampler audio continuation; visual re-noise is retained."""
import json
import math

import torch
from comfy.nested_tensor import NestedTensor

from .latent_upscale import _nested_parts


class AudioContinuationNoise:
    def __init__(self, parent):
        if not callable(getattr(parent, "generate_noise", None)):
            raise TypeError("Connected NOISE needs generate_noise(latent)")
        self.parent = parent
        self.seed = parent.seed

    def generate_noise(self, latent):
        # Delegate once, retaining the user's selected visual noise and seed.
        video, audio = _nested_parts(self.parent.generate_noise(latent), "selected AV noise")
        target_video, target_audio = _nested_parts(latent["samples"], "HIGH samples")
        if video.shape != target_video.shape or audio.shape != target_audio.shape:
            raise ValueError("Selected NOISE must match the actual HIGH video/audio shapes")
        return NestedTensor((video, torch.zeros_like(audio)))


def continue_audio(coarse_output, high_latent, visual_noise, low_sigmas, high_sigmas,
                   coarse_sampler=None):
    for name, values in (("LOW", low_sigmas), ("HIGH", high_sigmas)):
        if not torch.is_tensor(values) or values.ndim != 1 or len(values) < 2:
            raise ValueError(f"{name} SIGMAS must be a one-dimensional interval schedule")
        if not torch.isfinite(values).all() or torch.any(values < 0):
            raise ValueError(f"{name} SIGMAS must be finite and nonnegative")
    boundary = float(low_sigmas[-1])
    if boundary != float(high_sigmas[0]):
        raise ValueError("Audio continuation requires LOW end sigma == HIGH start sigma")
    video, clean_audio = _nested_parts(high_latent["samples"], "HIGH samples")
    _, continued_audio = _nested_parts(coarse_output["samples"], "LOW output samples")
    if continued_audio.shape != clean_audio.shape:
        raise ValueError("LOW output and HIGH audio timeline shapes must match")
    if not torch.isfinite(continued_audio).all():
        raise ValueError("LOW output audio contains NaN or Inf")
    # Native FLOW_AV already exports an audio-clock normalized LATENT. The T8
    # direct-clock sampler advances audio at sigma_a, but Core's generic
    # inverse_noise_scaling exports every stream divided by (1 - sigma_v).
    # Canonicalize that LOW socket0 before either native FLOW_AV or T8's
    # partial-start rebase initializes HIGH. This is not an x0 prediction.
    function = getattr(coarse_sampler, "sampler_function", None)
    shift_video = getattr(function, "_minimax_h3_shift_video", None)
    shift_audio = getattr(function, "_minimax_h3_shift_audio", None)
    factor = 1.0
    sigma_audio = None
    clock = "native_FLOW_AV_or_unspecified"
    if shift_video is not None and shift_audio is not None:
        shift_video, shift_audio = float(shift_video), float(shift_audio)
        if not all(math.isfinite(v) and v > 0 for v in (shift_video, shift_audio)):
            raise ValueError("LOW sampler video/audio clock shifts must be positive and finite")
        from .sampling import time_shift_sigma
        sigma_audio = float(time_shift_sigma(boundary, shift_video, shift_audio))
        if not 0 <= boundary < 1 or not 0 <= sigma_audio < 1:
            raise ValueError("Partial audio continuation needs a boundary below the initial pure-noise sigma")
        factor = (1.0 - boundary) / (1.0 - sigma_audio)
        clock = "T8_direct_audio_clock_canonicalized"
    elif coarse_sampler is not None:
        clock = "unlisted_sampler_preserved_clock_unverified"
    audio = continued_audio if factor == 1.0 else continued_audio * factor
    mask = high_latent.get("noise_mask")
    if mask is not None:
        _, audio_mask = _nested_parts(mask, "HIGH noise_mask")
        if audio_mask.shape != clean_audio.shape:
            raise ValueError("HIGH audio noise_mask must match its audio samples")
        # Known source regions continue to belong to the HIGH template.
        audio = torch.where(audio_mask.to(audio.device) == 0, clean_audio.to(audio.device), audio)
    output = {**high_latent, "samples": NestedTensor((video, audio))}
    report = dict(schema="t8.two_pass.audio_continuation.v1",
                  status="prepared_unqualified", boundary_sigma=boundary,
                  audio_source="coarse_sampler_output_not_denoised_output",
                  low_output_clock=clock, low_audio_sigma=sigma_audio,
                  low_output_audio_normalization=factor,
                  initial_audio_fresh_noise=False, generated_audio_force_frozen=False,
                  noise_scope="HIGH initial NOISE only; stochastic sampler internal noise is unchanged",
                  visual_noise="connected_provider_delegated_once_unchanged",
                  seed=visual_noise.seed, extra_denoiser_evaluations=0,
                  cache_identity="opaque_NOISE_requires_uncertified_execution",
                  quality_accepted=False,
                  contract="Use matching native LOW/HIGH audio clocks; does not certify arbitrary sampler/model equivalence.")
    return output, AudioContinuationNoise(visual_noise), json.dumps(report, sort_keys=True)
