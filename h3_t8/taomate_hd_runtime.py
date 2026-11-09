"""TaoMate-HD teacher-coupled 2 LOW + 1 HIGH, through native Core guiders.

T8 adaptation, 2026-11-05. Teacher states/coupled lift follow TaoLiveAIGC
and slmonker/ComfyUI-TaoMateHD 1e4fb7c. Unlike that adapter, keep native
keyframes, clean-source masks, user MODEL classes and all input metadata.
See THIRD_PARTY_NOTICES/TAOMATE-HD-LICENSE.txt and TAOMATE-HD-NOTICE.txt.
"""
from __future__ import annotations

import json
import math
import time
import uuid

import torch
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils

from . import taomate_hd_math as math_hd
from .progressive_masking import normalize_av_masks, resize_video_source
from .sampling import nested_av_parts, time_shift_sigma


def conditioning_at_size(conditioning, height, width):
    output = []
    for embedding, metadata in conditioning:
        details = dict(metadata)
        if "minimax_keyframes" in details:
            details["minimax_keyframes"] = [
                ({**frame, "latent": resize_video_source(frame["latent"], height, width)}
                 if frame.get("latent") is not None else dict(frame))
                for frame in details["minimax_keyframes"]]
        output.append([embedding, details])
    return output


class TeacherStage:
    """Euler continuation with an independent original clean inpaint anchor."""
    def __init__(self, continuation, clean, conditioning_noise, audio_clock=None):
        self.continuation = continuation
        self.clean = clean
        self.conditioning_noise = conditioning_noise
        self.calls = 0
        self.audio_clock = audio_clock

    @torch.no_grad()
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        sampling = model_wrap.inner_model.model_sampling
        x = latent_image if self.continuation else sampling.noise_scaling(
            sigmas[0], noise, latent_image, max_denoise=False)
        denoiser = model_wrap
        if denoise_mask is not None:
            denoiser = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
            clean, _ = comfy.utils.pack_latents(self.clean.unbind())
            anchor_noise, _ = comfy.utils.pack_latents(self.conditioning_noise.unbind())
            denoiser.latent_image = model_wrap.inner_model.process_latent_in(clean.to(x))
            denoiser.noise = anchor_noise.to(x)
        arguments = dict(extra_args)
        if denoise_mask is not None:
            arguments["denoise_mask"] = denoise_mask
        for step, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            comfy.model_management.throw_exception_if_processing_interrupted()
            denoised = denoiser(x, sigma * x.new_ones([x.shape[0]]), **arguments)
            self.calls += 1
            if callback is not None:
                callback(step, denoised, x, len(sigmas) - 1)
            following_state = denoised if float(following) == 0 else x + (denoised - x) * (1.0 - following / sigma)
            if self.audio_clock is not None:
                # The selected MODEL carries unscaled audio and returns raw
                # audio velocity, as in the existing dual-clock Euler route.
                # Native FLOW_AV already converts that velocity to the packed
                # video clock and must retain the original update above.
                shift_video, shift_audio = self.audio_clock
                video_values = math.prod(self.clean.unbind()[0].shape[1:])
                audio_delta = (time_shift_sigma(following, shift_video, shift_audio)
                               - time_shift_sigma(sigma, shift_video, shift_audio))
                if denoise_mask is not None:
                    video_delta = following - sigma
                    audio_delta = video_delta + denoise_mask[..., video_values:] * (audio_delta - video_delta)
                audio_derivative = (x[..., video_values:] - denoised[..., video_values:]) / sigma
                following_state = torch.cat((following_state[..., :video_values],
                    x[..., video_values:] + audio_derivative * audio_delta), dim=-1)
            x = following_state
        return x


def _audio_clock(model):
    """Observe actual selected interfaces, never replace the user's sampler."""
    sampling = model.get_model_object("model_sampling")
    base = getattr(model, "model", None)
    if not callable(getattr(base, "audio_scale", None)):
        return None, {"mode": "delegated_single_clock_unverified", "audio_scale": None}
    scale = getattr(sampling, "audio_scale", None)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("The selected packed-AV model needs a positive finite audio_scale")
    options = model.model_options.get("transformer_options", {})
    diffusion = model.get_model_object("diffusion_model")
    video_shift = options.get("minimax_h3_sigma_shift_video", getattr(diffusion, "sigma_shift_video", None))
    audio_shift = options.get("minimax_h3_sigma_shift_audio", getattr(diffusion, "sigma_shift_audio", None))
    if scale == 1.0:
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (video_shift, audio_shift)):
            raise ValueError("Unscaled H3 audio velocity needs actual positive video/audio shift values")
        clock = (float(video_shift), float(audio_shift))
        return clock, {"mode": "raw_audio_velocity_dual_clock", "audio_scale": scale,
                       "video_shift": clock[0], "audio_shift": clock[1]}
    return None, {"mode": "native_packed_flow_av", "audio_scale": scale,
                  "video_shift": video_shift, "audio_shift": audio_shift}


def _observe_model_calls(model, counts, active_stage):
    """Request-local Core APPLY_MODEL observer; all existing delegates remain."""
    if not callable(getattr(model, "clone", None)):
        return model, False
    selected = model.clone()
    if not callable(getattr(selected, "add_wrapper_with_key", None)):
        return model, False
    def observe(executor, *args, **kwargs):
        value = executor(*args, **kwargs)
        counts[active_stage[0]] += 1
        return value
    selected.add_wrapper_with_key("apply_model", "t8_taomate_hd_observe_" + uuid.uuid4().hex, observe)
    return selected, True


def _stage_counts(total):
    if type(total) is not int or total < 3:
        raise ValueError("TaoMate-HD has three teacher intervals; provide at least three evaluations")
    a = math.ceil(total / 3)
    b = math.ceil((total - a) / 2)
    return a, b, total - a - b


@torch.no_grad()
def sample_taomate_hd(model, positive, latent, *, negative=None, seed=8301, steps=3,
                      cfg=1.0, low_scale=.625, high_model=None):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not math.isfinite(cfg):
        raise ValueError("cfg must be finite")
    if cfg != 1.0 and negative is None:
        raise ValueError("CFG other than 1 requires negative conditioning")
    if not positive:
        raise ValueError("Positive conditioning is required")
    counts = _stage_counts(steps)
    video, audio = nested_av_parts(latent)
    if (tuple(video.shape[:2]) != (1, 24) or audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2)
            or not all(p.is_floating_point() and bool(torch.isfinite(p).all()) for p in (video, audio))):
        raise ValueError("TaoMate-HD noise geometry needs finite [1,24,T,H,W] video and [1,32,2,A] audio")
    high = math_hd.ProgressiveCanvas.from_video_latent(video)
    low = math_hd.derive_low_canvas(high, low_scale)
    selected_sampling = model.get_model_object("model_sampling")
    # Derive states with the selected model's clock; never silently patch it.
    shift = float(selected_sampling.shift)
    boundaries = math_hd.teacher_sigmas(video_shift=shift)
    low_noise, high_noise = math_hd.coupled_video_noise(low_canvas=low, high_canvas=high,
        temporal_length=video.shape[2], seed=seed)
    sound_noise = math_hd.audio_noise(audio_shape=tuple(audio.shape), high_canvas=high,
        temporal_length=video.shape[2], seed=seed)
    nested = comfy.nested_tensor.NestedTensor
    low_clean = nested((resize_video_source(video, low.latent_height, low.latent_width), audio))
    low_random = nested((low_noise, sound_noise))
    high_random = nested((high_noise, sound_noise))
    low_mask = normalize_av_masks(latent.get("noise_mask"), *low_clean.unbind())
    high_mask = normalize_av_masks(latent.get("noise_mask"), video, audio)
    negative = negative if negative is not None else []
    low_positive = conditioning_at_size(positive, low.latent_height, low.latent_width)
    low_negative = conditioning_at_size(negative, low.latent_height, low.latent_width)
    low_clock, low_clock_report = _audio_clock(model)
    high_selected = model if high_model is None else high_model
    high_clock, high_clock_report = _audio_clock(high_selected)
    apply_calls, active_stage = [0, 0, 0], [0]
    low_observed, low_observer = _observe_model_calls(model, apply_calls, active_stage)
    high_observed, high_observer = _observe_model_calls(high_selected, apply_calls, active_stage)
    guider = comfy.samplers.CFGGuider(low_observed)
    guider.set_conds(low_positive, low_negative)
    guider.set_cfg(cfg)
    high_guider = comfy.samplers.CFGGuider(high_observed)
    high_guider.set_conds(positive, negative)
    high_guider.set_cfg(cfg)
    state = low_clean
    stages = []
    intermediate = comfy.model_management.intermediate_device()
    for index, count in enumerate(counts):
        active_stage[0] = index
        current = guider if index < 2 else high_guider
        original = low_clean if index < 2 else latent["samples"]
        random = low_random if index < 2 else high_random
        sampler = TeacherStage(index != 0, original, random, low_clock if index < 2 else high_clock)
        noise = random if index == 0 else nested(tuple(torch.zeros_like(p) for p in state.unbind()))
        sigmas = torch.linspace(boundaries[index], boundaries[index+1], count + 1, dtype=torch.float32)
        started = time.perf_counter()
        state = current.sample(noise, state, sampler, sigmas,
            denoise_mask=low_mask if index < 2 else high_mask, disable_pbar=False, seed=seed).to(intermediate)
        stages.append(dict(stage=("low_0_16", "low_16_33", "high_33_49")[index],
                           denoiser_calls=sampler.calls, solver_evaluations=sampler.calls,
                           actual_model_apply_calls=apply_calls[index] if (low_observer if index < 2 else high_observer) else None,
                           audio_clock=low_clock_report if index < 2 else high_clock_report, sigmas=sigmas.tolist(),
                           elapsed_seconds=time.perf_counter()-started,
                           video_shape=list(state.unbind()[0].shape), audio_shape=list(state.unbind()[1].shape)))
        if index == 1:
            v, a = state.unbind()
            lifted = math_hd.lift_video_latent(low_video=v, low_noise=low_noise.to(v.device),
                high_noise=high_noise.to(v.device), sigma=boundaries[2], low_canvas=low, high_canvas=high)
            state = nested((lifted, a))
    if not all(bool(torch.isfinite(p).all()) for p in state.unbind()):
        raise RuntimeError("TaoMate-HD returned non-finite video/audio; no completed result")
    output = dict(latent)
    output["samples"] = state
    report = dict(schema="t8.taomate_hd.v1", status="sampled_quality_unverified",
                  recipe="teacher_0_16_33_49", official_evaluation_count=steps == 3 and cfg == 1.0,
                  total_denoiser_calls=sum(s["denoiser_calls"] for s in stages), stages=stages,
                  total_solver_evaluations=sum(s["solver_evaluations"] for s in stages),
                  actual_model_apply_calls=sum(apply_calls) if low_observer and high_observer else None,
                  model_call_observer="completed Core APPLY_MODEL delegation; bypassing custom wrappers are not observed",
                  video_shift=shift, low_canvas=low.as_dict(), high_canvas=high.as_dict(),
                  cfg=cfg, seed=seed, independent_high_model=high_model is not None,
                  lift="bilinear_state_plus_sigma_correlated_noise_residual",
                  audio_lift="same LOW2 audio state; no resample/no fresh audio draw",
                  noise="per_temporal_high_seed_plus_1000003; orthonormal_lowpass",
                  original_mask_retained=high_mask is not None, source_metadata_retained=True,
                  timing_scope="node-stage wall time; includes Core loading, no forced CUDA synchronization",
                  limitations="Single window. No upstream streaming history or eight-GPU pipeline claim. "
                              "More than 3 evaluations subdivides teacher intervals, not a new trained recipe.")
    return output, json.dumps(report, ensure_ascii=False, allow_nan=False)
