"""HyperFlow continuous split on the author's one trained AV trajectory.

This is *not* the old Director 4+4 schedule and is not an upscale. The two
stages may have independently patched content models while sharing the same
full-structure H3 base and original HyperFlow artifact.
"""

from __future__ import annotations

import json
import math
import time

import comfy.sample
import comfy.utils
import torch

from .core import nested_av_parts
from .hyperflow_runtime_advanced import ATTACHMENT_KEY, HyperFlowBinding
from .hyperflow_sampling_advanced import build_hyperflow_plan, setup_hyperflow_sampler
from .progressive_sampling_runtime import _restore_stage_objects


def validate_noise_mask(av_latent):
    """Check real Core resize geometry without allocating a second AV mask.

    Core supports resized video-only masks as well as nested video/audio masks.
    Keep its interpolation and default all-one audio mask, not a new mask policy.
    """
    mask = av_latent.get("noise_mask")
    if mask is None:
        return
    latents = nested_av_parts(av_latent)
    masks = mask.unbind() if getattr(mask, "is_nested", False) else [mask]
    for part, latent in zip(masks, latents):
        if (not isinstance(part, torch.Tensor) or part.is_complex() or not part.numel()
                or not bool(torch.isfinite(part).all()) or bool((part < 0).any()) or bool((part > 1).any())):
            raise ValueError("HyperFlow noise mask must contain finite real values in [0,1]")
        try:
            resized = comfy.utils.reshape_mask(torch.empty(part.shape, dtype=part.dtype, device="meta"), latent.shape)
        except (IndexError, RuntimeError, ValueError, TypeError) as error:
            raise ValueError(f"HyperFlow noise mask cannot be resized to native AV geometry: {error}") from error
        if resized.shape != latent.shape:
            raise ValueError("HyperFlow noise mask channels/batch do not match native AV geometry")


def _validate_pair(model_low, model_high, av_latent, split_interval: int) -> tuple[HyperFlowBinding, HyperFlowBinding]:
    low = model_low.get_attachment(ATTACHMENT_KEY)
    high = model_high.get_attachment(ATTACHMENT_KEY)
    if not isinstance(low, HyperFlowBinding) or not isinstance(high, HyperFlowBinding):
        raise ValueError("Both HyperFlow stages require their dedicated Loader, not a generic LoRA")
    if low.sha256 != high.sha256 or low.raw_sigmas != high.raw_sigmas:
        raise ValueError("HyperFlow stages must use the same original file and trained grid")
    if low.model_identity != high.model_identity or id(model_low.model) != id(model_high.model):
        raise ValueError("HyperFlow stages must clone one common full-structure H3 base")
    if type(split_interval) is not int or not 1 <= split_interval <= 7:
        raise ValueError("Continuous split must leave at least one absolute interval in each stage")
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected native AV LATENT dictionary")
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError("HyperFlow split expects native packed MiniMax H3 video/audio latent")
    validate_noise_mask(av_latent)
    return low, high


@torch.inference_mode()
def sample_hyperflow_split(model_low, model_high, positive, av_latent: dict, *,
                           seed: int, split_interval: int = 4, cfg: float = 1.0,
                           negative=None, callback=None, positive_high=None, negative_high=None) -> tuple[dict, str]:
    """Run 4+4 (or another 1..7 split) with exact x_sigma handoff.

    The LOW sampler captures its exact model-space endpoint x_sigma before
    Core's nonterminal clean-latent conversion. The HIGH typed sampler injects
    that state directly. Unmasked Core initialization is only a shape scaffold;
    masked continuation retains the original inpaint source and prepared noise.
    Neither path rebases the captured trajectory or draws new noise for HIGH.
    """
    low_binding, high_binding = _validate_pair(model_low, model_high, av_latent, split_interval)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("cfg must be finite in [0, 100]")
    plan_low = build_hyperflow_plan(model_low, 0, split_interval)
    plan_high = build_hyperflow_plan(model_high, split_interval, 8)
    if plan_low.video_sigmas != plan_high.video_sigmas or plan_low.audio_sigmas != plan_high.audio_sigmas:
        raise ValueError("HyperFlow stage clock mismatch")
    boundary_state = {}

    def capture(state):
        if "x_sigma" in boundary_state:
            raise RuntimeError("HyperFlow low-stage endpoint captured twice")
        boundary_state["x_sigma"] = state

    def continuation_state():
        if "x_sigma" not in boundary_state:
            raise RuntimeError("HyperFlow low-stage endpoint was not captured")
        return boundary_state["x_sigma"]

    branch_low, sampler_low, schedule_low = setup_hyperflow_sampler(
        model_low, av_latent, plan_low, internal_continuation=True,
        endpoint_capture=capture)
    branch_high, sampler_high, schedule_high = setup_hyperflow_sampler(
        model_high, av_latent, plan_high, internal_continuation=True,
        x_sigma_override=continuation_state)
    negative = [] if negative is None else negative
    positive_high = positive if positive_high is None else positive_high
    negative_high = negative if negative_high is None else negative_high
    mask = av_latent.get("noise_mask")
    started = time.perf_counter()
    stage_times = {}
    observed_callbacks = {"LOW": [], "HIGH": []}
    try:
        noise = comfy.sample.prepare_noise(av_latent["samples"], seed)
        low_started = time.perf_counter()

        def low_callback(index, prediction, state, count):
            observed_callbacks["LOW"].append({
                "local_index": index, "absolute_interval": index, "local_total": count,
            })
            if callback is not None:
                callback(index, prediction, state, 8)

        boundary_latent = comfy.sample.sample_custom(
            branch_low, noise, cfg, sampler_low, schedule_low, positive, negative,
            av_latent["samples"], noise_mask=mask, callback=low_callback, disable_pbar=True, seed=seed)
        stage_times["low_seconds"] = time.perf_counter() - low_started
    finally:
        _restore_stage_objects(branch_low)
    try:
        if "x_sigma" not in boundary_state:
            raise RuntimeError("HyperFlow low stage returned without a direct x_sigma state")
        if not all(bool(torch.isfinite(part).all()) for part in boundary_latent.unbind()):
            raise ValueError("HyperFlow boundary latent contains NaN/Inf")
        high_started = time.perf_counter()
        # Core returns a clean-latent view at nonterminal sigma. Keep that for
        # spatial/conditioning shape, but inject the exact captured x_sigma into
        # the typed high sampler. No inverse-scaling round-trip or new RNG.
        # Inpaint uses the original clean source and the *same* prepared noise
        # at every sigma. Replacing either with the HEAD's clean view/zeros
        # changes known regions, even though the generated x_sigma is exact.
        tail_source = boundary_latent if mask is None else av_latent["samples"]
        tail_noise = comfy.sample.prepare_empty_noise(boundary_latent) if mask is None else noise

        def high_callback(index, prediction, state, count):
            observed_callbacks["HIGH"].append({
                "local_index": index, "absolute_interval": split_interval + index, "local_total": count,
            })
            if callback is not None:
                callback(split_interval + index, prediction, state, 8)

        output = comfy.sample.sample_custom(
            branch_high, tail_noise, cfg, sampler_high, schedule_high,
            positive_high, negative_high, tail_source, noise_mask=mask, callback=high_callback,
            disable_pbar=True, seed=seed)
        stage_times["high_seconds"] = time.perf_counter() - high_started
    finally:
        _restore_stage_objects(branch_high)
    if not all(bool(torch.isfinite(part).all()) for part in output.unbind()):
        raise ValueError("HyperFlow final latent contains NaN/Inf")
    report = {
        "schema": "t8.minimax_h3.hyperflow.continuous_split.v1",
        "status": "sampled_quality_unverified",
        "recipe": "hyperflow8_continuous_split_exp_v1",
        "source_sha256": low_binding.sha256,
        "low_owner": low_binding.owner,
        "high_owner": high_binding.owner,
        "shared_base_identity": low_binding.model_identity,
        "absolute_intervals": [[0, split_interval], [split_interval, 8]],
        "stage_nfe": [split_interval, 8 - split_interval],
        "total_nfe": 8,
        "observed_sampler_callbacks": observed_callbacks,
        "nfe_evidence": {
            "legacy_nfe_fields": "declared_trained_intervals",
            "callback_source": "actual_Core_sample_custom_callback",
            "actual_DiT_forward_count_qualified": False,
            "scope": "Sampler callbacks do not count extra forwards inside user MODEL or CFG wrappers",
        },
        "video_boundary_sigma": plan_low.video_sigmas[split_interval],
        "audio_boundary_sigma": plan_low.audio_sigmas[split_interval],
        "noise_handoff": "direct_captured_x_sigma_no_new_noise_no_audio_rebase",
        "resolution_handoff": "none_continuous_validation_not_low_high_upscale",
        "stage_seconds": stage_times,
        "total_seconds": time.perf_counter() - started,
    }
    result = dict(av_latent)
    result["samples"] = output
    return result, json.dumps(report, ensure_ascii=False, allow_nan=False)
