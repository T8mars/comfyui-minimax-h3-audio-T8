"""One explicit SPEED stage-to-stage handoff; never samples either stage.

The legacy whole-chain SPEED runner remains the numerical reference.  A caller
must supply the completed public-flow state and the next stage's rebuilt AV
target.  This module deliberately does not manufacture a learned-upscale or
silently rebuild conditioning.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import comfy.nested_tensor

from ..speed_advanced import (
    SPEED_PLAN_SCHEMA,
    dct_expand_official,
    recover_raw_flow_state,
    reindex_joint_audio_state,
    solve_segment_noise,
)


def _parts(value: Any, name: str):
    if not getattr(value, "is_nested", False):
        raise ValueError(f"{name} must be a native nested H3 AV state")
    parts = tuple(value.unbind())
    if len(parts) != 2:
        raise ValueError(f"{name} must contain video and audio")
    return parts


def transition_speed_stage(
    plan: Mapping[str, Any],
    stage_index: int,
    external_output: Any,
    next_latent: Mapping[str, Any],
    *,
    audio_scale: float,
    noise_scale: float,
    seed: int,
    dct_chunk_size: int = 64,
) -> tuple[Any, dict[str, Any]]:
    """Return the exact solved NOISE for one next SPEED segment.

    ``external_output`` is Guider_Basic.sample's completed stage state, not its
    x0 estimate.  ``next_latent`` is the already rebuilt target canvas.  The
    caller must bind these to a completed stage receipt before exposing a
    persistent resume path; this mathematical adapter alone is not a receipt.
    """
    if plan.get("schema") != SPEED_PLAN_SCHEMA:
        raise ValueError("Expected a T8 H3 SPEED plan")
    if type(stage_index) is not int or not 0 <= stage_index < len(plan["transitions"]):
        raise ValueError("SPEED transition stage index is out of range")
    transitions = plan["transitions"]
    stages = plan["stages"]
    segments = plan["segments"]
    if len(stages) != len(segments) or len(transitions) != len(stages) - 1:
        raise ValueError("SPEED plan stage/transition counts disagree")
    transition = transitions[stage_index]
    current_segment = segments[stage_index]
    next_segment = segments[stage_index + 1]
    sigma_from = float(transition["sigma"])
    sigma_to = float(transition["aligned_sigma"])
    if (
        transition["from_stage"] != stage_index
        or transition["to_stage"] != stage_index + 1
        or current_segment["stage_index"] != stage_index
        or next_segment["stage_index"] != stage_index + 1
        or not math.isclose(float(current_segment["sigmas"][-1]), sigma_from, abs_tol=1e-7)
        or not math.isclose(float(next_segment["sigmas"][0]), sigma_to, abs_tol=1e-7)
    ):
        raise ValueError("SPEED transition is not bound to adjacent segment endpoints")
    if not all(math.isfinite(value) and value > 0 for value in (audio_scale, noise_scale)):
        raise ValueError("SPEED audio/noise scales must be finite and positive")
    video, audio = _parts(external_output, "Completed SPEED output")
    target_video, target_audio = _parts(next_latent.get("samples"), "Next SPEED target")
    old_shape = stages[stage_index]
    new_shape = stages[stage_index + 1]
    if tuple(video.shape[-2:]) != (old_shape["latent_height"], old_shape["latent_width"]):
        raise ValueError("Completed SPEED video does not match its planned canvas")
    if tuple(target_video.shape[-2:]) != (new_shape["latent_height"], new_shape["latent_width"]):
        raise ValueError("Next SPEED target does not match its planned canvas")
    if video.shape[:-2] != target_video.shape[:-2] or audio.shape != target_audio.shape:
        raise ValueError("SPEED transition changed AV batch, channels, time, or audio layout")

    raw = recover_raw_flow_state(external_output, sigma=sigma_from, audio_scale=audio_scale)
    raw_video, raw_audio = _parts(raw, "Recovered SPEED state")
    expanded_video, expanded_sigma, dct_report = dct_expand_official(
        raw_video,
        new_shape["latent_height"],
        new_shape["latent_width"],
        sigma=sigma_from,
        ratio=float(transition["ratio"]),
        seed=int(seed) + (stage_index + 1) * 10_000,
        chunk_size=int(dct_chunk_size),
    )
    if not math.isclose(expanded_sigma, sigma_to, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("SPEED DCT-aligned sigma disagrees with the plan")
    next_video = target_video.to(device=raw_video.device, dtype=raw_video.dtype)
    next_audio = target_audio.to(device=raw_audio.device, dtype=raw_audio.dtype)
    aligned_audio = reindex_joint_audio_state(
        raw_audio, next_audio * audio_scale, sigma_from=sigma_from, sigma_to=sigma_to
    )
    desired = comfy.nested_tensor.NestedTensor((expanded_video, aligned_audio))
    target = comfy.nested_tensor.NestedTensor((next_video, next_audio))
    noise = solve_segment_noise(
        desired, target, sigma=sigma_to, audio_scale=audio_scale, noise_scale=noise_scale
    )
    return noise, {
        "schema": "t8.modular-sampling.speed-transition.v1",
        "from_stage": stage_index,
        "to_stage": stage_index + 1,
        "sigma_from": sigma_from,
        "sigma_to": sigma_to,
        "seed": int(seed) + (stage_index + 1) * 10_000,
        "dct": dct_report,
        "audio_transport": "target_anchored_public_flow_reindex_exp",
        "audio_spatial_noise_expansion": False,
        "sampled": False,
        "cache_reuse_authorized": False,
    }
