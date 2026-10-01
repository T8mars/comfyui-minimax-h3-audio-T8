"""Explicit learned lift and one Chunked PASS2 segment, preserving old math.

The legacy all-in-one executor remains the compatibility path. These live
objects are deliberately not portable cache receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import comfy.nested_tensor
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from .chunked_source import ChunkedSourceSegment, SCHEMAS, _plan_sha
from .results import _input_identity


@dataclass(frozen=True)
class ChunkedPass2Context:
    plan_sha256: str
    source_identity: dict[str, Any]
    source_latent: dict
    original_audio: torch.Tensor
    global_video_noise: torch.Tensor | None
    global_audio_noise: torch.Tensor | None
    target_inherited_video_mask: torch.Tensor | None
    inherited_mask_report: dict | None
    global_noise_report: dict | None
    noise_seed: int | None


@dataclass(frozen=True)
class ChunkedPass2Result:
    plan_sha256: str
    source_identity: dict[str, Any]
    index: int
    count: int
    output_latent: dict
    output_identity: dict[str, Any]


def _check_plan(plan):
    if type(plan) is not dict or plan.get("schema") not in SCHEMAS:
        raise ValueError("Expected a v1-v4 Chunked Two-Pass Plan")


def prepare_chunked_pass2(source_latent, plan, noise):
    """Do once per full source, exactly as the old pre-loop noise/mask path."""
    _check_plan(plan)
    legacy._validate_chunked_source_geometry(plan, source_latent)
    video, audio = legacy._nested_parts(source_latent["samples"], name="source_latent")
    if (video.ndim != 5 or tuple(video.shape[:2]) != (1, 24)
            or audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2)):
        raise ValueError("Expected native batch-one H3 AV source")
    inherited_target = None
    inherited_report = None
    if plan["schema"] == legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4:
        inherited, inherited_report = legacy._normalize_inherited_video_mask(
            source_latent, video, policy=plan.get("video_mask_policy", "inherit_required"),
        )
        if inherited is not None:
            inherited_target = legacy._resize_video_mask_spatial_only(
                inherited, int(plan["target_height"]) // legacy.VAE_DOWNSAMPLE,
                int(plan["target_width"]) // legacy.VAE_DOWNSAMPLE,
            ).contiguous()
            inherited_report = dict(inherited_report)
            inherited_report.update({
                "target_shape": list(inherited_target.shape),
                "target_spatial_policy": "nearest_exact_spatial_only",
            })
    global_video = global_audio = global_report = None
    if plan["schema"] in (legacy.PLAN_SCHEMA_LOW_SIGMA_V3,
                          legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4):
        global_video, global_audio, global_report = legacy._build_global_target_av_noise(
            noise, source_latent, video, audio, plan,
        )
    elif plan["schema"] == legacy.PLAN_SCHEMA_GLOBAL_NOISE_V2:
        global_video, global_report = legacy._build_global_target_video_noise(
            noise, source_latent, video, audio, plan,
        )
    context = ChunkedPass2Context(
        _plan_sha(plan), _input_identity(source_latent), source_latent, audio,
        global_video, global_audio, inherited_target, inherited_report,
        global_report, getattr(noise, "seed", None) if global_video is not None else None,
    )
    report = {"schema": "t8.modular-sampling.chunked-pass2-context.v1",
              "plan_sha256": context.plan_sha256, "global_noise": global_report,
              "inherited_video_mask": inherited_report,
              "sampled": False, "portable_stage_result": False}
    return context, json.dumps(report, ensure_ascii=False, sort_keys=True)


def _check_segment(segment_latent, spec, context, plan):
    _check_plan(plan)
    if type(context) is not ChunkedPass2Context or type(spec) is not ChunkedSourceSegment:
        raise ValueError("Expected Chunked source segment and PASS2 context")
    if (spec.plan_sha256 != _plan_sha(plan) or context.plan_sha256 != spec.plan_sha256
            or spec.source_identity != context.source_identity):
        raise ValueError("Chunked plan/source identity mismatch")
    if _input_identity(context.source_latent) != context.source_identity:
        raise ValueError("Chunked full source mutated after PASS2 preparation")
    if _input_identity(segment_latent) != spec.slice_identity:
        raise ValueError("Chunked source segment mutated after slicing")


def lift_chunked_segment(segment_latent, spec, context, plan):
    """Run exactly the old per-segment learned upscaler, without PASS2."""
    _check_segment(segment_latent, spec, context, plan)
    upscaled, width, height, upscale_report = legacy.learned_upscale_h3_av_latent(
        segment_latent, plan["model_name"], "target_dimensions", 2.0, 1.0,
        int(plan["target_width"]), int(plan["target_height"]),
        "honor_dimensions_exp", 2.0, plan["precision"], plan["release_policy"],
    )
    lifted_video, _ = legacy._nested_parts(upscaled["samples"], name="lifted_segment")
    if (tuple(lifted_video.shape[:3]) != (1, 24, spec.end_token - spec.start_token)
            or tuple(lifted_video.shape[-2:]) != (
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE,
            )):
        raise ValueError("Learned lift returned unexpected Chunked video geometry")
    report = {"schema": "t8.modular-sampling.chunked-learned-lift.v1",
              "segment_index": spec.index, "width": width, "height": height,
              "upscale": json.loads(upscale_report), "sampled": False,
              "portable_stage_result": False}
    return upscaled, json.dumps(report, ensure_ascii=False, sort_keys=True)


def sample_chunked_pass2(model, conditioning, source_segment, lifted_segment, spec,
                         context, plan, noise, sampler, sigmas, previous=None,
                         negative=None, cfg=1.0, *, captured_audio=None):
    """Sample one explicit PASS2 chunk and return cumulative video + original audio."""
    _check_segment(source_segment, spec, context, plan)
    if hasattr(model, "get_attachment"):
        from .chunked_effects import assert_effect_binding
        from .chunked_relay import assert_relay_binding
        from .chunked_v1_relay import assert_v1_local_relay_binding
        assert_effect_binding(model, lifted_segment, spec)
        assert_relay_binding(model, conditioning, sigmas, lifted_segment, spec)
        assert_v1_local_relay_binding(
            model, conditioning, source_segment, lifted_segment,
            spec, context, plan, previous, sigmas,
        )
    if context.noise_seed is not None and getattr(noise, "seed", None) != context.noise_seed:
        raise ValueError("PASS2 NOISE seed differs from global-noise preparation")
    source_video, source_audio = legacy._nested_parts(source_segment["samples"], name="source_segment")
    lifted_video, _lifted_audio = legacy._nested_parts(lifted_segment["samples"], name="lifted_segment")
    if (tuple(lifted_video.shape[:3]) != tuple(source_video.shape[:3])
            or tuple(lifted_video.shape[-2:]) != (
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE,
            )):
        raise ValueError("PASS2 lifted segment has unexpected video geometry")
    inherited_target = None
    if context.target_inherited_video_mask is not None:
        inherited_target = context.target_inherited_video_mask[:, :, spec.start_token:spec.end_token]
        lifted_mask = lifted_segment.get("noise_mask")
        if lifted_mask is None:
            raise ValueError("Learned lift dropped inherited H3 noise_mask")
        lifted_video_mask, _ = legacy._nested_parts(lifted_mask, name="lifted noise_mask")
        if (tuple(lifted_video_mask.shape) != tuple(inherited_target.shape)
                or not torch.equal(lifted_video_mask.to(
                    device=inherited_target.device, dtype=inherited_target.dtype), inherited_target)):
            raise ValueError("Learned lift changed inherited H3 noise_mask")
    accumulated = None
    if spec.index:
        if (type(previous) is not ChunkedPass2Result or previous.index != spec.index - 1
                or previous.count != spec.count or previous.plan_sha256 != spec.plan_sha256
                or previous.source_identity != spec.source_identity
                or _input_identity(previous.output_latent) != previous.output_identity):
            raise ValueError("PASS2 previous segment is missing, stale or out of order")
        accumulated, previous_audio = legacy._nested_parts(
            previous.output_latent["samples"], name="previous_pass2")
        if previous_audio is not context.original_audio:
            raise ValueError("PASS2 previous segment lost original full audio identity")
    elif previous is not None:
        raise ValueError("First PASS2 segment must not have a previous result")
    chunk_conditioning = legacy.reanchor_conditioning(
        conditioning, spec.start_frame, spec.end_frame, tuple(lifted_video.shape[-2:]),
    )
    temporal_mask = None
    locked = transition = 0
    uses_owned_overlap = plan.get("temporal_merge_policy") == legacy.TEMPORAL_OWNERSHIP_POLICY
    if spec.index and accumulated is not None and uses_owned_overlap:
        locked = max(0, int(accumulated.shape[2]) - spec.start_token)
        locked = min(locked, int(lifted_video.shape[2]))
        if locked:
            lifted_video = lifted_video.clone()
            lifted_video[:, :, :locked] = accumulated[:, :, spec.start_token:spec.start_token + locked]
            temporal_mask, locked, transition = legacy._temporal_overlap_mask(
                int(lifted_video.shape[2]), locked, dtype=lifted_video.dtype, device=lifted_video.device,
            )
    elif spec.index and accumulated is not None:
        chunk_conditioning = legacy.anchor_conditioning(
            chunk_conditioning, accumulated, spec.start_frame, plan["anchor_strength"],
        )
    spatial_plan = plan
    if plan.get("spatial_strategy", "independent_tiles_exp") == "full_frame_safe":
        spatial_plan = dict(plan)
        spatial_plan.update({
            "tile_width": int(plan["target_width"]), "tile_height": int(plan["target_height"]),
            "spatial_overlap": 0, "spatial_fade": 0,
            "minimum_tile_size": min(int(plan["target_width"]), int(plan["target_height"])),
        })
    sampled_video, tile_report = legacy._spatial_resample(
        lifted_video, source_audio, chunk_conditioning, spatial_plan, model, noise,
        sampler, sigmas, negative, cfg,
        chunk_noise_video=(context.global_video_noise[:, :, spec.start_token:spec.end_token]
                           if context.global_video_noise is not None else None),
        chunk_noise_audio=(context.global_audio_noise[..., spec.audio_start:spec.audio_end]
                           if context.global_audio_noise is not None else None),
        chunk_temporal_mask=temporal_mask,
        chunk_inherited_video_mask=inherited_target,
        audio_sampling_policy=plan.get("second_pass_audio_policy", "locked_input_audio"),
        rebind_shape_bound_sampler=plan["schema"] in (
            legacy.PLAN_SCHEMA_LOW_SIGMA_V3, legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4),
        **({"captured_audio": captured_audio} if captured_audio is not None else {}),
    )
    if uses_owned_overlap:
        accumulated, merged, merged_transition = legacy._append_video_guarded_overlap(
            accumulated, sampled_video, spec.start_token, locked,
        )
        if merged != locked + transition or merged_transition != transition:
            raise RuntimeError("Chunked temporal overlap profile changed before merge")
    else:
        accumulated = legacy._append_video(accumulated, sampled_video, spec.start_token)
    output = {"samples": comfy.nested_tensor.NestedTensor((accumulated, context.original_audio))}
    result = ChunkedPass2Result(spec.plan_sha256, spec.source_identity, spec.index,
                                spec.count, output, _input_identity(output))
    report = {"schema": "t8.modular-sampling.chunked-pass2-segment.v1",
              "segment_index": spec.index, "segment_count": spec.count,
              "completed": spec.index + 1 == spec.count,
              "locked_overlap_tokens": locked, "transition_overlap_tokens": transition,
              "tile": tile_report, "original_full_audio_preserved": True,
              "portable_stage_result": False}
    return output, result, json.dumps(report, ensure_ascii=False, sort_keys=True)
