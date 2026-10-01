"""Bind external Stage EAV to one full-frame Chunked PASS2 segment.

The legacy spatial loop remains untouched. Multi-tile plans are deliberately
rejected: their repeated, shape-varying forwards need a different audit plan.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import torch
import comfy.nested_tensor

from .. import chunked_two_pass_upscale_advanced as legacy
from ..sampling import nested_av_parts
from .chunked_source import ChunkedSourceSegment
from .chunked_stages import ChunkedPass2Result, _check_segment
from .contracts import StageContext
from .eav import StageEAVRuntime, apply_stage_eav
from .results import _input_identity


RECIPE = "chunked.pass2.single-tile.v1"
KEY = "t8_modular_chunked_pass2_eav_v1"


@dataclass(frozen=True)
class ChunkedEAVOwner:
    context: StageContext
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    effect_latent_identity: dict
    segment_index: int
    sampling_object: object


def _single_tile(plan, video):
    if plan.get("spatial_strategy") != "full_frame_safe":
        raise ValueError("Chunked Stage EAV currently requires full_frame_safe (one tile)")
    if tuple(video.shape[-2:]) != (int(plan["target_height"]) // legacy.VAE_DOWNSAMPLE,
                                    int(plan["target_width"]) // legacy.VAE_DOWNSAMPLE):
        raise ValueError("Chunked Stage EAV target geometry differs from lifted video")


def bind_chunked_eav(model, sigmas, source_segment, lifted_segment, spec, pass2_context,
                     plan, config):
    """Build a real stage owner, then install the existing generic EAV adapter."""
    _check_segment(source_segment, spec, pass2_context, plan)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("Chunked Stage EAV needs floating one-dimensional SIGMAS")
    if len(sigmas) < 2 or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:]):
        raise ValueError("Chunked Stage EAV needs finite descending SIGMAS")
    video, audio = nested_av_parts(lifted_segment)
    _single_tile(plan, video)
    if plan.get("temporal_merge_policy") == legacy.TEMPORAL_OWNERSHIP_POLICY and spec.index:
        raise ValueError("Chunked Stage EAV guarded overlap needs its future per-segment mask adapter")
    source_video, source_audio = nested_av_parts(source_segment)
    if tuple(video.shape[:3]) != tuple(source_video.shape[:3]) or tuple(audio.shape) != tuple(source_audio.shape):
        raise ValueError("Chunked Stage EAV lifted AV geometry differs from source segment")
    options = model.model_options["transformer_options"]
    shift_video = options.get("minimax_h3_sigma_shift_video")
    shift_audio = options.get("minimax_h3_sigma_shift_audio")
    if shift_video is None or shift_audio is None:
        raise ValueError("Chunked Stage EAV requires the selected MODEL's native AV clock shifts")
    profile = json.dumps({"plan_sha256": spec.plan_sha256, "segment_index": spec.index,
                          "sigma_dtype": str(sigmas.dtype)}, sort_keys=True)
    context = StageContext(
        recipe=RECIPE, stage=f"pass2_segment_{spec.index}", profile=profile,
        start=0, end=len(sigmas) - 1,
        trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=float(shift_video), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="learned_upscaled_chunk_with_original_audio",
        output_semantics="sampled_chunk_merged_to_cumulative_video_original_full_audio",
        denoised_semantics="terminal_pass2_chunk_prediction",
    )
    selected = model.get_model_object("model_sampling")
    video_mask = torch.ones((1, 1, video.shape[2], video.shape[3], video.shape[4]),
                            dtype=video.dtype, device=video.device)
    if pass2_context.target_inherited_video_mask is not None:
        video_mask = video_mask * pass2_context.target_inherited_video_mask[
            :, :, spec.start_token:spec.end_token].to(video)
    audio_mask = (torch.ones_like(audio) if plan.get("second_pass_audio_policy") == "joint_av_preserve_input"
                  else torch.zeros_like(audio))
    effect_latent = dict(lifted_segment)
    effect_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    prepared = model.clone()
    if prepared.get_attachment(KEY) is not None:
        raise ValueError("Chunked Stage EAV is already bound on this MODEL branch")
    prepared.set_attachments(KEY, ChunkedEAVOwner(
        context, spec.plan_sha256, spec.source_identity,
        _input_identity(lifted_segment), _input_identity(effect_latent), spec.index, selected,
    ))
    return (*apply_stage_eav(prepared, sigmas, effect_latent, context, config), context)


def validate_stage(model, sigmas, av_latent, context):
    owner = model.get_attachment(KEY)
    if type(context) is not StageContext or context.recipe != RECIPE or type(owner) is not ChunkedEAVOwner:
        raise ValueError("Chunked Stage EAV needs its exact bound MODEL and context")
    if owner.context != context or model.get_model_object("model_sampling") is not owner.sampling_object:
        raise ValueError("Chunked Stage EAV MODEL owner or sampling object changed")
    if _input_identity(av_latent) != owner.effect_latent_identity:
        raise ValueError("Chunked Stage EAV bound mask/latent mutated after binding")
    profile = json.loads(context.profile)
    if (profile != {"plan_sha256": owner.plan_sha256, "segment_index": owner.segment_index,
                    "sigma_dtype": str(sigmas.dtype)} or context.stage != f"pass2_segment_{owner.segment_index}"):
        raise ValueError("Chunked Stage EAV plan, segment or SIGMAS dtype changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"), options.get("minimax_h3_sigma_shift_audio")) != (
            context.video_shift, context.audio_shift):
        raise ValueError("Chunked Stage EAV AV clocks changed")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or tuple(float(value) for value in sigmas.tolist()) != context.trajectory_sigmas):
        raise ValueError("Chunked Stage EAV SIGMAS differ from this segment")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("Chunked Stage EAV AV layout differs from this segment")
    return video, audio, SimpleNamespace(runtime=None, profile=context.profile)


def assert_effect_binding(model, lifted_segment, spec):
    owner = model.get_attachment(KEY)
    if owner is None:
        return
    if (type(owner) is not ChunkedEAVOwner or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity or owner.segment_index != spec.index
            or _input_identity(lifted_segment) != owner.lifted_identity):
        raise ValueError("Chunked PASS2 EAV MODEL is not bound to this lifted segment")


def audit_chunked_eav(result, spec, runtime):
    """Audit the live segment result, not the differently shaped cumulative AV."""
    if type(result) is not ChunkedPass2Result or type(spec) is not ChunkedSourceSegment:
        raise TypeError("Chunked Stage EAV audit needs the matching segment result and spec")
    if type(runtime) is not StageEAVRuntime or runtime.context.recipe != RECIPE:
        raise TypeError("Chunked Stage EAV audit needs its bound runtime")
    profile = json.loads(runtime.context.profile)
    if (result.index != spec.index or result.plan_sha256 != spec.plan_sha256
            or result.source_identity != spec.source_identity
            or profile["plan_sha256"] != spec.plan_sha256 or profile["segment_index"] != spec.index
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("Chunked Stage EAV result/runtime identity mismatch")
    report = runtime.snapshot()
    if report["status"] == "aborted":
        raise RuntimeError("Chunked Stage EAV aborted: " + report["feta"]["aborted"])
    report["chunked_segment_index"] = spec.index
    report["chunked_plan_sha256"] = spec.plan_sha256
    return result.output_latent, json.dumps(report, ensure_ascii=False, indent=2)
