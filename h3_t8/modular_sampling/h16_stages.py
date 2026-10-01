"""Explicit H16-3 PASS2 windows and its two audio-delivery policies.

The old all-in-one node remains the compatibility path. Captured audio is
scoped to this sampler call; unlike the old wrapper, no module-global sampler
function is replaced while another graph may be executing.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import comfy.nested_tensor
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import nodes_h16_chunked_pass2 as h16
from .chunked_source import ChunkedSourceSegment
from .chunked_stages import ChunkedPass2Context, ChunkedPass2Result, _check_segment, sample_chunked_pass2
from .results import _input_identity


AUDIO_OUTPUTS = ("preserve_first_pass", "refined_exp")


@dataclass(frozen=True)
class H16Pass2Result:
    core_result: ChunkedPass2Result
    audio_output: str
    audio_chunks: tuple
    audio_chunk_identities: tuple
    output_latent: dict
    output_identity: dict


def build_h16_plan(latent, temporal_strategy="guarded_overlap_exp",
                   temporal_chunk_frames=34, temporal_overlap_frames=17,
                   anchor_strength=0.999,
                   model_name=h16._DEFAULT_UPSCALER):
    video, _audio = h16._nested_av_parts(latent)
    width = int(video.shape[-1]) * h16._VAE_DOWNSAMPLE
    height = int(video.shape[-2]) * h16._VAE_DOWNSAMPLE
    plan, report = legacy.build_chunked_two_pass_masked_low_sigma_plan(
        model_name=model_name, target_width=width, target_height=height,
        temporal_chunk_frames=int(temporal_chunk_frames),
        temporal_overlap_frames=int(temporal_overlap_frames),
        anchor_strength=float(anchor_strength),
        tile_width=width, tile_height=height, spatial_overlap=0, spatial_fade=0,
        minimum_tile_size=256, overlap_blend="smoothstep", precision="fp16",
        release_policy="offload_after", spatial_strategy="full_frame_safe",
        temporal_strategy=temporal_strategy,
        second_pass_audio_policy="joint_av_preserve_input",
        video_mask_policy="inherit_if_present_else_generate_all",
    )
    return plan, report


def _verify_previous(previous, spec, audio_output):
    if spec.index == 0:
        if previous is not None:
            raise ValueError("First H16 PASS2 window cannot have a previous result")
        return None, ()
    if (type(previous) is not H16Pass2Result or previous.audio_output != audio_output
            or previous.core_result.index != spec.index - 1
            or previous.core_result.plan_sha256 != spec.plan_sha256
            or previous.core_result.source_identity != spec.source_identity
            or _input_identity(previous.output_latent) != previous.output_identity
            or len(previous.audio_chunks) != len(previous.audio_chunk_identities)
            or any(_input_identity(value) != expected for value, expected in zip(
                previous.audio_chunks, previous.audio_chunk_identities, strict=True))):
        raise ValueError("H16 previous window is stale, changed or has a different audio policy")
    return previous.core_result, previous.audio_chunks


def h16_window_piece(source_segment, lifted_segment, spec, context, plan,
                     previous=None, audio_output="preserve_first_pass"):
    """Reconstruct the single AV piece actually sent to this H16 window sampler.

    The returned mask includes both the old inherited mask and the guarded
    temporal takeover. It is a live binding contract, not a cache receipt.
    """
    _check_segment(source_segment, spec, context, plan)
    if (plan.get("schema") != legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4
            or plan.get("spatial_strategy") != "full_frame_safe"
            or plan.get("second_pass_audio_policy") != "joint_av_preserve_input"):
        raise ValueError("H16 needs the v4 full-frame joint-AV plan")
    if audio_output not in AUDIO_OUTPUTS:
        raise ValueError("Unknown H16 audio output policy")
    prior_core, _ = _verify_previous(previous, spec, audio_output)
    source_video, source_audio = legacy._nested_parts(source_segment["samples"], name="H16 source segment")
    lifted_video, _ = legacy._nested_parts(lifted_segment["samples"], name="H16 lifted segment")
    if (tuple(lifted_video.shape[:3]) != tuple(source_video.shape[:3])
            or tuple(lifted_video.shape[-2:]) != (
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE,
            )):
        raise ValueError("H16 lifted segment has unexpected video geometry")
    inherited = None
    if context.target_inherited_video_mask is not None:
        inherited = context.target_inherited_video_mask[:, :, spec.start_token:spec.end_token]
        lifted_mask = lifted_segment.get("noise_mask")
        if lifted_mask is None:
            raise ValueError("H16 learned lift dropped inherited noise_mask")
        lifted_video_mask, _ = legacy._nested_parts(lifted_mask, name="H16 lifted noise_mask")
        if (tuple(lifted_video_mask.shape) != tuple(inherited.shape)
                or not torch.equal(lifted_video_mask.to(inherited), inherited)):
            raise ValueError("H16 learned lift changed inherited noise_mask")
    temporal_mask = None
    locked = transition = 0
    if prior_core is not None and plan.get("temporal_merge_policy") == legacy.TEMPORAL_OWNERSHIP_POLICY:
        accumulated, original_audio = legacy._nested_parts(
            prior_core.output_latent["samples"], name="H16 previous PASS2",
        )
        if original_audio is not context.original_audio:
            raise ValueError("H16 previous window lost original audio identity")
        locked = min(max(0, int(accumulated.shape[2]) - spec.start_token), int(lifted_video.shape[2]))
        if locked:
            lifted_video = lifted_video.clone()
            lifted_video[:, :, :locked] = accumulated[:, :, spec.start_token:spec.start_token + locked]
            temporal_mask, locked, transition = legacy._temporal_overlap_mask(
                int(lifted_video.shape[2]), locked,
                dtype=lifted_video.dtype, device=lifted_video.device,
            )
    # _spatial_resample uses exactly one full-frame tile for this H16 plan.
    height, width = lifted_video.shape[-2:]
    video_mask = legacy.spatial_fade_mask(
        height, width, 0, 0, False, False, 0, 0,
    ).to(device=lifted_video.device)[None, None, None]
    if temporal_mask is not None:
        video_mask = video_mask * temporal_mask.to(device=lifted_video.device, dtype=video_mask.dtype)
    if inherited is not None:
        video_mask = video_mask * inherited.to(device=lifted_video.device, dtype=video_mask.dtype)
    piece = {
        "samples": comfy.nested_tensor.NestedTensor((lifted_video.clone(), source_audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, torch.ones_like(source_audio))),
    }
    return piece, locked, transition


def sample_h16_pass2(model, positive, source_segment, lifted_segment, spec,
                     context, plan, noise, sampler, sigmas, previous=None,
                     negative=None, cfg=1.0, audio_output="preserve_first_pass"):
    """Run exactly one v4 full-frame window; merge optional audio at the last."""
    if type(spec) is not ChunkedSourceSegment or type(context) is not ChunkedPass2Context:
        raise ValueError("H16 needs the explicit source segment and PASS2 context")
    if (type(plan) is not dict or plan.get("schema") != legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4
            or plan.get("spatial_strategy") != "full_frame_safe"
            or plan.get("second_pass_audio_policy") != "joint_av_preserve_input"):
        raise ValueError("H16 needs the v4 full-frame joint-AV plan")
    if audio_output not in AUDIO_OUTPUTS:
        raise ValueError("Unknown H16 audio output policy")
    prior_core, prior_audio = _verify_previous(previous, spec, audio_output)
    if hasattr(model, "get_attachment"):
        from .h16_relay import assert_h16_relay_binding
        from .h16_effects import assert_h16_eav_binding
        assert_h16_relay_binding(model, positive, source_segment, lifted_segment,
                                 spec, context, plan, previous, sigmas, audio_output)
        assert_h16_eav_binding(model, source_segment, lifted_segment, spec, context,
                               plan, previous, sigmas, audio_output)
    captured = [] if audio_output == "refined_exp" else None
    base_output, core_result, report_json = sample_chunked_pass2(
        model, positive, source_segment, lifted_segment, spec, context, plan,
        noise, sampler, sigmas, prior_core, negative, cfg,
        captured_audio=captured,
    )
    report = json.loads(report_json)
    chunks = prior_audio + tuple(captured or ())
    output = base_output
    report["audio_output"] = audio_output
    report["captured_audio_chunks"] = len(chunks)
    if audio_output == "refined_exp" and spec.index + 1 == spec.count:
        source_video, _ = legacy._nested_parts(context.source_latent["samples"], name="H16 source")
        segments = h16._executor_segments(source_video, plan)
        try:
            refined, count = h16._merge_audio_segments(
                context.original_audio, segments, chunks, legacy.FRAME_RESCALE,
            )
            video, _ = legacy._nested_parts(base_output["samples"], name="H16 PASS2 output")
            output = dict(base_output)
            output["samples"] = comfy.nested_tensor.NestedTensor((video, refined))
            report["refined_audio_chunks"] = count
            report["audio_merge"] = "absolute_frame_rescale_crossfade_energy_gate"
        except Exception as error:
            # Preserve the old H16 policy: only a merge failure falls back;
            # a failed sampling window itself still propagates its exception.
            report["audio_merge"] = "fallback_preserve_first_pass"
            report["audio_merge_error"] = f"{type(error).__name__}: {error}"
    elif audio_output == "preserve_first_pass":
        report["audio_merge"] = "preserve_first_pass"
    result = H16Pass2Result(core_result, audio_output, chunks,
                           tuple(_input_identity(value) for value in chunks),
                           output, _input_identity(output))
    report["portable_stage_result"] = False
    return output, result, core_result, json.dumps(report, ensure_ascii=False, sort_keys=True)
