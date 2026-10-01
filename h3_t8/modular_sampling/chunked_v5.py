"""Explicit global learned lift and joint AV refine windows for Chunked v5.

The released all-in-one executor remains untouched. These typed live objects
are not portable cache receipts; only the final window covers the full AV.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import comfy.model_management
import comfy.nested_tensor
import torch

from .. import chunked_two_pass_parity as parity
from .. import chunked_two_pass_upscale_advanced as legacy
from ..sampling import rebind_dual_clock_sampler
from .chunked_source import _plan_sha
from .results import _input_identity


@dataclass(frozen=True)
class StandardLift:
    plan_sha256: str
    source_identity: dict[str, Any]
    lifted_identity: dict[str, Any]
    source_latent: dict
    lifted_latent: dict
    segments: tuple[tuple[int, int, int, int], ...]
    audio_bounds: tuple[tuple[int, int], ...]
    upscale_report: dict


@dataclass(frozen=True)
class StandardPrepared:
    lift: StandardLift
    video_noise: torch.Tensor
    audio_noise: torch.Tensor
    noise_report: dict
    noise_seed: int | None


@dataclass(frozen=True)
class StandardWindowResult:
    plan_sha256: str
    source_identity: dict[str, Any]
    lifted_identity: dict[str, Any]
    index: int
    count: int
    output_latent: dict
    output_identity: dict[str, Any]


@dataclass(frozen=True)
class StandardWindowInput:
    piece: dict
    high_video: torch.Tensor
    high_audio: torch.Tensor
    start: int
    stop: int
    start_frame: int
    end_frame: int
    audio_start: int
    audio_stop: int
    video_overlap: int
    audio_overlap: int


def _source_contract(source, plan):
    if type(plan) is not dict or plan.get("schema") != parity.SCHEMA:
        raise ValueError("Expected the v5 standard joint 4+4 Chunked Plan")
    checked = parity.standard_plan(plan, json.dumps(plan.get("parity_report")))
    if not isinstance(source, dict):
        raise ValueError("Connect the partial4 denoised_output AV LATENT")
    for key in source:
        if key not in {"samples", "noise_mask", "batch_index", "type"}:
            raise ValueError(f"Unsupported partial latent metadata for temporal slicing: {key}")
    video, audio = parity._parts(source.get("samples"), "partial4 denoised input")
    frames = legacy.frames_for_tokens(video.shape[2])
    segments, _ = legacy.compute_temporal_segments(
        video.shape[2], plan["temporal_chunk_frames"], plan["temporal_overlap_frames"],
    )
    if len(segments) > 1 and plan["temporal_overlap_frames"] == 0:
        raise ValueError("Standard temporal 4+4 requires nonzero completed AV context overlap")
    bounds = tuple(parity._audio_bounds(sf, ef, frames, audio.shape[-1])
                   for _, sf, _, ef in segments)
    return checked, video, audio, tuple(segments), bounds


def lift_standard_joint(source, plan):
    """Run the original learned network once on the complete partial4 AV."""
    _checked, video, audio, segments, bounds = _source_contract(source, plan)
    comfy.model_management.throw_exception_if_processing_interrupted()
    lifted, width, height, report_json = legacy.learned_upscale_h3_av_latent(
        source, plan["model_name"], "target_dimensions", 2.0, 1.0,
        plan["target_width"], plan["target_height"], "honor_dimensions_exp", 2.0,
        plan["precision"], plan["release_policy"],
    )
    high_video, high_audio = parity._parts(lifted["samples"], "upscaled partial4 input")
    target = (plan["target_height"] // 16, plan["target_width"] // 16)
    if (tuple(high_video.shape[2:]) != (video.shape[2], *target)
            or not torch.equal(high_audio, audio)):
        raise RuntimeError("Learned upscaler changed time/audio or returned the wrong target canvas")
    receipt = StandardLift(_plan_sha(plan), _input_identity(source), _input_identity(lifted),
                           source, lifted, segments, bounds, json.loads(report_json))
    report = {"schema": "t8.modular-sampling.chunked-v5-lift.v1",
              "plan_sha256": receipt.plan_sha256, "segment_count": len(segments),
              "width": width, "height": height, "upscale": receipt.upscale_report,
              "sampled": False, "portable_stage_result": False}
    return lifted, receipt, json.dumps(report, ensure_ascii=False, sort_keys=True)


def _check_lift(source, lifted, receipt, plan):
    if type(receipt) is not StandardLift:
        raise ValueError("Expected the v5 global learned-lift receipt")
    if (_plan_sha(plan) != receipt.plan_sha256
            or _input_identity(source) != receipt.source_identity
            or _input_identity(lifted) != receipt.lifted_identity
            or source is not receipt.source_latent or lifted is not receipt.lifted_latent):
        raise ValueError("v5 plan, source or global lifted AV identity changed")
    _source_contract(source, plan)


def prepare_standard_joint(source, lifted, receipt, plan, noise):
    """Draw the original one-shot full-target video+audio noise after global lift."""
    _check_lift(source, lifted, receipt, plan)
    video, audio = parity._parts(source["samples"], "partial4 denoised input")
    video_noise, audio_noise, noise_report = legacy._build_global_target_av_noise(
        noise, source, video, audio, plan,
    )
    prepared = StandardPrepared(receipt, video_noise, audio_noise, noise_report,
                                getattr(noise, "seed", None))
    report = {"schema": "t8.modular-sampling.chunked-v5-prepare.v1",
              "plan_sha256": receipt.plan_sha256, "global_noise": noise_report,
              "sampled": False, "portable_stage_result": False}
    return prepared, json.dumps(report, ensure_ascii=False, sort_keys=True)


def _validate_sampling(sampler, sigmas, report):
    function = getattr(sampler, "sampler_function", None)
    if getattr(function, "__name__", "") != "sample_minimax_h3_dual_clock_euler":
        raise ValueError("Standard joint temporal 4+4 requires the T8 dual-clock Euler sampler")
    for clock in ("video", "audio"):
        if getattr(function, f"_minimax_h3_shift_{clock}", None) != report.get(f"shift_{clock}"):
            raise ValueError("Sampler clocks do not match the connected parity report")
    expected = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    actual = torch.as_tensor(sigmas).detach().to(device="cpu", dtype=torch.float32)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-6, rtol=0):
        raise ValueError("Connect parity refine_sigmas unchanged: exactly four standard intervals")
    return actual


def _window_input(source, lifted, prepared, plan, index, previous):
    """Build the exact joint AV/mask passed to the legacy sampler, without sampling."""
    if type(prepared) is not StandardPrepared:
        raise ValueError("Expected v5 global PASS2 preparation")
    receipt = prepared.lift
    _check_lift(source, lifted, receipt, plan)
    if type(index) is not int or not 0 <= index < len(receipt.segments):
        raise ValueError("v5 PASS2 window index is outside the source timeline")
    if index == 0:
        if previous is not None:
            raise ValueError("First v5 PASS2 window cannot have a previous result")
        published_video = published_audio = None
    else:
        if (type(previous) is not StandardWindowResult or previous.index != index - 1
                or previous.count != len(receipt.segments)
                or previous.plan_sha256 != receipt.plan_sha256
                or previous.source_identity != receipt.source_identity
                or previous.lifted_identity != receipt.lifted_identity
                or _input_identity(previous.output_latent) != previous.output_identity):
            raise ValueError("v5 previous PASS2 result is missing, stale or out of order")
        published_video, published_audio = parity._parts(
            previous.output_latent["samples"], "previous joint PASS2 output",
        )
    high_video, high_audio = parity._parts(lifted["samples"], "upscaled partial4 input")
    source_video, source_audio = parity._parts(source["samples"], "partial4 denoised input")
    target = (plan["target_height"] // 16, plan["target_width"] // 16)
    if tuple(high_video.shape[2:]) != (source_video.shape[2], *target) or not torch.equal(high_audio, source_audio):
        raise ValueError("v5 global lift changed AV geometry or source audio")
    inherited_video = inherited_audio = None
    if lifted.get("noise_mask") is not None:
        inherited_video, _ = legacy._normalize_inherited_video_mask(
            lifted, high_video, policy="inherit_required",
        )
        _, source_audio_mask = legacy._nested_parts(lifted["noise_mask"], name="inherited AV mask")
        try:
            inherited_audio = torch.broadcast_to(source_audio_mask, high_audio.shape)
        except RuntimeError as error:
            raise ValueError("Inherited audio mask does not match the full audio timeline") from error
        if not torch.isfinite(inherited_audio).all() or not bool(
                ((inherited_audio >= 0) & (inherited_audio <= 1)).all()):
            raise ValueError("Inherited audio mask must be finite and within [0,1]")
    start, sf, stop, ef = receipt.segments[index]
    audio_start, audio_stop = receipt.audio_bounds[index]
    video = high_video[:, :, start:stop].clone()
    audio = high_audio[..., audio_start:audio_stop].clone()
    video_overlap = 0 if published_video is None else min(published_video.shape[2] - start, video.shape[2])
    audio_overlap = 0 if published_audio is None else min(published_audio.shape[-1] - audio_start, audio.shape[-1])
    if min(video_overlap, audio_overlap) < 0:
        raise ValueError("Temporal chunk plan left a gap in the AV timeline")
    video_mask = torch.ones((1, 1, video.shape[2], *target), dtype=video.dtype, device=video.device)
    audio_mask = torch.ones_like(audio)
    if inherited_video is not None:
        video_mask *= inherited_video[:, :, start:stop].to(video_mask)
    if inherited_audio is not None:
        audio_mask *= inherited_audio[..., audio_start:audio_stop].to(audio_mask)
    if video_overlap:
        video[:, :, :video_overlap] = published_video[:, :, start:start + video_overlap]
        video_mask[:, :, :video_overlap] = 0
    if audio_overlap:
        audio[..., :audio_overlap] = published_audio[..., audio_start:audio_start + audio_overlap]
        audio_mask[..., :audio_overlap] = 0
    piece = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
             "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask))}
    return StandardWindowInput(piece, high_video, high_audio, start, stop, sf, ef,
                               audio_start, audio_stop, video_overlap, audio_overlap)


def sample_standard_window(model, positive, source, lifted, prepared, plan, noise,
                           sampler, sigmas, index, previous=None, negative=None, cfg=1.0):
    """Run exactly one remaining4 joint AV window, preserving completed overlap."""
    window = _window_input(source, lifted, prepared, plan, index, previous)
    receipt = prepared.lift
    if (prepared.noise_seed is not None
            and getattr(noise, "seed", None) != prepared.noise_seed):
        raise ValueError("v5 PASS2 NOISE seed differs from global preparation")
    actual = _validate_sampling(sampler, sigmas, plan["parity_report"])
    if not positive:
        raise ValueError("Connect HIGH conditioning with the same prompt/media/timeline")
    target = (plan["target_height"] // 16, plan["target_width"] // 16)
    for _, metadata in positive:
        for keyframe in metadata.get("minimax_keyframes", []):
            image = keyframe.get("latent")
            if image is not None and tuple(image.shape[-2:]) != target:
                raise ValueError("HIGH conditioning keyframes must match the target canvas")
    comfy.model_management.throw_exception_if_processing_interrupted()
    if hasattr(model, "get_attachment"):
        from .chunked_v5_effects import assert_v5_eav_binding
        from .chunked_v5_relay import assert_v5_relay_binding
        assert_v5_eav_binding(model, window, prepared, index, sigmas)
        assert_v5_relay_binding(model, positive, window, prepared, index, sigmas)
    bound_sampler = rebind_dual_clock_sampler(model, window.piece, sampler)
    local_positive = legacy.reanchor_conditioning(
        positive, window.start_frame, window.end_frame, target,
    )
    local_negative = None if negative is None else legacy.reanchor_conditioning(
        negative, window.start_frame, window.end_frame, target,
    )
    prepared_noise = comfy.nested_tensor.NestedTensor((
        prepared.video_noise[:, :, window.start:window.stop].contiguous(),
        prepared.audio_noise[..., window.audio_start:window.audio_stop].contiguous(),
    ))
    sampled = legacy.sample_piece(window.piece, local_positive, model, noise, bound_sampler,
                                  sigmas, local_negative, cfg, prepared_noise=prepared_noise)
    result_video, result_audio = parity._parts(sampled, "joint refine4 output")
    video, audio = window.piece["samples"].unbind()
    video_mask, audio_mask = window.piece["noise_mask"].unbind()
    if result_video.shape != video.shape or result_audio.shape != audio.shape:
        raise RuntimeError("Refine4 sampler changed AV geometry")
    result_video, video_roundoff = parity._restore_zero_mask(result_video, video, video_mask)
    result_audio, audio_roundoff = parity._restore_zero_mask(result_audio, audio, audio_mask)
    if index:
        published_video, published_audio = parity._parts(
            previous.output_latent["samples"], "previous joint PASS2 output",
        )
    else:
        published_video = published_audio = None
    published_video = parity._append_exact(
        published_video, result_video, window.start, 2, window.video_overlap,
    )
    published_audio = parity._append_exact(
        published_audio, result_audio, window.audio_start, -1, window.audio_overlap,
    )
    if index == len(receipt.segments) - 1 and (
            published_video.shape != window.high_video.shape or published_audio.shape != window.high_audio.shape):
        raise RuntimeError("Published refined AV does not cover the full source timeline")
    output = {key: value for key, value in source.items() if key in {"batch_index", "type"}}
    output["samples"] = comfy.nested_tensor.NestedTensor((published_video, published_audio))
    result = StandardWindowResult(receipt.plan_sha256, receipt.source_identity,
                                  receipt.lifted_identity, index, len(receipt.segments),
                                  output, _input_identity(output))
    report = {"schema": "t8.modular-sampling.chunked-v5-window.v1",
              "segment_index": index, "segment_count": len(receipt.segments),
              "completed": index + 1 == len(receipt.segments),
              "video_tokens": [window.start, window.stop],
              "pixel_frames": [window.start_frame, window.end_frame],
              "audio_tokens": [window.audio_start, window.audio_stop],
              "refine_nfe": len(actual) - 1,
              "locked_video_overlap_tokens": window.video_overlap,
              "locked_audio_overlap_tokens": window.audio_overlap,
              "video_read_only_roundoff": video_roundoff,
              "audio_read_only_roundoff": audio_roundoff,
              "audio_output": "refined_joint_audio", "portable_stage_result": False}
    return output, result, json.dumps(report, ensure_ascii=False, sort_keys=True)
