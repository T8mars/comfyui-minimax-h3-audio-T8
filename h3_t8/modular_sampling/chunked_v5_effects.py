"""Bind external Stage EAV to one exact joint-AV standard4+4 window.

This owner is intentionally live-process-only. The sampled window is rebuilt
from the same pure preparation as PASS2, including locked video/audio overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import torch

from .. import prompt_relay_advanced as relay
from ..sampling import nested_av_parts
from .chunked_v5 import StandardPrepared, StandardWindowResult, _window_input
from .contracts import StageContext
from .eav import EAVConfig, KEY as EAV_KEY, StageEAVRuntime, apply_stage_eav
from .results import _input_identity


RECIPE = "chunked.v5.joint-av-window.v1"
KEY = "t8_modular_chunked_v5_eav_v1"


@dataclass(frozen=True)
class V5EAVOwner:
    context: StageContext
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    piece_identity: dict
    window_index: int
    sampling_object: object
    mode: str


def bind_v5_eav(model, sigmas, source, lifted, prepared, plan, window_index,
                previous_result, config, relay_positive=None):
    """Install an external config on exactly one separately wired PASS2 MODEL."""
    if type(config) is not EAVConfig:
        raise TypeError("Use the external Stage EAV Config node")
    window = _window_input(source, lifted, prepared, plan, window_index, previous_result)
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or len(sigmas) < 2
            or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:])):
        raise ValueError("v5 Stage EAV needs finite descending PASS2 SIGMAS")
    from .chunked_v5 import parity
    expected = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    if sigmas.shape != expected.shape or not torch.allclose(
            sigmas.detach().to(device="cpu", dtype=torch.float32), expected, atol=1e-6, rtol=0):
        raise ValueError("v5 Stage EAV requires the unchanged standard remaining4 SIGMAS")
    if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        from .chunked_v5_relay import assert_v5_relay_binding
        if relay_positive is None:
            raise ValueError("v5 Relay+EAV needs the matching projected CONDITIONING")
        assert_v5_relay_binding(model, relay_positive, window, prepared, window_index, sigmas)
    elif relay_positive is not None:
        raise ValueError("v5 EAV received Relay CONDITIONING without an authenticated projected MODEL")
    options = model.model_options["transformer_options"]
    shift_video = options.get("minimax_h3_sigma_shift_video")
    shift_audio = options.get("minimax_h3_sigma_shift_audio")
    report = plan["parity_report"]
    if ((shift_video, shift_audio) != (report["shift_video"], report["shift_audio"])):
        raise ValueError("v5 Stage EAV MODEL AV clocks differ from the parity Plan")
    video, audio = nested_av_parts(window.piece)
    profile = json.dumps({"plan_sha256": prepared.lift.plan_sha256,
                          "window_index": window_index,
                          "sigma_dtype": str(sigmas.dtype)}, sort_keys=True)
    context = StageContext(
        recipe=RECIPE, stage=f"pass2_window_{window_index}", profile=profile,
        start=0, end=len(sigmas) - 1,
        trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=float(shift_video), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="global_learned_av_with_locked_video_audio_overlap",
        output_semantics="one_refined_joint_av_window_merged_to_cumulative_av",
        denoised_semantics="terminal_remaining4_joint_av_window_prediction",
    )
    selected = model.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("v5 Stage EAV is already bound to this MODEL branch")
    selected.set_attachments(KEY, V5EAVOwner(
        context, prepared.lift.plan_sha256, prepared.lift.source_identity,
        prepared.lift.lifted_identity, _input_identity(window.piece), window_index,
        model.get_model_object("model_sampling"), config.mode,
    ))
    return (*apply_stage_eav(selected, sigmas, window.piece, context, config), context)


def validate_stage(model, sigmas, av_latent, context):
    owner = model.get_attachment(KEY)
    if (type(context) is not StageContext or context.recipe != RECIPE
            or type(owner) is not V5EAVOwner or owner.context != context):
        raise ValueError("v5 Stage EAV needs its exact bound MODEL and context")
    if model.get_model_object("model_sampling") is not owner.sampling_object:
        raise ValueError("v5 Stage EAV sampling object changed")
    if _input_identity(av_latent) != owner.piece_identity:
        raise ValueError("v5 Stage EAV joint AV window or mask changed")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or tuple(float(value) for value in sigmas.tolist()) != context.trajectory_sigmas):
        raise ValueError("v5 Stage EAV SIGMAS differ from its window")
    profile = json.loads(context.profile)
    if profile != {"plan_sha256": owner.plan_sha256,
                   "window_index": owner.window_index,
                   "sigma_dtype": str(sigmas.dtype)}:
        raise ValueError("v5 Stage EAV plan/window profile changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"),
            options.get("minimax_h3_sigma_shift_audio")) != (
            context.video_shift, context.audio_shift):
        raise ValueError("v5 Stage EAV AV clocks changed")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("v5 Stage EAV AV layout differs from its window")
    return video, audio, SimpleNamespace(runtime=None, profile=context.profile)


def assert_v5_eav_binding(model, window, prepared, index, sigmas):
    owner = model.get_attachment(KEY)
    if owner is None:
        return
    if (type(owner) is not V5EAVOwner or type(prepared) is not StandardPrepared
            or owner.window_index != index or owner.plan_sha256 != prepared.lift.plan_sha256
            or owner.source_identity != prepared.lift.source_identity
            or owner.lifted_identity != prepared.lift.lifted_identity
            or owner.piece_identity != _input_identity(window.piece)
            or owner.context.trajectory_sigmas != tuple(float(value) for value in sigmas.tolist())
            or model.get_model_object("model_sampling") is not owner.sampling_object):
        raise ValueError("v5 PASS2 Stage EAV MODEL is not bound to this exact joint AV window")
    runtime = model.get_attachment(EAV_KEY)
    if owner.mode != "disabled" and (
            type(runtime) is not StageEAVRuntime or runtime.context != owner.context
            or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
        raise ValueError("v5 PASS2 Stage EAV wrapper was removed or replaced")


def audit_v5_eav(result, prepared, runtime):
    if (type(result) is not StandardWindowResult or type(prepared) is not StandardPrepared
            or type(runtime) is not StageEAVRuntime or runtime.context.recipe != RECIPE):
        raise TypeError("v5 Stage EAV audit needs a matching window result/preparation/runtime")
    profile = json.loads(runtime.context.profile)
    if (result.index != profile["window_index"]
            or result.plan_sha256 != prepared.lift.plan_sha256
            or result.source_identity != prepared.lift.source_identity
            or result.lifted_identity != prepared.lift.lifted_identity
            or profile["plan_sha256"] != prepared.lift.plan_sha256
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("v5 Stage EAV result/runtime identity mismatch")
    report = runtime.snapshot()
    if report["status"] == "aborted":
        raise RuntimeError("v5 Stage EAV aborted: " + report["feta"]["aborted"])
    report["chunked_v5_window_index"] = result.index
    report["chunked_v5_plan_sha256"] = result.plan_sha256
    return result.output_latent, json.dumps(report, ensure_ascii=False, indent=2)
