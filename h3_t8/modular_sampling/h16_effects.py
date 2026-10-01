"""External EAV bound to one exact guarded-overlap H16 PASS2 window."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import torch

from .. import prompt_relay_advanced as relay
from ..sampling import nested_av_parts
from .contracts import StageContext
from .eav import EAVConfig, KEY as EAV_KEY, StageEAVRuntime, apply_stage_eav
from .h16_stages import H16Pass2Result, h16_window_piece
from .results import _input_identity


RECIPE = "h16.pass2.guarded-window.v1"
KEY = "t8_modular_h16_eav_v1"


@dataclass(frozen=True)
class H16EAVOwner:
    context: StageContext
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    piece_identity: dict
    previous_identity: dict | None
    window_index: int
    audio_output: str
    sampling_object: object
    mode: str


def bind_h16_eav(model, sigmas, source_segment, lifted_segment, spec,
                 pass2_context, plan, previous_result, audio_output, config,
                 relay_positive=None):
    """Bind one separately editable effect without changing legacy H16 nodes."""
    if type(config) is not EAVConfig:
        raise TypeError("Use the external Stage EAV Config node")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or len(sigmas) < 2
            or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:])):
        raise ValueError("H16 Stage EAV needs finite descending PASS2 SIGMAS")
    if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        from .h16_relay import assert_h16_relay_binding
        if relay_positive is None:
            raise ValueError("H16 Relay+EAV needs the matching projected CONDITIONING")
        assert_h16_relay_binding(
            model, relay_positive, source_segment, lifted_segment, spec,
            pass2_context, plan, previous_result, sigmas, audio_output,
        )
    elif relay_positive is not None:
        raise ValueError("H16 EAV received Relay CONDITIONING without a projected MODEL")
    piece, locked, transition = h16_window_piece(
        source_segment, lifted_segment, spec, pass2_context, plan,
        previous_result, audio_output,
    )
    options = model.model_options["transformer_options"]
    shift_video = options.get("minimax_h3_sigma_shift_video")
    shift_audio = options.get("minimax_h3_sigma_shift_audio")
    if shift_video is None or shift_audio is None:
        raise ValueError("H16 Stage EAV needs the selected MODEL's native AV clocks")
    video, audio = nested_av_parts(piece)
    profile = json.dumps({"plan_sha256": spec.plan_sha256,
                          "window_index": spec.index, "audio_output": audio_output,
                          "locked_overlap_tokens": locked,
                          "transition_overlap_tokens": transition,
                          "sigma_dtype": str(sigmas.dtype)}, sort_keys=True)
    context = StageContext(
        recipe=RECIPE, stage=f"pass2_window_{spec.index}", profile=profile,
        start=0, end=len(sigmas) - 1,
        trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=float(shift_video), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="lifted_joint_av_with_inherited_and_guarded_overlap_masks",
        output_semantics="one_h16_window_merged_to_cumulative_video_with_selected_audio_policy",
        denoised_semantics="terminal_h16_pass2_window_prediction",
    )
    selected = model.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("H16 Stage EAV is already bound on this MODEL branch")
    selected.set_attachments(KEY, H16EAVOwner(
        context, spec.plan_sha256, spec.source_identity,
        _input_identity(lifted_segment), _input_identity(piece),
        previous_result.output_identity if previous_result is not None else None,
        spec.index, audio_output, model.get_model_object("model_sampling"), config.mode,
    ))
    return (*apply_stage_eav(selected, sigmas, piece, context, config), context)


def validate_stage(model, sigmas, av_latent, context):
    owner = model.get_attachment(KEY)
    if (type(context) is not StageContext or context.recipe != RECIPE
            or type(owner) is not H16EAVOwner or owner.context != context):
        raise ValueError("H16 Stage EAV needs its exact bound MODEL and context")
    if model.get_model_object("model_sampling") is not owner.sampling_object:
        raise ValueError("H16 Stage EAV sampling object changed")
    if _input_identity(av_latent) != owner.piece_identity:
        raise ValueError("H16 Stage EAV joint AV window or mask changed")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or tuple(float(value) for value in sigmas.tolist()) != context.trajectory_sigmas):
        raise ValueError("H16 Stage EAV SIGMAS differ from this window")
    profile = json.loads(context.profile)
    if (profile["plan_sha256"] != owner.plan_sha256
            or profile["window_index"] != owner.window_index
            or profile["audio_output"] != owner.audio_output
            or profile["sigma_dtype"] != str(sigmas.dtype)):
        raise ValueError("H16 Stage EAV plan, audio policy or SIGMAS dtype changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"),
            options.get("minimax_h3_sigma_shift_audio")) != (
            context.video_shift, context.audio_shift):
        raise ValueError("H16 Stage EAV AV clocks changed")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("H16 Stage EAV AV layout changed")
    return video, audio, SimpleNamespace(runtime=None, profile=context.profile)


def assert_h16_eav_binding(model, source_segment, lifted_segment, spec, context,
                           plan, previous_result, sigmas, audio_output):
    owner = model.get_attachment(KEY)
    if owner is None:
        if (model.get_attachment(EAV_KEY) is not None
                or model.get_wrappers("diffusion_model", EAV_KEY)):
            raise ValueError("H16 PASS2 Stage EAV MODEL lacks its H16 window binding")
        return
    piece, locked, transition = h16_window_piece(
        source_segment, lifted_segment, spec, context, plan,
        previous_result, audio_output,
    )
    profile = json.loads(owner.context.profile) if type(owner) is H16EAVOwner else {}
    if (type(owner) is not H16EAVOwner or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity or owner.window_index != spec.index
            or owner.audio_output != audio_output
            or owner.lifted_identity != _input_identity(lifted_segment)
            or owner.piece_identity != _input_identity(piece)
            or owner.previous_identity != (
                previous_result.output_identity if previous_result is not None else None)
            or owner.context.trajectory_sigmas != tuple(float(value) for value in sigmas.tolist())
            or profile.get("locked_overlap_tokens") != locked
            or profile.get("transition_overlap_tokens") != transition
            or model.get_model_object("model_sampling") is not owner.sampling_object
            or (model.model_options["transformer_options"].get("minimax_h3_sigma_shift_video"),
                model.model_options["transformer_options"].get("minimax_h3_sigma_shift_audio"))
            != (owner.context.video_shift, owner.context.audio_shift)):
        raise ValueError("H16 PASS2 Stage EAV MODEL is not bound to this exact window")
    runtime = model.get_attachment(EAV_KEY)
    if owner.mode != "disabled" and (
            type(runtime) is not StageEAVRuntime or runtime.context != owner.context
            or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
        raise ValueError("H16 PASS2 Stage EAV wrapper was removed or replaced")


def audit_h16_eav(result, spec, runtime):
    if (type(result) is not H16Pass2Result or type(runtime) is not StageEAVRuntime
            or runtime.context.recipe != RECIPE):
        raise TypeError("H16 Stage EAV audit needs its matching result and runtime")
    profile = json.loads(runtime.context.profile)
    if (result.core_result.index != spec.index
            or result.core_result.plan_sha256 != spec.plan_sha256
            or result.core_result.source_identity != spec.source_identity
            or result.audio_output != profile["audio_output"]
            or profile["plan_sha256"] != spec.plan_sha256
            or profile["window_index"] != spec.index
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("H16 Stage EAV result/runtime identity mismatch")
    report = runtime.snapshot()
    if report["status"] == "aborted":
        raise RuntimeError("H16 Stage EAV aborted: " + report["feta"]["aborted"])
    report["h16_window_index"] = spec.index
    report["h16_plan_sha256"] = spec.plan_sha256
    report["audio_output"] = result.audio_output
    return result.output_latent, json.dumps(report, ensure_ascii=False, indent=2)
