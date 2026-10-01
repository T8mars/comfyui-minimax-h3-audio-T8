"""Execution-local effects for an already validated Audio Refine tail.

The old Setup and Stage Bind remain authoritative for noise, masks, sampler and
SIGMAS. This adapter never samples, creates a replacement AV or accepts audio.
Relay encoding, Stage EAV and Core SamplerCustomAdvanced stay separate nodes.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .. import audio_refine_advanced as refine
from .. import enhance_a_video_advanced as feta
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..core import nested_av_parts
from ..long_video import LONG_VIDEO_CONDITIONING_KEY, LONG_VIDEO_SCHEMA, MOTION_FRAME_INDEX
from ..long_video_dual_identity import _v2_sampling_identity
from ..vdn_attention_compat import _factory_closure
from .audio_refine_stage import SCHEMA, _sha
from .chunked_relay import _assert_paired_conditioning
from .contracts import StageContext
from .results import _input_identity, canonical, request_conditions

RECIPE = "audio_refine.signed-tail.v1"
KEY = "t8_modular_audio_refine_effects_v1"


@dataclass(frozen=True)
class TailEffectsOwner:
    context: StageContext
    base: object
    sampling: object
    sampling_json: str | None
    stage_identity: dict
    guider: object
    conditions: dict
    boundary_sha256: str
    long_contract: dict | None


def _long_contract(model, entries, segment_index, context_frames):
    marked = [item.get(LONG_VIDEO_CONDITIONING_KEY) == LONG_VIDEO_SCHEMA for item in entries]
    if not any(marked):
        if segment_index != 0 or context_frames != 0:
            raise ValueError("Audio Refine non-long tail cannot declare a continuation window")
        return None
    if not all(marked):
        raise ValueError("Audio Refine scheduled conditions disagree on long-video scope")
    contract = feta._assert_long_video_contract(
        model, segment_index=segment_index, context_frames=context_frames)
    from ..long_video import step_offsets
    expected = list(step_offsets(contract["expected_motion_latent_steps"])) if segment_index else []
    for item in entries:
        offsets = [part[MOTION_FRAME_INDEX] for part in item.get("minimax_keyframes", [])
                   if MOTION_FRAME_INDEX in part]
        if offsets != expected:
            raise ValueError("Audio Refine long-video window differs from actual motion keyframes")
    return contract


def bind_tail_effects(stage_boundary, model, noise, guider, sampler, sigmas,
                      stage_latent, segment_index=0, context_frames=0):
    if not isinstance(stage_boundary, dict) or stage_boundary.get("schema") != SCHEMA:
        raise ValueError("Audio Refine effects require the existing signed Stage Bind boundary")
    unsigned = dict(stage_boundary)
    digest = unsigned.pop("context_sha256", None)
    if digest != _sha(unsigned):
        raise ValueError("Audio Refine effects boundary SHA-256 differs")
    controls = {"model": model, "noise": noise, "guider": guider, "sampler": sampler}
    if any(stage_boundary.get(name + "_object_id") != id(value) for name, value in controls.items()):
        raise ValueError("Audio Refine effects controls differ from Stage Bind")
    if (type(guider) is not refine.AudioRefineBasicGuider or guider.model_patcher is not model
            or guider.cfg != 1.0 or not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point()
            or stage_boundary.get("sigmas") != [float(value) for value in sigmas.tolist()]
            or stage_boundary.get("stage_latent") != _input_identity(stage_latent)):
        raise ValueError("Audio Refine effects source, guider or SIGMAS changed after Stage Bind")
    if type(segment_index) is not int or type(context_frames) is not int:
        raise ValueError("Audio Refine window indices must be integers")
    entries = guider.original_conds.get("positive", [])
    if not entries:
        raise ValueError("Audio Refine effects require the Setup positive conditioning")
    long_contract = _long_contract(model, entries, segment_index, context_frames)
    video, audio = nested_av_parts(stage_latent)
    bypassed = stage_boundary["bypassed"]
    if bypassed != (sigmas.numel() == 0):
        raise ValueError("Audio Refine abstain lost empty SIGMAS")
    context = StageContext(
        recipe=RECIPE, stage="audio_tail", profile=canonical({
            "variant": stage_boundary["variant"], "boundary_sha256": digest,
            "long_video": long_contract}), start=0, end=0 if bypassed else sigmas.numel() - 1,
        trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=12., audio_shift=3., video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="identity_noop" if bypassed else "final_video_av_audio_noise_only",
        output_semantics="identity_noop" if bypassed else "unaccepted_audio_candidate",
        denoised_semantics="identity_noop" if bypassed else "unaccepted_audio_prediction")
    if model.get_attachment(KEY) is not None:
        raise ValueError("Audio Refine tail already has an effects context")
    selected = model.get_model_object("model_sampling")
    patched = model.clone()
    patched.set_attachments(KEY, TailEffectsOwner(
        context, model.model, selected, None if bypassed else canonical(_v2_sampling_identity(selected)),
        _input_identity(stage_latent), guider, request_conditions(guider.original_conds),
        digest, long_contract))
    validate_stage(patched, sigmas, stage_latent, context)
    return patched, context, canonical({
        "schema": RECIPE, "status": "abstain_no_sample" if bypassed else "effects_ready",
        "stage_context": context.to_dict(), "sampled": False,
        "controls_preserved": True, "portable_cache_reuse_authorized": False,
        "boundary": "EAV enhances target-video attention during the joint AV tail; it is not an audio "
                    "enhancer or a quality claim. Original video is relocked only by the old Quality Gate."})


def capture_owner(model):
    owner = model.get_attachment(KEY)
    if type(owner) is not TailEffectsOwner or owner.context.recipe != RECIPE:
        raise ValueError("Audio Refine effects MODEL lacks the bound tail owner")
    if (model.model is not owner.base or model.get_model_object("model_sampling") is not owner.sampling
            or (owner.context.steps and canonical(_v2_sampling_identity(owner.sampling)) != owner.sampling_json)):
        raise ValueError("Audio Refine effects selected model/sampling changed")
    if request_conditions(owner.guider.original_conds) != owner.conditions:
        raise ValueError("Audio Refine original Setup conditioning changed")
    if owner.guider.cfg != 1.0 or owner.guider.model_patcher.model is not owner.base:
        raise ValueError("Audio Refine original Setup guider changed")
    if owner.context.steps:
        options = model.model_options["transformer_options"]
        if (options.get("minimax_h3_sigma_shift_video"), options.get("minimax_h3_sigma_shift_audio")) != (12., 3.):
            raise ValueError("Audio Refine effects changed native AV clock shifts")
    return owner


def validate_stage(model, sigmas, av_latent, context):
    owner = capture_owner(model)
    if type(context) is not StageContext or context != owner.context:
        raise ValueError("Audio Refine effects MODEL and context differ")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point()
            or tuple(float(value) for value in sigmas.tolist()) != context.trajectory_sigmas):
        raise ValueError("Audio Refine effects SIGMAS differ from bound tail")
    if _input_identity(av_latent) != owner.stage_identity:
        raise ValueError("Audio Refine effects source AV or masks changed")
    video, audio = nested_av_parts(av_latent)
    return video, audio, SimpleNamespace(runtime=None, profile=context.profile)


def _bound_relay(model):
    from . import eav
    wrappers = model.get_wrappers("diffusion_model", eav.KEY)
    if wrappers:
        if len(wrappers) != 1:
            raise ValueError("Audio Refine tail has ambiguous Stage EAV owners")
        state = _factory_closure(wrappers[0], eav.apply_stage_eav, "stage_wrapper")
        if state is None:
            raise ValueError("Audio Refine tail has an unauthenticated Stage EAV wrapper")
        return state.get("binding")
    if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        return relay.prompt_relay_model_contract(model)["binding"]
    return None


def tail_effects_guider(model, stage_context, sigmas, stage_latent, positive=None):
    """Rebind only this stage's guider, preserving noise/sampler/AV and cfg=1."""
    validate_stage(model, sigmas, stage_latent, stage_context)
    owner = capture_owner(model)
    original = owner.guider.original_conds["positive"]
    # Converted Core conditions are not re-encoded when positive is omitted.
    if positive is not None and (not isinstance(positive, (tuple, list)) or not all(
            isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict)
            for item in positive)):
        raise ValueError("Audio Refine external positive must be native CONDITIONING")
    entries = original if positive is None else [item[1] for item in positive]
    if not entries:
        raise ValueError("Audio Refine external positive cannot be empty")
    fields = ("minimax_keyframes", "minimax_refs", "minimax_frame_count", LONG_VIDEO_CONDITIONING_KEY)
    if positive is not None:
        # No-positive is an exact pass-through of all original scheduled media.
        # A replacement must retain each scheduled source, not just entry zero.
        if len(entries) != len(original):
            raise ValueError("Audio Refine external conditioning changed source schedule length")
        for item, expected in zip(entries, original, strict=True):
            if any(_input_identity(item.get(field)) != _input_identity(expected.get(field))
                   for field in (*fields, "start_percent", "end_percent")):
                raise ValueError("Audio Refine external conditioning changed source media or frame timeline")
    binding = _bound_relay(model)
    if binding is not None:
        paired = positive if positive is not None else [[item.get("cross_attn"), item] for item in original]
        _assert_paired_conditioning(paired, binding)
        video, audio = nested_av_parts(stage_latent)
        for item in entries:
            layout = build_packed_layout(
                binding["text_len"], *video.shape[2:], audio.shape[-1],
                keyframes=item.get("minimax_keyframes", []), refs=item.get("minimax_refs", []),
                frame_count=item.get("minimax_frame_count"))
            if owner.long_contract is not None:
                from ..long_video import repair_long_video_layout
                layout = repair_long_video_layout(layout, item.get("minimax_keyframes", []),
                    item.get("minimax_refs", []), item.get("minimax_frame_count"))
            if relay._layout_contract(layout) != binding["layout_contract"]:
                raise ValueError("Audio Refine Relay layout differs from frozen source AV")
    elif any(relay.PROMPT_RELAY_BINDING_KEY in item for item in entries):
        raise ValueError("Audio Refine Relay conditioning has no paired MODEL")
    if positive is None:
        guider = copy(owner.guider)
        guider.model_patcher = model
        guider.model_options = model.model_options
        guider.original_conds = dict(owner.guider.original_conds)
    else:
        guider = refine.AudioRefineBasicGuider(model, positive)
    return guider, canonical({"schema": RECIPE, "status": "guider_ready",
        "stage_context_sha256": stage_context.descriptor_sha256,
        "external_positive": positive is not None,
        "relay_binding_hash": binding["binding_hash"] if binding is not None else None,
        "cfg": 1., "sampled": False, "quality_accepted": False,
        "portable_cache_reuse_authorized": False})
