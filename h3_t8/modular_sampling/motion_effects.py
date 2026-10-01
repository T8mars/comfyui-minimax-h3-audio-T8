"""Bind an external full-retimed-clip Prompt Relay to Motion pass 2.

The stock Relay Conditioning node owns text/media encoding and its attention
wrapper. This adapter only authenticates that its complete time/layout matches
the signed retimed AV being refined; it never substitutes Relay's generated AV
for Motion's VAE/phase-vocoder result. Stage EAV stays a separate downstream
node and its runtime audit is required for any actual-call claim.
"""
from __future__ import annotations

import json

from .. import motion_recovery_advanced as motion
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..core import align_frame_count, nested_av_parts
from .chunked_relay import _assert_paired_conditioning
from .contracts import StageContext
from .motion_stage import SCHEMA as MOTION_SCHEMA
from .native_explicit import RECIPE as NATIVE_RECIPE
from .results import _input_identity, canonical

SCHEMA = "t8.modular-sampling.motion-relay-binding.v1"


def retimed_relay_length(motion_plan):
    plan = motion.validate_motion_plan(motion_plan)
    frames = plan["expanded_length"]
    if (plan["status"] != "ready" or frames <= plan["world_length"]
            or align_frame_count(frames) != frames):
        raise ValueError("Motion Relay needs a ready H3-aligned retimed plan")
    return frames, canonical({"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
                              "expanded_frames": frames, "sampled": False})


def bind_motion_relay(model, relay_positive, relay_av_latent, relay_plan,
                      motion_plan, stage_av_latent, stage_context):
    frames, _ = retimed_relay_length(motion_plan)
    plan = motion.validate_motion_plan(motion_plan)
    compiled = relay._validate_plan(relay_plan)
    if compiled["frame_count"] != frames:
        raise ValueError("Motion Relay timeline differs from retimed frame count")
    if (type(stage_context) is not StageContext or
            stage_context.recipe != NATIVE_RECIPE or stage_context.stage != "native_high"):
        raise ValueError("Motion Relay needs the source-bound pass-2 StageContext")
    source = json.loads(stage_context.profile).get("motion_recovery")
    if (not isinstance(source, dict) or source.get("schema") != MOTION_SCHEMA
            or source.get("plan_sha256") != plan["plan_sha256"]
            or source.get("av_latent") != _input_identity(stage_av_latent)):
        raise ValueError("Motion Relay StageContext does not bind this retimed source")
    contract = relay.prompt_relay_model_contract(model)
    binding = contract["binding"]
    if (binding["plan_hash"] != compiled["plan_hash"]
            or binding["task"] != "t2va"):
        raise ValueError("Motion Relay MODEL belongs to another timeline or task")
    _assert_paired_conditioning(relay_positive, binding)
    relay_video, relay_audio = nested_av_parts(relay_av_latent)
    stage_video, stage_audio = nested_av_parts(stage_av_latent)
    if (tuple(relay_video.shape) != tuple(stage_video.shape)
            or tuple(relay_audio.shape) != tuple(stage_audio.shape)
            or tuple(stage_video.shape) != stage_context.video_shape
            or tuple(stage_audio.shape) != stage_context.audio_shape):
        raise ValueError("Motion Relay packed AV layout differs from retimed pass 2")
    meta = relay_positive[0][1]
    layout = build_packed_layout(
        binding["text_len"], *relay_video.shape[2:], relay_audio.shape[-1],
        keyframes=meta.get("minimax_keyframes", []), refs=meta.get("minimax_refs", []),
        frame_count=meta.get("minimax_frame_count"))
    if relay._layout_contract(layout) != binding["layout_contract"]:
        raise ValueError("Motion Relay CONDITIONING layout is not authenticated")
    report = {"schema": SCHEMA, "status": "paired_retained_model_and_conditioning",
              "motion_plan_sha256": plan["plan_sha256"],
              "relay_plan_hash": compiled["plan_hash"],
              "relay_binding_hash": binding["binding_hash"],
              "stage_context_sha256": stage_context.descriptor_sha256,
              "frame_count": frames, "sampled": False,
              "cache_reuse_authorized": False, "quality_accepted": False,
              "boundary": "Relay's AV is layout evidence only. Motion's signed retimed AV stays the "
                          "sampler input; EAV/Relay actual forwards require the downstream audit."}
    return model, relay_positive, canonical(report)
