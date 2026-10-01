"""Authenticate a full-clip external Prompt Relay for standard Face Refine.

The stock Relay nodes own text encoding and the attention patch.  Face Refine
replaces only their blank video latent with encoded face crops; this adapter
checks that the resulting stage still has the same native packed AV layout.
Windowed, parity and multi-face routes need their own time projections.
"""
from __future__ import annotations

import json

from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..core import nested_av_parts
from .chunked_relay import _assert_paired_conditioning
from .contracts import StageContext
from .face_stage import SCHEMA as FACE_SCHEMA, _source_contract
from .native_explicit import RECIPE as NATIVE_RECIPE
from .results import canonical

SCHEMA = "t8.modular-sampling.face-relay-binding.v1"


def bind_standard_face_relay(model, relay_positive, relay_av_latent, relay_plan,
                             face_plan, source_frames, stage_av_latent, stage_context):
    """Keep the crop AV authoritative while pairing Relay MODEL/CONDITIONING."""
    compiled = relay._validate_plan(relay_plan)
    if (type(stage_context) is not StageContext or
            stage_context.recipe != NATIVE_RECIPE or stage_context.stage != "native_high"):
        raise ValueError("Face Relay needs the source-bound refinement StageContext")
    bound = json.loads(stage_context.profile).get("face_refine")
    if not isinstance(bound, dict) or bound.get("schema") != FACE_SCHEMA:
        raise ValueError("Face Relay supports only the standard full-clip Face stage")
    current = _source_contract(face_plan, source_frames, stage_av_latent,
                               bound["audio_policy"], variant="standard")
    if current != bound:
        raise ValueError("Face Relay StageContext does not bind this crop source")
    if compiled["frame_count"] != int(face_plan["source"]["frame_count"]):
        raise ValueError("Face Relay timeline differs from full-clip source")
    contract = relay.prompt_relay_model_contract(model)
    binding = contract["binding"]
    if (binding["plan_hash"] != compiled["plan_hash"] or
            binding["task"] != "t2va" or binding["query_route"] != "video_only_paper"):
        raise ValueError("Face Relay MODEL belongs to another timeline, task or query route")
    _assert_paired_conditioning(relay_positive, binding)
    relay_video, relay_audio = nested_av_parts(relay_av_latent)
    stage_video, stage_audio = nested_av_parts(stage_av_latent)
    if (tuple(relay_video.shape) != tuple(stage_video.shape) or
            tuple(relay_audio.shape) != tuple(stage_audio.shape) or
            tuple(stage_video.shape) != stage_context.video_shape or
            tuple(stage_audio.shape) != stage_context.audio_shape):
        raise ValueError("Face Relay packed AV layout differs from crop refinement")
    if (relay_audio.data_ptr() != stage_audio.data_ptr() or
            relay_av_latent.get("noise_mask") is not stage_av_latent.get("noise_mask")):
        raise ValueError("Face Relay crop must preserve the original audio and noise mask objects")
    metadata = relay_positive[0][1]
    for _, item in relay_positive:
        if (item.get("minimax_frame_count") != metadata.get("minimax_frame_count") or
                len(item.get("minimax_keyframes", [])) != len(metadata.get("minimax_keyframes", [])) or
                len(item.get("minimax_refs", [])) != len(metadata.get("minimax_refs", []))):
            raise ValueError("Face Relay scheduled conditioning changed packed media layout")
    layout = build_packed_layout(
        binding["text_len"], *stage_video.shape[2:], stage_audio.shape[-1],
        keyframes=metadata.get("minimax_keyframes", []),
        refs=metadata.get("minimax_refs", []),
        frame_count=metadata.get("minimax_frame_count"))
    if relay._layout_contract(layout) != binding["layout_contract"]:
        raise ValueError("Face Relay CONDITIONING layout is not authenticated")
    report = {"schema": SCHEMA, "status": "paired_crop_model_and_conditioning",
              "face_plan_sha256": bound["plan_sha256"],
              "relay_plan_hash": compiled["plan_hash"],
              "relay_binding_hash": binding["binding_hash"],
              "stage_context_sha256": stage_context.descriptor_sha256,
              "frame_count": compiled["frame_count"], "sampled": False,
              "cache_reuse_authorized": False, "quality_accepted": False,
              "boundary": "Relay's blank AV is layout evidence only. The face-crop AV is the sampler input; "
                          "actual attention calls and visual quality need downstream audit."}
    return model, relay_positive, canonical(report)
