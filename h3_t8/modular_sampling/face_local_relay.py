"""Pair stock local Ref2VA Relay with one source-bound Parity Face stage.

Each repair job owns a new local timeline. Parent-frame or studio-window
mapping is authenticated by the Face StageContext; no global Relay Plan is
silently sliced or shifted. The cropped AV and original source audio remain
the sampler/delivery authorities.
"""
from __future__ import annotations

import json

from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..core import nested_av_parts, split_noise_masks
from .chunked_relay import _assert_paired_conditioning
from .contracts import StageContext
from .face_stage import (_source_contract, PARITY_SCHEMA, MULTIFACE_SCHEMA,
                         WINDOW_SCHEMA)
from .native_explicit import RECIPE as NATIVE_RECIPE, capture_owner
from .results import canonical

SCHEMA = "t8.modular-sampling.face-local-relay-binding.v1"
SCHEMAS = {"parity": PARITY_SCHEMA, "multiface": MULTIFACE_SCHEMA,
           "window": WINDOW_SCHEMA}


def bind_local_face_relay(model, relay_positive, relay_av_latent, relay_plan,
                          face_plan, source_frames, stage_av_latent, stage_context,
                          parent_frames=None, window_plan=None, window_mapping=None,
                          source_audio=None, window_audio=None):
    """Return exactly the paired MODEL/CONDITIONING for a local face repair."""
    compiled = relay._validate_plan(relay_plan)
    if (type(stage_context) is not StageContext or stage_context.recipe != NATIVE_RECIPE
            or stage_context.stage != "native_high"):
        raise ValueError("Local Face Relay needs a source-bound refinement StageContext")
    bound = json.loads(stage_context.profile).get("face_refine")
    variant = bound.get("variant") if isinstance(bound, dict) else None
    if variant not in SCHEMAS or bound.get("schema") != SCHEMAS[variant]:
        raise ValueError("Local Face Relay requires Parity, multi-face or window StageContext")
    if capture_owner(model).context != stage_context:
        raise ValueError("Local Face Relay MODEL is not bound to this Face stage")
    current = _source_contract(face_plan, source_frames, stage_av_latent,
        bound["audio_policy"], variant=variant, parent_frames=parent_frames,
        window_plan=window_plan, window_mapping=window_mapping,
        source_audio=source_audio, window_audio=window_audio)
    if current != bound:
        raise ValueError("Local Face Relay parent/window/crop source differs from StageContext")
    expected_frames = int(face_plan["source"]["h3_aligned_frame_count"])
    if compiled["frame_count"] != expected_frames or len(compiled["events"]) < 2:
        raise ValueError("Local Face Relay needs two events on the exact aligned repair-window timeline")
    contract = relay.prompt_relay_model_contract(model)
    binding = contract["binding"]
    if (binding["plan_hash"] != compiled["plan_hash"] or binding["task"] != "ref2va"
            or binding["query_route"] != "video_only_paper"):
        raise ValueError("Local Face Relay MODEL belongs to another Plan, task or query route")
    _assert_paired_conditioning(relay_positive, binding)
    relay_video, relay_audio = nested_av_parts(relay_av_latent)
    stage_video, stage_audio = nested_av_parts(stage_av_latent)
    if (tuple(relay_video.shape) != tuple(stage_video.shape)
            or tuple(relay_audio.shape) != tuple(stage_audio.shape)
            or tuple(stage_video.shape) != stage_context.video_shape
            or tuple(stage_audio.shape) != stage_context.audio_shape):
        raise ValueError("Local Face Relay packed AV layout differs from crop refinement")
    if relay_audio.data_ptr() != stage_audio.data_ptr():
        raise ValueError("Local Face Relay crop replaced the locked source audio latent")
    _, relay_audio_mask = split_noise_masks(relay_av_latent, relay_video, relay_audio)
    if relay_audio_mask is None or bool(relay_audio_mask.count_nonzero()):
        raise ValueError("Local Face Relay requires a locked source audio mask")
    metadata = relay_positive[0][1]
    for _, item in relay_positive:
        if (item.get("minimax_frame_count") != metadata.get("minimax_frame_count")
                or len(item.get("minimax_keyframes", [])) != len(metadata.get("minimax_keyframes", []))
                or len(item.get("minimax_refs", [])) != len(metadata.get("minimax_refs", []))):
            raise ValueError("Local Face Relay scheduled CONDITIONING changed packed media layout")
    layout = build_packed_layout(binding["text_len"], *stage_video.shape[2:],
        stage_audio.shape[-1], keyframes=metadata.get("minimax_keyframes", []),
        refs=metadata.get("minimax_refs", []),
        frame_count=metadata.get("minimax_frame_count"))
    if relay._layout_contract(layout) != binding["layout_contract"]:
        raise ValueError("Local Face Relay CONDITIONING layout is not authenticated")
    report = {"schema": SCHEMA, "status": "paired_local_model_and_conditioning",
              "variant": variant, "face_plan_sha256": bound["plan_sha256"],
              "relay_plan_hash": compiled["plan_hash"],
              "relay_binding_hash": binding["binding_hash"],
              "stage_context_sha256": stage_context.descriptor_sha256,
              "local_render_frames": compiled["frame_count"],
              "source_frames": int(face_plan["source"]["frame_count"]),
              "absolute_parent_window": bound.get("absolute_window"),
              "sampled": False, "cache_reuse_authorized": False,
              "quality_accepted": False,
              "boundary": "This Plan is local to the aligned repair window, not a projection of parent-film "
                          "events. Face crop AV and locked source audio remain authoritative; actual Relay "
                          "attention calls and visual quality need downstream audit."}
    return model, relay_positive, canonical(report)
