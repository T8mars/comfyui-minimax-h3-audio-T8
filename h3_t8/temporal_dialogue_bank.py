"""Precompile native text once per actual Chunk window, before HIGH sampling.

Only the frozen encoded values enter bank identity, not an invented portable
identity for the live CLIP/Bridge provider. Unknown provider metadata is never
silently serialized or certified as a reproducible cache input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from . import chunked_two_pass_parity as parity
from . import chunked_two_pass_upscale_advanced as legacy
from .modular_sampling.results import _input_identity
from .temporal_dialogue_encoding import encode_scoped_window, validate_native_text_recipe
from .temporal_dialogue_identity import LiveMetadataIdentity
from .temporal_dialogue_scope import (
    CompiledWindowText, DialoguePlan, _digest, validate_compiled_window,
    validate_dialogue_plan, window_descriptors,
)


@dataclass(frozen=True)
class EncodedWindow:
    compiled: CompiledWindowText
    positive: list
    prepared_prompt: str
    tokens: dict
    report: dict
    content_identity: dict


@dataclass(frozen=True)
class WindowConditioningBank:
    dialogue_plan: DialoguePlan
    source_identity: dict
    chunk_plan_identity: dict
    width: int
    height: int
    executor_contract: str
    encoded: tuple[EncodedWindow, ...]
    sha256: str
    identity_context: LiveMetadataIdentity


def _executor_windows(source, plan):
    if type(source) is not dict or type(plan) is not dict:
        raise ValueError("Connect an actual Chunk Plan")
    video, audio = parity._parts(source.get("samples"), "dialogue bank source AV")
    frames = legacy.frames_for_tokens(int(video.shape[2]))
    if plan.get("schema") == parity.SCHEMA:
        parity.standard_plan(plan, json.dumps(plan.get("parity_report")))
        segments, _ = legacy.compute_temporal_segments(
            int(video.shape[2]), plan["temporal_chunk_frames"], plan["temporal_overlap_frames"])
        bounds = tuple(parity._audio_bounds(sf, ef, frames, int(audio.shape[-1]))
                       for _, sf, _, ef in segments)
        contract, padding = "v5_joint_refine4", "include_last"
    elif plan.get("schema") in {
        legacy.PLAN_SCHEMA_V1, legacy.PLAN_SCHEMA_GLOBAL_NOISE_V2,
        legacy.PLAN_SCHEMA_LOW_SIGMA_V3, legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
    }:
        if (plan["schema"] != legacy.PLAN_SCHEMA_V1
                and plan.get("temporal_strategy") == "full_clip_safe"):
            segments = [(0, 0, int(video.shape[2]), frames)]
        else:
            segments, _ = legacy.compute_temporal_segments(
                int(video.shape[2]), plan["temporal_chunk_frames"], plan["temporal_overlap_frames"])
        bounds = tuple((round(sf * legacy.FRAME_RESCALE),
                        min(int(audio.shape[-1]), round(ef * legacy.FRAME_RESCALE)))
                       for _, sf, _, ef in segments)
        contract, padding = "legacy_chunked_audio_context", "retain_source_tail"
    else:
        raise ValueError("Temporal Dialogue bank does not recognize this executor contract")
    descriptors = window_descriptors(segments, bounds, frames, int(audio.shape[-1]), padding)
    return tuple(segments), bounds, descriptors, contract


def _encoded_identity(positive, prepared_prompt, tokens, report, context):
    return context.describe({"positive": positive, "prepared_prompt": prepared_prompt,
                             "tokens": tokens, "report": report})


def _bank_payload(bank):
    return {"dialogue_plan_sha256": bank.dialogue_plan.sha256,
            "source_identity": bank.source_identity, "chunk_plan_identity": bank.chunk_plan_identity,
            "width": bank.width, "height": bank.height, "executor_contract": bank.executor_contract,
            "native_metadata_portable": bank.identity_context.portable,
            "encoded": [{"text_sha256": item.compiled.sha256,
                         "content_identity": item.content_identity} for item in bank.encoded]}


def prepare_window_bank(recipe, dialogue_plan, source, chunk_plan):
    """All CLIP work happens here; consuming a bank does not reload/encode CLIP."""
    validate_native_text_recipe(recipe)
    validate_dialogue_plan(dialogue_plan)
    _segments, _bounds, descriptors, contract = _executor_windows(source, chunk_plan)
    if (recipe.frame_count != dialogue_plan.total_frames
            or descriptors[0].total_frames != dialogue_plan.total_frames):
        raise ValueError("Native recipe, Dialogue Plan and actual source duration differ")
    if (recipe.width, recipe.height) != (chunk_plan.get("target_width"), chunk_plan.get("target_height")):
        raise ValueError("Capture HIGH native media at the actual Chunk target dimensions")
    encoded = []
    context = LiveMetadataIdentity()
    for window in descriptors:
        positive, compiled, prompt, report, tokens = encode_scoped_window(
            recipe, dialogue_plan, window, return_tokens=True)
        encoded.append(EncodedWindow(compiled, positive, prompt, tokens, report,
                                     _encoded_identity(positive, prompt, tokens, report, context)))
    draft = WindowConditioningBank(dialogue_plan, _input_identity(source), _input_identity(chunk_plan),
                                   recipe.width, recipe.height, contract, tuple(encoded), "", context)
    bank = WindowConditioningBank(dialogue_plan, draft.source_identity, draft.chunk_plan_identity,
                                  draft.width, draft.height, contract, draft.encoded,
                                  _digest(_bank_payload(draft)), context)
    report = {"bank_sha256": bank.sha256, "executor_contract": contract,
              "window_count": len(encoded), "native_encodes": len(encoded),
              "windows": [item.report for item in encoded], "sampling_executed": False,
              "quality_qualified": False, "provider_identity_certified": False,
              "native_metadata_portable": context.portable,
              "warnings": [] if context.portable else [
                  "Foreign metadata retained with live-only identity; internal state and Cold re-encoding equivalence unverified."],
              "boundary": "Frozen actual native encoding values, not a portable live provider."}
    return bank, report


def validate_window_bank(bank, source, chunk_plan):
    if type(bank) is not WindowConditioningBank or type(bank.identity_context) is not LiveMetadataIdentity:
        raise ValueError("Connect a precompiled Temporal Dialogue window bank")
    validate_dialogue_plan(bank.dialogue_plan)
    _segments, _bounds, descriptors, contract = _executor_windows(source, chunk_plan)
    if (bank.source_identity != _input_identity(source)
            or bank.chunk_plan_identity != _input_identity(chunk_plan)
            or bank.executor_contract != contract or len(bank.encoded) != len(descriptors)
            or (bank.width, bank.height) != (chunk_plan.get("target_width"), chunk_plan.get("target_height"))):
        raise ValueError("Window bank source, Chunk Plan, geometry or executor contract changed")
    for encoded, window in zip(bank.encoded, descriptors, strict=True):
        if type(encoded) is not EncodedWindow or encoded.compiled.window != window:
            raise ValueError("Window bank actual ownership descriptors changed")
        validate_compiled_window(bank.dialogue_plan, encoded.compiled)
        if encoded.content_identity != _encoded_identity(
                encoded.positive, encoded.prepared_prompt, encoded.tokens, encoded.report, bank.identity_context):
            raise ValueError("Window bank encoded native text, media or metadata changed")
    if _digest(_bank_payload(bank)) != bank.sha256:
        raise ValueError("Window bank content/SHA changed")
    return bank


def select_window_conditioning(bank, source, chunk_plan, index):
    validate_window_bank(bank, source, chunk_plan)
    if type(index) is not int or not 0 <= index < len(bank.encoded):
        raise ValueError("Temporal Dialogue window index is out of range")
    return bank.encoded[index]
