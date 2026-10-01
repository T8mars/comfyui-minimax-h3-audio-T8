"""Bind stock full-clip Prompt Relay to one exact H16 guarded PASS2 window.

The ordinary Relay nodes still own text and media encoding. The stock full
clip layout is not a valid layout for an H16 window; this adapter authenticates
the pair and projects only its event clock and PackedLayout. The CONDITIONING
keeps global keyframe coordinates because the legacy sampler reanchors it once
immediately before the actual window forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import comfy.conds
import comfy.patcher_extension as extension
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import enhance_a_video_advanced as feta
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..sampling import nested_av_parts
from .chunked_relay import _assert_paired_conditioning
from .h16_stages import H16Pass2Result, h16_window_piece
from .results import _input_identity


KEY = "t8_modular_h16_relay_v1"
RUNTIME_TYPE = "T8_H16_RELAY_RUNTIME"


@dataclass(frozen=True)
class H16RelayOwner:
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    piece_identity: dict
    previous_identity: dict | None
    window_index: int
    audio_output: str
    binding_hash: str
    sigma_values: tuple[float, ...]
    sampling_object: object


class H16RelayRuntime:
    def __init__(self, model, owner, blocks):
        self.model, self.owner, self.blocks = model, owner, int(blocks)
        self.baseline = {"completed_forwards": 0, "routed_attention_calls": 0}
        self.prepared = False
        self.closed = True

    def prepare(self, patcher, timestep, model_options):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("H16 Relay binding changed before PASS2")
        if self.closed:
            self.baseline = contract["execution_counts"]
            self.prepared = True
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("H16 Relay binding changed after PASS2")
        counts = contract["execution_counts"]
        deltas = {key: int(counts.get(key, 0)) - int(self.baseline.get(key, 0))
                  for key in self.baseline}
        forwards = len(self.owner.sigma_values) - 1
        expected = {"completed_forwards": forwards,
                    "routed_attention_calls": forwards * self.blocks}
        status = ("observed_relay_calls_quality_unverified"
                  if self.prepared and contract["attention_owner_verified"]
                  and deltas == expected else "unverified_incomplete_relay_coverage")
        return {"schema": "t8.modular-sampling.h16-relay-audit.v1",
                "status": status, "window_index": self.owner.window_index,
                "plan_sha256": self.owner.plan_sha256,
                "binding_hash": self.owner.binding_hash,
                "prepared": self.prepared, "actual_calls": deltas,
                "expected_calls": expected,
                "attention_owner_verified": contract["attention_owner_verified"],
                "cache_reuse_authorized": False, "quality_accepted": False}


def _sigma_values(sigmas):
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or len(sigmas) < 2
            or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:])):
        raise ValueError("H16 Relay needs finite descending PASS2 SIGMAS")
    return tuple(float(value) for value in sigmas.tolist())


def _paired_projected(positive, original, projected):
    paired = []
    for tensor, metadata in positive:
        updated = dict(metadata)
        if updated.get(relay.PROMPT_RELAY_BINDING_KEY) != original:
            raise ValueError("H16 Relay MODEL and CONDITIONING binding differ")
        model_conds = dict(updated.get("model_conds", {}))
        payload = model_conds.get(relay.PROMPT_RELAY_PAYLOAD_KEY)
        if type(payload) is not comfy.conds.CONDConstant or payload.cond != original["binding_hash"]:
            raise ValueError("H16 Relay CONDITIONING runtime payload differs")
        updated[relay.PROMPT_RELAY_BINDING_KEY] = projected
        model_conds[relay.PROMPT_RELAY_PAYLOAD_KEY] = comfy.conds.CONDConstant(
            projected["binding_hash"],
        )
        updated["model_conds"] = model_conds
        paired.append([tensor, updated])
    return paired


def project_h16_relay(raw_model, relay_model, relay_positive, relay_full_av_latent,
                      relay_plan, source_segment, lifted_segment, spec, context,
                      plan, sigmas, audio_output="preserve_first_pass",
                      previous_result=None):
    """Return a paired MODEL/CONDITIONING for one authenticated H16 window."""
    piece, locked, transition = h16_window_piece(
        source_segment, lifted_segment, spec, context, plan,
        previous_result, audio_output,
    )
    sigma_values = _sigma_values(sigmas)
    checked_plan = relay._validate_plan(relay_plan)
    source_video, source_audio = nested_av_parts(context.source_latent)
    if checked_plan["frame_count"] != legacy.frames_for_tokens(source_video.shape[2]):
        raise ValueError("H16 Relay Plan must cover the complete source timeline")
    contract = relay.prompt_relay_model_contract(relay_model)
    original = contract["binding"]
    if original["plan_hash"] != checked_plan["plan_hash"]:
        raise ValueError("H16 Relay MODEL belongs to another Plan")
    _assert_paired_conditioning(relay_positive, original)
    if (raw_model.model is not relay_model.model
            or raw_model.get_model_object("model_sampling") is not
            relay_model.get_model_object("model_sampling")):
        raise ValueError("H16 raw HIGH MODEL differs from external Relay Conditioning")
    relay_video, relay_audio = nested_av_parts(relay_full_av_latent)
    if (tuple(relay_video.shape) != tuple(source_video.shape)
            or tuple(relay_audio.shape) != tuple(source_audio.shape)):
        raise ValueError("H16 Relay full-clip AV layout differs from the source")
    meta = relay_positive[0][1]
    full_layout = build_packed_layout(
        original["text_len"], *relay_video.shape[2:], relay_audio.shape[-1],
        keyframes=meta.get("minimax_keyframes", []), refs=meta.get("minimax_refs", []),
        frame_count=meta.get("minimax_frame_count"),
    )
    if relay._layout_contract(full_layout) != original["layout_contract"]:
        raise ValueError("H16 Relay full-clip PackedLayout is not authenticated")
    video, audio = nested_av_parts(piece)
    local_positive = legacy.reanchor_conditioning(
        relay_positive, spec.start_frame, spec.end_frame, tuple(video.shape[-2:]),
    )
    local_positive = legacy.crop_conditioning(
        local_positive, *video.shape[-2:], 0, 0, *video.shape[-2:],
    )
    local_meta = local_positive[0][1]
    for _tensor, item in local_positive:
        if (item.get("minimax_frame_count") != local_meta.get("minimax_frame_count")
                or len(item.get("minimax_keyframes", [])) !=
                len(local_meta.get("minimax_keyframes", []))
                or len(item.get("minimax_refs", [])) != len(local_meta.get("minimax_refs", []))):
            raise ValueError("H16 Relay scheduled CONDITIONING changed local media layout")
    local_layout = build_packed_layout(
        original["text_len"], *video.shape[2:], audio.shape[-1],
        keyframes=local_meta.get("minimax_keyframes", []),
        refs=local_meta.get("minimax_refs", []),
        frame_count=local_meta.get("minimax_frame_count"),
    )
    projected = dict(original)
    local_keyframes = local_meta.get("minimax_keyframes", [])
    local_refs = local_meta.get("minimax_refs", [])
    # A global I2VA/FL2VA guide need not occur in every temporal window.
    # EAV classifies the actual packed payload, so the projected Relay task
    # must describe that same local payload rather than the stock full clip.
    projected["task"] = feta._classify_visual_task(
        local_keyframes, latent_frames=int(video.shape[2]), refs=local_refs,
    ).lower()
    projected["keyframe_count"] = len(local_keyframes)
    projected["reference_block_count"] = len(local_refs)
    projected["events"] = [
        {**event, "midpoint": float(event["midpoint"]) -
         (5.0 / 3.0) * spec.start_frame}
        for event in original["events"]
    ]
    projected["layout_contract"] = relay._layout_contract(local_layout)
    projected.pop("binding_hash")
    projected["binding_hash"] = relay._sha256_json(projected)
    paired = _paired_projected(relay_positive, original, projected)
    selected, _core_hashes = relay.patch_prompt_relay_model(
        raw_model, projected, contract["query_chunk_rows"],
    )
    verified = relay.prompt_relay_model_contract(selected)
    if verified["binding_hash"] != projected["binding_hash"]:
        raise RuntimeError("H16 projected Relay MODEL failed authentication")
    owner = H16RelayOwner(
        spec.plan_sha256, spec.source_identity, _input_identity(lifted_segment),
        _input_identity(piece),
        previous_result.output_identity if previous_result is not None else None,
        spec.index, audio_output, projected["binding_hash"], sigma_values,
        raw_model.get_model_object("model_sampling"),
    )
    selected = selected.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("H16 Relay is already projected on this MODEL branch")
    selected.set_attachments(KEY, owner)
    runtime = H16RelayRuntime(
        selected, owner, len(selected.get_model_object("diffusion_model").blocks),
    )
    selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    report = runtime.snapshot()
    report.update(sampled=False, global_plan_hash=checked_plan["plan_hash"],
                  window_frames=[spec.start_frame, spec.end_frame],
                  window_video_tokens=[spec.start_token, spec.end_token],
                  locked_overlap_tokens=locked, transition_overlap_tokens=transition,
                  projected_layout_hash=projected["layout_contract"]["contract_hash"])
    return selected, paired, runtime, json.dumps(report, ensure_ascii=False, indent=2)


def assert_h16_relay_binding(model, positive, source_segment, lifted_segment,
                             spec, context, plan, previous_result, sigmas, audio_output):
    owner = model.get_attachment(KEY)
    if owner is None:
        if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
            raise ValueError("Project full-clip Prompt Relay onto this H16 PASS2 window first")
        return
    piece, _locked, _transition = h16_window_piece(
        source_segment, lifted_segment, spec, context, plan,
        previous_result, audio_output,
    )
    if (type(owner) is not H16RelayOwner or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity
            or owner.lifted_identity != _input_identity(lifted_segment)
            or owner.piece_identity != _input_identity(piece)
            or owner.previous_identity != (
                previous_result.output_identity if previous_result is not None else None)
            or owner.window_index != spec.index or owner.audio_output != audio_output
            or owner.sigma_values != _sigma_values(sigmas)
            or model.get_model_object("model_sampling") is not owner.sampling_object):
        raise ValueError("H16 PASS2 Relay MODEL is not bound to this exact window")
    from .eav import KEY as EAV_KEY, StageEAVRuntime
    eav_runtime = model.get_attachment(EAV_KEY)
    if eav_runtime is None:
        contract = relay.prompt_relay_model_contract(model)
        if contract["binding_hash"] != owner.binding_hash:
            raise ValueError("H16 PASS2 projected Relay binding changed")
        binding = contract["binding"]
    else:
        if (type(eav_runtime) is not StageEAVRuntime or not eav_runtime.relay_required
                or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
            raise ValueError("H16 PASS2 composed Relay/EAV MODEL is not authenticated")
        binding = positive[0][1].get(relay.PROMPT_RELAY_BINDING_KEY) if positive else None
        if not isinstance(binding, dict) or binding.get("binding_hash") != owner.binding_hash:
            raise ValueError("H16 PASS2 composed Relay/EAV binding changed")
    _assert_paired_conditioning(positive, binding)


def audit_h16_relay(result, spec, runtime):
    if (type(result) is not H16Pass2Result or type(runtime) is not H16RelayRuntime
            or type(runtime.owner) is not H16RelayOwner):
        raise TypeError("H16 Relay audit needs its matching window result/runtime")
    owner = runtime.owner
    if (result.core_result.index != spec.index or owner.window_index != spec.index
            or result.core_result.plan_sha256 != owner.plan_sha256
            or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity
            or result.core_result.source_identity != spec.source_identity
            or result.audio_output != owner.audio_output
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("H16 Relay result/runtime identity mismatch")
    return result.output_latent, json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
