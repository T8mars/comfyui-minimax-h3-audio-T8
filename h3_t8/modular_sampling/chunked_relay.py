"""Authenticate an external Prompt Relay pair for a full-clip Chunked PASS2.

The ordinary Relay Plan/Conditioning nodes own text encoding and attention.
This adapter never rewrites their plan or substitutes a new sampling backend.
Temporal chunks and spatial tiles need separate local-layout adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import torch
import comfy.conds
import comfy.patcher_extension as extension

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..sampling import nested_av_parts
from .chunked_source import ChunkedSourceSegment
from .chunked_stages import ChunkedPass2Result, _check_segment
from .results import _input_identity


KEY = "t8_modular_chunked_pass2_relay_v1"
RUNTIME_TYPE = "T8_CHUNKED_PASS2_RELAY_RUNTIME"


@dataclass(frozen=True)
class ChunkedRelayOwner:
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    segment_index: int
    binding_hash: str
    sigma_values: tuple[float, ...]


class ChunkedRelayRuntime:
    def __init__(self, model, owner, blocks):
        self.model, self.owner, self.blocks = model, owner, int(blocks)
        self.baseline = {"completed_forwards": 0, "routed_attention_calls": 0}
        self.prepared = False
        self.closed = True

    def prepare(self, patcher, timestep, model_options):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("Chunked Relay binding changed before PASS2")
        if self.closed:
            self.baseline = contract["execution_counts"]
            self.prepared = True
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("Chunked Relay binding changed after PASS2")
        counts = contract["execution_counts"]
        deltas = {key: int(counts.get(key, 0)) - int(self.baseline.get(key, 0))
                  for key in self.baseline}
        expected_forwards = len(self.owner.sigma_values) - 1
        expected_attention = expected_forwards * self.blocks
        status = ("observed_relay_calls_quality_unverified"
                  if self.prepared and contract["attention_owner_verified"]
                  and deltas == {"completed_forwards": expected_forwards,
                                 "routed_attention_calls": expected_attention}
                  else "unverified_incomplete_relay_coverage")
        return {"schema": "t8.modular-sampling.chunked-relay-audit.v1",
                "status": status, "segment_index": self.owner.segment_index,
                "plan_sha256": self.owner.plan_sha256,
                "binding_hash": self.owner.binding_hash,
                "prepared": self.prepared,
                "actual_calls": deltas,
                "expected_calls": {"completed_forwards": expected_forwards,
                                   "routed_attention_calls": expected_attention},
                "attention_owner_verified": contract["attention_owner_verified"],
                "cache_reuse_authorized": False, "quality_accepted": False}


def bind_full_clip_relay(model, positive, relay_latent, relay_plan, source_segment,
                         lifted_segment, spec, pass2_context, plan, sigmas):
    """Pair stock external Relay outputs with one exact full-frame PASS2 input."""
    _check_segment(source_segment, spec, pass2_context, plan)
    if (plan["schema"] == legacy.PLAN_SCHEMA_V1 or spec.index != 0 or spec.count != 1
            or plan.get("spatial_strategy") != "full_frame_safe"):
        raise ValueError("Chunked Relay currently requires one full-clip, full-frame v2-v4 PASS2")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or len(sigmas) < 2
            or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:])):
        raise ValueError("Chunked Relay needs finite descending PASS2 SIGMAS")
    checked_plan = relay._validate_plan(relay_plan)
    if checked_plan["frame_count"] != spec.end_frame - spec.start_frame:
        raise ValueError("Chunked Relay Plan frame count differs from the PASS2 segment")
    contract = relay.prompt_relay_model_contract(model)
    binding = contract["binding"]
    if binding["plan_hash"] != checked_plan["plan_hash"]:
        raise ValueError("Chunked Relay MODEL belongs to another Plan")
    if not positive or not all(isinstance(item, (tuple, list)) and len(item) == 2
                               for item in positive):
        raise ValueError("Chunked Relay needs paired native CONDITIONING")
    _assert_paired_conditioning(positive, binding)
    video, audio = nested_av_parts(lifted_segment)
    relay_video, relay_audio = nested_av_parts(relay_latent)
    if (tuple(relay_video.shape) != tuple(video.shape)
            or tuple(relay_audio.shape) != tuple(audio.shape)):
        raise ValueError("Chunked Relay target AV layout differs from lifted PASS2 segment")
    metadata = positive[0][1]
    keyframes = metadata.get("minimax_keyframes", [])
    refs = metadata.get("minimax_refs", [])
    for _, item_metadata in positive:
        if (item_metadata.get("minimax_frame_count") != metadata.get("minimax_frame_count")
                or len(item_metadata.get("minimax_keyframes", [])) != len(keyframes)
                or len(item_metadata.get("minimax_refs", [])) != len(refs)):
            raise ValueError("Chunked Relay scheduled CONDITIONING changed media layout")
    if keyframes and metadata.get("minimax_frame_count") != checked_plan["frame_count"]:
        raise ValueError("Chunked Relay keyframe clock differs from PASS2")
    layout = build_packed_layout(binding["text_len"], *video.shape[2:], audio.shape[-1],
                                 keyframes=keyframes, refs=refs,
                                 frame_count=checked_plan["frame_count"])
    if relay._layout_contract(layout) != binding["layout_contract"]:
        raise ValueError("Chunked Relay exact PackedLayout differs from PASS2")
    chunk_conditioning = legacy.reanchor_conditioning(
        positive, spec.start_frame, spec.end_frame, tuple(video.shape[-2:]))
    chunk_conditioning = legacy.crop_conditioning(
        chunk_conditioning, *video.shape[-2:], 0, 0, *video.shape[-2:])
    actual_meta = chunk_conditioning[0][1]
    if (len(actual_meta.get("minimax_keyframes", [])) != len(keyframes)
            or len(actual_meta.get("minimax_refs", [])) != len(refs)):
        raise ValueError("Chunked PASS2 conditioning rewrite changed Relay layout")
    owner = ChunkedRelayOwner(
        spec.plan_sha256, spec.source_identity, _input_identity(lifted_segment), spec.index,
        binding["binding_hash"], tuple(float(value) for value in sigmas.tolist()),
    )
    selected = model.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("Chunked Relay already bound to this MODEL branch")
    selected.set_attachments(KEY, owner)
    runtime = ChunkedRelayRuntime(selected, owner,
                                  len(selected.get_model_object("diffusion_model").blocks))
    selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    report = runtime.snapshot()
    report["layout_contract_hash"] = binding["layout_contract"]["contract_hash"]
    report["sampled"] = False
    return selected, positive, runtime, json.dumps(report, ensure_ascii=False, indent=2)


def _assert_paired_conditioning(positive, binding):
    if not positive or not all(isinstance(item, (tuple, list)) and len(item) == 2
                               for item in positive):
        raise ValueError("Chunked Relay needs paired native CONDITIONING")
    for _, metadata in positive:
        if not isinstance(metadata, dict) or metadata.get(relay.PROMPT_RELAY_BINDING_KEY) != binding:
            raise ValueError("Chunked Relay MODEL and CONDITIONING binding differ")
        payload = metadata.get("model_conds", {}).get(relay.PROMPT_RELAY_PAYLOAD_KEY)
        if type(payload) is not comfy.conds.CONDConstant or payload.cond != binding["binding_hash"]:
            raise ValueError("Chunked Relay CONDITIONING has no matching runtime payload")


def assert_relay_binding(model, positive, sigmas, lifted_segment, spec):
    owner = model.get_attachment(KEY)
    if owner is None:
        from .chunked_v1_relay import KEY as V1_KEY
        from .h16_relay import KEY as H16_KEY
        if (model.get_attachment(V1_KEY) is None
                and model.get_attachment(H16_KEY) is None
                and model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)):
            raise ValueError("Project external Prompt Relay onto this Chunked PASS2 segment first")
        return
    if (type(owner) is not ChunkedRelayOwner or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity or owner.segment_index != spec.index
            or owner.lifted_identity != _input_identity(lifted_segment)
            or owner.sigma_values != tuple(float(value) for value in sigmas.tolist())):
        raise ValueError("Chunked PASS2 Relay MODEL is not bound to this lifted segment")
    from .eav import KEY as EAV_KEY, StageEAVRuntime
    eav_runtime = model.get_attachment(EAV_KEY)
    if eav_runtime is None:
        contract = relay.prompt_relay_model_contract(model)
        if contract["binding_hash"] != owner.binding_hash:
            raise ValueError("Chunked PASS2 Relay MODEL binding changed after selection")
        binding = contract["binding"]
    else:
        # Stage EAV authenticates the standalone Relay MODEL before composing
        # the two wrappers, then removes that standalone attachment. Its own
        # actual-call audit is the sole combined execution certificate.
        if (type(eav_runtime) is not StageEAVRuntime or not eav_runtime.relay_required
                or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
            raise ValueError("Chunked PASS2 composed Relay/EAV MODEL is not authenticated")
        binding = positive[0][1].get(relay.PROMPT_RELAY_BINDING_KEY) if positive else None
        if not isinstance(binding, dict) or binding.get("binding_hash") != owner.binding_hash:
            raise ValueError("Chunked PASS2 composed Relay/EAV binding changed")
    _assert_paired_conditioning(positive, binding)


def audit_full_clip_relay(result, spec, runtime):
    if type(result) is not ChunkedPass2Result or type(spec) is not ChunkedSourceSegment:
        raise TypeError("Chunked Relay audit needs its matching segment result and spec")
    if type(runtime) is not ChunkedRelayRuntime or type(runtime.owner) is not ChunkedRelayOwner:
        raise TypeError("Chunked Relay audit needs its own runtime")
    owner = runtime.owner
    if (owner.plan_sha256 != spec.plan_sha256 or owner.source_identity != spec.source_identity
            or owner.segment_index != spec.index or result.plan_sha256 != spec.plan_sha256
            or result.source_identity != spec.source_identity or result.index != spec.index
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("Chunked Relay result/runtime identity mismatch")
    return result.output_latent, json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
