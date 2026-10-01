"""Project a stock full-clip Prompt Relay pair onto one v1 temporal chunk.

Only the full-frame, locked-input-audio v1 contract is covered. The old
all-in-one sampler and its per-chunk anchor/merge math remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import comfy.patcher_extension as extension

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import enhance_a_video_advanced as feta
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..sampling import nested_av_parts
from .chunked_relay import _assert_paired_conditioning
from .chunked_stages import ChunkedPass2Result, _check_segment
from .h16_relay import _paired_projected, _sigma_values
from .results import _input_identity


KEY = "t8_modular_chunked_v1_local_relay_v1"
RUNTIME_TYPE = "T8_CHUNKED_V1_LOCAL_RELAY_RUNTIME"


@dataclass(frozen=True)
class V1RelayOwner:
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    previous_identity: dict | None
    segment_index: int
    binding_hash: str
    sigma_values: tuple[float, ...]
    sampling_object: object


class V1RelayRuntime:
    def __init__(self, model, owner, blocks):
        self.model, self.owner, self.blocks = model, owner, int(blocks)
        self.baseline = {"completed_forwards": 0, "routed_attention_calls": 0}
        self.prepared = False
        self.closed = True

    def prepare(self, patcher, timestep, model_options):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("Chunked v1 local Relay binding changed before PASS2")
        if self.closed:
            self.baseline = contract["execution_counts"]
            self.prepared = True
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("Chunked v1 local Relay binding changed after PASS2")
        counts = contract["execution_counts"]
        actual = {key: int(counts.get(key, 0)) - int(self.baseline.get(key, 0))
                  for key in self.baseline}
        forwards = len(self.owner.sigma_values) - 1
        expected = {"completed_forwards": forwards,
                    "routed_attention_calls": forwards * self.blocks}
        status = ("observed_relay_calls_quality_unverified"
                  if self.prepared and contract["attention_owner_verified"] and actual == expected
                  else "unverified_incomplete_relay_coverage")
        return {"schema": "t8.modular-sampling.chunked-v1-local-relay-audit.v1",
                "status": status, "segment_index": self.owner.segment_index,
                "plan_sha256": self.owner.plan_sha256,
                "binding_hash": self.owner.binding_hash,
                "prepared": self.prepared, "actual_calls": actual, "expected_calls": expected,
                "attention_owner_verified": contract["attention_owner_verified"],
                "cache_reuse_authorized": False, "quality_accepted": False}


def _v1_inputs(source_segment, lifted_segment, spec, context, plan, previous):
    _check_segment(source_segment, spec, context, plan)
    if (plan["schema"] != legacy.PLAN_SCHEMA_V1
            or plan.get("spatial_strategy") != "full_frame_safe"
            or plan.get("temporal_merge_policy") == legacy.TEMPORAL_OWNERSHIP_POLICY
            or plan.get("second_pass_audio_policy", "locked_input_audio") != "locked_input_audio"):
        raise ValueError("Chunked v1 local Relay requires full-frame, locked-audio v1 PASS2")
    source_video, source_audio = nested_av_parts(source_segment)
    video, lifted_audio = nested_av_parts(lifted_segment)
    if (tuple(video.shape[:3]) != tuple(source_video.shape[:3])
            or tuple(video.shape[-2:]) != (
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE,
            ) or tuple(lifted_audio.shape) != tuple(source_audio.shape)):
        raise ValueError("Chunked v1 local Relay lifted AV layout differs from this segment")
    previous_identity = None
    accumulated = None
    if spec.index:
        if (type(previous) is not ChunkedPass2Result or previous.index != spec.index - 1
                or previous.count != spec.count or previous.plan_sha256 != spec.plan_sha256
                or previous.source_identity != spec.source_identity
                or _input_identity(previous.output_latent) != previous.output_identity):
            raise ValueError("Chunked v1 local Relay needs the exact prior PASS2 result")
        accumulated, previous_audio = nested_av_parts(previous.output_latent)
        if previous_audio is not context.original_audio:
            raise ValueError("Chunked v1 prior PASS2 result lost the original full audio")
        previous_identity = previous.output_identity
    elif previous is not None:
        raise ValueError("First Chunked v1 segment cannot have a prior PASS2 result")
    return video, source_audio, accumulated, previous_identity


def project_v1_local_relay(raw_model, relay_model, relay_positive, relay_full_av_latent,
                           relay_plan, source_segment, lifted_segment, spec, context,
                           plan, sigmas, previous_result=None):
    """Return the paired local MODEL/CONDITIONING for exactly one v1 chunk."""
    video, audio, accumulated, previous_identity = _v1_inputs(
        source_segment, lifted_segment, spec, context, plan, previous_result,
    )
    sigma_values = _sigma_values(sigmas)
    checked_plan = relay._validate_plan(relay_plan)
    full_video, full_audio = nested_av_parts(context.source_latent)
    if checked_plan["frame_count"] != legacy.frames_for_tokens(full_video.shape[2]):
        raise ValueError("Chunked v1 Relay Plan must cover the complete source timeline")
    contract = relay.prompt_relay_model_contract(relay_model)
    original = contract["binding"]
    if original["plan_hash"] != checked_plan["plan_hash"]:
        raise ValueError("Chunked v1 Relay MODEL belongs to another Plan")
    _assert_paired_conditioning(relay_positive, original)
    if (raw_model.model is not relay_model.model
            or raw_model.get_model_object("model_sampling") is not
            relay_model.get_model_object("model_sampling")):
        raise ValueError("Chunked v1 raw HIGH MODEL differs from external Relay Conditioning")
    relay_video, relay_audio = nested_av_parts(relay_full_av_latent)
    if (tuple(relay_video.shape[:3]) != tuple(full_video.shape[:3])
            or tuple(relay_video.shape[-2:]) != tuple(video.shape[-2:])
            or tuple(relay_audio.shape) != tuple(full_audio.shape)):
        raise ValueError("Chunked v1 Relay full-clip AV layout differs from target/source")
    full_meta = relay_positive[0][1]
    for _, item in relay_positive:
        if (item.get("minimax_frame_count") != full_meta.get("minimax_frame_count")
                or len(item.get("minimax_keyframes", [])) !=
                len(full_meta.get("minimax_keyframes", []))
                or len(item.get("minimax_refs", [])) != len(full_meta.get("minimax_refs", []))):
            raise ValueError("Chunked v1 Relay scheduled CONDITIONING changed full media layout")
    if (full_meta.get("minimax_keyframes")
            and full_meta.get("minimax_frame_count") != checked_plan["frame_count"]):
        raise ValueError("Chunked v1 Relay keyframe clock differs from full Plan")
    full_layout = build_packed_layout(
        original["text_len"], *relay_video.shape[2:], relay_audio.shape[-1],
        keyframes=full_meta.get("minimax_keyframes", []),
        refs=full_meta.get("minimax_refs", []),
        frame_count=full_meta.get("minimax_frame_count"),
    )
    if relay._layout_contract(full_layout) != original["layout_contract"]:
        raise ValueError("Chunked v1 Relay full-clip PackedLayout is not authenticated")
    local_positive = legacy.reanchor_conditioning(
        relay_positive, spec.start_frame, spec.end_frame, tuple(video.shape[-2:]),
    )
    if accumulated is not None:
        local_positive = legacy.anchor_conditioning(
            local_positive, accumulated, spec.start_frame, plan["anchor_strength"],
        )
    local_positive = legacy.crop_conditioning(
        local_positive, *video.shape[-2:], 0, 0, *video.shape[-2:],
    )
    local_meta = local_positive[0][1]
    for _, item in local_positive:
        if (item.get("minimax_frame_count") != local_meta.get("minimax_frame_count")
                or len(item.get("minimax_keyframes", [])) !=
                len(local_meta.get("minimax_keyframes", []))
                or len(item.get("minimax_refs", [])) != len(local_meta.get("minimax_refs", []))):
            raise ValueError("Chunked v1 Relay scheduled CONDITIONING changed local media layout")
    local_layout = build_packed_layout(
        original["text_len"], *video.shape[2:], audio.shape[-1],
        keyframes=local_meta.get("minimax_keyframes", []),
        refs=local_meta.get("minimax_refs", []),
        frame_count=local_meta.get("minimax_frame_count"),
    )
    local_keyframes = local_meta.get("minimax_keyframes", [])
    local_refs = local_meta.get("minimax_refs", [])
    projected = dict(original)
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
    selected, _ = relay.patch_prompt_relay_model(
        raw_model, projected, contract["query_chunk_rows"],
    )
    if relay.prompt_relay_model_contract(selected)["binding_hash"] != projected["binding_hash"]:
        raise RuntimeError("Chunked v1 projected Relay MODEL failed authentication")
    owner = V1RelayOwner(
        spec.plan_sha256, spec.source_identity, _input_identity(lifted_segment),
        previous_identity, spec.index, projected["binding_hash"], sigma_values,
        raw_model.get_model_object("model_sampling"),
    )
    selected = selected.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("Chunked v1 Relay is already projected on this MODEL branch")
    selected.set_attachments(KEY, owner)
    runtime = V1RelayRuntime(
        selected, owner, len(selected.get_model_object("diffusion_model").blocks),
    )
    selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    report = runtime.snapshot()
    report.update(sampled=False, global_plan_hash=checked_plan["plan_hash"],
                  segment_frames=[spec.start_frame, spec.end_frame],
                  segment_video_tokens=[spec.start_token, spec.end_token],
                  previous_anchor_inserted=accumulated is not None,
                  projected_layout_hash=projected["layout_contract"]["contract_hash"])
    return selected, paired, runtime, json.dumps(report, ensure_ascii=False, indent=2)


def assert_v1_local_relay_binding(model, positive, source_segment, lifted_segment,
                                  spec, context, plan, previous_result, sigmas):
    owner = model.get_attachment(KEY)
    if owner is None:
        return
    _video, _audio, _accumulated, previous_identity = _v1_inputs(
        source_segment, lifted_segment, spec, context, plan, previous_result,
    )
    if (type(owner) is not V1RelayOwner or owner.plan_sha256 != spec.plan_sha256
            or owner.source_identity != spec.source_identity
            or owner.lifted_identity != _input_identity(lifted_segment)
            or owner.previous_identity != previous_identity
            or owner.segment_index != spec.index
            or owner.sigma_values != _sigma_values(sigmas)
            or model.get_model_object("model_sampling") is not owner.sampling_object):
        raise ValueError("Chunked v1 PASS2 Relay MODEL is not bound to this exact segment")
    from .eav import KEY as EAV_KEY, StageEAVRuntime
    eav_runtime = model.get_attachment(EAV_KEY)
    if eav_runtime is None:
        contract = relay.prompt_relay_model_contract(model)
        if contract["binding_hash"] != owner.binding_hash:
            raise ValueError("Chunked v1 projected Relay binding changed")
        binding = contract["binding"]
    else:
        if (type(eav_runtime) is not StageEAVRuntime or not eav_runtime.relay_required
                or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
            raise ValueError("Chunked v1 composed Relay/EAV MODEL is not authenticated")
        binding = positive[0][1].get(relay.PROMPT_RELAY_BINDING_KEY) if positive else None
        if not isinstance(binding, dict) or binding.get("binding_hash") != owner.binding_hash:
            raise ValueError("Chunked v1 composed Relay/EAV binding changed")
    _assert_paired_conditioning(positive, binding)


def audit_v1_local_relay(result, spec, runtime):
    if (type(result) is not ChunkedPass2Result or type(runtime) is not V1RelayRuntime
            or type(runtime.owner) is not V1RelayOwner):
        raise TypeError("Chunked v1 Relay audit needs its matching segment result/runtime")
    owner = runtime.owner
    if (owner.plan_sha256 != spec.plan_sha256 or owner.source_identity != spec.source_identity
            or owner.segment_index != spec.index or result.index != spec.index
            or result.plan_sha256 != spec.plan_sha256
            or result.source_identity != spec.source_identity
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("Chunked v1 Relay result/runtime identity mismatch")
    return result.output_latent, json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
