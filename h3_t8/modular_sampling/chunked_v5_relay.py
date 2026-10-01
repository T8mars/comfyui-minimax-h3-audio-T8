"""Project an external full-clip Prompt Relay onto one exact v5 AV window.

The stock Relay Conditioning owns text/media tokenization. Its full-clip
PackedLayout cannot be sent directly to a 136-frame v5 window, so this adapter
authenticates that pair and rebinds its event coordinates/layout to the actual
window without changing the global Plan or the legacy sampling trajectory.
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
from .chunked_v5 import StandardPrepared, StandardWindowResult, _window_input
from .chunked_relay import _assert_paired_conditioning
from .results import _input_identity


KEY = "t8_modular_chunked_v5_relay_v1"
RUNTIME_TYPE = "T8_CHUNKED_V5_RELAY_RUNTIME"


@dataclass(frozen=True)
class V5RelayOwner:
    plan_sha256: str
    source_identity: dict
    lifted_identity: dict
    piece_identity: dict
    window_index: int
    binding_hash: str
    sigma_values: tuple[float, ...]
    sampling_object: object


class V5RelayRuntime:
    def __init__(self, model, owner, blocks):
        self.model, self.owner, self.blocks = model, owner, int(blocks)
        self.baseline = {"completed_forwards": 0, "routed_attention_calls": 0}
        self.prepared = False
        self.closed = True

    def prepare(self, patcher, timestep, model_options):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("v5 Relay binding changed before PASS2")
        if self.closed:
            self.baseline = contract["execution_counts"]
            self.prepared = True
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        contract = relay.prompt_relay_model_contract(self.model)
        if contract["binding_hash"] != self.owner.binding_hash:
            raise RuntimeError("v5 Relay binding changed after PASS2")
        counts = contract["execution_counts"]
        deltas = {key: int(counts.get(key, 0)) - int(self.baseline.get(key, 0))
                  for key in self.baseline}
        forwards = len(self.owner.sigma_values) - 1
        expected = {"completed_forwards": forwards,
                    "routed_attention_calls": forwards * self.blocks}
        status = ("observed_relay_calls_quality_unverified"
                  if self.prepared and contract["attention_owner_verified"]
                  and deltas == expected else "unverified_incomplete_relay_coverage")
        return {"schema": "t8.modular-sampling.chunked-v5-relay-audit.v1",
                "status": status, "window_index": self.owner.window_index,
                "plan_sha256": self.owner.plan_sha256,
                "binding_hash": self.owner.binding_hash,
                "prepared": self.prepared, "actual_calls": deltas,
                "expected_calls": expected,
                "attention_owner_verified": contract["attention_owner_verified"],
                "cache_reuse_authorized": False, "quality_accepted": False}


def _validate_sigmas(sigmas, plan):
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or len(sigmas) < 2
            or not torch.isfinite(sigmas).all() or torch.any(sigmas[:-1] < sigmas[1:])):
        raise ValueError("v5 Relay needs finite descending PASS2 SIGMAS")
    from .. import chunked_two_pass_parity as parity
    expected = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    if sigmas.shape != expected.shape or not torch.allclose(
            sigmas.detach().to(device="cpu", dtype=torch.float32), expected, atol=1e-6, rtol=0):
        raise ValueError("v5 Relay requires the unchanged standard remaining4 SIGMAS")
    return tuple(float(value) for value in sigmas.tolist())


def _paired_projected(positive, original, projected):
    result = []
    for tensor, metadata in positive:
        updated = dict(metadata)
        if updated.get(relay.PROMPT_RELAY_BINDING_KEY) != original:
            raise ValueError("v5 Relay MODEL and CONDITIONING binding differ")
        model_conds = dict(updated.get("model_conds", {}))
        payload = model_conds.get(relay.PROMPT_RELAY_PAYLOAD_KEY)
        if type(payload) is not comfy.conds.CONDConstant or payload.cond != original["binding_hash"]:
            raise ValueError("v5 Relay CONDITIONING runtime payload differs")
        updated[relay.PROMPT_RELAY_BINDING_KEY] = projected
        model_conds[relay.PROMPT_RELAY_PAYLOAD_KEY] = comfy.conds.CONDConstant(
            projected["binding_hash"],
        )
        updated["model_conds"] = model_conds
        result.append([tensor, updated])
    return result


def project_v5_relay(raw_model, relay_model, relay_positive, relay_av_latent,
                     relay_plan, source, lifted, prepared, plan, sigmas,
                     window_index, previous_result=None):
    """Return a new matched MODEL/CONDITIONING pair for one exact AV window."""
    if type(prepared) is not StandardPrepared:
        raise ValueError("Expected the v5 global preparation")
    window = _window_input(source, lifted, prepared, plan, window_index, previous_result)
    sigma_values = _validate_sigmas(sigmas, plan)
    checked_plan = relay._validate_plan(relay_plan)
    source_video, _source_audio = nested_av_parts(source)
    if checked_plan["frame_count"] != legacy.frames_for_tokens(source_video.shape[2]):
        raise ValueError("v5 global Relay Plan must cover the complete partial4 timeline")
    contract = relay.prompt_relay_model_contract(relay_model)
    original = contract["binding"]
    if original["plan_hash"] != checked_plan["plan_hash"]:
        raise ValueError("v5 Relay MODEL belongs to another Plan")
    _assert_paired_conditioning(relay_positive, original)
    if (raw_model.model is not relay_model.model
            or raw_model.get_model_object("model_sampling") is not relay_model.get_model_object("model_sampling")):
        raise ValueError("v5 raw HIGH MODEL is not the one used by external Relay Conditioning")
    lifted_video, lifted_audio = nested_av_parts(lifted)
    relay_video, relay_audio = nested_av_parts(relay_av_latent)
    if (tuple(relay_video.shape) != tuple(lifted_video.shape)
            or tuple(relay_audio.shape) != tuple(lifted_audio.shape)):
        raise ValueError("v5 external Relay full-clip AV layout differs from global lift")
    meta = relay_positive[0][1]
    full_layout = build_packed_layout(
        original["text_len"], *relay_video.shape[2:], relay_audio.shape[-1],
        keyframes=meta.get("minimax_keyframes", []), refs=meta.get("minimax_refs", []),
        frame_count=meta.get("minimax_frame_count"),
    )
    if relay._layout_contract(full_layout) != original["layout_contract"]:
        raise ValueError("v5 external Relay full-clip PackedLayout is not authenticated")
    target = tuple(window.high_video.shape[-2:])
    local_positive = legacy.reanchor_conditioning(
        relay_positive, window.start_frame, window.end_frame, target,
    )
    local_meta = local_positive[0][1]
    for _tensor, item in local_positive:
        if (item.get("minimax_frame_count") != local_meta.get("minimax_frame_count")
                or len(item.get("minimax_keyframes", [])) != len(local_meta.get("minimax_keyframes", []))
                or len(item.get("minimax_refs", [])) != len(local_meta.get("minimax_refs", []))):
            raise ValueError("v5 Relay scheduled conditioning changed local media layout")
    video, audio = nested_av_parts(window.piece)
    local_layout = build_packed_layout(
        original["text_len"], *video.shape[2:], audio.shape[-1],
        keyframes=local_meta.get("minimax_keyframes", []),
        refs=local_meta.get("minimax_refs", []),
        frame_count=local_meta.get("minimax_frame_count"),
    )
    projected = dict(original)
    projected["events"] = [
        {**event, "midpoint": float(event["midpoint"]) - (5.0 / 3.0) * window.start_frame}
        for event in original["events"]
    ]
    # Full-clip FL2VA may become I2VA/T2VA/L2VA in one local window.
    # EAV classifies the *actual* per-window keyframes; Relay must bind the
    # same task after reanchoring. Leave otherwise unsupported layouts to the
    # existing Relay-only path so this projection does not narrow its scope.
    local_keyframes = local_meta.get("minimax_keyframes", [])
    local_refs = local_meta.get("minimax_refs", [])
    if str(original.get("task", "")).lower() in {
        "t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid",
    }:
        try:
            projected["task"] = feta._classify_visual_task(
                local_keyframes, latent_frames=video.shape[2], refs=local_refs,
            ).lower()
        except RuntimeError:
            pass
    projected["keyframe_count"] = len(local_keyframes)
    projected["reference_block_count"] = len(local_refs)
    projected["layout_contract"] = relay._layout_contract(local_layout)
    projected.pop("binding_hash")
    projected["binding_hash"] = relay._sha256_json(projected)
    paired = _paired_projected(relay_positive, original, projected)
    selected, _core_hashes = relay.patch_prompt_relay_model(
        raw_model, projected, contract["query_chunk_rows"],
    )
    verified = relay.prompt_relay_model_contract(selected)
    if verified["binding_hash"] != projected["binding_hash"]:
        raise RuntimeError("v5 projected Relay MODEL failed authentication")
    owner = V5RelayOwner(
        prepared.lift.plan_sha256, prepared.lift.source_identity,
        prepared.lift.lifted_identity, _input_identity(window.piece), window_index,
        projected["binding_hash"], sigma_values,
        raw_model.get_model_object("model_sampling"),
    )
    selected = selected.clone()
    if selected.get_attachment(KEY) is not None:
        raise ValueError("v5 Relay is already projected on this MODEL branch")
    selected.set_attachments(KEY, owner)
    runtime = V5RelayRuntime(selected, owner,
                             len(selected.get_model_object("diffusion_model").blocks))
    selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    report = runtime.snapshot()
    report.update(sampled=False, global_plan_hash=checked_plan["plan_hash"],
                  window_frames=[window.start_frame, window.end_frame],
                  window_video_tokens=[window.start, window.stop],
                  projected_layout_hash=projected["layout_contract"]["contract_hash"])
    return selected, paired, runtime, json.dumps(report, ensure_ascii=False, indent=2)


def assert_v5_relay_binding(model, positive, window, prepared, index, sigmas):
    owner = model.get_attachment(KEY)
    if owner is None:
        if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
            raise ValueError("Project full-clip Prompt Relay onto this v5 PASS2 window first")
        return
    if (type(owner) is not V5RelayOwner or type(prepared) is not StandardPrepared
            or owner.plan_sha256 != prepared.lift.plan_sha256
            or owner.source_identity != prepared.lift.source_identity
            or owner.lifted_identity != prepared.lift.lifted_identity
            or owner.piece_identity != _input_identity(window.piece)
            or owner.window_index != index
            or owner.sigma_values != tuple(float(value) for value in sigmas.tolist())
            or model.get_model_object("model_sampling") is not owner.sampling_object):
        raise ValueError("v5 PASS2 Relay MODEL is not bound to this joint AV window")
    from .eav import KEY as EAV_KEY, StageEAVRuntime
    eav_runtime = model.get_attachment(EAV_KEY)
    if eav_runtime is None:
        contract = relay.prompt_relay_model_contract(model)
        if contract["binding_hash"] != owner.binding_hash:
            raise ValueError("v5 PASS2 projected Relay binding changed")
        binding = contract["binding"]
    else:
        # Generic Stage EAV authenticates the projected Relay MODEL before
        # replacing its standalone wrapper with one combined owner. Only its
        # own audit can certify the actual joint execution.
        if (type(eav_runtime) is not StageEAVRuntime or not eav_runtime.relay_required
                or len(model.get_wrappers("diffusion_model", EAV_KEY)) != 1):
            raise ValueError("v5 PASS2 composed Relay/EAV MODEL is not authenticated")
        binding = positive[0][1].get(relay.PROMPT_RELAY_BINDING_KEY) if positive else None
        if not isinstance(binding, dict) or binding.get("binding_hash") != owner.binding_hash:
            raise ValueError("v5 PASS2 composed Relay/EAV binding changed")
    _assert_paired_conditioning(positive, binding)


def audit_v5_relay(result, prepared, runtime):
    if (type(result) is not StandardWindowResult or type(prepared) is not StandardPrepared
            or type(runtime) is not V5RelayRuntime or type(runtime.owner) is not V5RelayOwner):
        raise TypeError("v5 Relay audit needs matching PASS2 result/preparation/runtime")
    owner = runtime.owner
    if (result.index != owner.window_index or result.plan_sha256 != owner.plan_sha256
            or result.source_identity != owner.source_identity
            or result.lifted_identity != owner.lifted_identity
            or prepared.lift.plan_sha256 != owner.plan_sha256
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("v5 Relay result/runtime identity mismatch")
    return result.output_latent, json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
