"""Recording-bound Avatar stages; reuse native Progressive mathematics exactly.

Binding is an explicit user-selected encoded-AV/PCM pair, not a claim that a
particular VAE produced those bytes. The completed result retains a small audio
anchor for audit; delivery can load HIGH and the original PCM without encoding
or executing LOW again. No recording is silently substituted into model output.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..core import validate_audio
from . import progressive as stages
from .progressive_high_result import ProgressiveHighResult

KEY = "t8_modular_avatar_source_v1"
DELIVERY = "t8_modular_avatar_delivery_v1"
SCHEMA = "t8.modular-sampling.avatar-source.v1"


def _recording(value):
    waveform, _ = validate_audio(value, "Avatar original recording")
    stages.masks._finite(waveform, "Avatar original PCM")
    if type(value["sample_rate"]) is not int or value["sample_rate"] <= 0:
        raise ValueError("Avatar recording requires a positive integer sample rate")
    return stages.snapshot(value)


def _locked(source):
    if not isinstance(source, dict) or "samples" not in source:
        raise ValueError("Avatar requires encoded native AV")
    video, audio = stages.masks._av_parts(source["samples"], "Avatar source")
    mask = source.get("noise_mask")
    if not getattr(mask, "is_nested", False) or len(mask.unbind()) != 2:
        raise ValueError("Avatar requires explicit nested video/audio masks")
    normalized = stages.masks.normalize_av_masks(mask, video, audio)
    if bool(torch.count_nonzero(normalized.unbind()[1])):
        raise ValueError("Avatar requires audio mask=0 throughout")
    return video, audio, normalized


def _implementation():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class AvatarSource:
    source: dict
    recording: dict
    contract_json: str

    def verify(self):
        contract = json.loads(self.contract_json)
        if type(contract) is not dict or set(contract) != {
                "schema", "source", "recording", "encoded_audio", "implementation", "sha256"}:
            raise ValueError("Unknown Avatar source contract")
        unsigned = dict(contract)
        digest = unsigned.pop("sha256")
        _, audio, _ = _locked(self.source)
        if (contract["schema"] != SCHEMA or stages.sha(unsigned) != digest
                or self.source.get(KEY) != digest or DELIVERY in self.source
                or contract["source"] != stages.snapshot({k: v for k, v in self.source.items() if k != KEY})
                or contract["recording"] != _recording(self.recording)
                or contract["encoded_audio"] != stages._input_identity(audio)
                or contract["implementation"] != _implementation()):
            raise ValueError("Avatar source/recording/implementation changed")
        return contract


def bind_source(high_source, original_recording):
    _, audio, _ = _locked(high_source)
    if KEY in high_source or DELIVERY in high_source:
        raise ValueError("Avatar source is already bound; connect the original encoded source")
    contract = {"schema": SCHEMA, "source": stages.snapshot(high_source),
        "recording": _recording(original_recording), "encoded_audio": stages._input_identity(audio),
        "implementation": _implementation()}
    contract["sha256"] = stages.sha(contract)
    context = AvatarSource({**high_source, KEY: contract["sha256"]}, original_recording, stages.canonical(contract))
    context.verify()
    return context


def sample_low(avatar_source, model, sampler, plan, low_source, positive, negative, noise, **options):
    if type(avatar_source) is not AvatarSource:
        raise ValueError("Connect a bound Avatar source")
    contract = avatar_source.verify()
    _locked(low_source)
    expected = stages.prepare_low_source(avatar_source.source, plan, "initialized_av_exp")
    if stages.snapshot(low_source) != stages.snapshot(expected):
        raise ValueError("Avatar LOW source differs from its bound initialized source")
    boundary = stages.sample_low(model, sampler, plan, low_source, positive, negative, noise, **options)
    avatar_source.verify()
    receipt = boundary.verify()
    # Independent of generic input snapshots: unknown condition providers must
    # remain executable, without assuming a portable snapshot's internal shape.
    receipt["request"][KEY] = contract
    receipt["request_sha256"] = stages.sha(receipt["request"])
    receipt["portable_identity"] = bool(stages.portable(receipt["request"])
        and receipt["execution"]["actual_apply_calls"] >= plan.low_evaluations)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = stages.sha(receipt)
    result = stages.ProgressiveBoundary(boundary.tensors, stages.canonical(receipt))
    result.verify()
    return result


def prepare_high(avatar_source, low_boundary, model, sampler, lifted_av, video_noise, *,
                 high_source=None, high_sigmas=None):
    if type(avatar_source) is not AvatarSource:
        raise ValueError("Connect a bound Avatar source")
    contract = avatar_source.verify()
    receipt = low_boundary.verify()
    if receipt["request"].get(KEY) != contract:
        raise ValueError("Frozen LOW is not bound to this Avatar source/recording")
    selected = avatar_source.source if high_source is None else high_source
    _, audio, mask = _locked(selected)
    if stages._input_identity(audio) != contract["encoded_audio"]:
        raise ValueError("Avatar HIGH clean audio must retain the bound recording anchor")
    if DELIVERY in selected or (KEY in selected and selected[KEY] != contract["sha256"]):
        raise ValueError("Avatar HIGH contains a conflicting source binding")
    marker = {"source_contract": contract, "low_receipt_sha256": receipt["receipt_sha256"],
              "audio_anchor": audio.detach().clone(), "audio_mask": stages._input_identity(mask.unbind()[1])}
    source = {**selected, KEY: contract["sha256"], DELIVERY: marker}
    result = stages.prepare_high(low_boundary, model, sampler, lifted_av, source, video_noise,
                                 high_sigmas=high_sigmas)
    avatar_source.verify()
    return result


def deliver(high_result, original_recording):
    if type(high_result) is not ProgressiveHighResult:
        raise ValueError("Avatar delivery requires a completed typed HIGH result")
    receipt = high_result.verify()
    marker = high_result.output.get(DELIVERY)
    if type(marker) is not dict or set(marker) != {
            "source_contract", "low_receipt_sha256", "audio_anchor", "audio_mask"}:
        raise ValueError("Completed HIGH has no Avatar delivery binding")
    contract = marker["source_contract"]
    unsigned = dict(contract)
    digest = unsigned.pop("sha256", None)
    if (contract.get("schema") != SCHEMA or stages.sha(unsigned) != digest
            or high_result.output.get(KEY) != digest or contract.get("recording") != _recording(original_recording)):
        raise ValueError("Avatar delivery original recording does not match the frozen source")
    audio = marker["audio_anchor"]
    stages.masks._finite(audio, "Avatar frozen audio anchor")
    restart = receipt["sampling"]["restart"]
    metadata = {key: value for key, value in high_result.output.items() if key not in ("samples", "noise_mask")}
    audio_id = stages._input_identity(audio)
    if (audio_id != contract["encoded_audio"] or audio_id != restart["tensors"]["anchor"]["native_av"][1]
            or marker["audio_mask"] != restart["tensors"]["mask"]["native_av"][1]
            or marker["audio_mask"] != stages._input_identity(torch.zeros_like(audio, dtype=torch.float32))
            or restart["metadata"] != stages.snapshot(metadata)
            or marker["low_receipt_sha256"] != restart["low_receipt_sha256"]):
        raise ValueError("Avatar completed HIGH clean-anchor binding changed")
    actual = high_result.output["samples"].unbind()[1]
    difference = float((actual.to(device="cpu", dtype=torch.float64)
                        - audio.to(device="cpu", dtype=torch.float64)).abs().max())
    report = {"schema": "t8.modular-sampling.avatar-delivery.v1", "source_sha256": digest,
        "low_receipt_sha256": marker["low_receipt_sha256"], "high_receipt_sha256": receipt["receipt_sha256"],
        "source_audio_latent_max_abs_difference": difference, "audio_mask_zero": True,
        "original_recording_returned_unchanged": True, "sampling_calls": 0, "encoding_calls": 0,
        "recording_binding": "explicit_selected_pair_not_VAE_provenance", "voice_cloning": False,
        "trained_model_quality_qualified": False, "long_video_qualified": False}
    return high_result.output, original_recording, stages.canonical(report)
