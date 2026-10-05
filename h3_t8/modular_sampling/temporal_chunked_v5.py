"""Explicit scoped v5 entry point; released sampler and cache sources stay intact."""
from __future__ import annotations

from dataclasses import dataclass
import json

from . import chunked_v5
from .results import _input_identity
from .. import prompt_relay_advanced as relay
from ..temporal_dialogue_bank import select_window_conditioning
from ..temporal_dialogue_scope import DialogueReceipt, bind_published_window, verify_previous_window


@dataclass(frozen=True)
class ScopedWindowResult:
    base: chunked_v5.StandardWindowResult
    bank_sha256: str
    dialogue_receipt: DialogueReceipt


def _audio_sha(result):
    if (type(result) is not chunked_v5.StandardWindowResult
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("Scoped base AV result content changed")
    _video, audio = result.output_latent["samples"].unbind()
    return _input_identity(audio)["tensor_sha256"]


def _without_relay_pair(positive):
    # Relay may add only its known matched binding. Never strip arbitrary
    # user metadata/model_conds to make a changed condition appear unchanged.
    output = []
    for tensor, metadata in positive:
        values = dict(metadata)
        values.pop(relay.PROMPT_RELAY_BINDING_KEY, None)
        if "model_conds" in values:
            conditions = dict(values["model_conds"])
            conditions.pop(relay.PROMPT_RELAY_PAYLOAD_KEY, None)
            if conditions:
                values["model_conds"] = conditions
            else:
                values.pop("model_conds")
        output.append([tensor, values])
    return output


def sample_scoped_v5_window(model, source, lifted, prepared, plan, noise,
                            sampler, sigmas, bank, index, previous=None,
                            positive=None, negative=None, cfg=1.0):
    encoded = select_window_conditioning(bank, source, plan, index)
    if bank.executor_contract != "v5_joint_refine4":
        raise ValueError("Scoped v5 PASS2 requires the actual joint refine4 bank")
    previous_base = None
    previous_receipt = None
    previous_audio_sha = None
    if previous is not None:
        if type(previous) is not ScopedWindowResult or previous.bank_sha256 != bank.sha256:
            raise ValueError("Previous scoped window belongs to a different bank/policy")
        previous_base, previous_receipt = previous.base, previous.dialogue_receipt
        previous_audio_sha = _audio_sha(previous_base)
        if previous_base.index != previous_receipt.window.index:
            raise ValueError("Previous base result and scoped receipt window differ")
    verify_previous_window(bank.dialogue_plan, encoded.compiled.window,
                           previous_receipt, previous_audio_sha)
    selected = encoded.positive if positive is None else positive
    if bank.identity_context.describe(_without_relay_pair(selected)) != bank.identity_context.describe(encoded.positive):
        raise ValueError("Scoped PASS2 CONDITIONING does not match the precompiled native window")
    if hasattr(model, "get_attachment"):
        from .temporal_chunked_relay import assert_scoped_relay
        assert_scoped_relay(model, bank, index)
    # Core v5 still constructs the exact same AV/masks/noise and checks EAV/
    # Relay against that piece. We only replace text before its native guider.
    output, result, report_json = chunked_v5.sample_standard_window(
        model, selected, source, lifted, prepared, plan, noise, sampler, sigmas,
        index, previous_base, negative, cfg)
    scope_receipt = bind_published_window(
        bank.dialogue_plan, encoded.compiled, _audio_sha(result),
        previous_receipt, previous_audio_sha)
    scoped = ScopedWindowResult(result, bank.sha256, scope_receipt)
    report = json.loads(report_json)
    report.update(dialogue_scope={**encoded.report, "bank_sha256": bank.sha256,
                                 "receipt_sha256": scope_receipt.sha256},
                  scoped_native_conditioning_consumed=True, quality_qualified=False)
    return output, result, scoped, json.dumps(report, ensure_ascii=False, sort_keys=True)
