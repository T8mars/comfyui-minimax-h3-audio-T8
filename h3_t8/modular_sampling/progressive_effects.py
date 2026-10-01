"""Opt-in effects for ONE native Progressive stage, never a two-stage runner.

The native stage owns the execution lifetime, so cancellation and a cached
Apply node cannot leak telemetry into a later sample. Persistent executable
identity is separately authenticated, never inferred from this descriptor.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import threading

import torch

from .. import progressive_relay as relay_stages
from .. import progressive_eav as eav_stages
from .. import prompt_relay_advanced as relay
from ..progressive_eav_masks import NativeProgressiveMaskContract
from ..patch_stack_policy import warn_patch_stack
from . import progressive as stages
from .eav import EAVConfig

KEY = "t8_modular_progressive_effects_v1"
SCHEMA = "t8.modular-sampling.progressive-effects.v1"


def _phase(plan, phase):
    stages.validate_plan(plan)
    if phase not in ("low", "high"):
        raise ValueError("Unknown Progressive effects phase")
    return plan.low_evaluations if phase == "low" else plan.high_evaluations


class StageEffects:
    def __init__(self, plan, phase, *, relay_report=None, eav_runtime=None, mask=None, continuation=None):
        _phase(plan, phase)
        self.plan, self.phase = plan, phase
        self.relay_report, self.eav_runtime = relay_report, eav_runtime
        self.mask = mask
        self.continuation = continuation
        self.lock = threading.Lock()
        self.last_report = None
        # No counter/config alias escapes into the immutable expected values.
        self.binding = stages.canonical(self.descriptor())

    def descriptor(self):
        relay = None if self.relay_report is None else {
            key: value for key, value in self.relay_report.items() if key != "completed_calls"}
        config = None if self.eav_runtime is None else {
            key: value for key, value in self.eav_runtime.config.items()
            if key != "composed_attention_backend"}  # backend telemetry changes per forward
        result = {"plan": asdict(self.plan), "phase": self.phase, "relay": relay, "eav": config,
                  "mask": None if self.mask is None else self.mask.report()}
        if self.continuation is not None:
            from .continuation import _checked_phase
            _checked_phase(self.continuation, self.phase, self.plan)
            result["continuation"] = self.continuation.verify()
        return result

    def verify(self):
        if stages.canonical(self.descriptor()) != self.binding:
            raise ValueError("Progressive stage effect configuration changed after binding")

    def reset(self):
        if self.relay_report is not None:
            self.relay_report["completed_calls"].update(forward=0, routed_attention=0)
        if self.eav_runtime is not None:
            # This adapter owns only this freshly built runtime. A consumed
            # snapshot alone leaves old entries until the first new forward;
            # explicitly clear them so an early abort cannot report old work.
            with self.eav_runtime._lock:
                self.eav_runtime._consumed = True
                self.eav_runtime._aborted = None
                self.eav_runtime._forwards = []
        self.last_report = None

    def report(self, model):
        from .progressive_effect_identity import project
        from ..patch_stack_policy import UnverifiedModelStack
        expected = _phase(self.plan, self.phase)
        try:
            _, effect_identity = project(model)
        except UnverifiedModelStack:
            effect_identity = None
        result = {"schema": SCHEMA, "phase": self.phase, "plan": asdict(self.plan),
                  "assigned_nfe": expected, "quality_accepted": False,
                  "portable_effect_identity": effect_identity is not None}
        verified = True
        if self.relay_report is not None:
            result["relay"] = deepcopy(self.relay_report)
            counts = result["relay"]["completed_calls"]
            wanted_attention = expected * len(model.model.diffusion_model.blocks)
            relay_verified = counts == {"forward": expected, "routed_attention": wanted_attention}
            result["relay"]["composition_verified"] = relay_verified
            if not relay_verified:
                warn_patch_stack("Progressive stage Relay actual forward/attention coverage is incomplete")
            verified &= relay_verified
        if self.eav_runtime is not None:
            result["eav"] = eav_stages.audit_progressive_eav_stage(
                self.eav_runtime, self.plan, self.phase, model)
            verified &= result["eav"]["composition_verified"]
        result["status"] = "verified_stage_execution_quality_unverified" if verified else "executed_user_stack_unverified"
        return result


def owner(model):
    value = model.get_attachment(KEY)
    if value is not None and type(value) is not StageEffects:
        raise ValueError("Unknown Progressive stage effects owner")
    return value


def apply_relay(model, positive, negative, plan, phase, *, mode="apply_exp", guide_resize="legacy_bilinear"):
    """Use this phase's own external Relay plan/model/conditions, not LOW's plan."""
    _phase(plan, phase)
    if mode not in ("disabled", "apply_exp"):
        raise ValueError("Unknown Progressive Relay mode")
    if owner(model) is not None:
        raise ValueError("Apply Progressive Relay before its EAV adapter; do not stack phase owners")
    base, contract = relay_stages.detach_relay_input(model)
    if contract is None:
        selector = model.model_options.get("transformer_options", {}).get("optimized_attention_override")
        residual = bool(model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY))
        residual |= hasattr(selector, "_t8_prompt_relay_binding_hash")
        for conditioning in (positive, negative):
            residual |= any(relay.PROMPT_RELAY_BINDING_KEY in metadata
                or relay.PROMPT_RELAY_PAYLOAD_KEY in metadata.get("model_conds", {})
                for _, metadata in conditioning)
        if residual:
            raise ValueError("Unpaired Progressive Relay MODEL/CONDITIONING markers")
        # The original encoder returns an unpatched MODEL for report-only and
        # zero/single events. Preserve that real no-op, including user patches.
    else:
        positive = relay_stages.strip_paired_conditioning(positive, contract, required=True)
        negative = relay_stages.strip_paired_conditioning(negative, contract, required=False)
    index = 0 if phase == "low" else 1
    pos = stages.legacy.prepare_stage_conditioning(positive, plan, positive=True, guide_resize=guide_resize)[index]
    neg = stages.legacy.prepare_stage_conditioning(negative, plan, positive=False, guide_resize=guide_resize)[index]
    if mode == "disabled" or contract is None:
        return base, pos, neg
    selected, pos, neg, _, report = relay_stages.prepare_relay_stage(
        base, contract, pos, neg, plan, low=phase == "low", backend_contract=contract)
    selected.set_attachments(KEY, StageEffects(plan, phase, relay_report=report))
    return selected, pos, neg


def apply_eav(model, config, plan, phase, source):
    """Source is LOW's clean LATENT or HIGH's typed restart (never noisy anchor)."""
    _phase(plan, phase)
    if type(config) is not EAVConfig:
        raise ValueError("Expected the external Stage EAV Config")
    if config.mode == "disabled":
        return model
    current = owner(model)
    from ..prompt_relay_advanced import PROMPT_RELAY_WRAPPER_KEY
    if current is None and model.get_attachment(PROMPT_RELAY_WRAPPER_KEY) is not None:
        raise ValueError("Bind external Relay with Progressive Relay Stage Apply before applying Progressive EAV")
    if current is not None:
        current.verify()
        if current.plan != plan or current.phase != phase or current.eav_runtime is not None:
            raise ValueError("Progressive EAV and Relay must belong to the same single stage")
    if phase == "high":
        if type(source) is not stages.ProgressiveRestart:
            raise ValueError("HIGH EAV requires its typed restart, not an ordinary LATENT")
        source.verify()
        if source.plan != plan:
            raise ValueError("HIGH EAV plan differs from the restart")
        source = {"samples": source.tensors["anchor"], "noise_mask": source.tensors["mask"]}
    video, audio = stages._geometry(source, plan, phase)
    normalized = stages.masks.normalize_av_masks(source.get("noise_mask"), video, audio)
    mask = NativeProgressiveMaskContract(model, normalized, video.shape, audio.shape)
    relay_report = deepcopy(current.relay_report) if current is not None else None
    schedule = torch.tensor(plan.sigmas, dtype=getattr(torch, plan.sigma_dtype.split(".")[-1]))
    selected, telemetry = eav_stages.prepare_progressive_eav(model, schedule, plan,
        mode=config.mode, tau=config.tau, start=config.start_video_progress, end=config.end_video_progress,
        workspace=config.max_workspace_mib, hard_limit=config.g_hard_limit,
        relay_report=relay_report, mask_contract=mask)
    selected.set_attachments(KEY, StageEffects(plan, phase, relay_report=relay_report, eav_runtime=telemetry, mask=mask))
    return selected


@contextmanager
def execution(model, plan, phase, latent, cfg):
    """Only the explicit stage sampler calls this; an Apply node never samples."""
    runtime = owner(model)
    if runtime is None:
        yield None
        return
    runtime.verify()
    if runtime.plan != plan or runtime.phase != phase:
        raise ValueError("Progressive effects are connected to a different sampling stage")
    if cfg != 1.:
        raise ValueError("Progressive stage effects require native CFG1 batch-one routing")
    if runtime.mask is not None:
        video, audio = stages._geometry(latent, plan, phase)
        normalized = stages.masks.normalize_av_masks(latent.get("noise_mask"), video, audio)
        accepted = runtime.continuation.contexts.source if runtime.continuation is not None else None
        actual = NativeProgressiveMaskContract(model, normalized, video.shape, audio.shape, accepted_source=accepted)
        if actual.report() != runtime.mask.report():
            raise ValueError("Progressive effect mask differs from the actual sampling mask")
    if not runtime.lock.acquire(blocking=False):
        raise RuntimeError("This Progressive effects owner is already executing; use independent stage branches")
    report = {}
    try:
        runtime.reset()
        yield report
        runtime.verify()
        report.update(runtime.report(model))
        runtime.last_report = deepcopy(report)
    except BaseException as error:
        if runtime.eav_runtime is not None:
            runtime.eav_runtime.abort(error)
        runtime.last_report = {"schema": SCHEMA, "phase": phase, "status": "aborted",
                               "error": f"{type(error).__name__}: {error}", "quality_accepted": False}
        raise
    finally:
        runtime.lock.release()


def audit(result, phase):
    """Read the immutable sampled result, not a mutable Apply-node counter."""
    if phase == "low" and type(result) is stages.ProgressiveBoundary:
        data = result.verify()
    else:
        from .progressive_high_result import ProgressiveHighResult
        if phase != "high" or type(result) is not ProgressiveHighResult:
            raise ValueError("Connect the completed result of the selected Progressive stage")
        data = result.verify()["sampling"]
    report = data["execution"].get("effects")
    if report is None:
        return {"schema": SCHEMA, "phase": phase, "status": "no_external_stage_effects",
                "quality_accepted": False}
    if report.get("schema") != SCHEMA or report.get("phase") != phase:
        raise ValueError("Progressive effects report has the wrong phase/schema")
    return deepcopy(report)
