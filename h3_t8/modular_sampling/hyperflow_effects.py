"""External effects on a real continuous HyperFlow interval, never another plan.

Reuse the existing EAV operator and exact paired Relay router. The sampler owns
reset/lease/audit lifetime; UI Apply-node counters are not completion evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import json
import threading
import weakref

import torch

from .. import hyperflow_runtime_advanced as runtime
from .. import hyperflow_sampling_advanced as native
from .. import hyperflow_two_pass_advanced as legacy
from .. import prompt_relay_advanced as relay
from ..progressive_relay import strip_paired_conditioning
from ..patch_stack_policy import warn_patch_stack
from ..vdn_attention_compat import _factory_closure
from .contracts import StageContext
from .results import canonical

KEY = "t8_modular_hyperflow_effect_stage_v1"
RECIPE = "hyperflow.continuous_effect_stage.v1"
SCHEMA = "t8.modular-sampling.hyperflow-stage-effects.v1"
_relay_leases = weakref.WeakKeyDictionary()
_lease_lock = threading.Lock()


@dataclass(frozen=True)
class HyperFlowStageOwner:
    context: StageContext
    binding: runtime.HyperFlowBinding
    relay_binding_json: str | None
    relay_counts: dict | None
    relay_wrapper: object | None
    frozen: str
    lock: object = field(default_factory=threading.Lock, compare=False)

    @property
    def runtime(self):
        return None  # Native operator route, not a V2 sparse runtime.

    @property
    def profile(self):
        return "hyperflow_continuous"

    def descriptor(self):
        return {"context": self.context.to_dict(), "relay_binding_json": self.relay_binding_json}


def capture_owner(model):
    owner = model.get_attachment(KEY)
    if type(owner) is not HyperFlowStageOwner or set(vars(owner)) != set(HyperFlowStageOwner.__dataclass_fields__):
        raise ValueError("HyperFlow effects require their dedicated stage binding")
    if canonical(owner.descriptor()) != owner.frozen:
        raise ValueError("HyperFlow stage effect descriptor changed after binding")
    if owner.relay_binding_json is not None:
        relay_state = _factory_closure(owner.relay_wrapper, relay._install_prompt_relay_model, "_diffusion_wrapper")
        if (relay_state is None or relay_state.get("execution_counts") is not owner.relay_counts
                or canonical(relay_state.get("binding")) != owner.relay_binding_json):
            raise ValueError("HyperFlow Relay audit counter/binding differs from its real native owner")
    if model.get_attachment(runtime.ATTACHMENT_KEY) != owner.binding:
        raise ValueError("HyperFlow effects and actual two-time MODEL belong to different owners")
    plan = native.build_hyperflow_plan(model, owner.context.start, owner.context.end)
    if (owner.context.recipe != RECIPE or tuple(plan.video_sigmas) != owner.context.trajectory_sigmas
            or (owner.context.video_shift, owner.context.audio_shift) != (12., 3.)
            or owner.context.profile != canonical(plan.as_report())):
        raise ValueError("HyperFlow effects no longer match their actual trained interval plan")
    role = "head" if plan.start_interval == 0 and 1 <= plan.stop_interval <= 7 else "tail"
    if (owner.context.stage != role or (role == "tail" and not 1 <= plan.start_interval < plan.stop_interval == 8)):
        raise ValueError("HyperFlow effect stage role changed")
    return owner


def _relay_selection(model, positive, negative):
    attachment = model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY)
    if attachment is None:
        if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY) or any(
            relay.PROMPT_RELAY_BINDING_KEY in metadata
            or relay.PROMPT_RELAY_PAYLOAD_KEY in metadata.get("model_conds", {})
            for conditioning in (positive, negative) for _, metadata in conditioning):
            raise ValueError("Unpaired HyperFlow Relay MODEL/CONDITIONING markers")
        return None, None, None
    contract = relay.prompt_relay_model_contract(model)
    strip_paired_conditioning(positive, contract, required=True)
    strip_paired_conditioning(negative, contract, required=False)
    wrappers = model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    state = _factory_closure(wrappers[0], relay._install_prompt_relay_model, "_diffusion_wrapper")
    if state is None or type(state.get("execution_counts")) is not dict:
        raise ValueError("HyperFlow Relay lacks its actual native execution counter owner")
    return canonical(contract["binding"]), state["execution_counts"], wrappers[0]


def bind_stage(model, source, positive, negative, start, stop):
    if model.get_attachment(KEY) is not None:
        raise ValueError("Bind from the dedicated Loader/Relay branch, not an already bound HyperFlow stage")
    if model.get_attachment("t8_modular_stage_eav_v1") is not None:
        raise ValueError("Bind HyperFlow interval before applying external Stage EAV")
    split = stop if start == 0 else start
    legacy._validate_pair(model, model, source, split)
    if not (start == 0 and 1 <= stop <= 7 or 1 <= start < stop == 8):
        raise ValueError("HyperFlow effects require an actual continuous HEAD or TAIL")
    plan = native.build_hyperflow_plan(model, start, stop)
    video, audio = legacy.nested_av_parts(source)
    context = StageContext(RECIPE, "head" if start == 0 else "tail", canonical(plan.as_report()),
        start, stop, tuple(plan.video_sigmas), 12., 3., tuple(video.shape), tuple(audio.shape),
        "initial_av_template" if start == 0 else "exact_captured_model_space_x_sigma",
        "raw_x_sigma_and_separate_scaffold" if start == 0 else "completed_av", "not_an_upscale_x0_contract")
    relay_json, counts, wrapper = _relay_selection(model, positive, negative)
    descriptor = canonical({"context": context.to_dict(), "relay_binding_json": relay_json})
    owner = HyperFlowStageOwner(context, model.get_attachment(runtime.ATTACHMENT_KEY), relay_json, counts, wrapper, descriptor)
    clone = model.clone()
    clone.set_attachments(KEY, owner)
    capture_owner(clone)
    return clone, positive, negative, source, plan.video_segment, context, canonical({
        "schema": SCHEMA, "stage_context": context.to_dict(), "relay_bound": relay_json is not None,
        "sampled": False, "boundary": "Actual continuous interval only. Config/Bind is not effect execution proof."})


def bind_head(model, source, positive, negative, split=4):
    return bind_stage(model, source, positive, negative, 0, split)


def bind_tail(boundary, model, positive, negative):
    from .hyperflow import ContinuousBoundary
    if type(boundary) is not ContinuousBoundary:
        raise ValueError("TAIL effects require the exact continuous HEAD boundary")
    receipt = boundary.verify()
    return bind_stage(model, boundary.scaffold, positive, negative,
                      receipt["request"]["plan"]["absolute_interval"][1], 8)


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    video, audio = legacy.nested_av_parts(source)
    if context != owner.context or tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("HyperFlow effect geometry/interval differs from the bound stage")
    expected = native.build_hyperflow_plan(model, context.start, context.end).video_segment
    if (type(sigmas) is not torch.Tensor or sigmas.dtype != torch.float32
            or sigmas.shape != expected.shape or not torch.equal(sigmas.cpu(), expected)
            or source.get("noise_mask") is not None):
        raise ValueError("HyperFlow effects cannot change the continuous grid or no-mask contract")
    return video, audio, owner


def project_owner(model):
    if model.get_attachment(KEY) is None:
        return model, None
    owner = capture_owner(model)
    clone = model.clone()
    clone.remove_attachments(KEY)
    return clone, owner.descriptor()


def _relay_lock(wrapper):
    with _lease_lock:
        if wrapper not in _relay_leases:
            _relay_leases[wrapper] = threading.Lock()
        return _relay_leases[wrapper]


@contextmanager
def execution(model, plan, source, positive, negative, cfg):
    from . import eav
    owner = model.get_attachment(KEY)
    if owner is None:
        yield None
        return
    owner = capture_owner(model)
    if (owner.context.start, owner.context.end) != (plan.start_interval, plan.stop_interval):
        raise ValueError("HyperFlow effect phase differs from the sampler's actual interval")
    validate_stage(model, plan.video_segment, source, owner.context)
    telemetry = model.get_attachment(eav.KEY)
    if telemetry is not None and (type(telemetry) is not eav.StageEAVRuntime or telemetry.context != owner.context):
        raise ValueError("HyperFlow EAV owner belongs to another interval")
    if cfg != 1. and (telemetry is not None or owner.relay_binding_json is not None):
        raise ValueError("HyperFlow stage effects require native CFG1 batch-one routing")
    if owner.relay_binding_json is not None:
        contract = {"binding": json.loads(owner.relay_binding_json)}
        contract["binding_hash"] = contract["binding"]["binding_hash"]
        strip_paired_conditioning(positive, contract, required=True)
        strip_paired_conditioning(negative, contract, required=False)
    locks = [owner.lock] + ([_relay_lock(owner.relay_wrapper)] if owner.relay_wrapper is not None else [])
    acquired = []
    report = {}
    try:
        for lock in locks:
            if not lock.acquire(blocking=False):
                raise RuntimeError("HyperFlow effects owner is busy; use independent stage branches")
            acquired.append(lock)
        before = dict(owner.relay_counts or {})
        config = None
        if telemetry is not None:
            config = canonical(telemetry.snapshot()["config"])
            telemetry.closed = True
            telemetry.prepare(None, None, None)
        yield report
        capture_owner(model)
        report.update(schema=SCHEMA, phase=owner.context.stage, stage_context=owner.context.to_dict(),
                      assigned_nfe=plan.nfe, quality_accepted=False, composition_verified=True)
        if telemetry is not None:
            current = telemetry.snapshot()
            if canonical(current["config"]) != config:
                raise ValueError("HyperFlow EAV configuration changed during execution")
            report["eav"] = deepcopy(current)
            report["composition_verified"] &= current["status"] in {
                "observed_report_only", "observed_apply_exp", "observed_no_steps_in_effect_window"}
        if owner.relay_counts is not None:
            counts = {key: value - before.get(key, 0) for key, value in owner.relay_counts.items()}
            forward = telemetry.completed_forwards if telemetry is not None and telemetry.relay_required else counts["completed_forwards"]
            wanted = plan.nfe * len(model.model.diffusion_model.blocks)
            verified = forward == plan.nfe and counts["routed_attention_calls"] == wanted
            report["relay"] = {"binding": json.loads(owner.relay_binding_json), "completed_forwards": forward,
                "routed_attention_calls": counts["routed_attention_calls"], "planned_attention_calls": wanted,
                "composition_verified": verified}
            report["composition_verified"] &= verified
        if not report["composition_verified"]:
            warn_patch_stack("HyperFlow stage effect actual execution coverage is incomplete")
    except BaseException as error:
        if telemetry is not None and acquired:
            telemetry.telemetry.abort(error)
        raise
    finally:
        for lock in reversed(acquired):
            lock.release()


def audit(result, phase):
    from .hyperflow import ContinuousBoundary, ContinuousResult
    if phase == "head" and type(result) is ContinuousBoundary:
        report = result.verify()["execution"].get("effects")
    elif phase == "tail" and type(result) is ContinuousResult:
        report = result.verify()["sampling"]["execution"].get("effects")
    else:
        raise ValueError("Connect the matching completed HyperFlow typed phase result")
    if report is None:
        return {"schema": SCHEMA, "phase": phase, "status": "no_external_stage_binding", "quality_accepted": False}
    if report.get("schema") != SCHEMA or report.get("phase") != phase:
        raise ValueError("HyperFlow completed effects report has another phase")
    return deepcopy(report)
