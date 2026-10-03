"""External effects for the distinct fitted-curve continuous stage contract."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import json
import threading

import torch

from .. import hyperflow_curve_runtime_exp as runtime
from ..progressive_relay import strip_paired_conditioning
from ..patch_stack_policy import warn_patch_stack
from .contracts import StageContext
from .hyperflow_effects import _relay_selection, _relay_lock
from .results import canonical

KEY = "t8_hyperflow_curve_effect_stage_v1"
RECIPE = "hyperflow.curve_continuous_effect_stage.v1"
SCHEMA = "t8.hyperflow.curve_stage_effects.v1"


@dataclass(frozen=True)
class CurveStageOwner:
    context: StageContext
    binding: runtime.CurveBinding
    relay_binding_json: str | None
    relay_counts: dict | None
    relay_wrapper: object | None
    frozen: str
    lock: object = field(default_factory=threading.Lock, compare=False)

    @property
    def runtime(self):
        return None

    @property
    def profile(self):
        return "hyperflow_curve_continuous"

    def descriptor(self):
        return {"context": self.context.to_dict(), "relay_binding_json": self.relay_binding_json}


def capture_owner(model):
    from .. import prompt_relay_advanced as relay
    from ..vdn_attention_compat import _factory_closure
    owner = model.get_attachment(KEY)
    if type(owner) is not CurveStageOwner or set(vars(owner)) != set(CurveStageOwner.__dataclass_fields__):
        raise ValueError("Curve effects need their dedicated stage owner")
    if canonical(owner.descriptor()) != owner.frozen or model.get_attachment(runtime.KEY) != owner.binding:
        raise ValueError("Curve stage effect owner or actual fit binding changed")
    plan = runtime.build_plan(model, owner.context.start, owner.context.end)
    context = owner.context
    role = "head" if plan.start_interval == 0 else "tail"
    if (context.recipe != RECIPE or context.stage != role or context.profile != canonical(plan.as_report())
            or tuple(plan.video_sigmas) != context.trajectory_sigmas or (context.video_shift, context.audio_shift) != (12., 3.)
            or not (plan.start_interval == 0 and 1 <= plan.stop_interval <= 7
                    or 1 <= plan.start_interval < plan.stop_interval == 8)):
        raise ValueError("Curve effect context is not its actual HEAD/TAIL interval")
    if owner.relay_binding_json is not None:
        state = _factory_closure(owner.relay_wrapper, relay._install_prompt_relay_model, "_diffusion_wrapper")
        if (state is None or state.get("execution_counts") is not owner.relay_counts
                or canonical(state.get("binding")) != owner.relay_binding_json):
            raise ValueError("Curve Relay has another actual counter/binding owner")
    return owner


def bind_stage(model, source, positive, negative, start, stop):
    from .hyperflow_curve import _source
    if model.get_attachment(KEY) is not None or model.get_attachment("t8_modular_stage_eav_v1") is not None:
        raise ValueError("Bind from Curve Loader/paired Relay before external Stage EAV")
    if not (start == 0 and 1 <= stop <= 7 or 1 <= start < stop == 8):
        raise ValueError("Curve effects need an explicit continuous HEAD or TAIL")
    plan = runtime.build_plan(model, start, stop)
    video, audio = _source(source)
    context = StageContext(RECIPE, "head" if start == 0 else "tail", canonical(plan.as_report()),
        start, stop, tuple(plan.video_sigmas), 12., 3., tuple(video.shape), tuple(audio.shape),
        "initial_av_template" if start == 0 else "exact_curve_captured_x_sigma",
        "curve_raw_x_sigma_and_separate_scaffold" if start == 0 else "completed_curve_av",
        "not_an_upscale_or_full_hyperflow_contract")
    binding, counts, wrapper = _relay_selection(model, positive, negative)
    descriptor = canonical({"context": context.to_dict(), "relay_binding_json": binding})
    owner = CurveStageOwner(context, model.get_attachment(runtime.KEY), binding, counts, wrapper, descriptor)
    clone = model.clone()
    clone.set_attachments(KEY, owner)
    capture_owner(clone)
    return clone, positive, negative, source, plan.video_segment, context, canonical({"schema": SCHEMA,
        "stage_context": context.to_dict(), "relay_bound": binding is not None, "sampled": False})


def bind_head(model, source, positive, negative, split=4):
    return bind_stage(model, source, positive, negative, 0, split)


def bind_tail(boundary, model, positive, negative):
    from .hyperflow_curve import CurveBoundary
    if type(boundary) is not CurveBoundary:
        raise ValueError("Curve TAIL binding needs its exact curve HEAD boundary")
    value = boundary.verify()
    return bind_stage(model, boundary.scaffold, positive, negative, value["request"]["plan"]["absolute_interval"][1], 8)


def validate_stage(model, sigmas, source, context):
    from .hyperflow_curve import _source
    owner = capture_owner(model)
    video, audio = _source(source)
    expected = runtime.build_plan(model, context.start, context.end).video_segment
    if (context != owner.context or tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape
            or type(sigmas) is not torch.Tensor or sigmas.dtype != torch.float32
            or sigmas.shape != expected.shape or not torch.equal(sigmas.cpu(), expected)):
        raise ValueError("Curve effects changed actual stage geometry, owner or AV clock")
    return video, audio, owner


def project_owner(model):
    if model.get_attachment(KEY) is None:
        return model, None
    owner = capture_owner(model)
    clone = model.clone()
    clone.remove_attachments(KEY)
    return clone, owner.descriptor()


@contextmanager
def execution(model, plan, source, positive, negative, cfg):
    from . import eav
    if model.get_attachment(KEY) is None:
        # Effects without their independent binding must not acquire a receipt.
        if model.get_attachment(eav.KEY) is not None:
            raise ValueError("Bind the curve phase before Stage EAV")
        yield None
        return
    owner = capture_owner(model)
    validate_stage(model, plan.video_segment, source, owner.context)
    telemetry = model.get_attachment(eav.KEY)
    if telemetry is not None and (type(telemetry) is not eav.StageEAVRuntime or telemetry.context != owner.context):
        raise ValueError("Curve EAV belongs to another actual phase")
    if cfg != 1. and (telemetry is not None or owner.relay_binding_json is not None):
        raise ValueError("Curve stage effects require native CFG1 batch-one routing")
    if owner.relay_binding_json is not None:
        contract = {"binding": json.loads(owner.relay_binding_json)}
        contract["binding_hash"] = contract["binding"]["binding_hash"]
        strip_paired_conditioning(positive, contract, required=True)
        strip_paired_conditioning(negative, contract, required=False)
    locks = [owner.lock] + ([_relay_lock(owner.relay_wrapper)] if owner.relay_wrapper is not None else [])
    acquired, report = [], {}
    try:
        for lock in locks:
            if not lock.acquire(blocking=False):
                raise RuntimeError("Curve effects are busy; use independent phase branches")
            acquired.append(lock)
        before = dict(owner.relay_counts or {})
        config = None
        if telemetry is not None:
            config = canonical(telemetry.snapshot()["config"])
            telemetry.closed = True
            telemetry.prepare(None, None, None)
        yield report
        capture_owner(model)
        report.update(schema=SCHEMA, phase=owner.context.stage, assigned_nfe=plan.nfe,
            stage_context=owner.context.to_dict(), composition_verified=True, quality_accepted=False)
        if telemetry is not None:
            current = telemetry.snapshot()
            if canonical(current["config"]) != config:
                raise ValueError("Curve EAV configuration changed during execution")
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
            warn_patch_stack("Curve effect real execution coverage is incomplete")
    except BaseException as error:
        if telemetry is not None and acquired:
            telemetry.telemetry.abort(error)
        raise
    finally:
        for lock in reversed(acquired):
            lock.release()


def audit(result, phase):
    from .hyperflow_curve import CurveBoundary, CurveResult
    if phase == "head" and type(result) is CurveBoundary:
        report = result.verify()["execution"].get("effects")
    elif phase == "tail" and type(result) is CurveResult:
        report = result.verify()["sampling"]["execution"].get("effects")
    else:
        raise ValueError("Select the matching completed curve phase result")
    if report is None:
        return {"schema": SCHEMA, "phase": phase, "status": "no_external_curve_stage_binding", "quality_accepted": False}
    if report.get("schema") != SCHEMA or report.get("phase") != phase:
        raise ValueError("Curve sampled effect report has another phase")
    return deepcopy(report)
