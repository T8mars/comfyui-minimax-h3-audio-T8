"""Public RF base/restart stage contracts; never combines both descents."""
from dataclasses import dataclass
import json

import torch
import comfy.samplers

from .. import sampling
from ..long_video_dual_identity import _v2_sampling_identity
from . import native_dual, rf_restart, detail_effects
from .contracts import StageContext, clear_native_stage_descriptors

RECIPE = "rf_restart.native_joint_av.v1"
KEY = "t8_modular_rf_stage_v1"
ANCHOR = "t8_modular_rf_original_template_v1"
SIGMA_DTYPE = "t8_modular_rf_sigma_dtype_v1"
STAGES = ("rf_base", "rf_restart")


@dataclass(frozen=True)
class RFStageOwner(native_dual.NativeDualOwner):
    pass


def capture_owner(model):
    owner = native_dual.capture_owner(model, key=KEY, owner_type=RFStageOwner, recipe=RECIPE)
    if owner.context.stage not in STAGES:
        raise ValueError("RF stage has an unknown phase")
    if detail_effects.describe(model) != json.loads(owner.context.profile)["detail_effects"]:
        raise ValueError("RF stage Bias/STG configuration changed after Setup")
    return owner


def _anchor(source):
    # Keep only this stage's original source, not a recursively growing history.
    return {key: value for key, value in source.items() if key not in (ANCHOR, SIGMA_DTYPE, rf_restart.RAW_ENDPOINT)}


def handoff(completed_av, original_template=None, base_sigmas=None):
    if original_template is None:
        original_template = completed_av.get(ANCHOR)
        if original_template is None:
            raise ValueError("Connect the original RF base template or load a saved RF base result with its anchor")
    if base_sigmas is not None:
        if not isinstance(base_sigmas, torch.Tensor) or not base_sigmas.is_floating_point():
            raise ValueError("RF base SIGMAS must be a floating tensor")
        completed_av = {**completed_av, SIGMA_DTYPE: str(base_sigmas.dtype)}
    return rf_restart.prepare_handoff(completed_av, _anchor(original_template))


def _bind(prepared, sampler, sigmas, source, stage, profile, shift_video, shift_audio):
    video, audio = sampling.nested_av_parts(source)
    effects = detail_effects.describe(prepared)
    raw_endpoint = rf_restart.retained_endpoint(source) if stage == STAGES[1] else None
    model_time_dtype = raw_endpoint.dtype if raw_endpoint is not None else torch.float32
    profile = {**profile, "sigma_dtype": str(sigmas.dtype), "detail_effects": effects,
        "forward_plan": detail_effects.forward_plan(effects, sigmas.tolist(),
            len(prepared.get_model_object("diffusion_model").blocks), model_time_dtype=model_time_dtype)}
    noop = sigmas.numel() == 0
    context = StageContext(RECIPE, stage, native_dual._canonical(profile), 0, max(0, len(sigmas) - 1),
        tuple(float(x) for x in sigmas.tolist()), float(shift_video), float(shift_audio),
        tuple(video.shape), tuple(audio.shape),
        "identity_noop" if noop else "native_av_template" if stage == STAGES[0] else "completed_endpoint_with_original_rf_anchor",
        "identity_noop" if noop else "terminal_av", "identity_noop" if noop else "terminal_prediction_x0")
    # These are our own inert descriptors on this new branch, not effect hooks.
    clear_native_stage_descriptors(prepared)
    selected = prepared.object_patches["model_sampling"]
    prepared.set_attachments(KEY, RFStageOwner(context, selected,
        native_dual._canonical(_v2_sampling_identity(selected)), sampling.model_uses_raw_audio_velocity(prepared)))
    return prepared, sampler, sigmas, context


def build_base_stage(model, av_latent, sigmas, shift_video=12., shift_audio=3.):
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("RF base requires a floating SIGMAS vector")
    # The original strict schedule contract accepts complete or partial starts.
    rf_restart.legacy._validate_h3_sigmas(sigmas, "custom_strict")
    prepared, sampler, _ = sampling.setup_dual_clock_sampling(model, av_latent, len(sigmas) - 1, shift_video, shift_audio)
    result = _bind(prepared, sampler, sigmas, av_latent, STAGES[0],
                   {"anchor_identity": json.loads(rf_restart._identity(_anchor(av_latent)))}, shift_video, shift_audio)
    return (*result, json.dumps({"stage": STAGES[0], "sampled": False,
        "wiring": "Explicit full/partial/tail SIGMAS. One descent only. Stage Sampler retains the original anchor for restart."}))


def build_restart_stage(model, rf_handoff, shift_video=12., shift_audio=3., restart_video_sigma=.15,
                        restart_steps=3, restart_seed=1234):
    dtype = rf_handoff.completed_av.get(SIGMA_DTYPE, "torch.float32")
    dtypes = {str(value): value for value in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    if dtype not in dtypes:
        raise ValueError("RF saved SIGMAS dtype is unknown")
    prepared, sampler, sigmas, raw = rf_restart.build_restart_stage(model, rf_handoff,
        shift_video=shift_video, shift_audio=shift_audio, restart_video_sigma=restart_video_sigma,
        restart_steps=restart_steps, restart_seed=restart_seed, sigma_dtype=dtypes[dtype])
    profile = {"anchor_identity": json.loads(rf_handoff.template_identity),
        "endpoint_identity": json.loads(rf_handoff.completed_identity), "restart_report": json.loads(sampler.report_json)}
    result = _bind(prepared, sampler, sigmas, rf_handoff.completed_av, STAGES[1], profile, shift_video, shift_audio)
    details = json.loads(raw)
    details.update(stage_context=result[3].to_dict(), legacy_constructor_base_not_executed=True,
        retained_model_endpoint=rf_restart.RAW_ENDPOINT in rf_handoff.completed_av,
        boundary="Only the RF restart descent; original anchor and SIGMAS dtype retained. Native Bias/STG, "
                 "stage EAV/Relay and frozen result adapters have tiny CPU evidence, not full GPU/media qualification.")
    return (*result, json.dumps(details, ensure_ascii=False, indent=2))


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    if context != owner.context:
        raise ValueError("RF stage MODEL/context differ")
    profile = json.loads(context.profile)
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point()
            or str(sigmas.dtype) != profile["sigma_dtype"]
            or tuple(float(x) for x in sigmas.tolist()) != context.trajectory_sigmas):
        raise ValueError("RF stage SIGMAS differ from the bound trajectory")
    video, audio = sampling.nested_av_parts(source)
    if (tuple(video.shape), tuple(audio.shape)) != (context.video_shape, context.audio_shape):
        raise ValueError("RF stage AV layout differs from context")
    expected = profile.get("endpoint_identity") if context.stage == STAGES[1] else profile["anchor_identity"]
    actual = source if context.stage == STAGES[1] else _anchor(source)
    if json.loads(rf_restart._identity(actual)) != expected:
        raise ValueError("RF stage endpoint/original template differs from its context")
    return video, audio, owner


def sampler_is_known(model, sampler, context):
    owner = capture_owner(model)
    if context.stage == STAGES[0]:
        return native_dual.sampler_is_known(model, sampler, context, owner=owner)
    if type(sampler) is not rf_restart.RFRestartSampler or set(vars(sampler)) != {
            "sampler_function", "extra_options", "inpaint_options", "handoff", "report_json"}:
        return False
    sampler.handoff.verify()
    profile = json.loads(context.profile)
    if (json.loads(sampler.handoff.template_identity) != profile["anchor_identity"]
            or json.loads(sampler.handoff.completed_identity) != profile["endpoint_identity"]
            or json.loads(sampler.report_json) != profile["restart_report"]):
        raise ValueError("RF sampler handoff or restart parameters differ from context")
    native = comfy.samplers.KSAMPLER(sampler.sampler_function, sampler.extra_options, sampler.inpaint_options)
    return native_dual.sampler_is_known(model, native, context, owner=owner)


def project_identity(model):
    view, contract = native_dual.project_identity(model, owner=capture_owner(model), key=KEY)
    view, contract["detail_effects"] = detail_effects.project(view)
    return view, contract


def retain_anchor(output, denoised, source, sampler, context, raw_endpoint=None):
    if context.stage == STAGES[1] and type(sampler) is not rf_restart.RFRestartSampler:
        return output, denoised  # Unknown selection may run; no invented RF state.
    anchor = _anchor(source) if context.stage == STAGES[0] else sampler.handoff.original_template
    dtype = json.loads(context.profile)["sigma_dtype"]
    output, denoised = ({key: value for key, value in item.items() if key != rf_restart.RAW_ENDPOINT}
                        for item in (output, denoised))
    if raw_endpoint is not None:
        output = rf_restart.attach_endpoint(output, raw_endpoint)
    elif context.steps == 0 and rf_restart.RAW_ENDPOINT in source:
        output[rf_restart.RAW_ENDPOINT] = source[rf_restart.RAW_ENDPOINT]
    return ({**output, ANCHOR: anchor, SIGMA_DTYPE: dtype}, {**denoised, ANCHOR: anchor, SIGMA_DTYPE: dtype})


def forward_plan(context):
    return json.loads(context.profile)["forward_plan"]
