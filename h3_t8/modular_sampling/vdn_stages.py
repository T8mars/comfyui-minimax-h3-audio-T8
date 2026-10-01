"""One original VDN full trajectory or its own fresh-noise tail; no sampling."""
from dataclasses import dataclass
import json

import torch
import comfy.samplers

from .. import vdn_h3_advanced as vdn, vdn_two_pass, sampling
from ..long_video_dual_identity import _v2_sampling_identity, content_identity
from ..patch_stack_policy import UnverifiedModelStack
from . import native_dual, vdn_identity
from .contracts import StageContext, clear_native_stage_descriptors

RECIPE = "vdn.complete_then_own_grid_tail.v1"
KEY = "t8_modular_vdn_stage_v1"
STAGES = ("vdn_complete", "vdn_refine")


@dataclass(frozen=True)
class VDNStageOwner(native_dual.NativeDualOwner):
    pass


def _contract(model):
    receipt = model.get_attachment(vdn.ATTACHMENT_KEY)
    if (type(receipt) is not dict or receipt.get("schema") != vdn.SCHEMA
            or receipt.get("status") != "configured" or receipt.get("stage") not in vdn.STAGES
            or receipt.get("steps") != vdn.STAGES[receipt["stage"]]["steps"]):
        raise ValueError("Connect the original configured OpenVDN Composer MODEL")
    try:
        identity = vdn_identity.inspect(model)
    except UnverifiedModelStack:
        identity = None  # Preserve unknown user patches for actual fresh execution.
    return {"receipt": content_identity(receipt), "identity": identity}


def capture_owner(model):
    owner = native_dual.capture_owner(model, key=KEY, owner_type=VDNStageOwner, recipe=RECIPE)
    if owner.context.stage not in STAGES or _contract(model) != json.loads(owner.context.profile)["vdn"]:
        raise ValueError("VDN stage recipe/branch/Composer changed after binding")
    return owner


def build_stage(model, av_latent, stage=STAGES[0], refine_steps=4, first_pass_latent=None):
    if stage not in STAGES:
        raise ValueError("Unknown VDN stage")
    contract = _contract(model)
    count = int(model.get_attachment(vdn.ATTACHMENT_KEY)["steps"])
    if stage == STAGES[0]:
        prepared, sampler, sigmas, old_report = vdn.setup_vdn_execution(model, av_latent)
        full, start = sigmas, 0
    else:
        if first_pass_latent is None:
            raise ValueError("VDN refinement needs the completed first-pass latent")
        prepared, sampler, sigmas, old_report = vdn_two_pass.setup_vdn_refine(
            model, av_latent, first_pass_latent, refine_steps)
        # The old execution helper selects this precise native_flow grid. Keep
        # its returned actual tail, check provenance rather than replace it.
        full = sampling.native_flow_sigmas(count, 12.)
        start = count - len(sigmas) + 1
        if not torch.equal(sigmas, full[start:]):
            raise ValueError("VDN refine no longer matches its own complete stage grid")
    video, audio = sampling.nested_av_parts(av_latent)
    context = StageContext(RECIPE, stage, native_dual._canonical({"vdn": contract,
        "sigma_dtype": str(sigmas.dtype), "sampler": json.loads(old_report)["sampler"]}),
        start, count, tuple(full.tolist()), 12., 3., tuple(video.shape), tuple(audio.shape),
        "native_av_template" if start == 0 else "completed_av_learned_reconciled_fresh_noise",
        "terminal_av", "terminal_prediction_x0")
    selected = prepared.object_patches["model_sampling"]
    clear_native_stage_descriptors(prepared)
    prepared.set_attachments(KEY, VDNStageOwner(context, selected,
        native_dual._canonical(_v2_sampling_identity(selected)), sampling.model_uses_raw_audio_velocity(model)))
    return prepared, sampler, sigmas, context, json.dumps({"schema": "t8.modular-sampling.vdn-stage.v1",
        "stage_context": context.to_dict(), "legacy_plan": json.loads(old_report), "sampled": False,
        "boundary": "Original complete DMD8/B50 or own-stage fresh-noise tail, not LBH/partial4+4. "
                    "Keep learned upscale/reconcile and independent models/LoRA/noise outside. "
                    "VDN window/linear effects require their own actual producer adapters."}, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    if context != owner.context:
        raise ValueError("VDN MODEL and stage context differ")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or str(sigmas.dtype) != json.loads(context.profile)["sigma_dtype"]
            or tuple(sigmas.tolist()) != context.trajectory_sigmas[context.start:context.end + 1]):
        raise ValueError("VDN stage SIGMAS values/dtype differ from its own bound grid")
    video, audio = sampling.nested_av_parts(source)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("VDN source AV geometry changed after setup")
    return video, audio, owner


def sampler_is_known(model, sampler, context):
    owner = capture_owner(model)
    if json.loads(context.profile)["sampler"] == "dual_clock_euler":
        return native_dual.sampler_is_known(model, sampler, context, owner=owner)
    expected = comfy.samplers.sampler_object("euler")
    return (type(sampler) is comfy.samplers.KSAMPLER and sampler.sampler_function is expected.sampler_function
            and sampler.extra_options == expected.extra_options and sampler.inpaint_options == expected.inpaint_options)


def project_identity(model):
    view, contract = native_dual.project_identity(model, owner=capture_owner(model), key=KEY)
    view, contract["vdn"] = vdn_identity.project(view)
    return view, contract
