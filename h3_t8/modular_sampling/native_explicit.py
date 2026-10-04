"""Bind an already explicit native AV stage, without constructing a new recipe.

Existing native/base-flow/LBH plans and samplers remain authoritative. This
owner describes the connected stage; it is not proof of which upstream plan
produced it, nor a replacement for PDD/VDN/V2-specific numerical adapters.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json

import torch
import comfy.samplers
import comfy.k_diffusion.sampling as k_sampling

from .. import sampling
from ..learned_latent_upscale_advanced import _extract_h3_shifts
from ..long_video_dual_identity import _v2_sampling_identity
from . import native_dual, detail_effects
from .contracts import StageContext, clear_native_stage_descriptors

RECIPE = "native.explicit_single_stage.v1"
KEY = "t8_modular_native_explicit_v1"
STAGES = ("native_low", "native_high")
ER_SDE_SOURCE_SHA256 = "0f8db8e10b7d1b5311adaffe1ebfbdb41ced68c88fa31041eaa9e3a5c4d83543"
ER_SDE_FIRST_SIGMA_SHA256 = "38a667cea46183a84a6183d40cc0e67bb84ffe36cf09db197f4a5dee277d0ff6"
LCM_SOURCE_SHA256 = "7c3bb88e758dbc13825f26db141026e008016d18ae0615981e08130fe15df6cb"


@dataclass(frozen=True)
class NativeExplicitOwner(native_dual.NativeDualOwner):
    pass


def capture_owner(model):
    owner = native_dual.capture_owner(model, key=KEY, owner_type=NativeExplicitOwner, recipe=RECIPE)
    if owner.context.stage not in STAGES:
        raise ValueError("Native explicit stage has an unknown role")
    if detail_effects.describe(model) != json.loads(owner.context.profile)["detail_effects"]:
        raise ValueError("Native explicit stage Bias/STG changed after binding")
    return owner


def _sampler_kind(model, sampler, context, owner):
    if native_dual.sampler_is_known(model, sampler, context, owner=owner):
        return "dual_clock_euler"
    expected = comfy.samplers.sampler_object("euler")
    if (type(sampler) is comfy.samplers.KSAMPLER and sampler.sampler_function is expected.sampler_function
            and sampler.extra_options == expected.extra_options and sampler.inpaint_options == expected.inpaint_options):
        return "euler"
    if _exact_source_er_sde(sampler):
        return "er_sde_exact_source"
    if _exact_source_lcm(sampler):
        return "lcm_exact_source"
    return "unadapted"


def _lcm_core_source_hash():
    try:
        return hashlib.sha256(inspect.getsource(k_sampling.sample_lcm).encode()).hexdigest()
    except (AttributeError, OSError, TypeError):
        return None


def _exact_source_lcm(sampler):
    """Recognize only Core's unmodified one-forward/fresh-noise LCM.

    This authenticates completion of the connected stage, not an ELM checkpoint,
    author RNG replay or content quality. Non-default options remain unverified.
    """
    expected = comfy.samplers.sampler_object("lcm")
    return (type(sampler) is comfy.samplers.KSAMPLER
            and sampler.sampler_function is expected.sampler_function
            and sampler.sampler_function is k_sampling.sample_lcm
            and not sampler.extra_options and not sampler.inpaint_options
            and _lcm_core_source_hash() == LCM_SOURCE_SHA256)


def _exact_source_er_sde(sampler):
    expected = comfy.samplers.sampler_object("er_sde")
    return (type(sampler) is comfy.samplers.KSAMPLER
            and sampler.sampler_function is expected.sampler_function
            and sampler.sampler_function is k_sampling.sample_er_sde
            and not sampler.extra_options and not sampler.inpaint_options
            and _er_sde_core_source_hashes() == (ER_SDE_SOURCE_SHA256, ER_SDE_FIRST_SIGMA_SHA256))


def _er_sde_eav_forward_plan(model, sampler, sigmas, effects, block_count):
    """Recognize current Core's one-denoiser-call/interval ER-SDE.

    The source guard covers the solver's actual model call and first-sigma
    adjustment. Unknown Core revisions and sampler options stay unverified.
    """
    if not _exact_source_er_sde(sampler):
        return None
    solver_sha, offset_sha = _er_sde_core_source_hashes()
    selected = model.object_patches["model_sampling"]
    adjusted = k_sampling.offset_first_sigma_for_snr(sigmas, selected)
    plan = detail_effects.forward_plan(effects, adjusted.tolist(), block_count,
                                       model_time_dtype=adjusted.dtype)
    plan["er_sde_solver_sha256"] = solver_sha
    plan["er_sde_first_sigma_sha256"] = offset_sha
    plan["boundary"] = ("Current Core ER-SDE source has one model call per interval; derivative stages do not "
                        "call the model again. This plans EAV forwards only; portable StageResult identity "
                        "separately requires the exact sampler, actual callback count, model and source checks.")
    return plan


def _er_sde_core_source_hashes():
    try:
        return (hashlib.sha256(inspect.getsource(k_sampling.sample_er_sde).encode()).hexdigest(),
                hashlib.sha256(inspect.getsource(k_sampling.offset_first_sigma_for_snr).encode()).hexdigest())
    except (OSError, TypeError):
        return None, None


def bind_stage(model, sampler, sigmas, av_latent, stage=STAGES[0]):
    if stage not in STAGES:
        raise ValueError("Unknown native explicit stage")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point()
            or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all())
            or not bool(((sigmas >= 0) & (sigmas <= 1)).all())
            or not bool((sigmas[:-1] > sigmas[1:]).all())):
        raise ValueError("Native explicit SIGMAS must descend strictly within [0,1]")
    if "model_sampling" not in model.object_patches:
        raise ValueError("Connect the configured MODEL from an explicit native H3 sampler setup")
    shifts = _extract_h3_shifts(model)
    video, audio = sampling.nested_av_parts(av_latent)
    effects = detail_effects.describe(model)
    prepared = model.clone()
    selected = prepared.object_patches["model_sampling"]
    sampling_json = native_dual._canonical(_v2_sampling_identity(selected))
    terminal = float(sigmas[-1]) == 0.
    block_count = len(prepared.get_model_object("diffusion_model").blocks)
    er_sde_plan = _er_sde_eav_forward_plan(prepared, sampler, sigmas, effects, block_count)
    lcm_plan = None
    if _exact_source_lcm(sampler):
        lcm_plan = detail_effects.forward_plan(effects, sigmas.tolist(), block_count)
        lcm_plan["lcm_solver_sha256"] = _lcm_core_source_hash()
        lcm_plan["boundary"] = ("Exact Core LCM has one model call per interval; non-terminal fresh-noise "
                                "transitions make no additional model calls. Not author RNG or quality parity.")
    coverage_adapter = ("core_er_sde_exact_source" if er_sde_plan is not None else
                        "core_lcm_exact_source" if lcm_plan is not None else "none")
    initial_profile = {"sigma_dtype": str(sigmas.dtype), "detail_effects": effects,
        "forward_plan": er_sde_plan or lcm_plan or detail_effects.forward_plan(effects, sigmas.tolist(), block_count),
        "origin": "explicit_connected_native_stage_not_upstream_recipe_provenance"}
    def context(profile):
        return StageContext(RECIPE, stage, native_dual._canonical(profile), 0, len(sigmas) - 1,
            tuple(float(x) for x in sigmas.tolist()), *shifts, tuple(video.shape), tuple(audio.shape),
            "native_av_template" if stage == STAGES[0] else "explicit_reconciled_av_with_fresh_noise",
            "terminal_av" if terminal else "unfinished_av_x_sigma",
            "prediction_x0_for_learned_handoff" if stage == STAGES[0] else "native_prediction_x0")
    initial = context(initial_profile)
    owner = NativeExplicitOwner(initial, selected, sampling_json, sampling.model_uses_raw_audio_velocity(model))
    kind = _sampler_kind(model, sampler, initial, owner)
    initial_profile["sampler_kind"] = kind
    initial_profile["eav_forward_coverage_adapter"] = coverage_adapter
    if kind == "unadapted" and er_sde_plan is None:
        initial_profile["forward_plan"]["known"] = False
    bound = context(initial_profile)
    # Only our inert descriptors on the new clone. No live effect is removed.
    clear_native_stage_descriptors(prepared)
    prepared.set_attachments(KEY, NativeExplicitOwner(bound, selected, sampling_json, owner.audio_velocity_is_raw))
    report = {"schema": "t8.modular-sampling.native-explicit.v1", "stage_context": bound.to_dict(),
        "sampled": False, "sampler_replaced": False, "sigmas_replaced": False,
        "portable_completion_sampler_adapted": kind != "unadapted",
        "eav_forward_coverage_adapter": coverage_adapter,
        "boundary": "Binds the actual connected native stage only, not its upstream plan provenance. "
                    "Keep original base-flow/LBH/full-first schedules, separate learned upscaler/reconcile and fresh HIGH noise. "
                    "PDD/VDN/V2-specific effects and completion are not certified by this native binding."}
    return prepared, sampler, sigmas, bound, json.dumps(report, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    result = native_dual.validate_stage(model, sigmas, source, context, owner=owner)
    if str(sigmas.dtype) != json.loads(context.profile)["sigma_dtype"] or tuple(sigmas.tolist()) != context.trajectory_sigmas:
        raise ValueError("Native explicit SIGMAS values or dtype changed after binding")
    return result


def sampler_is_known(model, sampler, context):
    kind = _sampler_kind(model, sampler, context, capture_owner(model))
    return kind != "unadapted" and kind == json.loads(context.profile)["sampler_kind"]


def project_identity(model):
    view, contract = native_dual.project_identity(model, owner=capture_owner(model), key=KEY)
    view, contract["detail_effects"] = detail_effects.project(view)
    return view, contract


def forward_plan(context):
    plan = json.loads(context.profile)["forward_plan"]
    if "lcm_solver_sha256" in plan and _lcm_core_source_hash() != plan["lcm_solver_sha256"]:
        return {**plan, "known": False, "boundary": "Bound LCM Core source changed; EAV coverage is unverified."}
    if "er_sde_solver_sha256" in plan and _er_sde_core_source_hashes() != (
            plan["er_sde_solver_sha256"], plan["er_sde_first_sigma_sha256"]):
        return {**plan, "known": False, "boundary": "Bound ER-SDE Core source changed; EAV coverage is unverified."}
    return plan
