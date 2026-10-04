"""Expose PDD's original absolute 0:4 / 4:8 windows, never run either pass.

The existing PDD loader remains responsible for adapter validation/loading.
Rebuild only the native sampler geometry, keep its actual nine-point table,
and retain all head banks, LoRAs and injections on a new MODEL branch.
"""
from dataclasses import dataclass
import json

import torch
import comfy.samplers

from .. import pdd_advanced as pdd, sampling
from ..long_video_dual_identity import _v2_sampling_identity, content_identity
from ..patch_stack_policy import UnverifiedModelStack
from . import native_dual
from .contracts import StageContext, clear_native_stage_descriptors

RECIPE = "pdd.learned_absolute4plus4.v1"
KEY = "t8_modular_pdd_stage_v1"
STAGES = ("pdd_low_0_4", "pdd_high_4_8")
NATIVE = "comfyui_native_pdd_final_layer_plus_backbone_lora"
DYNAMIC = "comfyui_dynamic_model_only_bypass_plus_final_forward_injection"
RELATIVE = "comfyui_native_kijai_pdd_relative_heads_plus_backbone"


@dataclass(frozen=True)
class PDDStageOwner(native_dual.NativeDualOwner):
    pass


def _head_contract(model):
    receipt = model.get_attachment(pdd.PDD_ATTACHMENT_KEY)
    if not isinstance(receipt, dict) or receipt.get("schema") != "t8_minimax_h3_pdd_8step_setup_v2":
        raise ValueError("Connect MODEL from the existing PDD 8-Step Setup")
    mode = receipt.get("lora", {}).get("application_mode")
    result = {"loader_receipt": content_identity(receipt), "mode": mode,
              "metadata": content_identity(model.get_attachment("t8_minimax_h3_pdd_lora_metadata"))}
    if mode == NATIVE:
        final = model.get_model_object("diffusion_model.final_layer")
        banks = {}
        for stream in ("video", "audio"):
            head = getattr(final, stream + "_out")
            for field in ("weight", "bias"):
                key = f"diffusion_model.final_layer.{stream}_out.{field}"
                found = []
                for patch in model.patches.get(key, ()):
                    if not isinstance(patch, tuple) or len(patch) != 5:
                        continue
                    strength, payload, base_strength, offset, function = patch
                    if not (isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "diff"
                            and isinstance(payload[1], tuple) and len(payload[1]) == 2):
                        continue
                    tensor, options = payload[1]
                    if (strength == 1. and base_strength == 0. and offset is None and function is None
                            and isinstance(tensor, torch.Tensor) and options == {"pad_weight": True}):
                        expected = ((pdd.PDD_NUM_STEPS * head.out_features, head.in_features)
                                    if field == "weight" else (pdd.PDD_NUM_STEPS * head.out_features,))
                        if tuple(tensor.shape) != expected or not bool(torch.isfinite(tensor).all()):
                            raise ValueError("PDD native head bank has invalid shape or values")
                        found.append(content_identity(tensor))
                if len(found) != 1:
                    raise ValueError("PDD native MODEL is missing its unique 32-head replacement: " + key)
                banks[key] = found[0]
        result["heads"] = banks
    elif mode == RELATIVE:
        final = model.get_model_object("diffusion_model.final_layer")
        descriptor = receipt["lora"]
        strength = descriptor.get("strength")
        if (type(strength) not in (float, int) or not 0.0 <= strength <= 1.0
                or descriptor.get("native_head_patch_targets") != 4):
            raise ValueError("Kijai PDD relative head strength/receipt is invalid")
        banks = {}
        for stream in ("video", "audio"):
            head = getattr(final, stream + "_out")
            for field in ("weight", "bias"):
                key = f"diffusion_model.final_layer.{stream}_out.{field}"
                expected = ((pdd.PDD_NUM_STEPS * head.out_features, head.in_features)
                            if field == "weight" else (pdd.PDD_NUM_STEPS * head.out_features,))
                found = []
                for patch in model.patches.get(key, ()):
                    if not isinstance(patch, tuple) or len(patch) != 5:
                        continue
                    amount, payload, base_strength, offset, function = patch
                    if not (amount == strength and base_strength == 1.0 and offset is None and function is None
                            and isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "diff"
                            and isinstance(payload[1], tuple) and len(payload[1]) == 2):
                        continue
                    tensor, options = payload[1]
                    if not isinstance(tensor, torch.Tensor) or options != {"pad_weight": True}:
                        continue
                    if tuple(tensor.shape) != expected or not bool(torch.isfinite(tensor).all()):
                        raise ValueError("Kijai PDD relative head has invalid shape/values")
                    identity = content_identity(tensor)
                    if identity == descriptor.get("native_head_differences", {}).get(key):
                        found.append(identity)
                if len(found) != 1:
                    raise ValueError("Kijai PDD MODEL lacks its unique content-bound relative head: " + key)
                banks[key] = found[0]
        result["heads"] = banks
    elif mode == DYNAMIC:
        final = model.model_options.get("transformer_options", {}).get("minimax_h3_pdd_final")
        if type(final) is not pdd.PDDHeadFinalLayer or not model.get_injections(pdd.PDD_INJECTION_KEY):
            raise ValueError("PDD legacy MODEL is missing its dynamic heads or injection")
        if pdd.pdd_forward_wrapper not in model.get_wrappers("diffusion_model", pdd.PDD_WRAPPER_KEY):
            raise ValueError("PDD legacy MODEL is missing its absolute head selector")
        result["heads"] = content_identity({name: getattr(final, name) for name in
            ("video_weight", "video_bias", "audio_weight", "audio_bias", "strength", "variant")})
        from . import pdd_dynamic
        try:
            result["dynamic"] = pdd_dynamic.describe(model)
        except UnverifiedModelStack:
            # Unknown executable additions remain usable for a fresh run.
            # project_identity will explain why persisted reuse is unavailable.
            result["dynamic"] = None
    else:
        raise ValueError("PDD setup receipt has an unknown loader mode")
    return result


def capture_owner(model):
    owner = native_dual.capture_owner(model, key=KEY, owner_type=PDDStageOwner, recipe=RECIPE)
    if owner.context.stage not in STAGES:
        raise ValueError("Unknown PDD stage owner")
    if _head_contract(model) != json.loads(owner.context.profile)["pdd"]:
        raise ValueError("PDD heads or loader contract changed after stage setup")
    return owner


def build_stage(model, av_latent, full_sigmas, stage=STAGES[0]):
    if stage not in STAGES:
        raise ValueError("Unknown PDD absolute stage")
    if (not isinstance(full_sigmas, torch.Tensor) or full_sigmas.ndim != 1
            or not full_sigmas.is_floating_point()):
        raise ValueError("PDD needs the original floating nine-point SIGMAS")
    pdd.validate_pdd_sigmas(full_sigmas)
    contract = _head_contract(model)
    # Same public setup used by the old HIGH graph; no weight/adapter reload,
    # no first pass and no replacement of the connected full sigma table.
    start, end = (0, 4) if stage == STAGES[0] else (4, 8)
    # Old LOW consumes the loader's Euler/simple; old HIGH explicitly uses the
    # packed-geometry dual_clock_euler/native_flow setup. Its regenerated table
    # is unused: BOTH windows still use the original connected PDD nine points.
    prepared, sampler, _ = sampling.setup_dual_clock_sampling(model, av_latent, pdd.PDD_NFE,
        pdd.PDD_SHIFT_VIDEO, pdd.PDD_SHIFT_AUDIO,
        pdd.PDD_SAMPLER if start == 0 else "dual_clock_euler",
        pdd.PDD_SCHEDULER if start == 0 else "native_flow")
    video, audio = sampling.nested_av_parts(av_latent)
    profile = native_dual._canonical({"sigma_dtype": str(full_sigmas.dtype), "pdd": contract})
    context = StageContext(RECIPE, stage, profile, start, end, tuple(full_sigmas.tolist()),
        pdd.PDD_SHIFT_VIDEO, pdd.PDD_SHIFT_AUDIO, tuple(video.shape), tuple(audio.shape),
        "native_av_template" if start == 0 else "learned_reconciled_joint_av_fresh_noise",
        "unfinished_av_x_sigma" if start == 0 else "terminal_av",
        "prediction_x0_for_learned_handoff" if start == 0 else "terminal_prediction_x0")
    selected = prepared.object_patches["model_sampling"]
    clear_native_stage_descriptors(prepared)
    prepared.set_attachments(KEY, PDDStageOwner(context, selected,
        native_dual._canonical(_v2_sampling_identity(selected)), sampling.model_uses_raw_audio_velocity(model)))
    report = {"schema": "t8.modular-sampling.pdd-stage.v1", "stage_context": context.to_dict(),
        "absolute_head_groups": list(range(start, end)), "sampled": False,
        "legacy_dynamic_persistent_identity_adapted": contract.get("dynamic") is not None,
        "boundary": "Original PDD heads and actual nine-point schedule retained. Independent MODEL/LoRA, "
                    "condition and NOISE per pass. LOW denoised -> existing learned3D -> original joint-audio "
                    "reconcile -> HIGH fresh noise. Native banks and authenticated legacy injection have stage identity. "
                    "Unknown external patch stacks run without false portable qualification."}
    return prepared, sampler, full_sigmas[start:end + 1], context, json.dumps(report, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    if context != owner.context:
        raise ValueError("PDD MODEL and stage context are not paired")
    start, end = (0, 4) if context.stage == STAGES[0] else (4, 8)
    if (context.start, context.end) != (start, end):
        raise ValueError("PDD absolute head interval changed")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or str(sigmas.dtype) != json.loads(context.profile)["sigma_dtype"]
            or tuple(sigmas.tolist()) != context.trajectory_sigmas[start:end + 1]):
        raise ValueError("PDD SIGMAS differ from their original absolute window or dtype")
    video, audio = sampling.nested_av_parts(source)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("PDD AV geometry differs from stage setup")
    return video, audio, owner


def sampler_is_known(model, sampler, context):
    owner = capture_owner(model)
    if context.stage == STAGES[1]:
        return native_dual.sampler_is_known(model, sampler, context, owner=owner)
    expected = comfy.samplers.sampler_object(pdd.PDD_SAMPLER)
    return (type(sampler) is comfy.samplers.KSAMPLER and sampler.sampler_function is expected.sampler_function
            and sampler.extra_options == expected.extra_options and sampler.inpaint_options == expected.inpaint_options)


def project_identity(model):
    owner = capture_owner(model)
    view, contract = native_dual.project_identity(model, owner=owner, key=KEY)
    if json.loads(owner.context.profile)["pdd"]["mode"] == DYNAMIC:
        from . import pdd_dynamic
        view, contract["dynamic"] = pdd_dynamic.project(view)
    # These plain descriptions are already content-bound above. Actual patches
    # remain present and the ordinary identity hashes all their tensors/order.
    view.remove_attachments(pdd.PDD_ATTACHMENT_KEY)
    view.remove_attachments("t8_minimax_h3_pdd_lora_metadata")
    return view, contract
