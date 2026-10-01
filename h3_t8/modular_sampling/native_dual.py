"""Expose the legacy Dual math one stage at a time, without running its loop.

LOW4 is simple8[:5], LOW20 is full native_flow20, HIGH is its own published
LBH table. These are not V2 windows. Audio reconciliation delegates to the old
runner, including its partial-audio migration and locked template prefix.
"""
from copy import copy
from dataclasses import dataclass
import json
import math

import torch
import comfy.samplers

from .. import sampling
from ..long_video_dual_model_runner import DualModelSegmentRunner
from ..long_video_dual_identity import _v2_sampling_identity
from ..vdn_attention_compat import _factory_closure
from .contracts import StageContext, clear_native_stage_descriptors

RECIPE = "native_dual.lbh_or_stock20.v1"
KEY = "t8_modular_native_dual_v1"
STAGES = ("dual_low_4", "dual_low_20", "dual_high_3", "dual_high_4", "dual_high_5")
AUDIO_SOURCES = ("auto", "legacy_policy", "first_pass", "highres_template")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _runner(model=None, *, coarse_steps=4, refine_steps=4, shift_video=12., shift_audio=3.,
            second_audio_source="auto", second_audio_strength=0.):
    # Only _stage_sampling or _reconcile is used. No run(), cache, model load,
    # conditioning, context extraction, color correction or upscaler executes.
    return DualModelSegmentRunner(model, model, contract={}, low_width=32, low_height=32,
        upscaler_model="not_loaded_by_single_stage_adapter", coarse_steps=coarse_steps,
        refine_steps=refine_steps, first_shift_video=shift_video, first_shift_audio=shift_audio,
        second_shift_video=shift_video, second_shift_audio=shift_audio, color_match=False,
        second_audio_source=second_audio_source, second_audio_strength=second_audio_strength)


@dataclass(frozen=True)
class NativeDualOwner:
    context: StageContext
    sampling_object: object
    sampling_json: str
    audio_velocity_is_raw: bool

    @property
    def runtime(self):
        return None  # Not a V2 producer or a fake V2 dispatch receipt.

    @property
    def profile(self):
        return self.context.profile


def capture_owner(model, *, key=KEY, owner_type=NativeDualOwner, recipe=RECIPE):
    owner = model.get_attachment(key)
    if type(owner) is not owner_type or set(vars(owner)) != {
        "context", "sampling_object", "sampling_json", "audio_velocity_is_raw"
    }:
        raise ValueError("Native Dual stage MODEL is missing its exact owner")
    if type(owner.context) is not StageContext or owner.context.recipe != recipe:
        raise ValueError("Native Dual owner has another stage recipe")
    if model.object_patches.get("model_sampling") is not owner.sampling_object:
        raise ValueError("Native Dual selected sampling object was replaced")
    if _canonical(_v2_sampling_identity(owner.sampling_object)) != owner.sampling_json:
        raise ValueError("Native Dual sampling configuration changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"), options.get("minimax_h3_sigma_shift_audio")) != (
            owner.context.video_shift, owner.context.audio_shift):
        raise ValueError("Native Dual MODEL clock shifts differ from context")
    return owner


def build_stage(model, av_latent, stage=STAGES[0], shift_video=12., shift_audio=3.):
    if stage not in STAGES:
        raise ValueError("Unknown Native Dual stage")
    for value in (shift_video, shift_audio):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Native Dual clock shifts must be positive finite numbers")
    first = stage.startswith("dual_low_")
    steps = int(stage.rsplit("_", 1)[1])
    runner = _runner(model, coarse_steps=steps if first else 4, refine_steps=4 if first else steps,
                     shift_video=shift_video, shift_audio=shift_audio)
    prepared, sampler, sigmas, schedule = runner._stage_sampling(model, av_latent, first)
    video, audio = sampling.nested_av_parts(av_latent)
    profile = ("simple8_partial4" if steps == 4 else "native_flow20") if first else f"published_lbh{steps}"
    context = StageContext(recipe=RECIPE, stage=stage, profile=profile, start=0, end=steps,
        trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=float(shift_video), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="native_av_template" if first else "learned_upscaled_reconciled_prediction_x0",
        output_semantics="unfinished_av_x_sigma" if first and steps == 4 else "terminal_av",
        denoised_semantics="prediction_x0_for_learned_handoff" if first else "terminal_prediction_x0")
    selected = prepared.object_patches["model_sampling"]
    clear_native_stage_descriptors(prepared)
    prepared.set_attachments(KEY, NativeDualOwner(context, selected,
        _canonical(_v2_sampling_identity(selected)), sampling.model_uses_raw_audio_velocity(model)))
    report = {"schema": "t8.modular-sampling.native-dual.v1", "stage_context": context.to_dict(),
              "descriptor_sha256": context.descriptor_sha256, "legacy_schedule": json.loads(schedule),
              "planned_nfe": steps, "sampled": False, "cache_reuse_authorized": False,
              "wiring": "Independent MODEL/LoRA/condition/noise per stage. LOW denoised_output -> existing "
                        "learned3D -> Native Dual Handoff with rebuilt HIGH template -> HIGH. Decode HIGH output.",
              "boundary": "One legacy stage setup only; no loop or hidden second sampler. "
                          "Descriptor is not tensor provenance or completion evidence."}
    return prepared, sampler, sigmas, context, json.dumps(report, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, av_latent, context, *, owner=None):
    owner = capture_owner(model) if owner is None else owner
    if context != owner.context:
        raise ValueError("Native Dual MODEL is not paired with this stage context")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("Native Dual SIGMAS must be a floating one-dimensional tensor")
    expected = torch.tensor(context.trajectory_sigmas, dtype=sigmas.dtype, device=sigmas.device)
    if sigmas.shape != expected.shape or not torch.allclose(sigmas, expected, rtol=0, atol=1e-7):
        raise ValueError("Native Dual SIGMAS differ from their exact bound trajectory")
    video, audio = sampling.nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("Native Dual AV layout differs from context")
    return video, audio, owner


def sampler_is_known(model, sampler, context, *, owner=None):
    owner = capture_owner(model) if owner is None else owner
    if type(sampler) is not comfy.samplers.KSAMPLER or sampler.extra_options or sampler.inpaint_options:
        return False
    state = _factory_closure(sampler.sampler_function, sampling._build_dual_clock_sampler, "sampler_function")
    return state == {"video_values": math.prod(context.video_shape[1:]),
                     "packed_values": math.prod(context.video_shape[1:]) + math.prod(context.audio_shape[1:]),
                     "shift_video": context.video_shift, "shift_audio": context.audio_shift,
                     "audio_velocity_is_raw": owner.audio_velocity_is_raw}


def project_identity(model, *, owner=None, key=KEY):
    """Factor only the exact native stage owner on an inspection-only clone."""
    owner = capture_owner(model) if owner is None else owner
    view = model.clone()
    previous = getattr(model, "object_patches_backup", {}).get("model_sampling")
    if previous is not None:
        if model.model.model_sampling is not owner.sampling_object:
            # Core clones share the base network and object-patch backup. A
            # sibling LOW/HIGH may still occupy the live sampling slot before
            # Core installs this branch. Authenticate that inert native slot;
            # unknown executable sampling owners remain nonportable. Never
            # unload the sibling or write into the shared live network.
            _v2_sampling_identity(model.model.model_sampling)
        view.model = copy(model.model)
        view.model._modules = dict(model.model._modules)
        view.model._modules["model_sampling"] = previous
        view.object_patches_backup = dict(model.object_patches_backup)
        view.object_patches_backup.pop("model_sampling")
    view.object_patches.pop("model_sampling")
    view.remove_attachments(key)
    return view, {"stage_context": owner.context.to_dict(), "model_sampling": json.loads(owner.sampling_json),
                  "audio_velocity_is_raw": owner.audio_velocity_is_raw}


def reconcile(learned_latent, highres_template, positive, first_pass_steps=4,
              second_audio_source="auto", second_audio_strength=0.):
    if type(first_pass_steps) is not int or first_pass_steps not in (4, 20):
        raise ValueError("Native Dual handoff requires first_pass_steps 4 or 20")
    runner = _runner(coarse_steps=first_pass_steps, second_audio_source=second_audio_source,
                     second_audio_strength=second_audio_strength)
    prepared, bound, report = runner._reconcile(learned_latent, highres_template, positive)
    details = {"schema": "t8.modular-sampling.native-dual-handoff.v1", "legacy_report": json.loads(report),
               "audio_policy": runner.audio_policy, "diffusion_evaluations": 0,
               "boundary": "Original legacy handoff, not a learned upscaler or a sampler. "
                           "Connect LOW denoised_output through the existing learned3D first."}
    return prepared, bound, json.dumps(details, ensure_ascii=False, indent=2)
