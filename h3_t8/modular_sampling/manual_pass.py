"""The two original manual-second-pass calls as independently editable stages.

No loop, learned upscale, second sampler or hidden noise runs in this setup.
The handoff is first OUTPUT; fresh noise is a separate visible NOISE input.
"""
from dataclasses import dataclass
import json
import math

import comfy.samplers

from .. import sampling
from ..long_video_dual_identity import _v2_sampling_identity
from ..long_video_sampling_plan_advanced import build_long_video_sampling_plan, resolve_long_video_sample_schedules
from .contracts import StageContext, clear_native_stage_descriptors
from . import native_dual

RECIPE = "native.manual_second_pass.v1"
KEY = "t8_modular_manual_pass_v1"
STAGES = ("manual_first", "manual_second")


@dataclass(frozen=True)
class ManualPassOwner(native_dual.NativeDualOwner):
    """Distinct receipt, sharing only the inert native sampling identity shape."""


def capture_owner(model):
    return native_dual.capture_owner(model, key=KEY, owner_type=ManualPassOwner, recipe=RECIPE)


def build_stage(model, av_latent, stage=STAGES[0], first_steps=20, manual_sigmas="0.5,0.412,0.35,0",
                shift_video=12., shift_audio=3., sampler_name="dual_clock_euler", scheduler="native_flow"):
    if stage not in STAGES:
        raise ValueError("Unknown manual-pass stage")
    if type(first_steps) is not int or not 1 <= first_steps <= 10000:
        raise ValueError("Manual first_steps must be a positive integer no greater than 10000")
    for value in (shift_video, shift_audio):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Manual-pass AV shifts must be positive finite numbers")
    first = stage == "manual_first"
    plan = None
    if not first:
        plan, _ = build_long_video_sampling_plan("manual_second_pass", 0, "video_sigma_linear", manual_sigmas)
    count = first_steps if first else len(plan.manual_sigmas) - 1
    prepared, sampler, base = sampling.setup_dual_clock_sampling(model, av_latent, count,
        shift_video, shift_audio, sampler_name, scheduler)
    if first:
        sigmas, _, report = resolve_long_video_sample_schedules(base, None,
            shift_video=shift_video, shift_audio=shift_audio)
    else:
        _, sigmas, report = resolve_long_video_sample_schedules(base, plan,
            shift_video=shift_video, shift_audio=shift_audio)
    video, audio = sampling.nested_av_parts(av_latent)
    context = StageContext(recipe=RECIPE, stage=stage,
        profile=native_dual._canonical({"sampler": sampler_name, "scheduler": scheduler}),
        start=0, end=len(sigmas) - 1, trajectory_sigmas=tuple(float(value) for value in sigmas.tolist()),
        video_shift=float(shift_video), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="native_av_template" if first else "first_pass_output_with_fresh_noise",
        output_semantics="terminal_av_for_manual_restart" if first else "terminal_av",
        denoised_semantics="prediction_x0_not_manual_handoff" if first else "terminal_prediction_x0")
    selected = prepared.object_patches["model_sampling"]
    clear_native_stage_descriptors(prepared)
    prepared.set_attachments(KEY, ManualPassOwner(context, selected,
        native_dual._canonical(_v2_sampling_identity(selected)), sampling.model_uses_raw_audio_velocity(model)))
    details = {"schema": "t8.modular-sampling.manual-pass.v1", "stage_context": context.to_dict(),
        "legacy_schedule": report, "sampled": False, "planned_intervals": context.steps,
        "portable_completion_sampler_adapted": sampler_name in ("dual_clock_euler", "euler"),
        "wiring": "First OUTPUT -> second latent_image. Independent MODEL/conditioning/NOISE per stage. "
                  "To reproduce old runners, use the same seed and optional FreeNoise segment_index for both.",
        "boundary": "Not learned-upscale or tail_subdivide. Other Core samplers remain selectable, but their "
                    "multi-evaluation completion/effect audit needs a separate qualification adapter."}
    return prepared, sampler, sigmas, context, json.dumps(details, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, av_latent, context):
    return native_dual.validate_stage(model, sigmas, av_latent, context, owner=capture_owner(model))


def project_identity(model):
    return native_dual.project_identity(model, owner=capture_owner(model), key=KEY)


def sampler_is_known(model, sampler, context):
    owner = capture_owner(model)
    name = json.loads(context.profile)["sampler"]
    if name == "dual_clock_euler":
        return native_dual.sampler_is_known(model, sampler, context, owner=owner)
    if name != "euler" or type(sampler) is not comfy.samplers.KSAMPLER:
        return False
    expected = comfy.samplers.sampler_object(name)
    return (sampler.sampler_function is expected.sampler_function
            and sampler.extra_options == expected.extra_options and sampler.inpaint_options == expected.inpaint_options)


def allows_intermediate_evaluations(context):
    return context.recipe == RECIPE and json.loads(context.profile)["sampler"] not in ("dual_clock_euler", "euler")
