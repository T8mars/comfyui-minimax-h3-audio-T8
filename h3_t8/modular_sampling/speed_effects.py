"""Bind the generic stage EAV adapter to one exact SPEED Euler segment."""
from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from .. import sampling
from ..core import nested_av_parts
from ..long_video_dual_identity import _v2_sampling_identity
from .contracts import StageContext
from . import native_dual


RECIPE = "speed.dct-nstage.v1"
KEY = "t8_modular_speed_stage_v1"


def bind(model, plan, stage_index, sigmas, av_latent, shift_audio, scope):
    segment = plan["segments"][stage_index]
    video, audio = nested_av_parts(av_latent)
    values = tuple(float(value) for value in segment["sigmas"])
    if sigmas.ndim != 1 or not torch.allclose(sigmas.cpu(), torch.tensor(values), atol=1e-7, rtol=0):
        raise ValueError("SPEED effect stage SIGMAS differ from the plan")
    context = StageContext(
        recipe=RECIPE, stage=f"stage_{stage_index}", profile=scope,
        start=0, end=len(values) - 1, trajectory_sigmas=values,
        video_shift=float(plan["shift_video"]), audio_shift=float(shift_audio),
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="native_av_target" if stage_index == 0 else "dct_expanded_flow_target",
        output_semantics="terminal_av" if stage_index == len(plan["stages"]) - 1 else "public_flow_x_sigma",
        denoised_semantics="not_captured_by_speed_guider",
    )
    if model.get_attachment(KEY) is not None:
        raise ValueError("SPEED stage MODEL already carries a stage owner")
    prepared = model.clone()
    selected = prepared.get_model_object("model_sampling")
    prepared.set_attachments(KEY, native_dual.NativeDualOwner(
        context, selected,
        json.dumps(_v2_sampling_identity(selected), sort_keys=True, separators=(",", ":")),
        sampling.model_uses_raw_audio_velocity(model),
    ))
    return prepared, context


def capture_owner(model):
    return native_dual.capture_owner(
        model, key=KEY, owner_type=native_dual.NativeDualOwner, recipe=RECIPE,
    )


def project_identity(model):
    return native_dual.project_identity(model, owner=capture_owner(model), key=KEY)


def validate_stage(model, sigmas, av_latent, context):
    if type(context) is not StageContext or context.recipe != RECIPE:
        raise ValueError("Stage EAV requires a real SPEED stage context")
    if capture_owner(model).context != context:
        raise ValueError("Stage EAV SPEED MODEL and stage context differ")
    expected = torch.tensor(context.trajectory_sigmas, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.shape != expected.shape or not torch.allclose(sigmas.cpu(), expected, atol=1e-7, rtol=0):
        raise ValueError("Stage EAV SPEED SIGMAS differ from this stage")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("Stage EAV SPEED AV layout differs from this stage")
    # Generic EAV only uses runtime/profile to decide whether the old V2
    # sparse producer adapter applies. SPEED Euler has no V2 owner.
    return video, audio, SimpleNamespace(runtime=None, profile=context.profile)


def apply_relay(model, spec, source, relay_plan, *, execution_mode="apply_exp", query_chunk_rows=256):
    """Re-encode and bind one external Relay plan at this SPEED canvas."""
    from ..prompt_relay_advanced import build_prompt_relay_conditioning
    from ..speed_advanced import _source_conditioning_kwargs
    from .speed_stages import SpeedStageSpec

    if type(spec) is not SpeedStageSpec or source is not spec.source_ref:
        raise ValueError("SPEED Relay source must be this prepared stage's Source output")
    # The external plan owns the stage text.  It may be edited without also
    # changing Source.prompt; the rebuilt target below must still match the
    # already prepared media/AV contract exactly.
    shape = spec.plan["stages"][spec.index]
    expected_sigmas = torch.tensor(spec.plan["segments"][spec.index]["sigmas"], dtype=torch.float32)
    validate_stage(model, expected_sigmas, spec.stage.latent, spec.stage_context)
    kwargs = _source_conditioning_kwargs(source, shape["width"], shape["height"])
    kwargs.pop("prompt")
    kwargs.pop("length")
    selected, positive, latent, mux_audio, prompt, media_map, report = build_prompt_relay_conditioning(
        model=model, prompt_relay_plan=relay_plan, execution_mode=execution_mode,
        query_chunk_rows=query_chunk_rows, **kwargs,
    )
    for name in ("samples", "noise_mask"):
        expected = spec.stage.latent.get(name)
        actual = latent.get(name)
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"SPEED Relay rebuilt a different {name} contract")
        elif (not getattr(expected, "is_nested", False) or not getattr(actual, "is_nested", False)
              or not all(torch.equal(a, b) for a, b in zip(expected.unbind(), actual.unbind()))):
            raise ValueError(f"SPEED Relay rebuilt a different {name} contract")
    validate_stage(selected, expected_sigmas, latent, spec.stage_context)
    return selected, positive, latent, mux_audio, prompt, media_map, report
