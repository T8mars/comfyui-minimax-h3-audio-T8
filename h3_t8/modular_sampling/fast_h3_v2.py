"""Public single-stage adapter for the existing exact FastH3 V2 4+4 recipe."""
from __future__ import annotations

import json

from .. import fast_h3_v2_advanced as legacy
from ..sampling import nested_av_parts
from .contracts import StageContext

PROFILES = ("trained_vsa_exp", "dense_compat_exp")
STAGES = ("low_0_4", "high_4_8")
RECIPE = "fasth3_v2.dual4plus4.v1"


def build_stage(model, av_latent, stage="low_0_4", profile="trained_vsa_exp", min_tokens=12288):
    if stage not in STAGES:
        raise ValueError("FastH3 V2 stage must be low_0_4 or high_4_8")
    if profile not in PROFILES:
        raise ValueError("This split uses the exact DMD recipe, not the distinct official template sampler")
    start, end = (0, 4) if stage == "low_0_4" else (4, 8)
    prepared, sampler, trajectory, original_report = legacy.build_fast_h3_v2_setup(
        model, av_latent, profile, min_tokens
    )
    # This is the same absolute window already consumed by the legacy loop.
    # Do not reset HIGH to step zero, force terminal zero on LOW, or freeze audio.
    sampler.extra_options.update(stage_start=start, stage_end=end)
    sigmas = trajectory[start:end + 1].clone()
    video, audio = nested_av_parts(av_latent)
    context = StageContext(
        recipe=RECIPE, stage=stage, profile=profile, start=start, end=end,
        trajectory_sigmas=tuple(float(v) for v in trajectory.tolist()),
        video_shift=legacy.VIDEO_SHIFT, audio_shift=legacy.AUDIO_SHIFT,
        video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
        input_semantics="native_av_template" if start == 0 else "learned_upscaled_reconciled_prediction_x0",
        output_semantics="unfinished_av_x_sigma" if end < 8 else "terminal_av",
        denoised_semantics="prediction_x0_for_learned_handoff" if end < 8 else "terminal_prediction_x0",
    )
    report = {**json.loads(original_report), "schema": "t8.modular-sampling.fasth3-v2.v1",
              "nfe": context.steps, "planned_nfe": context.steps, "stage_context": context.to_dict(),
              "descriptor_sha256": context.descriptor_sha256,
              "sampled": False, "cache_reuse_authorized": False,
              "wiring": "MODEL -> BasicGuider, SAMPLER/SIGMAS -> SamplerCustomAdvanced. "
                        "LOW denoised_output -> existing learned upscaler -> high-resolution reconcile -> HIGH. "
                        "Use independent MODEL/LoRA/conditioning/noise branches. Decode HIGH output.",
              "boundary": "Descriptor records expected handoff, not proof of tensor provenance or completed sampling. "
                          "Existing V2 runtime audit measures dispatch separately. No hidden second stage."}
    return prepared, sampler, sigmas, context, json.dumps(report, ensure_ascii=False, indent=2)
