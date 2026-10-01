"""Verify a completed V2 LOW result before its separate learned handoff."""

import json

from .. import fast_h3_v2_advanced as legacy
from .contracts import StageContext
from .fast_h3_v2 import PROFILES, RECIPE
from .results import StageResult


def completed_low_x0(stage_result):
    if type(stage_result) is not StageResult:
        raise ValueError("FastH3 V2 LOW handoff needs a sampler-produced StageResult")
    receipt = stage_result.verify()
    context = StageContext.from_dict(receipt["request"]["stage_context"])
    expected = tuple(float(value) for value in legacy.dmd_sigmas().tolist())
    if (context.recipe != RECIPE or context.stage != "low_0_4" or context.profile not in PROFILES
            or (context.start, context.end) != (0, 4) or context.trajectory_sigmas != expected
            or (context.video_shift, context.audio_shift) != (legacy.VIDEO_SHIFT, legacy.AUDIO_SHIFT)
            or context.input_semantics != "native_av_template"
            or context.output_semantics != "unfinished_av_x_sigma"
            or context.denoised_semantics != "prediction_x0_for_learned_handoff"):
        raise ValueError("Stage result is not the exact FastH3 V2 LOW 0:4 handoff")
    if receipt["verified_recipe_completion"] is not True:
        raise ValueError("FastH3 V2 LOW 0:4 did not complete the verified sampler recipe")
    report = {"schema": "t8.modular-sampling.fasth3-v2-low-x0.v1",
              "status": "verified_low_x0", "request_sha256": receipt["request_sha256"],
              "receipt_sha256": receipt["receipt_sha256"], "profile": context.profile,
              "portable_identity": receipt["portable_identity"],
              "boundary": "Completed LOW prediction_x0 only. The external learned upscaler and "
                          "HIGH reconcile/sampler remain separate; this does not prove their execution."}
    return stage_result.denoised_output, json.dumps(report, ensure_ascii=False, sort_keys=True)
