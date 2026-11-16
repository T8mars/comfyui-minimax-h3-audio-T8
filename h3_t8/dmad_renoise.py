"""Explicit DMAD flow re-noising, using Core's existing LCM consumer.

No external implementation is copied: CONST + LCM already implements the
clean-prediction/fresh-noise rule. FLOW_AV carries audio on its native clock.
The previously accepted DMAD8/simple workflow remains a separate recipe.
"""

import json

from .sampling import setup_dual_clock_sampling


def prepare_dmad_renoise(model, av_latent, steps=8):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("DMAD steps must be a positive integer")
    selected, sampler, sigmas = setup_dual_clock_sampling(
        model, av_latent, steps, 12.0, 2.0, sampler_name="lcm", scheduler="native_flow"
    )
    sampling = selected.get_model_object("model_sampling")
    report = {
        "schema": "t8.dmad_renoise.prepare.v1",
        "status": "prepared_sampling_unobserved",
        "steps": steps,
        "sampler": "native_Core_lcm_flow_renoise",
        "video_shift": 12.0,
        "audio_shift": 2.0,
        "video_sigmas": sigmas.tolist(),
        "audio_scale": float(sampling.audio_scale),
        "rule": "clean prediction; (1 - sigma_next) * clean + sigma_next * fresh noise; no final re-noise",
        "randomness": "Core packed AV RNG, seeded by the connected NOISE; not bitwise WanGP's separate video/audio generator",
        "mask": "Native Core sampler inpaint/conditioning masks remain active",
        "adapter": "Load the selected distilled adapter upstream; no model or LoRA is selected, removed or replaced here",
        "recommended": "Matching DMAD 4/8-step adapter at strength 1, BasicGuider CFG 1; other user choices execute without quality claims",
        "patch_policy": "Existing weights, hooks and wrappers retained; unknown combinations remain unverified",
        "noise_scale": float(getattr(sampling, "noise_scale", 1.0)),
        "trained_GPU_quality_verified": False,
        "reference": "https://github.com/deepbeepmeep/WanGP/blob/4331e1a727a9c3a0b35466c2a7d2bfee72b912ae/models/minimax_h3/dmad.py",
    }
    return (
        selected,
        sampler,
        sigmas,
        json.dumps(report, ensure_ascii=False, sort_keys=True),
    )
