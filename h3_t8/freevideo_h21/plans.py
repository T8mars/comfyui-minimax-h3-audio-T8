"""New v4 plans are read from their exact source; legacy tables stay untouched."""
from __future__ import annotations

import copy
import hashlib

from .source import REVISION, canonical
from .cache_state import HISTORY_POLICY

RUNTIME_SCHEMA = "t8.freevideo.h21.runtime.v4"
STAGE_SCHEMA = "t8.freevideo.h21.stage.v4"


def plan(source, tier, width, height, frames, *, task="t2va", role="FULL"):
    tiers = source.module("tiers")
    if tier not in tiers.TIERS:
        raise ValueError("Choose a concrete H21 tier: light, medium, high or max")
    if role not in ("FULL", "LOW", "HIGH"):
        raise ValueError("H21 role must be FULL, LOW or HIGH")
    definition = copy.deepcopy(tiers.TIERS[tier])
    canvas = source.module("geometry").geometry(width, height, frames=frames)
    if canvas["frames"] != frames:
        raise ValueError("Conditioning must use the actual aligned H3 frame count, not a silently extended clock")
    # This API receives pre-encoded conditioning, exactly the author's
    # conditioned=True branch: no hidden render-size substitution.
    sampled, output_size = source.module("two_pass").sampling_canvas(canvas, conditioned=True)
    sampling = source.module("two_pass").plan(sampled, definition["two_pass"], task,
        base_steps=definition["steps"], refine_steps=tiers.REFINE_STEPS, scale=definition["upscale"])
    if role != "FULL" and not sampling["enabled"]:
        raise ValueError("This actual one-pass plan has no LOW or HIGH stage")
    expected = sampling["total_steps"] if role == "FULL" else sampling["base_steps"] if role == "LOW" else sampling["refine_steps"]
    selected = dict(schema="t8.freevideo.h21.plan.v4", source_revision=REVISION,
        tier=tier, role=role, task=task, requested_canvas=dict(width=width, height=height, frames=frames),
        actual_canvas={k: sampled[k] for k in ("width", "height", "frames")}, output_size=output_size,
        definition=definition, sampling=sampling, expected_callbacks=expected,
        step_cache=definition["step_cache"] if role != "HIGH" else None,
        requested_upscale=definition["upscale"] if definition["two_pass"] else None,
        actual_learned_upscale=sampling.get("scale", 2) if sampling.get("upscaler_sha256") else None,
        reference_audio_t=1., high_audio_policy="completed_LOW_preserved", approximation=definition["step_cache"] is not None)
    if selected["step_cache"] == "easycache-av":
        selected["cache_history_policy"] = HISTORY_POLICY
    selected["identity"] = hashlib.sha256(canonical(selected).encode()).hexdigest()
    return selected


def clock(source, selected, role):
    """Actual author scheduler/clock API, not a second rounded linspace."""
    if role == "HIGH":
        video, audio = source.module("refine_schedule").schedulers("cpu")
        start = 1
    else:
        scheduler = source.module("step_grid").scheduler
        video, audio = scheduler(12.), scheduler(3.)
        video.set_timesteps(selected["sampling"]["base_steps"], device="cpu")
        audio.set_timesteps(selected["sampling"]["base_steps"], device="cpu")
        start = 0
    times = source.module("adaln").modality_timesteps(video.timesteps[start:], audio.timesteps[start:], selected["task"])
    return dict(video_sigmas=video.sigmas[start:].tolist(), audio_sigmas=audio.sigmas[start:].tolist(),
                modulation_timesteps=[value.tolist() for value in times], reference_audio_t=1.,
                leading_initialization_not_NFE=role == "HIGH")
