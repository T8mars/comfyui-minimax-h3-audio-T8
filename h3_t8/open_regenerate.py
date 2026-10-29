"""Open-H3 refinement preparation; native Comfy sampler/VAEs own execution.

Schedule/method: NVlabs/Sana, sol-engine/models/Open-H3-Regenerate-2K.
The Kijai rank-384 adapter is only the refiner, not the merged release recipe.
"""
import json
import math

import torch

from .core import encode_audio_once, nested_av_parts
from .sampling import setup_dual_clock_sampling
from .source_av import frame_count_from_video_latent_t, prepare_source_av_latent


SCHEDULES = {
    "published_3step": (0.909375, 6 / 7, 2 / 3, 0.0),
    "published_2step": (0.7, 42 / 85, 0.0),
}
SCHEDULE_OPTIONS = (*SCHEDULES, "custom_shift6")


def refine_sigmas(schedule="published_3step", custom_steps=3, start_sigma=0.909375):
    if schedule in SCHEDULES:
        return torch.tensor(SCHEDULES[schedule], dtype=torch.float32)
    if schedule != "custom_shift6":
        raise ValueError(f"Unknown refinement schedule: {schedule}")
    if type(custom_steps) is not int or custom_steps < 1:
        raise ValueError("Custom refinement steps must be a positive integer")
    if not math.isfinite(start_sigma) or not 0 < start_sigma <= 1:
        raise ValueError("Refinement start sigma must be finite and in (0, 1]")
    raw = start_sigma / (6 - 5 * start_sigma)
    points = [raw * (1 - index / custom_steps) for index in range(1, custom_steps)]
    return torch.tensor([start_sigma, *(6 * x / (1 + 5 * x) for x in points), 0.0])


def prepare_refine(model, video_latent, audio_vae, schedule="published_3step",
                   custom_steps=3, start_sigma=0.909375):
    sigmas = refine_sigmas(schedule, custom_steps, start_sigma)
    samples = video_latent.get("samples")
    video = nested_av_parts(video_latent)[0] if getattr(samples, "is_nested", False) else samples
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("Connect an encoded H3 video LATENT [B,24,T,H,W] or joint AV LATENT")
    frames = frame_count_from_video_latent_t(video.shape[2])
    # Silence is encoded, never replaced by an all-zero audio latent. The
    # original soundtrack remains outside this refinement branch for muxing.
    silence = {"waveform": torch.zeros(1, 2, round(frames * 32000 / 24)), "sample_rate": 32000}
    with torch.no_grad():
        encoded = encode_audio_once(audio_vae, silence)
    if video.shape[0] != encoded.shape[0] and encoded.shape[0] == 1:
        encoded = encoded.expand(video.shape[0], -1, -1, -1)
    source = {key: value for key, value in video_latent.items() if key not in {"samples", "noise_mask"}}
    source["samples"] = video
    prepared, _, _, source_report = prepare_source_av_latent(
        source, {"samples": encoded}, "regenerate", 1.0, "lock", 0.0,
        "trim_to_video", "match_video",
    )
    # Sigmas are already in the video-noise domain. A shift of one prevents
    # applying the usual 12/3 clocks a second time; audio rows are locked.
    patched, sampler, _ = setup_dual_clock_sampling(
        model, prepared, len(sigmas) - 1, 1.0, 1.0, sampler_name="euler",
    )
    noise_scale = float(getattr(patched.get_model_object("model_sampling"), "noise_scale", 1.0))
    report = {
        "schema": "t8.open_h3_regenerate.prepare.v1",
        "status": "prepared_sampling_unobserved",
        "schedule": schedule, "video_sigmas": sigmas.tolist(), "steps": len(sigmas) - 1,
        "sampler": "euler", "sampling_shifts": {"video": 1.0, "audio": 1.0},
        "inherited_noise_scale": noise_scale,
        "guidance": "Connect BasicGuider (CFG 1) to the same second-stage MODEL and text conditioning",
        "audio_context": "VAE-encoded stereo silence; locked mask 0 and native audio timestep 1",
        "video_mask": "full refinement mask 1; original input mask is not reused",
        "input_mask_replaced": "noise_mask" in video_latent,
        "source_frames": frames, "source_video_shape": list(video.shape),
        "source_av": json.loads(source_report),
        "adapters": "No adapter is added, removed or inferred from a filename; use an independent selected MODEL branch",
        "recipe_boundary": "Kijai rank384 contains refiner only. Published merged recipe additionally uses Turbo v1.2 and Sharp including curve branches; this setup does not supply them or certify equivalent quality.",
        "soundtrack": "Mux original AUDIO separately; this branch neither regenerates nor replaces the source recording",
        "trained_GPU_quality_verified": False,
    }
    return patched, sampler, sigmas, prepared, json.dumps(report, ensure_ascii=False, sort_keys=True)
