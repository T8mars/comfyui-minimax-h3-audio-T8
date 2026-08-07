from __future__ import annotations

import json

import torch

from comfy.ldm.minimax.model import MiniMaxH3Model

from .core import (
    MAX_PIXELS,
    MAX_TRAINED_FRAMES,
    MIN_TRAINED_FRAMES,
    VRAM_CAUTION_PIXELS,
    align_frame_count,
    classify_h3_vae,
    sorted_autogrow_values,
    validate_audio,
)


def run_preflight(
    width: int,
    height: int,
    length: int,
    audio_mode: str,
    model=None,
    video_vae=None,
    audio_vae=None,
    drive_audio=None,
    ref_images=None,
    ref_videos=None,
    ref_audios=None,
):
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, object] = {}

    if width <= 0 or height <= 0 or width % 32 or height % 32:
        errors.append("width and height must be positive multiples of 32")
    pixels = width * height
    facts["pixels"] = pixels
    facts["max_pixel_area"] = MAX_PIXELS
    if pixels > MAX_PIXELS:
        errors.append(
            f"canvas has {pixels:,} pixels; configured H3 2.0MP cap is "
            f"{MAX_PIXELS:,} pixels (1920x1088)"
        )
    elif pixels > VRAM_CAUTION_PIXELS:
        warnings.append(
            f"high-resolution canvas has {pixels:,} pixels; this is supported up to "
            "1920x1088 but may require substantially more VRAM"
        )
    aligned = align_frame_count(length)
    facts["aligned_frames"] = aligned
    if aligned != length:
        warnings.append(f"length {length} will snap up to {aligned} on the 17n+5 grid")
    if not MIN_TRAINED_FRAMES <= aligned <= MAX_TRAINED_FRAMES:
        warnings.append(
            f"{aligned} frames is outside the approximately trained {MIN_TRAINED_FRAMES}-{MAX_TRAINED_FRAMES} range"
        )

    if model is not None:
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if not isinstance(diffusion_model, MiniMaxH3Model):
            errors.append("model is not a native ComfyUI MiniMax H3 diffusion model")
    if video_vae is not None:
        video_vae_kind = classify_h3_vae(video_vae)
        facts["video_vae_kind"] = video_vae_kind
        if video_vae_kind == "audio":
            errors.append("video_vae is an H3 audio VAE; connect the H3 video VAE")
    if audio_vae is not None:
        audio_vae_kind = classify_h3_vae(audio_vae)
        facts["audio_vae_kind"] = audio_vae_kind
        if audio_vae_kind == "video":
            errors.append("audio_vae is an H3 video VAE; connect the H3 audio VAE")
        facts["audio_vae_sample_rate"] = int(getattr(audio_vae, "audio_sample_rate", 32000))

    if audio_mode != "native" and drive_audio is None:
        errors.append(f"audio mode {audio_mode} requires drive_audio")
    if drive_audio is not None:
        try:
            waveform, sample_rate = validate_audio(drive_audio, "drive_audio")
            facts["drive_audio_sample_rate"] = sample_rate
            facts["drive_audio_duration_seconds"] = waveform.shape[-1] / sample_rate
            if not torch.isfinite(waveform).all():
                errors.append("drive_audio contains NaN or infinite samples")
            else:
                peak = float(waveform.abs().max())
                rms = float(torch.sqrt(torch.mean(waveform.float().square())))
                facts["drive_audio_peak"] = peak
                facts["drive_audio_rms"] = rms
                if peak > 1.0:
                    warnings.append(f"drive_audio peak {peak:.3f} exceeds 0 dBFS and may clip")
                if rms < 1e-4:
                    warnings.append("drive_audio is effectively silent")
        except ValueError as error:
            errors.append(str(error))

    pictures = sorted_autogrow_values(ref_images)
    videos = sorted_autogrow_values(ref_videos)
    audios = sorted_autogrow_values(ref_audios)
    facts.update({"reference_images": len(pictures), "reference_videos": len(videos), "reference_audios": len(audios)})
    if len(pictures) > 9 or len(videos) > 3 or len(audios) > 3:
        errors.append("reference counts exceed H3 limits (9 pictures / 3 videos / 3 audios)")
    for index, frames in enumerate(videos, 1):
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            errors.append(f"ref_video_{index} is not an IMAGE frame batch")
        elif not 48 <= frames.shape[0] <= 360:
            warnings.append(f"ref_video_{index} has {frames.shape[0]} frames; official guidance is 48-360 at 24fps")

    ready = not errors
    report = json.dumps(
        {"ready": ready, "errors": errors, "warnings": warnings, "facts": facts},
        ensure_ascii=False,
        indent=2,
    )
    return ready, len(warnings), report
