from __future__ import annotations

import json
import math
from typing import Mapping

import torch
import torch.nn.functional as functional
import torchaudio

import comfy.nested_tensor

from .core import (
    AUDIO_LATENT_FPS,
    FPS,
    align_frame_count_down,
    encode_audio_once,
    validate_audio,
)


SCHEMA = "t8.minimax_h3.lanpaint_av_bridge.v1"
UPSTREAM = "https://github.com/scraed/LanPaint"
UPSTREAM_REVISION = "32cf848e93971da380d868936e007f5611218bee"


def parse_audio_intervals(value: str, duration: float) -> list[tuple[float, float]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = []
        for line in text.replace("，", ",").splitlines():
            line = line.strip()
            if not line:
                continue
            separator = "," if "," in line else "-"
            parts = [part.strip() for part in line.split(separator, 1)]
            if len(parts) != 2:
                raise ValueError(
                    "audio_intervals must be JSON or one start-end interval per line"
                )
            raw.append({"start": float(parts[0]), "end": float(parts[1])})
    if isinstance(raw, Mapping) and isinstance(raw.get("audio_intervals"), list):
        # Connect Prepare.report_json directly to Composite.audio_intervals so
        # both stages share one normalized interval plan.
        raw = raw["audio_intervals"]
    if not isinstance(raw, list):
        raise ValueError("audio_intervals JSON must be a list")
    result = []
    for item in raw:
        if isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError("Each audio interval must contain start and end seconds")
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"Invalid audio interval {item!r}")
        clipped_start = max(0.0, start)
        clipped_end = min(float(duration), end)
        if clipped_end > clipped_start:
            result.append((clipped_start, clipped_end))
    result.sort()
    merged: list[tuple[float, float]] = []
    for start, end in result:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def audio_interval_mask(
    intervals: list[tuple[float, float]],
    length: int,
    rate: float,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    mask = torch.zeros(int(length), device=device, dtype=dtype)
    for start, end in intervals:
        first = max(0, int(math.floor(start * rate)))
        last = min(int(length), int(math.ceil(end * rate)))
        if last > first:
            mask[first:last] = 1.0
    return mask


def _video_mask_to_latent(mask, video: torch.Tensor, frame_count: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(video)
    if not isinstance(mask, torch.Tensor):
        raise ValueError("video_mask must be a connected MASK tensor")
    value = mask.float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"video_mask must be [F,H,W] or [H,W], got {tuple(value.shape)}")
    if value.shape[0] == 1:
        value = value.expand(frame_count, -1, -1)
    elif value.shape[0] < frame_count:
        raise ValueError(
            f"video_mask has {value.shape[0]} frames but source requires {frame_count}"
        )
    else:
        value = value[:frame_count]
    value = value.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
    target = (video.shape[2], video.shape[3], video.shape[4])
    source = tuple(value.shape[2:])
    if all(source[index] >= target[index] for index in range(3)):
        # Every source frame/pixel contributes to one output bin. This avoids
        # dropping a short painted event during H3 temporal compression.
        value = functional.adaptive_max_pool3d(value, target)
    else:
        value = functional.interpolate(value, size=target, mode="nearest")
    return value.to(device=video.device, dtype=video.dtype).expand_as(video)


def prepare_lanpaint_av_latent(
    frames: torch.Tensor,
    source_audio,
    video_vae,
    audio_vae,
    audio_intervals: str,
    frame_policy: str,
    require_lanpaint_sampler: bool,
    video_mask=None,
):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 5:
        raise ValueError("frames must be a non-empty ComfyUI IMAGE batch")
    frame_count = int(frames.shape[0])
    aligned = align_frame_count_down(frame_count)
    if aligned < 5:
        raise ValueError("Source contains fewer than 5 H3-compatible frames")
    if aligned != frame_count:
        if frame_policy == "strict":
            raise ValueError(
                f"Source has {frame_count} frames; H3 requires 17n+5. "
                f"Use trim_down to process the first {aligned} frames."
            )
        if frame_policy != "trim_down":
            raise ValueError(f"Unknown frame_policy {frame_policy!r}")
        frames = frames[:aligned]
        frame_count = aligned
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if width % 32 or height % 32:
        raise ValueError("LanPaint H3 source width and height must be divisible by 32")

    if require_lanpaint_sampler:
        try:
            import nodes

            available = "LanPaint_KSamplerAdvanced" in nodes.NODE_CLASS_MAPPINGS
        except Exception:
            available = False
        if not available:
            raise RuntimeError(
                "LanPaint KSampler Advanced is not installed. Install the GPL-3.0 "
                f"upstream node pack from {UPSTREAM}, then restart ComfyUI."
            )

    video = video_vae.encode(frames[:, :, :, :3])
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            "The connected video VAE did not return MiniMax H3 [B,24,T,H,W] latent data"
        )
    audio = encode_audio_once(audio_vae, source_audio)
    target_audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    if audio.shape[-1] > target_audio_t:
        audio = audio[..., :target_audio_t]
    elif audio.shape[-1] < target_audio_t:
        audio = functional.pad(audio, (0, target_audio_t - audio.shape[-1]))
    audio = audio.to(device=video.device, dtype=video.dtype)

    duration = frame_count / FPS
    intervals = parse_audio_intervals(audio_intervals, duration)
    video_noise_mask = _video_mask_to_latent(video_mask, video, frame_count)
    audio_1d = audio_interval_mask(
        intervals,
        audio.shape[-1],
        AUDIO_LATENT_FPS,
        device=audio.device,
        dtype=audio.dtype,
    )
    audio_noise_mask = audio_1d[None, None, None, :].expand_as(audio)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (video_noise_mask, audio_noise_mask)
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": "lanpaint_av_latent_ready",
        "upstream": UPSTREAM,
        "upstream_revision_reviewed": UPSTREAM_REVISION,
        "source_frames": frame_count,
        "source_resolution": [width, height],
        "duration_seconds": duration,
        "video_latent_shape": list(video.shape),
        "audio_latent_shape": list(audio.shape),
        "video_regenerate_fraction": float(video_noise_mask.float().mean().item()),
        "audio_regenerate_fraction": float(audio_noise_mask.float().mean().item()),
        "audio_intervals": [list(item) for item in intervals],
        "mask_semantics": "1=regenerate, 0=preserve",
        "claims": {
            "lanpaint_sampler_bundled": False,
            "external_sampler_required": True,
            "old_workflow_schema_changed": False,
        },
    }
    return (
        latent,
        frames,
        source_audio,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _frame_mask(mask, count: int, height: int, width: int, device) -> torch.Tensor:
    if mask is None:
        return torch.zeros((count, height, width), device=device)
    value = mask.float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError("video_mask must be [F,H,W] or [H,W]")
    if value.shape[0] == 1:
        value = value.expand(count, -1, -1)
    if value.shape[0] < count:
        raise ValueError("video_mask has fewer frames than the composite")
    value = value[:count].to(device=device).unsqueeze(1)
    if tuple(value.shape[-2:]) != (height, width):
        value = functional.interpolate(value, (height, width), mode="nearest")
    return value[:, 0].clamp(0.0, 1.0)


def _feather_video_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    pixels = max(1, int(pixels))
    if pixels % 2 == 0:
        pixels += 1
    value = functional.max_pool2d(
        mask.unsqueeze(1), pixels, stride=1, padding=pixels // 2
    )
    if pixels > 1:
        value = functional.avg_pool2d(
            value, pixels, stride=1, padding=pixels // 2
        )
    return value[:, 0].clamp(0.0, 1.0)


def _fit_audio_pair(source_audio, repaired_audio):
    source, source_rate = validate_audio(source_audio, "source_audio")
    repaired, repaired_rate = validate_audio(repaired_audio, "repaired_audio")
    if repaired_rate != source_rate:
        repaired = torchaudio.functional.resample(repaired, repaired_rate, source_rate)
    channels = max(source.shape[1], repaired.shape[1])

    def match(value):
        if value.shape[1] == channels:
            return value
        if value.shape[1] == 1:
            return value.expand(-1, channels, -1)
        return value.mean(dim=1, keepdim=True).expand(-1, channels, -1)

    source = match(source)
    repaired = match(repaired)
    length = min(source.shape[-1], repaired.shape[-1])
    return source[..., :length], repaired[..., :length], source_rate


def composite_lanpaint_av(
    source_frames: torch.Tensor,
    repaired_frames: torch.Tensor,
    source_audio,
    repaired_audio,
    audio_intervals: str,
    video_blend_pixels: int,
    audio_crossfade_seconds: float,
    video_mask=None,
):
    if source_frames.ndim != 4 or repaired_frames.ndim != 4:
        raise ValueError("source_frames and repaired_frames must be IMAGE batches")
    count = min(int(source_frames.shape[0]), int(repaired_frames.shape[0]))
    if count < 1:
        raise ValueError("Cannot composite empty frame batches")
    source = source_frames[:count, :, :, :3]
    repaired = repaired_frames[:count, :, :, :3].to(source)
    height, width = int(source.shape[1]), int(source.shape[2])
    if tuple(repaired.shape[1:3]) != (height, width):
        repaired = functional.interpolate(
            repaired.movedim(-1, 1),
            (height, width),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)
    mask = _frame_mask(video_mask, count, height, width, source.device)
    feathered = _feather_video_mask(mask, int(video_blend_pixels))
    frames = source * (1.0 - feathered[..., None]) + repaired * feathered[..., None]

    source_wave, repaired_wave, sample_rate = _fit_audio_pair(source_audio, repaired_audio)
    duration = source_wave.shape[-1] / sample_rate
    intervals = parse_audio_intervals(audio_intervals, duration)
    audio_mask = audio_interval_mask(
        intervals,
        source_wave.shape[-1],
        sample_rate,
        device=source_wave.device,
        dtype=source_wave.dtype,
    )
    crossfade_samples = max(0, round(float(audio_crossfade_seconds) * sample_rate))
    if crossfade_samples > 1 and audio_mask.numel() > 1:
        kernel = min(crossfade_samples, int(audio_mask.numel()))
        left = kernel // 2
        right = kernel - 1 - left
        padded = functional.pad(audio_mask[None, None], (left, right), mode="replicate")
        audio_mask = functional.avg_pool1d(padded, kernel, stride=1)[0, 0]
    audio = source_wave * (1.0 - audio_mask[None, None]) + repaired_wave * audio_mask[None, None]
    report = {
        "schema": SCHEMA,
        "status": "lanpaint_av_composited",
        "frame_count": count,
        "resolution": [width, height],
        "video_blend_pixels": int(video_blend_pixels),
        "video_replaced_fraction": float(feathered.mean().item()),
        "audio_crossfade_seconds": float(audio_crossfade_seconds),
        "audio_replaced_fraction": float(audio_mask.mean().item()),
        "audio_intervals": [list(item) for item in intervals],
        "source_audio_preserved_outside_intervals": True,
    }
    return (
        frames.clamp(0.0, 1.0),
        {"waveform": audio, "sample_rate": sample_rate},
        json.dumps(report, ensure_ascii=False, indent=2),
    )
