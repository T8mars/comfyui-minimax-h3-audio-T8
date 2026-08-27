from __future__ import annotations

import json
from collections.abc import Mapping

import torch
import torchaudio

import comfy.nested_tensor
import comfy.utils

from .core import (
    AUDIO_LATENT_FPS,
    FPS,
    REFERENCE_PIXEL_AREA,
    MAX_TRAINED_FRAMES,
    MIN_TRAINED_FRAMES,
    align_frame_count,
    resize_image,
    validate_audio,
)


VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO = 2
SOURCE_AV_SCHEMA = "h3_t8_source_av_v1"
STREAM_MODES = ("lock", "remix", "regenerate")
AUDIO_FIT_POLICIES = (
    "strict",
    "trim_to_video",
    "pad_to_video_generate_tail",
    "fit_to_video_generate_tail",
)
DTYPE_DEVICE_POLICIES = ("match_video", "strict")
SOURCE_AUDIO_SAMPLE_RATE = 32000
SHORT_VIDEO_POLICIES = ("strict", "hold_last_frame")
SHORT_AUDIO_POLICIES = ("strict", "pad_silence")


def frame_count_from_video_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(
            "MiniMax H3 video latent T must satisfy T = 5n+2; "
            f"got T={latent_t}. Pre-trim the source to a 17n+5 frame window before VAE encode."
        )
    return ((latent_t - 2) // 5) * 17 + 5


def prepare_source_media_window(
    frames: torch.Tensor,
    source_fps: float,
    width: int,
    height: int,
    length: int,
    start_seconds: float,
    short_video_policy: str,
    short_audio_policy: str,
    source_audio: Mapping | None = None,
):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"frames must be IMAGE [N,H,W,C], got {shape}")
    if frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError("source video frames are empty or do not contain RGB channels")
    if float(source_fps) <= 0:
        raise ValueError("source_fps must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds must be nonnegative")
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("H3 source output width and height must be positive multiples of 32")
    if short_video_policy not in SHORT_VIDEO_POLICIES:
        raise ValueError(f"Unknown short video policy {short_video_policy!r}")
    if short_audio_policy not in SHORT_AUDIO_POLICIES:
        raise ValueError(f"Unknown short audio policy {short_audio_policy!r}")

    frame_count = align_frame_count(length)
    target_times = start_seconds + torch.arange(frame_count, dtype=torch.float64) / FPS
    source_indices = torch.round(target_times * float(source_fps)).to(torch.long)
    beyond = source_indices >= frames.shape[0]
    held_video_frames = int(beyond.sum().item())
    if held_video_frames and short_video_policy == "strict":
        required_end = start_seconds + (frame_count - 1) / FPS
        available_end = (frames.shape[0] - 1) / float(source_fps)
        raise ValueError(
            "Source video is too short for the requested H3 window: "
            f"needs frame time {required_end:.6f}s, available through {available_end:.6f}s"
        )
    source_indices = source_indices.clamp(0, frames.shape[0] - 1).to(frames.device)
    selected = frames.index_select(0, source_indices)
    selected = resize_image(selected, width, height, "disabled")

    duration_seconds = frame_count / FPS
    target_audio_samples = round(duration_seconds * SOURCE_AUDIO_SAMPLE_RATE)
    audio_source = "connected_audio"
    padded_audio_samples = 0
    if source_audio is None:
        waveform = torch.zeros((1, 2, target_audio_samples), dtype=torch.float32)
        audio_source = "generated_silence"
        audio_input_rate = None
        audio_input_channels = None
    else:
        waveform, audio_input_rate = validate_audio(source_audio, "source_audio")
        audio_input_channels = waveform.shape[1]
        if waveform.shape[1] == 1:
            waveform = waveform.expand(-1, 2, -1)
        elif waveform.shape[1] != 2:
            waveform = waveform.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        if audio_input_rate != SOURCE_AUDIO_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                audio_input_rate,
                SOURCE_AUDIO_SAMPLE_RATE,
            )
        start_sample = round(start_seconds * SOURCE_AUDIO_SAMPLE_RATE)
        available = waveform[..., start_sample : start_sample + target_audio_samples]
        if available.shape[-1] < target_audio_samples:
            if short_audio_policy == "strict":
                raise ValueError(
                    "Source audio is too short for the requested H3 window: "
                    f"needs {target_audio_samples} samples from {start_sample}, "
                    f"got {available.shape[-1]}"
                )
            padded_audio_samples = target_audio_samples - available.shape[-1]
            padding = available.new_zeros((*available.shape[:-1], padded_audio_samples))
            available = torch.cat((available, padding), dim=-1)
        waveform = available

    output_audio = {
        "waveform": waveform,
        "sample_rate": SOURCE_AUDIO_SAMPLE_RATE,
    }
    warnings = []
    if width * height > REFERENCE_PIXEL_AREA:
        warnings.append(
            "Source output exceeds the 1920x1088 reference area; execution remains allowed "
            "and VRAM/runtime/OOM risk is owned by the user."
        )
    if held_video_frames:
        warnings.append(f"Held the last source frame for {held_video_frames} target frames.")
    if padded_audio_samples:
        warnings.append(f"Padded {padded_audio_samples} source-audio samples with silence.")
    if audio_source == "generated_silence":
        warnings.append("No source audio was connected; generated a silent stereo track for VAE encode.")
    if not MIN_TRAINED_FRAMES <= frame_count <= MAX_TRAINED_FRAMES:
        warnings.append(
            f"{frame_count} target frames are outside the approximate H3 trained range "
            f"{MIN_TRAINED_FRAMES}-{MAX_TRAINED_FRAMES}."
        )

    report = {
        "schema": SOURCE_AV_SCHEMA,
        "status": "source_media_window_ready",
        "facts": {
            "input_frame_count": frames.shape[0],
            "input_resolution": [frames.shape[2], frames.shape[1]],
            "source_fps": float(source_fps),
            "start_seconds": float(start_seconds),
            "target_fps": FPS,
            "requested_length": int(length),
            "aligned_frame_count": frame_count,
            "target_resolution": [width, height],
            "duration_seconds": duration_seconds,
            "first_source_frame_index": int(source_indices[0].item()),
            "last_source_frame_index": int(source_indices[-1].item()),
            "held_video_frames": held_video_frames,
            "audio_source": audio_source,
            "input_audio_sample_rate": audio_input_rate,
            "input_audio_channels": audio_input_channels,
            "target_audio_sample_rate": SOURCE_AUDIO_SAMPLE_RATE,
            "target_audio_samples": target_audio_samples,
            "padded_audio_samples": padded_audio_samples,
        },
        "warnings": warnings,
        "claims": {
            "streaming_decode": False,
            "memory_safe": False,
            "arbitrary_video_supported": False,
        },
    }
    return (
        selected,
        output_audio,
        frame_count,
        duration_seconds,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _require_latent(value: Mapping, name: str) -> tuple[Mapping, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a connected LATENT value")
    if "samples" not in value:
        raise ValueError(f"{name} is missing samples")
    return value, value["samples"]


def _nested_parts(samples, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not getattr(samples, "is_nested", False):
        raise ValueError(f"{name} is not a joint AV latent")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly video and audio streams; got {len(parts)}")
    if not all(isinstance(part, torch.Tensor) for part in parts):
        raise ValueError(f"{name} contains a non-tensor stream")
    return parts


def _extract_masks(latent: Mapping, samples, name: str):
    masks = latent.get("noise_mask")
    if masks is None:
        return (None, None) if getattr(samples, "is_nested", False) else None
    if getattr(samples, "is_nested", False):
        if not getattr(masks, "is_nested", False):
            if isinstance(masks, torch.Tensor):
                return masks, None
            raise ValueError(f"{name} has an unsupported AV noise_mask")
        parts = tuple(masks.unbind())
        if len(parts) != 2:
            raise ValueError(f"{name} AV noise_mask must contain two streams")
        return parts
    if getattr(masks, "is_nested", False) or not isinstance(masks, torch.Tensor):
        raise ValueError(f"{name} has a noise_mask incompatible with its samples")
    return masks


def _extract_video_source(video_latent: Mapping):
    latent, samples = _require_latent(video_latent, "video_latent")
    if getattr(samples, "is_nested", False):
        video, fallback_audio = _nested_parts(samples, "video_latent")
        video_mask, fallback_audio_mask = _extract_masks(latent, samples, "video_latent")
        return video, video_mask, fallback_audio, fallback_audio_mask, True
    if not isinstance(samples, torch.Tensor):
        raise ValueError("video_latent samples must be a torch.Tensor or joint AV NestedTensor")
    return samples, _extract_masks(latent, samples, "video_latent"), None, None, False


def _extract_audio_source(audio_latent: Mapping | None, fallback_audio, fallback_mask):
    if audio_latent is None:
        if fallback_audio is None:
            raise ValueError("audio_latent is required when video_latent is not already a joint AV latent")
        return fallback_audio, fallback_mask, "existing_av_stream", None

    latent, samples = _require_latent(audio_latent, "audio_latent")
    if getattr(samples, "is_nested", False):
        _video, audio = _nested_parts(samples, "audio_latent")
        _video_mask, audio_mask = _extract_masks(latent, samples, "audio_latent")
        return audio, audio_mask, "audio_input_av_stream", latent
    if not isinstance(samples, torch.Tensor):
        raise ValueError("audio_latent samples must be a torch.Tensor or joint AV NestedTensor")
    return samples, _extract_masks(latent, samples, "audio_latent"), "audio_input", latent


def _validate_video(video: torch.Tensor) -> tuple[int, int, int]:
    if video.ndim != 5:
        raise ValueError(f"H3 video latent must be [B,24,T,H,W], got {tuple(video.shape)}")
    if video.shape[0] != 1:
        raise ValueError(f"MiniMax H3 currently supports video batch size 1; got {video.shape[0]}")
    if video.shape[1] != VIDEO_CHANNELS:
        raise ValueError(f"H3 video latent must have 24 channels; got {video.shape[1]}")
    if not video.is_floating_point():
        raise ValueError(f"H3 video latent must use a floating dtype; got {video.dtype}")
    if min(video.shape[2:]) <= 0:
        raise ValueError(f"H3 video latent cannot contain an empty dimension: {tuple(video.shape)}")
    if video.shape[-2] % 2 or video.shape[-1] % 2:
        raise ValueError(
            "H3 source canvas must be divisible by 32; encoded latent H/W must both be even, "
            f"got {tuple(video.shape[-2:])}"
        )
    frames = frame_count_from_video_latent_t(video.shape[2])
    height = video.shape[-2] * 16
    width = video.shape[-1] * 16
    return frames, width, height


def _validate_audio(audio: torch.Tensor, video_batch: int) -> None:
    if audio.ndim != 4:
        raise ValueError(f"H3 audio latent must be [B,32,2,T], got {tuple(audio.shape)}")
    if audio.shape[0] != video_batch:
        raise ValueError(
            f"H3 video/audio latent batches must match; video={video_batch}, audio={audio.shape[0]}"
        )
    if audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STEREO:
        raise ValueError(
            "H3 audio latent must use 32 channels and stereo dimension 2; "
            f"got {tuple(audio.shape)}"
        )
    if audio.shape[-1] <= 0:
        raise ValueError("H3 audio latent cannot be empty")
    if not audio.is_floating_point():
        raise ValueError(f"H3 audio latent must use a floating dtype; got {audio.dtype}")


def _mode_scale(mode: str, strength: float) -> float:
    if mode not in STREAM_MODES:
        raise ValueError(f"Unknown source AV stream mode {mode!r}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("Source AV denoise strength must be between 0 and 1")
    if mode == "lock":
        return 0.0
    if mode == "regenerate":
        return 1.0
    return float(strength)


def _effective_mask(mask, samples: torch.Tensor, scale: float) -> torch.Tensor:
    if mask is None:
        return torch.full_like(samples, scale)
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Source AV noise masks must be torch.Tensor values")
    mask = mask.to(device=samples.device, dtype=samples.dtype)
    if tuple(mask.shape) != tuple(samples.shape):
        mask = comfy.utils.reshape_mask(mask, samples.shape)
    return mask.clamp(0.0, 1.0) * scale


def _fit_audio_to_video(
    audio: torch.Tensor,
    audio_mask: torch.Tensor,
    target_t: int,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if policy not in AUDIO_FIT_POLICIES:
        raise ValueError(f"Unknown audio fit policy {policy!r}")
    current_t = audio.shape[-1]
    if current_t == target_t:
        return audio, audio_mask, "exact"
    if current_t > target_t:
        if policy not in {"trim_to_video", "fit_to_video_generate_tail"}:
            raise ValueError(
                f"Audio latent is longer than the H3 video clock ({current_t}>{target_t}); "
                "choose trim_to_video or fit_to_video_generate_tail explicitly"
            )
        return (
            audio[..., :target_t],
            audio_mask[..., :target_t],
            f"trimmed_{current_t - target_t}_latent_steps",
        )

    if policy not in {"pad_to_video_generate_tail", "fit_to_video_generate_tail"}:
        raise ValueError(
            f"Audio latent is shorter than the H3 video clock ({current_t}<{target_t}); "
            "choose pad_to_video_generate_tail or fit_to_video_generate_tail explicitly"
        )
    pad_t = target_t - current_t
    pad_shape = (*audio.shape[:-1], pad_t)
    padded_audio = torch.cat((audio, audio.new_zeros(pad_shape)), dim=-1)
    padded_mask = torch.cat((audio_mask, audio_mask.new_ones(pad_shape)), dim=-1)
    return padded_audio, padded_mask, f"padded_{pad_t}_latent_steps_generate_tail"


def _merge_metadata(video_latent: Mapping, audio_latent: Mapping | None):
    output = {
        key: value
        for key, value in video_latent.items()
        if key not in {"samples", "noise_mask"}
    }
    merged = []
    conflicts = []
    if audio_latent is not None:
        for key, value in audio_latent.items():
            if key in {"samples", "noise_mask"}:
                continue
            if key in output:
                conflicts.append(key)
                continue
            output[key] = value
            merged.append(key)
    return output, merged, conflicts


def prepare_source_av_latent(
    video_latent: Mapping,
    audio_latent: Mapping | None,
    video_mode: str,
    video_denoise_strength: float,
    audio_mode: str,
    audio_denoise_strength: float,
    audio_fit_policy: str,
    dtype_device_policy: str,
):
    if dtype_device_policy not in DTYPE_DEVICE_POLICIES:
        raise ValueError(f"Unknown dtype/device policy {dtype_device_policy!r}")

    video, video_mask, fallback_audio, fallback_audio_mask, input_was_av = _extract_video_source(
        video_latent
    )
    audio, audio_mask, audio_source, audio_mapping = _extract_audio_source(
        audio_latent, fallback_audio, fallback_audio_mask
    )
    frames, width, height = _validate_video(video)
    _validate_audio(audio, video.shape[0])

    original_audio_device = str(audio.device)
    original_audio_dtype = str(audio.dtype)
    converted_audio = False
    if audio.device != video.device or audio.dtype != video.dtype:
        if dtype_device_policy == "strict":
            raise ValueError(
                "H3 video/audio dtype and device must match in strict mode: "
                f"video={video.device}/{video.dtype}, audio={audio.device}/{audio.dtype}"
            )
        audio = audio.to(device=video.device, dtype=video.dtype)
        if audio_mask is not None:
            audio_mask = audio_mask.to(device=video.device, dtype=video.dtype)
        converted_audio = True

    video_scale = _mode_scale(video_mode, video_denoise_strength)
    audio_scale = _mode_scale(audio_mode, audio_denoise_strength)
    effective_video_mask = _effective_mask(video_mask, video, video_scale)
    effective_audio_mask = _effective_mask(audio_mask, audio, audio_scale)

    expected_audio_t = round(frames * AUDIO_LATENT_FPS / FPS)
    input_audio_t = audio.shape[-1]
    audio, effective_audio_mask, fit_action = _fit_audio_to_video(
        audio,
        effective_audio_mask,
        expected_audio_t,
        audio_fit_policy,
    )

    output, merged_metadata, metadata_conflicts = _merge_metadata(video_latent, audio_mapping)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (effective_video_mask, effective_audio_mask)
    )

    video_output = {
        key: value for key, value in output.items() if key not in {"samples", "noise_mask"}
    }
    audio_output = video_output.copy()
    video_output["samples"] = video
    audio_output["samples"] = audio
    video_output["noise_mask"] = effective_video_mask
    audio_output["noise_mask"] = effective_audio_mask

    warnings = []
    if width * height > REFERENCE_PIXEL_AREA:
        warnings.append(
            "Source AV latent exceeds the 1920x1088 reference area; execution remains allowed "
            "and VRAM/runtime/OOM risk is owned by the user."
        )
    if not MIN_TRAINED_FRAMES <= frames <= MAX_TRAINED_FRAMES:
        warnings.append(
            f"{frames} source frames are outside the approximate H3 trained range "
            f"{MIN_TRAINED_FRAMES}-{MAX_TRAINED_FRAMES}."
        )
    if converted_audio:
        warnings.append(
            "Audio latent dtype/device was matched to the video latent; this may allocate a new audio tensor."
        )
    if fit_action != "exact":
        warnings.append(f"Audio timeline was explicitly adjusted: {fit_action}.")
    if metadata_conflicts:
        warnings.append(
            "Conflicting audio metadata keys were not allowed to overwrite video metadata: "
            + ", ".join(metadata_conflicts)
        )

    report = {
        "schema": SOURCE_AV_SCHEMA,
        "status": "experimental_ready",
        "facts": {
            "input_video_was_av": input_was_av,
            "audio_source": audio_source,
            "video_shape": list(video.shape),
            "input_audio_shape": [audio.shape[0], audio.shape[1], audio.shape[2], input_audio_t],
            "output_audio_shape": list(audio.shape),
            "canvas": [width, height],
            "frame_count": frames,
            "duration_seconds": frames / FPS,
            "video_latent_fps": f"17n+5 frames at {FPS}fps",
            "audio_latent_fps": AUDIO_LATENT_FPS,
            "expected_audio_t": expected_audio_t,
            "audio_fit_policy": audio_fit_policy,
            "audio_fit_action": fit_action,
            "video_mode": video_mode,
            "video_effective_denoise": video_scale,
            "audio_mode": audio_mode,
            "audio_effective_denoise": audio_scale,
            "video_device": str(video.device),
            "video_dtype": str(video.dtype),
            "input_audio_device": original_audio_device,
            "input_audio_dtype": original_audio_dtype,
            "audio_converted_to_video": converted_audio,
            "merged_audio_metadata_keys": merged_metadata,
            "metadata_conflicts_kept_from_video": metadata_conflicts,
        },
        "warnings": warnings,
        "claims": {
            "denoise_strength_is_calibrated_linear_weight": False,
            "memory_safe": False,
            "arbitrary_video_supported": False,
            "temporal_concat": False,
        },
    }
    return output, video_output, audio_output, json.dumps(report, ensure_ascii=False, indent=2)


def separate_source_av_latent(av_latent: Mapping):
    latent, samples = _require_latent(av_latent, "av_latent")
    video, audio = _nested_parts(samples, "av_latent")
    frames, width, height = _validate_video(video)
    _validate_audio(audio, video.shape[0])
    expected_audio_t = round(frames * AUDIO_LATENT_FPS / FPS)
    if audio.shape[-1] != expected_audio_t:
        raise ValueError(
            "H3 AV latent audio clock does not match its video stream: "
            f"audio T={audio.shape[-1]}, expected {expected_audio_t} for {frames} frames"
        )

    masks = _extract_masks(latent, samples, "av_latent")
    video_mask, audio_mask = masks
    metadata = {
        key: value for key, value in latent.items() if key not in {"samples", "noise_mask"}
    }
    video_output = metadata.copy()
    audio_output = metadata.copy()
    video_output["samples"] = video
    audio_output["samples"] = audio
    if video_mask is not None:
        video_output["noise_mask"] = video_mask
    if audio_mask is not None:
        audio_output["noise_mask"] = audio_mask

    report = {
        "schema": SOURCE_AV_SCHEMA,
        "status": "valid_h3_av_latent",
        "facts": {
            "video_shape": list(video.shape),
            "audio_shape": list(audio.shape),
            "canvas": [width, height],
            "frame_count": frames,
            "duration_seconds": frames / FPS,
            "audio_t": audio.shape[-1],
            "has_video_noise_mask": video_mask is not None,
            "has_audio_noise_mask": audio_mask is not None,
            "metadata_keys": sorted(metadata),
        },
    }
    return video_output, audio_output, json.dumps(report, ensure_ascii=False, indent=2)
