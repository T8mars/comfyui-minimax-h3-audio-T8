from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
MIN_TRAINED_FRAMES = 124
MAX_TRAINED_FRAMES = 362


def align_frame_count(frame_count: int) -> int:
    """Snap up to MiniMax H3's 17n+5 frame grid."""
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def align_frame_count_down(frame_count: int) -> int:
    frame_count = int(frame_count)
    return frame_count - ((frame_count - 5) % 17)


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image: torch.Tensor, width: int, height: int, crop: str = "disabled") -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def empty_av_latent(width: int, height: int, length: int) -> tuple[dict, int]:
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, audio_t), device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def nested_av_parts(av_latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected a MiniMax H3 joint AV LATENT")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected a nested MiniMax H3 joint video/audio latent")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two AV latent parts, got {len(parts)}")
    video, audio = parts
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "Unexpected MiniMax H3 AV latent layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 currently supports batch size 1 only")
    return video, audio


def validate_audio(audio: dict, name: str = "audio") -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"{name} must be a connected AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError(f"{name} is missing waveform or sample_rate")
    if waveform.ndim != 3:
        raise ValueError(f"{name} must use [batch,channels,samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"{name} is empty")
    if waveform.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1 for MiniMax H3")
    return waveform, int(sample_rate)


def encode_audio_once(audio_vae, audio: dict) -> torch.Tensor:
    waveform, sample_rate = validate_audio(audio)
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("The audio VAE did not return [B,C,stereo,T] latent data")
    return latent


def fit_audio_latent(encoded_audio: torch.Tensor, template_audio: torch.Tensor) -> torch.Tensor:
    if encoded_audio.ndim != 4 or template_audio.ndim != 4:
        raise ValueError("MiniMax H3 audio latents must use [B,C,stereo,T]")
    if encoded_audio.shape[1:-1] != template_audio.shape[1:-1]:
        raise ValueError(
            "Audio VAE latent layout mismatch: "
            f"got {tuple(encoded_audio.shape)}, target {tuple(template_audio.shape)}"
        )
    if encoded_audio.shape[0] != template_audio.shape[0]:
        if encoded_audio.shape[0] == 1:
            encoded_audio = encoded_audio.expand(template_audio.shape[0], -1, -1, -1)
        else:
            raise ValueError("Audio latent batch cannot be matched to the AV latent")
    target_t = template_audio.shape[-1]
    if encoded_audio.shape[-1] > target_t:
        encoded_audio = encoded_audio[..., :target_t]
    elif encoded_audio.shape[-1] < target_t:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - encoded_audio.shape[-1]))
        encoded_audio = torch.cat((encoded_audio, padding), dim=-1)
    return encoded_audio.to(device=template_audio.device, dtype=template_audio.dtype)


def sorted_autogrow_items(values: Mapping | None) -> list[tuple[int, object]]:
    if not values:
        return []

    def sort_key(item):
        key = str(item[0])
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return 10_000

    output = []
    for key, value in sorted(values.items(), key=sort_key):
        if value is None:
            continue
        try:
            ordinal = int(str(key).rsplit("_", 1)[-1])
        except ValueError:
            ordinal = len(output) + 1
        output.append((ordinal, value))
    return output


def sorted_autogrow_values(values: Mapping | None) -> list:
    return [value for _, value in sorted_autogrow_items(values)]


def split_noise_masks(av_latent: dict, video: torch.Tensor, audio: torch.Tensor):
    masks = av_latent.get("noise_mask")
    if masks is None:
        return None, None
    if getattr(masks, "is_nested", False):
        parts = tuple(masks.unbind())
        if len(parts) == 2:
            return parts
    # A legacy video-only mask must never be silently discarded.
    if isinstance(masks, torch.Tensor):
        return masks, None
    raise ValueError("Unsupported AV noise_mask layout")


def replace_audio_latent(av_latent: dict, encoded_audio: torch.Tensor, denoise_strength: float) -> dict:
    video, template_audio = nested_av_parts(av_latent)
    fitted = fit_audio_latent(encoded_audio, template_audio)
    video_mask, _ = split_noise_masks(av_latent, video, template_audio)
    if video_mask is None:
        video_mask = torch.ones_like(video)
    audio_mask = torch.full_like(fitted, float(denoise_strength))
    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((video, fitted))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return output
