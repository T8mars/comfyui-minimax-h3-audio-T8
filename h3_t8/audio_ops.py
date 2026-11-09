from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as torch_functional
import torchaudio

from comfy_extras.nodes_audio import vae_decode_audio

from .core import encode_audio_once, nested_av_parts, replace_audio_latent, validate_audio


def inject_audio_latent(av_latent: dict, source_audio: dict, audio_vae, mode: str, strength: float):
    mode = mode.lower()
    if mode not in {"lock", "remix"}:
        raise ValueError("Audio latent control mode must be lock or remix")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    encoded = encode_audio_once(audio_vae, source_audio)
    denoise = 0.0 if mode == "lock" else strength
    return replace_audio_latent(av_latent, encoded, denoise), source_audio


def decode_av_latent(av_latent: dict, video_vae, audio_vae):
    video, audio = nested_av_parts(av_latent)
    images = video_vae.decode(video)
    if not isinstance(images, torch.Tensor) or images.ndim not in (4, 5):
        raise ValueError("Video VAE returned invalid IMAGE geometry")
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    if min(images.shape) < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("Video VAE returned invalid IMAGE geometry")
    # Validate before any downstream saver can clamp/cast NaN to uint8. Bound
    # the temporary boolean storage to one frame, not the entire decoded clip.
    if any(not bool(torch.isfinite(frame).all()) for frame in images):
        raise ValueError("Video VAE returned nonfinite IMAGE values")
    decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
    waveform, rate = validate_audio(decoded_audio, "decoded_audio")
    if rate <= 0 or not bool(torch.isfinite(waveform).all()):
        raise ValueError("Audio VAE returned invalid or nonfinite AUDIO values")
    video_latent = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
    audio_latent = video_latent.copy()
    video_latent["samples"] = video
    audio_latent["samples"] = audio
    masks = av_latent.get("noise_mask")
    if getattr(masks, "is_nested", False):
        video_mask, audio_mask = masks.unbind()
        video_latent["noise_mask"] = video_mask
        audio_latent["noise_mask"] = audio_mask
    return images, decoded_audio, video_latent, audio_latent


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _match_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    channels = waveform.shape[1]
    if channels == target_channels:
        return waveform
    if channels == 1:
        return waveform.expand(-1, target_channels, -1)
    if target_channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    # For uncommon multichannel inputs, make a deterministic stereo-compatible downmix.
    mono = waveform.mean(dim=1, keepdim=True)
    return mono.expand(-1, target_channels, -1)


def mix_audio(
    source_audio: dict,
    generated_audio: dict,
    source_gain_db: float = 0.0,
    generated_gain_db: float = -6.0,
    duck_generated: float = 0.5,
    output_sample_rate: str = "source",
    peak_limit_dbfs: float = -1.0,
):
    source, source_rate = validate_audio(source_audio, "source_audio")
    generated, generated_rate = validate_audio(generated_audio, "generated_audio")
    if output_sample_rate == "source":
        target_rate = source_rate
    elif output_sample_rate == "generated":
        target_rate = generated_rate
    else:
        target_rate = int(output_sample_rate)

    source = _resample(source, source_rate, target_rate)
    generated = _resample(generated, generated_rate, target_rate)
    target_channels = max(source.shape[1], generated.shape[1])
    source = _match_channels(source, target_channels)
    generated = _match_channels(generated, target_channels)
    target_samples = max(source.shape[-1], generated.shape[-1])
    source = torch_functional.pad(source, (0, target_samples - source.shape[-1]))
    generated = torch_functional.pad(generated, (0, target_samples - generated.shape[-1]))

    source = source * (10.0 ** (source_gain_db / 20.0))
    generated = generated * (10.0 ** (generated_gain_db / 20.0))
    if duck_generated > 0:
        envelope = source.abs().mean(dim=1, keepdim=True)
        window = max(1, round(target_rate * 0.02))
        envelope = torch_functional.avg_pool1d(envelope, window, stride=1, padding=window // 2)
        envelope = envelope[..., :target_samples]
        activity = (envelope / 0.05).clamp(0.0, 1.0)
        generated = generated * (1.0 - float(duck_generated) * activity)

    mixed = source + generated
    limit = 10.0 ** (peak_limit_dbfs / 20.0)
    peak = mixed.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(peak), torch.full_like(peak, limit) / peak)
    mixed = mixed * scale
    return {"waveform": mixed, "sample_rate": target_rate}


def _absolute_sample_tick(seconds, sample_rate, source_origin=0.0):
    """Nearest global PCM tick, with stable ties across equivalent float sums.

    At 44.1kHz a 24fps boundary can be exactly half a sample. Independent
    representations of start+duration differ by a few floating point ulps;
    snap only that arithmetic uncertainty to the half tick before ties-to-even.
    """
    tick = seconds * sample_rate
    if not math.isfinite(tick):
        raise ValueError("Audio time coordinate exceeds finite sample precision")
    tolerance = 4 * max(math.ulp(tick), math.ulp(float(seconds)) * sample_rate,
                        math.ulp(float(source_origin)) * sample_rate)
    if tolerance >= .25:
        raise ValueError("Audio time coordinate cannot resolve individual PCM samples; rebase source and requested times together")
    half_tick = round(tick * 2) / 2
    if abs(tick - half_tick) <= tolerance:
        tick = half_tick
    return round(tick)


def trim_av_output(frames: torch.Tensor, start_seconds: float, duration_seconds: float, audio=None,
                   fps: float = 24.0, frame_count: int = 0, *, source_time_origin_seconds: float = 0.0,
                   audio_boundary_policy: str = "legacy_duration"):
    if frames.ndim != 4:
        raise ValueError(f"frames must be IMAGE [N,H,W,C], got {tuple(frames.shape)}")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           for value in (start_seconds, duration_seconds, fps, source_time_origin_seconds)):
        raise ValueError("trim times, source time origin and fps must be finite numbers")
    if start_seconds < 0 or duration_seconds <= 0 or fps <= 0:
        raise ValueError("trim start must be nonnegative; duration and fps must be positive")
    if audio_boundary_policy not in ("legacy_duration", "absolute_endpoints"):
        raise ValueError("Unknown audio_boundary_policy")
    local_start = start_seconds - source_time_origin_seconds
    if local_start < 0:
        raise ValueError("Requested trim starts before the source time origin; video frames cannot be padded")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a nonnegative integer; 0 keeps legacy duration rounding")
    start_frame = round(local_start * fps)
    # A compiler's discrete video endpoint is independent of the requested
    # audio interval. Keep old callers' rounding and PCM coordinates unchanged.
    frame_count = frame_count or max(1, round(duration_seconds * fps))
    end_frame = start_frame + frame_count
    if end_frame > frames.shape[0]:
        raise ValueError(
            f"Requested output frames [{start_frame}:{end_frame}] exceed decoded frame count {frames.shape[0]}"
        )
    trimmed_frames = frames[start_frame:end_frame]

    trimmed_audio = None
    audio_mapping = None
    if audio is not None:
        waveform, sample_rate = validate_audio(audio)
        if sample_rate <= 0 or not bool(torch.isfinite(waveform).all()):
            raise ValueError("trim AUDIO must have a positive sample rate and finite PCM")
        absolute_start = round(start_seconds * sample_rate)
        absolute_end = round((start_seconds + duration_seconds) * sample_rate)
        origin_sample = round(source_time_origin_seconds * sample_rate)
        if audio_boundary_policy == "absolute_endpoints":
            absolute_start = _absolute_sample_tick(start_seconds, sample_rate, source_time_origin_seconds)
            absolute_end = _absolute_sample_tick(start_seconds + duration_seconds, sample_rate, source_time_origin_seconds)
            origin_sample = _absolute_sample_tick(source_time_origin_seconds, sample_rate)
            start_sample, end_sample = absolute_start - origin_sample, absolute_end - origin_sample
        else:
            start_sample = round(local_start * sample_rate)
            end_sample = start_sample + round(duration_seconds * sample_rate)
        requested_absolute_start, requested_absolute_end = absolute_start, absolute_end
        absolute_start, absolute_end = origin_sample + start_sample, origin_sample + end_sample
        sample_count = end_sample - start_sample
        sliced = waveform[..., start_sample:end_sample]
        actual_samples = int(sliced.shape[-1])
        if sliced.shape[-1] < sample_count:
            sliced = torch_functional.pad(sliced, (0, sample_count - sliced.shape[-1]))
        trimmed_audio = {"waveform": sliced, "sample_rate": sample_rate}
        audio_mapping = {
            "sample_rate": sample_rate, "absolute_start_sample": absolute_start,
            "absolute_end_sample": absolute_end, "source_origin_sample": origin_sample,
            "requested_absolute_start_sample": requested_absolute_start,
            "requested_absolute_end_sample": requested_absolute_end,
            "local_start_sample": start_sample, "local_end_sample": end_sample,
            "source_samples": int(waveform.shape[-1]), "copied_samples": actual_samples,
            "right_padding_samples": sample_count - actual_samples,
            "output_samples": sample_count, "resampled": False,
            "boundary_rounding": ("global_nearest_ties_even_4ulp_half_tick" if audio_boundary_policy == "absolute_endpoints"
                                  else "legacy_local_start_plus_rounded_duration"),
        }

    report = json.dumps(
        {
            "start_seconds": start_seconds,
            "requested_duration_seconds": duration_seconds,
            "fps": fps,
            "start_frame": start_frame,
            "frame_count": frame_count,
            "actual_video_duration_seconds": frame_count / fps,
            "source_time_origin_seconds": source_time_origin_seconds,
            "local_start_seconds": local_start,
            "audio_boundary_policy": audio_boundary_policy,
            "audio_mapping": audio_mapping,
        },
        ensure_ascii=False,
        indent=2,
    )
    return trimmed_frames, trimmed_audio, report
