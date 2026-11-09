"""Atomic 24fps H3 IMAGE/AUDIO output using the established isolated encoder."""

import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import long_video_delivery as delivery
from .h3_world_advanced import _normalize_output_audio, _mux_h3_world_audio
from .director_media import inspect_media, source_timing


def _verify_delivery(path, *, count, width, height, fps, rate, samples):
    """A playable file is insufficient: bind its actual clock to this request."""
    measured = inspect_media(path)
    if any(measured[key] != expected for key, expected in (
            ("frames", count), ("width", width), ("height", height), ("fps", fps),
            ("has_audio", True), ("audio_rate", rate), ("audio_channels", 2))):
        raise ValueError("AV delivery geometry/rate differs from the requested IMAGE/AUDIO")
    # AAC decoders expose the padded final frame. This is codec padding, not
    # permission to lose a source sample or append a second silent clip.
    padding = measured["audio_samples"] - samples
    if not 0 <= padding < 1024:
        raise ValueError("AV delivery decoded PCM length differs beyond one AAC frame of padding")
    timing = source_timing(path)
    tracks = timing["streams"]
    if len(tracks) != 2 or {row["kind"] for row in tracks} != {"video", "audio"}:
        raise ValueError("AV delivery must contain exactly one video and one audio track")
    from fractions import Fraction
    for row in tracks:
        if row["start_pts"] != 0 or row["duration_pts"] is None:
            raise ValueError("AV delivery timeline must have a measured zero origin and end")
        tick = Fraction(row["timebase"]["num"], row["timebase"]["den"])
        expected = Fraction(count, 24) if row["kind"] == "video" else Fraction(samples, rate)
        if abs(row["duration_pts"] * tick - expected) > tick:
            raise ValueError("AV delivery presentation endpoint differs from the requested interval")
    return {"decoded": measured, "source_timing": timing,
            "logical_audio_samples": samples, "aac_decode_padding_samples": padding,
            "full_decode_verified": True, "quality_reviewed": False}


def save_h3_av_safe(images, audio, output, fps=24, crf=18, preserve_audio_length=False):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("IMAGE must be [frames,height,width,channels]")
    count, height, width, channels = images.shape
    if min(count, height, width) <= 0 or width % 2 or height % 2 or channels not in (3, 4):
        raise ValueError("H3 safe output requires nonempty even-size RGB/RGBA frames")
    if fps != 24:
        raise ValueError("H3 safe output requires 24 fps")
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer from 0 to 51")
    output = Path(output).resolve()
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("H3 safe output must be a new MP4")
    if not isinstance(audio, dict):
        raise ValueError("AUDIO mapping required")
    rate = audio.get("sample_rate")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("invalid AUDIO sample rate")
    if type(preserve_audio_length) is not bool:
        raise ValueError("preserve_audio_length must be a boolean")
    expected_samples = math.ceil(count * rate / fps)
    if preserve_audio_length:
        waveform = audio.get("waveform")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("AUDIO waveform must be [batch,channels,samples]")
        samples = int(waveform.shape[-1])
        if not 0 < samples <= expected_samples:
            raise ValueError("Preserved AUDIO must be nonempty and fit inside the video interval")
        expected_samples = samples
    # AAC decode can yield finite float peaks above full scale. They do not
    # prove the source was hard-clipped, and f32le accepts those exact samples.
    # Preserve them instead of rejecting, limiting or normalizing the source.
    # The shared helper's historical H3-World clipping default stays unchanged.
    audio_array, rate, audio_report = _normalize_output_audio(
        audio, expected_samples=expected_samples, clip_amplitude=False)
    amplitude_report = {
        "policy": "preserve_finite_float32_samples",
        "source_peak_abs": float(np.max(np.abs(audio_array), initial=0.)),
        "above_full_scale_sample_values": int(np.count_nonzero(np.abs(audio_array) > 1.)),
        "gain_applied": 1.0,
        "clipped_by_this_saver": False,
        "scope": "pre-encoder f32le; lossy AAC does not promise decoded PCM identity",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb_hash = hashlib.sha256()

    def chunks():
        for frame in images:
            rgb = frame[..., :3].detach().float().cpu()
            if not torch.isfinite(rgb).all():
                raise ValueError("nonfinite IMAGE")
            raw = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
            rgb_hash.update(raw)
            yield raw

    with tempfile.TemporaryDirectory(prefix=".h3-av-encode-", dir=output.parent) as temporary:
        temp = Path(temporary)
        video, raw_audio, combined = temp / "video.mp4", temp / "audio.f32le", temp / "combined.mp4"
        delivery._encode_rgb_frames_isolated(video, chunks, frame_count=count, width=width,
                                            height=height, fps=fps, bit_depth=8, crf=crf)
        delivery._strict_validate_mp4(video, require_audio=False)
        delivery._write_planar_audio_raw(raw_audio, audio_array)
        _mux_h3_world_audio(video, raw_audio, combined, sample_rate=rate, duration_seconds=count / fps)
        delivery._strict_validate_mp4(combined, require_audio=True)
        verification = _verify_delivery(combined, count=count, width=width, height=height,
                                        fps=fps, rate=rate, samples=expected_samples)
        # Flush our own completed private file before publishing its hard link.
        # Windows cannot fsync a read-only handle; never open user masters for
        # writing merely to bind a later postprocessing job.
        with combined.open("r+b") as handle:
            os.fsync(handle.fileno())
        # Same-volume publication must never replace a concurrently created file.
        os.link(combined, output)
    return {"schema": "t8.h3.safe_av_output.v1", "status": "pass",
            "encoder_policy": delivery.ISOLATED_VIDEO_ENCODER_POLICY,
            "source_rgb8_sha256": rgb_hash.hexdigest(), "frames": count, "width": width,
            "source_pcm_f32_sha256": hashlib.sha256(audio_array.tobytes()).hexdigest(),
            "height": height, "fps": fps, "crf": crf, "audio": audio_report,
            "audio_amplitude": amplitude_report,
            "delivery_verification": verification,
            "output_sha256": delivery._sha256_file(output), "output": str(output)}
