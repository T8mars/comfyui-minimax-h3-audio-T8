"""Atomic 24fps H3 IMAGE/AUDIO output using the established isolated encoder."""

import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from . import long_video_delivery as delivery
from .h3_world_advanced import _normalize_output_audio, _mux_h3_world_audio


def save_h3_av_safe(images, audio, output, fps=24, crf=18):
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
    audio_array, rate, audio_report = _normalize_output_audio(audio, expected_samples=math.ceil(count * rate / fps))
    if audio_report["clipped_sample_values"]:
        raise ValueError("H3 safe output refuses clipped source audio")
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
            "height": height, "fps": fps, "crf": crf, "audio": audio_report,
            "output_sha256": delivery._sha256_file(output), "output": str(output)}
