"""Explicit Windows Job-owned FFmpeg export, with bounded raw-data staging."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import av
import numpy as np
import psutil
import torch

from ..dlss_fi_backend.process import IsolatedTaskError, run_isolated
from .storage import _root, _path, file_sha
from .video_io import components_serial, _validate


def _tensor_identity(value, check):
    """Hash logical row-major bytes, including views, in at most 4 MiB copies."""
    if value.device.type == "meta" or value.layout != torch.strided or value.is_quantized:
        raise ValueError("Media identity needs a materialized strided tensor")
    digest = hashlib.sha256()
    limit = max(1, 4 * 1024**2 // value.element_size())

    def visit(part):
        check()
        if part.numel() <= limit:
            chunk = part.detach().cpu().contiguous()
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("Media identity contains non-finite values")
            digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif part.shape[0] == 1:
            visit(part[0])
        else:
            rows = max(1, limit // math.prod(part.shape[1:]))
            for start in range(0, part.shape[0], rows):
                visit(part[start:start + rows])

    visit(value)
    return {"tensor_sha256": digest.hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}


def bounded_component_identity(parts, check=lambda: None):
    """Same data identity as serial I/O, without materializing a whole video view."""
    return {"images": _tensor_identity(parts.images, check),
            "audio": None if parts.audio is None else {
                "waveform": _tensor_identity(parts.audio["waveform"], check),
                "sample_rate": parts.audio["sample_rate"]},
            "frame_rate": str(Fraction(parts.frame_rate))}


def save_isolated(video, output_root, prefix, metadata=None, interrupt=lambda: None,
                  *, timeout=600, max_staging_gib=16, min_free_disk_gib=2):
    if os.name != "nt":
        raise RuntimeError("The isolated video exporter currently requires Windows Job Objects")
    for name, value, maximum in (("timeout", timeout, 3600), ("max_staging_gib", max_staging_gib, 1024),
                                  ("min_free_disk_gib", min_free_disk_gib, 1024)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("A finite positive bounded " + name + " is required")
    started = time.monotonic()
    deadline = started + timeout
    def check():
        interrupt()
        if time.monotonic() >= deadline:
            raise TimeoutError("Isolated export deadline exceeded")
    check()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("Isolated export requires installed ffmpeg and ffprobe")
    parts, read_report = components_serial(video, check)
    depth, color = video.get_bit_depth(), video.get_color_space() or "sRGB"
    frames, original_fps, audio = _validate(parts, depth, color)
    fps = Fraction(round(original_fps * 1000), 1000)
    if fps <= 0:
        raise ValueError("Frame rate rounds to zero at native MP4 precision")
    n, height, width, _ = frames.shape
    video_bytes = n * width * height * 3 // 2 * (2 if depth == 10 else 1)
    audio_samples = min(audio["waveform"].shape[-1], math.ceil(audio["sample_rate"] / fps * n)) if audio else 0
    audio_bytes = audio_samples * audio["waveform"].shape[1] * 4 if audio else 0
    if video_bytes + audio_bytes > max_staging_gib * 1024**3:
        raise ValueError("Raw staging exceeds the selected disk budget")
    # Conversion is one frame/audio chunk at a time, not a second full RGB copy.
    if width * height * 48 + 64 * 1024**2 > psutil.virtual_memory().available:
        raise MemoryError("Insufficient host memory for bounded frame conversion")
    root = _root(output_root, create=True)
    reserve = int(min_free_disk_gib * 1024**3)
    if shutil.disk_usage(root).free < 2 * (video_bytes + audio_bytes) + reserve:
        raise OSError("Insufficient free disk for raw staging, candidate and reserve")
    selected = _path(root, prefix)
    owned = selected.with_name(selected.name + "-" + uuid.uuid4().hex)
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.mkdir()
    identity = bounded_component_identity(parts, check)
    report = {"status": "preparing", "input_identity": identity, "read_boundary": read_report,
              "width": width, "height": height, "frames": n, "fps": float(fps), "source_fps": str(original_fps),
              "bit_depth": depth, "color_space": color, "quality_accepted": False,
              "video_encoder_threads": 1, "color_conversion_threads": 1, "audio_encoder_threads": 1 if audio else None,
              "audio_policy": "original_rate_channels_and_waveform_core_duration_trim"}
    try:
        quantized, converted = hashlib.sha256(), hashlib.sha256()
        with (owned / "frames.yuv").open("xb") as output:
            for tensor in frames:
                check()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Cannot export non-finite RGB")
                pixels = ((tensor.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16) if depth == 10
                          else (tensor * 255).clamp(0, 255).byte().cpu().numpy())
                quantized.update(pixels.tobytes())
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le" if depth == 10 else "rgb24")
                frame = frame.reformat(format="yuv420p10le" if depth == 10 else "yuv420p",
                                       dst_colorspace=1 if color == "sRGB" else 9, threads=1)
                for plane in frame.planes:
                    values = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                    packed = values[:, :plane.width * (2 if depth == 10 else 1)].tobytes()
                    converted.update(packed)
                    output.write(packed)
                if shutil.disk_usage(owned).free < reserve:
                    raise OSError("Disk reserve was consumed while staging video")
            output.flush()
            os.fsync(output.fileno())
        audio_info = None
        if audio:
            with (owned / "audio.f32le").open("xb") as output:
                for start in range(0, audio_samples, 65536):
                    check()
                    chunk = audio["waveform"][0, :, start:min(audio_samples, start + 65536)]
                    output.write(chunk.T.float().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes())
                output.flush()
                os.fsync(output.fileno())
            audio_info = {"sample_rate": audio["sample_rate"], "channels": audio["waveform"].shape[1],
                          "bytes": audio_bytes, "sha": file_sha(owned / "audio.f32le")}
        if (owned / "frames.yuv").stat().st_size != video_bytes:
            raise ValueError("Staged YUV geometry changed")
        request = {"schema": "t8.raw-yuv-isolated-export.v1", "width": width, "height": height, "frames": n,
                   "fps": str(fps), "bit_depth": depth, "color_space": color, "video_bytes": video_bytes,
                   "video_sha": converted.hexdigest(), "audio": audio_info, "metadata": metadata or {},
                   "ffmpeg": str(Path(ffmpeg).resolve()), "ffprobe": str(Path(ffprobe).resolve())}
        encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf8")
        if len(encoded) > 4 * 1024**2 or any(type(k) is not str or "\x00" in k for k in request["metadata"]):
            raise ValueError("Oversized or invalid video metadata")
        request_path = owned / "request.json"
        with request_path.open("xb") as stream:
            stream.write(encoded)
        expected = hashlib.sha256(encoded).hexdigest()
        interruptions = []
        def worker_check():
            try:
                check()
            except BaseException as error:
                interruptions.append(error)
                raise
        check()
        try:
            isolation = run_isolated(Path(__file__).with_name("video_ffmpeg_worker.py"),
                [str(request_path), expected], timeout=min(3600, deadline - time.monotonic()), check=worker_check)
        except IsolatedTaskError as error:
            report["isolation"] = error.receipt
            if interruptions:
                raise interruptions[0]
            raise
        report["isolation"] = isolation
        check()
        worker = json.loads((owned / "worker-report.json").read_text(encoding="utf8"))
        partial, final = owned / "video.partial.mp4", owned / "video.mp4"
        if (worker["status"] != "validated_not_published" or worker["request_sha"] != expected
                or worker["strict_decode_verified"] is not True or file_sha(partial) != worker["file_sha256"]
                or file_sha(request_path) != expected or bounded_component_identity(parts, check) != identity):
            raise ValueError("Worker output or source changed before publication")
        with partial.open("rb+") as stream:
            os.fsync(stream.fileno())
        check()
        if final.exists():
            raise FileExistsError("Isolated output already exists")
        partial.rename(final)
        report.update(status="isolated_h264_written", strict_decode_verified=True, worker=worker,
                      quantized_rgb_sha256=quantized.hexdigest(), converted_yuv_sha256=converted.hexdigest(), path=str(final))
        # Only this invocation's verified staging buffers are disposable; retain
        # manifests/reports and all failed-run files. Never recurse/delete roots.
        try:
            (owned / "frames.yuv").unlink()
            if audio:
                (owned / "audio.f32le").unlink()
        except OSError as error:
            report["staging_cleanup_warning"] = str(error)
        return final, report
    except BaseException as error:
        report.update(status="failed_not_published", error=f"{type(error).__name__}: {error}")
        with (owned / "failure.json").open("x", encoding="utf8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        raise
