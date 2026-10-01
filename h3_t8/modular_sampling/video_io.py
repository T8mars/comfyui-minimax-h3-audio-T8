"""Explicit serial codec boundaries for separate graphs, never a global patch.

The writer follows ComfyUI's VideoFromComponents MP4/H.264 conversion order
(comfy_api/latest/_input_impl/video_types.py, GPL-3.0), with explicit per-stream
thread ownership, cancellation checkpoints and an exclusive publication path.
Old Core video objects and nodes remain unchanged.
"""
from fractions import Fraction
import hashlib
import json
import math
import os
import uuid

import av
import numpy as np
import torch

from .storage import _root, _path


class _CheckedPacket:
    def __init__(self, packet, interrupt):
        self.packet, self.interrupt = packet, interrupt

    def __getattr__(self, name):
        return getattr(self.packet, name)

    def decode(self):
        self.interrupt()
        try:
            for frame in self.packet.decode():
                self.interrupt()
                yield frame
        except av.error.InvalidDataError as error:
            # Core catches InvalidDataError and continues. At this opt-in
            # boundary an incomplete video is a failure, never a valid source.
            raise ValueError("Serial source packet failed to decode") from error


class _CheckedContainer:
    def __init__(self, container, interrupt):
        self.container, self.interrupt = container, interrupt

    def __getattr__(self, name):
        return getattr(self.container, name)

    def demux(self, *streams):
        for packet in self.container.demux(*streams):
            self.interrupt()
            yield _CheckedPacket(packet, self.interrupt)


def components_serial(video, interrupt=lambda: None):
    from comfy_api.latest import InputImpl
    interrupt()
    if isinstance(video, InputImpl.VideoFromFile):
        source = video.get_stream_source()
        if hasattr(source, "seek"):
            source.seek(0)
        with av.open(source, mode="r") as container:
            for stream in container.streams:
                if stream.type in ("video", "audio") and stream.codec_context is not None:
                    stream.codec_context.thread_count = 1
            # Native Core owns trim/crop/rotation, float RGB conversion and audio
            # selection. Its AUTO thread_type cannot override explicit count=1.
            result = video.get_components_internal(_CheckedContainer(container, interrupt))
        backend = "native_file_decode_serial"
    else:
        # Materialized components preserve tensor/audio identities. Unknown
        # providers keep their own decoder; do not claim to control that code.
        result = video.get_components()
        backend = "materialized_components" if isinstance(video, InputImpl.VideoFromComponents) else "external_provider_unverified"
    interrupt()
    return result, {"status": "components_ready", "backend": backend,
                    "decoder_threads": 1 if backend == "native_file_decode_serial" else None,
                    "strict_decode_verified": False,
                    "sampler_executed": False, "quality_accepted": False}


def component_identity(components):
    """Logical data identity for repeat-read comparisons, not a model receipt."""
    from .results import _input_identity
    return {"images": _input_identity(components.images),
            "audio": _input_identity(None if components.audio is None else {
                "waveform": components.audio["waveform"], "sample_rate": components.audio["sample_rate"]}),
            "frame_rate": str(Fraction(components.frame_rate))}


def _validate(components, bit_depth, color_space):
    from comfy_api.latest._input_impl.video_types import VIDEO_COLOR_TRANSFERS
    frames = components.images
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3
            or min(frames.shape) < 1 or not frames.is_floating_point()
            or frames.shape[1] % 2 or frames.shape[2] % 2):
        raise ValueError("H.264 serial output needs nonempty floating RGB with even dimensions")
    if type(bit_depth) is not int or bit_depth not in (8, 10) or color_space not in VIDEO_COLOR_TRANSFERS:
        raise ValueError("Explicit supported 8/10-bit depth and color space required")
    if getattr(components, "alpha", None) is not None:
        raise ValueError("H.264 cannot preserve alpha; explicitly composite it first")
    fps = Fraction(components.frame_rate)
    if fps <= 0 or not math.isfinite(float(fps)):
        raise ValueError("Video fps must be positive and finite")
    audio = components.audio
    if audio is not None:
        if not isinstance(audio, dict):
            raise ValueError("Audio must be a waveform/sample_rate dictionary")
        waveform = audio.get("waveform")
        if (type(audio.get("sample_rate")) is not int or audio["sample_rate"] <= 0
                or not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
                or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2, 6)
                or waveform.shape[2] < 1 or not waveform.is_floating_point()
                or not torch.isfinite(waveform).all()):
            raise ValueError("Audio must contain finite mono/stereo/5.1 samples at its original rate")
    return frames, fps, audio


def write_serial(components, path, bit_depth, color_space, metadata=None, interrupt=lambda: None):
    from comfy_api.latest._input_impl.video_types import set_video_color_properties
    frames, original_fps, audio = _validate(components, bit_depth, color_space)
    # Identical rounding to Core's existing writer, no silent fps resampling.
    fps = Fraction(round(original_fps * 1000), 1000)
    quantized_sha, yuv_sha = hashlib.sha256(), hashlib.sha256()
    interrupt()
    with av.open(str(path), mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
        for key, value in (metadata or {}).items():
            output.metadata[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
        stream.pix_fmt = pix_fmt
        stream.codec_context.thread_count = 1
        set_video_color_properties(stream.codec_context, color_space)
        audio_stream = None
        if audio is not None:
            rate = audio["sample_rate"]
            waveform = audio["waveform"][0, :, :math.ceil(rate / fps * frames.shape[0])]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}[waveform.shape[0]]
            audio_stream = output.add_stream("aac", rate=rate, layout=layout)
            audio_stream.codec_context.thread_count = 1
        for tensor in frames:
            interrupt()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Cannot export non-finite RGB")
            if bit_depth == 10:
                pixels = (tensor.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
            else:
                pixels = (tensor * 255).clamp(0, 255).byte().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            quantized_sha.update(pixels.tobytes())
            frame = frame.reformat(format=pix_fmt, dst_colorspace=1 if color_space == "sRGB" else 9, threads=1)
            # Hash visible values only: FFmpeg plane padding is not image data.
            for plane in frame.planes:
                visible = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                yuv_sha.update(visible[:, :plane.width * (2 if bit_depth == 10 else 1)].tobytes())
            set_video_color_properties(frame, color_space)
            output.mux(stream.encode(frame))
        output.mux(stream.encode(None))
        if audio_stream is not None:
            interrupt()
            frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
            frame.sample_rate, frame.pts = rate, 0
            output.mux(audio_stream.encode(frame))
            output.mux(audio_stream.encode(None))
    interrupt()
    return {"status": "serial_h264_written", "width": frames.shape[2], "height": frames.shape[1],
            "frames": frames.shape[0], "fps": float(fps), "source_fps": str(original_fps),
            "bit_depth": bit_depth, "color_space": color_space, "video_encoder_threads": 1,
            "color_conversion_threads": 1, "quantized_rgb_sha256": quantized_sha.hexdigest(),
            "converted_yuv_sha256": yuv_sha.hexdigest(),
            "audio_encoder_threads": 1 if audio is not None else None,
            "audio_policy": "original_rate_channels_and_waveform_core_duration_trim",
            "quality_accepted": False, "strict_decode_verified": False}


def save_serial(video, output_root, prefix, metadata=None, interrupt=lambda: None):
    interrupt()
    components, decode_report = components_serial(video, interrupt)
    depth, color = video.get_bit_depth(), video.get_color_space() or "sRGB"
    _validate(components, depth, color)
    root = _root(output_root, create=True)
    selected = _path(root, prefix)
    directory = selected.with_name(selected.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # One invocation owns this namespace. Never reuse files.
    partial, final = directory / "video.partial.mp4", directory / "video.mp4"
    report = write_serial(components, partial, depth, color, metadata, interrupt)
    with partial.open("rb+") as stream:
        os.fsync(stream.fileno())
    report["input_identity"] = component_identity(components)
    interrupt()
    if final.exists():
        raise FileExistsError("Refusing to overwrite serial video output")
    partial.rename(final)
    report["read_boundary"] = decode_report
    report["path"] = str(final)
    return final, report
