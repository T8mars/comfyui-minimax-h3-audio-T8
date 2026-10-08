"""One actual Source AV preparation, with content-bound temporal lineage.

Opt-in only. No alternate resampler, VAE, sampler, or global decoder patch.
Opaque controls can still be mapped, but same shape/JSON never certifies origin.
"""
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math

import torch
import comfy.utils

from .core import resize_image
from .source_av import prepare_source_media_window

_OWNER = object()
MAX_FRAMES = 4096
MAX_TENSOR_BYTES = 2 * 1024**3
SCHEMA = "h3_source_conform_v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    """Hash in frame-sized blocks: never duplicate a full resident video on CPU."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim < 1:
        raise ValueError("Lineage requires a floating tensor")
    size = value.numel() * value.element_size()
    if not size or size > MAX_TENSOR_BYTES or value.shape[0] > MAX_FRAMES:
        raise ValueError("Lineage tensor exceeds the explicit 2GiB/4096-frame budget or is empty")
    shape = tuple(value.shape)
    digest = hashlib.sha256(canonical({"shape": shape, "dtype": str(value.dtype)}).encode())
    for item in value.detach():
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        block = item.to(device="cpu").contiguous()
        if not torch.isfinite(block).all().item():
            raise ValueError("Lineage tensor contains NaN or infinity")
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audio_identity(audio):
    if audio is None:
        return None
    return (int(audio["sample_rate"]), identity(audio["waveform"]))


@dataclass(frozen=True)
class MediaClock:
    frames_sha: str
    audio_identity: tuple | None
    frame_rate: Fraction
    pts_seconds: tuple[Fraction, ...]
    active_origin: Fraction
    status: str
    evidence_json: str
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class FrameMap:
    source_sha: str
    output_sha: str
    output_audio_identity: tuple
    indices: tuple[int, ...]
    target_times: tuple[float, ...]
    source_count: int
    source_geometry: tuple[int, int]
    output_geometry: tuple[int, int]
    clock: MediaClock | None
    trace_json: str
    map_sha: str
    _source: torch.Tensor = field(repr=False, compare=False)
    _output: torch.Tensor = field(repr=False, compare=False)
    _audio: object = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DerivedControl:
    source_sha: str
    output_sha: str
    kind: str
    operation: str
    _owner: object = field(repr=False, compare=False)


def _map_payload(value):
    return dict(schema=SCHEMA, source_sha=value.source_sha, output_sha=value.output_sha,
                audio_identity=value.output_audio_identity, indices=value.indices,
                target_times=value.target_times, source_count=value.source_count,
                source_geometry=value.source_geometry, output_geometry=value.output_geometry,
                clock=None if value.clock is None else json.loads(value.clock.evidence_json),
                trace=json.loads(value.trace_json))


def validate_map(value):
    if not isinstance(value, FrameMap) or value._owner is not _OWNER:
        raise ValueError("Connect an actual Source Conform map, not a JSON declaration")
    if hashlib.sha256(canonical(_map_payload(value)).encode()).hexdigest() != value.map_sha:
        raise ValueError("Source frame-map integrity changed")
    if identity(value._source) != value.source_sha or identity(value._output) != value.output_sha:
        raise ValueError("Source or conformed RGB changed after preparation")
    if audio_identity(value._audio) != value.output_audio_identity:
        raise ValueError("Conformed PCM changed after preparation")
    return value


def decode_clock(video, interrupt=lambda: None):
    """Record PTS at the SAME native decode that produces the IMAGE/AUDIO.

    Native trim/crop/rotation/float conversion/audio selection remain Core-owned.
    Unknown providers run normally but get no certified CFR/PTS lineage.
    """
    import av
    from comfy_api.latest import InputImpl
    from .modular_sampling.video_io import _CheckedPacket, _CheckedContainer

    interrupt()
    observed = []
    audio_observed = []
    decoded_bytes = [0, 0]
    native = isinstance(video, InputImpl.VideoFromFile)
    origin = Fraction(0)
    if native:
        source = video.get_stream_source()
        if hasattr(source, "seek"):
            source.seek(0)
        with av.open(source, mode="r") as container:
            stream = video._get_first_video_stream(container)
            start, duration = video.get_active_trim_window()
            origin = Fraction(str(start))
            begin = int(start / stream.time_base)
            end = int((start + duration) / stream.time_base)
            for item in container.streams:
                if item.type in ("video", "audio") and item.codec_context is not None:
                    item.codec_context.thread_count = 1

            class Packet(_CheckedPacket):
                def decode(self):
                    for frame in super().decode():
                        if self.packet.stream.type == "video" and frame.pts is not None:
                            if frame.pts >= begin and (not duration or frame.pts < end):
                                if len(observed) >= MAX_FRAMES:
                                    raise ValueError("Source PTS observation exceeds 4096 frames")
                                decoded_bytes[0] += frame.width * frame.height * 3 * 4
                                if decoded_bytes[0] > MAX_TENSOR_BYTES:
                                    raise ValueError("Source decode exceeds conservative 2GiB RGB budget")
                                observed.append(Fraction(frame.pts) * frame.time_base)
                        elif self.packet.stream.type == "audio" and frame.pts is not None:
                            if len(audio_observed) >= MAX_FRAMES * 16:
                                raise ValueError("Source audio PTS observation exceeds bounded budget")
                            decoded_bytes[1] += frame.samples * len(frame.layout.channels) * 4
                            if decoded_bytes[1] > MAX_TENSOR_BYTES:
                                raise ValueError("Source decode exceeds conservative 2GiB PCM budget")
                            audio_observed.append((str(Fraction(frame.pts) * frame.time_base),
                                                   frame.samples, frame.sample_rate))
                        yield frame

            class Container(_CheckedContainer):
                def demux(self, *streams):
                    for packet in self.container.demux(*streams):
                        interrupt()
                        yield Packet(packet, interrupt)

            components = video.get_components_internal(Container(container, interrupt))
    else:
        components = video.get_components()
    interrupt()
    frames_sha = identity(components.images)
    audio_sha = audio_identity(components.audio)
    rate = Fraction(components.frame_rate)
    cfr = (native and rate > 0 and len(observed) == len(components.images)
           and len(observed) >= 2
           and all(right - left == 1 / rate for left, right in zip(observed, observed[1:])))
    status = "observed_CFR" if cfr else "clock_unverified"
    evidence = dict(schema="h3_source_clock_v1", status=status, frames_sha=frames_sha,
        audio_identity=audio_sha, actual_frame_rate=str(rate),
        pts_seconds=[str(value) for value in observed], active_trim_origin=str(origin),
        first_video_pts=None if not observed else str(observed[0]),
        audio_packets_observed=audio_observed, decoder="native_Core_serial" if native else "external_unverified",
        video_audio_origin_equal=(bool(observed) and observed[0] == origin),
        audio_normalization="native_Core_get_components_internal",
        audio_pts_alignment_certified=False, streaming=False, sampler_executed=False)
    clock = MediaClock(frames_sha, audio_sha, rate, tuple(observed), origin, status, canonical(evidence), _OWNER)
    return components, clock, evidence


def conform(frames, source_fps, width, height, length, start_seconds=0.,
            short_video_policy="strict", short_audio_policy="pad_silence", source_audio=None, clock=None):
    if not math.isfinite(float(source_fps)) or source_fps <= 0 or not math.isfinite(float(start_seconds)):
        raise ValueError("Source fps and start must be finite; fps must be positive")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Source Conform requires RGB IMAGE [N,H,W,3]")
    from .core import align_frame_count
    if align_frame_count(length) > MAX_FRAMES:
        raise ValueError("Conform map exceeds explicit 4096-frame budget")
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Conform width/height must be positive multiples of32")
    if align_frame_count(length) * width * height * 3 * frames.element_size() > MAX_TENSOR_BYTES:
        raise ValueError("Conform output would exceed the explicit 2GiB tensor budget")
    source_sha = identity(frames)
    source_audio_sha = audio_identity(source_audio)
    if clock is not None:
        if not isinstance(clock, MediaClock) or clock._owner is not _OWNER:
            raise ValueError("Connect the actual Source Clock object, not a JSON declaration")
        if (clock.frames_sha != source_sha or clock.audio_identity != source_audio_sha
                or not math.isclose(float(clock.frame_rate), float(source_fps), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Source clock belongs to different RGB, PCM, or fps")
        evidence = json.loads(clock.evidence_json)
        if (evidence["frames_sha"] != clock.frames_sha
                or evidence["actual_frame_rate"] != str(clock.frame_rate)
                or evidence["pts_seconds"] != [str(value) for value in clock.pts_seconds]
                or evidence["status"] != clock.status
                or evidence["active_trim_origin"] != str(clock.active_origin)):
            raise ValueError("Source clock observation record changed")
    selected, audio, count, duration, old_report, trace = prepare_source_media_window(
        frames, source_fps, width, height, length, start_seconds, short_video_policy,
        short_audio_policy, source_audio, return_frame_map=True)
    trace["source_audio_identity"] = source_audio_sha
    trace["actual_selected_pts"] = (None if clock is None or clock.status != "observed_CFR" else
                                    [str(clock.pts_seconds[index]) for index in trace["source_indices"]])
    values = dict(source_sha=source_sha, output_sha=identity(selected), output_audio_identity=audio_identity(audio),
        indices=trace["source_indices"], target_times=trace["target_times"], source_count=len(frames),
        source_geometry=(int(frames.shape[2]), int(frames.shape[1])), output_geometry=(width, height),
        clock=clock, trace_json=canonical(trace), map_sha="", _source=frames, _output=selected, _audio=audio, _owner=_OWNER)
    draft = FrameMap(**values)
    values["map_sha"] = hashlib.sha256(canonical(_map_payload(draft)).encode()).hexdigest()
    result = FrameMap(**values)
    report = _map_payload(result)
    report.update(map_sha=result.map_sha, status="source_prepared_once",
                  clock_status="declared_fps_unverified" if clock is None else clock.status,
                  legacy_report=json.loads(old_report), same_source_preparation=True,
                  audio_pts_alignment_certified=False, quality_accepted=False)
    return selected, audio, count, duration, result, report


def derive_control(source_frames, kind="grayscale", mask=None):
    """Owned preprocessing BEFORE selection, binding actual input/output bytes.

    Manual MASK association is explicitly a user's annotation, not inferred origin.
    More preprocessors must call this receipt factory around their actual execution;
    passing opaque tensors to map_control never becomes verified by dimensions alone.
    """
    if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
        raise ValueError("Control derivation needs source RGB IMAGE")
    source_sha = identity(source_frames)
    if kind == "identity_rgb":
        result = source_frames
        operation = "source_RGB_identity_no_preprocess"
    elif kind == "grayscale":
        result = source_frames.mean(dim=-1, keepdim=True).expand(-1, -1, -1, 3)
        operation = "RGB_channel_mean_expand"
    elif kind == "manual_mask":
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(source_frames.shape[:3]):
            raise ValueError("Manual per-frame MASK must match full source N,H,W; no implicit broadcast")
        if not mask.is_floating_point() or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError("MASK must be finite floating values in [0,1]")
        result = mask
        operation = "explicit_manual_annotation_unverified_semantics"
    else:
        raise ValueError("Unknown owned control derivation")
    lineage = DerivedControl(source_sha, identity(result), kind, operation, _OWNER)
    return result, lineage


def map_control(frame_map, control, kind="image", width=0, height=0, lineage=None):
    value = validate_map(frame_map)
    if kind not in ("image", "mask"):
        raise ValueError("Control kind must be image or mask")
    if not isinstance(control, torch.Tensor) or control.ndim != (4 if kind == "image" else 3):
        raise ValueError("Control must be full source IMAGE or MASK, not already re-timed frames")
    if tuple(control.shape[:3]) != (value.source_count, value.source_geometry[1], value.source_geometry[0]):
        raise ValueError("Control source grid differs; do not independently conform before mapping")
    control_sha = identity(control)
    verified = False
    if lineage is not None:
        if not isinstance(lineage, DerivedControl) or lineage._owner is not _OWNER:
            raise ValueError("Connect actual derived control lineage, not a declaration")
        if lineage.source_sha != value.source_sha or lineage.output_sha != control_sha:
            raise ValueError("Control lineage belongs to different or modified content")
        if (kind == "mask") != (lineage.kind == "manual_mask"):
            raise ValueError("Control lineage type differs from connected tensor")
        verified = True
    width = value.output_geometry[0] if width == 0 else int(width)
    height = value.output_geometry[1] if height == 0 else int(height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Control stage width/height must be positive multiples of32")
    channels = 1 if kind == "mask" else 3
    if len(value.indices) * width * height * channels * control.element_size() > MAX_TENSOR_BYTES:
        raise ValueError("Mapped control would exceed the explicit 2GiB tensor budget")
    selected = control.index_select(0, torch.tensor(value.indices, device=control.device))
    if kind == "mask":
        if not torch.isfinite(selected).all() or selected.min() < 0 or selected.max() > 1:
            raise ValueError("MASK must be finite within[0,1]")
        selected = comfy.utils.common_upscale(selected[:, None], width, height, "nearest-exact", "disabled")[:, 0]
        operation = "nearest-exact:crop_disabled"
    else:
        if control.shape[-1] != 3:
            raise ValueError("Control IMAGE needs three channels")
        selected = resize_image(selected, width, height, "disabled")
        operation = "lanczos:crop_disabled"
    report = dict(schema="h3_control_map_v1", map_sha=value.map_sha, source_sha=value.source_sha,
        control_sha=control_sha, output_sha=identity(selected), frame_indices=value.indices,
        stage_geometry=(width, height), kind=kind, spatial_operation=operation,
        lineage_status="owned_preprocess_verified" if verified else "external_origin_unverified",
        annotation_semantics_verified=False,
        clock_status="declared_fps_unverified" if value.clock is None else value.clock.status,
        controls_temporal_map_exact=True, rgb_pixels_equivalent=False, sampler_executed=False)
    return selected, report
