from __future__ import annotations

from collections import deque
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np

import folder_paths

from .studio_advanced import validate_timeline


REEL_PLAN_SCHEMA = "t8.minimax_h3.reel_delivery_plan.v1"
REEL_DELIVERY_SCHEMA = "t8.minimax_h3.reel_delivery.v1"
AUDIO_ROLES = {"dialogue", "music", "ambience", "sfx"}
PEAK_POLICIES = ("block_if_clipping", "normalize_peak", "allow_clipping")
FPS = 24
MAX_CLIPS = 512
MAX_AUDIO_EVENTS = 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: Any, name: str) -> str:
    import re

    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", text):
        raise ValueError(f"{name} contains unsupported characters")
    return text


def _allowed_roots() -> list[Path]:
    roots = []
    for getter in (
        folder_paths.get_input_directory,
        folder_paths.get_output_directory,
        folder_paths.get_temp_directory,
    ):
        try:
            roots.append(Path(getter()).resolve())
        except Exception:
            continue
    return roots


def _resolve_media(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("media path cannot be empty")
    prefixes = {
        "input:": folder_paths.get_input_directory,
        "output:": folder_paths.get_output_directory,
        "temp:": folder_paths.get_temp_directory,
    }
    for prefix, getter in prefixes.items():
        if text.lower().startswith(prefix):
            path = (Path(getter()).resolve() / text[len(prefix) :].lstrip("/\\")).resolve()
            break
    else:
        path = Path(text).resolve()
    roots = _allowed_roots()
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("media path must stay inside ComfyUI input/output/temp")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"media file is missing or symbolic: {path}")
    return path


def _probe_media(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path), mode="r") as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        result: dict[str, Any] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "has_video": video is not None,
            "has_audio": audio is not None,
        }
        if video is not None:
            rate = video.average_rate or video.guessed_rate
            if rate is None:
                raise ValueError(f"video frame rate is unavailable: {path}")
            fps = float(rate)
            if abs(fps - FPS) > 1e-6:
                raise ValueError(f"Reel Delivery currently requires exact 24fps input: {path}")
            frame_count = int(video.frames or 0)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(video))
            result.update(
                {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": int(video.width),
                    "height": int(video.height),
                }
            )
        if audio is not None:
            duration = None
            if audio.duration is not None and audio.time_base is not None:
                duration = float(audio.duration * audio.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            result.update(
                {
                    "audio_duration_seconds": duration,
                    "audio_sample_rate": int(audio.rate or 0),
                    "audio_channels": int(audio.channels or 0),
                }
            )
        return result


def _gain(value: Any, name: str) -> float:
    gain = float(value)
    if not math.isfinite(gain) or not -60 <= gain <= 24:
        raise ValueError(f"{name} must be between -60 and 24 dB")
    return gain


def _time(value: Any, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _event_plan(
    raw: Mapping[str, Any],
    *,
    lane_id: str,
    role: str,
    event_index: int,
    total_duration: float,
    sample_rate: int,
) -> dict[str, Any]:
    path = _resolve_media(raw.get("path"))
    probe = _probe_media(path)
    if not probe["has_audio"]:
        raise ValueError(f"audio event has no audio stream: {path}")
    start = _time(raw.get("start_seconds", 0.0), "start_seconds")
    trim_in = _time(raw.get("trim_in_seconds", 0.0), "trim_in_seconds")
    source_duration = probe.get("audio_duration_seconds")
    trim_out_raw = raw.get("trim_out_seconds")
    if trim_out_raw is None:
        if source_duration is None:
            raise ValueError("trim_out_seconds is required when source duration is unavailable")
        trim_out = float(source_duration)
    else:
        trim_out = _time(trim_out_raw, "trim_out_seconds")
    if trim_out <= trim_in:
        raise ValueError("audio trim_out_seconds must be greater than trim_in_seconds")
    if source_duration is not None and trim_out > source_duration + 0.05:
        raise ValueError("audio trim exceeds source duration")
    duration = trim_out - trim_in
    if start + duration > total_duration + 1e-6:
        raise ValueError("audio event extends beyond the reel duration")
    fade_in = _time(raw.get("fade_in_seconds", 0.0), "fade_in_seconds")
    fade_out = _time(raw.get("fade_out_seconds", 0.0), "fade_out_seconds")
    if fade_in + fade_out > duration + 1e-9:
        raise ValueError("audio event fades exceed event duration")
    return {
        "id": _safe_id(raw.get("id", f"{lane_id}_{event_index:03d}"), "audio event id"),
        "lane_id": lane_id,
        "role": role,
        "path": str(path),
        "source_sha256": probe["sha256"],
        "start_seconds": start,
        "start_sample": round(start * sample_rate),
        "trim_in_seconds": trim_in,
        "trim_out_seconds": trim_out,
        "duration_seconds": duration,
        "gain_db": _gain(raw.get("gain_db", 0.0), "gain_db"),
        "fade_in_samples": round(fade_in * sample_rate),
        "fade_out_samples": round(fade_out * sample_rate),
        "source_probe": probe,
    }


def build_reel_delivery_plan(
    project_id: str,
    reel_json: str,
    sample_rate: int,
    maximum_transition_seconds: float,
    maximum_transition_buffer_mib: float,
    timeline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_id = _safe_id(project_id, "project_id")
    if int(sample_rate) not in {32000, 44100, 48000}:
        raise ValueError("sample_rate must be 32000, 44100 or 48000")
    max_transition = _time(maximum_transition_seconds, "maximum_transition_seconds")
    if max_transition > 2.0:
        raise ValueError("maximum_transition_seconds cannot exceed 2 seconds")
    max_buffer = float(maximum_transition_buffer_mib)
    if not math.isfinite(max_buffer) or not 16 <= max_buffer <= 4096:
        raise ValueError("maximum_transition_buffer_mib must be between 16 and 4096")
    try:
        payload = json.loads(str(reel_json or ""))
    except json.JSONDecodeError as error:
        raise ValueError(f"reel_json is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("reel_json root must be an object")
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("reel_json clips must be a non-empty list")
    if len(raw_clips) > MAX_CLIPS:
        raise ValueError(f"Reel Delivery supports at most {MAX_CLIPS} clips")

    validated_timeline = validate_timeline(timeline) if timeline is not None else None
    clips: list[dict[str, Any]] = []
    width = height = None
    cumulative_frame = 0
    previous_transition = 0
    warnings = []
    for index, raw in enumerate(raw_clips):
        if not isinstance(raw, Mapping):
            raise ValueError(f"clip {index} must be an object")
        path = _resolve_media(raw.get("path"))
        probe = _probe_media(path)
        if not probe["has_video"]:
            raise ValueError(f"clip has no video stream: {path}")
        if width is None:
            width, height = probe["width"], probe["height"]
        elif (probe["width"], probe["height"]) != (width, height):
            raise ValueError("all reel clips must have the same dimensions; implicit resize is disabled")
        trim_in = _time(raw.get("trim_in_seconds", 0.0), "trim_in_seconds")
        source_duration = probe["frame_count"] / FPS
        trim_out = (
            source_duration
            if raw.get("trim_out_seconds") is None
            else _time(raw["trim_out_seconds"], "trim_out_seconds")
        )
        start_frame = round(trim_in * FPS)
        end_frame = round(trim_out * FPS)
        if start_frame < 0 or end_frame <= start_frame or end_frame > probe["frame_count"]:
            raise ValueError(f"clip {index} trim is outside exact source frame boundaries")
        selected_frames = end_frame - start_frame
        transition_seconds = _time(
            raw.get("crossfade_to_next_seconds", 0.0),
            "crossfade_to_next_seconds",
        )
        if transition_seconds > max_transition + 1e-9:
            raise ValueError("clip crossfade exceeds maximum_transition_seconds")
        transition_frames = round(transition_seconds * FPS)
        if index + 1 == len(raw_clips) and transition_frames:
            raise ValueError("the final clip cannot crossfade to a missing next clip")
        if transition_frames * 2 >= selected_frames:
            raise ValueError("crossfade must be shorter than half of its source clip")
        timeline_start = cumulative_frame - previous_transition
        timeline_end = timeline_start + selected_frames
        item = {
            "index": index,
            "id": _safe_id(raw.get("id", f"clip_{index:03d}"), "clip id"),
            "path": str(path),
            "source_sha256": probe["sha256"],
            "source_probe": probe,
            "trim_start_frame": start_frame,
            "trim_end_frame": end_frame,
            "trim_in_seconds": start_frame / FPS,
            "trim_out_seconds": end_frame / FPS,
            "selected_frames": selected_frames,
            "timeline_start_frame": timeline_start,
            "timeline_end_frame_before_out_transition": timeline_end,
            "transition_in_frames": previous_transition,
            "transition_out_frames": transition_frames,
            "include_source_audio": bool(raw.get("include_source_audio", True)),
            "source_audio_gain_db": _gain(
                raw.get("source_audio_gain_db", 0.0),
                "source_audio_gain_db",
            ),
        }
        if item["include_source_audio"] and not probe["has_audio"]:
            item["include_source_audio"] = False
            warnings.append(f"clip {item['id']} has no source audio; source lane disabled")
        clips.append(item)
        cumulative_frame = timeline_end
        previous_transition = transition_frames
    total_frames = sum(item["selected_frames"] for item in clips) - sum(
        item["transition_out_frames"] for item in clips
    )
    total_duration = total_frames / FPS
    total_samples = round(total_duration * int(sample_rate))
    for index in range(len(clips) - 1):
        overlap = clips[index]["transition_out_frames"]
        if overlap and overlap * 2 >= clips[index + 1]["selected_frames"]:
            raise ValueError("crossfade must be shorter than half of the next clip")
    transition_buffer_mib = (
        max((item["transition_out_frames"] for item in clips), default=0)
        * int(width)
        * int(height)
        * 3
        / (1024**2)
    )
    if transition_buffer_mib > max_buffer:
        raise ValueError(
            f"visual crossfade buffer estimate {transition_buffer_mib:.1f}MiB exceeds gate "
            f"{max_buffer:.1f}MiB"
        )

    events: list[dict[str, Any]] = []
    for clip in clips:
        if not clip["include_source_audio"]:
            continue
        duration = clip["selected_frames"] / FPS
        events.append(
            {
                "id": f"source_{clip['id']}",
                "lane_id": "source",
                "role": "dialogue",
                "path": clip["path"],
                "source_sha256": clip["source_sha256"],
                "start_seconds": clip["timeline_start_frame"] / FPS,
                "start_sample": round(clip["timeline_start_frame"] / FPS * sample_rate),
                "trim_in_seconds": clip["trim_in_seconds"],
                "trim_out_seconds": clip["trim_out_seconds"],
                "duration_seconds": duration,
                "gain_db": clip["source_audio_gain_db"],
                "fade_in_samples": round(clip["transition_in_frames"] / FPS * sample_rate),
                "fade_out_samples": round(clip["transition_out_frames"] / FPS * sample_rate),
                "source_probe": clip["source_probe"],
            }
        )

    raw_lanes = payload.get("audio_lanes", [])
    if not isinstance(raw_lanes, list):
        raise ValueError("audio_lanes must be a list")
    lane_ids = set()
    for lane_index, lane in enumerate(raw_lanes):
        if not isinstance(lane, Mapping):
            raise ValueError(f"audio lane {lane_index} must be an object")
        lane_id = _safe_id(lane.get("id", f"lane_{lane_index:03d}"), "audio lane id")
        if lane_id in lane_ids or lane_id == "source":
            raise ValueError(f"duplicate or reserved audio lane id: {lane_id}")
        lane_ids.add(lane_id)
        role = str(lane.get("role", "ambience")).strip().lower()
        if role not in AUDIO_ROLES:
            raise ValueError(f"audio lane {lane_id} has unsupported role")
        lane_events = lane.get("events", [])
        if not isinstance(lane_events, list):
            raise ValueError(f"audio lane {lane_id} events must be a list")
        for event_index, raw_event in enumerate(lane_events):
            if not isinstance(raw_event, Mapping):
                raise ValueError(f"audio event {lane_id}/{event_index} must be an object")
            events.append(
                _event_plan(
                    raw_event,
                    lane_id=lane_id,
                    role=role,
                    event_index=event_index,
                    total_duration=total_duration,
                    sample_rate=int(sample_rate),
                )
            )
    if len(events) > MAX_AUDIO_EVENTS:
        raise ValueError(f"Reel Delivery supports at most {MAX_AUDIO_EVENTS} audio events")
    if validated_timeline is not None:
        if len(clips) != validated_timeline["shot_count"]:
            raise ValueError("connected Studio Timeline shot count does not match reel clips")
        if total_frames != validated_timeline["total_frames"]:
            warnings.append(
                "reel trims/transitions change the Studio Timeline total frame count; "
                "delivery boundaries follow the explicit reel plan"
            )

    plan = {
        "schema": REEL_PLAN_SCHEMA,
        "project_id": project_id,
        "clips": clips,
        "audio_events": events,
        "clip_count": len(clips),
        "audio_event_count": len(events),
        "fps": FPS,
        "sample_rate": int(sample_rate),
        "width": int(width),
        "height": int(height),
        "total_frames": total_frames,
        "total_samples": total_samples,
        "total_duration_seconds": total_duration,
        "transition_buffer_estimate_mib": transition_buffer_mib,
        "maximum_transition_buffer_mib": max_buffer,
        "timeline_hash": (
            validated_timeline["timeline_hash"] if validated_timeline is not None else None
        ),
        "warnings": warnings,
        "streaming_memory_contract": (
            "visual encoding retains at most one transition tail plus one current frame; "
            "audio mixing and final mux run in bounded FFmpeg streams"
        ),
    }
    plan["plan_hash"] = _hash_json(plan)
    return plan


def validate_reel_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != REEL_PLAN_SCHEMA:
        raise ValueError("reel_plan must be a T8 Reel Delivery Plan")
    result = dict(value)
    expected = result.pop("plan_hash", None)
    if expected != _hash_json(result):
        raise ValueError("reel_plan fingerprint is invalid")
    result["plan_hash"] = expected
    for item in [*result["clips"], *result["audio_events"]]:
        path = _resolve_media(item["path"])
        if _sha256_file(path) != item["source_sha256"]:
            raise ValueError(f"reel source changed after planning: {path}")
    return result

def _reel_root(project_id: str) -> Path:
    root = (
        Path(folder_paths.get_output_directory()).resolve()
        / "minimax_h3_t8_reel_delivery"
        / _safe_id(project_id, "project_id")
    ).resolve()
    output = Path(folder_paths.get_output_directory()).resolve()
    if output not in root.parents:
        raise ValueError("reel delivery root escaped the ComfyUI output folder")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _interruption_check() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _selected_frames(clip: Mapping[str, Any]):
    import av

    with av.open(clip["path"], mode="r") as container:
        stream = container.streams.video[0]
        start = int(clip["trim_start_frame"])
        end = int(clip["trim_end_frame"])
        for index, frame in enumerate(container.decode(stream)):
            if index < start:
                continue
            if index >= end:
                break
            _interruption_check()
            yield frame.to_ndarray(format="rgb24")


def _compose_video(plan: Mapping[str, Any], path: Path, crf: int) -> dict[str, Any]:
    import av

    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".mp4.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    encoded = 0
    peak_tail_frames = 0
    try:
        with av.open(str(temporary), mode="w", format="mp4") as output:
            stream = output.add_stream("libx264", rate=Fraction(FPS, 1))
            stream.width = int(plan["width"])
            stream.height = int(plan["height"])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(int(crf)), "preset": "medium"}
            previous_tail: list[np.ndarray] = []
            for clip_index, clip in enumerate(plan["clips"]):
                source = iter(_selected_frames(clip))
                transition_in = int(clip["transition_in_frames"])
                if transition_in:
                    heads = []
                    for _ in range(transition_in):
                        try:
                            heads.append(next(source))
                        except StopIteration as error:
                            raise ValueError("clip ended during the declared visual crossfade") from error
                    if len(previous_tail) != transition_in:
                        raise RuntimeError("visual transition tail contract is inconsistent")
                    for index, (outgoing, incoming) in enumerate(zip(previous_tail, heads)):
                        weight = (index + 1) / (transition_in + 1)
                        blended = np.rint(
                            outgoing.astype(np.float32) * (1.0 - weight)
                            + incoming.astype(np.float32) * weight
                        ).clip(0, 255).astype(np.uint8)
                        output.mux(
                            stream.encode(av.VideoFrame.from_ndarray(blended, format="rgb24"))
                        )
                        encoded += 1
                elif previous_tail:
                    for array in previous_tail:
                        output.mux(
                            stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24"))
                        )
                        encoded += 1
                tail_count = int(clip["transition_out_frames"])
                pending: deque[np.ndarray] = deque()
                for array in source:
                    pending.append(array)
                    if len(pending) > tail_count:
                        output.mux(
                            stream.encode(
                                av.VideoFrame.from_ndarray(pending.popleft(), format="rgb24")
                            )
                        )
                        encoded += 1
                previous_tail = list(pending)
                peak_tail_frames = max(peak_tail_frames, len(previous_tail))
                if clip_index + 1 == len(plan["clips"]):
                    for array in previous_tail:
                        output.mux(
                            stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24"))
                        )
                        encoded += 1
                    previous_tail = []
            output.mux(stream.encode(None))
        if encoded != int(plan["total_frames"]):
            raise RuntimeError(
                f"reel video encoded {encoded} frames, expected {plan['total_frames']}"
            )
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "frame_count": encoded,
        "peak_transition_tail_frames": peak_tail_frames,
        "video_sha256": _sha256_file(path),
    }


def _run_process(args: list[str], log_path: Path) -> None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        try:
            while process.poll() is None:
                _interruption_check()
                time.sleep(0.1)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"FFmpeg failed with exit code {process.returncode}:\n{tail}")


def _audio_filter(plan: Mapping[str, Any]) -> tuple[list[str], str]:
    inputs: list[str] = []
    filters: list[str] = []
    labels = []
    sample_rate = int(plan["sample_rate"])
    for index, event in enumerate(plan["audio_events"]):
        inputs.extend(["-i", event["path"]])
        duration_samples = round(float(event["duration_seconds"]) * sample_rate)
        chain = (
            f"[{index}:a:0]atrim=start={event['trim_in_seconds']:.9f}:"
            f"end={event['trim_out_seconds']:.9f},asetpts=PTS-STARTPTS,"
            f"aresample={sample_rate},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={event['gain_db']:.6f}dB"
        )
        fade_in = int(event["fade_in_samples"])
        fade_out = int(event["fade_out_samples"])
        if fade_in:
            chain += f",afade=t=in:start_sample=0:nb_samples={fade_in}"
        if fade_out:
            chain += (
                f",afade=t=out:start_sample={max(0, duration_samples - fade_out)}:"
                f"nb_samples={fade_out}"
            )
        chain += f",adelay={int(event['start_sample'])}S:all=1[a{index}]"
        filters.append(chain)
        labels.append(f"[a{index}]")
    total = int(plan["total_samples"])
    if labels:
        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
            + f"atrim=end_sample={total},apad=whole_len={total}[aout]"
        )
    return inputs, ";\n".join(filters)


def _compose_audio(plan: Mapping[str, Any], path: Path, log_path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for Reel Delivery audio mixing")
    total_seconds = int(plan["total_samples"]) / int(plan["sample_rate"])
    if not plan["audio_events"]:
        args = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={plan['sample_rate']}:cl=stereo",
            "-t",
            f"{total_seconds:.9f}",
            "-c:a",
            "pcm_f32le",
            str(path),
        ]
        _run_process(args, log_path)
    else:
        inputs, filters = _audio_filter(plan)
        filter_path = path.with_suffix(".filters.txt")
        filter_path.write_text(filters, encoding="utf-8")
        try:
            args = [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-y",
                *inputs,
                "-filter_complex_script",
                str(filter_path),
                "-map",
                "[aout]",
                "-c:a",
                "pcm_f32le",
                str(path),
            ]
            _run_process(args, log_path)
        finally:
            if filter_path.exists():
                filter_path.unlink()
    peak = 0.0
    samples = 0
    import av

    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=int(plan["sample_rate"]),
        )
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                array = converted.to_ndarray()
                peak = max(peak, float(np.max(np.abs(array))) if array.size else 0.0)
                samples += int(array.shape[1])
        for converted in resampler.resample(None):
            array = converted.to_ndarray()
            peak = max(peak, float(np.max(np.abs(array))) if array.size else 0.0)
            samples += int(array.shape[1])
    if samples < int(plan["total_samples"]):
        raise RuntimeError("mixed audio is shorter than the reel sample boundary")
    return {
        "decoded_samples": samples,
        "used_samples": int(plan["total_samples"]),
        "peak_before_delivery": peak,
        "audio_sha256": _sha256_file(path),
    }


def _valid_phase_file(state: Mapping[str, Any], key: str, path: Path) -> bool:
    expected = state.get(key)
    return bool(expected and path.is_file() and _sha256_file(path) == expected)

def compose_reel_delivery(
    reel_plan: Mapping[str, Any],
    confirm_compose: bool,
    filename_prefix: str,
    crf: int,
    peak_policy: str,
) -> tuple[str, dict[str, Any]]:
    plan = validate_reel_plan(reel_plan)
    if not confirm_compose:
        return "", {
            "schema": REEL_DELIVERY_SCHEMA,
            "status": "planned_not_composed",
            "reason": "confirm_compose is false",
            "plan_hash": plan["plan_hash"],
            "source_files_mutated": False,
        }
    if not 0 <= int(crf) <= 51:
        raise ValueError("crf must be between 0 and 51")
    if peak_policy not in PEAK_POLICIES:
        raise ValueError(f"peak_policy must be one of {PEAK_POLICIES}")
    safe_prefix = _safe_id(filename_prefix, "filename_prefix")
    root = _reel_root(plan["project_id"])
    state_path = root / f"{safe_prefix}.state.json"
    video_path = root / f".{safe_prefix}.{plan['plan_hash'][:12]}.video.mp4"
    audio_path = root / f".{safe_prefix}.{plan['plan_hash'][:12]}.audio.wav"
    output_path = root / f"{safe_prefix}.{plan['plan_hash'][:12]}.mp4"
    log_path = root / f"{safe_prefix}.ffmpeg.log"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        state = {}
    if state.get("plan_hash") != plan["plan_hash"]:
        state = {
            "schema": REEL_DELIVERY_SCHEMA,
            "plan_hash": plan["plan_hash"],
            "project_id": plan["project_id"],
            "video_crf": int(crf),
            "phase": "planned",
        }
        _atomic_json(state_path, state)
    elif state.get("video_crf") != int(crf):
        state.pop("video_sha256", None)
        state["video_crf"] = int(crf)
        state["phase"] = "planned"
        _atomic_json(state_path, state)

    if not _valid_phase_file(state, "video_sha256", video_path):
        video_report = _compose_video(plan, video_path, int(crf))
        state.update(video_report)
        state["video_crf"] = int(crf)
        state["phase"] = "video_ready"
        _atomic_json(state_path, state)
    else:
        video_report = {
            "frame_count": plan["total_frames"],
            "video_sha256": state["video_sha256"],
            "resumed": True,
        }

    if not _valid_phase_file(state, "audio_sha256", audio_path):
        audio_report = _compose_audio(plan, audio_path, log_path)
        state.update(audio_report)
        state["phase"] = "audio_ready"
        _atomic_json(state_path, state)
    else:
        audio_report = {
            "used_samples": plan["total_samples"],
            "peak_before_delivery": state.get("peak_before_delivery", 0.0),
            "audio_sha256": state["audio_sha256"],
            "resumed": True,
        }

    peak = float(audio_report["peak_before_delivery"])
    if peak_policy == "block_if_clipping" and peak > 1.0 + 1e-6:
        raise ValueError(
            f"mixed audio peak {peak:.6f} exceeds 1.0; reduce gains or select normalize_peak"
        )
    gain = min(1.0, 0.98 / peak) if peak_policy == "normalize_peak" and peak > 0.98 else 1.0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for Reel Delivery final mux")
    descriptor, name = tempfile.mkstemp(prefix=f".{safe_prefix}.", suffix=".mp4.tmp", dir=root)
    os.close(descriptor)
    temporary = Path(name)
    try:
        args = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            f"volume={gain:.12f}",
            "-frames:v",
            str(plan["total_frames"]),
            "-t",
            f"{plan['total_duration_seconds']:.9f}",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ]
        _run_process(args, log_path)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    state.update(
        {
            "phase": "complete",
            "output_path": str(output_path),
            "output_sha256": _sha256_file(output_path),
            "peak_policy": peak_policy,
            "delivery_gain": gain,
        }
    )
    _atomic_json(state_path, state)
    report = {
        "schema": REEL_DELIVERY_SCHEMA,
        "status": "complete",
        "plan_hash": plan["plan_hash"],
        "output_path": str(output_path),
        "output_sha256": state["output_sha256"],
        "state_path": str(state_path),
        "fps": plan["fps"],
        "frame_count": plan["total_frames"],
        "sample_rate": plan["sample_rate"],
        "audio_samples": plan["total_samples"],
        "duration_seconds": plan["total_duration_seconds"],
        "video": video_report,
        "audio": audio_report,
        "peak_policy": peak_policy,
        "delivery_gain": gain,
        "source_files_mutated": False,
        "resume_contract": "valid completed video/audio phase artifacts are hash-verified and reused",
        "cancel_contract": (
            "ComfyUI interruption terminates FFmpeg, removes the active atomic temp file and "
            "retains only hash-verified completed phase artifacts"
        ),
        "streaming_memory_contract": plan["streaming_memory_contract"],
        "final_container_bit_exact_claim": False,
    }
    return str(output_path), report
