from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import comfy.conds
import comfy.ldm.minimax.model as minimax_model
import node_helpers

from .conditioning import build_conditioning
from .core import resize_image
from .h3_lora_compat_advanced import load_minimax_h3_lora_model
from .long_video_delivery import (
    ISOLATED_VIDEO_ENCODER_POLICY,
    STRICT_AV_DECODE_POLICY,
    _cleanup_temporary,
    _encode_rgb_frames_isolated,
    _run_isolated_ffmpeg,
    _sha256_file,
    _strict_validate_mp4,
    _write_planar_audio_raw,
)


SCHEMA = "t8.minimax_h3.world.v1"
PLAN_TYPE = "T8_H3_WORLD_ACTION_PLAN"
PAYLOAD_FLAG = "minimax_h3_world_schema"
ACTION_SPANS_KEY = "minimax_h3_world_action_spans"
HEAD_END_KEY = "minimax_h3_world_head_end"
PLAN_SHA_KEY = "minimax_h3_world_plan_sha256"
RUNTIME_KEY = "t8_minimax_h3_world_runtime_v1"
PATCH_VERSION = 1

WIDTH = 832
HEIGHT = 480
FRAME_COUNT = 124
LATENT_T = 37
FPS = 24
EXPECTED_LORA_PAIRS = 104
SAFE_OUTPUT_SCHEMA = "t8.minimax_h3.world.safe_output.v1"

KEYS = ("W", "A", "S", "D", "I", "J", "K", "L", "F")
MOTION_ORDER = ("W", "S", "A", "D")
MOTION = {
    "W": "walks forward",
    "S": "walks backward",
    "A": "strafes left",
    "D": "strafes right",
}
PRESETS: dict[str, tuple[str, ...]] = {
    "still": (),
    "forward": ("W",),
    "back": ("S",),
    "strafe-left": ("A",),
    "strafe-right": ("D",),
    "tilt-down": ("I",),
    "tilt-up": ("K",),
    "pan-left": ("J",),
    "pan-right": ("L",),
    "pan-left-fast": ("J", "F"),
    "pan-right-fast": ("L", "F"),
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def _normalize_output_audio(
    audio: Mapping[str, Any], *, expected_samples: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    if not isinstance(audio, Mapping):
        raise ValueError("H3-World safe output requires a ComfyUI AUDIO mapping")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("H3-World AUDIO waveform must be a [batch,channels,samples] tensor")
    if int(waveform.shape[0]) != 1 or int(waveform.shape[1]) not in {1, 2}:
        raise ValueError("H3-World safe output accepts one mono or stereo AUDIO batch")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("H3-World AUDIO sample_rate must be a positive integer")
    samples = waveform[0].detach().float().cpu()
    if not torch.isfinite(samples).all():
        raise ValueError("H3-World AUDIO contains NaN or infinity")
    source_samples = int(samples.shape[-1])
    if source_samples < expected_samples:
        samples = torch.nn.functional.pad(samples, (0, expected_samples - source_samples))
    else:
        samples = samples[:, :expected_samples]
    if int(samples.shape[0]) == 1:
        samples = samples.repeat(2, 1)
    clipped_values = int(torch.count_nonzero(samples.abs() > 1.0).item())
    samples = samples.clamp(-1.0, 1.0).contiguous()
    return samples.numpy(), int(sample_rate), {
        "source_samples": source_samples,
        "encoded_samples": int(expected_samples),
        "source_channels": int(waveform.shape[1]),
        "encoded_channels": 2,
        "clipped_sample_values": clipped_values,
    }


def _mux_h3_world_audio(
    video_path: Path,
    raw_audio_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    duration_seconds: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for H3-World safe MP4 output")
    descriptor, log_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".ffmpeg.log.tmp", dir=output_path.parent
    )
    os.close(descriptor)
    log_path = Path(log_name)
    duration = f"{float(duration_seconds):.12f}"
    try:
        _run_isolated_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-f",
                "f32le",
                "-ar",
                str(int(sample_rate)),
                "-ac",
                "2",
                "-i",
                str(raw_audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-af",
                f"atrim=end={duration}",
                "-t",
                duration,
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(output_path),
            ],
            log_path,
            operation="H3-World exact-duration AAC mux",
        )
    finally:
        _cleanup_temporary(log_path, sys.exc_info()[1])


def save_h3_world_video_safe(
    images: torch.Tensor,
    audio: Mapping[str, Any],
    output_path: str | Path,
    *,
    fps: int = FPS,
    crf: int = 18,
) -> tuple[Path, dict[str, Any]]:
    """Atomically encode the fixed H3-World clip outside ComfyUI's PyAV encoder."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("H3-World images must be an [frames,height,width,channels] tensor")
    frame_count, height, width, channels = (int(value) for value in images.shape)
    if (width, height, frame_count) != (WIDTH, HEIGHT, FRAME_COUNT):
        raise ValueError(
            "H3-World safe output requires exactly "
            f"{WIDTH}x{HEIGHT}x{FRAME_COUNT}; got {width}x{height}x{frame_count}"
        )
    if channels not in {3, 4}:
        raise ValueError("H3-World safe output requires RGB or RGBA IMAGE frames")
    if int(fps) != FPS:
        raise ValueError(f"H3-World safe output requires exactly {FPS} fps")
    if not 0 <= int(crf) <= 51:
        raise ValueError("H3-World safe output crf must stay within 0..51")
    if not torch.isfinite(images).all():
        raise ValueError("H3-World IMAGE frames contain NaN or infinity")

    output = Path(output_path).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("H3-World safe output path must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(audio, Mapping):
        raise ValueError("H3-World safe output requires a ComfyUI AUDIO mapping")
    source_sample_rate = audio.get("sample_rate")
    if (
        isinstance(source_sample_rate, bool)
        or not isinstance(source_sample_rate, int)
        or source_sample_rate <= 0
    ):
        raise ValueError("H3-World AUDIO sample_rate must be a positive integer")
    expected_samples = math.ceil(frame_count * source_sample_rate / int(fps))
    audio_array, sample_rate, audio_report = _normalize_output_audio(
        audio, expected_samples=expected_samples
    )
    token = uuid.uuid4().hex
    video_only = output.with_name(f".{output.stem}.{token}.video-only.mp4")
    raw_audio = output.with_name(f".{output.stem}.{token}.audio.f32le")
    combined = output.with_name(f".{output.stem}.{token}.combined.mp4")

    rgb = images[..., :3].detach()

    def chunks():
        for frame in rgb:
            yield (
                (frame.float().clamp(0.0, 1.0) * 255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )

    try:
        _encode_rgb_frames_isolated(
            video_only,
            chunks,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=int(fps),
            bit_depth=8,
            crf=int(crf),
        )
        _strict_validate_mp4(video_only, require_audio=False)
        _write_planar_audio_raw(raw_audio, audio_array)
        _mux_h3_world_audio(
            video_only,
            raw_audio,
            combined,
            sample_rate=sample_rate,
            duration_seconds=frame_count / int(fps),
        )
        _strict_validate_mp4(combined, require_audio=True)
        os.replace(combined, output)
    finally:
        for temporary in (video_only, raw_audio, combined):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    report = {
        "schema": SAFE_OUTPUT_SCHEMA,
        "status": "ATOMICALLY_PUBLISHED",
        "output_path": str(output),
        "output_sha256": _sha256_file(output).upper(),
        "video": {
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": int(fps),
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "crf": int(crf),
            "encoder_policy": ISOLATED_VIDEO_ENCODER_POLICY,
        },
        "audio": {
            "codec": "aac",
            "sample_rate": sample_rate,
            **audio_report,
        },
        "strict_decode_validated": True,
        "strict_decode_policy": STRICT_AV_DECODE_POLICY,
        "atomic_publish": True,
    }
    return output, report


def _normalize_keys(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("H3-World keys must be a JSON list")
    requested = {str(value).upper() for value in values}
    unknown = sorted(requested - set(KEYS))
    if unknown:
        raise ValueError(f"Unknown H3-World key(s): {', '.join(unknown)}")
    for first, second in (("W", "S"), ("A", "D"), ("J", "L"), ("I", "K")):
        if first in requested and second in requested:
            requested.remove(first)
            requested.remove(second)
    if "F" in requested and not ({"J", "L"} & requested):
        raise ValueError("H3-World key F is only valid together with J or L")
    return tuple(key for key in KEYS if key in requested)


def annotation_for_keys(values: Any) -> str:
    keys = set(_normalize_keys(values))
    words = [MOTION[key] for key in MOTION_ORDER if key in keys]
    motion = " and ".join(words) if words else "stands still"
    camera: list[str] = []
    if "J" in keys:
        camera.append(f"pans left {'sharply' if 'F' in keys else 'slowly'}")
    if "L" in keys:
        camera.append(f"pans right {'sharply' if 'F' in keys else 'slowly'}")
    if "I" in keys:
        camera.append("tilts down")
    if "K" in keys:
        camera.append("tilts up")
    camera_text = " and ".join(camera) if camera else (
        "follows him" if motion != "stands still" else "holds steady"
    )
    return f"the man {motion}, camera {camera_text}"


def _custom_rows(timeline_json: str) -> list[tuple[str, ...]]:
    try:
        segments = json.loads(timeline_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid H3-World custom timeline JSON: {exc}") from exc
    if not isinstance(segments, list) or not segments:
        raise ValueError("Custom H3-World timeline must be a non-empty JSON list")
    rows: list[tuple[str, ...]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"H3-World segment {index} must be a JSON object")
        start = segment.get("start_latent")
        end = segment.get("end_latent")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"H3-World segment {index} start_latent must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"H3-World segment {index} end_latent must be an integer")
        if start != cursor:
            raise ValueError(
                f"H3-World custom timeline must tile 0..{LATENT_T} without gaps; "
                f"segment {index} starts at {start}, expected {cursor}"
            )
        if not start < end <= LATENT_T:
            raise ValueError(f"H3-World segment {index} has invalid range [{start}, {end})")
        keys = _normalize_keys(segment.get("keys", []))
        rows.extend([keys] * (end - start))
        cursor = end
    if cursor != LATENT_T:
        raise ValueError(
            f"H3-World custom timeline ends at {cursor}; it must end at {LATENT_T}"
        )
    return rows


def compile_action_plan(
    action_preset: str = "forward",
    custom_timeline_json: str = "[]",
) -> tuple[dict[str, Any], str, str]:
    if action_preset == "custom":
        rows = _custom_rows(custom_timeline_json)
    else:
        if action_preset not in PRESETS:
            raise ValueError(f"Unknown H3-World action preset: {action_preset}")
        keys = _normalize_keys(PRESETS[action_preset])
        rows = [keys] * LATENT_T
    script = [annotation_for_keys(row) for row in rows]
    payload = {
        "schema": SCHEMA,
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "latent_t": LATENT_T,
        "preset": action_preset,
        "keys": [list(row) for row in rows],
        "script": script,
    }
    payload["payload_sha256"] = _canonical_sha(payload)
    report = {
        "schema": SCHEMA,
        "status": "READY",
        "contract": "832x480, 124 frames, first-frame I2VA, 37 action latents",
        "preset": action_preset,
        "distinct_sentences": len(set(script)),
        "payload_sha256": payload["payload_sha256"],
        "first_annotation": script[0],
    }
    return payload, _json(script), _json(report)


def validate_action_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("Connect a valid MiniMax H3-World Action Timeline plan")
    expected = dict(plan)
    claimed = expected.pop("payload_sha256", None)
    if claimed != _canonical_sha(expected):
        raise ValueError("H3-World action plan hash mismatch")
    if (
        int(plan.get("width", 0)) != WIDTH
        or int(plan.get("height", 0)) != HEIGHT
        or int(plan.get("frame_count", 0)) != FRAME_COUNT
        or int(plan.get("latent_t", 0)) != LATENT_T
    ):
        raise ValueError("H3-World v1 currently requires 832x480, 124 frames and 37 latents")
    keys = plan.get("keys")
    script = plan.get("script")
    if not isinstance(keys, list) or len(keys) != LATENT_T:
        raise ValueError("H3-World action plan must contain 37 key rows")
    normalized = [_normalize_keys(row) for row in keys]
    expected_script = [annotation_for_keys(row) for row in normalized]
    if script != expected_script:
        raise ValueError("H3-World action text does not match its key timeline")
    return plan


def _single_condition_entry(conditioning: Any, label: str) -> tuple[torch.Tensor, dict]:
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError(f"H3-World {label} must encode to exactly one conditioning entry")
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f"H3-World {label} returned an invalid conditioning entry")
    tensor, metadata = entry
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"H3-World {label} must be a [1, tokens, channels] tensor")
    if not isinstance(metadata, dict):
        raise ValueError(f"H3-World {label} conditioning metadata must be a dict")
    return tensor, metadata


def _prepare_first_frame(first_frame: torch.Tensor) -> torch.Tensor:
    """Match upstream H3-World's scale-to-cover then center-crop policy."""
    if not isinstance(first_frame, torch.Tensor) or first_frame.ndim != 4:
        raise ValueError("H3-World I2VA requires one IMAGE batch as first_frame")
    image = first_frame[:1]
    if int(image.shape[1]) == HEIGHT and int(image.shape[2]) == WIDTH:
        return image
    return resize_image(image, WIDTH, HEIGHT, "center")


def build_h3_world_i2va_conditioning(
    clip,
    video_vae,
    audio_vae,
    first_frame: torch.Tensor,
    prompt: str,
    action_plan: dict[str, Any],
) -> tuple[Any, dict, str, str, str]:
    plan = validate_action_plan(action_plan)
    prepared_first_frame = _prepare_first_frame(first_frame)
    base = build_conditioning(
        clip,
        video_vae,
        audio_vae,
        prompt,
        WIDTH,
        HEIGHT,
        FRAME_COUNT,
        "I2VA",
        "native",
        0.35,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
        None,
        None,
        prepared_first_frame,
        None,
        None,
        None,
        None,
        None,
    )
    conditioning, latent, _mux_audio, conditioned_prompt, _media_map, base_report = base
    head, head_metadata = _single_condition_entry(conditioning, "scene prompt")
    if head.shape[-1] != 5120:
        raise ValueError(
            "H3-World requires raw MiniMax H3 Qwen conditioning with 5120 channels; "
            f"received {head.shape[-1]}"
        )

    cache: dict[str, torch.Tensor] = {}
    action_rows: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    cursor = int(head.shape[1])
    for sentence in plan["script"]:
        encoded = cache.get(sentence)
        if encoded is None:
            tokens = clip.tokenize(sentence)
            encoded, _metadata = _single_condition_entry(
                clip.encode_from_tokens_scheduled(tokens), "action sentence"
            )
            if encoded.shape[-1] != head.shape[-1]:
                raise ValueError("H3-World scene and action text encoders do not match")
            cache[sentence] = encoded
        count = int(encoded.shape[1])
        spans.append((cursor, cursor + count))
        cursor += count
        action_rows.append(encoded)

    combined_tensor = torch.cat([head, *action_rows], dim=1)
    head_tags = head_metadata.get("minimax_token_tags")
    if head_tags is None:
        head_tags = torch.ones(head.shape[1], dtype=torch.long)
    head_tags = torch.as_tensor(head_tags, dtype=torch.long).reshape(-1)
    if head_tags.numel() != head.shape[1]:
        raise ValueError("MiniMax H3 scene token tags do not match the encoded prompt length")
    combined_tags = torch.cat(
        (head_tags, torch.ones(cursor - head.shape[1], dtype=torch.long)), dim=0
    )
    combined = [[combined_tensor, dict(head_metadata)]]
    combined = node_helpers.conditioning_set_values(
        combined,
        {
            "minimax_token_tags": combined_tags,
            PAYLOAD_FLAG: PATCH_VERSION,
            ACTION_SPANS_KEY: spans,
            HEAD_END_KEY: int(head.shape[1]),
            PLAN_SHA_KEY: plan["payload_sha256"],
        },
    )
    report = {
        "schema": SCHEMA,
        "status": "READY",
        "route": "first_frame_i2va_native_av",
        "canvas": [WIDTH, HEIGHT],
        "frame_count": FRAME_COUNT,
        "latent_t": LATENT_T,
        "head_tokens": int(head.shape[1]),
        "action_tokens": cursor - int(head.shape[1]),
        "action_spans": len(spans),
        "distinct_action_sentences_encoded": len(cache),
        "plan_sha256": plan["payload_sha256"],
        "base_conditioning": base_report,
    }
    return combined, latent, conditioned_prompt, _json(plan["script"]), _json(report)


def directed_mask_allows(aq: int, akv: int, fq: int, fkv: int) -> bool:
    same_annotation = aq >= 0 and akv >= 0 and aq == akv
    video_reads_own = fq >= 0 and akv >= 0 and fq == akv
    leak_out = akv >= 0 and not same_annotation and not video_reads_own
    leak_in = aq >= 0 and fkv >= 0 and aq != fkv
    return not (leak_out or leak_in)


@dataclass
class _MaskCache:
    key: tuple | None = None
    block_mask: Any = None


class H3WorldFlexRuntime:
    def __init__(self, compile_flex_attention: bool):
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as exc:
            raise RuntimeError(
                "H3-World requires torch.nn.attention.flex_attention"
            ) from exc
        self.compile_flex_attention = bool(compile_flex_attention)
        self._flex_attention = (
            torch.compile(flex_attention, fullgraph=True, dynamic=False)
            if self.compile_flex_attention
            else flex_attention
        )
        self._mask_cache = _MaskCache()

    def block_mask(self, payload: dict, device: torch.device, seq_len: int):
        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except ImportError as exc:
            raise RuntimeError(
                "H3-World requires torch.nn.attention.flex_attention"
            ) from exc
        rows = torch.as_tensor(payload["h3_world_action_text_rows"], dtype=torch.long)
        video_start = int(payload["h3_world_video_start"])
        frame_rows = int(payload["h3_world_frame_rows"])
        latent_t = int(payload["h3_world_latent_t"])
        if rows.shape != (latent_t, 2):
            raise RuntimeError("H3-World action row table changed after conditioning")
        key = (
            str(device),
            int(seq_len),
            video_start,
            frame_rows,
            latent_t,
            tuple(map(tuple, rows.tolist())),
        )
        if self._mask_cache.key == key and self._mask_cache.block_mask is not None:
            return self._mask_cache.block_mask

        annotation = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        frame = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for index, (start, stop) in enumerate(rows.tolist()):
            annotation[int(start) : int(stop)] = index
        video_stop = video_start + latent_t * frame_rows
        if not 0 <= video_start < video_stop <= seq_len:
            raise RuntimeError("H3-World video row span is outside the packed sequence")
        positions = torch.arange(video_start, video_stop, device=device)
        frame[video_start:video_stop] = (
            (positions - video_start) // frame_rows
        ).to(torch.int32)

        def mask_mod(_batch, _head, q, kv, ann=annotation, frm=frame):
            aq, akv, fq, fkv = ann[q], ann[kv], frm[q], frm[kv]
            same_annotation = (aq >= 0) & (akv >= 0) & (aq == akv)
            video_reads_own = (fq >= 0) & (akv >= 0) & (fq == akv)
            leak_out = (akv >= 0) & ~same_annotation & ~video_reads_own
            leak_in = (aq >= 0) & (fkv >= 0) & (aq != fkv)
            return (q == kv) | ~(leak_out | leak_in)

        block_mask = create_block_mask(
            mask_mod, None, None, seq_len, seq_len, device=device, _compile=True
        )
        self._mask_cache = _MaskCache(key=key, block_mask=block_mask)
        return block_mask

    def attention(self, attention, x, rope_freqs, block_mask):
        sequence = x.shape[0]
        q, k, v = attention.qkv_proj(x).split(
            attention.heads * attention.head_dim, dim=-1
        )
        v = v.view(sequence, attention.heads, attention.head_dim)
        if rope_freqs is not None:
            q = q.view(1, sequence, attention.heads, attention.head_dim)
            k = k.view(1, sequence, attention.heads, attention.head_dim)
            qw = minimax_model.comfy.model_management.cast_to(
                attention.q_norm.weight, device=x.device
            )
            kw = minimax_model.comfy.model_management.cast_to(
                attention.k_norm.weight, device=x.device
            )
            rot = rope_freqs.shape[-3] * 2
            if minimax_model.comfy.model_management.in_training:
                q, k = minimax_model.comfy.quant_ops.ck.rms_rope_split_half(
                    q,
                    k,
                    rope_freqs,
                    qw,
                    kw,
                    epsilon=attention.q_norm.eps,
                    rot_dim=rot,
                )
            else:
                minimax_model.comfy.quant_ops.ck.rms_rope_split_half_(
                    q,
                    k,
                    rope_freqs,
                    qw,
                    kw,
                    epsilon=attention.q_norm.eps,
                    rot_dim=rot,
                )
            q, k = q[0], k[0]
        else:
            q = attention.q_norm(q.view(sequence, attention.heads, attention.head_dim))
            k = attention.k_norm(k.view(sequence, attention.heads, attention.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.clone().transpose(0, 1).unsqueeze(0)
        output = self._flex_attention(q, k, v, block_mask=block_mask)
        output = output.transpose(1, 2).reshape(1, sequence, -1).squeeze(0)
        return attention.out_proj(output)


def _validate_action_spans(spans: Any, head_end: int, text_len: int) -> list[tuple[int, int]]:
    if not isinstance(spans, (list, tuple)) or len(spans) != LATENT_T:
        raise RuntimeError("H3-World conditioning must contain 37 action text spans")
    normalized: list[tuple[int, int]] = []
    cursor = head_end
    for index, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise RuntimeError(f"H3-World action span {index} is invalid")
        start, stop = int(span[0]), int(span[1])
        if start != cursor or stop <= start or stop > text_len:
            raise RuntimeError(
                f"H3-World action spans must tile [{head_end}, {text_len}); "
                f"span {index} is [{start}, {stop}), expected start {cursor}"
            )
        normalized.append((start, stop))
        cursor = stop
    if cursor != text_len:
        raise RuntimeError("H3-World action spans do not reach the text end")
    return normalized


def _refine_text_segments(diffusion, cross_attn, spans, head_end, transformer_options):
    raw = cross_attn[0]
    projected = diffusion.condition_proj(raw)
    pieces = [(0, head_end), *spans]
    refined = [
        diffusion.token_refiner(
            projected[start:stop], transformer_options=transformer_options
        )
        for start, stop in pieces
    ]
    return torch.cat(refined, dim=0).unsqueeze(0)


def _repair_layout_for_actions(layout, spans, head_end: int, latent_t: int) -> None:
    text_len = int(layout.signature[0])
    time_grid = minimax_model._video_t_grid(latent_t, 0.0)
    origin = float(text_len) - float(time_grid[-1]) - 1.0
    if origin < head_end:
        raise RuntimeError(
            "H3-World action positions overlap the scene prompt; shorten the prompt "
            "or use shorter action annotations"
        )
    for index, (start, stop) in enumerate(spans):
        layout.position_ids[start:stop, 0] = origin + float(time_grid[index])


def _ensure_patch_compatibility(model) -> None:
    existing = set(getattr(model, "object_patches", {}))
    owned = [
        path
        for path in existing
        if path == "extra_conds"
        or path == "diffusion_model._forward"
        or (path.startswith("diffusion_model.blocks.") and path.endswith(".attn.forward"))
    ]
    if owned:
        raise RuntimeError(
            "H3-World is an isolated EXP route and cannot stack with an existing "
            f"H3 layout/attention patch: {owned[:4]}"
        )
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    patches = options.get("patches", {})
    replacements = options.get("patches_replace", {})
    if patches.get("attn1_patch") or patches.get("attn1_output_patch"):
        raise RuntimeError("H3-World cannot stack with attention hook patches in v1")
    if options.get("optimized_attention_override") is not None:
        raise RuntimeError("H3-World cannot stack with an attention override in v1")
    if replacements.get("dit"):
        raise RuntimeError("H3-World cannot stack with DiT block replacements in v1")


def patch_h3_world_model(model, compile_flex_attention: bool = True):
    if not hasattr(model, "clone") or not hasattr(model, "add_object_patch"):
        raise ValueError("H3-World requires a ComfyUI MODEL")
    diffusion = model.get_model_object("diffusion_model")
    if diffusion.__class__.__name__ != "MiniMaxH3Model":
        raise ValueError("H3-World requires the native ComfyUI MiniMaxH3Model")
    if len(getattr(diffusion, "blocks", ())) != 50:
        raise ValueError("H3-World requires the 50-block MiniMax H3 architecture")
    _ensure_patch_compatibility(model)
    runtime = H3WorldFlexRuntime(compile_flex_attention)
    patched = model.clone()
    base_model = patched.model
    original_extra_conds = patched.get_model_object("extra_conds")
    original_forward = patched.get_model_object("diffusion_model._forward")
    forward_function = getattr(original_forward, "__func__", original_forward)
    required_forward_markers = (
        "layout = payload.get(\"layout\")",
        "self.token_refiner",
        "self.blocks",
        "self.final_layer",
    )
    try:
        forward_source = inspect.getsource(forward_function)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Cannot inspect the active MiniMax H3 forward contract") from exc
    missing = [marker for marker in required_forward_markers if marker not in forward_source]
    if missing:
        raise RuntimeError(
            "The active ComfyUI MiniMax H3 forward contract is not compatible with "
            f"H3-World; missing markers: {missing}"
        )

    def patched_extra_conds(_self, **kwargs):
        if int(kwargs.get(PAYLOAD_FLAG, 0) or 0) != PATCH_VERSION:
            return original_extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn")
        latent_shapes = kwargs.get("latent_shapes")
        if not isinstance(cross_attn, torch.Tensor) or cross_attn.ndim != 3:
            raise RuntimeError("H3-World conditioning is missing raw Qwen cross-attention")
        if latent_shapes is None or len(latent_shapes) < 2:
            raise RuntimeError("H3-World conditioning is missing packed AV latent shapes")
        if int(cross_attn.shape[-1]) != 5120:
            raise RuntimeError("H3-World needs raw 5120-channel Qwen states")
        text_len = int(cross_attn.shape[1])
        head_end = int(kwargs.get(HEAD_END_KEY, 0) or 0)
        spans = _validate_action_spans(kwargs.get(ACTION_SPANS_KEY), head_end, text_len)
        device = kwargs["device"]
        raw = cross_attn.to(device=device, dtype=_self.get_dtype_inference())
        refined = _refine_text_segments(
            _self.diffusion_model,
            raw,
            spans,
            head_end,
            {},
        )

        core_kwargs = dict(kwargs)
        core_kwargs["cross_attn"] = None
        out = original_extra_conds(**core_kwargs)
        out["c_crossattn"] = comfy.conds.CONDRegular(refined)
        payload_cond = out.get("minimax_payload")
        payload = dict(getattr(payload_cond, "cond", {}) or {})
        vs = latent_shapes[0]
        layout = minimax_model.PackedLayout(
            text_len,
            int(vs[2]),
            (int(vs[3]) + 1) // 2 * 2,
            (int(vs[4]) + 1) // 2 * 2,
            int(latent_shapes[1][-1]),
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )
        if int(vs[2]) != LATENT_T:
            raise RuntimeError(
                f"H3-World v1 expects {LATENT_T} video latents, got {int(vs[2])}"
            )
        _repair_layout_for_actions(layout, spans, head_end, LATENT_T)
        video_start, video_stop, _kind = next(
            segment for segment in layout.segments if segment[2] == "video"
        )
        video_rows = video_stop - video_start
        if video_rows % LATENT_T:
            raise RuntimeError("H3-World video rows do not divide into 37 latent frames")
        payload.update(
            {
                "layout": layout,
                PAYLOAD_FLAG: PATCH_VERSION,
                "h3_world_action_text_rows": torch.tensor(spans, dtype=torch.long),
                "h3_world_video_start": int(video_start),
                "h3_world_frame_rows": int(video_rows // LATENT_T),
                "h3_world_latent_t": LATENT_T,
                "h3_world_plan_sha256": kwargs.get(PLAN_SHA_KEY),
                "h3_world_refiner_segments": LATENT_T + 1,
            }
        )
        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        return out

    patched_extra_conds._t8_h3_world_patch_version = PATCH_VERSION
    patched.add_object_patch(
        "extra_conds", types.MethodType(patched_extra_conds, base_model)
    )

    def patched_forward(_self, *args, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        if int(payload.get(PAYLOAD_FLAG, 0) or 0) != PATCH_VERSION:
            return original_forward(*args, **kwargs)
        positional_options = args[3] if len(args) >= 4 else None
        transformer_options = dict(
            positional_options or kwargs.get("transformer_options") or {}
        )
        if transformer_options.get("patches", {}).get("attn1_patch"):
            raise RuntimeError("H3-World refuses runtime attention patches")
        layout = payload.get("layout")
        if layout is None:
            raise RuntimeError("H3-World packed layout is missing")
        device = args[0][0].device if args else kwargs["x"][0].device
        block_mask = runtime.block_mask(payload, device, int(layout.seq_len))
        transformer_options[RUNTIME_KEY] = {
            "runtime": runtime,
            "block_mask": block_mask,
        }
        if len(args) >= 4:
            forwarded_args = list(args)
            forwarded_args[3] = transformer_options
            kwargs.pop("transformer_options", None)
            return original_forward(*forwarded_args, **kwargs)
        kwargs["transformer_options"] = transformer_options
        return original_forward(*args, **kwargs)

    patched_forward._t8_h3_world_patch_version = PATCH_VERSION
    patched.add_object_patch(
        "diffusion_model._forward", types.MethodType(patched_forward, diffusion)
    )

    for index, block in enumerate(diffusion.blocks):
        attention = block.attn
        original_attention = attention.forward

        def h3_world_attention_forward(
            _attention,
            x,
            rope_freqs=None,
            transformer_options=None,
            _original=original_attention,
        ):
            runtime_spec = (transformer_options or {}).get(RUNTIME_KEY)
            if runtime_spec is None:
                return _original(
                    x, rope_freqs=rope_freqs, transformer_options=transformer_options or {}
                )
            return runtime_spec["runtime"].attention(
                _attention, x, rope_freqs, runtime_spec["block_mask"]
            )

        h3_world_attention_forward._t8_h3_world_patch_version = PATCH_VERSION
        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            types.MethodType(h3_world_attention_forward, attention),
        )
    return patched, {
        "patch_version": PATCH_VERSION,
        "main_attention_patches": len(diffusion.blocks),
        "refiner_policy": "scene head plus each action sentence refined independently",
        "attention_policy": "directed FlexAttention action-to-own-latent binding",
        "compiled_flex_attention": bool(compile_flex_attention),
        "model_class": diffusion.__class__.__name__,
    }


def compose_h3_world_model(
    model,
    lora_path: str | Path,
    strength_model: float = 1.0,
    compile_flex_attention: bool = True,
) -> tuple[Any, str]:
    loaded, lora_report_json = load_minimax_h3_lora_model(
        model, lora_path, strength_model
    )
    lora_report = json.loads(lora_report_json)
    applied = int(lora_report.get("applied_patch_count", 0))
    missed = int(lora_report.get("missed_patch_target_count", 0))
    if applied != EXPECTED_LORA_PAIRS or missed:
        raise RuntimeError(
            "H3-World requires all 104 LoRA targets; "
            f"the selected MODEL applied {applied}/{EXPECTED_LORA_PAIRS} with {missed} missed"
        )
    patched, patch_report = patch_h3_world_model(
        loaded, compile_flex_attention=compile_flex_attention
    )
    report = {
        "schema": SCHEMA,
        "status": "READY",
        "lora": lora_report,
        "runtime": patch_report,
        "supported_route": "832x480x124 first-frame I2VA, native AV, CFG 1.0",
        "stacking": "isolated EXP route; no SLA/VSA/Sage/DiT replacement/attention hooks",
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_world_contract", report)
    return patched, _json(report)
