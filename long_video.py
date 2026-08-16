from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unicodedata

import torch

import folder_paths
import node_helpers
from safetensors import safe_open
from safetensors.torch import save_file

from .conditioning import (
    _encode_reference_audio,
    _resize_reference_image,
    build_packed_layout,
    resolve_task_type,
)
from .core import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    MAX_PIXELS,
    MIN_TRAINED_FRAMES,
    adapt_canvas,
    align_frame_count,
    align_frame_count_down,
    empty_av_latent,
    encode_audio_once,
    fit_audio_latent,
    nested_av_parts,
    replace_audio_latent,
    resize_image,
    sorted_autogrow_items,
    sorted_autogrow_values,
)
from .prompt_tags import media_map_json, prepare_prompt


LONG_VIDEO_SCHEMA = 1
LONG_VIDEO_PATCH_VERSION = 1
LONG_VIDEO_CONDITIONING_KEY = "t8_long_video_schema"
MOTION_FRAME_INDEX = "t8_long_video_frame_index"
MOTION_AUDIO_END_FRAME = "t8_long_video_audio_end_frame"
HYBRID_KEYFRAME_SENTINEL = "t8_keyframe_latent"
CONTEXT_TYPE_NAME = "H3_T8_CONTEXT"
CONTEXT_FRAME_STEPS = {5: 2, 22: 7, 39: 12}
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
MAX_CONTEXT_FRAMES = 39
STATE_FOLDER = "minimax_h3_t8_long_video"
FIRST_FRAME_REUSE_POLICIES = {"segment0_only", "persistent_identity_reference"}
PERSISTENT_IDENTITY_STRATEGIES = {"single_reference", "scene_plus_identity"}


def pixel_frames_from_latent_t(latent_t: int) -> int:
    if latent_t < 1:
        raise ValueError("MiniMax H3 video latent T must be positive")
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t))


def step_offsets(latent_t: int) -> list[int]:
    offsets: list[int] = []
    current = 0
    for index in range(latent_t):
        offsets.append(current)
        current += FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
    return offsets


@dataclass(frozen=True)
class LongVideoPlan:
    chain_id: str
    segment_index: int
    requested_new_duration_seconds: float
    requested_frame_count: int
    render_frames: int
    render_duration_seconds: float
    context_frames: int
    context_duration_seconds: float
    trim_start_seconds: float
    final_duration_seconds: float
    final_frame_count: int
    timeline_start_seconds: float
    timeline_end_seconds: float
    minimum_render_frames: int
    is_final_segment: bool
    save_context: bool
    hidden_tail_frames: int

    def report(self) -> str:
        payload = asdict(self)
        payload["trained_range_warning"] = not (
            MIN_TRAINED_FRAMES <= self.render_frames <= 362
        )
        if self.context_frames:
            payload["video_condition_row_ratio"] = (
                CONTEXT_FRAME_STEPS[self.context_frames]
                / ((self.render_frames - 5) // 17 * 5 + 2)
            )
        else:
            payload["video_condition_row_ratio"] = 0.0
        return json.dumps(payload, ensure_ascii=False, indent=2)


def make_long_video_plan(
    chain_id: str,
    segment_index: int,
    new_duration_seconds: float,
    context_frames: int = 22,
    minimum_render_frames: int = MIN_TRAINED_FRAMES,
    timeline_start_seconds: float = -1.0,
    is_final_segment: bool = False,
) -> LongVideoPlan:
    normalized_chain_id = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    if new_duration_seconds <= 0:
        raise ValueError("new_duration_seconds must be positive")
    if int(context_frames) not in CONTEXT_FRAME_STEPS:
        raise ValueError("context_frames must be 5, 22, or 39")
    minimum_render_frames = align_frame_count(max(5, int(minimum_render_frames)))

    active_context = 0 if segment_index == 0 else int(context_frames)
    requested_frames = max(1, round(float(new_duration_seconds) * FPS))
    required_frames = max(minimum_render_frames, requested_frames + active_context)
    render_frames = align_frame_count(required_frames)
    available_after_context = render_frames - active_context
    if requested_frames > available_after_context:
        raise RuntimeError("Long-video planner failed to allocate enough post-context frames")

    if is_final_segment:
        final_frame_count = requested_frames
        hidden_tail_frames = available_after_context - requested_frames
    else:
        # A continuable segment must end at the sampled latent tail. Otherwise
        # the next context would come from frames the user trimmed away.
        final_frame_count = available_after_context
        hidden_tail_frames = 0
    final_duration_seconds = final_frame_count / FPS

    if timeline_start_seconds < 0:
        first_render_frames = align_frame_count(max(minimum_render_frames, requested_frames))
        continuation_render_frames = align_frame_count(
            max(minimum_render_frames, requested_frames + int(context_frames))
        )
        continuation_frames = continuation_render_frames - int(context_frames)
        timeline_start_seconds = (
            0.0
            if segment_index == 0
            else (first_render_frames + (segment_index - 1) * continuation_frames) / FPS
        )
    if timeline_start_seconds < 0:
        raise ValueError("timeline_start_seconds cannot be negative")

    return LongVideoPlan(
        chain_id=normalized_chain_id,
        segment_index=segment_index,
        requested_new_duration_seconds=float(new_duration_seconds),
        requested_frame_count=requested_frames,
        render_frames=render_frames,
        render_duration_seconds=render_frames / FPS,
        context_frames=active_context,
        context_duration_seconds=active_context / FPS,
        trim_start_seconds=active_context / FPS,
        final_duration_seconds=final_duration_seconds,
        final_frame_count=final_frame_count,
        timeline_start_seconds=float(timeline_start_seconds),
        timeline_end_seconds=float(timeline_start_seconds) + final_duration_seconds,
        minimum_render_frames=minimum_render_frames,
        is_final_segment=bool(is_final_segment),
        save_context=not bool(is_final_segment),
        hidden_tail_frames=hidden_tail_frames,
    )


def sanitize_chain_id(chain_id: str) -> str:
    value = unicodedata.normalize("NFKC", str(chain_id or "").strip())
    value = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", "_", value)
    value = value.strip(" .")
    if not value or value in {".", ".."}:
        raise ValueError("chain_id must contain at least one safe character")
    if len(value) > 64:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        value = f"{value[:48]}_{digest}"
    return value


def context_state_path(chain_id: str, source_segment_index: int) -> Path:
    safe_chain = sanitize_chain_id(chain_id)
    source_segment_index = int(source_segment_index)
    if source_segment_index < 0:
        raise ValueError("source_segment_index cannot be negative")
    output_root = Path(folder_paths.get_output_directory()).resolve()
    folder = (output_root / STATE_FOLDER / safe_chain).resolve()
    if output_root not in folder.parents:
        raise ValueError("Resolved long-video state directory escaped the ComfyUI output folder")
    return folder / f"segment_{source_segment_index:05d}.context.safetensors"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _empty_context(chain_id: str, segment_index: int) -> dict:
    return {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": True,
        "chain_id": sanitize_chain_id(chain_id),
        "target_segment_index": int(segment_index),
    }


def save_context_state(
    av_latent: dict,
    chain_id: str,
    segment_index: int,
    model_id: str = "unknown",
    sampling_summary: str = "unknown",
    save_context: bool = True,
) -> tuple[str, str]:
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    if not save_context:
        report = {
            "schema": LONG_VIDEO_SCHEMA,
            "chain_id": safe_chain,
            "source_segment_index": segment_index,
            "saved": False,
            "reason": "final segment: exact tail trim is not a valid continuation checkpoint",
        }
        return "", json.dumps(report, ensure_ascii=False, indent=2)
    video, audio = nested_av_parts(av_latent)
    total_frames = pixel_frames_from_latent_t(int(video.shape[2]))

    supported = [
        (frames, steps)
        for frames, steps in sorted(CONTEXT_FRAME_STEPS.items(), reverse=True)
        if steps <= int(video.shape[2])
    ]
    if not supported:
        raise ValueError("The sampled video latent is too short to save H3 motion context")
    max_context_frames, video_steps = supported[0]
    audio_steps = min(
        int(audio.shape[-1]),
        round(max_context_frames / FPS * AUDIO_LATENT_FPS),
    )
    if audio_steps < 1:
        raise ValueError("The sampled AV latent has no usable audio tail")

    video_tail = video[:1, :, -video_steps:].detach().cpu().contiguous()
    audio_tail = audio[:1, :, :, -audio_steps:].detach().cpu().contiguous()
    audio_overhang = int(audio.shape[-1]) - FRAME_RESCALE * total_frames
    if not 0.0 <= audio_overhang < 1.0:
        audio_overhang = 0.0

    metadata = {
        "schema": str(LONG_VIDEO_SCHEMA),
        "chain_id": safe_chain,
        "source_segment_index": str(segment_index),
        "model_id": str(model_id or "unknown"),
        "sampling_summary": str(sampling_summary or "unknown"),
        "fps": str(FPS),
        "source_total_frames": str(total_frames),
        "max_context_frames": str(max_context_frames),
        "video_shape": json.dumps(list(video_tail.shape)),
        "audio_shape": json.dumps(list(audio_tail.shape)),
        "video_dtype": str(video_tail.dtype),
        "audio_dtype": str(audio_tail.dtype),
        "audio_overhang": repr(float(audio_overhang)),
        "video_sha256": _tensor_sha256(video_tail),
        "audio_sha256": _tensor_sha256(audio_tail),
    }

    target = context_state_path(safe_chain, segment_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file({"video_tail": video_tail, "audio_tail": audio_tail}, str(temporary), metadata)
        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "schema": LONG_VIDEO_SCHEMA,
        "chain_id": safe_chain,
        "source_segment_index": segment_index,
        "path": str(target),
        "source_total_frames": total_frames,
        "max_context_frames": max_context_frames,
        "video_shape": list(video_tail.shape),
        "audio_shape": list(audio_tail.shape),
        "video_bytes": video_tail.numel() * video_tail.element_size(),
        "audio_bytes": audio_tail.numel() * audio_tail.element_size(),
        "atomic_replace": True,
    }
    return str(target), json.dumps(report, ensure_ascii=False, indent=2)


def load_context_state(chain_id: str, segment_index: int) -> tuple[dict, bool, str]:
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    if segment_index == 0:
        context = _empty_context(safe_chain, segment_index)
        return context, False, json.dumps(context, ensure_ascii=False, indent=2)

    source_index = segment_index - 1
    path = context_state_path(safe_chain, source_index)
    if not path.is_file():
        raise FileNotFoundError(
            f"Long-video segment {segment_index} needs context from segment {source_index}, "
            f"but {path} does not exist"
        )

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
        if keys != {"video_tail", "audio_tail"}:
            raise ValueError(f"Invalid H3 T8 context tensor keys in {path}: {sorted(keys)}")
        video_tail = handle.get_tensor("video_tail")
        audio_tail = handle.get_tensor("audio_tail")

    required = {
        "schema", "chain_id", "source_segment_index", "fps", "source_total_frames",
        "max_context_frames", "video_shape", "audio_shape", "video_dtype", "audio_dtype",
        "audio_overhang", "video_sha256", "audio_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"H3 T8 context metadata is incomplete: {', '.join(missing)}")
    if int(metadata["schema"]) != LONG_VIDEO_SCHEMA:
        raise ValueError(
            f"Unsupported H3 T8 context schema {metadata['schema']}; expected {LONG_VIDEO_SCHEMA}"
        )
    if metadata["chain_id"] != safe_chain or int(metadata["source_segment_index"]) != source_index:
        raise ValueError("H3 T8 context chain/segment metadata does not match the requested slot")
    if int(metadata["fps"]) != FPS:
        raise ValueError(f"H3 T8 context fps must be {FPS}")
    if json.loads(metadata["video_shape"]) != list(video_tail.shape):
        raise ValueError("H3 T8 context video shape metadata does not match the tensor")
    if json.loads(metadata["audio_shape"]) != list(audio_tail.shape):
        raise ValueError("H3 T8 context audio shape metadata does not match the tensor")
    if metadata["video_dtype"] != str(video_tail.dtype):
        raise ValueError("H3 T8 context video dtype metadata does not match the tensor")
    if metadata["audio_dtype"] != str(audio_tail.dtype):
        raise ValueError("H3 T8 context audio dtype metadata does not match the tensor")
    max_context_frames = int(metadata["max_context_frames"])
    expected_video_steps = CONTEXT_FRAME_STEPS.get(max_context_frames)
    if expected_video_steps is None or int(video_tail.shape[2]) != expected_video_steps:
        raise ValueError("H3 T8 context video tail is not a supported 5/22/39-frame window")
    if int(metadata["source_total_frames"]) < max_context_frames:
        raise ValueError("H3 T8 context source frame metadata is shorter than its tail")
    if metadata["video_sha256"] != _tensor_sha256(video_tail):
        raise ValueError("H3 T8 context video tensor checksum failed")
    if metadata["audio_sha256"] != _tensor_sha256(audio_tail):
        raise ValueError("H3 T8 context audio tensor checksum failed")

    parsed_metadata = {
        "schema": LONG_VIDEO_SCHEMA,
        "chain_id": safe_chain,
        "source_segment_index": source_index,
        "target_segment_index": segment_index,
        "fps": FPS,
        "source_total_frames": int(metadata["source_total_frames"]),
        "max_context_frames": max_context_frames,
        "audio_overhang": float(metadata["audio_overhang"]),
        "model_id": metadata.get("model_id", "unknown"),
        "sampling_summary": metadata.get("sampling_summary", "unknown"),
        "path": str(path),
    }
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": video_tail,
        "audio_tail": audio_tail,
        "metadata": parsed_metadata,
    }
    report = parsed_metadata | {
        "video_shape": list(video_tail.shape),
        "audio_shape": list(audio_tail.shape),
        "checksums_valid": True,
    }
    return context, True, json.dumps(report, ensure_ascii=False, indent=2)


def context_fingerprint(chain_id: str, segment_index: int) -> str:
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index <= 0:
        return f"{safe_chain}:segment0"
    path = context_state_path(safe_chain, segment_index - 1)
    if not path.is_file():
        return f"{safe_chain}:missing:{segment_index - 1}"
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _validate_context(
    context: dict,
    segment_index: int,
    context_frames: int,
    width: int,
    height: int,
) -> bool:
    if not isinstance(context, dict) or int(context.get("schema", -1)) != LONG_VIDEO_SCHEMA:
        raise ValueError("Connect an H3 T8 Long Video Context Load output")
    if context.get("empty"):
        if int(segment_index) != 0:
            raise ValueError("Only segment 0 may run without a previous H3 T8 context")
        if int(context_frames) != 0:
            raise ValueError("Segment 0 must use context_frames=0 from the Segment Planner")
        return False

    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("H3 T8 context metadata is missing")
    if int(segment_index) <= 0:
        raise ValueError("A loaded previous context cannot be used for segment 0")
    if int(metadata.get("source_segment_index", -2)) != int(segment_index) - 1:
        raise ValueError("H3 T8 context is not from the immediately previous segment")
    if int(context_frames) not in CONTEXT_FRAME_STEPS:
        raise ValueError("Continuation context_frames must be 5, 22, or 39")
    if int(metadata.get("max_context_frames", 0)) < int(context_frames):
        raise ValueError("Saved H3 T8 context does not contain the requested frame window")
    video = context.get("video_tail")
    audio = context.get("audio_tail")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("H3 T8 context video tail must be [B,C,T,H,W]")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
        raise ValueError("H3 T8 context audio tail must be [B,C,stereo,T]")
    if video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("H3 T8 context video tail must use batch 1 and 24 channels")
    if audio.shape[0] != 1 or audio.shape[1:3] != (32, 2):
        raise ValueError("H3 T8 context audio tail must use [1,32,2,T]")
    if tuple(video.shape[-2:]) != (height // 16, width // 16):
        raise ValueError(
            "Long-video continuation requires the same canvas as the previous segment: "
            f"context latent is {video.shape[-1] * 16}x{video.shape[-2] * 16}, "
            f"target is {width}x{height}"
        )
    required_video_steps = CONTEXT_FRAME_STEPS[int(context_frames)]
    if int(video.shape[2]) < required_video_steps:
        raise ValueError("Saved H3 T8 context does not contain enough video latent steps")
    return True


def _motion_context_blocks(
    context: dict,
    context_frames: int,
    continue_audio: bool,
) -> tuple[list[dict], dict | None]:
    latent_steps = CONTEXT_FRAME_STEPS[int(context_frames)]
    video = context["video_tail"][:, :, -latent_steps:].contiguous()
    offsets = step_offsets(latent_steps)
    if pixel_frames_from_latent_t(latent_steps) != int(context_frames):
        raise RuntimeError("H3 motion context grid changed unexpectedly")
    keyframes = [
        {
            "resolved_frame_index": 0,
            MOTION_FRAME_INDEX: offset,
            "latent": video[:, :, index : index + 1],
        }
        for index, offset in enumerate(offsets)
    ]

    audio_ref = None
    if continue_audio:
        audio_steps = round(int(context_frames) / FPS * AUDIO_LATENT_FPS)
        audio = context["audio_tail"]
        if audio_steps > int(audio.shape[-1]):
            raise ValueError("Saved H3 T8 context does not contain enough audio tail")
        overhang = float(context["metadata"].get("audio_overhang", 0.0))
        audio_ref = {
            "kind": "audio",
            "ref_audio_t": audio_steps,
            "audio_latent": audio[:, :, :, -audio_steps:].contiguous(),
            MOTION_AUDIO_END_FRAME: int(context_frames) + overhang / FRAME_RESCALE,
        }
    return keyframes, audio_ref


def _resolve_long_task_type(
    task_type: str,
    context_active: bool,
    first_frame,
    last_frame,
    has_user_refs: bool,
) -> str:
    if not context_active:
        return resolve_task_type(task_type, first_frame, last_frame, has_user_refs)

    requested = (task_type or "auto").lower()
    if requested == "auto":
        if has_user_refs:
            return "hybrid"
        if last_frame is not None:
            return "fl2va"
        return "i2va-motion"
    if requested not in {"t2va", "i2va", "fl2va", "l2va", "ref2va", "hybrid"}:
        raise ValueError(f"Unknown MiniMax H3 task type: {task_type}")
    if requested in {"t2va", "i2va"} and (last_frame is not None or has_user_refs):
        raise ValueError(f"{requested.upper()} continuation cannot include last_frame or references")
    if requested in {"fl2va", "l2va"}:
        if last_frame is None or has_user_refs:
            raise ValueError(f"{requested.upper()} continuation requires last_frame and no references")
    if requested == "ref2va" and (not has_user_refs or last_frame is not None):
        raise ValueError("REF2VA continuation requires references and no last_frame")
    if requested == "hybrid" and not has_user_refs:
        raise ValueError("HYBRID continuation requires at least one reference media input")
    return f"{requested}-motion"


def build_long_video_conditioning(
    clip,
    video_vae,
    audio_vae,
    context: dict,
    segment_index: int,
    context_frames: int,
    context_audio: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_type: str = "auto",
    audio_mode: str = "native",
    audio_denoise_strength: float = 0.35,
    add_source_as_reference: bool = True,
    prompt_primary_audio_ordinal: int = 1,
    strict_prompt_tags: bool = True,
    ref_image_size: str = "match",
    reference_video_policy: str = "official_2_to_15s",
    drive_audio=None,
    final_audio=None,
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    first_frame_reuse: str = "segment0_only",
    persistent_identity_image=None,
    persistent_identity_strategy: str = "single_reference",
    persistent_identity_interval: int = 1,
):
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Requested canvas has {width * height:,} pixels and exceeds the configured "
            f"MiniMax H3 2.0MP cap of {MAX_PIXELS:,} pixels (1920x1088)"
        )
    if not 0.0 <= audio_denoise_strength <= 1.0:
        raise ValueError("audio_denoise_strength must be between 0 and 1")
    if context_audio not in {"video_and_audio", "video_only"}:
        raise ValueError("context_audio must be video_and_audio or video_only")
    if first_frame_reuse not in FIRST_FRAME_REUSE_POLICIES:
        raise ValueError(
            "first_frame_reuse must be segment0_only or persistent_identity_reference"
        )
    if first_frame_reuse == "persistent_identity_reference" and first_frame is None:
        raise ValueError("persistent_identity_reference requires a connected first_frame")
    if persistent_identity_strategy not in PERSISTENT_IDENTITY_STRATEGIES:
        raise ValueError(
            "persistent_identity_strategy must be single_reference or scene_plus_identity"
        )
    persistent_identity_interval = int(persistent_identity_interval)
    if persistent_identity_interval < 1:
        raise ValueError("persistent_identity_interval must be at least 1")
    if (
        first_frame_reuse == "persistent_identity_reference"
        and persistent_identity_strategy == "scene_plus_identity"
        and persistent_identity_image is None
    ):
        raise ValueError("scene_plus_identity requires a connected persistent_identity_image")

    context_active = _validate_context(context, segment_index, context_frames, width, height)
    latent, frame_count = empty_av_latent(width, height, length)
    _, template_audio = nested_av_parts(latent)
    if context_active and int(context_frames) >= frame_count:
        raise ValueError("Motion context must be shorter than the current H3 target")

    warnings: list[str] = []
    keyframes: list[dict] = []
    keyframe_images: list[torch.Tensor] = []
    picture_labels: list[str] = []
    motion_audio_ref = None
    persistent_identity_reference = False
    persistent_identity_source = "none"
    persistent_identity_sources: list[str] = []
    persistent_identity_images: list[torch.Tensor] = []
    persistent_identity_requested = (
        first_frame_reuse == "persistent_identity_reference"
    )
    persistent_identity_due = False
    if context_active:
        motion_keyframes, motion_audio_ref = _motion_context_blocks(
            context,
            int(context_frames),
            context_audio == "video_and_audio",
        )
        keyframes.extend(motion_keyframes)
        persistent_identity_due = (
            persistent_identity_requested
            and (int(segment_index) - 1) % persistent_identity_interval == 0
        )
        if first_frame is not None and persistent_identity_due:
            persistent_identity_reference = True
            if persistent_identity_strategy == "scene_plus_identity":
                persistent_identity_sources = ["first_frame", "persistent_identity_image"]
                persistent_identity_images = [first_frame, persistent_identity_image]
            elif persistent_identity_image is not None:
                persistent_identity_sources = ["persistent_identity_image"]
                persistent_identity_images = [persistent_identity_image]
            else:
                persistent_identity_sources = ["first_frame"]
                persistent_identity_images = [first_frame]
            persistent_identity_source = "+".join(persistent_identity_sources)
            warnings.append(
                f"{persistent_identity_source} is used as a non-timeline identity reference; "
                "this adds reference rows and is not a guarantee of identity preservation"
            )
        elif first_frame is not None and persistent_identity_requested:
            warnings.append(
                f"persistent identity references were skipped on segment {int(segment_index)} "
                f"by interval {persistent_identity_interval}; only bounded motion context is used"
            )
        elif first_frame is not None:
            warnings.append(
                "initial first_frame was ignored because the previous segment motion context owns the head"
            )
        if (
            persistent_identity_image is not None
            and not persistent_identity_requested
        ):
            warnings.append(
                "persistent_identity_image was ignored because first_frame_reuse is segment0_only"
            )
    elif first_frame is not None:
        image = resize_image(first_frame[:1], width, height, "disabled")
        keyframe_images.append(image)
        picture_labels.append("first_frame (exact frame 0)")
        keyframes.append({"resolved_frame_index": 0, "latent": video_vae.encode(image)})
        if persistent_identity_image is not None:
            warnings.append(
                "persistent_identity_image is continuation-only and was ignored on segment 0"
            )
    elif persistent_identity_image is not None:
        warnings.append(
            "persistent_identity_image is continuation-only and was ignored because no motion "
            "context is active"
        )

    if last_frame is not None:
        image = resize_image(last_frame[:1], width, height, "center")
        keyframe_images.append(image)
        picture_labels.append(f"last_frame (exact frame {frame_count - 1})")
        keyframes.append({"resolved_frame_index": frame_count - 1, "latent": video_vae.encode(image)})

    ref_image_values = sorted_autogrow_values(ref_images)
    ref_video_entries = sorted_autogrow_items(ref_videos)
    ref_audio_values = sorted_autogrow_values(ref_audios)
    ref_video_audio_by_ordinal = dict(sorted_autogrow_items(ref_video_audios))
    if len(ref_image_values) > 9 or len(ref_video_entries) > 3 or len(ref_audio_values) > 3:
        raise ValueError("MiniMax H3 reference limits are 9 pictures, 3 videos, and 3 standalone audios")
    persistent_identity_reference_count = len(persistent_identity_images)
    if len(ref_image_values) + persistent_identity_reference_count > 9:
        raise ValueError(
            "Persistent identity reference image(s) plus user reference images must not exceed "
            "9 pictures"
        )
    video_ordinals = {ordinal for ordinal, _ in ref_video_entries}
    orphan_soundtracks = sorted(set(ref_video_audio_by_ordinal) - video_ordinals)
    if orphan_soundtracks:
        raise ValueError(
            "Reference-video soundtrack(s) have no same-numbered video: "
            + ", ".join(map(str, orphan_soundtracks))
        )

    mode = audio_mode.lower()
    if mode not in {"native", "reference_only", "lock_source", "remix_source"}:
        raise ValueError(f"Unknown audio mode: {audio_mode}")
    if mode != "native" and drive_audio is None:
        raise ValueError(f"Audio mode {audio_mode} requires drive_audio")
    if drive_audio is None:
        add_source_as_reference = False

    ref_items: list[dict] = []
    ref_blocks: list[dict] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    for identity_source, identity_image in zip(
        persistent_identity_sources, persistent_identity_images, strict=True
    ):
        resized, ref_width, ref_height = _resize_reference_image(
            identity_image, width, height, ref_image_size
        )
        encoded = video_vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": ref_height // 16,
            "latent_w": ref_width // 16,
            "latent": encoded,
        })
        picture_labels.append(f"{identity_source} (persistent identity reference)")

    for index, image in enumerate(ref_image_values, 1):
        resized, ref_width, ref_height = _resize_reference_image(image, width, height, ref_image_size)
        encoded = video_vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": ref_height // 16,
            "latent_w": ref_width // 16,
            "latent": encoded,
        })
        picture_labels.append(f"ref_image_{index}")

    for index, (video_ordinal, frames) in enumerate(ref_video_entries, 1):
        if frames.ndim != 4 or frames.shape[0] < 5:
            raise ValueError(f"ref_video_{index} must contain at least 5 IMAGE frames")
        input_frame_count = int(frames.shape[0])
        if reference_video_policy == "official_2_to_15s" and not (2 * FPS <= input_frame_count <= 15 * FPS):
            raise ValueError(
                f"ref_video_{index} has {input_frame_count} frames; official guidance is 48-360 frames at 24fps"
            )
        source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
        canvas_width, canvas_height = adapt_canvas(source_width, source_height)
        if source_width * source_height < canvas_width * canvas_height:
            canvas_width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            canvas_height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = resize_image(frames, canvas_width, canvas_height)
        frames = frames[:frame_count]
        aligned_count = align_frame_count_down(int(frames.shape[0]))
        if aligned_count < 5:
            raise ValueError(f"ref_video_{index} is too short after 17n+5 alignment")
        frames = frames[:aligned_count]
        encoded_video = video_vae.encode(frames)

        soundtrack = ref_video_audio_by_ordinal.get(video_ordinal)
        encoded_soundtrack, soundtrack_t = None, 0
        if soundtrack is not None:
            encoded_soundtrack, soundtrack_t = _encode_reference_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
            audio_labels.append(f"ref_video_audio_{video_ordinal}")
        sample_indices = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({
            "type": "video",
            "data": frames[sample_indices],
            "timestamps": [sample_index / FPS for sample_index in sample_indices],
        })
        ref_blocks.append({
            "kind": "video_audio" if soundtrack_t else "video",
            "latent_t": int(encoded_video.shape[2]),
            "latent_h": canvas_height // 16,
            "latent_w": canvas_width // 16,
            "ref_audio_t": soundtrack_t,
            "latent": encoded_video,
            "audio_latent": encoded_soundtrack,
        })
        video_labels.append(f"ref_video_{video_ordinal}")

    encoded_source = None
    source_audio_ordinal = 0
    if drive_audio is not None:
        encoded_source = fit_audio_latent(encode_audio_once(audio_vae, drive_audio), template_audio)
        if add_source_as_reference:
            ref_items.append({"type": "audio"})
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": int(encoded_source.shape[-1]),
                "audio_latent": encoded_source,
            })
            audio_labels.append("drive_audio (primary source)")
            source_audio_ordinal = len(audio_labels)

    for index, audio in enumerate(ref_audio_values, 1):
        encoded_audio, audio_t = _encode_reference_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": encoded_audio})
        audio_labels.append(f"ref_audio_{index}")

    has_user_refs = bool(ref_blocks)
    resolved_task = _resolve_long_task_type(
        task_type,
        context_active,
        None if context_active else first_frame,
        last_frame,
        has_user_refs,
    )
    counts = {
        "pictures": len(picture_labels),
        "videos": len(video_labels),
        "audios": len(audio_labels),
    }
    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        counts,
        source_audio_ordinal=source_audio_ordinal,
        prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
        strict=strict_prompt_tags,
    )
    warnings.extend(prompt_warnings)

    if ref_blocks:
        token_items = [{"type": "image", "data": image} for image in keyframe_images] + ref_items
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=token_items)
    elif keyframe_images:
        tokens = clip.tokenize(conditioned_prompt, images=keyframe_images)
    else:
        tokens = clip.tokenize(conditioned_prompt)

    refs = list(ref_blocks)
    if motion_audio_ref is not None:
        refs.append(motion_audio_ref)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    values = {LONG_VIDEO_CONDITIONING_KEY: LONG_VIDEO_SCHEMA}
    if keyframes:
        values.update({"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
    if refs:
        values["minimax_refs"] = refs
    conditioning = node_helpers.conditioning_set_values(conditioning, values)

    if mode == "lock_source":
        latent = replace_audio_latent(latent, encoded_source, 0.0)
    elif mode == "remix_source":
        latent = replace_audio_latent(latent, encoded_source, audio_denoise_strength)

    media_map = media_map_json(picture_labels, video_labels, audio_labels, source_audio_ordinal)
    report = {
        "schema": LONG_VIDEO_SCHEMA,
        "segment_index": int(segment_index),
        "context_active": context_active,
        "context_frames": int(context_frames),
        "context_audio": context_audio,
        "task": resolved_task,
        "audio_mode": mode,
        "render_frames": frame_count,
        "motion_keyframes": sum(MOTION_FRAME_INDEX in item for item in keyframes),
        "first_frame_reuse": first_frame_reuse,
        "persistent_identity_requested": persistent_identity_requested,
        "persistent_identity_reference": persistent_identity_reference,
        "persistent_identity_strategy": persistent_identity_strategy,
        "persistent_identity_interval": persistent_identity_interval,
        "persistent_identity_due": persistent_identity_due,
        "persistent_identity_source": persistent_identity_source,
        "persistent_identity_sources": persistent_identity_sources,
        "persistent_identity_reference_count": persistent_identity_reference_count,
        "persistent_identity_image_connected": persistent_identity_image is not None,
        "user_reference_blocks": len(ref_blocks) - persistent_identity_reference_count,
        "reference_blocks_total": len(ref_blocks),
        "timeline_audio_ref": motion_audio_ref is not None,
        "warnings": warnings,
    }
    output_audio = final_audio if final_audio is not None else drive_audio
    return (
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _ref_advance(ref: dict) -> float:
    kind = ref.get("kind")
    if kind == HYBRID_KEYFRAME_SENTINEL:
        return 0.0
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(ref.get("ref_audio_t", 0))
    if kind in {"video", "video_audio"}:
        audio_t = float(ref.get("ref_audio_t", 0))
        video_t = int(ref.get("latent_t", 0))
        return max(audio_t, FRAME_RESCALE * pixel_frames_from_latent_t(video_t))
    raise RuntimeError(f"Unsupported MiniMax H3 reference kind in long-video layout: {kind!r}")


def _locate_ref_segments(layout, keyframes: list[dict], refs: list[dict]) -> dict[int, tuple[int, int]]:
    segments = list(layout.segments)
    cursor = 1 + len(keyframes)
    located: dict[int, tuple[int, int]] = {}

    for ref_index, ref in enumerate(refs):
        kind = ref.get("kind")
        if kind == HYBRID_KEYFRAME_SENTINEL:
            continue
        if kind == "image":
            expected = ["ref_img"]
        elif kind == "audio":
            expected = ["ref_audio"] if int(ref.get("ref_audio_t", 0)) > 0 else []
        elif kind in {"video", "video_audio"}:
            expected = []
            if int(ref.get("ref_audio_t", 0)) > 0:
                expected.append("ref_audio")
            expected.append("ref_img")
        else:
            raise RuntimeError(f"Unsupported MiniMax H3 reference kind: {kind!r}")

        for expected_kind in expected:
            if cursor >= len(segments) or segments[cursor][2] != expected_kind:
                raise RuntimeError(
                    f"MiniMax H3 layout/ref segment mismatch at ref {ref_index}: "
                    f"expected {expected_kind}"
                )
            if expected_kind == "ref_audio":
                located[ref_index] = (segments[cursor][0], segments[cursor][1])
            cursor += 1

    if cursor + 2 != len(segments) or [segments[cursor][2], segments[cursor + 1][2]] != ["audio", "video"]:
        raise RuntimeError("MiniMax H3 target audio/video segments changed; long-video patch refused")
    return located


def repair_long_video_payload(out: dict, kwargs: dict) -> dict:
    if int(kwargs.get(LONG_VIDEO_CONDITIONING_KEY, 0) or 0) != LONG_VIDEO_SCHEMA:
        return out
    cond = out.get("minimax_payload")
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError("Long-video model patch could not access the MiniMax H3 payload")

    keyframes = list(kwargs.get("minimax_keyframes") or [])
    refs = list(kwargs.get("minimax_refs") or [])
    payload["cond_video_latents"] = [
        keyframe["latent"] for keyframe in keyframes if "latent" in keyframe
    ] + [
        ref["latent"]
        for ref in refs
        if ref.get("kind") != HYBRID_KEYFRAME_SENTINEL and "latent" in ref
    ]
    payload["cond_audio_latents"] = [
        ref["audio_latent"] for ref in refs if ref.get("audio_latent") is not None
    ]
    if kwargs.get("minimax_frame_count") is not None:
        payload["frame_count"] = kwargs["minimax_frame_count"]

    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("Long-video MiniMax H3 payload has no PackedLayout")
    text_len, latent_t = int(layout.signature[0]), int(layout.signature[1])
    frame_count = payload.get("frame_count")
    ref_offset = sum(_ref_advance(ref) for ref in refs)

    cond_segments = [(start, stop) for start, stop, kind in layout.segments if kind == "cond"]
    if len(cond_segments) != len(keyframes):
        raise RuntimeError("MiniMax H3 keyframe/layout count changed; long-video patch refused")
    for (start, stop), keyframe in zip(cond_segments, keyframes):
        pixel_index = int(keyframe.get(MOTION_FRAME_INDEX, keyframe["resolved_frame_index"]))
        if pixel_index < 0 or (frame_count is not None and pixel_index >= int(frame_count)):
            raise RuntimeError(f"Long-video keyframe index {pixel_index} is outside the target")
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == int(frame_count) - 1:
            cond_t = float(text_len) + FRAME_RESCALE * pixel_frames_from_latent_t(latent_t) - FRAME_RESCALE
        else:
            cond_t = float(text_len) + FRAME_RESCALE * pixel_index
        layout.position_ids[start:stop, 0] = cond_t + ref_offset

    marked = [index for index, ref in enumerate(refs) if ref.get(MOTION_AUDIO_END_FRAME) is not None]
    if len(marked) > 1:
        raise RuntimeError("Long-video layout supports one marked continuation audio window")
    if marked:
        ref_index = marked[0]
        ref = refs[ref_index]
        if ref.get("kind") != "audio":
            raise RuntimeError("Long-video audio timeline marker is only valid on an audio ref")
        located = _locate_ref_segments(layout, keyframes, refs)
        if ref_index not in located:
            raise RuntimeError("Long-video marked audio ref emitted no layout rows")
        ref_origin = float(text_len) + sum(_ref_advance(item) for item in refs[:ref_index])
        target_origin = float(text_len) + ref_offset
        audio_t = int(ref.get("ref_audio_t", 0))
        end_frame = float(ref[MOTION_AUDIO_END_FRAME])
        shift = target_origin + FRAME_RESCALE * end_frame - audio_t - ref_origin
        start, stop = located[ref_index]
        layout.position_ids[start:stop, 0] += shift

    expected_video_latents = sum(kind in {"cond", "ref_img"} for _, _, kind in layout.segments)
    expected_audio_latents = sum(kind == "ref_audio" for _, _, kind in layout.segments)
    if len(payload["cond_video_latents"]) != expected_video_latents:
        raise RuntimeError(
            "Long-video video payload/layout count mismatch: "
            f"{len(payload['cond_video_latents'])} latents for {expected_video_latents} segments"
        )
    if len(payload["cond_audio_latents"]) != expected_audio_latents:
        raise RuntimeError(
            "Long-video audio payload/layout count mismatch: "
            f"{len(payload['cond_audio_latents'])} latents for {expected_audio_latents} segments"
        )
    payload["t8_long_video_patch_version"] = LONG_VIDEO_PATCH_VERSION
    return out


def patch_long_video_model(model):
    if not hasattr(model, "clone") or not hasattr(model, "add_object_patch"):
        raise ValueError("model is not a ComfyUI MODEL patcher")
    patched = model.clone()
    original = patched.get_model_object("extra_conds")
    if getattr(original, "_t8_multikeyframe_patch_version", None) is not None:
        raise ValueError(
            "MiniMax H3 Long Video Conditioning and Multi-Keyframe Advanced cannot be "
            "stacked until their patch order has been validated"
        )
    if getattr(original, "_t8_long_video_patch_version", None) == LONG_VIDEO_PATCH_VERSION:
        return patched

    base_model = patched.model
    if type(base_model).__name__ != "MiniMaxH3":
        diffusion_name = type(getattr(base_model, "diffusion_model", None)).__name__
        if diffusion_name != "MiniMaxH3Model":
            raise ValueError("MiniMax H3 Long Video Conditioning requires a MiniMax H3 MODEL")

    def _patched_extra_conds(_self, **kwargs):
        out = original(**kwargs)
        return repair_long_video_payload(out, kwargs)

    _patched_extra_conds._t8_long_video_patch_version = LONG_VIDEO_PATCH_VERSION
    patched.add_object_patch("extra_conds", types.MethodType(_patched_extra_conds, base_model))
    return patched


def packed_layout_contract_probe() -> dict:
    text_len, latent_t, latent_h, latent_w, audio_t = 7, 37, 8, 8, 207
    frame_count = pixel_frames_from_latent_t(latent_t)
    keyframes = [
        {"resolved_frame_index": 0, MOTION_FRAME_INDEX: offset, "latent": torch.zeros(1)}
        for offset in step_offsets(CONTEXT_FRAME_STEPS[22])
    ]
    refs = [{
        "kind": "audio",
        "ref_audio_t": round(22 / FPS * AUDIO_LATENT_FPS),
        "audio_latent": torch.zeros(1),
        MOTION_AUDIO_END_FRAME: 22.0,
    }]
    layout = build_packed_layout(
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=keyframes,
        refs=refs,
        frame_count=frame_count,
    )
    return {
        "frame_count": frame_count,
        "segments": [(a, b, kind) for a, b, kind in layout.segments],
        "signature": list(layout.signature),
    }
