from __future__ import annotations

import inspect
import math

import torch

import node_helpers
from comfy.ldm.minimax.model import PackedLayout
from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel

from .core import (
    CANVAS_MULTIPLE,
    FPS,
    MAX_PIXELS,
    REF_IMAGE_SHORT_EDGE,
    adapt_canvas,
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


HYBRID_KEYFRAME_SENTINEL = "t8_keyframe_latent"


def assert_hybrid_layout_contract() -> None:
    """Fail loudly if upstream starts packing our sentinel as a real reference.

    Current ComfyUI replaces keyframe cond latents with the ref latent list when
    both payloads are present. The sentinel makes that latent list start with the
    keyframes while PackedLayout intentionally ignores its unknown ref kind.
    """
    extra_conds_source = inspect.getsource(MiniMaxH3BaseModel.extra_conds)
    required_contract = 'payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]'
    if required_contract not in extra_conds_source:
        raise RuntimeError(
            "This ComfyUI build changed MiniMax H3 ref/keyframe latent assembly; "
            "the T8 Hybrid path is disabled until its ordering can be revalidated."
        )

    keyframe = {"resolved_frame_index": 0, "latent": torch.zeros(1)}
    baseline = PackedLayout(1, 2, 2, 2, 1, keyframes=[keyframe], frame_count=5)
    hybrid = PackedLayout(
        1,
        2,
        2,
        2,
        1,
        keyframes=[keyframe],
        refs=[{"kind": HYBRID_KEYFRAME_SENTINEL, "latent": torch.zeros(1)}],
        frame_count=5,
    )
    if baseline.segments != hybrid.segments or baseline.seq_len != hybrid.seq_len:
        raise RuntimeError(
            "This ComfyUI build changed MiniMax H3 PackedLayout reference handling; "
            "the T8 exact-keyframe + reference compatibility path is disabled to prevent corrupt conditioning."
        )


def resolve_task_type(task_type: str, first_frame, last_frame, has_refs: bool) -> str:
    first = first_frame is not None
    last = last_frame is not None
    requested = (task_type or "auto").lower()
    if requested == "auto":
        if has_refs and (first or last):
            return "hybrid"
        if has_refs:
            return "ref2va"
        if first and last:
            return "fl2va"
        if first:
            return "i2va"
        if last:
            return "l2va"
        return "t2va"

    requirements = {
        "t2va": (False, False, False),
        "i2va": (True, False, False),
        "fl2va": (True, True, False),
        "l2va": (False, True, False),
        "ref2va": (False, False, True),
        "hybrid": (None, None, True),
    }
    if requested not in requirements:
        raise ValueError(f"Unknown MiniMax H3 task type: {task_type}")
    need_first, need_last, need_refs = requirements[requested]
    if need_first is not None and first != need_first:
        raise ValueError(f"{requested.upper()} first_frame connection does not match the selected task")
    if need_last is not None and last != need_last:
        raise ValueError(f"{requested.upper()} last_frame connection does not match the selected task")
    if need_refs and not has_refs:
        raise ValueError(f"{requested.upper()} requires at least one reference media input")
    if requested == "hybrid" and not (first or last):
        raise ValueError("HYBRID requires first_frame and/or last_frame")
    if requested not in {"ref2va", "hybrid"} and has_refs:
        raise ValueError(f"{requested.upper()} cannot include reference media; use Auto or Hybrid")
    return requested


def _resize_reference_image(image, width: int, height: int, ref_image_size: str):
    h, w = int(image.shape[1]), int(image.shape[2])
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    target_width = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return resize_image(image[:1], target_width, target_height), target_width, target_height


def _encode_reference_audio(audio_vae, audio: dict):
    latent = encode_audio_once(audio_vae, audio)
    return latent, int(latent.shape[-1])


def build_conditioning(
    clip,
    video_vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_type: str = "auto",
    audio_mode: str = "lock_source",
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
):
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Requested canvas has {width * height:,} pixels and exceeds the configured "
            f"MiniMax H3 2.0MP cap of {MAX_PIXELS:,} pixels (1920x1088); reduce width/height"
        )
    if not 0.0 <= audio_denoise_strength <= 1.0:
        raise ValueError("audio_denoise_strength must be between 0 and 1")

    ref_image_values = sorted_autogrow_values(ref_images)
    ref_video_entries = sorted_autogrow_items(ref_videos)
    ref_video_values = [value for _, value in ref_video_entries]
    ref_audio_values = sorted_autogrow_values(ref_audios)
    ref_video_audio_by_ordinal = dict(sorted_autogrow_items(ref_video_audios))
    if len(ref_image_values) > 9 or len(ref_video_values) > 3 or len(ref_audio_values) > 3:
        raise ValueError("MiniMax H3 reference limits are 9 pictures, 3 videos, and 3 standalone audios")
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
    if drive_audio is None and add_source_as_reference:
        add_source_as_reference = False

    latent, frame_count = empty_av_latent(width, height, length)
    _, template_audio = nested_av_parts(latent)

    keyframes = []
    keyframe_images = []
    picture_labels: list[str] = []
    if first_frame is not None:
        image = resize_image(first_frame[:1], width, height, "disabled")
        keyframe_images.append(image)
        picture_labels.append("first_frame (exact frame 0)")
        keyframes.append({"resolved_frame_index": 0, "latent": video_vae.encode(image)})
    if last_frame is not None:
        image = resize_image(last_frame[:1], width, height, "center")
        keyframe_images.append(image)
        picture_labels.append(f"last_frame (exact frame {frame_count - 1})")
        keyframes.append({"resolved_frame_index": frame_count - 1, "latent": video_vae.encode(image)})

    real_ref_items: list[dict] = []
    real_ref_blocks: list[dict] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    for index, image in enumerate(ref_image_values, 1):
        resized, ref_width, ref_height = _resize_reference_image(image, width, height, ref_image_size)
        encoded = video_vae.encode(resized)
        real_ref_items.append({"type": "image", "data": resized})
        real_ref_blocks.append(
            {
                "kind": "image",
                "latent_h": ref_height // 16,
                "latent_w": ref_width // 16,
                "latent": encoded,
            }
        )
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
            real_ref_items.append({"type": "audio"})
            audio_labels.append(f"ref_video_audio_{video_ordinal}")
        sample_indices = list(range(0, frames.shape[0], FPS // 2))
        real_ref_items.append(
            {
                "type": "video",
                "data": frames[sample_indices],
                "timestamps": [sample_index / FPS for sample_index in sample_indices],
            }
        )
        real_ref_blocks.append(
            {
                "kind": "video_audio" if soundtrack_t else "video",
                "latent_t": int(encoded_video.shape[2]),
                "latent_h": canvas_height // 16,
                "latent_w": canvas_width // 16,
                "ref_audio_t": soundtrack_t,
                "latent": encoded_video,
                "audio_latent": encoded_soundtrack,
            }
        )
        video_labels.append(f"ref_video_{video_ordinal}")

    encoded_source = None
    source_audio_ordinal = 0
    if drive_audio is not None:
        encoded_source = fit_audio_latent(encode_audio_once(audio_vae, drive_audio), template_audio)
        if add_source_as_reference:
            real_ref_items.append({"type": "audio"})
            real_ref_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": int(encoded_source.shape[-1]),
                    "audio_latent": encoded_source,
                }
            )
            audio_labels.append("drive_audio (primary source)")
            source_audio_ordinal = len(audio_labels)

    for index, audio in enumerate(ref_audio_values, 1):
        encoded_audio, audio_t = _encode_reference_audio(audio_vae, audio)
        real_ref_items.append({"type": "audio"})
        real_ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": encoded_audio})
        audio_labels.append(f"ref_audio_{index}")

    has_refs = bool(real_ref_blocks)
    resolved_task = resolve_task_type(task_type, first_frame, last_frame, has_refs)
    counts = {"pictures": len(picture_labels), "videos": len(video_labels), "audios": len(audio_labels)}
    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        counts,
        source_audio_ordinal=source_audio_ordinal,
        prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
        strict=strict_prompt_tags,
    )

    if keyframes and real_ref_blocks:
        assert_hybrid_layout_contract()
        ref_items = [{"type": "image", "data": image} for image in keyframe_images] + real_ref_items
        refs = [
            {"kind": HYBRID_KEYFRAME_SENTINEL, "latent": keyframe["latent"]}
            for keyframe in keyframes
        ] + real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=ref_items)
    elif real_ref_blocks:
        refs = real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=real_ref_items)
    else:
        refs = []
        tokens = clip.tokenize(conditioned_prompt, images=keyframe_images)

    conditioning = clip.encode_from_tokens_scheduled(tokens)
    values = {}
    if keyframes:
        values.update({"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
    if refs:
        values["minimax_refs"] = refs
    if values:
        conditioning = node_helpers.conditioning_set_values(conditioning, values)

    if mode == "lock_source":
        latent = replace_audio_latent(latent, encoded_source, 0.0)
    elif mode == "remix_source":
        latent = replace_audio_latent(latent, encoded_source, audio_denoise_strength)

    media_map = media_map_json(picture_labels, video_labels, audio_labels, source_audio_ordinal)
    report_lines = [
        f"task={resolved_task}",
        f"audio_mode={mode}",
        f"frames={frame_count} ({frame_count / FPS:.3f}s at 24fps)",
        f"pictures={len(picture_labels)}, videos={len(video_labels)}, audios={len(audio_labels)}",
        f"source_audio_tag={'<Audio ' + str(source_audio_ordinal) + '>' if source_audio_ordinal else 'none'}",
    ]
    report_lines.extend(f"warning: {warning}" for warning in prompt_warnings)
    output_audio = final_audio if final_audio is not None else drive_audio
    return conditioning, latent, output_audio, conditioned_prompt, media_map, "\n".join(report_lines)
