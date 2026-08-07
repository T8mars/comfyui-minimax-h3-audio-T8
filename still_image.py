from __future__ import annotations

import json
import math

import torch

import comfy.model_management
import comfy.nested_tensor
import node_helpers
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout

from .core import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    MAX_PIXELS,
    REF_IMAGE_SHORT_EDGE,
    VRAM_CAUTION_PIXELS,
    adapt_canvas,
    classify_h3_vae,
    nested_av_parts,
    resize_image,
    sorted_autogrow_values,
)
from .prompt_tags import media_map_json, prepare_prompt


# Native H3 video timing maps frames 17k+5 to latent T=5k+2.
# The direct one-frame target is an explicitly experimental exception.
STILL_TARGETS = {
    "direct_1_frame": (1, 1),
    "micro_video_5_frames": (5, 2),
    "short_video_22_frames": (22, 7),
    "trained_124_frames": (124, 37),
}


def resolve_still_target(target_mode: str) -> tuple[int, int, int]:
    if target_mode not in STILL_TARGETS:
        raise ValueError(f"Unknown MiniMax H3 still target mode: {target_mode}")
    frame_count, latent_t = STILL_TARGETS[target_mode]
    audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    return frame_count, latent_t, audio_t


def resolve_still_canvas(
    edit_image: torch.Tensor,
    canvas_mode: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    if not isinstance(edit_image, torch.Tensor) or edit_image.ndim != 4:
        raise ValueError("edit_image must be an IMAGE tensor [B,H,W,C]")
    if edit_image.shape[0] < 1 or edit_image.shape[-1] < 3:
        raise ValueError("edit_image must contain at least one RGB image")

    if canvas_mode == "from_edit_image":
        width, height = adapt_canvas(int(edit_image.shape[2]), int(edit_image.shape[1]))
        while width * height > MAX_PIXELS:
            if width >= height:
                width -= CANVAS_MULTIPLE
            else:
                height -= CANVAS_MULTIPLE
    elif canvas_mode != "custom":
        raise ValueError(f"Unknown still canvas mode: {canvas_mode}")

    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("MiniMax H3 still width and height must be positive multiples of 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Still canvas has {width * height:,} pixels; configured H3 2.0MP cap is "
            f"{MAX_PIXELS:,} pixels (1920x1088)"
        )
    return width, height


def collect_still_references(edit_image, ref_images=None) -> list[torch.Tensor]:
    if edit_image is None:
        raise ValueError("MiniMax H3 image editing requires edit_image as Picture 1")
    images = [edit_image] + sorted_autogrow_values(ref_images)
    if len(images) > 9:
        raise ValueError("MiniMax H3 image editing supports at most 9 total reference pictures")
    for index, image in enumerate(images, 1):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(f"Picture {index} must be an IMAGE tensor [B,H,W,C]")
        if image.shape[0] < 1 or image.shape[-1] < 3:
            raise ValueError(f"Picture {index} must contain at least one RGB image")
    return images


def _resize_still_reference(
    image: torch.Tensor,
    width: int,
    height: int,
    ref_image_size: str,
) -> tuple[torch.Tensor, int, int]:
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    elif ref_image_size == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
    else:
        raise ValueError(f"Unknown reference image size policy: {ref_image_size}")
    target_width = max(
        CANVAS_MULTIPLE,
        round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    target_height = max(
        CANVAS_MULTIPLE,
        round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    resized = resize_image(image[:1], target_width, target_height)
    return resized, target_width, target_height


def empty_still_av_latent(
    width: int,
    height: int,
    target_mode: str,
    audio_target: str,
) -> tuple[dict, int]:
    frame_count, latent_t, audio_t = resolve_still_target(target_mode)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, audio_t), device=device)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if audio_target == "lock_silence":
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.zeros_like(audio))
        )
    elif audio_target != "generate_and_discard":
        raise ValueError(f"Unknown still audio target strategy: {audio_target}")
    return latent, frame_count


def assert_still_layout_contract(
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    refs: list[dict],
) -> None:
    layout = PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t, refs=refs)
    kinds = [kind for _, _, kind in layout.segments]
    if kinds[-2:] != ["audio", "video"] or kinds.count("ref_img") != len(refs):
        raise RuntimeError(
            "This ComfyUI build changed MiniMax H3 still/reference packing; "
            "the experimental image-edit path is disabled until revalidated."
        )


def build_still_image_edit_conditioning(
    clip,
    video_vae,
    prompt: str,
    canvas_mode: str,
    width: int,
    height: int,
    target_mode: str,
    reference_strength: float,
    audio_target: str,
    strict_prompt_tags: bool,
    ref_image_size: str,
    edit_image,
    ref_images=None,
):
    if not 0.0 <= reference_strength <= 1.0:
        raise ValueError("reference_strength must be between 0 and 1")
    references = collect_still_references(edit_image, ref_images)
    width, height = resolve_still_canvas(edit_image, canvas_mode, width, height)
    latent, frame_count = empty_still_av_latent(width, height, target_mode, audio_target)
    video, audio = nested_av_parts(latent)

    ref_items = []
    ref_blocks = []
    picture_labels = ["edit_image (primary source)"]
    for index, image in enumerate(references, 1):
        resized, ref_width, ref_height = _resize_still_reference(
            image, width, height, ref_image_size
        )
        encoded = video_vae.encode(resized)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5 or encoded.shape[2] != 1:
            raise ValueError(
                f"MiniMax H3 video VAE must encode Picture {index} as [1,24,1,H,W]"
            )
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append(
            {
                "kind": "image",
                "latent_h": ref_height // 16,
                "latent_w": ref_width // 16,
                "latent": encoded,
            }
        )
        if index > 1:
            picture_labels.append(f"ref_image_{index - 1}")

    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        {"pictures": len(ref_blocks), "videos": 0, "audios": 0},
        source_audio_ordinal=0,
        prompt_primary_audio_ordinal=0,
        strict=strict_prompt_tags,
    )
    tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    text_len = int(conditioning[0][0].shape[1])
    assert_still_layout_contract(
        text_len,
        int(video.shape[2]),
        int(video.shape[3]),
        int(video.shape[4]),
        int(audio.shape[-1]),
        ref_blocks,
    )
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {
            "minimax_refs": ref_blocks,
            "minimax_visual_cond_noise_aug": float(reference_strength),
        },
    )

    warnings = [
        "experimental Ref2VA still-image path; not an official T2I/img2img task",
    ]
    if min(width, height) < 512:
        warnings.append(
            "canvas short edge below 512 is far below normal H3 use and may collapse image structure"
        )
    if frame_count < 124:
        warnings.append(f"{frame_count}-frame target is outside the approximate 124-362 frame training range")
    if audio_target == "lock_silence":
        warnings.append("target audio is locked to silence; compare against generate_and_discard for quality")
    warnings.extend(prompt_warnings)
    report = json.dumps(
        {
            "task": "reference_image_edit",
            "status": "experimental",
            "required_checkpoint_family": "MiniMax H3 Ref2VA",
            "canvas": {"width": width, "height": height},
            "target": {
                "mode": target_mode,
                "frames": frame_count,
                "video_latent_shape": list(video.shape),
                "audio_latent_shape": list(audio.shape),
                "audio_strategy": audio_target,
            },
            "reference_strength": reference_strength,
            "pictures": len(ref_blocks),
            "warnings": warnings,
        },
        ensure_ascii=False,
        indent=2,
    )
    media_map = media_map_json(picture_labels, [], [], 0)
    return conditioning, latent, conditioned_prompt, media_map, report


def run_still_preflight(
    canvas_mode: str,
    width: int,
    height: int,
    target_mode: str,
    reference_strength: float,
    audio_target: str,
    model=None,
    video_vae=None,
    edit_image=None,
    ref_images=None,
):
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, object] = {
        "required_checkpoint_family": "MiniMax H3 Ref2VA",
        "experimental": True,
    }

    references = []
    try:
        references = collect_still_references(edit_image, ref_images)
    except ValueError as error:
        errors.append(str(error))

    if edit_image is not None:
        try:
            resolved_width, resolved_height = resolve_still_canvas(
                edit_image, canvas_mode, width, height
            )
            facts["canvas"] = {"width": resolved_width, "height": resolved_height}
            if min(resolved_width, resolved_height) < 512:
                warnings.append(
                    "canvas short edge below 512 may collapse image structure; use from_edit_image or a larger custom canvas"
                )
            if resolved_width * resolved_height > VRAM_CAUTION_PIXELS:
                warnings.append(
                    "high-resolution still canvas is supported up to 1920x1088 but may require substantially more VRAM"
                )
        except ValueError as error:
            errors.append(str(error))

    try:
        frame_count, latent_t, audio_t = resolve_still_target(target_mode)
        facts["target"] = {
            "frames": frame_count,
            "video_latent_t": latent_t,
            "audio_latent_t": audio_t,
        }
        if frame_count < 124:
            warnings.append(
                f"{frame_count}-frame still target is outside the approximate 124-362 frame training range"
            )
    except ValueError as error:
        errors.append(str(error))

    if not 0.0 <= reference_strength <= 1.0:
        errors.append("reference_strength must be between 0 and 1")
    elif reference_strength < 0.95:
        warnings.append("reference_strength below 0.95 adds substantial noise to reference latents")

    if audio_target not in {"generate_and_discard", "lock_silence"}:
        errors.append(f"Unknown still audio target strategy: {audio_target}")
    elif audio_target == "lock_silence":
        warnings.append("locked-silence audio is experimental and may change image quality")

    if model is not None:
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if not isinstance(diffusion_model, MiniMaxH3Model):
            errors.append("model is not a native ComfyUI MiniMax H3 diffusion model")
    else:
        warnings.append("connect model to verify the native MiniMax H3 model class")
    if video_vae is None:
        warnings.append("connect video_vae to include it in the preflight graph")
    else:
        video_vae_kind = classify_h3_vae(video_vae)
        facts["video_vae_kind"] = video_vae_kind
        if video_vae_kind == "audio":
            errors.append("video_vae is an H3 audio VAE; connect the H3 video VAE")

    facts["pictures"] = len(references)
    warnings.append("checkpoint variant cannot be inferred reliably; select a Ref2VA checkpoint")
    ready = not errors
    return ready, len(warnings), json.dumps(
        {"ready": ready, "errors": errors, "warnings": warnings, "facts": facts},
        ensure_ascii=False,
        indent=2,
    )


def decode_still_image(
    av_latent: dict,
    video_vae,
    frame_selection: str,
    frame_index: int,
):
    video, _audio = nested_av_parts(av_latent)
    frames = video_vae.decode(video)
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("MiniMax H3 video VAE did not decode an IMAGE frame batch")

    frame_count = int(frames.shape[0])
    if frame_selection == "first":
        selected_index = 0
    elif frame_selection == "middle":
        selected_index = frame_count // 2
    elif frame_selection == "last":
        selected_index = frame_count - 1
    elif frame_selection == "index":
        selected_index = int(frame_index)
    else:
        raise ValueError(f"Unknown still frame selection: {frame_selection}")
    if not 0 <= selected_index < frame_count:
        raise ValueError(
            f"frame_index {selected_index} is outside the decoded range 0-{frame_count - 1}"
        )

    report = json.dumps(
        {
            "decoded_frames": frame_count,
            "selection": frame_selection,
            "selected_index": selected_index,
            "video_latent_shape": list(video.shape),
        },
        ensure_ascii=False,
        indent=2,
    )
    return frames[selected_index:selected_index + 1], frames, report
