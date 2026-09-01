from __future__ import annotations

# Temporal/spatial orchestration is adapted from the MIT-licensed
# Comfyui-MMH3-UltimateUpscale project by bbaudio-2025. The learned 3D model
# inference itself reuses this project's existing implementation.

import json
import math
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .learned_latent_upscale_advanced import learned_upscale_h3_av_latent
from .sampling import rebind_dual_clock_sampler


try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except ImportError:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0


VAE_DOWNSAMPLE = 16
GRID_PIXELS = 32
PLAN_SCHEMA_V1 = "t8.minimax_h3.chunked_two_pass.v1"
PLAN_SCHEMA_GLOBAL_NOISE_V2 = "t8.minimax_h3.chunked_two_pass.global_noise.v2"
PLAN_SCHEMA_LOW_SIGMA_V3 = "t8.minimax_h3.chunked_two_pass.low_sigma.v3"
PLAN_SCHEMA_MASKED_LOW_SIGMA_V4 = (
    "t8.minimax_h3.chunked_two_pass.masked_low_sigma.v4"
)
GLOBAL_NOISE_POLICY = "one_full_target_video_noise_then_exact_coordinate_slices"
TEMPORAL_OWNERSHIP_POLICY = "previous_overlap_guarded_progressive_takeover"
TEMPORAL_FULL_CLIP_POLICY = "one_full_clip_no_temporal_stitching"
AUDIO_SAMPLING_POLICIES = (
    "joint_av_preserve_input",
    "locked_input_audio",
)
VIDEO_MASK_POLICIES = (
    "inherit_required",
    "inherit_if_present_else_generate_all",
    "disabled",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def frames_for_tokens(count: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(count)))


def tokens_for_frames(frame_count: int) -> int:
    tokens = 0
    covered = 0
    while covered < int(frame_count):
        covered += FRAME_PER_TOKEN[tokens % 5]
        tokens += 1
    return tokens


def _snap_frame(frame: int, maximum_tokens: int) -> tuple[int, int]:
    choices = [
        (token, frames_for_tokens(token))
        for token in range(0, int(maximum_tokens) + 1, 5)
    ]
    return min(choices, key=lambda item: abs(item[1] - int(frame)))


def compute_temporal_segments(
    video_tokens: int, chunk_length: int, overlap: int
) -> tuple[list[tuple[int, int, int, int]], int]:
    total_frames = frames_for_tokens(video_tokens)
    if chunk_length <= 0 or overlap < 0 or chunk_length <= overlap:
        raise ValueError("chunk_length must be positive and larger than overlap")
    hop = chunk_length - overlap
    segments = []
    previous_end = 0
    index = 0
    while True:
        requested_start = index * hop
        requested_end = min(requested_start + chunk_length, total_frames)
        if index == 0:
            start_token, start_frame = 0, 0
        else:
            start_token, start_frame = _snap_frame(requested_start, video_tokens)
            if start_token > previous_end:
                start_token = previous_end
                start_frame = frames_for_tokens(start_token)
        if requested_end >= total_frames:
            end_token, end_frame = video_tokens, total_frames
        else:
            end_token, end_frame = _snap_frame(requested_end, video_tokens)
            if end_token <= start_token:
                end_token = min(video_tokens, start_token + 5)
                end_frame = frames_for_tokens(end_token)
        segments.append((start_token, start_frame, end_token, end_frame))
        if end_token >= video_tokens:
            break
        previous_end = end_token
        index += 1
    return segments, total_frames


def _grid_axis(size: int, tile: int, overlap: int, minimum: int):
    if size <= tile:
        return [0], [size], [0]
    stride = tile - overlap
    count = math.ceil((size - overlap) / stride)
    origins = [index * stride for index in range(count)]
    lengths = [min(tile, size - origin) for origin in origins]
    if minimum > 0 and len(origins) >= 2 and lengths[-1] < minimum:
        moved = size - minimum
        if origins[-2] < moved < origins[-2] + lengths[-2]:
            origins[-1] = moved
            lengths[-1] = minimum
    overlaps = [0]
    overlaps.extend(
        max(0, origins[index - 1] + lengths[index - 1] - origins[index])
        for index in range(1, len(origins))
    )
    return origins, lengths, overlaps


def compute_spatial_grid(
    height: int,
    width: int,
    tile_height: int,
    tile_width: int,
    overlap_height: int,
    overlap_width: int,
    minimum_tile: int,
):
    if min(tile_height, tile_width) <= 0:
        raise ValueError("tile dimensions must be positive")
    if overlap_height >= tile_height or overlap_width >= tile_width:
        raise ValueError("tile overlap must be smaller than the tile")
    rows, row_sizes, row_overlaps = _grid_axis(
        height, tile_height, overlap_height, minimum_tile
    )
    cols, col_sizes, col_overlaps = _grid_axis(
        width, tile_width, overlap_width, minimum_tile
    )
    return rows, cols, row_sizes, col_sizes, row_overlaps, col_overlaps


def spatial_fade_mask(
    height: int,
    width: int,
    overlap_height: int,
    overlap_width: int,
    done_top: bool,
    done_left: bool,
    fade_height: int,
    fade_width: int,
) -> torch.Tensor:
    mask = torch.ones(height, width, dtype=torch.float32)
    if done_left and overlap_width:
        fade = min(fade_width, overlap_width)
        frozen = overlap_width - fade
        mask[:, :frozen] = 0
        if fade:
            mask[:, frozen:overlap_width] = torch.linspace(0, 1, fade)[None, :]
    if done_top and overlap_height:
        fade = min(fade_height, overlap_height)
        frozen = overlap_height - fade
        mask[:frozen] = 0
        if fade:
            ramp = torch.linspace(0, 1, fade)[:, None]
            mask[frozen:overlap_height] = torch.minimum(
                mask[frozen:overlap_height], ramp
            )
    return mask


def _blend_weights(values: torch.Tensor, style: str) -> torch.Tensor:
    if style == "smoothstep":
        return values * values * (3.0 - 2.0 * values)
    return values


def _crossfade(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    count = left.shape[dim]
    weights = torch.linspace(0, 1, count, device=left.device, dtype=left.dtype)
    shape = [1] * left.ndim
    shape[dim] = count
    return left + (right - left) * weights.view(shape)


def _trim_keyframe(keyframe: dict, start_frame: int, end_frame: int):
    index = int(keyframe["resolved_frame_index"])
    video = keyframe.get("latent")
    audio = keyframe.get("audio_latent")
    if video is None and audio is None:
        if not start_frame <= index < end_frame:
            return None
        return {"resolved_frame_index": index - start_frame}
    output = {}
    if video is not None:
        first = None
        stop = None
        position = index
        for token in range(video.shape[2]):
            span = FRAME_PER_TOKEN[token % 5]
            if start_frame <= position and position + span <= end_frame:
                first = token if first is None else first
                stop = token + 1
            position += span
        if first is not None:
            output["latent"] = video[:, :, first:stop].contiguous()
            output["resolved_frame_index"] = (
                index + frames_for_tokens(first) - start_frame
            )
    if audio is not None:
        audio_start = max(0, math.ceil((start_frame - index) * FRAME_RESCALE))
        audio_end = min(
            audio.shape[-1], math.floor((end_frame - index) * FRAME_RESCALE)
        )
        if audio_end > audio_start:
            output["audio_latent"] = audio[..., audio_start:audio_end].contiguous()
            output.setdefault("resolved_frame_index", max(0, index - start_frame))
    return output or None


def reanchor_conditioning(conditioning, start_frame: int, end_frame: int, spatial):
    output = []
    for tensor, metadata in conditioning:
        updated = dict(metadata)
        keyframes = updated.get("minimax_keyframes")
        if keyframes:
            trimmed = [
                item
                for item in (
                    _trim_keyframe(keyframe, start_frame, end_frame)
                    for keyframe in keyframes
                )
                if item is not None
            ]
            for keyframe in trimmed:
                latent = keyframe.get("latent")
                if latent is not None and tuple(latent.shape[-2:]) != tuple(spatial):
                    batch, channels, time, height, width = latent.shape
                    keyframe["latent"] = F.interpolate(
                        latent.reshape(batch * time, channels, height, width),
                        size=spatial,
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(batch, channels, time, *spatial)
            if trimmed:
                updated["minimax_keyframes"] = trimmed
            else:
                updated.pop("minimax_keyframes", None)
        output.append([tensor, updated])
    return output


def anchor_conditioning(conditioning, previous_video, start_frame: int, strength: float):
    token = tokens_for_frames(start_frame)
    if token >= previous_video.shape[2]:
        raise ValueError("previous chunk does not reach the next chunk anchor")
    anchor = {
        "resolved_frame_index": 0,
        "latent": previous_video[:, :, token : token + 1].contiguous(),
    }
    output = []
    for tensor, metadata in conditioning:
        updated = dict(metadata)
        keyframes = [
            keyframe
            for keyframe in updated.get("minimax_keyframes", [])
            if keyframe.get("resolved_frame_index") != 0
            or keyframe.get("latent") is None
        ]
        updated["minimax_keyframes"] = [anchor, *keyframes]
        updated["minimax_visual_cond_noise_aug"] = max(
            0.0, min(1.0, float(strength))
        )
        output.append([tensor, updated])
    return output


def crop_conditioning(conditioning, source_h, source_w, row, col, height, width):
    output = []
    for tensor, metadata in conditioning:
        updated = dict(metadata)
        cropped = []
        for keyframe in updated.get("minimax_keyframes", []):
            copy = dict(keyframe)
            latent = keyframe.get("latent")
            if latent is not None:
                if tuple(latent.shape[-2:]) != (source_h, source_w):
                    batch, channels, time, old_h, old_w = latent.shape
                    latent = F.interpolate(
                        latent.reshape(batch * time, channels, old_h, old_w),
                        size=(source_h, source_w),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(batch, channels, time, source_h, source_w)
                copy["latent"] = latent[
                    :, :, :, row : row + height, col : col + width
                ].contiguous()
            cropped.append(copy)
        if cropped:
            updated["minimax_keyframes"] = cropped
        output.append([tensor, updated])
    return output


def _build_guider(model, positive, negative, cfg: float):
    guider = comfy.samplers.CFGGuider(model)
    if negative is None:
        guider.inner_set_conds({"positive": positive})
    else:
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
    return guider


def sample_piece(
    piece,
    positive,
    model,
    noise,
    sampler,
    sigmas,
    negative,
    cfg,
    *,
    prepared_noise=None,
):
    latent = dict(piece)
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    latent["samples"] = latent_image
    guider = _build_guider(model, positive, negative, cfg)
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, {}
    )
    sample_noise = (
        noise.generate_noise(latent) if prepared_noise is None else prepared_noise
    )
    output = guider.sample(
        sample_noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=noise.seed,
    )
    return output.to(comfy.model_management.intermediate_device())


def _spatial_resample(
    chunk_video,
    chunk_audio,
    conditioning,
    plan,
    model,
    noise,
    sampler,
    sigmas,
    negative,
    cfg,
    *,
    chunk_noise_video=None,
    chunk_noise_audio=None,
    chunk_temporal_mask=None,
    chunk_inherited_video_mask=None,
    audio_sampling_policy="locked_input_audio",
    rebind_shape_bound_sampler=False,
):
    tile_w = plan["tile_width"] // VAE_DOWNSAMPLE
    tile_h = plan["tile_height"] // VAE_DOWNSAMPLE
    overlap_w = plan["spatial_overlap"] // VAE_DOWNSAMPLE
    overlap_h = plan["spatial_overlap"] // VAE_DOWNSAMPLE
    fade = plan["spatial_fade"] // VAE_DOWNSAMPLE
    minimum = plan["minimum_tile_size"] // VAE_DOWNSAMPLE
    _, channels, time, height, width = chunk_video.shape
    if chunk_noise_video is not None and tuple(chunk_noise_video.shape) != tuple(
        chunk_video.shape
    ):
        raise ValueError(
            "global video noise must match the current upscaled temporal chunk: "
            f"noise={tuple(chunk_noise_video.shape)}, video={tuple(chunk_video.shape)}"
        )
    if audio_sampling_policy not in AUDIO_SAMPLING_POLICIES:
        raise ValueError(
            "unknown second-pass audio sampling policy: "
            f"{audio_sampling_policy!r}"
        )
    if chunk_noise_audio is not None and tuple(chunk_noise_audio.shape) != tuple(
        chunk_audio.shape
    ):
        raise ValueError(
            "global audio noise must match the current temporal audio slice: "
            f"noise={tuple(chunk_noise_audio.shape)}, audio={tuple(chunk_audio.shape)}"
        )
    if chunk_temporal_mask is not None:
        expected_mask_shape = (1, 1, time, 1, 1)
        if tuple(chunk_temporal_mask.shape) != expected_mask_shape:
            raise ValueError(
                "temporal ownership mask must match the current temporal chunk: "
                f"expected={expected_mask_shape}, actual={tuple(chunk_temporal_mask.shape)}"
            )
    if chunk_inherited_video_mask is not None:
        expected_inherited_shape = (1, 1, time, height, width)
        if tuple(chunk_inherited_video_mask.shape) != expected_inherited_shape:
            raise ValueError(
                "inherited video mask must match the current upscaled temporal chunk: "
                f"expected={expected_inherited_shape}, "
                f"actual={tuple(chunk_inherited_video_mask.shape)}"
            )
        if not torch.isfinite(chunk_inherited_video_mask).all():
            raise ValueError("inherited video mask contains NaN or Inf")
        if (
            float(chunk_inherited_video_mask.amin().item()) < 0.0
            or float(chunk_inherited_video_mask.amax().item()) > 1.0
        ):
            raise ValueError("inherited video mask values must stay within [0,1]")
    rows, cols, row_sizes, col_sizes, row_overlaps, col_overlaps = compute_spatial_grid(
        height,
        width,
        tile_h,
        tile_w,
        overlap_h,
        overlap_w,
        minimum,
    )
    accumulated = chunk_video.clone()
    for row_index, row in enumerate(rows):
        for col_index, col in enumerate(cols):
            tile_height = row_sizes[row_index]
            tile_width = col_sizes[col_index]
            row_overlap = row_overlaps[row_index]
            col_overlap = col_overlaps[col_index]
            tile = chunk_video[
                :, :, :, row : row + tile_height, col : col + tile_width
            ].clone()
            if col_index and col_overlap:
                tile[..., :col_overlap] = accumulated[
                    :, :, :, row : row + tile_height, col : col + col_overlap
                ]
            if row_index and row_overlap:
                tile[..., :row_overlap, :] = accumulated[
                    :, :, :, row : row + row_overlap, col : col + tile_width
                ]
            video_mask = spatial_fade_mask(
                tile_height,
                tile_width,
                row_overlap,
                col_overlap,
                bool(row_index),
                bool(col_index),
                fade,
                fade,
            ).to(device=tile.device)[None, None, None]
            if chunk_temporal_mask is not None:
                video_mask = video_mask * chunk_temporal_mask.to(
                    device=tile.device, dtype=video_mask.dtype
                )
            if chunk_inherited_video_mask is not None:
                inherited_tile_mask = chunk_inherited_video_mask[
                    :, :, :, row : row + tile_height, col : col + tile_width
                ]
                video_mask = video_mask * inherited_tile_mask.to(
                    device=tile.device, dtype=video_mask.dtype
                )
            joint_audio = audio_sampling_policy == "joint_av_preserve_input"
            audio_mask = (
                torch.ones_like(chunk_audio)
                if joint_audio
                else torch.zeros_like(chunk_audio)
            )
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_audio)),
                "noise_mask": comfy.nested_tensor.NestedTensor(
                    (video_mask, audio_mask)
                ),
            }
            piece_sampler = (
                rebind_dual_clock_sampler(model, piece, sampler)
                if rebind_shape_bound_sampler
                else sampler
            )
            prepared_noise = None
            if chunk_noise_video is not None:
                tile_noise = chunk_noise_video[
                    :, :, :, row : row + tile_height, col : col + tile_width
                ].contiguous()
                if joint_audio:
                    if chunk_noise_audio is None:
                        raise ValueError(
                            "joint AV second-pass sampling requires the matching global "
                            "audio-noise slice"
                        )
                    audio_noise = chunk_noise_audio.to(
                        device=tile_noise.device,
                        dtype=chunk_audio.dtype,
                    ).contiguous()
                else:
                    audio_noise = torch.zeros(
                        chunk_audio.shape,
                        dtype=chunk_audio.dtype,
                        device=tile_noise.device,
                    )
                prepared_noise = comfy.nested_tensor.NestedTensor(
                    (tile_noise, audio_noise)
                )
            tile_conditioning = crop_conditioning(
                conditioning,
                height,
                width,
                row,
                col,
                tile_height,
                tile_width,
            )
            sampled = sample_piece(
                piece,
                tile_conditioning,
                model,
                noise,
                piece_sampler,
                sigmas,
                negative,
                cfg,
                prepared_noise=prepared_noise,
            ).tensors[0]
            region = accumulated[
                :, :, :, row : row + tile_height, col : col + tile_width
            ].clone()
            if col_index and col_overlap:
                values = torch.linspace(
                    0, 1, col_overlap, device=region.device, dtype=region.dtype
                )
                weights = _blend_weights(values, plan["overlap_blend"])
                region[..., :col_overlap] = (
                    region[..., :col_overlap] * (1 - weights[None, None, None, None])
                    + sampled[..., :col_overlap]
                    * weights[None, None, None, None]
                )
            if row_index and row_overlap:
                values = torch.linspace(
                    0, 1, row_overlap, device=region.device, dtype=region.dtype
                )
                weights = _blend_weights(values, plan["overlap_blend"])
                region[..., :row_overlap, :] = (
                    region[..., :row_overlap, :]
                    * (1 - weights[None, None, None, :, None])
                    + sampled[..., :row_overlap, :]
                    * weights[None, None, None, :, None]
                )
            band = torch.zeros(
                (1, 1, 1, tile_height, tile_width),
                dtype=torch.bool,
                device=region.device,
            )
            if col_index and col_overlap:
                band[..., :col_overlap] = True
            if row_index and row_overlap:
                band[..., :row_overlap, :] = True
            accumulated[
                :, :, :, row : row + tile_height, col : col + tile_width
            ] = torch.where(band, region, sampled)
    return accumulated, {
        "rows": rows,
        "cols": cols,
        "row_sizes": row_sizes,
        "col_sizes": col_sizes,
        "row_overlaps": row_overlaps,
        "col_overlaps": col_overlaps,
        "noise_policy": (
            GLOBAL_NOISE_POLICY
            if chunk_noise_video is not None
            else "legacy_noise_object_per_piece"
        ),
        "audio_sampling_policy": audio_sampling_policy,
        "inherited_video_mask_applied": chunk_inherited_video_mask is not None,
        "video_mask_combine": (
            "inherited_times_spatial_ownership_times_temporal_ownership"
            if chunk_inherited_video_mask is not None
            else "spatial_ownership_times_temporal_ownership"
        ),
    }


def _append_video(accumulated, chunk, start_token: int):
    if accumulated is None:
        return chunk
    total = max(accumulated.shape[2], start_token + chunk.shape[2])
    output = torch.zeros(
        (1, accumulated.shape[1], total, accumulated.shape[3], accumulated.shape[4]),
        device=accumulated.device,
        dtype=accumulated.dtype,
    )
    output[:, :, : accumulated.shape[2]] = accumulated
    overlap = max(0, accumulated.shape[2] - start_token)
    overlap = min(overlap, chunk.shape[2])
    if overlap:
        output[:, :, start_token : start_token + overlap] = _crossfade(
            output[:, :, start_token : start_token + overlap].clone(),
            chunk[:, :, :overlap],
            2,
        )
    if chunk.shape[2] > overlap:
        output[:, :, start_token + overlap : start_token + chunk.shape[2]] = chunk[
            :, :, overlap:
        ]
    return output


def _temporal_overlap_mask(time: int, overlap: int, *, dtype, device):
    """Build a read-only guard followed by a smooth takeover inside the overlap."""

    overlap = max(0, min(int(overlap), int(time)))
    mask = torch.ones((1, 1, int(time), 1, 1), dtype=dtype, device=device)
    if not overlap:
        return mask, 0, 0
    locked = max(1, overlap // 2)
    transition = overlap - locked
    mask[:, :, :locked] = 0
    if transition:
        values = torch.linspace(
            0,
            1,
            transition + 1,
            dtype=dtype,
            device=device,
        )[1:]
        mask[:, :, locked:overlap, 0, 0] = _blend_weights(
            values, "smoothstep"
        )
    return mask, locked, transition


def _append_video_guarded_overlap(
    accumulated,
    chunk,
    start_token: int,
    locked_overlap_tokens: int,
):
    """Keep the guard exact, publish the sampled bridge, then append new tokens."""

    if accumulated is None:
        return chunk, 0, 0
    overlap = max(0, int(accumulated.shape[2]) - int(start_token))
    overlap = min(overlap, int(chunk.shape[2]))
    if start_token > accumulated.shape[2]:
        raise ValueError("temporal segment starts after the published video")
    locked = max(0, min(int(locked_overlap_tokens), overlap))
    transition = overlap - locked
    published = accumulated
    if transition:
        published = accumulated.clone()
        published[:, :, start_token + locked : start_token + overlap] = chunk[
            :, :, locked:overlap
        ]
    if overlap >= chunk.shape[2]:
        return published, overlap, transition
    return (
        torch.cat((published, chunk[:, :, overlap:]), dim=2),
        overlap,
        transition,
    )


def build_chunked_two_pass_plan(
    model_name: str,
    target_width: int,
    target_height: int,
    temporal_chunk_frames: int,
    temporal_overlap_frames: int,
    anchor_strength: float,
    tile_width: int,
    tile_height: int,
    spatial_overlap: int,
    spatial_fade: int,
    minimum_tile_size: int,
    overlap_blend: str,
    precision: str,
    release_policy: str,
    spatial_strategy: str = "full_frame_safe",
) -> tuple[dict[str, Any], str]:
    integer_fields = {
        "target_width": target_width,
        "target_height": target_height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "spatial_overlap": spatial_overlap,
        "spatial_fade": spatial_fade,
        "minimum_tile_size": minimum_tile_size,
    }
    if any(int(value) <= 0 for key, value in integer_fields.items() if key not in {"spatial_overlap", "spatial_fade"}):
        raise ValueError("target and tile sizes must be positive")
    if any(int(value) % GRID_PIXELS for value in integer_fields.values()):
        raise ValueError("target, tile, overlap and fade values must be multiples of 32 pixels")
    if temporal_chunk_frames % 17 or temporal_overlap_frames % 17:
        raise ValueError("temporal chunk and overlap must be multiples of 17 frames")
    if temporal_overlap_frames >= temporal_chunk_frames:
        raise ValueError("temporal overlap must be smaller than the chunk")
    if spatial_overlap >= min(tile_width, tile_height):
        raise ValueError("spatial overlap must be smaller than each tile axis")
    if spatial_fade > spatial_overlap:
        raise ValueError("spatial fade cannot exceed overlap")
    if minimum_tile_size > min(tile_width, tile_height):
        raise ValueError("minimum_tile_size cannot exceed tile dimensions")
    if not 0 <= float(anchor_strength) <= 1:
        raise ValueError("anchor_strength must be in [0,1]")
    if overlap_blend not in {"linear", "smoothstep"}:
        raise ValueError("overlap_blend must be linear or smoothstep")
    if spatial_strategy not in {"full_frame_safe", "independent_tiles_exp"}:
        raise ValueError(
            "spatial_strategy must be full_frame_safe or independent_tiles_exp"
        )
    plan = {
        "schema": PLAN_SCHEMA_V1,
        "model_name": str(model_name),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "temporal_chunk_frames": int(temporal_chunk_frames),
        "temporal_overlap_frames": int(temporal_overlap_frames),
        "anchor_strength": float(anchor_strength),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "spatial_overlap": int(spatial_overlap),
        "spatial_fade": int(spatial_fade),
        "minimum_tile_size": int(minimum_tile_size),
        "overlap_blend": overlap_blend,
        "precision": precision,
        "release_policy": release_policy,
        "spatial_strategy": spatial_strategy,
        "spatial_quality_boundary": (
            "full_frame_preserves_global_h3_spatial_context"
            if spatial_strategy == "full_frame_safe"
            else "experimental_independent_canvases_may_diverge"
        ),
        "audio_policy": "exact_input_tensor_passthrough",
        "pixel_limit_policy": "no_project_pixel_area_limit",
    }
    return plan, _json(plan)


def build_chunked_two_pass_global_noise_plan(
    model_name: str,
    target_width: int,
    target_height: int,
    temporal_chunk_frames: int,
    temporal_overlap_frames: int,
    anchor_strength: float,
    tile_width: int,
    tile_height: int,
    spatial_overlap: int,
    spatial_fade: int,
    minimum_tile_size: int,
    overlap_blend: str,
    precision: str,
    release_policy: str,
    spatial_strategy: str = "full_frame_safe",
    temporal_strategy: str = "full_clip_safe",
) -> tuple[dict[str, Any], str]:
    """Build the opt-in v2 plan without changing the released v1 contract."""

    plan, _ = build_chunked_two_pass_plan(
        model_name,
        target_width,
        target_height,
        temporal_chunk_frames,
        temporal_overlap_frames,
        anchor_strength,
        tile_width,
        tile_height,
        spatial_overlap,
        spatial_fade,
        minimum_tile_size,
        overlap_blend,
        precision,
        release_policy,
        spatial_strategy,
    )
    if temporal_strategy not in {"full_clip_safe", "guarded_overlap_exp"}:
        raise ValueError(
            "temporal_strategy must be full_clip_safe or guarded_overlap_exp"
        )
    temporal_merge_policy = (
        TEMPORAL_FULL_CLIP_POLICY
        if temporal_strategy == "full_clip_safe"
        else TEMPORAL_OWNERSHIP_POLICY
    )
    temporal_overlap_policy = (
        "sample the complete source timeline as one H3 Transformer trajectory; "
        "temporal chunk and overlap values are ignored"
        if temporal_strategy == "full_clip_safe"
        else (
            "replace the next target overlap with the exact previous output; keep "
            "its first half read-only, progressively transfer the second half to "
            "the new trajectory, then append only newly owned tokens"
        )
    )
    plan.update(
        {
            "schema": PLAN_SCHEMA_GLOBAL_NOISE_V2,
            "noise_policy": GLOBAL_NOISE_POLICY,
            "global_noise_scope": "target_video_latent_only",
            "audio_noise_policy": "zero_per_piece_with_existing_zero_noise_mask",
            "sampler_boundary": (
                "coordinates the external initial noise exactly; ancestral or SDE "
                "samplers may still add independent internal per-step noise"
            ),
            "temporal_strategy": temporal_strategy,
            "temporal_merge_policy": temporal_merge_policy,
            "temporal_overlap_policy": temporal_overlap_policy,
            "compatibility": "old v1 plan and executor inputs remain unchanged",
        }
    )
    return plan, _json(plan)


def build_chunked_two_pass_low_sigma_plan(
    model_name: str,
    target_width: int,
    target_height: int,
    temporal_chunk_frames: int,
    temporal_overlap_frames: int,
    anchor_strength: float,
    tile_width: int,
    tile_height: int,
    spatial_overlap: int,
    spatial_fade: int,
    minimum_tile_size: int,
    overlap_blend: str,
    precision: str,
    release_policy: str,
    spatial_strategy: str = "full_frame_safe",
    temporal_strategy: str = "full_clip_safe",
    second_pass_audio_policy: str = "joint_av_preserve_input",
) -> tuple[dict[str, Any], str]:
    """Build the corrected full-first-pass plus low-noise-refine contract.

    This remains append-only: v1 and v2 retain their exact schemas and behavior.  The
    sampler schedule stays external so users can inspect and replace it; the published
    workflow uses ComfyUI BasicScheduler(simple, 3 steps, denoise 0.30).
    """

    if second_pass_audio_policy not in AUDIO_SAMPLING_POLICIES:
        raise ValueError(
            "second_pass_audio_policy must be joint_av_preserve_input or "
            "locked_input_audio"
        )
    plan, _ = build_chunked_two_pass_global_noise_plan(
        model_name,
        target_width,
        target_height,
        temporal_chunk_frames,
        temporal_overlap_frames,
        anchor_strength,
        tile_width,
        tile_height,
        spatial_overlap,
        spatial_fade,
        minimum_tile_size,
        overlap_blend,
        precision,
        release_policy,
        spatial_strategy,
        temporal_strategy,
    )
    plan.update(
        {
            "schema": PLAN_SCHEMA_LOW_SIGMA_V3,
            "first_pass_contract": "complete_trajectory_to_zero_before_upscale",
            "recommended_refine": {
                "scheduler": "simple",
                "steps": 3,
                "denoise": 0.30,
                "upstream_readme_max_denoise": 0.40,
            },
            "second_pass_audio_policy": second_pass_audio_policy,
            "audio_noise_policy": (
                "one_global_audio_noise_reused_for_joint_model_context_then_discard_output"
                if second_pass_audio_policy == "joint_av_preserve_input"
                else "zero_per_piece_with_zero_audio_noise_mask"
            ),
            "final_audio_policy": "return_exact_first_pass_audio_tensor",
            "compatibility": (
                "append_only_v3; old v1/v2 plans, executor inputs, and workflows remain "
                "unchanged"
            ),
        }
    )
    return plan, _json(plan)


def build_chunked_two_pass_masked_low_sigma_plan(
    model_name: str,
    target_width: int,
    target_height: int,
    temporal_chunk_frames: int,
    temporal_overlap_frames: int,
    anchor_strength: float,
    tile_width: int,
    tile_height: int,
    spatial_overlap: int,
    spatial_fade: int,
    minimum_tile_size: int,
    overlap_blend: str,
    precision: str,
    release_policy: str,
    spatial_strategy: str = "full_frame_safe",
    temporal_strategy: str = "full_clip_safe",
    second_pass_audio_policy: str = "joint_av_preserve_input",
    video_mask_policy: str = "inherit_required",
) -> tuple[dict[str, Any], str]:
    """Build an append-only low-sigma route that preserves the first-pass VIDEO mask."""

    if video_mask_policy not in VIDEO_MASK_POLICIES:
        raise ValueError(
            "video_mask_policy must be inherit_required, "
            "inherit_if_present_else_generate_all, or disabled"
        )
    plan, _ = build_chunked_two_pass_low_sigma_plan(
        model_name,
        target_width,
        target_height,
        temporal_chunk_frames,
        temporal_overlap_frames,
        anchor_strength,
        tile_width,
        tile_height,
        spatial_overlap,
        spatial_fade,
        minimum_tile_size,
        overlap_blend,
        precision,
        release_policy,
        spatial_strategy,
        temporal_strategy,
        second_pass_audio_policy,
    )
    plan.update(
        {
            "schema": PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
            "video_mask_policy": video_mask_policy,
            "video_mask_resize": "nearest_exact_spatial_only",
            "video_mask_temporal_policy": (
                "exact_latent_tokens_or_verified_static_expand; no temporal interpolation"
            ),
            "video_mask_combine": (
                "inherited_times_spatial_ownership_times_temporal_ownership"
            ),
            "compatibility": (
                "append_only_v4; old v1/v2/v3 plans, node IDs, widget orders, and "
                "workflows remain unchanged"
            ),
        }
    )
    return plan, _json(plan)


def _nested_parts(value, *, name: str):
    if not getattr(value, "is_nested", False):
        raise ValueError(f"{name} must be a nested H3 video/audio tensor")
    tensors = getattr(value, "tensors", None)
    if tensors is None:
        tensors = value.unbind()
    tensors = tuple(tensors)
    if len(tensors) != 2:
        raise ValueError(f"{name} must contain exactly video and audio tensors")
    return tensors


def _resize_video_mask_spatial_only(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize only H/W while preserving every H3 latent-time token exactly."""

    if mask.ndim != 5 or int(mask.shape[1]) != 1:
        raise ValueError("normalized MiniMax H3 video mask must be [B,1,T,H,W]")
    if tuple(mask.shape[-2:]) == (int(target_height), int(target_width)):
        return mask
    batch, _channels, time, height, width = mask.shape
    work = mask.permute(0, 2, 1, 3, 4).reshape(batch * time, 1, height, width)
    resized = F.interpolate(
        work.to(torch.float32),
        size=(int(target_height), int(target_width)),
        mode="nearest-exact",
    )
    return resized.reshape(batch, time, 1, target_height, target_width).permute(
        0, 2, 1, 3, 4
    )


def _normalize_inherited_video_mask(
    latent: dict,
    video: torch.Tensor,
    *,
    policy: str,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Normalize an inherited H3 mask without interpolating its time dimension."""

    if policy not in VIDEO_MASK_POLICIES:
        raise ValueError(f"unknown video mask policy: {policy!r}")
    report: dict[str, Any] = {
        "policy": policy,
        "status": "disabled" if policy == "disabled" else "missing",
    }
    if policy == "disabled":
        return None, report

    nested_mask = latent.get("noise_mask") if isinstance(latent, dict) else None
    if nested_mask is None:
        if policy == "inherit_required":
            raise ValueError(
                "masked low-sigma v4 requires a nested first-pass noise_mask"
            )
        report["status"] = "missing_generate_all_fallback"
        return None, report
    video_mask, _audio_mask = _nested_parts(nested_mask, name="first-pass noise_mask")
    if not isinstance(video_mask, torch.Tensor):
        raise ValueError("first-pass video noise_mask must be a torch.Tensor")
    report["source_shape"] = list(video_mask.shape)

    if video_mask.ndim == 2:
        work = video_mask[None, None, None]
    elif video_mask.ndim == 3:
        work = video_mask[None, None]
    elif video_mask.ndim == 4:
        if int(video_mask.shape[1]) == 1:
            # Core SetLatentNoiseMask stores image/video masks as [frames,1,H,W].
            work = video_mask.permute(1, 0, 2, 3)[None]
        elif int(video_mask.shape[0]) == int(video.shape[0]):
            work = video_mask[:, None]
        else:
            raise ValueError(
                "four-dimensional first-pass video mask must be [frames,1,H,W] "
                "or [B,T,H,W]"
            )
    elif video_mask.ndim == 5:
        work = video_mask
    else:
        raise ValueError("first-pass video noise_mask must have rank 2 through 5")

    if int(work.shape[0]) == 1 and int(video.shape[0]) > 1:
        work = work.expand(int(video.shape[0]), -1, -1, -1, -1)
    if int(work.shape[0]) != int(video.shape[0]):
        raise ValueError(
            "first-pass video mask batch does not match the H3 video latent"
        )
    if int(work.shape[1]) == 1:
        channel_policy = "single_channel"
    else:
        work = work.amax(dim=1, keepdim=True)
        channel_policy = "any_generate_max"

    work = work.to(torch.float32)
    if not torch.isfinite(work).all():
        raise ValueError("first-pass video noise_mask contains NaN or Inf")
    if float(work.amin().item()) < 0.0 or float(work.amax().item()) > 1.0:
        raise ValueError("first-pass video noise_mask values must stay within [0,1]")

    source_time = int(work.shape[2])
    target_time = int(video.shape[2])
    if source_time == target_time:
        temporal_policy = "exact_latent_tokens"
    elif source_time == 1:
        work = work.expand(-1, -1, target_time, -1, -1)
        temporal_policy = "single_frame_expanded"
    else:
        reference = work[:, :, :1]
        if not torch.equal(work, reference.expand_as(work)):
            raise ValueError(
                "first-pass video mask time does not match H3 latent tokens and is "
                "not static; temporal interpolation is forbidden"
            )
        work = reference.expand(-1, -1, target_time, -1, -1)
        temporal_policy = "verified_static_then_expanded"

    work = _resize_video_mask_spatial_only(
        work,
        int(video.shape[-2]),
        int(video.shape[-1]),
    ).contiguous()
    pooled = F.max_pool3d(work, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    protected = float((work <= 1.0e-3).to(torch.float32).mean().item())
    editable = float((work >= 1.0 - 1.0e-3).to(torch.float32).mean().item())
    report.update(
        {
            "status": "normalized",
            "normalized_shape": list(work.shape),
            "channel_policy": channel_policy,
            "temporal_policy": temporal_policy,
            "spatial_policy": "nearest_exact_to_source_latent_grid",
            "minimum": float(work.amin().item()),
            "maximum": float(work.amax().item()),
            "protected_fraction": protected,
            "editable_fraction": editable,
            "partial_fraction": max(0.0, 1.0 - protected - editable),
            "h3_2x2_pooled_editable_fraction": float(
                (pooled >= 1.0 - 1.0e-3).to(torch.float32).mean().item()
            ),
        }
    )
    return work, report


def _build_global_target_av_noise(noise, latent, video, audio, plan):
    target_height = int(plan["target_height"]) // VAE_DOWNSAMPLE
    target_width = int(plan["target_width"]) // VAE_DOWNSAMPLE
    expected_shape = (
        int(video.shape[0]),
        int(video.shape[1]),
        int(video.shape[2]),
        target_height,
        target_width,
    )
    template_video = torch.zeros(expected_shape, dtype=video.dtype, device="cpu")
    template_audio = torch.zeros(tuple(audio.shape), dtype=audio.dtype, device="cpu")
    noise_template = {
        "samples": comfy.nested_tensor.NestedTensor(
            (template_video, template_audio)
        )
    }
    if "batch_index" in latent:
        noise_template["batch_index"] = latent["batch_index"]
    generated = noise.generate_noise(noise_template)
    video_noise, audio_noise = _nested_parts(generated, name="generated global noise")
    if tuple(video_noise.shape) != expected_shape:
        raise ValueError(
            "the connected NOISE source did not return the requested full target video "
            f"shape: expected={expected_shape}, actual={tuple(video_noise.shape)}"
        )
    if not torch.isfinite(video_noise).all():
        raise ValueError("the connected NOISE source returned non-finite video noise")
    if tuple(audio_noise.shape) != tuple(audio.shape):
        raise ValueError(
            "the connected NOISE source did not return the requested audio shape: "
            f"expected={tuple(audio.shape)}, actual={tuple(audio_noise.shape)}"
        )
    if not torch.isfinite(audio_noise).all():
        raise ValueError("the connected NOISE source returned non-finite audio noise")
    report = {
        "policy": GLOBAL_NOISE_POLICY,
        "generate_noise_calls": 1,
        "target_video_noise_shape": list(expected_shape),
        "target_video_noise_dtype": str(video_noise.dtype),
        "target_video_noise_device": str(video_noise.device),
        "seed": getattr(noise, "seed", None),
        "target_audio_noise_shape": list(audio_noise.shape),
        "audio_noise": "generated_once_full_timeline",
        "full_noise_bytes": int(video_noise.numel() * video_noise.element_size()),
    }
    report["full_noise_bytes"] += int(
        audio_noise.numel() * audio_noise.element_size()
    )
    return video_noise.contiguous(), audio_noise.contiguous(), report


def _build_global_target_video_noise(noise, latent, video, audio, plan):
    """Preserve the released v2 zero-audio-noise contract."""

    video_noise, _audio_noise, report = _build_global_target_av_noise(
        noise, latent, video, audio, plan
    )
    report = dict(report)
    report["audio_noise"] = "zero_per_piece"
    report["full_noise_bytes"] = int(
        video_noise.numel() * video_noise.element_size()
    )
    return video_noise, report


def execute_chunked_two_pass_upscale(
    model,
    conditioning,
    latent,
    noise,
    sampler,
    sigmas,
    plan,
    negative=None,
    cfg: float = 1.0,
):
    if not isinstance(plan, dict) or plan.get("schema") not in {
        PLAN_SCHEMA_V1,
        PLAN_SCHEMA_GLOBAL_NOISE_V2,
        PLAN_SCHEMA_LOW_SIGMA_V3,
        PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
    }:
        raise ValueError("plan must come from the T8 Chunked Two-Pass Plan node")
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("expected a nested MiniMax H3 AV latent")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError("unexpected MiniMax H3 AV latent shapes")
    if video.shape[0] != 1:
        raise ValueError("chunked two-pass currently supports batch 1")

    inherited_video_mask = None
    target_inherited_video_mask = None
    inherited_mask_report = None
    if plan["schema"] == PLAN_SCHEMA_MASKED_LOW_SIGMA_V4:
        inherited_video_mask, inherited_mask_report = (
            _normalize_inherited_video_mask(
                latent,
                video,
                policy=plan.get("video_mask_policy", "inherit_required"),
            )
        )
        if inherited_video_mask is not None:
            target_inherited_video_mask = _resize_video_mask_spatial_only(
                inherited_video_mask,
                int(plan["target_height"]) // VAE_DOWNSAMPLE,
                int(plan["target_width"]) // VAE_DOWNSAMPLE,
            ).contiguous()
            inherited_mask_report = dict(inherited_mask_report)
            inherited_mask_report.update(
                {
                    "target_shape": list(target_inherited_video_mask.shape),
                    "target_spatial_policy": "nearest_exact_spatial_only",
                }
            )

    global_video_noise = None
    global_audio_noise = None
    global_noise_report = None
    if plan["schema"] in {
        PLAN_SCHEMA_LOW_SIGMA_V3,
        PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
    }:
        (
            global_video_noise,
            global_audio_noise,
            global_noise_report,
        ) = _build_global_target_av_noise(noise, latent, video, audio, plan)
    elif plan["schema"] == PLAN_SCHEMA_GLOBAL_NOISE_V2:
        global_video_noise, global_noise_report = _build_global_target_video_noise(
            noise, latent, video, audio, plan
        )

    frame_count = frames_for_tokens(int(video.shape[2]))
    if (
        plan["schema"]
        in {
            PLAN_SCHEMA_GLOBAL_NOISE_V2,
            PLAN_SCHEMA_LOW_SIGMA_V3,
            PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
        }
        and plan.get("temporal_strategy", "full_clip_safe") == "full_clip_safe"
    ):
        segments = [(0, 0, int(video.shape[2]), frame_count)]
    else:
        segments, frame_count = compute_temporal_segments(
            int(video.shape[2]),
            int(plan["temporal_chunk_frames"]),
            int(plan["temporal_overlap_frames"]),
        )
    accumulated = None
    segment_reports = []
    tile_reports = []
    for index, (start_token, start_frame, end_token, end_frame) in enumerate(segments):
        chunk_video = video[:, :, start_token:end_token].contiguous()
        audio_start = round(start_frame * FRAME_RESCALE)
        audio_end = min(audio.shape[-1], round(end_frame * FRAME_RESCALE))
        chunk_audio = audio[..., audio_start:audio_end].contiguous()
        chunk_latent = {
            "samples": comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))
        }
        chunk_inherited_source_mask = None
        if inherited_video_mask is not None:
            chunk_inherited_source_mask = inherited_video_mask[
                :, :, start_token:end_token
            ].contiguous()
            chunk_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (chunk_inherited_source_mask, torch.ones_like(chunk_audio))
            )
        upscaled, _, _, upscale_report = learned_upscale_h3_av_latent(
            chunk_latent,
            plan["model_name"],
            "target_dimensions",
            2.0,
            1.0,
            int(plan["target_width"]),
            int(plan["target_height"]),
            "honor_dimensions_exp",
            2.0,
            plan["precision"],
            plan["release_policy"],
        )
        chunk_video = upscaled["samples"].tensors[0]
        chunk_inherited_target_mask = None
        if chunk_inherited_source_mask is not None:
            upscaled_mask = upscaled.get("noise_mask")
            if upscaled_mask is None:
                raise RuntimeError(
                    "learned latent upscaler dropped the inherited H3 noise_mask"
                )
            chunk_inherited_target_mask, _upscaled_audio_mask = _nested_parts(
                upscaled_mask,
                name="upscaled noise_mask",
            )
            expected_target_mask = target_inherited_video_mask[
                :, :, start_token:end_token
            ]
            if tuple(chunk_inherited_target_mask.shape) != tuple(
                expected_target_mask.shape
            ):
                raise RuntimeError(
                    "learned latent upscaler changed the inherited mask geometry: "
                    f"expected={tuple(expected_target_mask.shape)}, "
                    f"actual={tuple(chunk_inherited_target_mask.shape)}"
                )
            if not torch.equal(
                chunk_inherited_target_mask.to(
                    device=expected_target_mask.device,
                    dtype=expected_target_mask.dtype,
                ),
                expected_target_mask,
            ):
                raise RuntimeError(
                    "learned latent upscaler changed inherited mask values"
                )
            chunk_inherited_target_mask = expected_target_mask
        chunk_conditioning = reanchor_conditioning(
            conditioning,
            start_frame,
            end_frame,
            tuple(chunk_video.shape[-2:]),
        )
        temporal_mask = None
        locked_overlap_tokens = 0
        transition_overlap_tokens = 0
        uses_owned_overlap = (
            plan.get("temporal_merge_policy") == TEMPORAL_OWNERSHIP_POLICY
        )
        if index and accumulated is not None and uses_owned_overlap:
            locked_overlap_tokens = max(0, int(accumulated.shape[2]) - start_token)
            locked_overlap_tokens = min(
                locked_overlap_tokens, int(chunk_video.shape[2])
            )
            if locked_overlap_tokens:
                chunk_video = chunk_video.clone()
                chunk_video[:, :, :locked_overlap_tokens] = accumulated[
                    :, :, start_token : start_token + locked_overlap_tokens
                ]
                temporal_mask, locked_overlap_tokens, transition_overlap_tokens = (
                    _temporal_overlap_mask(
                        int(chunk_video.shape[2]),
                        locked_overlap_tokens,
                        dtype=chunk_video.dtype,
                        device=chunk_video.device,
                    )
                )
        elif index and accumulated is not None:
            chunk_conditioning = anchor_conditioning(
                chunk_conditioning,
                accumulated,
                start_frame,
                plan["anchor_strength"],
            )
        spatial_plan = plan
        if plan.get("spatial_strategy", "independent_tiles_exp") == "full_frame_safe":
            spatial_plan = dict(plan)
            spatial_plan.update(
                {
                    "tile_width": int(plan["target_width"]),
                    "tile_height": int(plan["target_height"]),
                    "spatial_overlap": 0,
                    "spatial_fade": 0,
                    "minimum_tile_size": min(
                        int(plan["target_width"]), int(plan["target_height"])
                    ),
                }
            )
        chunk_output, tile_report = _spatial_resample(
            chunk_video,
            chunk_audio,
            chunk_conditioning,
            spatial_plan,
            model,
            noise,
            sampler,
            sigmas,
            negative,
            cfg,
            chunk_noise_video=(
                global_video_noise[:, :, start_token:end_token]
                if global_video_noise is not None
                else None
            ),
            chunk_noise_audio=(
                global_audio_noise[..., audio_start:audio_end]
                if global_audio_noise is not None
                else None
            ),
            chunk_temporal_mask=temporal_mask,
            chunk_inherited_video_mask=chunk_inherited_target_mask,
            audio_sampling_policy=plan.get(
                "second_pass_audio_policy", "locked_input_audio"
            ),
            rebind_shape_bound_sampler=(
                plan["schema"]
                in {PLAN_SCHEMA_LOW_SIGMA_V3, PLAN_SCHEMA_MASKED_LOW_SIGMA_V4}
            ),
        )
        if uses_owned_overlap:
            (
                accumulated,
                merged_overlap_tokens,
                merged_transition_tokens,
            ) = _append_video_guarded_overlap(
                accumulated,
                chunk_output,
                start_token,
                locked_overlap_tokens,
            )
            if merged_overlap_tokens != (
                locked_overlap_tokens + transition_overlap_tokens
            ) or merged_transition_tokens != transition_overlap_tokens:
                raise RuntimeError(
                    "temporal overlap profile changed before ownership merge: "
                    f"locked={locked_overlap_tokens}, "
                    f"transition={transition_overlap_tokens}, "
                    f"merged={merged_overlap_tokens}, "
                    f"merged_transition={merged_transition_tokens}"
                )
        else:
            accumulated = _append_video(accumulated, chunk_output, start_token)
        segment_reports.append(
            {
                "index": index,
                "pixel_frames": [start_frame, end_frame],
                "video_tokens": [start_token, end_token],
                "audio_tokens_read_only": [audio_start, audio_end],
                "locked_overlap_tokens": locked_overlap_tokens,
                "transition_overlap_tokens": transition_overlap_tokens,
                "published_new_tokens": int(chunk_output.shape[2])
                - locked_overlap_tokens
                - transition_overlap_tokens,
                "upscale": json.loads(upscale_report),
            }
        )
        tile_report["segment_index"] = index
        tile_reports.append(tile_report)

    output = {
        "samples": comfy.nested_tensor.NestedTensor((accumulated, audio))
    }
    report = {
        "schema": "t8.minimax_h3.chunked_two_pass.execution.v1",
        "status": "completed",
        "source_frame_count": frame_count,
        "segment_count": len(segments),
        "segments": segment_reports,
        "tiles": tile_reports,
        "audio_preserved_by_identity": output["samples"].tensors[1] is audio,
        "audio_resampled": False,
        "spatial_strategy": plan.get("spatial_strategy", "independent_tiles_exp"),
        "temporal_strategy": plan.get("temporal_strategy", "legacy_chunked"),
        "pixel_limit_policy": "no_project_pixel_area_limit",
        "noise_policy": plan.get("noise_policy", "legacy_noise_object_per_piece"),
        "temporal_merge_policy": plan.get(
            "temporal_merge_policy", "legacy_final_latent_crossfade"
        ),
        "global_noise": global_noise_report,
        "inherited_video_mask": inherited_mask_report,
        "first_pass_contract": plan.get("first_pass_contract", "not_declared"),
        "recommended_refine": plan.get("recommended_refine"),
        "second_pass_audio_policy": plan.get(
            "second_pass_audio_policy", "locked_input_audio"
        ),
        "second_pass_sampler_binding": (
            "rebound_per_upscaled_piece"
            if plan["schema"]
            in {PLAN_SCHEMA_LOW_SIGMA_V3, PLAN_SCHEMA_MASKED_LOW_SIGMA_V4}
            else "legacy_external_sampler_object"
        ),
        "refine_sigma_start": (
            float(torch.as_tensor(sigmas).flatten()[0])
            if torch.as_tensor(sigmas).numel()
            else None
        ),
        "refine_nfe": max(0, int(torch.as_tensor(sigmas).numel()) - 1),
    }
    return output, _json(report)
