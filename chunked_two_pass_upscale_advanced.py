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


try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except ImportError:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0


VAE_DOWNSAMPLE = 16
GRID_PIXELS = 32


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


def sample_piece(piece, positive, model, noise, sampler, sigmas, negative, cfg):
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
    output = guider.sample(
        noise.generate_noise(latent),
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
):
    tile_w = plan["tile_width"] // VAE_DOWNSAMPLE
    tile_h = plan["tile_height"] // VAE_DOWNSAMPLE
    overlap_w = plan["spatial_overlap"] // VAE_DOWNSAMPLE
    overlap_h = plan["spatial_overlap"] // VAE_DOWNSAMPLE
    fade = plan["spatial_fade"] // VAE_DOWNSAMPLE
    minimum = plan["minimum_tile_size"] // VAE_DOWNSAMPLE
    _, channels, time, height, width = chunk_video.shape
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
            audio_mask = torch.zeros_like(chunk_audio)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_audio)),
                "noise_mask": comfy.nested_tensor.NestedTensor(
                    (video_mask, audio_mask)
                ),
            }
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
                sampler,
                sigmas,
                negative,
                cfg,
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
        "schema": "t8.minimax_h3.chunked_two_pass.v1",
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
    if not isinstance(plan, dict) or plan.get("schema") != "t8.minimax_h3.chunked_two_pass.v1":
        raise ValueError("plan must come from the T8 Chunked Two-Pass Plan node")
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("expected a nested MiniMax H3 AV latent")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError("unexpected MiniMax H3 AV latent shapes")
    if video.shape[0] != 1:
        raise ValueError("chunked two-pass currently supports batch 1")

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
        chunk_conditioning = reanchor_conditioning(
            conditioning,
            start_frame,
            end_frame,
            tuple(chunk_video.shape[-2:]),
        )
        if index and accumulated is not None:
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
        )
        accumulated = _append_video(accumulated, chunk_output, start_token)
        segment_reports.append(
            {
                "index": index,
                "pixel_frames": [start_frame, end_frame],
                "video_tokens": [start_token, end_token],
                "audio_tokens_read_only": [audio_start, audio_end],
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
        "pixel_limit_policy": "no_project_pixel_area_limit",
    }
    return output, _json(report)
