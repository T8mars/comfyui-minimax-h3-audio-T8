"""Prepare control time indices without changing original RGB coordinates.

The existing Fun Apply owns spatial crop/fit and VAE encoding. This adapter
does not estimate pose/depth, process source audio, observe PTS or sample.
"""

from __future__ import annotations

import math

import torch

from .control_regions import tensor_sha
from .director_controls import normalize_controls, resolve_control_geometry


SCHEMA = "t8.director.control_source.v1"
TARGET_FPS = 24


def _rate(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("控制视频source_fps必须是实际提供的有限正数字，不能猜测帧率")
    return float(value)


def _video_selection(frames, source_fps, length, start_seconds, short_video_policy):
    # Source AV's exact float64/ties-even law, limited to temporal selection.
    # The original helper also resizes RGB and creates PCM, so cannot be
    # called here without changing the source-pixel crop contract or audio.
    target_times = (
        start_seconds + torch.arange(length, dtype=torch.float64) / TARGET_FPS
    )
    positions = target_times * source_fps
    if not bool(torch.isfinite(target_times).all() and torch.isfinite(positions).all()):
        raise ValueError("控制视频时间换算产生非有限帧坐标，原输入未修改")
    rounded = torch.round(positions)
    beyond = rounded >= len(frames)
    held = int(beyond.sum().item())
    if held and short_video_policy == "strict":
        raise ValueError(
            "Control source video is too short for the requested H3 window: "
            f"needs frame time {float(target_times[-1]):.6f}s, available through "
            f"{(len(frames) - 1) / source_fps:.6f}s; select hold_last_frame explicitly to extend"
        )
    # Clamp before the int64 conversion, not after: a large finite supplied
    # rate may legitimately request a held last frame, never wrap to index0.
    indices = rounded.clamp(0, len(frames) - 1).to(
        dtype=torch.long, device=frames.device
    )
    return (
        frames.index_select(0, indices).detach(),
        indices.cpu().tolist(),
        target_times.tolist(),
        held,
    )


def _target_mask(geometry, device):
    region = geometry["region"]
    if region is None:
        return None
    left, top, right, bottom = region["box_xyxy"]
    x = torch.arange(geometry["width"], dtype=torch.float64, device=device)
    y = torch.arange(geometry["height"], dtype=torch.float64, device=device)
    # Each original target pixel gets its actual covered area. Fractional
    # edges remain fractional rather than rounding the rectangle or center.
    wx = (
        torch.minimum(x + 1, x.new_tensor(right)) - torch.maximum(x, x.new_tensor(left))
    ).clamp(0, 1)
    wy = (
        torch.minimum(y + 1, y.new_tensor(bottom)) - torch.maximum(y, y.new_tensor(top))
    ).clamp(0, 1)
    return (wy[:, None] * wx[None, :]).to(dtype=torch.float32).unsqueeze(0)


def prepare_director_control_source(
    control_frames,
    spec,
    *,
    width,
    height,
    length,
    source_kind="video",
    source_fps=None,
    stage="single",
):
    """Return original-spatial IMAGE, optional target MASK, crop JSON, report.

    source_fps is only a supplied decoder/declaration value. The IMAGE port
    contains no PTS, so even a plausible rate never certifies VFR timing or
    physical source lineage. Static inputs use the first image only; the old
    Fun Apply retains its existing hold-last expansion to H3's frame grid.
    """
    if (
        not isinstance(control_frames, torch.Tensor)
        or control_frames.is_nested
        or control_frames.layout != torch.strided
        or control_frames.device.type == "meta"
        or control_frames.ndim != 4
        or control_frames.shape[-1] < 3
        or any(size <= 0 for size in control_frames.shape)
        or not control_frames.is_floating_point()
        or not bool(torch.isfinite(control_frames).all())
    ):
        raise ValueError("控制源必须是非空有限浮点RGB IMAGE [B,H,W,C]，C至少为3")
    if not isinstance(source_kind, str) or source_kind not in {"image", "video"}:
        raise ValueError("控制源种类必须明确选择image或video，不会将音频当控制画面")
    normalized = normalize_controls({"version": 1, "items": [spec]})["items"][0]
    geometry = resolve_control_geometry(
        normalized,
        width=width,
        height=height,
        length=length,
        source_size=(int(control_frames.shape[2]), int(control_frames.shape[1])),
        stage=stage,
    )
    source_settings = normalized["source"]
    warnings = []
    rate = None
    ignored = 0
    if source_kind == "image":
        selected = control_frames[:1].detach()
        indices, target_times, held = [0], [0.0], 0
        ignored = len(control_frames) - 1
        if ignored:
            warnings.append(
                f"静态控制使用首张图片，另{ignored}张未作为动画时间线；需要逐帧控制请明确使用视频。"
            )
        if source_settings["start_seconds"]:
            warnings.append("静态控制没有视频时轴，开始时间草稿保留但未应用。")
        clock_status = "static_image_no_temporal_clock"
    else:
        rate = _rate(source_fps)
        selected, indices, target_times, held = _video_selection(
            control_frames,
            rate,
            length,
            source_settings["start_seconds"],
            source_settings["short_video_policy"],
        )
        clock_status = "supplied_fps_CFR_assumption_unverified"
        warnings.append(
            "控制视频仅按输入FPS假设CFR对齐；没有实际PTS，VFR时间与物理同源关系均未认证。"
        )
        if held:
            warnings.append(
                f"按明确hold_last_frame策略保持末帧{held}个目标帧；未循环或生成新画面。"
            )
    mask = _target_mask(geometry, selected.device)
    report = {
        "schema": SCHEMA,
        "status": "temporal_geometry_prepared_not_sampled",
        "stage": geometry["stage"],
        "source_kind": source_kind,
        "source_size_actual": [
            int(control_frames.shape[2]),
            int(control_frames.shape[1]),
        ],
        "input_frame_count": len(control_frames),
        "output_frame_count": len(selected),
        "target_frame_count": length,
        "target_fps": TARGET_FPS,
        "source_fps": rate,
        "source_start_seconds_requested": source_settings["start_seconds"],
        "source_start_seconds_applied": source_settings["start_seconds"]
        if source_kind == "video"
        else None,
        "source_indices": indices,
        "target_times_seconds": target_times,
        "held_video_frames": held,
        "ignored_image_batch_frames": ignored,
        "static_expansion": "original_fun_apply_hold_last"
        if source_kind == "image"
        else None,
        "rounding": "torch.round_float64_ties_to_even_then_clamp"
        if source_kind == "video"
        else None,
        "clock_status": clock_status,
        "pts_timing_certified": False,
        "spatial_operation": "none_original_RGB_dimensions",
        "geometry": geometry,
        "source_tensor_sha256": tensor_sha(control_frames),
        "selected_tensor_sha256": tensor_sha(selected),
        "region_mask_sha256": tensor_sha(mask) if mask is not None else None,
        "source_audio_consumed": False,
        "preprocessor_estimated": False,
        "streaming": False,
        "memory_safe_claimed": False,
        "portable_cache_identity": False,
        "quality_accepted": False,
        "warnings": warnings,
    }
    return selected, mask, geometry["crop_plan_json"], report
