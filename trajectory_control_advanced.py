from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


TRAJECTORY_CONTROL_SCHEMA = "h3_t8_trajectory_control_plan/v1"
EASING_MODES = ("linear", "smoothstep", "hold")
CLIP_POLICIES = ("clip_to_canvas", "reject_outside")
RENDER_MODES = ("soft_region", "box_outline", "reference_sprite")
PALETTE = (
    (0.95, 0.20, 0.15),
    (0.10, 0.85, 0.30),
    (0.15, 0.45, 1.00),
    (0.95, 0.75, 0.10),
    (0.75, 0.20, 0.95),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _parse_keyframes(raw: str, length: int, clip_policy: str) -> tuple[list[dict[str, Any]], int]:
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError(f"keyframes_json is invalid JSON: {error.msg}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("keyframes_json must be a non-empty JSON list")
    output = []
    clipped = 0
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"keyframe {index} must be an object")
        frame_raw = item.get("frame")
        if isinstance(frame_raw, bool) or not isinstance(frame_raw, int):
            raise ValueError(f"keyframe {index}.frame must be an integer")
        frame = int(frame_raw)
        if not 0 <= frame < length:
            raise ValueError(f"keyframe {index}.frame must stay within 0..{length - 1}")
        object_id = str(item.get("object_id", "subject")).strip()
        if not object_id:
            raise ValueError(f"keyframe {index}.object_id cannot be empty")
        x = _number(item.get("x"), f"keyframe {index}.x")
        y = _number(item.get("y"), f"keyframe {index}.y")
        width = _number(item.get("width", item.get("w")), f"keyframe {index}.width")
        height = _number(item.get("height", item.get("h")), f"keyframe {index}.height")
        strength = _number(item.get("strength", 1.0), f"keyframe {index}.strength")
        if width <= 0 or height <= 0:
            raise ValueError(f"keyframe {index} width and height must be positive")
        if not 0.0 <= strength <= 2.0:
            raise ValueError(f"keyframe {index}.strength must stay within 0..2")
        values = (x, y, width, height)
        inside = x >= 0 and y >= 0 and x + width <= 1 and y + height <= 1
        if not inside:
            if clip_policy == "reject_outside":
                raise ValueError(
                    f"keyframe {index} bbox leaves the normalized canvas; use clip_to_canvas to clip"
                )
            clipped += 1
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            width = min(max(width, 1e-4), 1.0 - x)
            height = min(max(height, 1e-4), 1.0 - y)
        output.append(
            {
                "frame": frame,
                "object_id": object_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "strength": strength,
                "input_bbox": list(values),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in output:
        grouped[item["object_id"]].append(item)
    for object_id, values in grouped.items():
        frames = [item["frame"] for item in values]
        if frames != sorted(set(frames)):
            raise ValueError(
                f"object {object_id!r} keyframes must be unique and strictly increasing"
            )
    return output, clipped


def _ease(alpha: float, mode: str) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    if mode == "linear":
        return alpha
    if mode == "smoothstep":
        return alpha * alpha * (3.0 - 2.0 * alpha)
    if mode == "hold":
        return 0.0
    raise ValueError(f"Unknown easing mode {mode!r}")


def _interpolate_track(values: list[dict[str, Any]], length: int, easing: str) -> list[dict[str, float]]:
    keys = sorted(values, key=lambda item: item["frame"])
    output = []
    for frame in range(length):
        if frame <= keys[0]["frame"]:
            left = right = keys[0]
            alpha = 0.0
        elif frame >= keys[-1]["frame"]:
            left = right = keys[-1]
            alpha = 0.0
        else:
            right_index = next(index for index, key in enumerate(keys) if key["frame"] >= frame)
            left = keys[right_index - 1]
            right = keys[right_index]
            alpha = (frame - left["frame"]) / max(1, right["frame"] - left["frame"])
        alpha = _ease(alpha, easing)
        output.append(
            {
                name: float(left[name] + (right[name] - left[name]) * alpha)
                for name in ("x", "y", "width", "height", "strength")
            }
        )
    return output


def _render_plan_preview(plan: Mapping[str, Any], maximum: int = 12) -> torch.Tensor:
    length = int(plan["length"])
    indices = torch.linspace(0, length - 1, min(maximum, length)).round().int().tolist()
    height = min(384, int(plan["height"]))
    width = max(64, int(round(int(plan["width"]) * height / int(plan["height"]))))
    canvas = torch.zeros((len(indices), height, width, 3), dtype=torch.float32)
    for track_index, track in enumerate(plan["tracks"]):
        color = torch.tensor(PALETTE[track_index % len(PALETTE)])
        for preview_index, frame in enumerate(indices):
            bbox = track["frames"][frame]
            x0 = max(0, min(width - 1, int(round(bbox["x"] * width))))
            y0 = max(0, min(height - 1, int(round(bbox["y"] * height))))
            x1 = max(x0 + 1, min(width, int(round((bbox["x"] + bbox["width"]) * width))))
            y1 = max(y0 + 1, min(height, int(round((bbox["y"] + bbox["height"]) * height))))
            line = max(1, min(height, width) // 180)
            canvas[preview_index, y0 : min(y1, y0 + line), x0:x1] = color
            canvas[preview_index, max(y0, y1 - line) : y1, x0:x1] = color
            canvas[preview_index, y0:y1, x0 : min(x1, x0 + line)] = color
            canvas[preview_index, y0:y1, max(x0, x1 - line) : x1] = color
    return canvas


def build_trajectory_control_plan(
    *,
    keyframes_json: str,
    width: int,
    height: int,
    length: int,
    fps: float,
    easing: str,
    clip_policy: str,
) -> tuple[dict[str, Any], torch.Tensor, str, int]:
    width, height, length = int(width), int(height), int(length)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % 32 or height % 32:
        raise ValueError("H3 trajectory-control width and height must be divisible by 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("H3 trajectory-control length must follow 17n+5 (22, 39, 56, ...)")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be finite and positive")
    if easing not in EASING_MODES:
        raise ValueError(f"Unknown easing {easing!r}")
    if clip_policy not in CLIP_POLICIES:
        raise ValueError(f"Unknown clip_policy {clip_policy!r}")
    keyframes, clipped = _parse_keyframes(keyframes_json, length, clip_policy)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in keyframes:
        grouped[item["object_id"]].append(item)
    tracks = [
        {
            "object_id": object_id,
            "keyframes": values,
            "frames": _interpolate_track(values, length, easing),
        }
        for object_id, values in sorted(grouped.items())
    ]
    plan = {
        "schema": TRAJECTORY_CONTROL_SCHEMA,
        "status": "ready",
        "width": width,
        "height": height,
        "length": length,
        "fps": float(fps),
        "easing": easing,
        "clip_policy": clip_policy,
        "clipped_keyframe_count": clipped,
        "tracks": tracks,
        "scientific_scope": (
            "TrailBlazer-inspired keyframed spatial trajectory representation for H3 control-video "
            "conditioning; it does not reproduce TrailBlazer's U-Net cross-attention surgery"
        ),
        "audio_modified": False,
    }
    plan["plan_sha256"] = _hash(plan)
    report = {
        "schema": "h3_t8_trajectory_control_report/v1",
        "status": "ready",
        "object_count": len(tracks),
        "keyframe_count": len(keyframes),
        "clipped_keyframe_count": clipped,
        "plan_sha256": plan["plan_sha256"],
        "recommended_connection": (
            "Render -> MiniMax H3 Fun Control Apply control_video; start with control_kind=custom "
            "or preprocess the rendered sprite/region as canny/depth/pose before Apply"
        ),
        "claim_boundary": plan["scientific_scope"],
    }
    return plan, _render_plan_preview(plan), _canonical_json(report), len(tracks)


def validate_trajectory_control_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or plan.get("schema") != TRAJECTORY_CONTROL_SCHEMA:
        raise ValueError("trajectory_plan must be a connected H3 trajectory-control plan")
    output = dict(plan)
    expected = output.pop("plan_sha256", None)
    if not isinstance(expected, str) or expected != _hash(output):
        raise ValueError("trajectory-control plan SHA-256 does not match its payload")
    output["plan_sha256"] = expected
    return output


def _soft_rectangle(
    height: int,
    width: int,
    bbox: Mapping[str, float],
    feather: float,
) -> torch.Tensor:
    yy = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xx = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    y = yy[:, None]
    x = xx[None, :]
    x0, y0 = float(bbox["x"]), float(bbox["y"])
    x1, y1 = x0 + float(bbox["width"]), y0 + float(bbox["height"])
    if feather <= 0:
        return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).float()
    edge = max(1e-6, float(feather))
    return (
        torch.sigmoid((x - x0) / edge)
        * torch.sigmoid((x1 - x) / edge)
        * torch.sigmoid((y - y0) / edge)
        * torch.sigmoid((y1 - y) / edge)
    ).clamp(0.0, 1.0)


def _reference_for_track(reference_images: torch.Tensor, track_index: int) -> torch.Tensor:
    if reference_images.ndim != 4 or reference_images.shape[-1] < 3:
        raise ValueError("reference_images must be a ComfyUI IMAGE batch")
    if int(reference_images.shape[0]) not in (1, track_index + 1) and track_index >= int(
        reference_images.shape[0]
    ):
        raise ValueError("reference_images must contain one image or at least one image per object")
    index = 0 if int(reference_images.shape[0]) == 1 else track_index
    return reference_images[index, ..., :3].detach().float().cpu().clamp(0.0, 1.0)


def _paste_reference(
    canvas: torch.Tensor,
    union_mask: torch.Tensor,
    reference: torch.Tensor,
    reference_mask: torch.Tensor | None,
    bbox: Mapping[str, float],
    strength: float,
) -> None:
    height, width = canvas.shape[:2]
    x0 = max(0, min(width - 1, int(round(float(bbox["x"]) * width))))
    y0 = max(0, min(height - 1, int(round(float(bbox["y"]) * height))))
    x1 = max(x0 + 1, min(width, int(round((float(bbox["x"]) + float(bbox["width"])) * width))))
    y1 = max(y0 + 1, min(height, int(round((float(bbox["y"]) + float(bbox["height"])) * height))))
    box_h, box_w = y1 - y0, x1 - x0
    source_h, source_w = reference.shape[:2]
    scale = min(box_w / max(1, source_w), box_h / max(1, source_h))
    target_w = max(1, min(box_w, int(round(source_w * scale))))
    target_h = max(1, min(box_h, int(round(source_h * scale))))
    sprite = F.interpolate(
        reference.permute(2, 0, 1)[None],
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    if reference_mask is None:
        alpha = torch.ones((target_h, target_w), dtype=torch.float32)
    else:
        alpha = F.interpolate(
            reference_mask[None, None].float().cpu(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp(0.0, 1.0)
    alpha = (alpha * min(max(float(strength), 0.0), 1.0)).clamp(0.0, 1.0)
    offset_x = x0 + (box_w - target_w) // 2
    offset_y = y0 + (box_h - target_h) // 2
    region = canvas[offset_y : offset_y + target_h, offset_x : offset_x + target_w]
    region.copy_(region * (1.0 - alpha[..., None]) + sprite * alpha[..., None])
    union_mask[offset_y : offset_y + target_h, offset_x : offset_x + target_w] = torch.maximum(
        union_mask[offset_y : offset_y + target_h, offset_x : offset_x + target_w], alpha
    )


def render_trajectory_control(
    *,
    trajectory_plan: Mapping[str, Any],
    render_mode: str,
    feather: float,
    line_width: int,
    background_level: float,
    reference_images: torch.Tensor | None = None,
    reference_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    plan = validate_trajectory_control_plan(trajectory_plan)
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unknown render_mode {render_mode!r}")
    if render_mode == "reference_sprite" and reference_images is None:
        raise ValueError("reference_sprite requires connected reference_images")
    length, height, width = int(plan["length"]), int(plan["height"]), int(plan["width"])
    control = torch.full(
        (length, height, width, 3), float(background_level), dtype=torch.float32
    )
    union = torch.zeros((length, height, width), dtype=torch.float32)
    for track_index, track in enumerate(plan["tracks"]):
        color = torch.tensor(PALETTE[track_index % len(PALETTE)], dtype=torch.float32)
        reference = (
            _reference_for_track(reference_images, track_index)
            if reference_images is not None
            else None
        )
        mask = None
        if reference_masks is not None:
            if reference_masks.ndim != 3:
                raise ValueError("reference_masks must be a ComfyUI MASK batch")
            index = 0 if int(reference_masks.shape[0]) == 1 else track_index
            if index >= int(reference_masks.shape[0]):
                raise ValueError("reference_masks must contain one mask or one mask per object")
            mask = reference_masks[index]
        for frame, bbox in enumerate(track["frames"]):
            strength = min(max(float(bbox["strength"]), 0.0), 1.0)
            if render_mode == "reference_sprite":
                assert reference is not None
                _paste_reference(control[frame], union[frame], reference, mask, bbox, strength)
                continue
            region = _soft_rectangle(height, width, bbox, feather)
            if render_mode == "box_outline":
                inner_bbox = dict(bbox)
                inner_bbox["x"] += line_width / max(width, 1)
                inner_bbox["y"] += line_width / max(height, 1)
                inner_bbox["width"] = max(
                    1e-4, inner_bbox["width"] - 2 * line_width / max(width, 1)
                )
                inner_bbox["height"] = max(
                    1e-4, inner_bbox["height"] - 2 * line_width / max(height, 1)
                )
                region = (region - _soft_rectangle(height, width, inner_bbox, feather)).clamp(0.0, 1.0)
            alpha = (region * strength).clamp(0.0, 1.0)
            control[frame] = control[frame] * (1.0 - alpha[..., None]) + color * alpha[..., None]
            union[frame] = torch.maximum(union[frame], alpha)
    preview_indices = torch.linspace(0, length - 1, min(12, length)).round().int()
    preview = control[preview_indices]
    report = {
        "schema": "h3_t8_trajectory_control_render/v1",
        "status": "ok",
        "render_mode": render_mode,
        "plan_sha256": plan["plan_sha256"],
        "frames": length,
        "height": height,
        "width": width,
        "object_count": len(plan["tracks"]),
        "audio_modified": False,
        "recommended_connection": (
            "connect control_video to MiniMaxH3FunControlApplyT8Advanced; custom is the direct "
            "experimental route, while canny/depth/pose should be produced by the matching preprocessor"
        ),
        "claim_boundary": plan["scientific_scope"],
    }
    return control.clamp(0.0, 1.0), union.clamp(0.0, 1.0), preview, _canonical_json(report)

