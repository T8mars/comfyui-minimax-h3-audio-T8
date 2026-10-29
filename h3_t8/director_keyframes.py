"""Versioned middle-image drafts and explicit visible/generated frame clocks.

Pure data only: importing the Director must not import Torch or model runtimes.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .keyframe_positions import resolve_position
from .frame_grid import FPS, align_frame_count
from .timing_plan import make_timing_plan


FIELD = "middleKeyframes"
BASES = {"visible_delivery", "generated_canvas"}
ENTRY_FIELDS = {"id", "asset_id", "position_mode", "position", "visual_noise_aug", "resize_mode", "enabled"}


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数字")
    return value


def validate_keyframes(value, identity):
    if value is None:
        return
    if (not isinstance(value, dict) or type(value.get("version")) is not int
            or value["version"] != 1 or not isinstance(value.get("time_basis", "visible_delivery"), str)
            or value.get("time_basis", "visible_delivery") not in BASES
            or not isinstance(value.get("items"), list)):
        raise ValueError("中间关键帧需要版本1、明确时轴和图片列表")
    ids = set()
    for number, item in enumerate(value["items"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"中间关键帧第{number}项必须是对象")
        uid = identity(item.get("id"))
        if uid in ids:
            raise ValueError("中间关键帧 UUID 重复")
        ids.add(uid)
        if item.get("asset_id") is not None:
            identity(item["asset_id"])
        if not isinstance(item.get("position_mode"), str) or item["position_mode"] not in {"frame", "seconds", "percent"}:
            raise ValueError(f"中间关键帧第{number}项定位方式无效")
        _number(item.get("position"), "中间关键帧位置")
        aug = _number(item.get("visual_noise_aug", .999), "关键帧原始混噪值")
        if not 0 <= aug <= 1:
            raise ValueError("关键帧原始混噪值必须在0到1之间")
        if (not isinstance(item.get("resize_mode", "center_crop"), str)
                or item.get("resize_mode", "center_crop") not in {"center_crop", "stretch"}):
            raise ValueError("关键帧缩放方式无效")
        if type(item.get("enabled", True)) is not bool:
            raise ValueError("关键帧启用开关必须是布尔值")


def keyframe_asset_ids(shot):
    value = shot.get(FIELD)
    return [item["asset_id"] for item in value["items"] if item.get("asset_id")] if value else []


def shot_clock(shot, assets=None):
    """The same requested/AudioWindow clock used by the actual graph consumers."""
    duration = shot["manualDuration"]
    events = shot["events"] if shot["writingMode"] == "advanced" else []
    if shot["autoDuration"]:
        if events:
            duration = max(item["end"] for item in events)
        elif shot["sound"] == "record" and shot.get("audio"):
            duration = shot["end"] - shot["start"]
    _number(duration, "镜头时长")
    if duration <= 0 or not math.isfinite(duration * FPS):
        raise ValueError("镜头时长必须大于0且形成有限帧数")
    visible = math.ceil(duration * FPS)
    requested = max(5, visible)
    generated, trim_start = align_frame_count(requested), 0.0
    if shot["sound"] == "record" and shot.get("audio"):
        audio = (assets or {}).get(shot["audio"], {})
        plan = make_timing_plan(shot["start"], duration,
                               source_duration_seconds=audio.get("duration", 0))
        generated, trim_start = plan.frame_count, plan.final_trim_start_seconds
    start_frame = round(trim_start * FPS)  # OutputTrim's exact video rounding.
    return {"fps": FPS, "requested_seconds": duration, "requested_frames": requested,
            "aligned_frames": generated, "generated_seconds": generated / FPS,
            "delivery_trim_frames": visible, "delivery_trim_start_seconds": trim_start,
            "delivery_start_frame": start_frame, "delivery_end_frame": start_frame + visible}


def resolve_keyframes(shot, assets, *, clock=None):
    """Resolve each explicit request afresh; do not clamp after a duration edit."""
    value = shot.get(FIELD)
    if not value or not any(item.get("enabled", True) for item in value["items"]):
        return []
    clock = shot_clock(shot, assets) if clock is None else clock
    visible = clock["delivery_trim_frames"]
    generated = clock["aligned_frames"]
    basis = value.get("time_basis", "visible_delivery")
    count = visible if basis == "visible_delivery" else generated
    used, result = {}, []
    for number, item in enumerate(value["items"], 1):
        if not item.get("enabled", True):
            continue
        label = f"中间关键帧第{number}项"
        asset = assets.get(item.get("asset_id"))
        if asset is None or asset.get("kind") != "image":
            raise ValueError(f"{label}缺少实际图片，请选择图片或关闭该项")
        frame = resolve_position(item["position_mode"], item["position"], count, label=label)
        if basis == "visible_delivery":
            frame += clock["delivery_start_frame"]
        if frame in used:
            raise ValueError(f"{label}与第{used[frame]}项都定位到帧{frame}，请明确调整而非覆盖")
        used[frame] = number
        result.append({"id": item["id"], "asset_id": item["asset_id"],
            "position_mode": item["position_mode"], "position": item["position"],
            "frame_index": frame, "time_basis": basis, "visible_frames": visible,
            "generated_frames": generated, "visible_in_delivery": clock["delivery_start_frame"] <= frame < clock["delivery_end_frame"],
            "visual_noise_aug": item.get("visual_noise_aug", .999),
            "resize_mode": item.get("resize_mode", "center_crop")})
    return sorted(result, key=lambda item: item["frame_index"])


def portable_keyframes(value):
    if value is None:
        return None
    return {"version": value["version"], "time_basis": value.get("time_basis", "visible_delivery"),
            "items": [{key: deepcopy(item[key]) for key in ENTRY_FIELDS if key in item} for item in value["items"]]}


def remap_keyframes(shot, asset_map, fresh):
    value = shot.get(FIELD)
    if value:
        for item in value["items"]:
            item["id"] = fresh("keyframe", item["id"])
            if item.get("asset_id"):
                item["asset_id"] = asset_map[item["asset_id"]]
