"""Versioned Director control drafts and independent source/target geometry.

Pure standard-library contracts: no Torch/model imports, preprocessing claims,
sampler changes or portable executable identity. Project/compiler/UI adapters
must explicitly consume these values; their existence is not an integration.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import uuid


FIELD = "controlRegions"
VERSION = 1
CROP_SCHEMA = "t8.h3.control_crop.v1"
KINDS = {"depth", "pose", "canny", "HED", "MLSD", "custom"}
FIT_MODES = {"exact", "center_crop", "stretch"}
CROP_FITS = {"exact", "stretch", "contain", "cover"}
ENTRY_FIELDS = {
    "id",
    "asset_id",
    "enabled",
    "control_net_name",
    "control_kind",
    "fit_mode",
    "strength",
    "start_percent",
    "end_percent",
    "stages",
    "source",
    "region",
    "crop",
}
NESTED_FIELDS = {
    "stages": {"low", "high"},
    "source": {"start_seconds", "short_video_policy"},
    "region": {"mode", "xyxy_normalized"},
    "crop": {"enabled", "xyxy_normalized", "fit", "flip_x", "flip_y"},
}


def _identity(value):
    try:
        result = str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("区域控制身份必须是稳定 UUID") from error
    if result != value:
        raise ValueError("区域控制 UUID 必须采用规范格式")
    return result


def _number(value, label):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label}必须是有限数字")
    return value


def _rectangle(value, label, *, active=False, target=False):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label}需要四个归一化边界数字")
    box = [_number(item, label) for item in value]
    if active:
        left, top, right, bottom = box
        if (
            right <= left
            or bottom <= top
            or not math.isfinite(right - left)
            or not math.isfinite(bottom - top)
        ):
            raise ValueError(f"{label}需要有限的正宽高，原请求未修改")
        if target and any(item < 0 or item > 1 for item in box):
            raise ValueError("目标控制区域必须位于归一化画布0到1内，不能当作源像素crop")
    return box


def _stage(stage):
    if not isinstance(stage, str) or stage.lower() not in {"single", "low", "high"}:
        raise ValueError("区域控制阶段必须是 SINGLE、LOW 或 HIGH")
    return stage.lower()


def _active(item, stage=None):
    return (
        item["enabled"]
        and item["strength"] != 0
        and (stage in (None, "single") or item["stages"][stage])
    )


def normalize_controls(value, identity=_identity):
    """Copy a structurally valid draft, retaining inactive rows and extra data.

    Empty selections are editable drafts. Actual generation resolves active
    assets and rectangles separately; no missing selection is invented here.
    """
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != VERSION
        or not isinstance(value.get("items"), list)
    ):
        raise ValueError("局部控制需要版本1和控制列表；未知版本请保留原项目")
    normalized = deepcopy(value)
    ids = set()
    for number, item in enumerate(normalized["items"], 1):
        label = f"局部控制第{number}项"
        if not isinstance(item, dict):
            raise ValueError(f"{label}必须是对象")
        uid = identity(item.get("id"))
        if uid in ids:
            raise ValueError("局部控制 UUID 重复")
        ids.add(uid)
        item.setdefault("asset_id", None)
        if item["asset_id"] is not None:
            identity(item["asset_id"])
        for name, default in {
            "enabled": True,
            "control_net_name": "",
            "control_kind": "depth",
            "fit_mode": "center_crop",
            "strength": 0.7,
            "start_percent": 0.0,
            "end_percent": 0.75,
        }.items():
            item.setdefault(name, default)
        if type(item["enabled"]) is not bool or not isinstance(
            item["control_net_name"], str
        ):
            raise ValueError(f"{label}启用开关或模型名称类型无效")
        if (
            not isinstance(item["control_kind"], str)
            or item["control_kind"] not in KINDS
            or not isinstance(item["fit_mode"], str)
            or item["fit_mode"] not in FIT_MODES
        ):
            raise ValueError(f"{label}控制类别或画面适配方式无效")
        for name in ("strength", "start_percent", "end_percent"):
            _number(item[name], label + name)
        defaults = {
            "stages": {"low": True, "high": True},
            "source": {"start_seconds": 0.0, "short_video_policy": "hold_last_frame"},
            "region": {"mode": "full", "xyxy_normalized": [0.0, 0.0, 1.0, 1.0]},
            "crop": {
                "enabled": False,
                "xyxy_normalized": [0.0, 0.0, 1.0, 1.0],
                "fit": "contain",
                "flip_x": False,
                "flip_y": False,
            },
        }
        for name, fields in defaults.items():
            item.setdefault(name, {})
            if not isinstance(item[name], dict):
                raise ValueError(f"{label}{name}必须是对象")
            for key, default in fields.items():
                item[name].setdefault(key, deepcopy(default))
        if any(type(item["stages"][name]) is not bool for name in ("low", "high")):
            raise ValueError(f"{label}阶段开关必须是布尔值")
        source, region, crop = item["source"], item["region"], item["crop"]
        _number(source["start_seconds"], label + "控制视频开始时间")
        if not isinstance(source["short_video_policy"], str) or source[
            "short_video_policy"
        ] not in {"strict", "hold_last_frame"}:
            raise ValueError(f"{label}源视频不足时策略无效")
        if not isinstance(region["mode"], str) or region["mode"] not in {
            "full",
            "rect",
        }:
            raise ValueError(f"{label}目标区域模式无效")
        _rectangle(region["xyxy_normalized"], label + "目标区域")
        _rectangle(crop["xyxy_normalized"], label + "源裁框")
        if (
            any(
                type(crop[name]) is not bool for name in ("enabled", "flip_x", "flip_y")
            )
            or not isinstance(crop["fit"], str)
            or crop["fit"] not in CROP_FITS
        ):
            raise ValueError(f"{label}源裁框开关或适配方式无效")
    json.dumps(normalized, allow_nan=False)  # Also reject hidden NaN/non-JSON fields.
    return normalized


def validate_controls(value, identity=_identity):
    normalize_controls(value, identity)


def control_asset_ids(shot, *, active=False, stage=None):
    stage = _stage(stage) if stage is not None else None
    value = normalize_controls(shot.get(FIELD))
    return (
        [
            item["asset_id"]
            for item in value["items"]
            if item["asset_id"] and (not active or _active(item, stage))
        ]
        if value
        else []
    )


def _execution_values(item):
    if not 0 <= item["start_percent"] <= item["end_percent"] <= 1:
        raise ValueError("局部控制时间窗需要0≤开始≤结束≤1，未自动夹取")
    if item["source"]["start_seconds"] < 0:
        raise ValueError("控制视频开始时间不能为负数")
    if item["region"]["mode"] == "rect":
        _rectangle(
            item["region"]["xyxy_normalized"], "目标区域", active=True, target=True
        )
    if item["crop"]["enabled"]:
        # The underlying crop supports a virtual exterior with zero padding.
        # This is not the bounded target mask, so do not clip it to [0,1].
        _rectangle(item["crop"]["xyxy_normalized"], "源裁框", active=True)


def _asset_path(root, relative):
    if not isinstance(relative, str) or not relative or "\0" in relative:
        raise ValueError("控制素材缺少实际相对路径")
    parts = PurePosixPath(relative.replace("\\", "/"))
    if (
        not parts.parts
        or parts.is_absolute()
        or any(part in {".", ".."} or ":" in part for part in parts.parts)
    ):
        raise ValueError("控制素材路径越界或不是普通相对路径")
    if root is None:
        return None
    base = Path(root).resolve()
    path = (base / parts).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("控制素材实际路径越界")
    if not path.is_file():
        raise FileNotFoundError("控制素材文件不存在，请重连原素材")
    return path


def resolve_controls(
    shot, assets, *, stage="single", asset_root=None, identity=_identity
):
    """Resolve active dependencies, optionally checking actual source bytes.

    Without asset_root the result binds declarations only, not a media read.
    Runtime geometry must still use actual decoded IMAGE dimensions and fps.
    """
    stage = _stage(stage)
    value = normalize_controls(shot.get(FIELD), identity)
    result = []
    for item in value["items"] if value else ():
        if not _active(item, stage):
            continue
        _execution_values(item)
        if not item["control_net_name"].strip():
            raise ValueError(
                "请选择局部控制所用的 Fun Control 模型；不会猜测或替换模型"
            )
        asset = assets.get(item["asset_id"])
        if (
            not isinstance(asset, dict)
            or asset.get("id") != item["asset_id"]
            or asset.get("kind") not in {"image", "video"}
            or asset.get("missing")
        ):
            raise ValueError(
                "局部控制缺少已上传的控制图片或视频；不会将录音或普通参考图自动代用"
            )
        digest = asset.get("sha256")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            raise ValueError("控制素材缺少有效的内容SHA256")
        if any(
            type(asset.get(name)) is not int or asset[name] <= 0
            for name in ("width", "height")
        ):
            raise ValueError("控制素材缺少有效的源画面宽高")
        path = _asset_path(asset_root, asset.get("server_path"))
        if path is not None:
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(block)
            if actual.hexdigest() != digest.lower():
                raise ValueError("控制素材实际内容SHA256不匹配，请重连而非沿用旧声明")
        result.append(
            {
                "id": item["id"],
                "asset_id": item["asset_id"],
                "asset_kind": asset["kind"],
                "asset_sha256": digest.lower(),
                "server_path": asset["server_path"],
                "source_size_declared": [asset["width"], asset["height"]],
                "asset_content_verified": path is not None,
                "target_fps": 24,
                "stage": stage.upper(),
                "spec": portable_controls({"version": VERSION, "items": [item]})[
                    "items"
                ][0],
                "portable_cache_identity": False,
            }
        )
    return result


def resolve_control_geometry(
    item, *, width, height, length, source_size, stage="single"
):
    """Resolve pure geometry against an actual stage and decoded source size."""
    stage = _stage(stage)
    item = normalize_controls({"version": VERSION, "items": [item]})["items"][0]
    _execution_values(item)
    if any(type(size) is not int or size <= 0 or size % 32 for size in (width, height)):
        raise ValueError("控制目标画布需要32对齐的正整数宽高")
    if type(length) is not int or length < 5 or length % 17 != 5:
        raise ValueError("控制目标帧数必须遵守H3的17n+5帧格")
    if (
        not isinstance(source_size, (tuple, list))
        or len(source_size) != 2
        or any(type(size) is not int or size <= 0 for size in source_size)
    ):
        raise ValueError("源裁框需要实际解码RGB的正整数宽高")

    def pixels(box, dimensions):
        try:
            values = [
                coordinate * dimensions[index % 2]
                for index, coordinate in enumerate(box)
            ]
        except OverflowError as error:
            raise ValueError("控制坐标无法表示为有限的实际像素") from error
        if any(not math.isfinite(value) for value in values):
            raise ValueError("控制坐标转换后的实际像素不是有限数字")
        _rectangle(values, "实际像素裁框", active=True)
        return values

    region = None
    if item["region"]["mode"] == "rect":
        normalized = item["region"]["xyxy_normalized"]
        region = {
            "xyxy_normalized": deepcopy(normalized),
            "box_xyxy": pixels(normalized, (width, height)),
            "coordinate_space": "original_target_canvas",
        }
    crop = None
    if item["crop"]["enabled"]:
        requested = item["crop"]
        crop = {
            "schema": CROP_SCHEMA,
            "source_size": list(source_size),
            "boxes": [pixels(requested["xyxy_normalized"], source_size)],
            "fit": requested["fit"],
            "flip_x": requested["flip_x"],
            "flip_y": requested["flip_y"],
        }
    return {
        "stage": stage.upper(),
        "width": width,
        "height": height,
        "length": length,
        "region": region,
        "crop_plan": crop,
        "crop_plan_json": ""
        if crop is None
        else json.dumps(crop, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "geometry_only": True,
        "portable_cache_identity": False,
    }


def portable_controls(value):
    normalized = normalize_controls(value)
    if normalized is None:
        return None
    items = []
    for item in normalized["items"]:
        clean = {name: deepcopy(item[name]) for name in ENTRY_FIELDS}
        for name, fields in NESTED_FIELDS.items():
            clean[name] = {key: deepcopy(item[name][key]) for key in fields}
        items.append(clean)
    return {"version": VERSION, "items": items}


def remap_controls(shot, asset_map, fresh):
    """Atomic in-place ID remap, including inactive and explicitly missing rows."""
    value = shot.get(FIELD)
    normalize_controls(value)
    if value is None:
        return
    mapped = [
        (item, asset_map[item["asset_id"]] if item.get("asset_id") else None)
        for item in value["items"]
    ]
    changes = [
        (
            item,
            _identity(fresh("control", item["id"])),
            _identity(asset) if asset else None,
        )
        for item, asset in mapped
    ]
    if len({new_id for _, new_id, _ in changes}) != len(changes):
        raise ValueError("区域控制迁移产生重复 UUID，原草稿保留")
    for item, new_id, asset in changes:
        item["id"] = new_id
        if item.get("asset_id"):
            item["asset_id"] = asset
