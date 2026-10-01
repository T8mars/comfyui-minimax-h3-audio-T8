"""Versioned Director sampling settings and the exact two-pass canvas plan."""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping


DEFAULT_TWO_PASS = {
    "mode": "two_pass",
    "preset": "standard_4plus4_v1",
    "output_mp": "auto",
    "upscaler": "auto",
    "low_loras": [],
    "high_loras": [],
}


def _lora_rows(value: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{stage} LoRA 必须是列表")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{stage} 第 {index + 1} 条 LoRA 无效")
        name = row.get("name")
        enabled = row.get("enabled", True)
        strength = row.get("strength", 1.0)
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise ValueError(f"{stage} LoRA 需要文件名与启用状态")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError(f"{stage} LoRA 强度必须是 -2–2 的有限数字")
        if enabled and not name:
            raise ValueError(f"{stage} 已启用 LoRA 必须选择文件")
        result.append({
            "id": str(row.get("id") or f"{stage}-{index}"),
            "name": name,
            "strength": float(strength),
            "enabled": enabled,
            "source": str(row.get("source") or "user"),
        })
    if len({row["id"] for row in result}) != len(result):
        raise ValueError(f"{stage} LoRA 行 ID 重复")
    return result


def _hyperflow_stage_eav(value: Any, stage: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{stage} EAV 配置必须是对象")
    mode = value.get("mode", "disabled")
    if not isinstance(mode, str) or mode not in {"disabled", "report_only", "apply_exp"}:
        raise ValueError(f"{stage} EAV 模式无效")
    result = {"mode": mode}
    for key, default, minimum, maximum in (
        ("tau", 4.0, -32.0, 32.0),
        ("start_video_progress", 0.15, 0.0, 0.99),
        ("end_video_progress", 0.90, 0.01, 1.0),
        ("g_hard_limit", 1.5, 1.0, 3.0),
    ):
        raw = value.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not minimum <= raw <= maximum:
            raise ValueError(f"{stage} EAV {key} 超出范围")
        result[key] = float(raw)
    if result["start_video_progress"] >= result["end_video_progress"]:
        raise ValueError(f"{stage} EAV 时间窗必须先开始后结束")
    workspace = value.get("max_workspace_mib", 32)
    if isinstance(workspace, bool) or not isinstance(workspace, int) or not 4 <= workspace <= 512:
        raise ValueError(f"{stage} EAV 工作区必须为 4–512 MiB 整数")
    result["max_workspace_mib"] = workspace
    return result


def _hyperflow_stage_checkpoint(value: Any) -> dict[str, str]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("HyperFlow HEAD 阶段回执必须是对象")
    mode = value.get("mode", "off")
    if not isinstance(mode, str) or mode not in {"off", "save", "resume_tail"}:
        raise ValueError("HyperFlow HEAD 阶段回执模式无效")
    if mode != "resume_tail":
        return {"mode": mode}
    path = value.get("artifact_path")
    digest = value.get("artifact_sha256")
    if not isinstance(path, str) or not path or len(path) > 1024:
        raise ValueError("仅 TAIL 恢复须填已保存 HEAD 的相对路径")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("仅 TAIL 恢复须填 HEAD 文件的64位 SHA256")
    return {"mode": mode, "artifact_path": path, "artifact_sha256": digest.lower()}


def _hyperflow_stage_relay(value: Any, stage: str) -> dict[str, str]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{stage} Relay 配置必须是对象")
    mode = value.get("mode", "inherit")
    if mode == "inherit":
        return {"mode": "inherit"}
    if mode != "custom":
        raise ValueError(f"{stage} Relay 文本模式无效")
    global_prompt = value.get("global_prompt")
    local_prompts = value.get("local_prompts", "")
    timing_mode = value.get("timing_mode", "auto_equal")
    time_ranges = value.get("time_ranges", "")
    if not isinstance(global_prompt, str) or not global_prompt.strip():
        raise ValueError(f"{stage} Relay 自定义全局提示词不能为空")
    if not isinstance(local_prompts, str) or not isinstance(time_ranges, str):
        raise ValueError(f"{stage} Relay 局部提示与时间范围必须是文本")
    if not isinstance(timing_mode, str) or timing_mode not in {"auto_equal", "frames", "seconds", "percent"}:
        raise ValueError(f"{stage} Relay 时间模式无效")
    if timing_mode == "auto_equal" and time_ranges.strip():
        raise ValueError(f"{stage} Relay 自动均分不能同时填写时间范围")
    return {"mode": "custom", "global_prompt": global_prompt.strip(),
            "local_prompts": local_prompts, "timing_mode": timing_mode,
            "time_ranges": time_ranges}


def normalize_sampling(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"mode": "single"}
    if not isinstance(raw, Mapping):
        raise ValueError("采样设置必须是对象")
    mode = raw.get("mode", "single")
    if mode == "single":
        result = {"mode": "single"}
        if "loras" in raw:
            result["loras"] = _lora_rows(raw["loras"], "单采")
            result["lora_mode"] = str(raw.get("lora_mode", "manual"))
            if result["lora_mode"] not in {"auto", "none", "manual"}:
                raise ValueError("单采 LoRA 模式无效")
        if "resolution_mp" in raw:
            resolution = raw["resolution_mp"]
            if resolution != "auto":
                try:
                    resolution = float(resolution)
                except (TypeError, ValueError) as error:
                    raise ValueError("单采最终 MP 无效") from error
                if not math.isfinite(resolution) or not 0.2 <= resolution <= 2.1:
                    raise ValueError("单采最终 MP 必须在 0.2–2.1 范围内")
            result["resolution_mp"] = resolution
        return result
    if mode == "hyperflow":
        variant = raw.get("variant", "single8")
        if variant not in {"single8", "continuous4plus4", "continuous4plus4separate",
                           "upscale8plus4", "upscale4plus4"}:
            raise ValueError("不支持的 HyperFlow 实验路线")
        file = raw.get("hyperflow_file")
        if not isinstance(file, str) or not file.startswith(("hyperflow/", "loras/")):
            raise ValueError("HyperFlow 必须单独选择原始适配器文件")
        target = raw.get("output_mp", "auto")
        if target != "auto":
            if isinstance(target, bool):
                raise ValueError("HyperFlow 最终 MP 无效")
            try:
                target = float(target)
            except (TypeError, ValueError) as error:
                raise ValueError("HyperFlow 最终 MP 无效") from error
            if not math.isfinite(target) or not 0.2 <= target <= 2.1:
                raise ValueError("HyperFlow 最终 MP 必须在 0.2–2.1 范围内")
        upscaler = raw.get("upscaler", "auto")
        if not isinstance(upscaler, str):
            raise ValueError("HyperFlow 学习型 3D 放大模型必须是文件名")
        checkpoint = (_hyperflow_stage_checkpoint(raw.get("stage_checkpoint"))
                      if variant == "continuous4plus4separate" else {"mode": "off"})
        result = {"mode": "hyperflow", "variant": variant, "hyperflow_file": file,
                "output_mp": target, "upscaler": upscaler,
                "low_loras": ([] if checkpoint["mode"] == "resume_tail" else
                              _lora_rows(raw.get("low_loras", []), "HyperFlow 一采")),
                # An inactive HIGH draft is persisted by the project but must
                # not invalidate or execute a single-stage recipe.
                "high_loras": [] if variant == "single8" else _lora_rows(raw.get("high_loras", []), "HyperFlow 二采")}
        if variant == "continuous4plus4separate":
            stages = raw.get("stage_eav", {})
            if not isinstance(stages, Mapping):
                raise ValueError("HyperFlow 分离式阶段 EAV 配置必须是对象")
            result["stage_eav"] = {stage: _hyperflow_stage_eav(
                None if stage == "head" and checkpoint["mode"] == "resume_tail" else stages.get(stage),
                stage.upper())
                                   for stage in ("head", "tail")}
            result["stage_checkpoint"] = checkpoint
            relay = raw.get("stage_relay", {})
            if not isinstance(relay, Mapping):
                raise ValueError("HyperFlow 分离式阶段 Relay 配置必须是对象")
            result["stage_relay"] = {
                stage: (_hyperflow_stage_relay(relay.get(stage), stage.upper())
                        if not (stage == "head" and checkpoint["mode"] == "resume_tail")
                        else {"mode": "inherit"})
                for stage in ("head", "tail")}
        return result
    if mode != "two_pass":
        raise ValueError("采样方式必须是单采或双采")
    preset = raw.get("preset", DEFAULT_TWO_PASS["preset"])
    if preset != "standard_4plus4_v1":
        raise ValueError(f"不支持的双采预设：{preset}")
    target = raw.get("output_mp", "auto")
    if target != "auto":
        if isinstance(target, bool) or not isinstance(target, (int, float, str)):
            raise ValueError("双采最终 MP 无效")
        try:
            target = float(target)
        except ValueError as error:
            raise ValueError("双采最终 MP 无效") from error
        if not math.isfinite(target) or not 0.2 <= target <= 2.1:
            raise ValueError("双采最终 MP 必须在 0.2–2.1 范围内")
    upscaler = raw.get("upscaler", "auto")
    if not isinstance(upscaler, str):
        raise ValueError("3D 放大模型必须是文件名")
    return {
        "mode": "two_pass",
        "preset": preset,
        "output_mp": target,
        "upscaler": upscaler,
        "low_loras": _lora_rows(raw.get("low_loras", []), "一采"),
        "high_loras": _lora_rows(raw.get("high_loras", []), "二采"),
    }


def effective_sampling(doc: Mapping[str, Any], shot: Mapping[str, Any]) -> dict[str, Any]:
    inherited = shot.get("samplingInherit", True)
    if not isinstance(inherited, bool):
        raise ValueError("镜头采样继承开关无效")
    source = doc.get("sampling") if inherited else shot.get("sampling")
    if source is None and not inherited:
        raise ValueError("本镜独立采样设置缺失")
    return normalize_sampling(source)


def two_pass_canvas(fraction: float, requested_mp: float | str) -> dict[str, Any]:
    from .director_project import director_canvas
    from .learned_latent_upscale_advanced import learned_upscale_geometry

    wanted_width, wanted_height = director_canvas(fraction, requested_mp)
    choices = []
    for dx in (-32, 0, 32):
        for dy in (-32, 0, 32):
            low_width = max(32, round((wanted_width / 2 + dx) / 32) * 32)
            low_height = max(32, round((wanted_height / 2 + dy) / 32) * 32)
            try:
                actual = learned_upscale_geometry(
                    source_latent_width=low_width // 16,
                    source_latent_height=low_height // 16,
                    size_mode="target_megapixels",
                    scale_by=2.0,
                    target_megapixels=wanted_width * wanted_height / 1_000_000,
                    target_width=wanted_width,
                    target_height=wanted_height,
                    aspect_policy="preserve_source",
                    max_anisotropy=1.05,
                )
            except ValueError:
                continue
            high_width, high_height = actual["output_width"], actual["output_height"]
            score = (
                abs(math.log((high_width / high_height) / fraction)),
                abs(high_width * high_height - wanted_width * wanted_height),
                abs(low_width * low_height - wanted_width * wanted_height / 4),
            )
            choices.append((score, low_width, low_height, actual))
    if not choices:
        raise ValueError("所选画幅／MP 无法得到合法双采尺寸，请调整最终总像素")
    _, low_width, low_height, actual = min(choices, key=lambda item: item[0])
    return {
        "low_width": low_width,
        "low_height": low_height,
        "width": actual["output_width"],
        "height": actual["output_height"],
        "actual_megapixels": actual["output_pixels"] / 1_000_000,
        "requested_megapixels": requested_mp,
        "scale_x": actual["scale_x"],
        "scale_y": actual["scale_y"],
        "memory_warning": actual["memory_warning"],
        "geometry_version": 1,
    }


def sampling_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(normalize_sampling(value))
