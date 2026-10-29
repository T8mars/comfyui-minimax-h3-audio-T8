"""Native director generation bridges for the validated H3 recipe families."""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

import folder_paths

from .director_project import ProjectStore, compile_project, validate_project
from .director_sampling_settings import effective_sampling, lora_interpretation
from .director_two_pass import apply_two_pass_graph
from .director_hyperflow import apply_hyperflow_graph
from .director_control_graph import append_director_controls, audit_control_latent, control_condition_length
from .ffmpeg_utils import resolve_ffmpeg
from .h3_weight_diagnostics import inspect_h3_weight_file


_UNET_CANDIDATES = (
    "minimax_h3_fl2va_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
)
_REF2VA_UNET_CANDIDATES = (
    "minimax_h3_ref2va_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    *_UNET_CANDIDATES,
)
_FAST_H3_UNET_CANDIDATES = (
    "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
)
_CLIP_CANDIDATES = (
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
)
_VIDEO_VAE_CANDIDATES = (
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors",
)
_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _pick(folder: str, candidates: tuple[str, ...], label: str) -> str:
    available = set(folder_paths.get_filename_list(folder))
    for name in candidates:
        if name in available and folder_paths.get_full_path(folder, name):
            return name
    # Never invent a model path.  A useful error lets the user install the
    # missing official asset without creating a queue item that cannot run.
    raise ValueError(
        f"导演台找不到 {label}；请安装以下任一正式文件：{', '.join(candidates)}"
    )


def _pick_requested(folder: str, requested: str, candidates: tuple[str, ...], label: str) -> str:
    if requested in (None, "", "auto"):
        return _pick(folder, candidates, label)
    if requested == "none":
        raise ValueError(f"导演台不能关闭必需的 {label}")
    if requested not in set(folder_paths.get_filename_list(folder)) or not folder_paths.get_full_path(folder, requested):
        raise ValueError(f"导演台找不到所选 {label}：{requested}")
    return requested


def _optional_turbo_lora() -> str | None:
    names = folder_paths.get_filename_list("loras")
    for name in names:
        low = name.lower()
        if "minimax_h3_turbo" in low and "comfyui" in low:
            return name
    return None


def _director_generation_settings(project: Mapping[str, Any]) -> dict[str, Any]:
    raw = project.get("doc", {}).get("generation", {})
    if not isinstance(raw, Mapping):
        raw = {}
    mode = str(raw.get("lora_mode", "manual" if any(str(v) not in ("auto", "none") for v in (raw.get("lora", "auto") if isinstance(raw.get("lora", "auto"), list) else [raw.get("lora", "auto")])) else "auto"))
    legacy = raw.get("lora", "auto")
    legacy = [legacy] if isinstance(legacy, str) else legacy
    rows = raw.get("loras")
    if rows is None or (not rows and any(name not in ("auto", "none") for name in legacy)):
        rows = [{"name": name, "strength": raw.get("lora_strength", 1.0), "enabled": True} for name in legacy if name not in ("auto", "none")]
    if not isinstance(rows, list):
        raise ValueError("LoRA 必须是列表")
    lora_rows = [{"name": str(row["name"]), "strength": float(row.get("strength", 1.0)),
                  "enabled": bool(row.get("enabled", True)),
                  **({"id": row["id"]} if "id" in row else {}),
                  **lora_interpretation(row)} for row in rows]
    resolution = raw.get("resolution_mp", "auto")
    if resolution != "auto":
        try:
            resolution = float(resolution)
        except (TypeError, ValueError) as error:
            raise ValueError("总像素必须选择自动或 0.2–1.0 MP") from error
        if resolution < 0.2 or resolution > 1.0:
            raise ValueError("总像素必须在 0.2–1.0 MP 之间")
    return {
        "unet": str(raw.get("unet", "auto") or "auto"),
        "clip": str(raw.get("clip", "auto") or "auto"),
        "video_vae": str(raw.get("video_vae", "auto") or "auto"),
        "audio_vae": str(raw.get("audio_vae", "auto") or "auto"),
        "lora_mode": mode,
        "loras": lora_rows,
        "resolution_mp": resolution,
    }


def director_model_catalog() -> dict[str, list[dict[str, str]]]:
    def entries(folder: str, predicate) -> list[dict[str, str]]:
        result = []
        for name in sorted(set(folder_paths.get_filename_list(folder))):
            if predicate(name) and folder_paths.get_full_path(folder, name):
                result.append({"value": name, "label": name})
        return result

    from .nodes_hyperflow_advanced import _weight_options
    from .h3_fun_control_advanced import _available_fun_control_names

    hyperflow = [name for name in _weight_options() if name != "missing_hyperflow_weights"]
    return {
        "hyperflow": [{"value": name, "label": name} for name in hyperflow],
        "fun_control_names": [{"value": name, "label": name} for name in _available_fun_control_names()],
        "unet": [{"value": "auto", "label": "自动（按镜头类型）"}] + entries(
            "diffusion_models", lambda n: n.lower().endswith(".safetensors") and ("minimax_h3" in n.lower() or "fasth3" in n.lower())
        ),
        "clip": [{"value": "auto", "label": "自动 H3 文本编码器"}] + entries(
            "clip", lambda n: n.lower().endswith(".safetensors") and ("qwen3vl" in n.lower() or "minimax_h3" in n.lower())
        ),
        "video_vae": [{"value": "auto", "label": "自动 H3 视频 VAE"}] + entries(
            "vae", lambda n: "minimax_h3_video_vae" in n.lower()
        ),
        "audio_vae": [{"value": "auto", "label": "自动 H3 音频 VAE"}] + entries(
            "vae", lambda n: "minimax_h3_audio_vae" in n.lower()
        ),
        "lora": [
            {"value": "auto", "label": "自动 Turbo LoRA"},
            {"value": "none", "label": "不使用 LoRA"},
        ] + entries("loras", lambda n: n.lower().endswith(".safetensors")),
        "upscaler": entries("latent_upscale_models", lambda n: n.lower().endswith(".safetensors")),
    }


def _optional_semantic_bridge_model() -> str:
    """Return an installed Bridge asset name without inventing a path."""

    # Semantic Bridge owns a recursive, duplicate-safe model resolver instead
    # of a normal ComfyUI folder_paths entry.  Use the same labels the node's
    # combo exposes so nested ``t8_compat/...`` files are executable.
    try:
        from .nodes_semantic_bridge import model_paths

        names = tuple(model_paths())
    except (ImportError, OSError, RuntimeError):
        names = ()
    try:
        names = tuple(names) + tuple(folder_paths.get_filename_list("semantic_bridge"))
    except (KeyError, AttributeError, TypeError):
        pass
    # Prefer the packaged ComfyUI-compatible conversion over original teacher
    # checkpoints when both are present.
    unique_names = {str(name) for name in names}
    preferred = sorted(
        (name for name in unique_names if name.replace("\\", "/").startswith("t8_compat/")),
        key=lambda name: (
            0 if "minimaxh3_semanticbridge" in name.lower() else 1,
            name,
        ),
    )
    ordered = preferred + sorted(unique_names.difference(preferred))
    for name in ordered:
        if str(name).lower().endswith(".safetensors"):
            if "/" in str(name) or "\\" in str(name):
                return str(name)
            if folder_paths.get_full_path("semantic_bridge", name):
                return str(name)
    raise ValueError(
        "导演台已启用 Semantic Bridge，但 models/semantic_bridge 没有可用 .safetensors；"
        "请安装 Semantic-Bridge-Comfy 后刷新模型列表"
    )


def _director_d3_settings(project: Mapping[str, Any], shot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge optional D3 settings while keeping old projects byte-compatible."""

    merged: dict[str, Any] = {}
    for owner in (project.get("d3"), project.get("doc", {}).get("d3")):
        if owner is None:
            continue
        if not isinstance(owner, Mapping):
            raise ValueError("导演台 D3 配置必须是对象")
        for key, value in owner.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    local = shot.get("d3")
    inherit = shot.get("d3Inherit")
    local_active = isinstance(local, Mapping) and any(
        isinstance(value, Mapping) and any(bool(item) for item in value.values())
        for value in local.values()
    )
    if inherit is False or (inherit is None and local_active):
        if not isinstance(local, Mapping):
            raise ValueError("导演台镜头 D3 配置必须是对象")
        for key, value in local.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    result = {}
    for key in ("semantic_bridge", "prompt_relay", "fast_h3_v2", "memory"):
        value = merged.get(key, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"导演台 D3 配置 {key} 必须是对象")
        result[key] = dict(value)
    return result


def _relay_plan_inputs(shot: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the Director's advanced event list into native Relay inputs."""

    global_prompt = str(shot.get("global_prompt") or "").strip()
    local_prompt = str(shot.get("local_prompt") or "").strip()
    events = list(shot.get("events") or [])
    if not global_prompt:
        global_prompt = local_prompt or "Cinematic continuous shot with coherent subject and stable lighting."
    if str(shot.get("writing_mode")) != "advanced" or not events:
        local_prompts = ""
        time_ranges = ""
        timing_mode = "auto_equal"
    else:
        local_prompts = "\n".join(str(event["text"]) for event in events)
        visible_frames = int(shot["time"].get("delivery_trim_frames",
            shot["time"].get("requested_frames", shot["time"]["aligned_frames"])))
        generated_frames = int(shot["time"]["aligned_frames"])
        ranges = []
        for index, event in enumerate(events):
            end = int(event["end_frame"])
            # The complete native grid includes hidden alignment frames after
            # delivery. Only inherit an event that actually reaches the visible
            # endpoint; do not fill a user's gap, drop events or edit their draft.
            if index == len(events) - 1 and end == visible_frames < generated_frames:
                end = generated_frames
            ranges.append(f"{int(event['start_frame'])}-{end - 1}")
        time_ranges = "\n".join(ranges)
        timing_mode = "frames"
    return {
        "global_prompt": global_prompt,
        "local_prompts": local_prompts,
        "length": int(shot["time"]["aligned_frames"]),
        "timing_mode": timing_mode,
        "time_ranges": time_ranges,
        "math_profile": "paper_v1",
        "epsilon": 0.1,
        "allow_gaps": False,
        "allow_overlaps": False,
    }


def _asset_for_role(shot: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    for item in shot.get("media_map", []):
        if item.get("role") == role:
            return item
    return None


def _assets_for_role(shot: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    return [item for item in shot.get("media_map", []) if item.get("role") == role]


def build_director_generation_prompt(
    project: Mapping[str, Any],
    shot_id: str,
    store: ProjectStore,
    *,
    seed: int = 26091901,
) -> dict[str, Any]:
    """Compile one D2a–D2c shot into a real native ComfyUI API prompt.

    The returned prompt is still pure data and can be validated without a
    model.  Only the explicit queue helper below mutates Core's queue.
    """

    checked = validate_project(project)
    raw_shot = next((item for item in checked["doc"]["shots"] if item["id"] == shot_id), None)
    if raw_shot is None:
        raise ValueError("请选择项目内有效的镜头 UUID")
    report = compile_project(checked, store, shot_id=shot_id)
    if not report["ready"]:
        number = checked["doc"]["shots"].index(raw_shot) + 1
        raise ValueError(f"第 {number} 镜「{raw_shot['name']}」预检未通过：" + "; ".join(e["message"] for e in report["errors"]))
    shot = next((item for item in report["shots"] if item["id"] == shot_id), None)
    if shot is None:
        raise ValueError("请选择项目内有效的镜头 UUID")

    # Discover the actual encoder before spending GPU time on this recipe.
    resolve_ffmpeg()

    task = shot["task_type"].lower()
    sound = raw_shot["sound"]
    if task not in {"t2va", "i2va", "fl2va", "l2va", "ref2va", "hybrid"}:
        raise ValueError(f"导演台暂不支持任务类型：{task}")
    d3 = _director_d3_settings(checked, raw_shot)
    sampling = effective_sampling(checked["doc"], raw_shot)
    generation = _director_generation_settings(checked)
    lora_source = "generation_rows"
    if sampling["mode"] == "single" and "loras" in sampling:
        generation["loras"] = sampling["loras"]
        generation["lora_mode"] = sampling["lora_mode"]
        generation["resolution_mp"] = sampling.get("resolution_mp", generation["resolution_mp"])
        lora_source = "sampling_rows"
    bridge_cfg = d3["semantic_bridge"]
    relay_cfg = d3["prompt_relay"]
    fast_cfg = d3["fast_h3_v2"]
    memory_cfg = d3["memory"]
    bridge_enabled = bool(bridge_cfg.get("enabled", False))
    relay_enabled = bool(relay_cfg.get("enabled", False))
    fast_enabled = bool(fast_cfg.get("enabled", False))
    low_vram_enabled = bool(memory_cfg.get("low_vram", False))
    chunk_ffn_enabled = bool(memory_cfg.get("chunk_ffn", False))
    if fast_enabled and (task != "t2va" or shot["audio_mode"] != "native" or shot.get("middle_keyframes")):
        report["warnings"].append({"shot_id": shot_id,
            "message": "FastH3 V2 保留所选模型及首尾/参考/原音/中间图条件；非文字组合属于 EXP，训练资格仍为 T2VA，效果须单独验证。不会自动换模型、移除条件或更改采样路线。"})
    relay_events = []
    if relay_enabled and str(relay_cfg.get("execution_mode", "apply_exp")) == "apply_exp":
        from .prompt_relay_advanced import _local_prompt_lines

        relay_events = _local_prompt_lines(_relay_plan_inputs(shot)["local_prompts"])
    if fast_enabled and len(relay_events) > 1 and str(fast_cfg.get("profile", "trained_vsa_exp")) != "dense_compat_exp":
        raise ValueError("FastH3 V2 训练 VSA 无法执行 Prompt Relay 的逐查询时间偏置；请显式选择 dense_compat_exp，或关闭 Relay。不会静默忽略 Relay 或自动切换后端")
    if sampling["mode"] == "two_pass" and fast_enabled:
        raise ValueError("标准 4+4 双采不能与 FastH3 V2 的独立 8 步模型同时启用")
    if sampling["mode"] == "hyperflow":
        if fast_enabled:
            raise ValueError("HyperFlow 的两时钟训练网格与 FastH3 V2 的独立采样器同时占用 SAMPLER/SIGMAS；当前没有联合采样协议。保留配置，请显式选择其中一条采样路线，不会静默覆盖")
        if bridge_enabled or low_vram_enabled or chunk_ffn_enabled:
            report["warnings"].append({"shot_id": shot_id,
                "message": "HyperFlow EXP 保留已启用的 Bridge/分块注意力/FFN 路线，但此叠加尚未完成真实 GPU 组合验收；请检查实际输出。"})
        if task != "t2va" or shot["audio_mode"] != "native" or relay_enabled or shot.get("middle_keyframes"):
            report["warnings"].append({"shot_id": shot_id,
                "message": "HyperFlow 将按各阶段实际 MODEL 配对首尾/参考/中间图及声音条件；这些 EXP 组合仍需独立 GPU 成片和人审，不借用文字单采资格。"})
        if sampling.get("stage_checkpoint", {}).get("mode") == "resume_tail" and sound == "record":
            recording = _asset_for_role(shot, "drive_audio") or {}
            report["warnings"].append({"shot_id": shot_id,
                "frozen_head_sha256": sampling["stage_checkpoint"]["artifact_sha256"],
                "current_recording_sha256": recording.get("sha256"),
                "message": "仅TAIL恢复保持冻结HEAD的原素材、遮罩与噪声；当前录音用于TAIL条件与最终交付，不证明它与历史HEAD同源。若更换录音并需要全片按新录音生成，请重新生成HEAD；不会静默替换冻结状态。"})
        if sampling["variant"] != "single8":
            from comfy.cli_args import args
            import comfy.memory_management

            if not args.disable_comfy_compiler and comfy.memory_management.aimdo_enabled:
                raise ValueError("当前 Core 启用了 comfy-aimdo 编译器；HyperFlow 双分支实测会造成原生崩溃。请以 --disable-comfy-compiler 启动独立 Core 后再选该实验路线")
    unet_candidates = (
        _FAST_H3_UNET_CANDIDATES
        if fast_enabled
        else _REF2VA_UNET_CANDIDATES
        if task in {"ref2va", "hybrid"}
        else _UNET_CANDIDATES
    )
    unet = _pick_requested("diffusion_models", generation["unet"], unet_candidates, "H3 diffusion model")
    # A bounded header read is evidence for a warning, never a compatibility
    # gate.  Model filenames and unknown user LoRAs are not proof of a merge.
    unet_path = folder_paths.get_full_path("diffusion_models", unet)
    merged_acceleration = inspect_h3_weight_file(unet_path).get("merged_acceleration", {}) if unet_path else {}
    if merged_acceleration.get("declared"):
        report["warnings"].append({
            "shot_id": shot_id,
            "message": "所选底模文件元数据声明已合并加速适配器，但未数值验证；叠加其他加速 LoRA 可能重复。保留手动选择，请核对模型来源和实际输出。",
        })
    if sampling["mode"] == "two_pass" and unet == "minimax_h3_fl2va_pruned_int8_convrot.safetensors" and any(
        row["enabled"] and row["name"] == "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
        for stage in ("low_loras", "high_loras") for row in sampling[stage]
    ):
        report["warnings"].append({
            "shot_id": shot_id,
            "message": "本机 GPU 实测裁剪版 FL2VA INT8＋EMA B LoRA 出现 AdaLN patch 形状错误；即使视频生成成功也不能认定该 LoRA 完整生效。保留您的选择，请查看 Core 日志或改用完整底模复测。",
        })
    clip = _pick_requested("clip", generation["clip"], _CLIP_CANDIDATES, "Qwen3-VL H3 text encoder")
    video_vae = _pick_requested("vae", generation["video_vae"], _VIDEO_VAE_CANDIDATES, "H3 video VAE")
    audio_vae = _pick_requested("vae", generation["audio_vae"], (_AUDIO_VAE,), "H3 audio VAE")

    task_type_value = {"ref2va": "Ref2VA", "hybrid": "Hybrid"}.get(task, task.upper())
    if relay_enabled and bridge_enabled and not bridge_cfg.get("enabled", False):
        raise ValueError("导演台 D3 Semantic Bridge 配置无效")

    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}},
    }
    model_id = "1"
    # Keep the established default graph IDs (first media node is 14) intact;
    # D3-enabled graphs intentionally append their explicit bridge/patch nodes.
    next_id = 13
    # FastH3 V2 is already a trained 8-step diffusion checkpoint; applying the
    # ordinary H3 Turbo LoRA on top changes its learned-gate contract.
    lora_entries = []
    legacy_turbo_route = task in {"t2va", "i2va"} and sound == "native" and not relay_enabled and not fast_enabled
    if sampling["mode"] in {"two_pass", "hyperflow"}:
        pass  # Stage-specific LoRA stacks are built from the unpatched base below.
    elif legacy_turbo_route:
        requested_rows = [(index, row) for index, row in enumerate(generation["loras"], 1) if row["enabled"]]
        if generation["lora_mode"] == "auto" and not requested_rows:
            selected = _optional_turbo_lora()
            if selected and not merged_acceleration.get("declared"):
                lora_entries = [{"name": selected, "strength": 1.0}]
        elif generation["lora_mode"] != "none":
            lora_entries = [
                {"name": _pick_requested("loras", row["name"], (row["name"],), "LoRA"),
                 "strength": row["strength"], "row_index": index, "row_id": row.get("id"),
                 "source": lora_source, **lora_interpretation(row)}
                for index, row in requested_rows
            ]
    elif generation["lora_mode"] == "manual" and any(row["enabled"] for row in generation["loras"]):
        lora_entries = [
            {"name": _pick_requested("loras", row["name"], (row["name"],), "LoRA"),
             "strength": row["strength"], "row_index": index, "row_id": row.get("id"),
             "source": lora_source, **lora_interpretation(row)}
            for index, row in enumerate(generation["loras"], 1) if row["enabled"]
        ]
        report["warnings"].append({"shot_id": shot_id,
            "message": "保留手动 LoRA 及顺序；本路线组合未经效果验证，风险由用户承担，不据此自动缩短采样步数。"})
    # Keep the old public tuple report and graph IDs, not a second LoRA chain.
    lora_names = [(row["name"], row["strength"]) for row in lora_entries]
    lora_loaders = []
    for row in lora_entries:
        graph[str(next_id)] = {
            "class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
            "inputs": {"model": [model_id, 0], "lora_name": row["name"],
                       "strength_model": row["strength"], **lora_interpretation(row)},
        }
        lora_loaders.append({"node_id": str(next_id), "stage": "SINGLE",
            "row_index": row.get("row_index", 1), "row_id": row.get("row_id"),
            "source": row.get("source", "automatic")})
        model_id = str(next_id)
        next_id += 1

    # Memory nodes are real MODEL patches, chained before Relay/FastH3 so their
    # authenticated receipts can be inspected by those downstream runtimes.
    if low_vram_enabled:
        graph[str(next_id)] = {
            "class_type": "MiniMaxH3LowVRAMAttentionT8Advanced",
            "inputs": {"model": [model_id, 0], "head_chunks": int(memory_cfg.get("head_chunks", 4))},
        }
        model_id = str(next_id)
        next_id += 1
    if chunk_ffn_enabled:
        graph[str(next_id)] = {
            "class_type": "MiniMaxH3ChunkFeedForwardT8Advanced",
            "inputs": {
                "model": [model_id, 0],
                "chunks": int(memory_cfg.get("chunks", 2)),
                "seq_threshold": int(memory_cfg.get("seq_threshold", 4096)),
            },
        }
        model_id = str(next_id)
        next_id += 1

    bridge_id = None
    if bridge_enabled:
        bridge_id = str(next_id)
        graph[bridge_id] = {
            "class_type": "MiniMaxH3SemanticBridgeConfigT8",
            "inputs": {
                "model_name": bridge_cfg.get("model_name") or _optional_semantic_bridge_model(),
                "enabled": True,
                "alpha": float(bridge_cfg.get("alpha", 0.10)),
                "magnitude_match": str(bridge_cfg.get("magnitude_match", "per_token")),
                "token_scope": str(bridge_cfg.get("token_scope", "all_tokens")),
                "device": str(bridge_cfg.get("device", "auto")),
                "chunk_tokens": int(bridge_cfg.get("chunk_tokens", 256)),
            },
        }
        next_id += 1

    condition_id = "5"
    positive_id = "5"
    latent_id = "5"
    mux_audio_id = "5"
    if relay_enabled:
        plan_id = str(next_id)
        graph[plan_id] = {
            "class_type": "MiniMaxH3PromptRelayPlanT8Advanced",
            "inputs": {
                **_relay_plan_inputs(shot),
                "epsilon": float(relay_cfg.get("epsilon", 0.1)),
            },
        }
        next_id += 1
        condition_id = str(next_id)
        relay_inputs: dict[str, Any] = {
            "model": [model_id, 0],
            "clip": ["4", 0],
            "video_vae": ["2", 0],
            "audio_vae": ["3", 0],
            "prompt_relay_plan": [plan_id, 0],
            "width": shot["canvas"]["width"],
            "height": shot["canvas"]["height"],
            "task_type": task_type_value,
            "audio_mode": "lock_source" if shot["audio_mode"] == "lock_source" else "native",
            "audio_denoise_strength": 0.0 if shot["audio_mode"] == "lock_source" else 1.0,
            "add_source_as_reference": shot["audio_mode"] == "lock_source",
            # Director has already resolved stable asset aliases to their
            # native audio slots. Do not redirect a video's Audio1 to the
            # separately selected drive recording after that compilation.
            "prompt_primary_audio_ordinal": 1 if sound == "voice" else 0,
            "strict_prompt_tags": True,
            "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s",
            "execution_mode": str(relay_cfg.get("execution_mode", "apply_exp")),
            "query_chunk_rows": int(relay_cfg.get("query_chunk_rows", 256)),
        }
        if bridge_id:
            relay_inputs["semantic_bridge"] = [bridge_id, 0]
        graph[condition_id] = {
            "class_type": "MiniMaxH3PromptRelayConditioningT8Advanced",
            "inputs": relay_inputs,
        }
        positive_id = condition_id
        latent_id = condition_id
        mux_audio_id = condition_id
        next_id += 1
    else:
        graph["5"] = {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "clip": ["4", 0],
                "video_vae": ["2", 0],
                "audio_vae": ["3", 0],
                "prompt": shot["prompt"],
                "width": shot["canvas"]["width"],
                "height": shot["canvas"]["height"],
                "length": shot["time"]["aligned_frames"],
                "task_type": task_type_value,
                "audio_mode": "lock_source" if shot["audio_mode"] == "lock_source" else "native",
                "audio_denoise_strength": 0.0 if shot["audio_mode"] == "lock_source" else 1.0,
                "add_source_as_reference": shot["audio_mode"] == "lock_source",
                "prompt_primary_audio_ordinal": 1 if sound == "voice" else 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
        }
        if bridge_id:
            apply_id = str(next_id)
            graph[apply_id] = {
                "class_type": "MiniMaxH3SemanticBridgeApplyT8",
                "inputs": {"conditioning": ["5", 0], "semantic_bridge": [bridge_id, 0]},
            }
            positive_id = apply_id
            next_id += 1

    condition_model_id = condition_id if relay_enabled else model_id
    fast_setup_id = None
    if fast_enabled:
        fast_setup_id = str(next_id)
        graph[fast_setup_id] = {
            "class_type": "MiniMaxH3FastH3V2SetupEXPT8",
            "inputs": {
                "model": [condition_model_id, 0],
                "av_latent": [latent_id, 2 if relay_enabled else 1],
                "profile": str(fast_cfg.get("profile", "trained_vsa_exp")),
                "min_tokens": int(fast_cfg.get("min_tokens", 12288)),
            },
        }
        next_id += 1

    sampler_inputs = {
        "model": [fast_setup_id, 0] if fast_setup_id else [condition_model_id, 0],
        "av_latent": [latent_id, 2 if relay_enabled else 1],
        "steps": 4 if lora_names and legacy_turbo_route else 8,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
    }
    if fast_setup_id:
        sampler_inputs.update({"sampler": [fast_setup_id, 1], "sigmas": [fast_setup_id, 2]})
    graph["6"] = {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": sampler_inputs} if not fast_setup_id else {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["8", 0],
            "guider": ["7", 0],
            "sampler": [fast_setup_id, 1],
            "sigmas": [fast_setup_id, 2],
            "latent_image": [latent_id, 2 if relay_enabled else 1],
        },
    }
    # FastH3 provides the complete sampler and sigmas, so the ordinary H3
    # DualClock node is bypassed and the custom sampler keeps the existing
    # ComfyUI queue contract.
    if fast_setup_id:
        graph["7"] = {"class_type": "BasicGuider", "inputs": {"model": [fast_setup_id, 0], "conditioning": [positive_id, 1 if relay_enabled else 0]}}
        graph["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
        graph["9"] = graph["6"]
        del graph["6"]
    else:
        graph["7"] = {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": [positive_id, 1 if relay_enabled else 0]}}
        graph["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
        graph["9"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["7", 0],
                "sampler": ["6", 1],
                "sigmas": ["6", 2],
                "latent_image": [latent_id, 2 if relay_enabled else 1],
            },
        }

    sampled_id = "9"
    if fast_setup_id:
        audit_id = str(next_id)
        graph[audit_id] = {
            "class_type": "MiniMaxH3FastH3V2RuntimeAuditEXPT8",
            "inputs": {"model": [fast_setup_id, 0], "sampled_av_latent": ["9", 0]},
        }
        sampled_id = audit_id
        next_id += 1
    graph["10"] = {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {"av_latent": [sampled_id, 0], "video_vae": ["2", 0], "audio_vae": ["3", 0]}}
    graph["11"] = {
        "class_type": "MiniMaxH3OutputTrimT8",
        "inputs": {
            "frames": ["10", 0],
            "audio": ["10", 1],
            "start_seconds": 0.0,
            "duration_seconds": shot["time"]["requested_seconds"],
            "frame_count": shot["time"]["delivery_trim_frames"],
            "fps": 24.0,
        },
    }
    graph["12"] = {
        "class_type": "MiniMaxH3SafeAVSaveT8Advanced",
        "inputs": {"images": ["11", 0], "audio": ["11", 1], "filename_prefix": f"T8_Director/{project['id'][:8]}/{shot_id[:8]}", "crf": 18,
                   "preserve_audio_length": True},
    }

    # For an audio-drive shot the conditioning node's mux_audio is the selected
    # windowed source track. Feeding the decoded generated track here would
    # silently replace the user's recording, which the Director contract
    # explicitly forbids. Native/reference-voice shots keep AVDecode audio.
    if sound == "record":
        graph["11"]["inputs"]["audio"] = [mux_audio_id, 3 if relay_enabled else 2]

    if next_id == 13 and not any((bridge_enabled, relay_enabled, fast_enabled, low_vram_enabled, chunk_ffn_enabled)):
        next_id = 14

    for input_name, role in (("first_frame", "first_frame"), ("last_frame", "last_frame")):
        item = _asset_for_role(shot, role)
        if item:
            graph[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": item["server_path"]}}
            graph[condition_id]["inputs"][input_name] = [str(next_id), 0]
            next_id += 1

    for index, item in enumerate(_assets_for_role(shot, "ref_image")):
        graph[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": item["server_path"]}}
        graph[condition_id]["inputs"][f"ref_images.ref_image_{index}"] = [str(next_id), 0]
        next_id += 1

    for index, item in enumerate(_assets_for_role(shot, "ref_video")):
        load_id = str(next_id)
        graph[load_id] = {"class_type": "LoadVideo", "inputs": {"file": item["server_path"]}}
        next_id += 1
        components_id = str(next_id)
        graph[components_id] = {
            "class_type": "GetVideoComponents",
            "inputs": {"video": [load_id, 0]},
        }
        graph[condition_id]["inputs"][f"ref_videos.ref_video_{index}"] = [components_id, 0]
        # A reference video's soundtrack is a separate native AUDIO slot.  The
        # compiler emits it immediately after the corresponding video entry.
        if any(a.get("asset_id") == item.get("asset_id") for a in _assets_for_role(shot, "ref_video_audio")):
            graph[condition_id]["inputs"][f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        next_id += 1

    for index, item in enumerate(_assets_for_role(shot, "ref_audio")):
        graph[str(next_id)] = {"class_type": "LoadAudio", "inputs": {"audio": item["server_path"]}}
        reference_audio = [str(next_id), 0]
        next_id += 1
        selection = shot.get("audio_selection") or {}
        if sound == "voice" and item["asset_id"] == selection.get("asset_id"):
            # The voice picker exposes a source selection too. Use that
            # exact range as the reference without locking the generated
            # audio or changing the requested video duration.
            graph[str(next_id)] = {
                "class_type": "TrimAudioDuration",
                "inputs": {"audio": reference_audio, "start_index": selection["start"],
                           "duration": selection["end"] - selection["start"]},
            }
            reference_audio = [str(next_id), 0]
            next_id += 1
        graph[condition_id]["inputs"][f"ref_audios.ref_audio_{index}"] = reference_audio

    drive = _asset_for_role(shot, "drive_audio")
    if drive:
        graph[str(next_id)] = {"class_type": "LoadAudio", "inputs": {"audio": drive["server_path"]}}
        load_id = str(next_id)
        next_id += 1
        selection = shot.get("audio_selection") or {}
        graph[str(next_id)] = {
            "class_type": "MiniMaxH3AudioWindowT8",
            "inputs": {
                "audio": [load_id, 0],
                "scene_start_seconds": selection.get("start", 0.0),
                "scene_duration_seconds": shot["time"]["requested_seconds"],
                "warmup_seconds": 0.0,
                "cooldown_seconds": 0.0,
                "ensure_minimum_context": True,
            },
        }
        graph[condition_id]["inputs"]["drive_audio"] = [str(next_id), 0]
        graph[condition_id]["inputs"]["final_audio"] = [str(next_id), 0]
        if sound == "record":
            # AudioWindow may left-pad a short recording to H3's minimum
            # context. Trim both picture and original audio from its exact
            # scene offset; zero would deliver the padding and cut the tail.
            graph["11"]["inputs"]["start_seconds"] = [str(next_id), 2]
        if not relay_enabled:
            graph[condition_id]["inputs"]["length"] = [str(next_id), 1]
        next_id += 1  # Reserve AudioWindow before appending middle-image nodes.

    endpoints = shot.get("keyframe_endpoints", {})
    endpoint_override = (endpoints.get("first_frame_index", -1) not in (-1, 0)
                         or endpoints.get("last_frame_index", -1) not in (-1, shot["time"]["aligned_frames"] - 1))
    if shot.get("middle_keyframes") or endpoint_override:
        # Reuse the existing native inputs and downstream graph, with explicit
        # MODEL/positive pairing. Ordinary prompts do not acquire Relay.
        previous = None
        for keyframe in shot.get("middle_keyframes", []):
            media = next(item for item in shot["media_map"] if item.get("keyframe_id") == keyframe["id"])
            image_id = str(next_id)
            graph[image_id] = {"class_type": "LoadImage", "inputs": {"image": media["server_path"]}}
            next_id += 1
            plan_id = str(next_id)
            inputs = {"image": [image_id, 0], "position_mode": "frame", "position": keyframe["frame_index"],
                "visual_noise_aug": keyframe["visual_noise_aug"], "resize_mode": keyframe["resize_mode"], "enabled": True}
            if previous is not None:
                inputs["previous_plan"] = [previous, 0]
            graph[plan_id] = {"class_type": "MiniMaxH3KeyframePlanT8Advanced", "inputs": inputs}
            previous = plan_id
            next_id += 1
        if not relay_enabled:
            # Stable AudioConditioning omits MODEL; this additive entry puts it
            # first, shifting its other outputs by one. Rewire exact links only.
            for node in graph.values():
                for key, value in node["inputs"].items():
                    if isinstance(value, list) and len(value) == 2 and value[0] == condition_id and type(value[1]) is int:
                        node["inputs"][key] = [condition_id, value[1] + 1]
            graph[fast_setup_id or "6"]["inputs"]["model"] = [condition_id, 0]
            graph[condition_id]["inputs"]["model"] = [model_id, 0]
        else:
            graph[condition_id]["inputs"].update(prompt=shot["prompt"], length=shot["time"]["aligned_frames"])
        graph[condition_id]["class_type"] = "MiniMaxH3DirectorKeyframeConditioningEXPT8"
        if previous is not None:
            graph[condition_id]["inputs"]["keyframe_plan"] = [previous, 0]
        graph[condition_id]["inputs"].update(endpoints)
        for key, default in {"first_frame_noise_aug": .999, "last_frame_noise_aug": .999,
                "reference_visual_noise_aug": .999, "execution_mode": "report_only", "query_chunk_rows": 256}.items():
            graph[condition_id]["inputs"].setdefault(key, default)
        report["warnings"].append({"shot_id": shot_id,
            "message": "关键帧已按实际交付位置接入条件；保持原采样和声音路由，GPU画质与组合效果仍需单独验收。"})

    if sampling["mode"] == "single" and shot.get("control_regions"):
        def add_control(kind, inputs):
            nonlocal next_id
            node_id = str(next_id)
            next_id += 1
            graph[node_id] = {"class_type": kind, "inputs": inputs}
            return node_id

        selected_model = graph[fast_setup_id or "6"]["inputs"]["model"]
        condition_inputs = graph[condition_id]["inputs"]
        controls = append_director_controls(graph, shot, stage="SINGLE",
            model=selected_model, positive=graph["7"]["inputs"]["conditioning"],
            width=condition_inputs["width"], height=condition_inputs["height"],
            length=control_condition_length(graph, condition_inputs), video_vae=["2", 0], add=add_control)
        graph[fast_setup_id or "6"]["inputs"]["model"] = controls["model"]
        graph["7"]["inputs"]["conditioning"] = controls["positive"]
        graph["10"]["inputs"]["av_latent"] = audit_control_latent(
            add_control, graph["10"]["inputs"]["av_latent"], controls["states"])

    two_pass = None
    if sampling["mode"] == "two_pass":
        two_pass = apply_two_pass_graph(graph, shot, sampling, store, int(seed), d3)
    elif sampling["mode"] == "hyperflow":
        two_pass = apply_hyperflow_graph(graph, shot, sampling, store, int(seed), d3)
    report["lora_loaders"] = two_pass.pop("lora_loaders") if two_pass else lora_loaders

    enabled_d3_routes = []
    if bridge_enabled:
        enabled_d3_routes.append("semantic_bridge")
    if relay_enabled:
        enabled_d3_routes.append("prompt_relay")
    if fast_enabled:
        enabled_d3_routes.append("fast_h3_v2")
    if low_vram_enabled:
        enabled_d3_routes.append("low_vram")
    if chunk_ffn_enabled:
        enabled_d3_routes.append("chunk_ffn")
    route_name = f"director_{task}_{sound}"
    if two_pass:
        route_name += "_" + (two_pass["recipe"] if two_pass["mode"] == "hyperflow" else "two_pass_4plus4")
    if enabled_d3_routes:
        route_name += "_" + "_".join(enabled_d3_routes)
    return {
        "prompt": graph,
        "report": report,
        "shot": shot,
        "recipe": route_name,
        "d3_routes": enabled_d3_routes,
        "seed": int(seed),
        "turbo_lora": lora_names or None,
        "sampling": two_pass or {"mode": "single"},
        "created_at": time.time(),
    }


async def queue_director_prompt(api_prompt: Mapping[str, Any], client_id: str | None = None, prompt_id: str | None = None) -> str:
    """Submit through Core's in-process queue, preserving normal validation."""

    from server import PromptServer
    import execution

    server = PromptServer.instance
    prompt_id = prompt_id or str(uuid.uuid4())
    prompt = server.trigger_on_prompt(dict(api_prompt))
    server.node_replace_manager.apply_replacements(prompt)
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        raise ValueError(valid[1])
    extra_data: dict[str, Any] = {"create_time": int(time.time() * 1000), "t8_director": True}
    if client_id:
        extra_data["client_id"] = client_id
    number = server.number
    server.number += 1
    server.prompt_queue.put((number, prompt_id, prompt, extra_data, valid[2], {}))
    return prompt_id


def director_job_status(prompt_id: str) -> dict[str, Any]:
    from server import PromptServer

    server = PromptServer.instance
    prompt_id = str(prompt_id)

    def progress_payload() -> dict[str, Any] | None:
        try:
            from comfy_execution.progress import get_progress_state

            registry = get_progress_state()
            if str(registry.prompt_id) != prompt_id:
                return None
            nodes = {
                str(node_id): {
                    "state": state["state"].value,
                    "value": float(state["value"]),
                    "max": float(state["max"]),
                }
                for node_id, state in registry.nodes.items()
            }
            total = sum(item["max"] for item in nodes.values())
            value = sum(min(item["value"], item["max"]) for item in nodes.values())
            return {
                "value": value,
                "max": total,
                "fraction": value / total if total else 0.0,
                "nodes": nodes,
            }
        except Exception:
            # Progress is an optional observation; queue/history state remains
            # authoritative when a Core version has no registry yet.
            return None

    record = server.prompt_queue.get_history(prompt_id=prompt_id).get(prompt_id)
    if record is not None:
        status = record.get("status") if isinstance(record, dict) else None
        status_str = status.get("status_str") if isinstance(status, dict) else None
        if not isinstance(status_str, str) or status_str not in {"success", "error"}:
            return {"prompt_id": prompt_id, "state": "unknown", "outputs": {}, "progress": progress_payload()}
        state = "success" if status_str == "success" else "error"
        if state == "error":
            messages = status.get("messages")
            events = {entry[0] for entry in (messages if isinstance(messages, (tuple, list)) else ())
                      if isinstance(entry, (tuple, list)) and len(entry) == 2
                      and isinstance(entry[0], str) and isinstance(entry[1], dict)
                      and str(entry[1].get("prompt_id")) == prompt_id}
            if "execution_interrupted" in events and "execution_error" not in events:
                state = "cancelled"
        return {"prompt_id": prompt_id, "state": state, "status": status, "outputs": record.get("outputs", {}), "progress": progress_payload()}
    running = server.prompt_queue.get_current_queue()
    if any(str(item[1]) == prompt_id for item in running[0]):
        return {"prompt_id": prompt_id, "state": "running", "progress": progress_payload()}
    if any(str(item[1]) == prompt_id for item in running[1]):
        return {"prompt_id": prompt_id, "state": "queued", "progress": None}
    return {"prompt_id": prompt_id, "state": "unknown", "progress": None}


def cancel_director_prompt(prompt_id: str) -> dict[str, Any]:
    from server import PromptServer

    server = PromptServer.instance
    prompt_id = str(prompt_id)
    deleted = server.prompt_queue.delete_queue_item(lambda item: str(item[1]) == prompt_id)
    interrupted = False if deleted else server.prompt_queue.interrupt_if_running(prompt_id)
    return {"prompt_id": prompt_id, "deleted_from_queue": bool(deleted), "interrupted": bool(interrupted)}
