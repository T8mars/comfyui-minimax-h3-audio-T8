"""Compile the Director's LOW → learned 3D → HIGH AV recipe from native nodes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import folder_paths

from .director_sampling_settings import lora_interpretation
from .director_control_graph import append_director_controls, audit_control_latent, control_condition_length


UPSCALE_CANDIDATES = ("minimax_h3_latent_upscaler_3d_fp16.safetensors",)


def _pick_upscaler(requested: str) -> str:
    choices = set(folder_paths.get_filename_list("latent_upscale_models"))
    name = next((candidate for candidate in UPSCALE_CANDIDATES if candidate in choices), None) if requested == "auto" else requested
    if not name or name not in choices or not folder_paths.get_full_path("latent_upscale_models", name):
        raise ValueError("双采缺少所选学习型 3D 放大模型，请在 models/latent_upscale_models 安装后重试")
    return name


def apply_two_pass_graph(
    graph: dict[str, dict[str, Any]],
    shot: dict[str, Any],
    sampling: dict[str, Any],
    store: Any,
    seed: int,
    d3: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replace the existing one-pass sampler with two native passes.

    The first pass returns denoised_output, which is a partial prediction for
    learned latent resizing. Only the second pass output reaches AVDecode.
    """
    if "6" not in graph or graph["6"]["class_type"] != "MiniMaxH3DualClockSamplerT8":
        raise ValueError("当前 FastH3 路线不能使用标准 4+4 双采；请关闭 FastH3")
    plan = shot["two_pass_canvas"]
    if not plan:
        raise ValueError("双采缺少实际 LOW/HIGH 尺寸计划")
    upscaler_name = _pick_upscaler(sampling["upscaler"])
    next_id = max(map(int, graph)) + 1
    lora_loaders = []

    def add(kind: str, inputs: dict[str, Any]) -> str:
        nonlocal next_id
        node_id = str(next_id)
        next_id += 1
        graph[node_id] = {"class_type": kind, "inputs": inputs}
        return node_id

    def stage_model(stage: str) -> str:
        model = "1"
        for row_index, row in enumerate(sampling[f"{stage}_loras"], 1):
            if not row["enabled"]:
                continue
            name = row["name"]
            if name not in set(folder_paths.get_filename_list("loras")) or not folder_paths.get_full_path("loras", name):
                raise ValueError(f"{'一采' if stage == 'low' else '二采'}找不到 LoRA：{name}")
            model = add("MiniMaxH3LoRACompatibilityLoaderT8Advanced", {
                "model": [model, 0], "lora_name": name, "strength_model": row["strength"],
                **lora_interpretation(row),
            })
            lora_loaders.append({"node_id": model, "stage": stage.upper(),
                "row_index": row_index, "row_id": row.get("id"), "source": "sampling_rows"})
        memory = d3["memory"]
        if memory.get("low_vram"):
            model = add("MiniMaxH3LowVRAMAttentionT8Advanced", {
                "model": [model, 0], "head_chunks": int(memory.get("head_chunks", 4)),
            })
        if memory.get("chunk_ffn"):
            model = add("MiniMaxH3ChunkFeedForwardT8Advanced", {
                "model": [model, 0], "chunks": int(memory.get("chunks", 2)),
                "seq_threshold": int(memory.get("seq_threshold", 4096)),
            })
        return model

    low_model, high_model = stage_model("low"), stage_model("high")
    relay = bool(d3["prompt_relay"].get("enabled"))
    middle = any(node["class_type"] == "MiniMaxH3DirectorKeyframeConditioningEXPT8" for node in graph.values())
    returns_model = relay or middle
    bridge = bool(d3["semantic_bridge"].get("enabled"))
    high_condition = next(
        (node_id for node_id, node in graph.items()
         if node["class_type"] == ("MiniMaxH3DirectorKeyframeConditioningEXPT8" if middle else
                                   "MiniMaxH3PromptRelayConditioningT8Advanced" if relay else "MiniMaxH3AudioConditioningT8")),
        None,
    )
    if high_condition is None:
        raise ValueError("双采未找到 HIGH 条件节点")
    high_inputs = graph[high_condition]["inputs"]
    if returns_model:
        high_inputs["model"] = [high_model, 0]
    high_bridge = next(
        (node_id for node_id, node in graph.items()
         if node["class_type"] == "MiniMaxH3SemanticBridgeApplyT8" and node["inputs"].get("conditioning") == [high_condition, 1 if returns_model else 0]),
        None,
    ) if bridge and not relay else None
    low_inputs = deepcopy(high_inputs)
    low_inputs["width"], low_inputs["height"] = plan["low_width"], plan["low_height"]
    if returns_model:
        low_inputs["model"] = [low_model, 0]
    for key, role in (("first_frame", "first_frame"), ("last_frame", "last_frame")):
        if key not in low_inputs:
            continue
        asset = next((item for item in shot["media_map"] if item["role"] == role), None)
        if asset:
            prepared = store.prepare_image(asset["asset_id"], plan["low_width"], plan["low_height"])
            low_inputs[key] = [add("LoadImage", {"image": prepared["server_path"]}), 0]
    low_condition = add(graph[high_condition]["class_type"], low_inputs)
    low_positive = low_condition
    if high_bridge:
        bridge_inputs = deepcopy(graph[high_bridge]["inputs"])
        bridge_inputs["conditioning"] = [low_condition, 1 if returns_model else 0]
        low_positive = add("MiniMaxH3SemanticBridgeApplyT8", bridge_inputs)
    low_model_output = [low_condition, 0] if returns_model else [low_model, 0]
    high_model_output = [high_condition, 0] if returns_model else [high_model, 0]
    low_latent = [low_condition, 2 if returns_model else 1]
    high_latent = [high_condition, 2 if returns_model else 1]
    low_positive_output = [low_positive, 1 if returns_model and not high_bridge else 0]
    high_positive_output = [high_bridge or high_condition, 1 if returns_model and not high_bridge else 0]
    low_controls = append_director_controls(graph, shot, stage="LOW",
        model=low_model_output, positive=low_positive_output,
        width=low_inputs["width"], height=low_inputs["height"], length=control_condition_length(graph, low_inputs),
        video_vae=["2", 0], add=add)
    low_model_output, low_positive_output = low_controls["model"], low_controls["positive"]

    graph["6"]["inputs"].update({
        "model": low_model_output, "av_latent": low_latent,
        "steps": 8, "shift_video": 12.0, "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler", "scheduler": "native_flow",
    })
    parity = add("MiniMaxH3LearnedTwoPassParityPlanT8Advanced", {
        "model": ["6", 0], "base_steps": 8, "coarse_steps": 4, "refine_steps": 4,
    })
    graph["7"]["inputs"].update({"model": ["6", 0], "conditioning": low_positive_output})
    graph["9"]["inputs"].update({
        "guider": ["7", 0], "sampler": ["6", 1], "sigmas": [parity, 0],
        "latent_image": low_latent,
    })
    upscaler = add("MiniMaxH3LearnedLatentUpscaleT8Advanced", {
        "av_latent": audit_control_latent(add, ["9", 1], low_controls["states"]), "model_name": upscaler_name,
        "size_mode": "target_megapixels", "scale_by": 2.0,
        "target_megapixels": plan["actual_megapixels"],
        "target_width": plan["width"], "target_height": plan["height"],
        "aspect_policy": "preserve_source", "max_anisotropy": 1.05,
        "precision": "fp16", "release_policy": "offload_after",
    })
    high_inputs["width"], high_inputs["height"] = [upscaler, 1], [upscaler, 2]
    reconciled = add("MiniMaxH3TwoPassLatentReconcileT8Advanced", {
        "learned_latent": [upscaler, 0],
        "highres_template": high_latent,
        "positive": high_positive_output,
        "audio_policy": "auto",
        "second_pass_audio_source": "legacy_policy",
        "second_pass_audio_strength": 0.0,
    })
    high_controls = append_director_controls(graph, shot, stage="HIGH",
        model=high_model_output, positive=[reconciled, 1],
        width=high_inputs["width"], height=high_inputs["height"], length=control_condition_length(graph, high_inputs),
        video_vae=["2", 0], add=add)
    mixer = add("MiniMaxH3TwoPassDetailMixerT8Advanced", {
        "model": high_controls["model"], "av_latent": [reconciled, 0],
        "refine_sigmas": [parity, 1], "shift_video": 12.0, "shift_audio": 3.0,
        "enable_tail": False, "extra_tail_steps": 3,
        "tail_spacing": "video_sigma_linear",
        "enable_model_time_bias": False, "bias": -0.025,
        "bias_start_progress": 0.7, "bias_end_progress": 0.95,
        "bias_domain": "video_sigma", "enable_stg": False, "stg_scale": 0.35,
        "stg_double_blocks": "25", "stg_start_progress": 0.25,
        "stg_end_progress": 0.85, "enable_restart": False,
        "restart_video_sigma": 0.15, "restart_steps": 3,
        "restart_seed": int(seed),
    })
    high_guider = add("BasicGuider", {"model": [mixer, 0], "conditioning": high_controls["positive"]})
    high_noise = add("RandomNoise", {"noise_seed": int(seed)})
    high_sampler = add("SamplerCustomAdvanced", {
        "noise": [high_noise, 0], "guider": [high_guider, 0],
        "sampler": [mixer, 1], "sigmas": [mixer, 2],
        "latent_image": [reconciled, 0],
    })
    graph["10"]["inputs"]["av_latent"] = audit_control_latent(add, [high_sampler, 0], high_controls["states"])
    if shot["delivery_audio"] == "original_selected_recording":
        graph["11"]["inputs"]["audio"] = [high_condition, 3 if returns_model else 2]
    return {
        "mode": "two_pass", "preset": sampling["preset"],
        "low": {"width": plan["low_width"], "height": plan["low_height"],
                "loras": [row for row in sampling["low_loras"] if row["enabled"]]},
        "high": {"width": plan["width"], "height": plan["height"],
                 "loras": [row for row in sampling["high_loras"] if row["enabled"]]},
        "upscaler": upscaler_name, "high_output_node": high_sampler,
        "lora_loaders": lora_loaders,
    }
