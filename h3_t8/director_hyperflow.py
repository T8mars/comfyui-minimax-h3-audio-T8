"""Opt-in Director graph compilation for the dedicated HyperFlow runtime.

These recipes never replace the legacy single sampler or standard 4+4 graph.
Each phase rebuilds its own MODEL/conditioning pair; media quality is separate.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import folder_paths

from .director_sampling_settings import lora_interpretation
from .director_two_pass import _pick_upscaler
from .h3_weight_diagnostics import inspect_h3_weight_file
from .nodes_hyperflow_advanced import _resolve
from .hyperflow_weights_advanced import DEFAULT_RAW_GRID
from .director_control_graph import (
    append_director_controls, audit_control_latent, audit_control_head, control_condition_length,
)


def apply_hyperflow_graph(graph, shot, sampling, store, seed, d3):
    if graph.get("6", {}).get("class_type") != "MiniMaxH3DualClockSamplerT8":
        raise ValueError("HyperFlow 需要接管实际 SAMPLER/SIGMAS；另一独立采样器没有联合网格协议")
    # The Loader validates actual backbone structures and adapter tensor
    # targets. A filename alone cannot establish or refute full-H3 support.
    try:
        original = _resolve(sampling["hyperflow_file"])
    except FileNotFoundError as error:
        raise ValueError("HyperFlow 原始权重未安装，请在独立加载器目录中选择文件") from error
    if not original.is_file():
        raise ValueError("HyperFlow 原始权重未安装")
    variant = sampling["variant"]
    next_id = max(map(int, graph)) + 1
    lora_loaders = []

    def add(kind, inputs):
        nonlocal next_id
        node = str(next_id)
        next_id += 1
        graph[node] = {"class_type": kind, "inputs": inputs}
        return node

    def stage_model(stage):
        model = "1"
        for row_index, row in enumerate(sampling[f"{stage}_loras"], 1):
            if not row["enabled"]:
                continue
            name = row["name"]
            content_path = folder_paths.get_full_path("loras", name)
            if name not in set(folder_paths.get_filename_list("loras")) or not content_path:
                raise ValueError(f"HyperFlow {'一采' if stage == 'low' else '二采'}找不到内容 LoRA：{name}")
            # This is a known incompatible role, not a generic unknown-LoRA
            # whitelist: the two-time adapter must be applied only by its
            # dedicated loader. A copied file may have a different path, so
            # inspect the bounded header as well as the exact selection.
            report = inspect_h3_weight_file(content_path)
            special = report.get("special_runtime_requirements", ())
            if (Path(content_path).resolve() == original.resolve()
                    or any(item.get("kind") == "hyperflow" for item in special)):
                raise ValueError("HyperFlow 原始权重不能放在一采/二采内容 LoRA 列表；请只在专用 HyperFlow 权重栏选择")
            model = add("MiniMaxH3LoRACompatibilityLoaderT8Advanced", {
                "model": [model, 0], "lora_name": name, "strength_model": row["strength"],
                **lora_interpretation(row)})
            lora_loaders.append({"node_id": model, "stage": stage.upper(),
                "row_index": row_index, "row_id": row.get("id"), "source": "sampling_rows"})
        memory = d3["memory"]
        if memory.get("low_vram"):
            model = add("MiniMaxH3LowVRAMAttentionT8Advanced", {
                "model": [model, 0], "head_chunks": int(memory.get("head_chunks", 4))})
        if memory.get("chunk_ffn"):
            model = add("MiniMaxH3ChunkFeedForwardT8Advanced", {
                "model": [model, 0], "chunks": int(memory.get("chunks", 2)),
                "seq_threshold": int(memory.get("seq_threshold", 4096))})
        return add("MiniMaxH3HyperFlowFreshLoaderEXPT8", {
            "model": [model, 0], "hyperflow_file": sampling["hyperflow_file"]})

    positive = graph["7"]["inputs"]["conditioning"]
    factory_id = positive[0]
    bridge_id = None
    if graph[factory_id]["class_type"] == "MiniMaxH3SemanticBridgeApplyT8":
        bridge_id = factory_id
        factory_id = graph[factory_id]["inputs"]["conditioning"][0]
    factory_kind = graph[factory_id]["class_type"]
    if factory_kind not in {"MiniMaxH3AudioConditioningT8", "MiniMaxH3PromptRelayConditioningT8Advanced",
                            "MiniMaxH3DirectorKeyframeConditioningEXPT8"}:
        raise ValueError("HyperFlow 缺少可重建的原生条件输入，不能丢弃未知条件接线")
    base_inputs = deepcopy(graph[factory_id]["inputs"])
    base_relay_plan = (deepcopy(graph[base_inputs["prompt_relay_plan"][0]]["inputs"])
                       if "prompt_relay_plan" in base_inputs else None)
    bridge_inputs = deepcopy(graph[bridge_id]["inputs"]) if bridge_id else None
    returns_model = factory_kind != "MiniMaxH3AudioConditioningT8"

    def condition(stage, model, *, width=None, height=None, reuse=False, controls=True):
        """Clone the native factory, not a previously encoded condition tensor."""
        inputs = deepcopy(base_inputs)
        if returns_model:
            inputs["model"] = [model, 0]
        if width is not None:
            inputs.update(width=width, height=height)
            if isinstance(width, int) and isinstance(height, int):
                for key in ("first_frame", "last_frame"):
                    asset = next((item for item in shot["media_map"] if item["role"] == key), None)
                    if key in inputs and asset:
                        prepared = store.prepare_image(asset["asset_id"], width, height)
                        inputs[key] = [add("LoadImage", {"image": prepared["server_path"]}), 0]
        if "prompt_relay_plan" in inputs and "stage_relay" in sampling:
            from .prompt_relay_advanced import build_prompt_relay_plan

            plan_inputs = deepcopy(base_relay_plan)
            selected = sampling["stage_relay"]["head" if stage == "low" else "tail"]
            if selected["mode"] == "custom":
                for key in ("global_prompt", "local_prompts", "timing_mode", "time_ranges"):
                    plan_inputs[key] = selected[key]
                build_prompt_relay_plan(**plan_inputs)
            if reuse:
                graph[base_inputs["prompt_relay_plan"][0]]["inputs"] = plan_inputs
            else:
                inputs["prompt_relay_plan"] = [add("MiniMaxH3PromptRelayPlanT8Advanced", plan_inputs), 0]
        if reuse:
            node_id = factory_id
            graph[node_id]["inputs"] = inputs
        else:
            node_id = add(factory_kind, inputs)
        positive_link = [node_id, 1 if returns_model else 0]
        if bridge_inputs is not None:
            selected_bridge = deepcopy(bridge_inputs)
            selected_bridge["conditioning"] = positive_link
            if reuse:
                graph[bridge_id]["inputs"] = selected_bridge
                selected_id = bridge_id
            else:
                selected_id = add("MiniMaxH3SemanticBridgeApplyT8", selected_bridge)
            positive_link = [selected_id, 0]
        result = {"model": [node_id, 0] if returns_model else [model, 0],
                  "positive": positive_link, "latent": [node_id, 2 if returns_model else 1],
                  "mux": [node_id, 3 if returns_model else 2], "states": [], "inputs": inputs}
        if controls:
            phase = "SINGLE" if variant == "single8" else stage.upper()
            result.update(append_director_controls(graph, shot, stage=phase,
                model=result["model"], positive=positive_link,
                width=inputs["width"], height=inputs["height"], length=control_condition_length(graph, inputs),
                video_vae=["2", 0], add=add))
        return result

    checkpoint = sampling.get("stage_checkpoint", {"mode": "off"})
    resume_tail = variant == "continuous4plus4separate" and checkpoint["mode"] == "resume_tail"
    low_model = None if resume_tail else stage_model("low")
    upscale = variant in {"upscale8plus4", "upscale4plus4"}
    low_condition = None
    if not resume_tail:
        if upscale:
            canvas = shot["two_pass_canvas"]
            if not canvas:
                raise ValueError("HyperFlow 低高清路线缺少独立 LOW 尺寸计划")
            low_condition = condition("low", low_model, width=canvas["low_width"], height=canvas["low_height"])
        else:
            low_condition = condition("low", low_model, reuse=True)
        positive, latent = low_condition["positive"], low_condition["latent"]
    if variant in {"single8", "upscale8plus4", "upscale4plus4"}:
        plan_id = add("MiniMaxH3HyperFlowHeadPlanT8Advanced" if variant == "upscale4plus4"
                      else "MiniMaxH3HyperFlowPlanT8Advanced", {"model": low_condition["model"]})
        setup = add("MiniMaxH3HyperFlowCoarseSamplerT8Advanced" if variant == "upscale4plus4"
                    else "MiniMaxH3HyperFlowSamplerT8Advanced", {
            "model": low_condition["model"], "av_latent": latent, "hyperflow_plan": [plan_id, 0]})
        graph["7"]["inputs"].update(model=[setup, 0], conditioning=positive)
        graph["9"]["inputs"].update(guider=["7", 0], sampler=[setup, 1], sigmas=[setup, 2], latent_image=latent)
        if upscale:
            upscaler = _pick_upscaler(sampling["upscaler"])
            resized = add("MiniMaxH3LearnedLatentUpscaleT8Advanced", {
                # Core's nonterminal sampler output 0 is x_sigma/latent view.
                # Its output 1 is the predicted clean x0 required by the
                # trained 3D upscaler; full8 has already reached clean x0.
                "av_latent": audit_control_latent(add,
                    ["9", 1 if variant == "upscale4plus4" else 0], low_condition["states"]),
                "model_name": upscaler,
                "size_mode": "target_megapixels", "scale_by": 2.0,
                "target_megapixels": shot["two_pass_canvas"]["actual_megapixels"],
                "target_width": shot["two_pass_canvas"]["width"],
                "target_height": shot["two_pass_canvas"]["height"],
                "aspect_policy": "preserve_source", "max_anisotropy": 1.05,
                "precision": "fp16", "release_policy": "offload_after"})
            high_model = stage_model("high")
            high_condition = condition("high", high_model, width=[resized, 1], height=[resized, 2],
                                       reuse=True, controls=False)
            reconciled = add("MiniMaxH3TwoPassLatentReconcileT8Advanced", {
                "learned_latent": [resized, 0], "highres_template": high_condition["latent"],
                "positive": high_condition["positive"], "audio_policy": "auto",
                "second_pass_audio_source": "legacy_policy", "second_pass_audio_strength": 0.0})
            high_condition.update(append_director_controls(graph, shot, stage="HIGH",
                model=high_condition["model"], positive=[reconciled, 1],
                width=[resized, 1], height=[resized, 2],
                length=control_condition_length(graph, high_condition["inputs"]),
                video_vae=["2", 0], add=add))
            high_plan = add("MiniMaxH3HyperFlowTailPlanT8Advanced", {"model": high_condition["model"]})
            high_setup = add("MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced" if variant == "upscale4plus4"
                             else "MiniMaxH3HyperFlowRefineSamplerT8Advanced", {
                "model": high_condition["model"], "av_latent": [reconciled, 0], "hyperflow_plan": [high_plan, 0]})
            high_guider = add("BasicGuider", {"model": [high_setup, 0], "conditioning": high_condition["positive"]})
            high_noise = add("RandomNoise", {"noise_seed": (seed + 1) % (2**64)})
            high_sampler = add("SamplerCustomAdvanced", {
                "noise": [high_noise, 0], "guider": [high_guider, 0],
                "sampler": [high_setup, 1], "sigmas": [high_setup, 2], "latent_image": [reconciled, 0]})
            graph["10"]["inputs"]["av_latent"] = audit_control_latent(add,
                [high_sampler, 0], high_condition["states"])
            if shot["delivery_audio"] == "original_selected_recording":
                graph["11"]["inputs"]["audio"] = high_condition["mux"]
            recipe = ("hyperflow4plus4_partial_x0_upscale_exp_v1" if variant == "upscale4plus4"
                      else "hyperflow8plus4_new_noise_upscale_exp_v1")
        else:
            recipe = "hyperflow8_single_v1"
            graph["10"]["inputs"]["av_latent"] = audit_control_latent(
                add, ["9", 0], low_condition["states"])
    elif variant == "continuous4plus4":
        high_model = stage_model("high")
        high_condition = condition("high", high_model)
        graph["9"] = {"class_type": "MiniMaxH3HyperFlowSplitT8Advanced", "inputs": {
            "model_low": low_condition["model"], "model_high": high_condition["model"],
            "positive": positive, "positive_high": high_condition["positive"], "av_latent": latent,
            "seed": seed, "split_interval": 4, "cfg": 1.0}}
        recipe = "hyperflow8_continuous_split_exp_v1"
    elif variant == "continuous4plus4separate":
        high_model = stage_model("high")
        head_model = None if resume_tail else low_condition["model"]
        high_condition = condition("high", high_model)
        tail_model, high_positive = high_condition["model"], high_condition["positive"]
        if shot["delivery_audio"] == "original_selected_recording":
            graph["11"]["inputs"]["audio"] = high_condition["mux"]
        if resume_tail:
            from .modular_sampling.hyperflow_storage import fingerprint

            storage_root = Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "hyperflow_stage_artifacts"
            actual = fingerprint(storage_root, checkpoint["artifact_path"], "head")
            if actual != checkpoint["artifact_sha256"]:
                raise ValueError("仅 TAIL 恢复的 HEAD 文件 SHA256 不匹配")
            source_boundary = [add("MiniMaxH3HyperFlowHeadLoadEXPT8", {
                "artifact_path": checkpoint["artifact_path"],
                "artifact_sha256": checkpoint["artifact_sha256"]}), 0]
        else:
            head_bind = add("MiniMaxH3HyperFlowHeadEffectsBindEXPT8", {
                "model": head_model, "av_latent": latent, "positive": positive,
                "negative": positive, "split_interval": 4})
            head_config = add("MiniMaxH3StageEAVConfigEXPT8", sampling["stage_eav"]["head"])
            head_effect = add("MiniMaxH3StageEAVApplyEXPT8", {
                "model": [head_bind, 0], "av_latent": [head_bind, 3],
                "sigmas": [head_bind, 4], "stage_context": [head_bind, 5],
                "eav_config": [head_config, 0]})
            graph["9"] = {"class_type": "MiniMaxH3HyperFlowHeadStageEXPT8", "inputs": {
                "model": [head_effect, 0], "av_latent": latent, "noise": ["8", 0],
                "positive": [head_bind, 1], "negative": [head_bind, 2], "split_interval": 4,
                "cfg": 1.0, "reserve_vram_mib": 1024}}
            head_audit = add("MiniMaxH3HyperFlowHeadEffectsAuditEXPT8", {
                "continuous_boundary": ["9", 0]})
            source_boundary = audit_control_head(add, [head_audit, 0], low_condition["states"])
            if checkpoint["mode"] == "save":
                source_boundary = [add("MiniMaxH3HyperFlowHeadSaveEXPT8", {
                    "continuous_boundary": source_boundary, "prefix": "T8_Director/HyperFlow/HEAD"}), 0]
        tail_bind = add("MiniMaxH3HyperFlowTailEffectsBindEXPT8", {
            "continuous_boundary": source_boundary, "model": tail_model,
            "positive": high_positive, "negative": high_positive})
        tail_config = add("MiniMaxH3StageEAVConfigEXPT8", sampling["stage_eav"]["tail"])
        tail_effect = add("MiniMaxH3StageEAVApplyEXPT8", {
            "model": [tail_bind, 0], "av_latent": [tail_bind, 3],
            "sigmas": [tail_bind, 4], "stage_context": [tail_bind, 5],
            "eav_config": [tail_config, 0]})
        tail = add("MiniMaxH3HyperFlowTailStageEXPT8", {
            "continuous_boundary": source_boundary, "model": [tail_effect, 0],
            "positive": [tail_bind, 1], "negative": [tail_bind, 2],
            "seed": seed, "cfg": 1.0, "reserve_vram_mib": 1024})
        tail_audit = add("MiniMaxH3HyperFlowTailEffectsAuditEXPT8", {
            "completed_result": [tail, 2]})
        graph["10"]["inputs"]["av_latent"] = audit_control_latent(
            add, [tail_audit, 1], high_condition["states"])
        recipe = ("hyperflow8_continuous_separate_tail_resume_exp_v1" if resume_tail
                  else "hyperflow8_continuous_separate_stages_exp_v1")
        if resume_tail:
            used = set()

            def include(node_id):
                if node_id in used:
                    return
                used.add(node_id)
                for value in graph[node_id]["inputs"].values():
                    if (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
                            and isinstance(value[1], int) and value[0] in graph):
                        include(value[0])

            include("12")
            for node_id in list(graph):
                if node_id not in used:
                    del graph[node_id]
    else:
        raise ValueError("未知 HyperFlow 实验路线")
    # In the split route the old sampler's output is no longer selected;
    # unreachable standard nodes remain in the graph as harmless provenance.
    if variant == "continuous4plus4":
        graph["10"]["inputs"]["av_latent"] = audit_control_latent(
            add, ["9", 0], low_condition["states"] + high_condition["states"])
    intervals = {
        "single8": ((0, 8),),
        "continuous4plus4": ((0, 4), (4, 8)),
        "continuous4plus4separate": ((0, 4), (4, 8)),
        "upscale8plus4": ((0, 8), (4, 8)),
        "upscale4plus4": ((0, 4), (4, 8)),
    }[variant]
    if resume_tail:
        intervals = ((4, 8),)
    trained_grid = {"raw_sigmas": DEFAULT_RAW_GRID, "video_shift": 12.0, "audio_shift": 3.0}
    grid_digest = hashlib.sha256(json.dumps(trained_grid, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"mode": "hyperflow", "variant": variant, "recipe": recipe,
            "hyperflow_file": sampling["hyperflow_file"],
            "trained_grid_contract": trained_grid,
            "trained_grid_contract_sha256": grid_digest,
            "absolute_intervals": intervals,
            "stage_nfe": [stop - start for start, stop in intervals],
            "total_nfe": sum(stop - start for start, stop in intervals),
            "grid_identity_note": "expected_v1_contract; runtime loader validates selected file metadata; frozen batches separately hash model bytes",
            "low_loras": [] if resume_tail else [row for row in sampling["low_loras"] if row["enabled"]],
            "high_loras": [row for row in sampling["high_loras"] if row["enabled"]] if variant != "single8" else [],
            "quality_status": "experimental_not_human_reviewed", "lora_loaders": lora_loaders}
