"""Opt-in Director graph compilation for the dedicated HyperFlow runtime.

These recipes never replace the legacy single sampler or standard 4+4 graph.
Only T2VA/native without D3 model mutations has been structurally qualified.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import folder_paths

from .director_two_pass import _pick_upscaler
from .h3_weight_diagnostics import inspect_h3_weight_file
from .nodes_hyperflow_advanced import _resolve
from .hyperflow_weights_advanced import DEFAULT_RAW_GRID


def apply_hyperflow_graph(graph, shot, sampling, store, seed, d3):
    if shot["task_type"].lower() != "t2va" or shot["audio_mode"] != "native":
        raise ValueError("HyperFlow EXP 目前仅验证 T2VA 原生声音；其他镜头请选择现有采样路线")
    if graph["6"]["class_type"] != "MiniMaxH3DualClockSamplerT8":
        raise ValueError("HyperFlow EXP 不能与 FastH3 或另一采样器叠加")
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

    def add(kind, inputs):
        nonlocal next_id
        node = str(next_id)
        next_id += 1
        graph[node] = {"class_type": kind, "inputs": inputs}
        return node

    def stage_model(stage):
        model = "1"
        for row in sampling[f"{stage}_loras"]:
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
                "model": [model, 0], "lora_name": name, "strength_model": row["strength"]})
        memory = d3["memory"]
        if memory.get("low_vram"):
            model = add("MiniMaxH3LowVRAMAttentionT8Advanced", {
                "model": [model, 0], "head_chunks": int(memory.get("head_chunks", 4))})
        if memory.get("chunk_ffn"):
            model = add("MiniMaxH3ChunkFeedForwardT8Advanced", {
                "model": [model, 0], "chunks": int(memory.get("chunks", 2)),
                "seq_threshold": int(memory.get("seq_threshold", 4096))})
        return add("MiniMaxH3HyperFlowLoaderT8Advanced", {
            "model": [model, 0], "hyperflow_file": sampling["hyperflow_file"]})

    positive = graph["7"]["inputs"]["conditioning"]
    latent = graph["9"]["inputs"]["latent_image"]
    checkpoint = sampling.get("stage_checkpoint", {"mode": "off"})
    resume_tail = variant == "continuous4plus4separate" and checkpoint["mode"] == "resume_tail"
    low_model = None if resume_tail else stage_model("low")
    upscale = variant in {"upscale8plus4", "upscale4plus4"}
    if variant in {"single8", "upscale8plus4", "upscale4plus4"}:
        low_condition = positive
        if upscale:
            plan = shot["two_pass_canvas"]
            if not plan or graph["5"]["class_type"] != "MiniMaxH3AudioConditioningT8":
                raise ValueError("HyperFlow 低高清路线缺少独立 LOW 尺寸计划")
            low_inputs = deepcopy(graph["5"]["inputs"])
            low_inputs["width"], low_inputs["height"] = plan["low_width"], plan["low_height"]
            low_condition_id = add("MiniMaxH3AudioConditioningT8", low_inputs)
            low_condition, latent = [low_condition_id, 0], [low_condition_id, 1]
            bridge = next((node for node in graph.values()
                           if node["class_type"] == "MiniMaxH3SemanticBridgeApplyT8"
                           and node["inputs"].get("conditioning") == ["5", 0]), None)
            if bridge is not None:
                inputs = deepcopy(bridge["inputs"])
                inputs["conditioning"] = low_condition
                low_condition = [add("MiniMaxH3SemanticBridgeApplyT8", inputs), 0]
        plan_id = add("MiniMaxH3HyperFlowHeadPlanT8Advanced" if variant == "upscale4plus4"
                      else "MiniMaxH3HyperFlowPlanT8Advanced", {"model": [low_model, 0]})
        setup = add("MiniMaxH3HyperFlowCoarseSamplerT8Advanced" if variant == "upscale4plus4"
                    else "MiniMaxH3HyperFlowSamplerT8Advanced", {
            "model": [low_model, 0], "av_latent": latent, "hyperflow_plan": [plan_id, 0]})
        graph["7"]["inputs"].update(model=[setup, 0], conditioning=low_condition)
        graph["9"]["inputs"].update(guider=["7", 0], sampler=[setup, 1], sigmas=[setup, 2], latent_image=latent)
        if upscale:
            upscaler = _pick_upscaler(sampling["upscaler"])
            resized = add("MiniMaxH3LearnedLatentUpscaleT8Advanced", {
                # Core's nonterminal sampler output 0 is x_sigma/latent view.
                # Its output 1 is the predicted clean x0 required by the
                # trained 3D upscaler; full8 has already reached clean x0.
                "av_latent": ["9", 1 if variant == "upscale4plus4" else 0],
                "model_name": upscaler,
                "size_mode": "target_megapixels", "scale_by": 2.0,
                "target_megapixels": shot["two_pass_canvas"]["actual_megapixels"],
                "target_width": shot["two_pass_canvas"]["width"],
                "target_height": shot["two_pass_canvas"]["height"],
                "aspect_policy": "preserve_source", "max_anisotropy": 1.05,
                "precision": "fp16", "release_policy": "offload_after"})
            high_model = stage_model("high")
            high_plan = add("MiniMaxH3HyperFlowTailPlanT8Advanced", {"model": [high_model, 0]})
            high_setup = add("MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced" if variant == "upscale4plus4"
                             else "MiniMaxH3HyperFlowRefineSamplerT8Advanced", {
                "model": [high_model, 0], "av_latent": [resized, 0], "hyperflow_plan": [high_plan, 0]})
            high_guider = add("BasicGuider", {"model": [high_setup, 0], "conditioning": positive})
            high_noise = add("RandomNoise", {"noise_seed": (seed + 1) % (2**64)})
            high_sampler = add("SamplerCustomAdvanced", {
                "noise": [high_noise, 0], "guider": [high_guider, 0],
                "sampler": [high_setup, 1], "sigmas": [high_setup, 2], "latent_image": [resized, 0]})
            graph["10"]["inputs"]["av_latent"] = [high_sampler, 0]
            recipe = ("hyperflow4plus4_partial_x0_upscale_exp_v1" if variant == "upscale4plus4"
                      else "hyperflow8plus4_new_noise_upscale_exp_v1")
        else:
            recipe = "hyperflow8_single_v1"
    elif variant == "continuous4plus4":
        high_model = stage_model("high")
        graph["9"] = {"class_type": "MiniMaxH3HyperFlowSplitT8Advanced", "inputs": {
            "model_low": [low_model, 0], "model_high": [high_model, 0],
            "positive": positive, "av_latent": latent,
            "seed": seed, "split_interval": 4, "cfg": 1.0}}
        recipe = "hyperflow8_continuous_split_exp_v1"
    elif variant == "continuous4plus4separate":
        high_model = stage_model("high")
        # The old combined route above remains unchanged. Only this opt-in
        # branch gets independently paired Relay/conditioning and EAV owners.
        head_model = None if resume_tail else [low_model, 0]
        tail_model = [high_model, 0]
        source = graph.get(positive[0])
        if source is not None and source["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced":
            plan_link = source["inputs"]["prompt_relay_plan"]
            plan = graph.get(plan_link[0])
            if plan is None or plan["class_type"] != "MiniMaxH3PromptRelayPlanT8Advanced":
                raise ValueError("分离式 HyperFlow Relay 缺少原生 Plan")
            base_plan_inputs = deepcopy(plan["inputs"])

            def stage_plan(stage):
                inputs = deepcopy(base_plan_inputs)
                selected = sampling["stage_relay"][stage]
                if selected["mode"] == "custom":
                    from .prompt_relay_advanced import build_prompt_relay_plan

                    for key in ("global_prompt", "local_prompts", "timing_mode", "time_ranges"):
                        inputs[key] = selected[key]
                    # The native planner owns event-count, frame-alignment,
                    # range, gap and overlap validation for both stages.
                    build_prompt_relay_plan(**inputs)
                return inputs

            if not resume_tail:
                plan["inputs"] = stage_plan("head")
                source["inputs"]["model"] = head_model
                head_model = [positive[0], 0]
            high_plan = add("MiniMaxH3PromptRelayPlanT8Advanced", stage_plan("tail"))
            high_relay_inputs = deepcopy(source["inputs"])
            high_relay_inputs.update(model=tail_model, prompt_relay_plan=[high_plan, 0])
            high_relay = add("MiniMaxH3PromptRelayConditioningT8Advanced", high_relay_inputs)
            tail_model, high_positive = [high_relay, 0], [high_relay, 1]
        else:
            if graph.get("5", {}).get("class_type") != "MiniMaxH3AudioConditioningT8":
                raise ValueError("分离式 HyperFlow 需要原生 T2VA 条件节点")
            high_condition = add("MiniMaxH3AudioConditioningT8", deepcopy(graph["5"]["inputs"]))
            high_positive = [high_condition, 0]
            if positive != ["5", 0]:
                if source is None or source["class_type"] != "MiniMaxH3SemanticBridgeApplyT8":
                    raise ValueError("分离式 HyperFlow 无法安全复制 HIGH 条件")
                high_bridge_inputs = deepcopy(source["inputs"])
                high_bridge_inputs["conditioning"] = high_positive
                high_positive = [add("MiniMaxH3SemanticBridgeApplyT8", high_bridge_inputs), 0]
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
            source_boundary = [head_audit, 0]
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
        graph["10"]["inputs"]["av_latent"] = [tail_audit, 1]
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
        graph["10"]["inputs"]["av_latent"] = ["9", 0]
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
            "quality_status": "experimental_not_human_reviewed"}
