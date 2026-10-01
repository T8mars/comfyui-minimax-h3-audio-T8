"""Three RF entry recipes as editable graphs; no sampling or user Core writes."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_manual_pass_workflow as manual  # noqa: E402

shared, native = manual.shared, manual.native
ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ("standalone", "detail_mixer", "two_pass_detail_mixer")


def _effects(graph, model, source, prefix):
    bias, stg = str(prefix), str(prefix + 1)
    graph[bias] = {"class_type": "MiniMaxH3ModelTimeBiasSamplerT8Advanced", "inputs": {
        "model": model, "av_latent": source, "steps": 4, "shift_video": 12., "shift_audio": 3.,
        "bias": -.05, "start_progress": .7, "end_progress": 1., "bias_domain": "video_sigma"}}
    graph[stg] = {"class_type": "MiniMaxH3SpatioTemporalGuidanceT8Advanced", "inputs": {
        "model": [bias, 0], "scale": .3, "double_blocks": "0", "start_progress": .2,
        "end_progress": .8, "shift_video": 12., "rescale": 0.}}
    return [stg, 0]


def split_graph(entry="standalone", variant="minimal", *, artifact_path="REPLACE_WITH_RF_BASE_PATH/manifest.json",
                artifact_sha256="0" * 64):
    if entry not in ENTRIES or variant not in native.VARIANTS:
        raise ValueError("Unknown RF entry or workflow variant")
    effects = variant != "minimal"
    saved = variant in ("save_effects", "resume_effects")
    build_variant = "save_effects" if variant == "resume_effects" else variant
    third = entry == "two_pass_detail_mixer"
    graph = native.split_graph(4, 3, build_variant) if third else manual.split_graph(build_variant)
    base_id, base_sample = ("26", "29") if third else ("10", "13")
    original = deepcopy(graph[base_id]["inputs"])
    if third:
        graph["88"] = deepcopy(graph[base_id])  # Original published partial HIGH3 table, no sampling.
    else:
        graph["88"] = {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": original["model"], "av_latent": original["av_latent"], "steps": 20,
            "shift_video": 12., "shift_audio": 3., "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}}
    schedule = ["88", 2]
    selected = original["model"]
    if entry != "standalone":
        graph["89"] = {"class_type": "MiniMaxH3AVTailDetailScheduleT8Advanced", "inputs": {
            "sigmas": schedule, "extra_tail_steps": 2, "spacing": "video_sigma_linear",
            "shift_video": 12., "shift_audio": 3., "profile": "custom_strict"}}
        schedule = ["89", 0]
        selected = _effects(graph, selected, original["av_latent"], 110)
    graph[base_id] = {"class_type": "MiniMaxH3RFBaseStageSetupEXPT8", "inputs": {
        "model": selected, "av_latent": original["av_latent"], "sigmas": schedule,
        "shift_video": 12., "shift_audio": 3.}}
    graph[base_sample]["class_type"] = "MiniMaxH3StageSamplerEXPT8"
    graph[base_sample]["inputs"]["stage_context"] = [base_id, 3]
    base_output = ["46" if third else "45", 0] if effects else [base_sample, 0]
    graph["90"] = {"class_type": "MiniMaxH3RFHandoffEXPT8", "inputs": {"completed_av": base_output}}
    if third:
        # Preserve both spatial passes and append a real independently editable RF third descent.
        graph["93"] = deepcopy(graph["22"])
        if graph["93"]["class_type"] == "MiniMaxH3StageUNETLoaderAfterEXPT8":
            graph["93"]["inputs"]["completed_stage"] = ["29", 2]
        graph["94"] = deepcopy(graph["71"])
        graph["94"]["inputs"]["model"] = ["93", 0]
        graph["95"] = deepcopy(graph["24"])
        if effects:
            graph["95"]["inputs"]["model"] = ["94", 0]
            graph["104"] = deepcopy(graph["47"])
            graph["95"]["inputs"]["prompt_relay_plan"] = ["104", 0]
        restart_id, final_sample = "96", "99"
        restart_model = ["95", 0] if effects else ["94", 0]
        condition = ["95", 1 if effects else 0]
        graph["97"] = deepcopy(graph["27"])
        graph["98"] = {"class_type": "BasicGuider", "inputs": {"model": [restart_id, 0], "conditioning": condition}}
        graph[final_sample] = {"class_type": "MiniMaxH3StageSamplerEXPT8", "inputs": {
            "noise": ["97", 0], "guider": ["98", 0], "sampler": [restart_id, 1], "sigmas": [restart_id, 2],
            "latent_image": [restart_id, 3], "stage_context": [restart_id, 4]}}
        condition_id = "95"
        if effects:
            graph["100"] = deepcopy(graph["42"])
            graph["101"] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
                "model": [restart_id, 0], "sigmas": [restart_id, 2], "av_latent": [restart_id, 3],
                "stage_context": [restart_id, 4], "eav_config": ["100", 0]}}
            graph["98"]["inputs"]["model"] = ["101", 0]
            graph["102"] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
                "av_latent": ["103", 0] if saved else [final_sample, 0], "runtime": ["101", 1]}}
        if saved:
            graph["103"] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
                "stage_result": [final_sample, 2], "prefix": "RF/RESTART"}}
            graph["51"]["inputs"]["prefix"] = "RF/BASE"
        graph["14"]["inputs"]["av_latent"] = ["102", 0] if effects else [final_sample, 0]
        graph["32"]["inputs"]["source"] = [restart_id, 5]
    else:
        restart_id, final_sample, condition_id = "26", "29", "24"
        restart_model = graph[restart_id]["inputs"]["model"]
        graph[final_sample]["class_type"] = "MiniMaxH3StageSamplerEXPT8"
        graph[final_sample]["inputs"].update(latent_image=[restart_id, 3], stage_context=[restart_id, 4])
        graph["32"]["inputs"]["source"] = [restart_id, 5]
        if effects:
            graph["44"]["inputs"].update(av_latent=[restart_id, 3], stage_context=[restart_id, 4])
        if saved:
            graph["50"]["inputs"]["prefix"] = "RF/BASE"
            graph["51"]["inputs"]["prefix"] = "RF/RESTART"
    graph[condition_id]["inputs"].update(width=["90", 3], height=["90", 4])
    if entry != "standalone":
        restart_model = _effects(graph, restart_model, ["90", 1], 120)
    graph[restart_id] = {"class_type": "MiniMaxH3RFRestartStageSetupEXPT8", "inputs": {
        "model": restart_model, "rf_handoff": ["90", 0], "shift_video": 12., "shift_audio": 3.,
        "restart_video_sigma": .15, "restart_steps": 3, "restart_seed": 1234}}
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_RF_{entry}_EXP"
    if variant == "resume_effects":
        loader_key = "93" if third else "22"
        loader = graph[loader_key]
        if loader["class_type"] != "MiniMaxH3StageUNETLoaderAfterEXPT8":
            raise ValueError("Expected a serialized restart-stage UNET loader")
        graph[loader_key] = {"class_type": "UNETLoader", "inputs": {
            name: value for name, value in loader["inputs"].items()
            if name != "completed_stage"}}
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "rf_base"}}
        graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
        graph["90"]["inputs"]["completed_av"] = ["60", 0]
        keep = set()
        def include(key):
            if key in keep:
                return
            keep.add(key)
            for value in graph[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    include(str(value[0]))
        for key in ("16", "32", "103" if third else "51", "61"):
            include(key)
        graph = {key: value for key, value in graph.items() if key in keep}
    return graph


def build_candidate(entry, variant, info):
    graph, workflow, _ = shared.build_candidate(split_graph(entry, variant), info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    for key, node in graph.items():
        item = next(value for value in workflow["nodes"] if value["id"] == ids[key])
        if int(key) >= 88:
            item["pos"] = [((int(key) - 88) % 4) * 430, 3000 + ((int(key) - 88) // 4) * 360]
        if node["class_type"].startswith("MiniMaxH3RF"):
            item["title"] = node["class_type"].replace("MiniMaxH3", "").replace("EXPT8", " — separate EXP")
    note = ("RF Restart 分离：BASE output＋原始模板 → 显式Handoff → 只执行RESTART。"
            "原模板、sigma精度与Core转float32之前的原始终点包含在Stage Sampler输出/保存资产中；Handoff提供真实宽高。"
            "各阶段独立MODEL/LoRA/条件/噪声/EAV/Relay Plan；tail/Bias/STG外置。"
            "NOISE控制原inpaint噪声，restart_seed控制设备上的联合AV重噪。"
            "两种Mixer原先隐藏的RF下降不再藏在SAMPLER里，two_pass图保留LOW＋learned＋HIGH再加RF第三采。"
            "关闭RF为0步旁路，不能删除原audio rebase。\n\n"
            "恢复图读取RF BASE的path/SHA及原模板，只跑RESTART；必须匹配冻结片段长度/提示词布局。"
            "本批为tiny CPU与Core图校验，不是浏览器编辑重载/完整GPU/人审认证。旧正式图不变。")
    for item in workflow["nodes"]:
        if item["type"] == "MarkdownNote":
            item["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"RF {entry} — {variant} EXP"
    _, selected = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = manual.load_info()
    import execution
    records = []
    for entry in ENTRIES:
        for variant in native.VARIANTS:
            graph, workflow, audit = build_candidate(entry, variant, info)
            validation = asyncio.run(execution.validate_prompt("rf-separate-candidate", deepcopy(graph), None))
            name = f"RF_{entry}_{variant}_EXP"
            shared.write_new(destination / f"{name}.api.json", graph)
            shared.write_new(destination / f"{name}.json", workflow)
            records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "Live Core CPU and serialization; no browser edit/reload, GPU, long-video body or human review."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
