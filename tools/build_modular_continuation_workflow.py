"""Accepted-parent windows with editable stages/effects and explicit frozen recovery."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_progressive_workflow as base  # noqa: E402

ROOT = base.ROOT
COMBINATIONS = (("plain", "both"),) + tuple((kind, scope)
    for kind in ("eav", "relay", "combined") for scope in ("low", "high", "both"))


def split_graph(frames=22, kind="plain", scope="both", variant="minimal"):
    if frames not in (22, 39) or (kind, scope) not in COMBINATIONS or variant not in base.VARIANTS:
        raise ValueError("Unknown continuation candidate")
    graph = base.split_graph("minimal", "t2va")
    templates = base.shared.split_graph_with_effects()
    graph["200"] = {"class_type": "MiniMaxH3ContinuationSourceEXPT8", "inputs": {
        "chain_id": "REPLACE_WITH_ACCEPTED_CHAIN", "segment_index": 1,
        "parent_candidate_id": "REPLACE_WITH_ACCEPTED_CANDIDATE", "parent_revision": 1,
        "previous_job_sha256": "0" * 64, "context_frames": frames,
        "width": 896, "height": 448, "low_width": 448, "low_height": 224}}
    graph["201"] = {"class_type": "MiniMaxH3ContinuationContextsEXPT8", "inputs": {
        "accepted_source": ["200", 0], "video_vae": ["7", 0]}}
    graph["90"]["inputs"].update(width=["200", 2], height=["200", 3], length=124)
    graph["26"] = {"class_type": "MiniMaxH3ContinuationPlanEXPT8", "inputs": {
        "contexts": ["201", 0], "full_sigmas": ["10", 2], "length": 124, "low_evaluations": 4, "low_scale": .5}}
    for key in ("10", "92"):
        graph[key]["inputs"]["steps"] = 8
    for phase, encoder in (("low", "9"), ("high", "24")):
        graph[encoder] = {"class_type": "MiniMaxH3ContinuationConditioningEXPT8", "inputs": {
            "contexts": ["201", 0], "phase": phase, "clip": ["6", 0], "video_vae": ["7", 0], "audio_vae": ["8", 0],
            "context_audio": "video_and_audio", "prompt": "Continue the woman's walking motion in the same quiet concert hall.",
            "length": 124, "task_type": "auto", "audio_mode": "native", "audio_denoise_strength": .35,
            "add_source_as_reference": True, "prompt_primary_audio_ordinal": 0, "strict_prompt_tags": True,
            "ref_image_size": "match", "reference_video_policy": "official_2_to_15s"}}
    graph["13"]["class_type"] = "MiniMaxH3ContinuationLowStageEXPT8"
    graph["13"]["inputs"].pop("cfg")
    graph["13"]["inputs"].pop("low_source")
    graph["13"]["inputs"].update(prepared_phase=["9", 0], positive=["9", 1], negative=["9", 2])
    graph["25"]["class_type"] = "MiniMaxH3ContinuationHighHandoffEXPT8"
    graph["25"]["inputs"].pop("high_source")
    graph["25"]["inputs"]["prepared_phase"] = ["24", 0]
    graph["29"]["class_type"] = "MiniMaxH3ContinuationHighStageEXPT8"
    graph["29"]["inputs"].pop("cfg")
    graph["29"]["inputs"].update(prepared_phase=["24", 0], positive=["24", 1], negative=["24", 2])
    outputs = ["16", "30", "203"]
    for phase, encoder, sampler, base_model, group, plan in (
        ("low", "9", "13", "10", 100, ["26", 0]), ("high", "24", "29", "92", 110, ["25", 1])):
        if kind == "plain" or scope not in (phase, "both"):
            continue
        if kind in ("relay", "combined"):
            plan_key, apply_key, projection_key = str(group), str(group+1), str(group+6)
            graph[plan_key] = deepcopy(templates["40"])
            graph[plan_key]["inputs"].update(length=345,
                global_prompt="Same scene and character throughout the selected accepted timeline.",
                local_prompts="She walks through the hall.\nShe slows near the doorway.\nShe turns to the right.")
            graph[projection_key] = {"class_type": "MiniMaxH3ContinuationRelayProjectEXPT8", "inputs": {
                "contexts": ["201", 0], "global_plan": [plan_key, 0], "length": 124, "accepted_end_frame": -1}}
            graph[encoder]["inputs"]["prompt"] = [projection_key, 1]
            graph[apply_key] = {"class_type": "MiniMaxH3ContinuationRelayApplyEXPT8", "inputs": {
                "model": [base_model, 0], "prepared_phase": [encoder, 0], "plan": plan,
                "clip": ["6", 0], "projected_relay": [projection_key, 0], "query_chunk_rows": 256, "mode": "apply_exp"}}
            graph[sampler]["inputs"].update(model=[apply_key, 0], positive=[apply_key, 1], negative=[apply_key, 2])
        if kind in ("eav", "combined"):
            config_key, apply_key = str(group+2), str(group+3)
            graph[config_key] = deepcopy(templates["41"])
            graph[apply_key] = {"class_type": "MiniMaxH3Continuation" + phase.title() + "EAVApplyEXPT8", "inputs": {
                "model": graph[sampler]["inputs"]["model"], "prepared_phase": [encoder, 0], "eav_config": [config_key, 0]}}
            graph[apply_key]["inputs"].update({"plan": plan} if phase == "low" else {"high_restart": ["25", 0]})
            graph[sampler]["inputs"]["model"] = [apply_key, 0]
        audit_key, preview_key = str(group+4), str(group+5)
        graph[audit_key] = {"class_type": "MiniMaxH3Progressive" + phase.title() + "EffectsAuditEXPT8", "inputs": {
            "low_boundary" if phase == "low" else "high_result": [sampler, 0 if phase == "low" else 2]}}
        graph[preview_key] = {"class_type": "PreviewAny", "inputs": {"source": [audit_key, 1]}}
        outputs.append(preview_key)
        if phase == "low":
            for key in ("20", "25"):
                graph[key]["inputs"]["low_boundary"] = [audit_key, 0]
    high_result = ["114", 0] if "114" in graph else ["29", 2]
    if variant == "save":
        graph["50"] = {"class_type": "MiniMaxH3ProgressiveLowSaveEXPT8", "inputs": {
            "low_boundary": ["104" if "104" in graph else "13", 0], "prefix": "Continuation/LOW"}}
        for key in ("20", "25"):
            graph[key]["inputs"]["low_boundary"] = ["50", 0]
        outputs.append("50")
    if variant in ("save", "resume_high"):
        graph["51"] = {"class_type": "MiniMaxH3ProgressiveHighSaveEXPT8", "inputs": {
            "high_result": high_result, "prefix": "Continuation/HIGH"}}
        outputs.append("51")
    if variant == "resume_high":
        graph["60"] = {"class_type": "MiniMaxH3ProgressiveLowLoadEXPT8", "inputs": {
            "artifact_path": "REPLACE_WITH_SAVED_LOW_PATH/progressive-low.safetensors", "artifact_sha256": "0" * 64}}
        for key in ("20", "25"):
            graph[key]["inputs"]["low_boundary"] = ["60", 0]
        outputs = [key for key in outputs if key != "105"]
    elif variant == "load_high":
        graph["61"] = {"class_type": "MiniMaxH3ProgressiveHighLoadEXPT8", "inputs": {
            "artifact_path": "REPLACE_WITH_SAVED_HIGH_PATH/progressive-high.safetensors", "artifact_sha256": "0" * 64}}
        high_result = ["61", 1]
        outputs = ["16", "30", "203"]
    graph["203"] = {"class_type": "MiniMaxH3ContinuationDeliveryEXPT8", "inputs": {
        "high_result": high_result, "accepted_source": ["200", 0]}}
    graph["14"]["inputs"]["av_latent"] = ["203", 0]
    graph["30"]["inputs"]["source"] = ["203", 4]
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Continuation_{frames}_{kind}_{scope}_{variant}_EXP"
    return base.prune(graph, outputs)


def build_candidate(frames, kind, scope, variant, info, *, graph_override=None):
    source = split_graph(frames, kind, scope, variant) if graph_override is None else graph_override
    graph, selected = base.shared.selected_frontend_schema(source, info)
    workflow = base.shared.convert(graph, selected, f"Continuation {frames} / {kind} / {scope} / {variant} EXP")
    mapping = {key: i + 1 for i, key in enumerate(graph)}
    positions = {
        "200": (-1300, -700), "201": (-850, -700), "90": (-400, -700), "26": (0, -700),
        "1": (-400, 200), "10": (0, 200), "9": (420, 200), "13": (1800, 200), "11": (1380, 700),
        "22": (-400, 1700), "92": (0, 1700), "24": (420, 1700), "20": (880, 1700),
        "23": (1280, 1700), "25": (1700, 1700), "27": (1700, 2150), "29": (2550, 1700),
        "50": (2250, 200), "51": (2950, 1700), "60": (0, -700), "61": (0, -700),
        "203": (3400, 1700), "14": (3850, 1700), "15": (4250, 1700), "16": (4650, 1700),
        "6": (-1300, 300), "7": (-1300, 700), "8": (-1300, 1050), "30": (3400, 2400)}
    for group, y in ((100, -1800), (110, 3300)):
        for offset, x in ((0, 0), (6, 450), (1, 900), (2, 1250), (3, 1650), (4, 2200), (5, 2600)):
            positions[str(group+offset)] = (x, y)
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, node_id in mapping.items():
        by_id[node_id]["pos"] = list(positions.get(key, (-1800, 0)))
    note = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note, "type": "MarkdownNote", "title": "续段分离 / Read first",
        "pos": [-2600, -1700], "size": [1250, 1000], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "已接受直接前段 → LOW/HIGH 独立续段 / EXP。不是自动长视频Loop；不自动Candidate/Accept/Compose。\n\n"
            "必须先有原生已接受链。Source里的chain_id、直接前段candidate、revision和previous_job_sha256都需填写实际值，"
            "目标尺寸须与该前段相同；LOW尺寸须与Plan scale匹配。占位符不可运行，不能把任意视频或任意job字符串当验证凭据。"
            "previous_job_sha256仅描述已选择的前段，不代表今天修改的模型/提示词仍是原job。\n\n"
            "LOW参考＝真实已接受MP4末39帧→原resize→当前VAE；HIGH参考＝完成AV上下文。两者共享完成音频，"
            "只HIGH锁已完成视频前缀；context选择22或39，时间坐标不变。两个条件节点独立，不调用配对整体编译器。\n\n"
            "Relay从全局计划按实际accepted结束位置投影，再接本阶段条件编译和Apply；两采Plan可独立。"
            "EAV配置外置，可仅LOW/仅HIGH/两采，默认report_only，apply_exp才施加。原生未知用户补丁保持执行，"
            "效果是否覆盖与持久资格按实际报告，不偷偷更换后端。\n\n"
            "模型/LoRA/条件/噪声两路独立，原learned3D节点外置；默认native8=4+4，非FastH3或HyperFlow。"
            "HIGH video seed默认LOW+1，HIGH sampler seed默认同LOW；改LOW seed请显式同步相应种子。\n\n"
            "Save记录实际path/SHA；resume_high已删除LOW链，load_high删除全部条件编译/模型/放大/采样，仅验证前段并解码已完成HIGH。"
            "本图默认native生成音频，CreateVideo接AV Decode音频；如显式配置final_audio，应改接Delivery的mux_audio，不能混用。\n\n"
            "当前为CPU tiny/VAE与learned替身、Core序列化资格；完整trained/GPU/浏览器/音画接缝人审仍待。"
            "旧一体节点、旧图、已接受成片和导演台不替换。"]})
    workflow["last_node_id"] = note
    return graph, workflow, base.shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = base.load_live_info()
    import execution
    rows = []
    for frames in (22, 39):
        for kind, scope in COMBINATIONS:
            for variant in base.VARIANTS:
                graph, workflow, audit = build_candidate(frames, kind, scope, variant, info)
                validation = asyncio.run(execution.validate_prompt("continuation-modular", deepcopy(graph), None))
                expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
                valid = bool(validation[0] and not validation[3] and set(validation[2]) == expected)
                name = f"Continuation_{frames}_{kind}_{scope}_{variant}_EXP"
                base.shared.write_new(target / (name + ".api.json"), graph)
                base.shared.write_new(target / (name + ".json"), workflow)
                rows.append({"name": name, "serialization": audit, "core_validation": validation, "all_outputs_valid": valid})
    base.shared.write_new(target / "audit.json", {"candidates": rows,
        "qualification": "CPU live Core/API/frontend serialization only; source placeholders need real accepted chain. "
                         "Not browser/trained/GPU/media/auto-loop qualification."})
    base.shared.write_new(target / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(target)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
