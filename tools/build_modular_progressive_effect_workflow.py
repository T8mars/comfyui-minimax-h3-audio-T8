"""External EAV/Relay on independent Progressive phases; no old graph edits."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_progressive_workflow as base  # noqa: E402

ROOT = base.ROOT
KINDS = ("eav", "relay", "combined")
SCOPES = ("low", "high", "both")


def split_graph(task="t2va", kind="combined", scope="both", variant="minimal"):
    if kind not in KINDS or scope not in SCOPES or variant not in base.VARIANTS:
        raise ValueError("Unknown Progressive effects candidate")
    graph = base.split_graph("minimal", task)
    templates = base.shared.split_graph_with_effects()
    outputs = ["16", "30"]
    for phase, encoder, stage, sampler, model, group, plan in (
        ("low", "9", "12", "13", "10", 100, ["26", 0]),
        ("high", "24", "28", "29", "92", 110, ["25", 1])):
        if scope not in (phase, "both"):
            continue
        if kind in ("relay", "combined"):
            plan_id = str(group)
            graph[plan_id] = deepcopy(templates["40"])
            graph[plan_id]["inputs"]["global_prompt"] = (
                f"{phase.upper()} independent Relay plan. A continuous cinematic shot in a quiet concert hall.")
            source = graph[encoder]
            source["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
            source["inputs"].pop("prompt")
            source["inputs"].pop("length")
            source["inputs"].update(model=[model, 0], prompt_relay_plan=[plan_id, 0],
                                    execution_mode="apply_exp", query_chunk_rows=256)
            if phase == "high":
                graph["25"]["inputs"]["high_source"] = [encoder, 2]
            graph[stage] = {"class_type": "MiniMaxH3ProgressiveRelayStageApplyEXPT8", "inputs": {
                "model": [encoder, 0], "positive": [encoder, 1], "negative": [encoder, 1],
                "plan": plan, "phase": phase, "mode": "apply_exp", "guide_resize": "legacy_bilinear"}}
            graph[sampler]["inputs"].update(model=[stage, 0], positive=[stage, 1], negative=[stage, 2])
        if kind in ("eav", "combined"):
            config_id, apply_id = str(group + 2), str(group + 3)
            graph[config_id] = deepcopy(templates["41"])
            graph[apply_id] = {"class_type": "MiniMaxH3Progressive" + phase.title() + "EAVApplyEXPT8", "inputs": {
                "model": graph[sampler]["inputs"]["model"], "eav_config": [config_id, 0]}}
            if phase == "low":
                graph[apply_id]["inputs"].update(plan=plan, low_source=["26", 1])
            else:
                graph[apply_id]["inputs"]["high_restart"] = ["25", 0]
            graph[sampler]["inputs"]["model"] = [apply_id, 0]
        audit_id, preview_id = str(group + 4), str(group + 5)
        graph[audit_id] = {"class_type": "MiniMaxH3Progressive" + phase.title() + "EffectsAuditEXPT8",
            "inputs": {"low_boundary" if phase == "low" else "high_result": [sampler, 0 if phase == "low" else 2]}}
        if phase == "low":
            for consumer in ("20", "25"):
                graph[consumer]["inputs"]["low_boundary"] = [audit_id, 0]
        graph[preview_id] = {"class_type": "PreviewAny", "inputs": {"source": [audit_id, 1]}}
        outputs.append(preview_id)
    if variant != "minimal":
        stored = base.split_graph(variant, task)
        for key in ("50", "51", "60", "61"):
            if key in stored:
                graph[key] = deepcopy(stored[key])
        if variant == "save":
            graph["50"]["inputs"]["low_boundary"] = ["104" if "104" in graph else "13", 0]
            for key in ("20", "25"):
                graph[key]["inputs"]["low_boundary"] = ["50", 0]
            outputs.append("50")
        if variant in ("save", "resume_high"):
            if "114" in graph:
                graph["51"]["inputs"]["high_result"] = ["114", 0]
            graph["14"]["inputs"]["av_latent"] = ["51", 0]
            outputs.append("51")
        if variant == "resume_high":
            for key in ("20", "25"):
                graph[key]["inputs"]["low_boundary"] = ["60", 0]
            outputs = [key for key in outputs if key != "105"]
        elif variant == "load_high":
            graph["14"]["inputs"]["av_latent"] = ["61", 0]
            graph["30"]["inputs"]["source"] = ["61", 2]
            outputs = ["16", "30"]
            if "114" in graph:
                graph["114"]["inputs"]["high_result"] = ["61", 1]
                outputs.append("115")
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Progressive_{task}_{kind}_{scope}_{variant}_EXP"
    return base.prune(graph, outputs)


def build_candidate(task, kind, scope, info, variant="minimal", *, graph_override=None):
    source = graph_override if graph_override is not None else split_graph(task, kind, scope, variant)
    graph, selected = base.shared.selected_frontend_schema(source, info)
    title = f"Progressive {task} — external {kind} — {scope} — {variant} EXP"
    workflow = base.shared.convert(graph, selected, title)
    _, plain, _ = base.build_candidate(variant, task, info)
    old_ids = {key: index + 1 for index, key in enumerate(base.split_graph(variant, task))}
    new_ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in plain["nodes"]}
    current = {node["id"]: node for node in workflow["nodes"]}
    for key, old_id in old_ids.items():
        if key in new_ids:
            node = current[new_ids[key]]
            node.update(pos=by_id[old_id]["pos"], title=by_id[old_id]["title"])
            if key in ("12", "28") and "Relay" in graph[key]["class_type"]:
                node["title"] = ("LOW" if key == "12" else "HIGH") + " Relay rebind + stage conditioning"
    for group, label, y in ((100, "LOW", -1600), (110, "HIGH", 3400)):
        for offset, caption, x in ((0, "independent Relay Plan", 0), (2, "independent EAV Config", 1200),
                                  (3, "native phase EAV Apply", 1580), (4, "immutable execution Audit", 2000),
                                  (5, "actual effects report", 2400)):
            key = str(group + offset)
            if key in new_ids:
                current[new_ids[key]].update(pos=[x, y], title=label + " — " + caption)
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "External effects / EXP boundaries",
        "pos": [-1500, -1000], "size": [1000, 680], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "Progressive 逐阶段外置效果 / EXP。LOW/HIGH模型、LoRA、提示词、Relay Plan与EAV Config独立。"
            "可只给一采、只给二采或两采分别配置，不调用隐藏双采Loop。\n\n"
            "Relay Conditioning按目标尺寸编码，专用Relay Stage Apply负责LOW实际布局/guide缩放；"
            "不要再串一个Stage Conditioning重复缩放。EAV默认report_only，需明确改apply_exp才应用。"
            "EAV的时间窗始终是1-video_sigma，不在二采重新从0计时。\n\n"
            "外置Audit读取对应完成结果的实际调用，不依赖可变运行时的旧计数。当前CFG1；"
            "未知用户补丁保留执行但可能绕过效果，检查unverified报告。\n\n"
            "可认证的原生效果支持LOW保存/只HIGH恢复/完成HIGH读取；未知组合仍可执行归档，但不认证跨进程复用。"
            "Save后保留实际artifact_path与SHA，Load占位符必须替换；恢复图没有LOW模型/采样/效果链。"
            "图中的learned3D是真实外置节点，但本批CPU验证使用tiny模型/明确放大替身；"
            "完整预训练GPU、实际learned权重、浏览器重载、成片人审仍待。旧工作流不替换。"]})
    workflow["last_node_id"] = note_id
    return graph, workflow, base.shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=base.VARIANTS, default="minimal")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = base.load_live_info()
    import execution
    rows = []
    for task in ("t2va", "i2va"):
        for kind in KINDS:
            for scope in SCOPES:
                graph, workflow, audit = build_candidate(task, kind, scope, info, args.variant)
                validation = asyncio.run(execution.validate_prompt("progressive-external-effects", deepcopy(graph), None))
                name = f"Progressive_{task}_{kind}_{scope}_{args.variant}_EXP"
                base.shared.write_new(destination / (name + ".api.json"), graph)
                base.shared.write_new(destination / (name + ".json"), workflow)
                expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
                valid = bool(validation[0] and not validation[3] and set(validation[2]) == expected)
                rows.append({"name": name, "serialization": audit, "core_validation": validation, "all_outputs_valid": valid})
    base.shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "Live CPU Core schema/API and serialization, not GPU/media/browser qualification."})
    base.shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
