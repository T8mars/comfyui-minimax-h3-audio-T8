"""Continuous HyperFlow saved-HEAD, TAIL-only replay and completed-AV graphs."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_hyperflow_workflow as continuous  # noqa: E402

shared, ROOT = continuous.shared, continuous.ROOT
VARIANTS = ("save_head", "save_both", "resume_tail", "load_completed")


def split_graph(variant="save_both", split=4, *, artifact_path=None, artifact_sha256="0" * 64):
    if variant not in VARIANTS:
        raise ValueError("Unknown continuous HyperFlow storage variant")
    graph = continuous.split_graph(split)
    graph["50"] = {"class_type": "MiniMaxH3HyperFlowHeadSaveEXPT8", "inputs": {
        "continuous_boundary": ["13", 0], "prefix": f"HEAD_{split}plus{8-split}"}}
    graph["29"]["inputs"]["continuous_boundary"] = ["50", 0]
    graph["51"] = {"class_type": "MiniMaxH3HyperFlowTailSaveEXPT8", "inputs": {
        "completed_result": ["29", 2], "prefix": f"TAIL_{split}plus{8-split}"}}
    graph["14"]["inputs"]["av_latent"] = ["51", 0]
    graph["16"]["inputs"]["filename_prefix"] += "_" + variant
    if variant == "save_head":
        return continuous.common.prune(graph, ["50"])
    if variant == "resume_tail":
        graph["50"] = {"class_type": "MiniMaxH3HyperFlowHeadLoadEXPT8", "inputs": {
            "artifact_path": artifact_path or "SELECT_SAVED_HEAD/hyperflow-head.safetensors",
            "artifact_sha256": artifact_sha256}}
        graph["31"]["inputs"]["source"] = ["50", 1]
    elif variant == "load_completed":
        graph["51"] = {"class_type": "MiniMaxH3HyperFlowTailLoadEXPT8", "inputs": {
            "artifact_path": artifact_path or "SELECT_SAVED_TAIL/hyperflow-tail.safetensors",
            "artifact_sha256": artifact_sha256}}
        graph["30"]["inputs"]["source"] = ["51", 2]
        return continuous.common.prune(graph, ["16", "30"])
    return continuous.common.prune(graph, ["16", "30", "31"])


def build_candidate(variant, split, info):
    graph, selected = shared.selected_frontend_schema(split_graph(variant, split), info)
    workflow = shared.convert(graph, selected, f"HyperFlow {split}+{8-split} — {variant} EXP")
    # Reuse the authored minimal graph layout by stable API ids, not graph
    # enumeration after pruning. Load-only graphs contain no dormant samplers.
    minimal_graph, minimal_workflow, _ = continuous.build_candidate(split, info)
    layouts = {key: node for key, node in zip(minimal_graph, minimal_workflow["nodes"])}
    labels = {"50": "Save exact HEAD raw state" if variant != "resume_tail" else "Load frozen HEAD — NO HEAD execution",
              "51": "Save completed TAIL AV" if variant != "load_completed" else "Load completed AV — NO sampling"}
    for key, node in zip(graph, workflow["nodes"]):
        if key in layouts:
            node.update(pos=layouts[key]["pos"], title=layouts[key]["title"])
        else:
            node.update(pos=[2030, 0] if key == "50" else [2050, 790], title=labels[key])
    note = ("连续HyperFlow独立存取候选：HEAD保存原始x_sigma与独立scaffold；不是clean x0，不能接普通learned放大。\n\n"
        "save_head只执行并保存一采；save_both保存两采；resume_tail图已删除一采/初始噪声/一采输出分支；"
        "load_completed图只读已完成AV并解码，没有底模、CLIP或扩散采样。\n\n"
        "在Load填入Save输出的相对路径和精确SHA，两者缺一不可。占位值仅作编辑模板，不能直接队列执行。"
        "存储根output/MiniMaxH3/hyperflow_stage_artifacts。缺失/坏文件/未知持久身份报错，不自动重跑。"
        "一采冻结后允许二采内容LoRA/提示独立修改，但原底模和HyperFlow adapter内容必须匹配。\n\n"
        "旧HEAD节点说明中的process-local是初版历史说明；当前已新增专用认证存取，未改旧schema。"
        "本图不是8+4/partial4+4/P7；外置EAV/Relay、完整权重GPU/浏览器/音画人审仍待，不自动下载或替换旧图。")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / Explicit frozen stage",
        "pos": [0, -660], "size": [1350, 600], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = note_id
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    destination = options.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = continuous.load_live_info()
    import execution
    rows = []
    for split in continuous.SPLITS:
        for variant in VARIANTS:
            graph, workflow, audit = build_candidate(variant, split, info)
            validation = asyncio.run(execution.validate_prompt("hyperflow-storage", deepcopy(graph), None))
            expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
            name = f"HyperFlow_Continuous_{split}plus{8-split}_{variant}_EXP"
            rows.append({"name": name, "serialization": audit, "core_validation": validation,
                "all_outputs_valid": bool(validation[0] and not validation[3] and set(validation[2]) == expected)})
            shared.write_new(destination / (name + ".api.json"), graph)
            shared.write_new(destination / (name + ".json"), workflow)
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "Live CPU schema/serialization; explicit Load placeholders need actual saved path/SHA. "
                         "Not browser/pretrained GPU/media/human qualification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
