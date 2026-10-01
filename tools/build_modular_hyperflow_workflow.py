"""True editable HyperFlow continuous stages, not upscale or a hidden split."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_progressive_workflow as common  # noqa: E402

shared = common.shared
ROOT = common.ROOT
SPLITS = (1, 4, 7)
WEIGHT = "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors"


def split_graph(split=4):
    if type(split) is not int or split not in SPLITS:
        raise ValueError("Select an authored continuous1+7/4+4/7+1 candidate")
    original = common.split_graph()
    graph = {key: deepcopy(original[key]) for key in ("1", "6", "7", "8", "9", "11", "14", "15", "16", "24", "90")}
    for key in ("9", "24", "90"):
        graph[key]["inputs"].update(width=512, height=512, length=73)
    for key in ("101", "102"):
        graph[key] = {"class_type": "MiniMaxH3HyperFlowLoaderT8Advanced",
            "inputs": {"model": ["1", 0], "hyperflow_file": WEIGHT}}
    graph["13"] = {"class_type": "MiniMaxH3HyperFlowHeadStageEXPT8", "inputs": {
        "model": ["101", 0], "av_latent": ["90", 0], "noise": ["11", 0],
        "positive": ["9", 0], "negative": ["9", 0], "split_interval": split, "cfg": 1., "reserve_vram_mib": 1024}}
    graph["29"] = {"class_type": "MiniMaxH3HyperFlowTailStageEXPT8", "inputs": {
        "continuous_boundary": ["13", 0], "model": ["102", 0], "positive": ["24", 0], "negative": ["24", 0],
        "seed": graph["11"]["inputs"]["noise_seed"], "cfg": 1., "reserve_vram_mib": 1024}}
    graph["14"]["inputs"]["av_latent"] = ["29", 0]
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_HyperFlow_Continuous_{split}plus{8-split}_EXP"
    graph["30"] = {"class_type": "PreviewAny", "inputs": {"source": ["29", 1]}}
    graph["31"] = {"class_type": "PreviewAny", "inputs": {"source": ["13", 1]}}
    return graph


def load_live_info():
    return common.load_live_info()


def build_candidate(split, info):
    graph, selected = shared.selected_frontend_schema(split_graph(split), info)
    workflow = shared.convert(graph, selected, f"HyperFlow continuous {split}+{8-split} — independent stages EXP")
    mapping = {index + 1: key for index, key in enumerate(graph)}
    layout = {
        "1": (0, 0, "One common full H3 base — branch BEFORE independent content LoRAs"),
        "101": (400, 0, "HEAD dedicated HyperFlow Loader — insert own content LoRAs upstream"),
        "102": (400, 1320, "TAIL dedicated HyperFlow Loader — independent content LoRAs upstream"),
        "6": (-450, 370, "Shared CLIP"), "7": (-450, 710, "Video VAE"), "8": (-450, 1000, "Audio VAE"),
        "90": (400, 530, "Common geometry only — TAIL prompt is not an ancestor of HEAD"),
        "9": (800, 0, "HEAD independent prompt / conditions"),
        "24": (800, 1320, "TAIL independent prompt / conditions"),
        "11": (1210, 530, "Initial joint AV noise — HEAD only"),
        "13": (1220, 0, f"HEAD only — absolute0:{split}; exact x_sigma output"),
        "29": (1610, 1320, f"TAIL only — absolute{split}:8; NO new noise / NO upscale"),
        "14": (2040, 1320, "Decode completed AV"), "15": (2460, 1320, "Create video with generated audio"),
        "16": (2870, 1320, "Save video"), "30": (1620, 1960, "TAIL actual execution report"),
        "31": (1640, 0, "HEAD completed boundary receipt")}
    for node in workflow["nodes"]:
        x, y, title = layout[mapping[node["id"]]]
        node.update(pos=[x, y], title=title)
    note = ("HyperFlow continuous分离候选：不是learned放大，也不是8+4/partial4+4/P7。\n\n"
        "一个完整H3底模分两支，各自插入内容LoRA后使用专用原HyperFlow Loader。不能用普通LoRA加载器加载HyperFlow。"
        "HEAD/TAIL提示词与条件各自编码；共同几何不依赖TAIL。CFG1候选复用正条件作负输入，改CFG须提供真实负条件。\n\n"
        "HEAD只走0:split，输出原始model-space x_sigma强类型；TAIL直接继续split:8，不加入新噪声、不audio rebase、不放大。"
        "TAIL的seed仅供模型执行，不是新RandomNoise；默认与HEAD一致。改TAIL模型/LoRA/提示只依赖TAIL链。\n\n"
        "当前只有tiny/synthetic adapter CPU数值与Core接口/序列化证据；原权重必须已安装，不自动下载。"
        "持久Save/Load、阶段EAV/Relay、浏览器保存重载、完整预训练GPU/音画/人审仍待，不以本候选代替全路线完成。"
        "未知用户wrapper保留执行，但不可移植身份不认证跨进程恢复。现有mask连续边界限制保持。旧一体split/旧图不替换。")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / Continuous only",
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
    info = load_live_info()
    import execution
    rows = []
    for split in SPLITS:
        graph, workflow, audit = build_candidate(split, info)
        validation = asyncio.run(execution.validate_prompt("t8-hyperflow-continuous-modular", deepcopy(graph), None))
        name = f"HyperFlow_Continuous_{split}plus{8-split}_EXP"
        expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
        rows.append({"name": name, "serialization": audit, "core_validation": validation,
            "all_outputs_valid": bool(validation[0] and not validation[3] and set(validation[2]) == expected)})
        shared.write_new(destination / (name + ".api.json"), graph)
        shared.write_new(destination / (name + ".json"), workflow)
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "CPU live schema/serialization only, not browser/GPU/media/human qualification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
