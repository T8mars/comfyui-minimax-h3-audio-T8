"""Editable native Progressive candidates; all sampling stages are explicit."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_fast_h3_v2_workflow as shared  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("minimal", "save", "resume_high", "load_high")
REFERENCE = "t8_h4c2_bund_korean_mv_ref_20260916.png"


def prune(graph, outputs):
    keep = set()
    def include(key):
        if key in keep:
            return
        keep.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                include(str(value[0]))
    for key in outputs:
        include(key)
    return {key: item for key, item in graph.items() if key in keep}


def split_graph(variant="minimal", task="t2va", *, reference=REFERENCE,
                artifact_path=None, artifact_sha256="0" * 64):
    if variant not in VARIANTS or task not in ("t2va", "i2va"):
        raise ValueError("Unknown Progressive candidate variant/task")
    original = shared.split_graph()
    graph = {key: deepcopy(original[key]) for key in ("1", "6", "7", "8", "9", "11", "14", "15", "16", "22", "23", "24", "27")}
    width, height, length, seed = 896, 512, 73, 26092301
    for key in ("1", "22"):
        graph[key]["inputs"]["unet_name"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    # This neutral source owns only geometry. Neither HIGH prompt nor HIGH
    # MODEL/LoRA may become an ancestor of LOW through an innocent LATENT edge.
    graph["90"] = {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {
        "width": width, "height": height, "length": length}}
    for key, model in (("10", "1"), ("92", "22")):
        graph[key] = {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": [model, 0], "av_latent": ["90", 0], "steps": 20,
            "shift_video": 12., "shift_audio": 3., "sampler_name": "euler", "scheduler": "native_flow"}}
    graph["26"] = {"class_type": "MiniMaxH3ProgressiveStagePlanEXPT8", "inputs": {
        "high_source": ["90", 0], "full_sigmas": ["10", 2], "low_evaluations": 10,
        "low_scale": .5, "task": task, "input_mode": "empty"}}
    for key in ("9", "24"):
        graph[key]["inputs"].update(width=width, height=height, length=length,
            task_type="I2VA" if task == "i2va" else "T2VA",
            prompt="A continuous cinematic shot of a woman walking through a quiet concert hall, natural motion and ambience.")
        if task == "i2va":
            graph[key]["inputs"]["first_frame"] = ["91", 0]
    if task == "i2va":
        graph["91"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
    graph["11"]["inputs"]["noise_seed"] = seed
    graph["27"]["inputs"]["noise_seed"] = (seed + 1) % 2**64
    graph["12"] = {"class_type": "MiniMaxH3ProgressiveStageConditioningEXPT8", "inputs": {
        "positive": ["9", 0], "negative": ["9", 0], "plan": ["26", 0], "phase": "low", "guide_resize": "legacy_bilinear"}}
    graph["13"] = {"class_type": "MiniMaxH3ProgressiveLowStageEXPT8", "inputs": {
        "model": ["10", 0], "sampler": ["10", 1], "noise": ["11", 0], "plan": ["26", 0],
        "low_source": ["26", 1], "positive": ["12", 0], "negative": ["12", 1], "cfg": 1., "reserve_vram_mib": 1024}}
    graph["20"] = {"class_type": "MiniMaxH3ProgressiveLiftInputEXPT8", "inputs": {
        "low_boundary": ["13", 0], "model": ["92", 0], "sampler": ["92", 1]}}
    graph["23"]["inputs"].update(av_latent=["20", 0], size_mode="target_dimensions",
        target_width=["20", 1], target_height=["20", 2])
    graph["24"]["inputs"].update(width=["23", 1], height=["23", 2])
    graph["25"] = {"class_type": "MiniMaxH3ProgressiveHighHandoffEXPT8", "inputs": {
        "low_boundary": ["13", 0], "model": ["92", 0], "sampler": ["92", 1],
        "lifted_av": ["23", 0], "high_source": ["24", 1], "video_noise": ["27", 0]}}
    graph["28"] = {"class_type": "MiniMaxH3ProgressiveStageConditioningEXPT8", "inputs": {
        "positive": ["24", 0], "negative": ["24", 0], "plan": ["25", 1], "phase": "high", "guide_resize": "legacy_bilinear"}}
    graph["29"] = {"class_type": "MiniMaxH3ProgressiveHighStageEXPT8", "inputs": {
        "high_restart": ["25", 0], "model": ["92", 0], "sampler": ["92", 1],
        "positive": ["28", 0], "negative": ["28", 1], "seed": seed, "cfg": 1., "reserve_vram_mib": 1024}}
    graph["14"]["inputs"]["av_latent"] = ["29", 0]
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_Progressive_{task}_{variant}_EXP"
    graph["30"] = {"class_type": "PreviewAny", "inputs": {"source": ["29", 1]}}
    outputs = ["16", "30"]
    if variant == "save":
        graph["50"] = {"class_type": "MiniMaxH3ProgressiveLowSaveEXPT8", "inputs": {
            "low_boundary": ["13", 0], "prefix": f"Progressive/{task}/LOW"}}
        for key in ("20", "25"):
            graph[key]["inputs"]["low_boundary"] = ["50", 0]
        outputs.append("50")
    if variant in ("save", "resume_high"):
        graph["51"] = {"class_type": "MiniMaxH3ProgressiveHighSaveEXPT8", "inputs": {
            "high_result": ["29", 2], "prefix": f"Progressive/{task}/HIGH"}}
        graph["14"]["inputs"]["av_latent"] = ["51", 0]
        outputs.append("51")
    if variant == "resume_high":
        graph["60"] = {"class_type": "MiniMaxH3ProgressiveLowLoadEXPT8", "inputs": {
            "artifact_path": artifact_path or "REPLACE_WITH_SAVED_LOW_PATH/progressive-low.safetensors",
            "artifact_sha256": artifact_sha256}}
        for key in ("20", "25"):
            graph[key]["inputs"]["low_boundary"] = ["60", 0]
    elif variant == "load_high":
        graph["61"] = {"class_type": "MiniMaxH3ProgressiveHighLoadEXPT8", "inputs": {
            "artifact_path": artifact_path or "REPLACE_WITH_SAVED_HIGH_PATH/progressive-high.safetensors",
            "artifact_sha256": artifact_sha256}}
        graph["14"]["inputs"]["av_latent"] = ["61", 0]
        graph["30"]["inputs"]["source"] = ["61", 2]
    return prune(graph, outputs)


def load_live_info():
    info = shared.load_live_info()
    import nodes
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    nodes.NODE_CLASS_MAPPINGS["EmptyMiniMaxH3LatentAV"] = EmptyMiniMaxH3LatentAV
    info["EmptyMiniMaxH3LatentAV"] = shared.native_info("EmptyMiniMaxH3LatentAV", EmptyMiniMaxH3LatentAV)
    info["LoadImage"] = shared.native_info("LoadImage", nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    return json.loads(json.dumps(info))


def build_candidate(variant, task, info, *, graph_override=None):
    source = graph_override if graph_override is not None else split_graph(variant, task)
    graph, selected = shared.selected_frontend_schema(source, info)
    title = f"Progressive {task} — editable native LOW / learned lift / HIGH — {variant} EXP"
    workflow = shared.convert(graph, selected, title)
    mapping = {key: index + 1 for index, key in enumerate(graph)}
    layout = {
        "90": (0, 0, "Shared geometry only — no prompt/model dependency"),
        "1": (0, 420, "LOW MODEL — insert independent LoRAs"),
        "10": (370, 0, "Native Euler 20 — full video/audio clocks"),
        "26": (740, 0, "Plan + LOW source — no HIGH prompt/MODEL input"),
        "9": (370, 420, "LOW prompt encoded at target geometry; guide resized explicitly"),
        "12": (1100, 0, "LOW conditioning only"), "13": (1480, 0, "LOW sampling only"),
        "11": (1110, 350, "LOW actual joint noise"), "50": (1850, 0, "Save typed evolving LOW boundary"),
        "22": (0, 1450, "HIGH MODEL — independent LoRAs"),
        "92": (370, 1450, "HIGH native Euler coordinates — no sampling"),
        "20": (740, 1450, "Boundary → VAE coordinates only; evolving audio stays in boundary"),
        "23": (1120, 1450, "Existing learned 3D video upscale — externally editable"),
        "24": (1480, 1450, "Independent HIGH prompt — actual learned output size"),
        "25": (1840, 1450, "HIGH restart / original evolving audio / clean anchor"),
        "27": (1850, 2020, "HIGH video noise — default LOW seed + 1"),
        "28": (2240, 1450, "HIGH conditioning only"), "29": (2600, 1450, "HIGH sampling only"),
        "51": (2980, 1450, "Save completed HIGH AV"), "14": (3370, 1450, "Decode final HIGH AV"),
        "15": (3770, 1450, "Create joint video"), "16": (4140, 1450, "Save video"),
        "60": (0, 0, "Frozen LOW path + SHA — LOW body is absent"),
        "61": (0, 0, "Frozen completed HIGH — all sampling bodies absent"),
        "6": (-450, 0, "Shared text encoder"), "7": (-450, 400, "Video VAE"),
        "8": (-450, 720, "Audio VAE"), "91": (-450, 1030, "First-frame reference"),
        "30": (2600, 2230, "Actual execution / loading report")}
    for node in workflow["nodes"]:
        key = next(key for key, node_id in mapping.items() if node_id == node["id"])
        if key in layout:
            x, y, label = layout[key]
            node.update(pos=[x, y], title=label)
    text = ("原生 Progressive 分离式候选 / EXP：不是 FastH3 V2，也不是整体 Loop。默认Stock20=LOW10+HIGH10。\n\n"
        "LOW模型/LoRA/提示词/联合噪声与HIGH模型/LoRA/提示词/视频噪声分别外露。共同几何不依赖HIGH提示词。"
        "LOW边界是模型坐标clean_video+仍演化的audio_next，不能当普通LATENT或锁死完成音轨。"
        "原learned 3D节点在图中实际执行，target_dimensions取冻结Plan目标，不插值代替。\n\n"
        "默认HIGH视频噪声seed=LOW seed+1，HIGH采样器seed=LOW seed；改LOW seed时按需同步这两处。"
        "需要改变HIGH剩余表可另接high_sigmas，但起点必须等于冻结resume_sigma且dtype一致。\n\n"
        "Save生成新artifact_path和精确SHA，二者都要保留。恢复图真正删除LOW链；加载完成HIGH图不加载扩散模型、"
        "不运行任何采样或放大。必须填写真实path/SHA，占位符不可运行。\n\n"
        "I2VA示例采用已有本机参考文件；在其它机器需重新选择实际首帧。改变共享尺寸/首帧/计划会按依赖重算。"
        "这是empty T2VA/I2VA图，不冒充Avatar/accepted-parent续段；负条件复用本阶段输入只适用于此CFG1候选。\n\n"
        "当前仍待Progressive独立EAV/Relay效果适配、浏览器保存重载、完整预训练GPU与人审。旧节点、旧示例不替换。")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / Qualification boundaries",
        "pos": [0, -780], "size": [1350, 710], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [text]})
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
    for task in ("t2va", "i2va"):
        for variant in VARIANTS:
            graph, workflow, audit = build_candidate(variant, task, info)
            validation = asyncio.run(execution.validate_prompt("t8-progressive-modular", deepcopy(graph), None))
            name = f"Progressive_{task}_{variant}_EXP"
            shared.write_new(destination / (name + ".api.json"), graph)
            shared.write_new(destination / (name + ".json"), workflow)
            expected_outputs = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
            all_outputs_valid = bool(validation[0] and not validation[3]
                                     and set(validation[2]) == expected_outputs)
            rows.append({"name": name, "serialization": audit, "core_validation": validation,
                         "all_outputs_valid": all_outputs_valid})
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "Live CPU schema/API serialization only, no browser roundtrip or trained GPU/media quality."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
