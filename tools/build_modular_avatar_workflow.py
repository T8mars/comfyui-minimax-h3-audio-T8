"""Recording-driven Avatar graphs, independent stage effects and explicit recovery."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_progressive_effect_workflow as effects  # noqa: E402

base = effects.base
ROOT = base.ROOT
COMBINATIONS = (("plain", "both"),) + tuple((kind, scope) for kind in effects.KINDS for scope in effects.SCOPES)
RECORDING = "h3_twopass_voice_5683_5p152s.flac"


def split_graph(task="i2va", kind="plain", scope="both", variant="minimal"):
    if (kind, scope) not in COMBINATIONS or variant not in base.VARIANTS:
        raise ValueError("Unknown Avatar graph variant")
    graph = (base.split_graph(variant, task) if kind == "plain"
             else effects.split_graph(task, kind, scope, variant))
    graph["130"] = {"class_type": "LoadAudio", "inputs": {"audio": RECORDING}}
    graph["131"] = {"class_type": "MiniMaxH3AudioWindowT8", "inputs": {
        "audio": ["130", 0], "scene_start_seconds": 0., "scene_duration_seconds": 73 / 24,
        "warmup_seconds": 0., "cooldown_seconds": 0., "ensure_minimum_context": False}}
    if variant != "load_high":
        graph["90"]["inputs"]["length"] = ["131", 1]
        graph["132"] = {"class_type": "MiniMaxH3AudioLatentControlT8", "inputs": {
            "av_latent": ["90", 0], "source_audio": ["131", 0], "audio_vae": ["8", 0],
            "mode": "lock", "strength": .35}}
        graph["133"] = {"class_type": "MiniMaxH3AvatarSourceBindEXPT8", "inputs": {
            "high_source": ["132", 0], "original_recording": ["131", 0]}}
        if "26" in graph:
            graph["26"]["inputs"].update(high_source=["133", 0], input_mode="initialized_av_exp", low_evaluations=4)
            graph["13"]["class_type"] = "MiniMaxH3AvatarLowStageEXPT8"
            graph["13"]["inputs"]["avatar_source"] = ["133", 1]
        graph["25"]["class_type"] = "MiniMaxH3AvatarHighHandoffEXPT8"
        graph["25"]["inputs"].pop("high_source")
        graph["25"]["inputs"]["avatar_source"] = ["133", 1]
        for key in ("10", "92"):
            if key in graph:
                graph[key]["inputs"]["steps"] = 8
        # Only the neutral source encodes/locks the recording. Stage encoders
        # produce independently editable native conditions, not another audio anchor.
        for key in ("9", "24"):
            if key in graph:
                graph[key]["inputs"].update(audio_mode="native", add_source_as_reference=False)
                graph[key]["inputs"].pop("drive_audio", None)
                if "prompt" in graph[key]["inputs"]:
                    graph[key]["inputs"].update(length=["131", 1], prompt=
                        "Stable shot of a woman speaking in synchronization with the supplied recording, "
                        "natural facial movements, stable lighting, no cut, no subtitles.")
        for key in ("100", "110"):
            if key in graph:
                graph[key]["inputs"].update(length=["131", 1],
                    global_prompt="Stable shot of a woman speaking in synchronization with the supplied recording.",
                    local_prompts="She looks toward the camera while speaking.\nShe continues speaking with restrained natural movements.")
    result = ["61", 1] if variant == "load_high" else (["114", 0] if "114" in graph else ["29", 2])
    graph["134"] = {"class_type": "MiniMaxH3AvatarDeliveryAuditEXPT8", "inputs": {
        "high_result": result, "original_recording": ["131", 0]}}
    graph["135"] = {"class_type": "MiniMaxH3AVLatentSeparateT8", "inputs": {"av_latent": ["134", 0]}}
    graph["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["135", 0], "vae": ["7", 0]}}
    graph["15"]["inputs"].update(images=["14", 0], audio=["134", 1])
    graph["30"]["inputs"]["source"] = ["134", 2]
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Avatar_{task}_{kind}_{scope}_{variant}_EXP"
    outputs = ["16", "30", "134"] + [key for key in ("50", "51", "105", "115") if key in graph]
    return base.prune(graph, outputs)


def load_live_info():
    info = base.load_live_info()
    import nodes
    from comfy_extras.nodes_audio import LoadAudio
    nodes.NODE_CLASS_MAPPINGS["LoadAudio"] = LoadAudio
    for name in ("LoadAudio", "VAEDecode"):
        info[name] = base.shared.native_info(name, nodes.NODE_CLASS_MAPPINGS[name])
    return json.loads(json.dumps(info))


def build_candidate(task, kind, scope, variant, info, *, graph_override=None):
    source = split_graph(task, kind, scope, variant) if graph_override is None else graph_override
    graph, selected = base.shared.selected_frontend_schema(source, info)
    workflow = base.shared.convert(graph, selected, f"Avatar {task} / {kind} / {scope} / {variant} EXP")
    if kind == "plain":
        old_graph, previous, _ = base.build_candidate(variant, task, info)
    else:
        old_graph, previous, _ = effects.build_candidate(task, kind, scope, info, variant)
    old_ids = {key: i + 1 for i, key in enumerate(old_graph)}
    current_ids = {key: i + 1 for i, key in enumerate(graph)}
    previous_nodes = {node["id"]: node for node in previous["nodes"]}
    current_nodes = {node["id"]: node for node in workflow["nodes"]}
    for key in set(old_ids) & set(current_ids):
        current_nodes[current_ids[key]].update(pos=previous_nodes[old_ids[key]]["pos"], title=previous_nodes[old_ids[key]]["title"])
    labels = {"130": (-1400, 300, "原录音 / Select your actual recording"),
        "131": (-1400, 700, "显式选取录音窗口 / no hidden trim"),
        "132": (-900, 1100, "单独编码并锁定原音 / no HIGH prompt dependency"),
        "133": (-300, -400, "绑定原音与编码AV / selected pair, not VAE provenance"),
        "13": (1480, 0, "Avatar LOW only / original audio locked"),
        "25": (1840, 1450, "Avatar HIGH handoff / retain source recording anchor"),
        "134": (3350, 2400, "原音交付及绑定核验 / never generated audio mux"),
        "135": (3750, 2400, "Separate completed video latent"),
        "14": (4100, 2400, "Video VAE decode only"),
        "26": (740, 0, "Initialized Avatar Plan / LOW4 + HIGH4"),
        "10": (370, 0, "LOW native Euler 8-step clock"),
        "92": (370, 1450, "Independent HIGH native Euler coordinates")}
    for key, (x, y, label) in labels.items():
        if key in current_ids:
            current_nodes[current_ids[key]].update(pos=[x, y], title=label)
    note = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note, "type": "MarkdownNote", "title": "Avatar split / read first",
        "pos": [-1500, -1350], "size": [1300, 820], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "Avatar 原录音驱动分离图 / EXP。LOW与HIGH独立MODEL/LoRA/提示词，learned3D放大外置；"
            "EAV/Relay可只LOW、只HIGH或两采分别启用。默认4+4 native Euler，不是FastH3 V2。\n\n"
            "原录音先经显式AudioWindow选择，再单独Audio Latent Control锁定编码音频；Bind记录所选编码AV/PCM对，"
            "不证明任意手接LATENT确由该PCM编码。Plan必须initialized_av_exp，音频mask必须全0。"
            "修改共同录音或源几何会使LOW身份改变；HIGH专属提示/模型改动不应重跑LOW。\n\n"
            "HIGH交付使用原PCM，不接AV Decode的生成音频。此图仅视频VAE解码，无音频重编码掩饰。"
            "Load HIGH只需同一录音窗口与完成HIGH path/SHA，无LOW模型、音频编码、放大或任何采样。"
            "Resume HIGH重建并核对同一编码源/录音，不重跑LOW。更换VAE导致字节改变时不能冒认同一冻结源。\n\n"
            "Save保留真实path和SHA；恢复占位符必须替换。本机录音/首帧文件在其它机器需重选。"
            "EAV默认report_only，apply_exp才实际应用；未知后端保持执行，但持久恢复资格另验。\n\n"
            "这批证据是tiny CPU/明确插值替身、Core图校验，不是trained learned/GPU/口型声音/接缝/浏览器人审资格。"
            "旧一体节点和旧图保留，不强制迁移。"]})
    workflow["last_node_id"] = note
    return graph, workflow, base.shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = load_live_info()
    import execution
    rows = []
    for task in ("t2va", "i2va"):
        for kind, scope in COMBINATIONS:
            for variant in base.VARIANTS:
                graph, workflow, audit = build_candidate(task, kind, scope, variant, info)
                validation = asyncio.run(execution.validate_prompt("modular-avatar", deepcopy(graph), None))
                expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
                valid = bool(validation[0] and not validation[3] and set(validation[2]) == expected)
                name = f"Avatar_{task}_{kind}_{scope}_{variant}_EXP"
                base.shared.write_new(target / (name + ".api.json"), graph)
                base.shared.write_new(target / (name + ".json"), workflow)
                rows.append({"name": name, "serialization": audit, "core_validation": validation, "all_outputs_valid": valid})
    base.shared.write_new(target / "audit.json", {"candidates": rows,
        "qualification": "CPU live Core/API and frontend serialization, not browser/trained learned/GPU/media qualification."})
    base.shared.write_new(target / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(target)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
