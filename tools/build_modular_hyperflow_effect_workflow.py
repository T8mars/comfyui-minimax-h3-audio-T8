"""Continuous HyperFlow independent stage effects, storage and TAIL-only replay."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_hyperflow_workflow as base  # noqa: E402

ROOT, shared = base.ROOT, base.shared
KINDS = ("eav", "relay", "combined")
SCOPES = ("head", "tail", "both")
VARIANTS = ("minimal", "save", "resume_tail")


def split_graph(split=4, kind="combined", scope="both", variant="minimal", *,
                artifact_path="SELECT_SAVED_HEAD/hyperflow-head.safetensors", artifact_sha256="0" * 64):
    if kind not in KINDS or scope not in SCOPES or variant not in VARIANTS:
        raise ValueError("Unknown HyperFlow external effect candidate")
    graph = base.split_graph(split)
    templates = shared.split_graph_with_effects()
    for phase, encoder, sampler, loader, group in (("head", "9", "13", "101", 200),
                                                   ("tail", "24", "29", "102", 210)):
        active = scope in (phase, "both")
        bind, config, apply, audit, plan = (str(group + offset) for offset in range(1, 6))
        model, positive = [loader, 0], [encoder, 0]
        if active and kind in ("relay", "combined"):
            graph[plan] = deepcopy(templates["40"])
            graph[plan]["inputs"].update(length=73,
                global_prompt="A continuous cinematic shot in a quiet concert hall, natural motion and ambience.",
                local_prompts="The woman walks toward the doorway.\nShe slows and turns to the right." if phase == "head"
                              else "The woman approaches the doorway in warm light.\nShe turns, her clothing moving naturally.")
            source = graph[encoder]
            source["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
            source["inputs"].pop("prompt")
            source["inputs"].pop("length")
            source["inputs"].update(model=model, prompt_relay_plan=[plan, 0], execution_mode="apply_exp", query_chunk_rows=256)
            model, positive = [encoder, 0], [encoder, 1]
        graph[bind] = {"class_type": "MiniMaxH3HyperFlow" + phase.title() + "EffectsBindEXPT8",
            "inputs": {"model": model, "positive": positive, "negative": positive}}
        graph[bind]["inputs"].update({"av_latent": ["90", 0], "split_interval": split} if phase == "head"
                                    else {"continuous_boundary": ["204", 0]})
        graph[sampler]["inputs"].update(model=[bind, 0], positive=[bind, 1], negative=[bind, 2])
        if active and kind in ("eav", "combined"):
            graph[config] = deepcopy(templates["41"])
            graph[config]["inputs"].update(tau=4. if phase == "head" else 2.)
            graph[apply] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
                "model": [bind, 0], "av_latent": [bind, 3], "sigmas": [bind, 4],
                "stage_context": [bind, 5], "eav_config": [config, 0]}}
            graph[sampler]["inputs"]["model"] = [apply, 0]
        graph[audit] = {"class_type": "MiniMaxH3HyperFlow" + phase.title() + "EffectsAuditEXPT8",
            "inputs": {"continuous_boundary" if phase == "head" else "completed_result":
                       [sampler, 0 if phase == "head" else 2]}}
    graph["29"]["inputs"]["continuous_boundary"] = ["204", 0]
    graph["14"]["inputs"]["av_latent"] = ["214", 1]
    graph["30"]["inputs"]["source"] = ["214", 2]
    graph["31"]["inputs"]["source"] = ["204", 1]
    if variant in ("save", "resume_tail"):
        graph["50"] = {"class_type": "MiniMaxH3HyperFlowHeadSaveEXPT8", "inputs": {
            "continuous_boundary": ["204", 0], "prefix": "HyperFlowEffects/HEAD"}}
        graph["51"] = {"class_type": "MiniMaxH3HyperFlowTailSaveEXPT8", "inputs": {
            "completed_result": ["214", 0], "prefix": "HyperFlowEffects/TAIL"}}
        for consumer in ("29", "211"):
            graph[consumer]["inputs"]["continuous_boundary"] = ["50", 0]
        graph["14"]["inputs"]["av_latent"] = ["51", 0]
    if variant == "resume_tail":
        graph["50"] = {"class_type": "MiniMaxH3HyperFlowHeadLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256}}
        graph["31"]["inputs"]["source"] = ["50", 1]
    graph["16"]["inputs"]["filename_prefix"] += f"_{kind}_{scope}_{variant}"
    return base.common.prune(graph, ["16", "30", "31"])


def build_candidate(split, kind, scope, variant, info, *, graph_override=None):
    graph, selected = shared.selected_frontend_schema(
        split_graph(split, kind, scope, variant) if graph_override is None else graph_override, info)
    workflow = shared.convert(graph, selected, f"HyperFlow {split}+{8-split} / {kind} / {scope} / {variant} EXP")
    positions = {"1": (-450, 0), "6": (-900, 400), "7": (-900, 800), "8": (-900, 1100),
        "90": (-450, -650), "101": (0, 0), "102": (0, 1750), "9": (450, 0), "24": (450, 1750),
        "11": (1200, 600), "13": (2100, 0), "29": (2100, 1750), "14": (3400, 1750),
        "15": (3800, 1750), "16": (4200, 1750), "30": (3000, 2400), "31": (3000, 650),
        "50": (3000, 0), "51": (3000, 1750)}
    for group, y in ((200, 0), (210, 1750)):
        for offset, x, dy in ((1, 900, 0), (2, 1250, 700), (3, 1650, 0), (4, 2550, 0), (5, 0, -1000)):
            positions[str(group+offset)] = (x, y+dy)
    for key, node in zip(graph, workflow["nodes"]):
        node["pos"] = list(positions[key])
        node["title"] = selected[graph[key]["class_type"]].get("display_name", graph[key]["class_type"])
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / External continuous effects",
        "pos": [-2000, -900], "size": [1000, 1000], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "HyperFlow continuous 外置效果候选：不是8+4/partial4+4/P7，不放大、不给二采重新加噪声。\n\n"
            "两路专用HyperFlow Loader上游可分别插内容LoRA；不要用普通LoRA节点加载HyperFlow本身。"
            "Relay Plan、配对MODEL/CONDITIONING和EAV Config两路独立，可只一采/只二采/两采。"
            "HEAD Bind的split须与HEAD采样器相同；TAIL从实际边界取绝对区间。"
            "Bind的stage_template只是EAV几何验证输入，不能代替TAIL的raw边界。\n\n"
            "EAV默认report_only不施加增益，改apply_exp才启用。时间窗始终为1-video_sigma，二采不从0重计。"
            "Relay默认apply_exp；要关闭，可选其disabled模式。当前候选CFG1，负条件复用本阶段正条件。"
            "审计读取已完成采样的不可变实际调用，不把配置存在当效果生效。未知补丁保留执行但覆盖/持久身份可能未验证。\n\n"
            "save保存真实HEAD/TAIL；resume_tail已移除HEAD模型/条件/噪声/效果/输出链。"
            "Load必须填写Save输出的真实path与SHA，占位符不能运行；不自动重采。冻结HEAD后可独立改TAIL内容与效果，"
            "原底模及HyperFlow adapter须一致。范围head的恢复图只保留冻结HEAD已有效果，TAIL不会自动启用。\n\n"
            "只有CPU tiny/synthetic adapter及Core图验证；完整trained权重/GPU/浏览器保存重载/音画人审仍待。"
            "不自动下载、不替换旧图，也不改导演台。"]})
    workflow["last_node_id"] = note_id
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = base.load_live_info()
    import execution
    rows = []
    for split in base.SPLITS:
        for kind in KINDS:
            for scope in SCOPES:
                for variant in VARIANTS:
                    graph, workflow, audit = build_candidate(split, kind, scope, variant, info)
                    validation = asyncio.run(execution.validate_prompt("hyperflow-effects", deepcopy(graph), None))
                    expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
                    name = f"HyperFlow_{split}plus{8-split}_{kind}_{scope}_{variant}_EXP"
                    rows.append({"name": name, "serialization": audit, "core_validation": validation,
                        "all_outputs_valid": bool(validation[0] and not validation[3] and set(validation[2]) == expected)})
                    shared.write_new(destination / (name + ".api.json"), graph)
                    shared.write_new(destination / (name + ".json"), workflow)
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "Live CPU schema/serialization only. Load placeholders need actual saved path/SHA. "
                         "Not pretrained GPU/media/browser/human qualification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
