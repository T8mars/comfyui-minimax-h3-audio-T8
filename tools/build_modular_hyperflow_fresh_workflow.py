"""Editable full8/partial4 -> original learned3D -> fresh4, never hidden two-pass."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_hyperflow_workflow as base  # noqa: E402

ROOT, shared = base.ROOT, base.shared
ROUTES = ("full8plus4", "partial4plus4")
EFFECTS = (("none", "both"), *((kind, scope) for kind in ("eav", "relay", "combined")
                              for scope in ("low", "high", "both")))
CASES = tuple((route, kind, scope, variant) for route in ROUTES for kind, scope in EFFECTS
              for variant in ("minimal", "save", "resume_high")) + tuple(
                  (route, "none", "both", "load_high") for route in ROUTES)


def split_graph(route="full8plus4", kind="combined", scope="both", variant="minimal", *,
                artifact_path="SELECT_SAVED_STAGE/manifest.json", artifact_sha256="0" * 64):
    if (route, kind, scope, variant) not in CASES:
        raise ValueError("Unknown HyperFlow fresh route/effect/storage candidate")
    old = shared.split_graph()
    graph = {key: deepcopy(old[key]) for key in
             ("1", "22", "6", "7", "8", "9", "11", "23", "24", "27", "14", "15", "16")}
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/HyperFlow_{route}_{kind}_{scope}_{variant}_EXP"
    low_stage = "hyperflow_low_full8" if route == ROUTES[0] else "hyperflow_low_partial4"
    high_stage = "hyperflow_high_after_full8" if route == ROUTES[0] else "hyperflow_high_after_partial4"
    templates = shared.split_graph_with_effects()
    for phase, encoder, loader, bare, setup, sampler, noise, guider, group in (
        ("low", "9", "101", "1", "10", "13", "11", "12", 200),
        ("high", "24", "102", "22", "26", "29", "27", "28", 210)):
        graph[bare]["inputs"]["unet_name"] = "minimax_h3_fl2va_int8_convrot.safetensors"
        graph[loader] = {"class_type": "MiniMaxH3HyperFlowFreshLoaderEXPT8", "inputs": {
            "model": [bare, 0], "hyperflow_file": base.WEIGHT}}
        model, positive, template = [loader, 0], [encoder, 0], [encoder, 1]
        active = scope in (phase, "both")
        config, apply, audit, plan = (str(group + offset) for offset in (2, 3, 4, 5))
        if active and kind in ("relay", "combined"):
            graph[plan] = deepcopy(templates["40"])
            graph[plan]["inputs"]["global_prompt"] += f" Independent {phase} stage prompt."
            graph[encoder]["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
            graph[encoder]["inputs"].pop("prompt")
            graph[encoder]["inputs"].pop("length")
            graph[encoder]["inputs"].update(model=model, prompt_relay_plan=[plan, 0],
                                            execution_mode="apply_exp", query_chunk_rows=256)
            model, positive, template = [encoder, 0], [encoder, 1], [encoder, 2]
        source = template if phase == "low" else ["23", 0]
        graph[setup] = {"class_type": "MiniMaxH3HyperFlowFreshStageSetupEXPT8", "inputs": {
            "model": model, "av_latent": source, "stage": low_stage if phase == "low" else high_stage}}
        model = [setup, 0]
        if active and kind in ("eav", "combined"):
            graph[config] = deepcopy(templates["41"])
            graph[apply] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
                "model": model, "sigmas": [setup, 2], "av_latent": source,
                "stage_context": [setup, 3], "eav_config": [config, 0]}}
            model = [apply, 0]
        graph[guider] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": positive}}
        graph[sampler] = {"class_type": "MiniMaxH3StageSamplerEXPT8", "inputs": {
            "noise": [noise, 0], "guider": [guider, 0], "sampler": [setup, 1],
            "sigmas": [setup, 2], "latent_image": source, "stage_context": [setup, 3]}}
        graph[audit] = {"class_type": "MiniMaxH3HyperFlowFreshStageAuditEXPT8", "inputs": {
            "stage_result": [sampler, 2]}}
    graph["20"] = {"class_type": "MiniMaxH3HyperFlowFreshLiftInputEXPT8", "inputs": {"stage_result": ["204", 0]}}
    graph["23"]["inputs"]["av_latent"] = ["20", 0]
    # Original public/Director fresh routes feed learned AV directly to HIGH.
    # Do not add a legacy audio-reconcile policy or freeze the audio silently.
    graph["14"]["inputs"]["av_latent"] = ["214", 1]
    graph["30"] = {"class_type": "PreviewAny", "inputs": {"source": ["214", 3]}}
    graph["31"] = {"class_type": "PreviewAny", "inputs": {"source": ["204", 3]}}
    outputs = ["16", "30", "31"]
    if variant in ("save", "resume_high"):
        for key, audit, stage in (("50", "204", "LOW"), ("51", "214", "HIGH")):
            graph[key] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
                "stage_result": [audit, 0], "prefix": f"HyperFlowFresh/{route}/{stage}"}}
        graph["14"]["inputs"]["av_latent"] = ["51", 0]
        outputs.extend(["50", "51"])
    if variant == "resume_high":
        graph["60"] = {"class_type": "MiniMaxH3HyperFlowFreshStageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": low_stage}}
        graph["20"]["inputs"]["stage_result"] = ["60", 3]
        graph["31"]["inputs"]["source"] = ["60", 4]
        outputs.remove("50")
    if variant == "load_high":
        graph["60"] = {"class_type": "MiniMaxH3HyperFlowFreshStageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": high_stage}}
        graph["214"]["inputs"]["stage_result"] = ["60", 3]
        outputs.remove("31")
    return base.common.prune(graph, outputs)


def build_candidate(route, kind, scope, variant, info, *, graph_override=None):
    graph, selected = shared.selected_frontend_schema(
        split_graph(route, kind, scope, variant) if graph_override is None else graph_override, info)
    title = f"HyperFlow {route} / {kind} / {scope} / {variant} EXP"
    workflow = shared.convert(graph, selected, title)
    positions = {"1": (-400, 0), "22": (-400, 1700), "101": (0, 0), "102": (0, 1700),
        "6": (-900, 400), "7": (-900, 750), "8": (-900, 1100), "9": (420, 0), "24": (420, 1700),
        "10": (900, 0), "26": (900, 1700), "11": (950, 650), "27": (950, 2350),
        "12": (1800, 0), "28": (1800, 1700), "13": (2200, 0), "29": (2200, 1700),
        "20": (3000, 0), "23": (0, 2900), "14": (3400, 1700), "15": (3800, 1700),
        "16": (4200, 1700), "30": (2950, 2400), "31": (2950, 650),
        "50": (3400, 0), "51": (3400, 2900), "60": (-500, 0)}
    for group, y in ((200, 0), (210, 1700)):
        for offset, x, dy in ((2, 1400, 650), (3, 1400, 0), (4, 2600, 0), (5, 0, -850)):
            positions[str(group + offset)] = (x, y + dy)
    for key, node in zip(graph, workflow["nodes"]):
        node["pos"] = list(positions[key])
        node["title"] = selected[graph[key]["class_type"]].get("display_name", graph[key]["class_type"])
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / Explicit fresh-noise stages",
        "pos": [-2100, -700], "size": [1100, 950], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "HyperFlow full8+fresh4 / partial4+fresh4 分离候选，不是 continuous 或 P7。\n\n"
            "两路原生完整 H3 底模和新 Fresh Route Loader，上游可各插自己的内容 LoRA；"
            "新 Loader 从 Core 原始备份读取 endpoint，不把前一阶段驻留的 LoRA 当原底模，不卸载共享模型。"
            "不要用普通 LoRA Loader 加载 HyperFlow 本身。LOW/HIGH 提示、条件和 NOISE 独立。"
            "一采 full8 用 output；partial4 用 denoised_output。Lift Input 按真实结果选择，"
            "不是把 x_sigma 假装 clean x0。现有 learned3D 留在外部，HIGH 条件尺寸接其真实输出。"
            "沿用原 fresh 路线的 learned AV 直接交接，不额外添加音频锁定策略；HIGH 保留原 fresh joint noise/audio rebase。\n\n"
            "EAV Config 可仅 LOW/仅 HIGH/两边；默认 report_only 不施加增益，apply_exp 才启用。"
            "Relay 各自 Plan + 配对 MODEL/CONDITIONING；CFG1。Audit 读取完成回执的实际调用。"
            "未知补丁保留执行，但可能不满足持久复用认证。\n\n"
            "save 保存两采；resume_high 已裁掉 LOW 模型/编码/噪声/采样/效果和输出节点；"
            "Load 需填真实 Save path/SHA。load_high 仅读完成结果再解码，无模型/编码/扩散。"
            "不自动匹配当前 LOW 设置、不隐式重采、不改旧图或导演台。\n\n"
            "本批候选只做 tiny CPU/synthetic HyperFlow 与 Core 图核验；完整预训练/GPU/"
            "浏览器编辑重载/完整音画人审仍待。所需模型必须已安装，不自动下载。"]})
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
    for case in CASES:
        graph, workflow, audit = build_candidate(*case, info)
        validation = asyncio.run(execution.validate_prompt("hyperflow-fresh", deepcopy(graph), None))
        expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
        name = "HyperFlow_" + "_".join(case) + "_EXP"
        rows.append({"name": name, "serialization": audit, "core_validation": validation,
            "all_outputs_valid": bool(validation[0] and not validation[3] and set(validation[2]) == expected)})
        shared.write_new(destination / (name + ".api.json"), graph)
        shared.write_new(destination / (name + ".json"), workflow)
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "CPU live schemas/serialization only; not browser, pretrained GPU or human qualification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
