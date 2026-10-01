"""VDN complete LOW -> clean native EMA/LBH HIGH, each independently editable."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_vdn_workflow as vdn  # noqa: E402

ROOT, shared = vdn.ROOT, vdn.shared


def split_graph(training=vdn.TRAINING[0], refine=4, variant="minimal"):
    if type(refine) is not int or refine not in (3, 4, 5):
        raise ValueError("Original native LBH HIGH supports 3/4/5")
    graph = vdn.split_graph(training, variant)
    graph["71"] = {"class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced", "inputs": {
        "model": ["22", 0], "lora_name": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors", "strength_model": 1.}}
    high_model = ["71", 0]
    if variant != "minimal":
        # This older candidate selects HIGH Relay only. For LOW VDN's dedicated
        # linear text-state adapter use build_modular_vdn_relay_workflow.
        high = graph["24"]
        high["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
        high["inputs"].pop("prompt")
        high["inputs"].pop("length")
        high["inputs"].update(model=high_model, prompt_relay_plan=["40", 0],
                               execution_mode="apply_exp", query_chunk_rows=256)
        for item in graph.values():
            for name, value in tuple(item["inputs"].items()):
                if isinstance(value, list) and len(value) == 2 and value[0] == "24":
                    item["inputs"][name] = ["24", {0: 1, 1: 2}[value[1]]]
        graph["40"] = shared.split_graph_with_effects()["40"]
        high_model = ["24", 0]
    graph["92"] = {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
        "model": high_model, "av_latent": ["25", 0], "steps": 8, "shift_video": 12., "shift_audio": 3.,
        "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}}
    graph["93"] = {"class_type": "MiniMaxH3LearnedTwoPassParityPlanT8Advanced", "inputs": {
        "model": ["92", 0], "base_steps": 8, "coarse_steps": 4, "refine_steps": refine}}
    graph["26"] = {"class_type": "MiniMaxH3NativeStageBindEXPT8", "inputs": {
        "model": ["92", 0], "sampler": ["92", 1], "sigmas": ["93", 1],
        "av_latent": ["25", 0], "stage": "native_high"}}
    for key, label in (("50", "LOW"), ("51", "HIGH")):
        if key in graph:
            graph[key]["inputs"]["prefix"] = f"VDNToNative/{training}/HIGH{refine}/{label}"
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/VDN_{training}_Native_HIGH{refine}_EXP"
    return graph


def build_candidate(training, refine, variant, info):
    graph, workflow, _ = shared.build_candidate(split_graph(training, refine, variant), info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title, position in (
        ("10", "VDN complete LOW — original DMD8 / B50", (800, 0)),
        ("13", "PASS 1 completed OUTPUT — never LBH coarse", (1550, 0)),
        ("70", "LOW original VDN Composer", (-400, 0)),
        ("71", "HIGH clean MODEL + native EMA B — no VDN Composer", (-400, 1300)),
        ("24", "HIGH actual-size conditioning / independent Relay", (250, 1500)),
        ("26", "HIGH native stage — not VDN own-grid tail", (1550, 1300)),
        ("29", f"PASS 2 native LBH {refine} steps", (2300, 1300)),
        ("40", "HIGH-only external Relay plan — LOW VDN pending", (200, 2700)),
        ("92", "HIGH original Dual-Clock setup", (750, 2500)),
        ("93", "HIGH original LBH refine table only", (1200, 2500)),
        ("94", "Original locked-audio audit / relock before decode", (3300, 1500)),
        ("72", "Audio audit report", (3700, 1500))):
        if key in ids:
            by_id[ids[key]].update(title=title, pos=list(position))
    note = (f"VDN {training}完整首采→原learned3D→独立原生EMA B / LBH HIGH{refine}。\n\n"
        "这是原VDN→native比较路线，不把首采替换成普通native8，也不把LOW接LBH coarse表。"
        "HIGH来自第二个干净UNET Loader，内容LoRA可另接，绝不继承VDN branch/DiT/layout。"
        "LOW完成OUTPUT槽0进入learned，HIGH条件尺寸跟随实际放大输出；原Reconcile和AudioAudit保持完成音频。\n\n"
        "两阶段EAV分别外置、默认report_only；本图仅native HIGH接Relay，LOW专属Relay另有独立构图器。"
        "不以HIGH Relay存在声称LOW也生效。恢复图真正去掉LOW模型/采样/效果，只读冻结LOW path/SHA后跑HIGH。"
        "tiny CPU/Core/序列化候选，非完整权重/learned/GPU/浏览器/人审；旧工作流不改。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"VDN {training} -> native HIGH{refine} / {variant} EXP"
    _, selected = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = shared.load_live_info()
    import execution
    records = []
    for training in vdn.TRAINING:
        for refine in (3, 4, 5):
            for variant in vdn.VARIANTS:
                graph, workflow, audit = build_candidate(training, refine, variant, info)
                validation = asyncio.run(execution.validate_prompt("vdn-native-stage", deepcopy(graph), None))
                name = f"VDN_{training}_Native_HIGH{refine}_{variant}_EXP"
                shared.write_new(destination / f"{name}.api.json", graph)
                shared.write_new(destination / f"{name}.json", workflow)
                records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "Core/serialization only; LOW VDN Relay is not selected in this graph. No browser/GPU/human certification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
