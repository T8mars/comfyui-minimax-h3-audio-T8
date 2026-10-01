"""VDN Relay-only/Relay+EAV/save/restore, including clean native HIGH branches."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_vdn_workflow as vdn  # noqa: E402
import build_modular_vdn_native_workflow as mixed  # noqa: E402

ROOT, shared = vdn.ROOT, vdn.shared
VARIANTS = ("relay", "relay_eav", "save_relay_eav", "resume_relay_eav")
BACKENDS = (("vdn", 4), ("native", 3), ("native", 4), ("native", 5))


def split_graph(training=vdn.TRAINING[0], backend="vdn", refine=4, variant="relay"):
    supported = (backend, refine) in BACKENDS or (
        training == "stage_b_50nfe" and (backend, refine) == ("vdn", 5))
    if training not in vdn.TRAINING or not supported or variant not in VARIANTS:
        raise ValueError("Unknown VDN external Relay workflow recipe")
    effects, saved = variant != "relay", variant in ("save_relay_eav", "resume_relay_eav")
    base = "save_eav" if saved else "eav" if effects else "minimal"
    graph = (vdn.split_graph(training, base, refine_steps=refine)
             if backend == "vdn" else mixed.split_graph(training, refine, base))
    for key, plan, model in (("9", "40", "70"), ("24", "140", "71")):
        item = graph[key]
        if item["class_type"] != "MiniMaxH3PromptRelayConditioningT8Advanced":
            item["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
            item["inputs"].pop("prompt")
            item["inputs"].pop("length")
            for target in graph.values():
                for name, value in tuple(target["inputs"].items()):
                    if isinstance(value, list) and len(value) == 2 and value[0] == key:
                        target["inputs"][name] = [key, {0: 1, 1: 2}[value[1]]]
        item["inputs"].update(model=[model, 0], prompt_relay_plan=[plan, 0], execution_mode="apply_exp", query_chunk_rows=256)
        graph[plan] = deepcopy(shared.split_graph_with_effects()["40"])
    graph["10"]["inputs"]["model"] = ["9", 0]
    if backend == "vdn":
        graph["26"]["inputs"]["model"] = ["24", 0]
    else:
        graph["92"]["inputs"]["model"] = ["24", 0]
    stages = [("110", "112", "10", "13", "12", "43", "45", "50", ["9", 2])]
    if backend == "vdn":
        stages.append(("111", "113", "26", "29", "28", "44", "46", "51", ["25", 0]))
    for apply, audit, setup, sampler, guider, effect, effect_audit, save, latent in stages:
        graph[apply] = {"class_type": "MiniMaxH3VDNRelayApplyEXPT8", "inputs": {
            "model": [setup, 0], "sigmas": [setup, 2], "av_latent": latent,
            "stage_context": [setup, 3], "mode": "apply_exp", "max_workspace_mib": 64}}
        graph[effect if effects else guider]["inputs"]["model"] = [apply, 0]
        completed = [effect_audit, 0] if effects else [save, 0] if saved else [sampler, 0]
        graph[audit] = {"class_type": "MiniMaxH3VDNRelayAuditEXPT8", "inputs": {
            "av_latent": completed, "runtime": [apply, 1]}}
    graph["23"]["inputs"]["av_latent"] = ["112", 0]
    if backend == "vdn":
        graph["26"]["inputs"]["first_pass_latent"] = ["112", 0]
        graph["94"]["inputs"]["second_pass_output"] = ["113", 0]
        graph["115"] = {"class_type": "PreviewAny", "inputs": {"source": ["113", 1]}}
    graph["114"] = {"class_type": "PreviewAny", "inputs": {"source": ["112", 1]}}
    for key, label in (("50", "LOW"), ("51", "HIGH")):
        if key in graph:
            graph[key]["inputs"]["prefix"] = f"VDNRelay/{training}/{backend}{refine}/{label}"
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/VDNRelay_{training}_{backend}{refine}_EXP"
    if variant == "resume_relay_eav":
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": "REPLACE_WITH_VDN_LOW/manifest.json", "artifact_sha256": "0" * 64,
            "expected_stage": "vdn_complete"}}
        graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
        graph["23"]["inputs"]["av_latent"] = ["60", 0]
        if backend == "vdn":
            graph["26"]["inputs"]["first_pass_latent"] = ["60", 0]
        keep = set()
        def include(key):
            if key in keep:
                return
            keep.add(key)
            for value in graph[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    include(str(value[0]))
        for key in ("16", "32", "51", "61", "72", *(["115"] if backend == "vdn" else [])):
            include(key)
        graph = {key: item for key, item in graph.items() if key in keep}
    return graph


def build_candidate(training, backend, refine, variant, info, *, graph_override=None):
    source = (graph_override if graph_override is not None
              else split_graph(training, backend, refine, variant))
    graph, workflow, _ = shared.build_candidate(source, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title, pos in (
        ("10", "LOW original complete VDN trajectory", (950, 0)),
        ("26", f"HIGH independent {backend} stage", (1550, 1450)),
        ("70", "LOW original Composer / independent MODEL + LoRA", (-450, 0)),
        ("71", "HIGH independent Composer" if backend == "vdn" else "HIGH clean native MODEL + EMA B", (-450, 1450)),
        ("40", "LOW external Relay Plan", (-900, 0)), ("140", "HIGH external Relay Plan", (-900, 1450)),
        ("110", "LOW actual VDN window + linear Relay", (500, 2600)),
        ("111", "HIGH actual VDN window + linear Relay", (1000, 2600)),
        ("112", "LOW actual Relay audit", (2050, 500)), ("113", "HIGH actual Relay audit", (2700, 1450)),
        ("94", "Original completed-audio audit / relock", (3400, 1450)),
        ("92", "HIGH original native Dual-Clock", (750, 2500)), ("93", "HIGH original LBH refine table", (1200, 2500)),
        ("114", "LOW Relay report", (2400, 500)), ("115", "HIGH Relay report", (3200, 2200))):
        if key in ids:
            by_id[ids[key]].update(title=title, pos=list(pos))
    note = (f"VDN {training}完整LOW → {backend} HIGH{refine}；两阶段独立Relay Plan / MODEL / 条件 / NOISE。\n\n"
        "VDN路线明确接 Relay Conditioning→VDN Stage→VDN Relay Apply→可选Stage EAV→Guider。"
        "窗口保留原key集合和双anchor，叠加真实时间偏置；linear使用beta_weighted_text_seed_exp_v1，"
        "逐query帧重算非线性文本seed，保留video扫描。这是VDN实验扩展，不宣称paper softmax等价或已训练画质。"
        "关闭适配器返回原MODEL，中性权重精确旁路；观察实际Audit，未知producer可能绕过。\n\n"
        "EAV有各自配置，默认report_only，可只接一段。VDN LOW完成OUTPUT槽0→原learned→实际HIGH条件→"
        "原Reconcile/AudioAudit保留完成音频。native HIGH独立干净MODEL/EMA/LBH，不继承VDN分支。"
        "恢复图删除LOW整链及其Plan/效果/输出，只读冻结path/SHA后跑HIGH。\n\n"
        "本图为Core/序列化候选；tiny CPU数值/恢复不代表完整预训练/learned/GPU/浏览器/人审。"
        "workspace仅bias与head-chunk工作区估计，不是总显存保证；内核不支持正常报错，无静默Dense回退。旧图不动。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"VDN Relay {training} -> {backend}{refine} / {variant} EXP"
    _, selected = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    destination = parser.parse_args().output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = shared.load_live_info()
    import execution
    records = []
    for training in vdn.TRAINING:
        for backend, refine in BACKENDS:
            for variant in VARIANTS:
                graph, workflow, audit = build_candidate(training, backend, refine, variant, info)
                validation = asyncio.run(execution.validate_prompt("vdn-relay-stage", deepcopy(graph), None))
                name = f"VDNRelay_{training}_{backend}{refine}_{variant}_EXP"
                shared.write_new(destination / f"{name}.api.json", graph)
                shared.write_new(destination / f"{name}.json", workflow)
                records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "Core/serialization only. Experimental linear text weighting; no full-model/GPU/browser/human claim."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
