"""Build CPU-validated legacy Dual split candidates; never queue sampling."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_fast_h3_v2_workflow as shared  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("minimal", "effects", "save_effects", "resume_effects")


def split_graph(coarse=4, refine=4, variant="minimal", *, artifact_path="REPLACE_WITH_SAVED_LOW_PATH/manifest.json",
                artifact_sha256="0" * 64):
    if type(coarse) is not int or coarse not in (4, 20) or type(refine) is not int or refine not in (3, 4, 5):
        raise ValueError("Native Dual supports LOW4/LOW20 and HIGH3/4/5")
    if variant not in VARIANTS:
        raise ValueError("Unknown Native Dual workflow variant")
    effects = variant != "minimal"
    graph = shared.split_graph_with_effects() if effects else shared.split_graph()
    if variant in ("save_effects", "resume_effects"):
        graph = shared.split_graph_with_results(graph)
        graph["50"]["inputs"]["prefix"] = f"NativeDual/LOW{coarse}"
        graph["51"]["inputs"]["prefix"] = f"NativeDual/HIGH{refine}"
    for key in ("1", "22"):
        graph[key]["inputs"]["unet_name"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    # Stock20 LOW uses the base MODEL. Partial4 LOW and the LBH HIGH branches
    # retain explicitly editable EMA B loaders; content LoRA chains may follow.
    for key, base in (("70", "1"), ("71", "22")):
        if key == "70" and coarse == 20:
            continue
        graph[key] = {"class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced", "inputs": {
            "model": [base, 0], "lora_name": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors",
            "strength_model": 1.0}}
    low_model = "70" if coarse == 4 else "1"
    if effects:
        graph["9"]["inputs"]["model"] = [low_model, 0]
        graph["24"]["inputs"]["model"] = ["71", 0]
    for key, stage, selected_model in (("10", f"dual_low_{coarse}", low_model),
                                        ("26", f"dual_high_{refine}", "71")):
        item = graph[key]
        item["class_type"] = "MiniMaxH3NativeDualStageSetupEXPT8"
        item["inputs"].pop("profile")
        item["inputs"].pop("min_tokens")
        item["inputs"].update(stage=stage, shift_video=12., shift_audio=3.)
        if not effects:
            item["inputs"]["model"] = [selected_model, 0]
    handoff = graph["25"]
    handoff["class_type"] = "MiniMaxH3NativeDualHandoffEXPT8"
    handoff["inputs"].pop("audio_policy")
    handoff["inputs"].pop("second_pass_audio_source")
    handoff["inputs"].pop("second_pass_audio_strength")
    handoff["inputs"].update(first_pass_steps=str(coarse), second_audio_source="auto", second_audio_strength=0.)
    # The shared V2 saved graph also inserts a V2-only completed LOW guard.
    # Native Dual has a different verified recipe; its own sampler's x0 port
    # is the correct learned handoff. Never run the V2 guard on this result.
    graph.pop("62", None)
    if effects:
        graph["45"]["inputs"]["av_latent"] = ["13", 1]
        graph["23"]["inputs"]["av_latent"] = ["45", 0]
    else:
        graph["23"]["inputs"]["av_latent"] = ["13", 1]
    # V2 runtime auditing must not be attached to a native Dual MODEL.
    for key in ("19", "20", "30", "31"):
        graph.pop(key)
    graph["14"]["inputs"]["av_latent"] = ["46", 0] if effects else ["29", 0]
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_Native_Dual_{coarse}plus{refine}_EXP"
    graph["72"] = {"class_type": "PreviewAny", "inputs": {"source": ["25", 2]}}
    if variant == "resume_effects":
        # The full graph delays HIGH loading until LOW completes. A cold
        # HIGH-only graph must load directly from its frozen LOW artifact;
        # leaving completed_stage attached would pull LOW back into closure.
        loader = graph["22"]
        if loader["class_type"] != "MiniMaxH3StageUNETLoaderAfterEXPT8":
            raise ValueError("Expected the serialized native HIGH loader")
        graph["22"] = {"class_type": "UNETLoader", "inputs": {
            name: value for name, value in loader["inputs"].items()
            if name != "completed_stage"}}
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256,
            "expected_stage": f"dual_low_{coarse}"}}
        graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
        graph["23"]["inputs"]["av_latent"] = ["60", 1]
        keep = set()

        def include(key):
            if key in keep:
                return
            keep.add(key)
            for value in graph[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    include(str(value[0]))

        for key in ("16", "32", "51", "61", "72"):
            include(key)
        graph = {key: item for key, item in graph.items() if key in keep}
    return graph


def build_candidate(coarse, refine, variant, info, *, graph_override=None):
    source = graph_override if graph_override is not None else split_graph(coarse, refine, variant)
    graph, workflow, _ = shared.build_candidate(source, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    note = (f"原生 Dual 分离式 LOW{coarse} → learned3D → HIGH{refine} / EXP 候选，未做完整GPU或人审。\n\n"
            "LOW4是simple8前四区间，LOW20是完整native-flow；HIGH3/4/5是独立LBH表，各自MODEL和shift。"
            "不是FastH3 V2模型或DMD切片。两路模型、LoRA、条件和noise均可编辑，不运行隐藏Loop。\n\n"
            "LOW必须取denoised_output进入原learned3D；HIGH条件的尺寸来自放大器输出。"
            "Native Dual Handoff保留旧音频策略：LOW4 auto继续联合音频、LOW20 auto保留已完成音频；"
            "模板锁定前缀及历史first_pass/零强度迁移保留。最终解码HIGH output。\n\n"
            "LOW4和HIGH默认显式EMA B加速LoRA；LOW20使用原生底模。可独立追加内容LoRA。"
            "此新组合没有继承旧长片样例的人审资格；没有改变旧一体节点或正式工作流。")
    if variant != "minimal":
        note += ("\n\nRelay外置Plan在实际LOW/HIGH尺寸分别绑定MODEL与CONDITIONING；可复制Plan独立设置。"
                 "两个EAV配置独立，默认report_only，仅审计。绝对1-video_sigma窗口不在阶段开始重置；"
                 "默认窗口可能没有覆盖LOW4。未知后端保留执行但审计覆盖和持久身份可能未适配。")
    if variant in ("save_effects", "resume_effects"):
        note += ("\n\nStage Sampler只执行一个原生阶段；Save生成新冻结目录，保留artifact_path和SHA。"
                 "Load显式选择该旧LOW结果，不自动判断当前更改的LOW设置是否匹配。坏SHA/错阶段拒绝。"
                 "CPU tiny保存/恢复不等于完整实尺寸learned3D或GPU效果已验收。")
    if variant == "resume_effects":
        note += "\n\n本图已实际移除全部LOW模型/条件/采样/效果/输出节点；先填真实保存path与SHA，占位符不能运行。"
    title_map = {"10": f"LOW {coarse} — original native Dual schedule",
                 "26": f"HIGH {refine} — independent published LBH schedule",
                 "25": "Original Dual audio handoff — locked prefix / joint policy",
                 "70": "LOW acceleration LoRA — insert content LoRAs independently",
                 "71": "HIGH acceleration LoRA — insert content LoRAs independently",
                 "72": "Actual handoff / audio-policy report"}
    positions = {"70": (-400, 0), "71": (-400, 1300), "72": (1200, 2300)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title in title_map.items():
        if key in ids:
            by_id[ids[key]]["title"] = title
            if key in positions:
                by_id[ids[key]]["pos"] = list(positions[key])
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    # The shared serializer's layout helpers do not change the graph's actual
    # algorithms. Replace its human-facing heading and re-audit serialization.
    workflow["extra"]["workflow_title"] = f"Native Dual LOW{coarse} / HIGH{refine} — {variant} EXP"
    _, selected_info = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected_info)


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
    for coarse in (4, 20):
        for refine in (3, 4, 5):
            for variant in VARIANTS:
                graph, workflow, audit = build_candidate(coarse, refine, variant, info)
                validation = asyncio.run(execution.validate_prompt("t8-native-dual-candidate", deepcopy(graph), None))
                name = f"Native_Dual_LOW{coarse}_HIGH{refine}_{variant}_EXP"
                shared.write_new(destination / f"{name}.api.json", graph)
                shared.write_new(destination / f"{name}.json", workflow)
                records.append({"name": name, "serialization": audit, "core_validation": validation})
    report = {"candidates": records, "qualification": "24 serialized frontend/API candidates with live CPU Core "
              "validation only; no browser roundtrip, models loaded, GPU sampling or human review."}
    shared.write_new(destination / "audit.json", report)
    shared.write_new(destination / "object-info.json", info)
    failures = [record["name"] for record in records if not record["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failures, "output": str(destination)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
