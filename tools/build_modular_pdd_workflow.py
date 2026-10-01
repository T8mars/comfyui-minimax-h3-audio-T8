"""Separate PDD 4+4 candidates, original heads/learned/reconcile; no queue."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_native_dual_workflow as dual  # noqa: E402

shared = dual.shared
ROOT = Path(__file__).resolve().parents[1]
BASES = ("FL2VA", "Ref2VA")


def split_graph(base="FL2VA", variant="minimal", *, artifact_path="REPLACE_WITH_PDD_LOW_PATH/manifest.json",
                artifact_sha256="0" * 64):
    if base not in BASES or variant not in dual.VARIANTS:
        raise ValueError("Unknown PDD base/graph variant")
    graph = dual.split_graph(4, 4, "save_effects" if variant == "resume_effects" else variant)
    effects = variant != "minimal"
    for key in ("70", "71"):
        graph.pop(key)
    for key in ("1", "22"):
        graph[key]["inputs"]["unet_name"] = f"minimax_h3_{base.lower()}_int8_convrot.safetensors"
    # Explicit original media inputs, not generated test images. User replaces
    # these example selections; the builder never reads image pixels or queues.
    graph["95"] = {"class_type": "LoadImage", "inputs": {"image": "10A.jpg"}}
    if base == "FL2VA":
        graph["96"] = deepcopy(graph["95"])
        for key, image in (("97", "95"), ("98", "96")):
            graph[key] = {"class_type": "ImageScale", "inputs": {"image": [image, 0],
                "upscale_method": "bicubic", "width": 1024, "height": 576, "crop": "center"}}
    for key, base_model in (("9", "1"), ("24", "22")):
        inputs = graph[key]["inputs"]
        inputs["task_type"] = base
        if base == "FL2VA":
            inputs.update(first_frame=["97", 0], last_frame=["98", 0])
        else:
            inputs["ref_images.ref_image_0"] = ["95", 0]
        if effects:
            inputs["model"] = [base_model, 0]
        else:
            inputs["length"] = 124 if base == "FL2VA" else 22
        if key == "9":
            inputs.update(width=512 if base == "FL2VA" else 864, height=288 if base == "FL2VA" else 480)
    if effects:
        frame_count = 124 if base == "FL2VA" else 22
        graph["40"]["inputs"]["length"] = frame_count
        graph["47"]["inputs"]["length"] = frame_count
    graph["23"]["inputs"]["scale_by"] = 2. if base == "FL2VA" else 1.5
    for key, loader, base_model, condition, stage in (("10", "90", "1", "9", "pdd_low_0_4"),
                                                    ("26", "92", "22", "24", "pdd_high_4_8")):
        source = graph[key]["inputs"]["av_latent"]
        graph[loader] = {"class_type": "MiniMaxH3PDD8StepSetupT8Advanced", "inputs": {
            "model": [condition if effects else base_model, 0], "av_latent": source,
            "pdd_lora_name": f"MiniMax-H3-{base}-Acc-8Step_comfyui_pdd.safetensors", "base_variant": base,
            "strength": 1.}}
        graph[key] = {"class_type": "MiniMaxH3PDDStageSetupEXPT8", "inputs": {
            "model": [loader, 0], "av_latent": source, "full_sigmas": [loader, 2], "stage": stage}}
    handoff = graph["25"]["inputs"]
    graph["25"] = {"class_type": "MiniMaxH3TwoPassLatentReconcileT8Advanced", "inputs": {
        "learned_latent": handoff["learned_latent"], "highres_template": handoff["highres_template"],
        "positive": handoff["positive"], "audio_policy": "auto", "second_pass_audio_source": "legacy_policy",
        "second_pass_audio_strength": 0.}}
    for key, stage in (("50", "LOW"), ("51", "HIGH")):
        if key in graph:
            graph[key]["inputs"]["prefix"] = f"PDD/{base}/{stage}"
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_PDD_{base}_4plus4_EXP"
    if variant == "resume_effects":
        # The complete graph defers HIGH loading until LOW has finished. A
        # cold HIGH-only graph must load directly from the frozen LOW result;
        # otherwise completed_stage pulls the entire LOW branch back in.
        loader = graph["22"]
        if loader["class_type"] != "MiniMaxH3StageUNETLoaderAfterEXPT8":
            raise ValueError("Expected the serialized PDD HIGH loader")
        graph["22"] = {"class_type": "UNETLoader", "inputs": {
            name: value for name, value in loader["inputs"].items()
            if name != "completed_stage"}}
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "pdd_low_0_4"}}
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
        graph = {key: value for key, value in graph.items() if key in keep}
    return graph


def build_candidate(base, variant, info, *, graph_override=None):
    graph, workflow, _ = shared.build_candidate(
        split_graph(base, variant) if graph_override is None else graph_override, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title, position in (
        ("10", "PDD LOW — absolute heads 0–3", (800, 0)),
        ("26", "PDD HIGH — absolute heads 4–7, actual HIGH geometry", (1560, 1300)),
        ("90", "Independent LOW PDD loader — editable LoRA chain after MODEL", (0, 2600)),
        ("92", "Independent HIGH PDD loader — no LOW MODEL dependency", (600, 2600)),
        ("95", "Choose reference / FIRST image", (-800, 0)),
        ("96", "Choose LAST image", (-800, 450)),
        ("97", "Shared FIRST crop before both Conditioning branches", (-400, 0)),
        ("98", "Shared LAST crop before both Conditioning branches", (-400, 450))):
        if key in ids:
            by_id[ids[key]].update(title=title, pos=list(position))
    note = (f"PDD {base}真正分离4+4 / EXP候选。已有PDD Loader保留原32头和8步表；新Setup只取绝对0:4或4:8，"
        "不执行采样，不是8+8。两路独立MODEL/PDD/LoRA、条件、NOISE。\n\n"
        "LOW denoised_output→原learned3D→实际HIGH尺寸条件→原Reconcile legacy_policy→HIGH fresh noise。"
        "原生未完成联合音频继续采样，不插完整首采音频冻结审计。FL2VA首尾图在分支前共用一次目标比例crop；"
        "Ref2VA两阶段使用同一参考输入，可显式编辑。请替换示例10A.jpg和提示词。\n\n"
        "EAV外置且默认report_only，两阶段可分别开关；Relay Plan可复制为独立事件，在实际尺寸各自绑定MODEL/条件。"
        "保存/恢复已接原生head-bank与经源码核对的旧动态注入路径；strength<1保留原插值数学。"
        "未知第三方补丁保留执行，可能不具备持久复用资格。恢复图需填写真实LOW path+SHA，只有HIGH采样器。\n\n"
        "本批为tiny CPU/Core与序列化候选，不是完整预训练权重/GPU/浏览器保存重载/人审验收。旧图不动。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"PDD {base} absolute 4+4 — {variant} EXP"
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
    import nodes as core_nodes
    for name in ("LoadImage", "ImageScale"):
        info[name] = shared.native_info(name, core_nodes.NODE_CLASS_MAPPINGS[name])
    import execution
    records = []
    for base in BASES:
        for variant in dual.VARIANTS:
            graph, workflow, audit = build_candidate(base, variant, info)
            validation = asyncio.run(execution.validate_prompt("pdd-stage-candidate", deepcopy(graph), None))
            name = f"PDD_{base}_4plus4_{variant}_EXP"
            shared.write_new(destination / f"{name}.api.json", graph)
            shared.write_new(destination / f"{name}.json", workflow)
            records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "CPU Core/serialization only, not browser roundtrip, GPU, pretrained learned or human."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
