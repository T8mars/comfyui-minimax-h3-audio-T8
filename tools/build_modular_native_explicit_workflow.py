"""Native base-flow/LBH/full-first graphs using existing plans plus stage bind."""
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
RECIPES = (("base_flow", 4, 4), *(('lbh', 4, n) for n in (3, 4, 5)),
           *(('complete', n, r) for n in (8, 20) for r in (3, 4, 5)))


def split_graph(recipe=RECIPES[0], variant="minimal", *, artifact_path="REPLACE_WITH_NATIVE_LOW_PATH/manifest.json",
                artifact_sha256="0" * 64):
    if recipe not in RECIPES or variant not in dual.VARIANTS:
        raise ValueError("Unknown explicit native candidate recipe/variant")
    name, coarse, refine = recipe
    # In a HIGH-only graph the native dual builder already removes the LOW
    # branch and replaces the stage-ordered HIGH loader with a direct loader.
    # Starting from save_effects retains a completed_stage dependency and can
    # silently pull the LOW sampler back into the cold-resume graph.
    graph = dual.split_graph(20 if coarse == 20 else 4, refine, variant)
    for key, setup, plan, stage in (("10", "90", "91", "native_low"), ("26", "92", "93", "native_high")):
        if key not in graph:
            continue
        source = graph[key]["inputs"]
        graph[setup] = {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": source["model"], "av_latent": source["av_latent"],
            "steps": coarse if stage == "native_low" and name == "complete" else 8,
            "shift_video": 12., "shift_audio": 3., "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}}
        if name == "base_flow":
            graph[plan] = {"class_type": "MiniMaxH3TwoPassSigmaPlanT8Advanced", "inputs": {
                "model": [setup, 0], "coarse_steps": coarse, "refine_steps": refine, "restart_base_noise": .5}}
        else:
            graph[plan] = {"class_type": "MiniMaxH3LearnedTwoPassParityPlanT8Advanced", "inputs": {
                "model": [setup, 0], "base_steps": 8, "coarse_steps": 4, "refine_steps": refine}}
        sigmas = [plan, 0 if stage == "native_low" else 1]
        if name == "complete" and stage == "native_low":
            sigmas = [setup, 2]
            graph.pop(plan)  # No misleading unused coarse LBH plan for a full first pass.
        graph[key] = {"class_type": "MiniMaxH3NativeStageBindEXPT8", "inputs": {
            "model": [setup, 0], "sampler": [setup, 1], "sigmas": sigmas,
            "av_latent": source["av_latent"], "stage": stage}}
    handoff = graph["25"]["inputs"]
    graph["25"] = {"class_type": "MiniMaxH3TwoPassLatentReconcileT8Advanced", "inputs": {
        "learned_latent": handoff["learned_latent"], "highres_template": handoff["highres_template"],
        "positive": handoff["positive"], "audio_policy": "auto",
        "second_pass_audio_source": "first_pass" if name == "complete" else "legacy_policy",
        "second_pass_audio_strength": 0.}}
    if name == "complete":
        graph["94"] = {"class_type": "MiniMaxH3TwoPassAudioAuditT8Advanced", "inputs": {
            "second_pass_input": ["25", 0], "second_pass_output": graph["14"]["inputs"]["av_latent"],
            "expected_audio_strength": 0., "fail_on_locked_mismatch": True, "locked_atol": 1e-5}}
        graph["14"]["inputs"]["av_latent"] = ["94", 0]
    for key, label in (("50", "LOW"), ("51", "HIGH")):
        if key in graph:
            graph[key]["inputs"]["prefix"] = f"NativeExplicit/{name}{coarse}plus{refine}/{label}"
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_Native_{name}_{coarse}plus{refine}_EXP"
    if variant == "resume_effects":
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "native_low"}}
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


def build_candidate(recipe, variant, info, *, graph_override=None):
    graph = split_graph(recipe, variant) if graph_override is None else deepcopy(graph_override)
    graph, workflow, _ = shared.build_candidate(graph, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title, position in (
        ("10", "LOW bind — original native sampler/table unchanged", (800, 0)),
        ("26", "HIGH bind — independently selected native model/table", (1560, 1300)),
        ("90", "LOW original Dual-Clock setup", (0, 2600)),
        ("91", "LOW original schedule plan", (420, 2600)),
        ("92", "HIGH original Dual-Clock setup", (840, 2600)),
        ("93", "HIGH original schedule plan — no LOW dependency", (1260, 2600)),
        ("94", "Completed first-pass audio — verify/relock before decode", (3000, 2200)),
        ("70", "LOW acceleration LoRA — independent content stack", (-400, 0)),
        ("71", "HIGH acceleration LoRA — independent content stack", (-400, 1300)),
        ("72", "Original reconcile/audio report", (1200, 2300))):
        if key in ids:
            by_id[ids[key]].update(title=title, pos=list(position))
    name, coarse, refine = recipe
    note = (f"原生{name} {coarse}+{refine}独立阶段 / EXP候选。所有调度与采样器来自原节点，Bind不改数学。\n\n"
        "base_flow使用各MODEL实际shift投影同一base q；LBH使用原simple8前4及独立发布HIGH3/4/5表；"
        "complete使用完整native_flow8或20，再learned放大和独立LBH后段，不能写成总8步。"
        "LOW denoised_output进入原learned3D，HIGH条件尺寸接实际放大器输出；原Reconcile在外部。"
        "complete默认保留完成音频并在解码前校验重锁；其它两类保留legacy_policy联合音频。\n\n"
        "独立MODEL/LoRA/条件/NOISE，Relay成对绑定，EAV默认report_only。"
        "Bind描述实际连接的单阶段，不凭标签证明上游来自某个配方。PDD/VDN/V2有各自适配，不借本节点认证。"
        "恢复图只读冻结LOW path/SHA及denoised，只跑HIGH；旧图不动，未知补丁保留但可能无法认证持久复用。"
        "本批仅tiny CPU/Core与序列化；未做浏览器编辑重载/完整learned GPU/人审，也不是长片body或自动缓存。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"Native {name} {coarse}+{refine} — {variant} EXP"
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
    for recipe in RECIPES:
        for variant in dual.VARIANTS:
            graph, workflow, audit = build_candidate(recipe, variant, info)
            validation = asyncio.run(execution.validate_prompt("native-explicit-candidate", deepcopy(graph), None))
            name = f"Native_{recipe[0]}_{recipe[1]}plus{recipe[2]}_{variant}_EXP"
            shared.write_new(destination / f"{name}.api.json", graph)
            shared.write_new(destination / f"{name}.json", workflow)
            records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "CPU live Core/serialization, not browser edit/reload, GPU, learned weights or human review."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
