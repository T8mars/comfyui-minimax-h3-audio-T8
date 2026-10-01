"""Manual second pass graphs with visible noise and no learned-upscale handoff."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_native_dual_workflow as native  # noqa: E402

shared = native.shared
ROOT = Path(__file__).resolve().parents[1]


def split_graph(variant="minimal", *, free_noise=False, artifact_path="REPLACE_WITH_SAVED_FIRST_PATH/manifest.json",
                artifact_sha256="0" * 64):
    if variant not in native.VARIANTS:
        raise ValueError("Unknown manual-pass workflow variant")
    # Start with a graph, not an executor. Replace all native Dual-specific
    # numerical nodes/edges; no Dual or V2 owner survives in the manual recipe.
    graph = native.split_graph(20, 3, "save_effects" if variant == "resume_effects" else variant)
    with_effects = variant != "minimal"
    for key in ("23", "25", "62", "71", "72"):
        graph.pop(key, None)
    for key, stage in (("10", "manual_first"), ("26", "manual_second")):
        graph[key]["class_type"] = "MiniMaxH3ManualPassStageSetupEXPT8"
        graph[key]["inputs"].update(stage=stage, first_steps=20, manual_sigmas="0.5,0.412,0.35,0",
                                    sampler_name="dual_clock_euler", scheduler="native_flow")
    if with_effects:
        graph["24"]["inputs"]["model"] = ["22", 0]
        graph["28"]["inputs"]["conditioning"] = ["24", 1]
        graph["45"]["inputs"]["av_latent"] = ["50", 0] if "50" in graph else ["13", 0]
        first_output = ["45", 0]
    else:
        graph["26"]["inputs"]["model"] = ["22", 0]
        graph["28"]["inputs"]["conditioning"] = ["24", 0]
        first_output = ["13", 0]
    graph["26"]["inputs"]["av_latent"] = first_output
    graph["29"]["inputs"]["latent_image"] = first_output
    if with_effects:
        graph["44"]["inputs"]["av_latent"] = first_output
    # Both passes use a single explicit geometry, without a dummy upscaler.
    for key, field, value in (("80", "width", 448), ("81", "height", 256), ("82", "length", 73)):
        graph[key] = {"class_type": "PrimitiveInt", "inputs": {"value": value}}
        if field == "length" and with_effects:
            graph["40"]["inputs"][field] = [key, 0]
        else:
            graph["9"]["inputs"][field] = [key, 0]
            graph["24"]["inputs"][field] = [key, 0]
    if with_effects:
        graph["47"]["inputs"]["length"] = ["82", 0]
    graph["27"]["inputs"]["noise_seed"] = graph["11"]["inputs"]["noise_seed"]
    for key, source, setup, guider in (("83", "11", "10", "13"), ("84", "27", "26", "29")):
        graph[key] = {"class_type": "MiniMaxH3StageNoiseEXPT8", "inputs": {
            "noise": [source, 0], "mode": "variance_preserving_blend" if free_noise else "disabled",
            "base_seed": 123456789, "reuse_ratio": .65, "segment_index": 0, "model": [setup, 0]}}
        graph[guider]["inputs"]["noise"] = [key, 0]
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/Modular_Manual_Second_Pass_EXP"
    if "50" in graph:
        graph["50"]["inputs"]["prefix"] = "ManualPass/FIRST"
        graph["51"]["inputs"]["prefix"] = "ManualPass/SECOND"
    if variant == "resume_effects":
        # The full graph delays the second UNET load until FIRST has completed.
        # A cold SECOND-only graph instead loads it directly from the frozen
        # artifact; retaining completed_stage would silently pull FIRST back in.
        loader = graph["22"]
        if loader["class_type"] != "MiniMaxH3StageUNETLoaderAfterEXPT8":
            raise ValueError("Expected the serialized second-stage UNET loader")
        graph["22"] = {"class_type": "UNETLoader", "inputs": {
            name: value for name, value in loader["inputs"].items()
            if name != "completed_stage"}}
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "manual_first"}}
        graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
        for key, field in (("26", "av_latent"), ("29", "latent_image"), ("44", "av_latent")):
            graph[key]["inputs"][field] = ["60", 0]
        keep = set()

        def include(key):
            if key in keep:
                return
            keep.add(key)
            for value in graph[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    include(str(value[0]))

        for key in ("16", "32", "51", "61"):
            include(key)
        graph = {key: item for key, item in graph.items() if key in keep}
    return graph


def load_info():
    info = shared.load_live_info()
    import nodes
    from comfy_extras.nodes_primitive import Int
    nodes.NODE_CLASS_MAPPINGS["PrimitiveInt"] = Int
    info["PrimitiveInt"] = json.loads(json.dumps(shared.native_info("PrimitiveInt", Int)))
    return info


def build_candidate(variant, free_noise, info, *, graph_override=None):
    source = graph_override if graph_override is not None else split_graph(variant, free_noise=free_noise)
    graph, workflow, _ = shared.build_candidate(source, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    titles = {"10": "FIRST complete trajectory — original native schedule",
              "26": "SECOND independent manual tail — fresh noise",
              "22": "SECOND MODEL — load after FIRST in full graph, directly in resume",
              "40": "FIRST external Prompt Relay Plan",
              "47": "SECOND external Prompt Relay Plan — edit without changing FIRST",
              "24": "SECOND conditioning — shared source geometry, independently editable prompt",
              "13": "FIRST — use OUTPUT, not denoised_output, for manual handoff",
              "29": "SECOND — final joint AV", "45": "FIRST effect audit — OUTPUT handoff",
              "80": "Shared width — match the frozen FIRST when restoring",
              "81": "Shared height — match the frozen FIRST when restoring",
              "82": "Shared frame count — match the frozen FIRST when restoring",
              "83": "FIRST external noise — explicit FreeNoise / legacy plan",
              "84": "SECOND external noise — same segment_index for legacy parity"}
    positions = {"40": (0, 650), "47": (800, 1500),
                 "80": (-400, 0), "81": (-400, 200), "82": (-400, 400),
                 "83": (1160, 700), "84": (1920, 2100), "24": (800, 1300)}
    for key, title in titles.items():
        if key in ids:
            by_id[ids[key]]["title"] = title
            if key in positions:
                by_id[ids[key]]["pos"] = list(positions[key])
    note = ("手动二采分离 / EXP：完整FIRST output → 独立SECOND手工尾段；不是learned放大或V2 DMD。\n\n"
            "两路MODEL／条件／noise均可编辑。默认同seed和同segment_index复现旧runner；二采先独立准备噪声。"
            "第一采交接使用output口；不要接denoised_output。无上采样，宽高帧数三个原语同时驱动两路条件。\n\n"
            "默认dual_clock_euler；也验证了原生euler。其他Core采样器可选择，但多次模型求值的完成／效果审计"
            "尚未认证，不会冒称Euler步数或签发可恢复资格。FreeNoise是显式NOISE节点，disabled精确旁路；"
            "from_model_plan需要连接带旧FreeNoise计划的MODEL。配置报告不是已经生成噪声的证明。")
    if variant != "minimal":
        note += ("\n\nFIRST/SECOND各有独立Relay Plan，分别绑定自己的MODEL／条件；"
                 "EAV各自配置，默认report_only。"
                 "第二采可独立开EAV，这是新增外置能力，不冒称旧effects runner默认会二采EAV。")
    if variant in ("save_effects", "resume_effects"):
        note += ("\n\n保存FIRST的路径与SHA；读取的是冻结首采，不自动比较今天的FIRST参数。"
                 "恢复图已删除全部FIRST计算／噪声／效果／输出节点；必须填真实path、SHA，并让共享宽高帧数"
                 "匹配该冻结结果。只执行SECOND，原输出文件不覆盖。")
    note += ("\n\n此候选只通过CPU tiny与真实Core图校验；没有完整GPU、浏览器保存重载或人审资格。"
             "旧一体节点、正式图与导演台均未替换。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"Manual Second Pass — {variant} — FreeNoise {free_noise} EXP"
    _, selected_info = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected_info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = load_info()
    import execution
    records = []
    for variant in native.VARIANTS:
        for free_noise in (False, True):
            graph, workflow, audit = build_candidate(variant, free_noise, info)
            validation = asyncio.run(execution.validate_prompt("t8-manual-pass-candidate", deepcopy(graph), None))
            name = f"Manual_Pass_{variant}_{'FreeNoise' if free_noise else 'NativeNoise'}_EXP"
            shared.write_new(destination / f"{name}.api.json", graph)
            shared.write_new(destination / f"{name}.json", workflow)
            records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "Live Core CPU / serialization only; not browser, full GPU, long-video body or human review."})
    shared.write_new(destination / "object-info.json", info)
    failed = [record["name"] for record in records if not record["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
