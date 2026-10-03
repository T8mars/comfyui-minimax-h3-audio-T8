"""Three opt-in, genuinely separated curve-approximation Full/Cold graphs."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path

from tools import build_modular_hyperflow_effect_workflow as base
from tools.audit_modular_sampling_compat import write_new
from tools.modular_frontend_layout import spread_frontend_columns

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/workflows/68-radar-hyperflow-curves"
BASE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VARIANTS = ("full_save", "cold_tail", "cold_delivery")
FILES = {name: "HyperFlow_Curve_4plus4_" + name + "_EXP.json" for name in VARIANTS}


def graph_for(variant, *, fit_file="missing_curve_fit", fit_path="", fit_sha256="",
              artifact_path=None, artifact_sha256="0" * 64):
    if variant not in VARIANTS:
        raise ValueError("Choose explicit curve FullSave, ColdTAIL or ColdDelivery")
    graph = base.split_graph(4, "combined", "both", "save" if variant == "full_save" else "resume_tail",
        artifact_path=artifact_path or "SELECT_SAVED_HEAD/curve-head.safetensors", artifact_sha256=artifact_sha256)
    # Reuse only graph structure; every full-time producer/consumer is replaced
    # by the dedicated typed curve node. No old split executor is invoked.
    for node in graph.values():
        kind = node["class_type"]
        if kind.startswith("MiniMaxH3HyperFlow") and kind.endswith("EXPT8"):
            node["class_type"] = kind.replace("MiniMaxH3HyperFlow", "MiniMaxH3HyperFlowCurve", 1)
    graph["1"] = {"class_type": "MiniMaxH3HyperFlowCurveBaseLoaderEXPT8", "inputs": {"base_file": BASE}}
    graph["103"] = {"class_type": "MiniMaxH3HyperFlowCurveFitLoadEXPT8", "inputs": {
        "fit_file": fit_file, "absolute_path": fit_path, "expected_sha256": fit_sha256}}
    for key in ("101", "102"):
        if key in graph:
            graph[key]["class_type"] = "MiniMaxH3HyperFlowCurveModelApplyEXPT8"
            graph[key]["inputs"].update(curve_fit=["103", 0], base_file=BASE)
    for key in ("9", "24"):
        if key in graph:
            graph[key]["inputs"].update(width=512, height=512)
    for key in ("205", "215"):
        if key in graph:
            graph[key]["inputs"].update(global_prompt="One adult walks through a quiet corridor in an uninterrupted cinematic shot. Natural motion and quiet ambience.",
                local_prompts="The adult walks steadily toward a doorway.\nThe adult approaches the doorway without changing identity or location.")
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/HyperFlow_Curves/4plus4_" + variant
    for key, prefix in (("50", "curve-head"), ("51", "curve-tail")):
        if key in graph and "prefix" in graph[key]["inputs"]:
            graph[key]["inputs"]["prefix"] = "HyperFlowCurves/" + prefix
    if variant == "cold_delivery":
        graph["51"] = {"class_type": "MiniMaxH3HyperFlowCurveTailLoadEXPT8", "inputs": {
            "artifact_path": artifact_path or "SELECT_SAVED_TAIL/curve-tail.safetensors", "artifact_sha256": artifact_sha256}}
        graph["30"]["inputs"]["source"] = ["51", 2]
        return base.base.common.prune(graph, ["16", "30"])
    return base.base.common.prune(graph, ["16", "30", "31"])


def build_candidate(variant, info, **kwargs):
    graph, selected = base.shared.selected_frontend_schema(graph_for(variant, **kwargs), info)
    workflow = base.shared.convert(graph, selected, "HyperFlow Curve 4+4 / " + variant + " EXP")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "曲线近似 / Explicit fit, separate stages",
        "pos": [0, -800], "size": [1100, 720], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "R10/S30 原生pruned H3 HyperFlow曲线近似，非完整时间教师等价。原完整HyperFlow/旧图保持。"
            "\n先独立Build Fit：所选pruned底模＋匹配完整教师＋原632tensor HyperFlow；然后把新asset放models/hyperflow/curve_fits，"
            "或在Fit Load填明确绝对路径。此图不自动拟合/下载/替换模型。Fit SHA可固定精确文件。专用Base Loader首次加载保留Core声明FP32层，"
            "不修补已加载MODEL/不改单独UNETLoader。先从该底模分支，再在两路Apply前各插不同内容LoRA。"
            "\nHEAD只0:4、TAIL只4:8；直接传原model-space x_sigma与独立scaffold，不抽二次噪声/放大/重置音频。"
            "两路条件/Prompt Relay Plan与配对MODEL、EAV Config各自外置。CFG1；EAV默认report_only，apply_exp才改数学，Relay默认apply_exp。"
            "\nFullSave保存两份实际阶段path/SHA；ColdTAIL删除HEAD模型/噪声/效果/输出，只载实际HEAD再续4步；"
            "ColdDelivery只读完成AV与两VAE解码，无MODEL/CLIP/fit/采样。Load占位path/SHA不可直接执行，坏文件不隐藏重采。"
            "HEAD/TAIL必须同actual base/adapter/fit/producer implementation；未知用户owner保留执行但不认证持久恢复。"
            "\n拟合残差和CPU逐值通过不是画质/完整权重GPU或人审。当前资格见专题文档，confirm/质量审核由用户决定。"]})
    workflow["last_node_id"] = note_id
    workflow.setdefault("extra", {})["t8_split_example"] = {"schema": "t8.hyperflow.curve.split-example.v1",
        "route": "S30", "variant": variant, "split_interval": 4, "approximation": True,
        "status": "opt_in_importable_quality_unreviewed"}
    spread_frontend_columns(workflow)
    return graph, workflow, base.shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    options = parser.parse_args()
    destination = options.output_dir.resolve()
    if not destination.is_relative_to(ROOT) or any((destination / name).exists() for name in FILES.values()):
        parser.error("Use new target files within the project; no overwrite")
    info = base.base.load_live_info()
    import execution
    rows = []
    for variant in VARIANTS:
        fit_file = info["MiniMaxH3HyperFlowCurveFitLoadEXPT8"]["input"]["required"]["fit_file"][1]["options"][0]
        graph, workflow, audit = build_candidate(variant, info, fit_file=fit_file)
        validation = asyncio.run(execution.validate_prompt("hyperflow-curves-" + variant, deepcopy(graph), None))
        expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
        if not validation[0] or validation[3] or set(validation[2]) != expected:
            raise RuntimeError("Curve graph validation failed: " + str(validation))
        write_new(destination / FILES[variant], workflow)
        rows.append({"variant": variant, "serialization": audit, "actual_Core_typed_validation": True})
    print(json.dumps({"destination": str(destination), "candidates": rows}))


if __name__ == "__main__":
    main()
