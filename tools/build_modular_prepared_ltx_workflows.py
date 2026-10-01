"""Build private full and decode-only Prepared LTX split candidates."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import uuid

from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/development/modular-sampling-m4-prepared-ltx-20260923/candidate-v1"
NODES = ("MiniMaxH3PreparedGenerationBundleEXPT8",
         "MiniMaxH3PreparedLTXGenerateEXPT8",
         "MiniMaxH3PreparedLTXLoadGenerationEXPT8",
         "MiniMaxH3PreparedLTXDecodeEXPT8")


def build_prompt(*, resume=False, bundle_path="", lease_path="", expected_sha256=""):
    graph = {"1": {"class_type": NODES[0], "inputs": {"prepared_bundle_path": bundle_path}}}
    if resume:
        graph["2"] = {"class_type": NODES[2], "inputs": {
            "prepared_bundle": ["1", 0], "noise_seed": 8301,
            "chain_id": "ltx_split_trial_01", "expected_sha256": expected_sha256}}
    else:
        graph["2"] = {"class_type": NODES[1], "inputs": {
            "prepared_bundle": ["1", 0], "noise_seed": 8301,
            "chain_id": "ltx_split_trial_01", "resume_existing": True,
            "serial_lease_path": lease_path}}
    graph["3"] = {"class_type": NODES[3], "inputs": {
        "prepared_bundle": ["1", 0], "generation_receipt": ["2", 0],
        "serial_lease_path": lease_path}}
    return graph


def build_workflow(info, *, resume=False, **paths):
    graph = build_prompt(resume=resume, **paths)
    title = "H3 Prepared LTX / Decode Frozen Latent / EXP" if resume else "H3 Prepared LTX / Separate Generation and Decode / EXP"
    workflow = convert(graph, info, title)
    normalize_native_widget_inputs(workflow)
    for index, node in enumerate(workflow["nodes"]):
        node["pos"] = [index * 640, 0]
        node["size"] = [590, 340]
    note = (
        "## Prepared LTX 两阶段实验图\n\n"
        "只接受匹配的 normalized LTX AV prepared bundle；它不是普通 H3 latent 的自动转换器。"
        "生成节点只运行固定 LTX 三次更新并保存 refined-latents.safetensors；解码节点只从完成回执"
        "加载 latent，复用原解码 worker 保留输入原声，不隐式重新生成。\n\n"
        "请先填写本机 prepared_bundle_path 与双方共用的 serial_lease_path。"
        "新提示词/模型/输入要重新准备清单；改变 seed 或内容使用新 chain_id。"
        "另一个旧一体节点、旧图及其缓存目录不受此候选影响。"
        + ("本图不含生成节点：从先前生成报告复制准确 SHA256 到 expected_sha256，"
           "仅解码/恢复后段；空占位不能运行。" if resume else
           "本图可先运行生成节点，再运行解码；生成输出路径与 SHA 可供独立恢复图使用。")
        + "\n\n仅候选图和 CPU 合同；未证明新分离 worker 的预训练 GPU 音画、人审或任意输入画质。"
    )
    nid = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": nid, "type": "MarkdownNote", "title": "必读：独立阶段与恢复边界",
        "pos": [1920, 0], "size": [850, 660], "flags": {}, "order": 3, "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = nid
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL,
        "t8:modular-prepared-ltx:20260923:" + ("resume" if resume else "full")))
    workflow["extra"]["prepared_route"] = "ltx_refine"
    workflow["extra"]["split_stage"] = "decode_only" if resume else "generation_then_decode"
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    output = options.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Choose a new private artifacts output directory")
    package_name = "h3_audio_t8_pkg"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    from h3_audio_t8_pkg.nodes_prepared_generation import MiniMaxH3PreparedGenerationBundleEXPT8
    from h3_audio_t8_pkg.modular_sampling.prepared_ltx_nodes import NODES as split_nodes
    classes = [MiniMaxH3PreparedGenerationBundleEXPT8, *split_nodes]
    info = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}
    for value in info.values():
        value["cnr_id"] = "minimax-h3-audio-T8"
    output.mkdir(parents=True)
    for resume in (False, True):
        stem = "Prepared_LTX_DecodeOnly_EXP" if resume else "Prepared_LTX_Generate_Decode_EXP"
        frontend = build_workflow(info, resume=resume)
        api = build_prompt(resume=resume)
        (output / (stem + ".json")).write_text(json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        (output / (stem + ".api.json")).write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(str(output))


if __name__ == "__main__":
    raise SystemExit(main())
