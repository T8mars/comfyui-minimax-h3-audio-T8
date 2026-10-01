"""Build a separate source-gated FastH3 V2 accepted-chain compose graph."""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

from tools import build_modular_fast_h3_v2_workflow as base
from tools.audit_modular_sampling_compat import write_new


ROOT = base.ROOT


def compose_graph(chain_id="REPLACE_WITH_ACCEPTED_V2_CHAIN"):
    return {
        "1": {"class_type": "MiniMaxH3FastH3V2CurrentAcceptedChainVerifyEXPT8",
              "inputs": {"chain_id": chain_id}},
        "2": {"class_type": "MiniMaxH3LongVideoComposeAcceptedT8", "inputs": {
            "chain_id": ["1", 0], "filename_prefix": "FastH3V2_SourceBound_8s",
            "require_final_segment": True, "audio_seam_policy": "cosine_bridge",
            "bridge_ms": 5.0, "crf": 18}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["1", 1]}},
    }


def build_candidate(graph, info):
    graph, selected = base.selected_frontend_schema(graph, info)
    workflow = base.convert(graph, selected,
        "FastH3 V2 · verify two accepted segments · compose separately")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote",
        "title": "两段已接受后再单独拼接", "pos": [-900, -560], "size": [740, 460],
        "flags": {}, "order": len(graph), "mode": 0, "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [
            "填入同一条已接受 FastH3 V2 链的 chain_id。只读来源门会逐个核对两段保存候选的 "
            "source-provenance.json、原候选和已接受媒体/上下文 SHA，以及124+68帧、同一作业SHA。"
            "通过后才调用原长视频 Compose 节点；默认5ms cosine_bridge，不改变原拼接算法。"
            "本图不采样、不保存或接受候选。画音质量仍须人工从接缝前看到片尾；"
            "来源侧记是本地完整性追踪，不是防恶意改写的签名。"]})
    workflow["last_node_id"] = note_id
    return graph, workflow, base.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = base.load_live_info()
    import execution
    graph, workflow, audit = build_candidate(compose_graph(), info)
    validation = asyncio.run(execution.validate_prompt(
        "fast-v2-source-bound-compose", deepcopy(graph), None))
    name = "FastH3V2_SourceBound_Accepted_124plus68_Compose"
    write_new(destination / (name + ".api.json"), graph)
    write_new(destination / (name + ".json"), workflow)
    write_new(destination / "audit.json", {"name": name, "serialization": audit,
        "core_validation": validation,
        "all_outputs_valid": bool(validation[0] and not validation[3]),
        "qualification": "Static Core/frontend only; accepted chain is a placeholder. "
                         "No composition executed, GPU or human quality pass."})
    if not validation[0] or validation[3]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
