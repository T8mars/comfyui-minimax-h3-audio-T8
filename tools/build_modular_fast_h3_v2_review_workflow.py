"""Build a standalone FastH3 V2 saved-candidate review/accept workflow.

Paste a candidate.json produced by the source-bound Save node. The workflow
starts with accept_candidate=false; it never runs sampling or composition.
"""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

from tools import build_modular_fast_h3_v2_workflow as base
from tools.audit_modular_sampling_compat import write_new


ROOT = base.ROOT


def review_graph(candidate_json_path="REPLACE_WITH_SAVED_V2_CANDIDATE_JSON"):
    return {
        "1": {"class_type": "MiniMaxH3FastH3V2CurrentCandidateReviewAcceptEXPT8",
              "inputs": {"candidate_json_path": candidate_json_path,
                         "accept_candidate": False}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 7]}},
    }


def build_candidate(graph, info):
    graph, selected = base.selected_frontend_schema(graph, info)
    workflow = base.convert(graph, selected,
        "FastH3 V2 · saved split candidate · separate review and explicit accept")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote",
        "title": "只读预览 → 人工确认 → 显式接受", "pos": [-850, -500], "size": [720, 440],
        "flags": {}, "order": len(graph), "mode": 0, "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [
            "仅接受来源绑定的 FastH3 V2 候选。先从首段或续段 Candidate Save 节点复制 "
            "candidate.json 绝对路径并保持 accept_candidate=false，检查完整画音及来源报告。"
            "确认后显式改为 true 才会调用原长视频接受事务；默认拒绝覆盖已有接受段。"
            "输出 chain_id / parent_candidate_id / parent_revision / current_job_sha256 "
            "供下一段 Accepted Source 填写。本图不重新采样、不自动接受、不拼接。"
            "本地来源侧记只提供完整性/追踪，不是防恶意改写的数字签名。"]})
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
    graph, workflow, audit = build_candidate(review_graph(), info)
    validation = asyncio.run(execution.validate_prompt(
        "fast-v2-saved-candidate-review", deepcopy(graph), None))
    name = "FastH3V2_Saved_Candidate_Review_ExplicitAccept"
    write_new(destination / (name + ".api.json"), graph)
    write_new(destination / (name + ".json"), workflow)
    write_new(destination / "audit.json", {"name": name, "serialization": audit,
        "core_validation": validation,
        "all_outputs_valid": bool(validation[0] and not validation[3]),
        "qualification": "Static Core/frontend only; saved candidate path is a placeholder. "
                         "No acceptance executed, GPU, composition or human review."})
    if not validation[0] or validation[3]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
