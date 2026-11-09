"""Wire the existing immutable Stage store into compiled Director graphs."""

MODES = ("disabled", "reuse_completed", "refresh")


def apply_stage_reuse(graph, mode="disabled"):
    if mode not in MODES:
        raise ValueError("无效的阶段复用模式")
    rows = []
    if mode != "disabled":
        for node_id, node in graph.items():
            kind = node["class_type"]
            if kind not in {"SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8"}:
                continue
            node["class_type"] = "MiniMaxH3StageReuseEXPT8"
            node["inputs"]["mode"] = mode
            if "stage_context" not in node["inputs"]:
                node["inputs"]["stage_role"] = "native_low" if str(node_id) == "9" else "native_high"
            rows.append(str(node_id))
    return {"mode": mode, "sampler_nodes": rows,
            "warning": ("本配方使用专用阶段接口，保持原来的保存／恢复方式；本次未启用自动阶段复用。"
                        if mode != "disabled" and not rows else None),
            "boundary": "编译接入不代表命中；实际输入与已完成文件核验后才复用。未适配的执行栈正常采样。"}


def reuse_reports(outputs):
    if not isinstance(outputs, dict):
        return []
    return [{"node_id": str(node_id), **report}
            for node_id, value in outputs.items() if isinstance(value, dict)
            for report in (value.get("t8_stage_reuse") or []) if isinstance(report, dict)]
