"""Export an opt-in Director split recipe as an editable native workflow.

The legacy D1 preflight export and Director generation path are unchanged.
Only the selected, already compiled split HyperFlow graph is materialized.
"""

from __future__ import annotations

from copy import deepcopy

from .director_generation import build_director_generation_prompt


def _selected_graph(prompt: dict) -> dict:
    """Keep only the delivery dependency closure, not dead legacy samplers."""
    outputs = [node_id for node_id, node in prompt.items()
               if node["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"]
    if len(outputs) != 1:
        raise ValueError("分离图必须恰有一个导演台交付节点")
    selected: set[str] = set()

    def include(node_id: str) -> None:
        if node_id in selected:
            return
        node = prompt.get(node_id)
        if node is None:
            raise ValueError(f"分离图存在缺失的上游节点：{node_id}")
        selected.add(node_id)
        for value in node.get("inputs", {}).values():
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str) and isinstance(value[1], int)):
                parent = value[0]
                if parent not in prompt:
                    raise ValueError(f"分离图存在缺失的上游节点：{parent}")
                include(parent)

    include(outputs[0])
    return {node_id: deepcopy(node) for node_id, node in prompt.items() if node_id in selected}


def _live_object_info(prompt: dict, registry: dict | None = None) -> dict:
    if registry is None:
        import nodes

        registry = nodes.NODE_CLASS_MAPPINGS
    result = {}
    for kind in {node["class_type"] for node in prompt.values()}:
        cls = registry.get(kind)
        if cls is None:
            raise ValueError(f"当前 Core 未注册分离图节点：{kind}")
        native_info = getattr(cls, "GET_NODE_INFO_V1", None)
        if callable(native_info):
            result[kind] = native_info()
            continue
        inputs = cls.INPUT_TYPES()
        outputs = cls.RETURN_TYPES
        result[kind] = {
            "input": inputs,
            "input_order": {section: list(fields) for section, fields in inputs.items()},
            "output": outputs,
            "output_name": getattr(cls, "RETURN_NAMES", outputs),
            "display_name": kind,
            "python_module": getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes"),
        }
    return result


def export_director_split_workflow(project, shot_id, store, *, seed=26091901,
                                   registry=None):
    """Compile without queueing, then export the actual split execution graph."""
    built = build_director_generation_prompt(project, shot_id, store, seed=seed)
    sampling = built["sampling"]
    if (sampling.get("mode"), sampling.get("variant")) != (
            "hyperflow", "continuous4plus4separate"):
        raise ValueError("当前仅支持导出 HyperFlow 连续4+4分离式镜头图；旧路线仍由原入口处理")
    prompt = _selected_graph(built["prompt"])
    kinds = [node["class_type"] for node in prompt.values()]
    if (kinds.count("MiniMaxH3HyperFlowTailStageEXPT8") != 1
            or "MiniMaxH3HyperFlowSplitT8Advanced" in kinds
            or "MiniMaxH3DualClockSamplerT8" in kinds):
        raise ValueError("导出图未保持分离式阶段合同")
    if sampling["recipe"] == "hyperflow8_continuous_separate_tail_resume_exp_v1":
        if ("MiniMaxH3HyperFlowHeadStageEXPT8" in kinds
                or kinds.count("MiniMaxH3HyperFlowHeadLoadEXPT8") != 1):
            raise ValueError("仅 TAIL 恢复图意外包含 HEAD 采样或缺少冻结读取")
    elif kinds.count("MiniMaxH3HyperFlowHeadStageEXPT8") != 1:
        raise ValueError("完整分离图缺少独立 HEAD 采样")
    info = _live_object_info(prompt, registry)
    # Keep the existing, audited native serializer rather than creating a
    # second ordering/seed-widget implementation for Director alone.
    from .tools.api_to_frontend_workflow import convert

    workflow = convert(prompt, info, "T8 Director · HyperFlow HEAD/TAIL separate EXP")
    workflow.setdefault("extra", {})["t8_director_split"] = {
        "schema": "t8.director.split-workflow-export.v1",
        "project_id": project["id"], "shot_id": shot_id,
        "recipe": sampling["recipe"], "seed": built["seed"],
        "warning": "实验性可编辑图；未排队，恢复图必须使用已验证的真实 HEAD 回执。",
    }
    return {"schema": "t8.director.split-workflow-export.v1",
            "workflow": workflow, "api_snapshot": prompt,
            "recipe": sampling["recipe"], "report": built["report"],
            "warning": "只导出图，不排队、不加载模型；不代表 GPU 音画或人审通过。"}
