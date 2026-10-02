"""Build opt-in, editable FL2VA/Ref2VA PDD 4+4 split examples."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from tools.audit_modular_sampling_compat import write_new
from tools import build_modular_pdd_workflow as pdd
from tools.modular_frontend_layout import spread_frontend_columns


DESTINATION = pdd.ROOT / "examples/workflows/19-pdd-acceleration"
GLOBAL_PROMPT = (
    "A single adult cyclist rides a bicycle through a quiet city street at dusk. "
    "Keep the rider, bicycle, direction, clothing and ambient sound consistent."
)
LOCAL_PROMPTS = (
    "The cyclist passes warm streetlights with steady forward motion.\n"
    "The camera tracks alongside as the cyclist continues into the distance."
)
FIRST_IMAGE = "SELECT_YOUR_FIRST_FRAME.png"
LAST_IMAGE = "SELECT_YOUR_LAST_FRAME.png"
REFERENCE_IMAGE = "SELECT_YOUR_REFERENCE_IMAGE.png"
VARIANTS = ("save_effects", "resume_effects")
FILES = {
    (base, variant):
        f"PDD_Split_{base}_4plus4_{'Full_Stages' if variant == 'save_effects' else 'Cold_HIGH'}_EXP.json"
    for base in pdd.BASES for variant in VARIANTS
}


def graph_for(base: str, variant: str) -> dict:
    if base not in pdd.BASES or variant not in VARIANTS:
        raise ValueError("Expected a reviewed PDD base and full/cold split variant")
    graph = pdd.split_graph(base, variant)
    for node in graph.values():
        if node["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced":
            node["inputs"].update(global_prompt=GLOBAL_PROMPT,
                                  local_prompts=LOCAL_PROMPTS)
    graph["95"]["inputs"]["image"] = (FIRST_IMAGE if base == "FL2VA" else REFERENCE_IMAGE)
    if base == "FL2VA":
        graph["96"]["inputs"]["image"] = LAST_IMAGE
    return graph


def load_info() -> dict:
    info = pdd.shared.load_live_info()
    import nodes as core_nodes

    for name in ("LoadImage", "ImageScale"):
        info[name] = pdd.shared.native_info(name, core_nodes.NODE_CLASS_MAPPINGS[name])
    return info


def build_suite(info: dict) -> dict:
    result = {}
    for base, variant in FILES:
        graph = graph_for(base, variant)
        selected, workflow, audit = pdd.build_candidate(
            base, variant, info, graph_override=deepcopy(graph))
        notes = [node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]
        if len(notes) != 1:
            raise ValueError("Expected one source-owned PDD note")
        notes[0]["widgets_values"] = [
            f"PDD {base} 分离式4+4 EXP：LOW与HIGH各自的MODEL／PDD LoRA、条件、NOISE、"
            "外置Prompt Relay和Stage EAV均可单独接线／编辑；总共仍为原PDD八次前向，"
            "不是完整8步跑两遍。LOW denoised_output经原learned3D和legacy_policy音频交接HIGH。"
            "EAV默认report_only，只观察不增强。"
            + ("本图显式保存LOW/HIGH；记下LOW的artifact_path与SHA。" if variant == "save_effects"
               else "本图只读已保存LOW并运行HIGH；填入准确manifest路径和SHA，图中没有LOW模型或采样器。")
            + "替换占位首帧／尾帧或参考图，确认底模、对应PDD权重、Qwen、双VAE和尺寸。"
            "历史真实GPU仅覆盖两种底模各自固定小画布22帧及report_only的机械／冷恢复，"
            "不代表本图默认画布、多素材／LoRA／后端、apply_exp画质或人工画音通过。"
            "旧一体PDD节点与工作流不迁移。"
        ]
        workflow.setdefault("extra", {})["t8_split_example"] = {
            "schema": "t8.pdd.split-example.v1", "route": "S04", "base": base,
            "variant": variant, "status": "experimental_importable_not_quality_accepted",
        }
        spread_frontend_columns(workflow)
        result[(base, variant)] = {"graph": selected, "workflow": workflow, "audit": audit}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(pdd.ROOT.resolve()):
        parser.error("Output must remain inside this project")
    suite = build_suite(load_info())
    for key, filename in FILES.items():
        write_new(destination / filename, suite[key]["workflow"])
        print(f"{filename}: {suite[key]['audit']['nodes']} nodes, "
              f"{suite[key]['audit']['edges']} edges")


if __name__ == "__main__":
    main()
