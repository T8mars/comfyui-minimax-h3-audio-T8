"""Build importable S17 SPEED 2/3-stage T2VA research workflows."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from tools import build_modular_speed_workflow as speed
from tools.audit_modular_sampling_compat import write_new
from tools.modular_frontend_layout import spread_frontend_columns


DESTINATION = speed.ROOT / "examples/workflows/50-speed-split"
KINDS = ("eav", "relay", "combined")
SCOPES = {2: ("stage0", "stage1", "all"),
          3: ("stage0", "stage1", "stage2", "all")}
VARIANTS = {2: ("full_no_save", "full_save", "cold_1"),
            3: ("full_no_save", "full_save", "cold_1", "cold_2")}
EFFECTS = {stages: (("none", "none"), *product(KINDS, SCOPES[stages]))
           for stages in (2, 3)}
FILES = {
    (stages, kind, scope, variant):
        f"SPEED_S17_{stages}Stage_{kind}_{scope}_{variant}_EXP.json"
    for stages in (2, 3)
    for kind, scope in EFFECTS[stages]
    for variant in VARIANTS[stages]
}


def effects_for(stages: int, kind: str, scope: str) -> dict[int, frozenset[str]]:
    if stages not in EFFECTS or (kind, scope) not in EFFECTS[stages]:
        raise ValueError("Expected a reviewed S17 SPEED stage/effect variant")
    enabled = {"eav", "relay"} if kind == "combined" else set() if kind == "none" else {kind}
    selected = set(range(stages)) if scope == "all" else (
        set() if scope == "none" else {int(scope[-1])})
    return {index: frozenset(enabled if index in selected else ())
            for index in range(stages)}


def graph_for(stages: int, kind: str, scope: str, variant: str) -> dict:
    if stages not in VARIANTS or variant not in VARIANTS[stages]:
        raise ValueError("Expected a reviewed S17 SPEED storage variant")
    resume_stage = int(variant[-1]) if variant.startswith("cold_") else None
    return speed.split_graph(
        stages, resume_stage=resume_stage,
        effects_by_stage=effects_for(stages, kind, scope),
        save_stages=variant != "full_no_save")


def note_for(stages: int, kind: str, scope: str, variant: str) -> str:
    effect_note = ("无外置效果。" if kind == "none" else
                   f"外置 {kind} 仅按 {scope} 阶段配置；Relay Plan 与 EAV Config 分开编辑，"
                   "EAV 默认 report_only，不表示画质增强。")
    storage_note = {
        "full_no_save": "完整运行，不写中间阶段回执；相邻阶段仍由真实 DCT 转段连接。",
        "full_save": "完整运行，显式保存每个非末段回执，记录实际 manifest 路径与 SHA。",
        "cold_1": "填写 Stage0 的真实 manifest 路径和 SHA，仅从 Stage1 继续；冻结阶段不再加载模型或重采。",
        "cold_2": "填写 Stage1 的真实 manifest 路径和 SHA，仅从 Stage2 继续；Stage0/1 不再运行。",
    }[variant]
    return (
        f"S17 SPEED {stages} 阶段 T2VA／{kind}:{scope}／{variant} 分离式研究 EXP。"
        "SPEED 是多分辨率 DCT 转段，不是 learned3D LOW→HIGH；每段有独立 MODEL、"
        "可插 LoRA、Source/条件、Sampler、噪声及显式 StageSample，转段单独处理视频频域"
        "和联合音频时间重索引。后段改动在图结构上不重新执行已冻结前段。"
        + effect_note + storage_note +
        "冷图只选定已保存的旧阶段，不会根据当前前段编辑自动命中；坏路径/SHA须拒绝。"
        "本组固定手工 sigma 仅供机械示范，旧用户盲评已否决固定 SPEED 计划，"
        "绝非质量或速度推荐，也不改旧整链默认。通用 T2VA 图不代表 I2VA/FL2VA/"
        "L2VA/Ref2VA/Hybrid 或参考音视频已交付。CLIP、VAE、底模、素材需按本机检查；"
        "冷图占位回执不可直接排队。旧一体图保留。可导入/保存重开与小尺寸机械相等"
        "不等于原尺寸真实权重音画、EAV apply_exp 质量或人工验收。")


def build_suite(info: dict) -> dict:
    result = {}
    for key in FILES:
        stages, kind, scope, variant = key
        graph, selected = speed.selected_frontend_schema(graph_for(*key), info)
        title = f"S17 SPEED {stages}-stage {kind}:{scope} {variant} EXP"
        workflow = speed.convert(graph, selected, title)
        by_id = {item["id"]: item for item in workflow["nodes"]}
        for node_id, node in zip(graph, by_id):
            if graph[node_id]["class_type"] == "UNETLoader":
                by_id[node]["title"] = f"Stage MODEL {node_id} — independent LoRA input"
        note_id = workflow["last_node_id"] + 1
        workflow["nodes"].append({
            "id": note_id, "type": "MarkdownNote", "title": "先读 / S17 SPEED EXP",
            "pos": [0, -800], "size": [1180, 520], "flags": {}, "order": len(graph),
            "mode": 0, "inputs": [], "outputs": [], "properties": {},
            "widgets_values": [note_for(*key)],
        })
        workflow["last_node_id"] = note_id
        workflow.setdefault("extra", {})["t8_split_example"] = {
            "schema": "t8.speed.split-example.v1", "route": "S17",
            "stages": stages, "effect_kind": kind, "effect_scope": scope,
            "variant": variant, "task_type": "T2VA",
            "status": "experimental_importable_not_quality_accepted",
        }
        workflow["extra"]["workflow_title"] = title
        spread_frontend_columns(workflow)
        audit = speed.audit_candidate(graph, workflow, selected)
        result[key] = {"graph": graph, "workflow": workflow, "audit": audit}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    options = parser.parse_args()
    destination = options.output_dir.resolve()
    if not destination.is_relative_to(speed.ROOT.resolve()):
        parser.error("Output must remain inside this project")
    suite = build_suite(speed.load_live_info())
    for key, filename in FILES.items():
        write_new(destination / filename, suite[key]["workflow"])
        print(f"{filename}: {suite[key]['audit']['nodes']} nodes, "
              f"{suite[key]['audit']['edges']} edges")


if __name__ == "__main__":
    main()
