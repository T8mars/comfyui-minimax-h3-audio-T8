#!/usr/bin/env python3
"""Build the dated SLA Precision V2 frontend workflow from the released router graph."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE_TYPE = "MiniMaxH3TurboSLAProfileRouterT8Advanced"
LOADER_TYPE = "MiniMaxH3SLADynamicLoRABypassV2T8Advanced"
ATTENTION_TYPE = "MiniMaxH3SLAPrecisionV2T8Advanced"
AUDIT_TYPE = "MiniMaxH3SLAPrecisionV2AuditT8Advanced"
RUNTIME_TYPE = "MINIMAX_H3_SLA_PRECISION_V2_RUNTIME"
FP8_BASE = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
SLA_LORA = "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def _reset_links(workflow: dict) -> None:
    workflow["links"] = []
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = None


def _connect(
    workflow: dict,
    link_id: int,
    source_id: int,
    source_slot: int,
    target_id: int,
    target_slot: int,
    data_type: str,
) -> None:
    source = _node(workflow, source_id)["outputs"][source_slot]
    target = _node(workflow, target_id)["inputs"][target_slot]
    source.setdefault("links", None)
    source["links"] = list(source["links"] or []) + [link_id]
    target["link"] = link_id
    workflow["links"].append(
        [link_id, source_id, source_slot, target_id, target_slot, data_type]
    )


def build(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    by_type = {node["type"]: node for node in workflow["nodes"]}
    if SOURCE_TYPE not in by_type:
        raise ValueError("source must contain the released Turbo/SLA Profile Router")
    if any(node["type"] in {LOADER_TYPE, ATTENTION_TYPE, AUDIT_TYPE} for node in workflow["nodes"]):
        raise ValueError("source already contains SLA Precision V2 nodes")

    _node(workflow, 4)["widgets_values"] = [FP8_BASE, "default"]
    loader = _node(workflow, 9)
    loader.update(
        {
            "type": LOADER_TYPE,
            "title": "SLA LoRA · dynamic model-only bypass (no base re-quantization)",
            "size": [390, 132],
            "inputs": [{"name": "model", "type": "MODEL", "link": None}],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": None},
                {"name": "report_json", "type": "STRING", "links": None},
            ],
            "properties": {
                "cnr_id": "minimax-h3-audio-T8",
                "Node name for S&R": LOADER_TYPE,
            },
            "widgets_values": [SLA_LORA],
        }
    )

    for node in workflow["nodes"]:
        if int(node.get("order", 0)) >= 10:
            node["order"] = int(node["order"]) + 1
        if node["type"] not in {"MarkdownNote"} and node["pos"][0] >= 2200:
            node["pos"][0] += 440

    attention_id = int(workflow["last_node_id"]) + 1
    attention = {
        "id": attention_id,
        "type": ATTENTION_TYPE,
        "title": "SLA Precision V2 · PlagueKind v1.4.3 precision route",
        "pos": [2200, 0],
        "size": [390, 492],
        "flags": {},
        "order": 10,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "sigmas", "type": "SIGMAS", "link": None},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": None},
            {"name": "runtime", "type": RUNTIME_TYPE, "links": None},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "properties": {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": ATTENTION_TYPE,
        },
        "widgets_values": [
            "recommended_8nfe_12v_3a",
            0.9,
            "32",
            8192,
            1,
            True,
            "0",
            "comfy_kitchen",
            True,
            False,
            False,
        ],
    }
    workflow["nodes"].append(attention)

    audit = _node(workflow, 13)
    audit.update(
        {
            "type": AUDIT_TYPE,
            "title": "SLA Precision V2 Runtime Audit · fail closed",
            "inputs": [
                {"name": "av_latent", "type": "LATENT", "link": None},
                {"name": "runtime", "type": RUNTIME_TYPE, "link": None},
            ],
            "outputs": [
                {"name": "av_latent", "type": "LATENT", "links": None},
                {"name": "report_json", "type": "STRING", "links": None},
            ],
            "properties": {
                "cnr_id": "minimax-h3-audio-T8",
                "Node name for S&R": AUDIT_TYPE,
            },
            "widgets_values": [],
        }
    )
    _node(workflow, 15)["widgets_values"][2] = (
        "MiniMaxH3_SLA/precision_v2_fp8_8nfe_12v3a_736x416_124f"
    )

    notes = {
        "① 正确连接与硬约束": (
            "## Precision V2 推荐连接\n\n"
            "`FP8 FL2VA → Dual-Clock(8 NFE / native_flow / 12V / 3A) → "
            "SLA Dynamic LoRA Bypass V2 → SLA Precision V2 Attention → BasicGuider`。"
            "同一组`sigmas`必须同时接入Precision V2，采样后必须通过它自己的Runtime Audit。"
            "不要再叠加旧SLA、KJ/Sage、Sol-Attn、FETA、BlockCache、STG或其他Attention owner。"
        ),
        "② 模式、模型与参数": (
            "## 已固定的质量修复参数\n\n"
            "该模板使用当前PlagueKind v1.4.3固定提交`066ada9`的精度路线：FP32块路由、"
            "直接Triton FP32 online-softmax稀疏核、32×32块、请求90%稀疏、首步与末步Dense、"
            "中间6步Sparse、语言与音频块全保护，并关闭reduced-precision accumulation。"
            "LoRA以动态model-only residual注入，避免把FP8底模合并后再次量化。旧节点保持兼容，"
            "但不再是此模板的推荐实现。"
        ),
        "③ 审计结果与科学边界": (
            "## 必须看Audit与真实成片\n\n"
            "推荐档应逐逻辑步报告`0/7 Dense`、`1-6 Sparse`，每个主步骤50个H3 attention块，"
            "总计300次Sparse、至少100次Dense、保护块大于0、kernel fallback为0。"
            "同输入同Seed的736×416×124对白实测通过严格解码、ASR与带400ms负对照的SyncNet，"
            "但RTX 4060 Ti 16GB最低空闲仅211MiB，未过项目512MiB安全门，因此仍标Advanced EXP，"
            "不宣称所有16GB环境安全。运行审计也不能替代正常速度的人脸、运动、音频与口型人审。"
        ),
    }
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote" and node.get("title") in notes:
            node["widgets_values"] = notes[node["title"]]

    _reset_links(workflow)
    connections = [
        (3, 0, 7, 0, "CLIP"),
        (1, 0, 7, 1, "VAE"),
        (2, 0, 7, 2, "VAE"),
        (5, 0, 7, 5, "IMAGE"),
        (6, 0, 7, 6, "IMAGE"),
        (4, 0, 8, 0, "MODEL"),
        (7, 1, 8, 1, "LATENT"),
        (8, 0, 9, 0, "MODEL"),
        (9, 0, attention_id, 0, "MODEL"),
        (8, 2, attention_id, 1, "SIGMAS"),
        (attention_id, 0, 11, 0, "MODEL"),
        (7, 0, 11, 1, "CONDITIONING"),
        (10, 0, 12, 0, "NOISE"),
        (11, 0, 12, 1, "GUIDER"),
        (8, 1, 12, 2, "SAMPLER"),
        (8, 2, 12, 3, "SIGMAS"),
        (7, 1, 12, 4, "LATENT"),
        (12, 0, 13, 0, "LATENT"),
        (attention_id, 1, 13, 1, RUNTIME_TYPE),
        (13, 0, 14, 0, "LATENT"),
        (1, 0, 14, 1, "VAE"),
        (2, 0, 14, 2, "VAE"),
        (14, 0, 15, 0, "IMAGE"),
        (14, 1, 15, 1, "AUDIO"),
    ]
    for link_id, connection in enumerate(connections, start=1):
        _connect(workflow, link_id, *connection)

    workflow["last_node_id"] = attention_id
    workflow["last_link_id"] = len(connections)
    workflow["extra"]["workflow_name"] = (
        "MiniMax H3 SLA Precision V2 FL2VA FP8 8-Step (Advanced EXP)"
    )
    workflow["groups"][0]["title"] = (
        "MiniMax H3 SLA Precision V2 · FP8 FL2VA · 8 NFE · 12V/3A"
    )
    workflow["groups"][0]["bounding"][2] += 440
    return workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mirror", type=Path)
    args = parser.parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    workflow = build(source)
    rendered = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    if args.mirror is not None:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
        args.mirror.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
