#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


KJ_TYPE = "MiniMaxH3MemoryEfficientSageAttentionPatch"
SLA_TYPE = "MiniMaxH3LightX2VSLAT8Advanced"
COMPOSER_TYPE = "MiniMaxH3LightX2VSLAKJSageComposerT8Advanced"


def build(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    by_type = {node["type"]: node for node in workflow["nodes"]}
    if SLA_TYPE not in by_type or "MiniMaxH3DualClockSamplerT8" not in by_type:
        raise ValueError("source must be the public T8 LightX2V SLA workflow")
    if KJ_TYPE in by_type or COMPOSER_TYPE in by_type:
        raise ValueError("source already contains a KJ Sage composer route")

    dual_clock = by_type["MiniMaxH3DualClockSamplerT8"]
    sla_node = by_type[SLA_TYPE]
    old_link_id = int(sla_node["inputs"][0]["link"])
    model_link = next(link for link in workflow["links"] if int(link[0]) == old_link_id)
    if int(model_link[1]) != int(dual_clock["id"]) or int(model_link[2]) != 0:
        raise ValueError("source Dual-Clock to SLA model link changed")

    new_node_id = int(workflow["last_node_id"]) + 1
    new_link_id = int(workflow["last_link_id"]) + 1
    for node in workflow["nodes"]:
        if int(node.get("order", 0)) >= int(sla_node["order"]):
            node["order"] = int(node["order"]) + 1
        if node.get("pos", [0])[0] >= sla_node["pos"][0] and node["type"] != "MarkdownNote":
            node["pos"][0] += 440

    kj_node = {
        "id": new_node_id,
        "type": KJ_TYPE,
        "title": "KJ MiniMax H3 Memory-Efficient Sage (required upstream)",
        "pos": list(sla_node["pos"]),
        "size": [390, 96],
        "flags": {},
        "order": int(sla_node["order"]) - 1,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": old_link_id}],
        "outputs": [{"name": "model", "type": "MODEL", "links": [new_link_id]}],
        "properties": {
            "Node name for S&R": KJ_TYPE,
            "cnr_id": "comfyui-kjnodes",
        },
        "widgets_values": [],
    }
    workflow["nodes"].append(kj_node)
    model_link[3] = new_node_id
    model_link[4] = 0
    workflow["links"].append(
        [new_link_id, new_node_id, 0, int(sla_node["id"]), 0, "MODEL"]
    )
    sla_node["inputs"][0]["link"] = new_link_id
    sla_node["type"] = COMPOSER_TYPE
    sla_node["title"] = "SLA + KJ Sage Composer (one attention owner)"
    sla_node["properties"]["Node name for S&R"] = COMPOSER_TYPE

    note_updates = {
        "① 正确连接与硬约束": (
            "## 正确连接\n\n"
            "`原生 H3 UNET → Dual-Clock(4步 / native_flow / 6V / 3A) → KJ MiniMax H3 Sage "
            "→ SLA + KJ Composer → BasicGuider`。不要在KJ与Composer之间加入"
            "ModelAttentionBackend、Sol-Attn或其他Attention节点。Composer只接受完整50-block KJ "
            "Sage补丁；缺块或被覆盖会直接报错。当前仍只开放FL2VA首帧+尾帧。"
        ),
        "② 模式、模型与参数": (
            "## 两条互斥执行路径\n\n"
            "`apply_lightx2v_sla`：主路径使用SLA动态块路由和block-sparse Sage2；KJ调用数必须为0。"
            "`dense_lora_control`：保持同一SLA LoRA，但50个主块全部委托KJ Sage，供同seed对照。"
            "每次Attention只运行一个后端，并不是SLA与KJ重复计算。`disabled_identity`直接保留上游"
            "KJ模型且不加载SLA LoRA。正式基座仍是BF16 FL2VA，量化基座仅为兼容实验。"
        ),
        "③ 审计结果与科学边界": (
            "## Audit必须通过\n\n"
            "SLA生成模式应为4次前向×50次block-sparse、0次KJ绕过；KJ稠密对照应为4×50次KJ "
            "Sage、0次SLA稀疏调用。任何forward替换、kernel failure或静默fallback都失败。"
            "此组合器解决的是KJ完整forward与SLA override的hook冲突，不承诺把两个加速比相乘，"
            "也不表示画质、速度、音频或所有16GB场景已经优于单独SLA。"
        ),
    }
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote" and node.get("title") in note_updates:
            node["widgets_values"] = note_updates[node["title"]]

    workflow["last_node_id"] = new_node_id
    workflow["last_link_id"] = new_link_id
    workflow["extra"]["workflow_name"] = (
        "MiniMax H3 LightX2V SLA + KJ Sage Composer FL2VA 4-Step (Advanced EXP)"
    )
    if workflow.get("groups"):
        workflow["groups"][0]["title"] = (
            "MiniMax H3 LightX2V SLA + KJ Sage · one Attention owner"
        )
        workflow["groups"][0]["bounding"][2] += 440
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    workflow = build(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
