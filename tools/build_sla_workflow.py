#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

try:
    from .api_to_frontend_workflow import convert
except ImportError:
    from api_to_frontend_workflow import convert


NOTES = (
    (
        "① 正确连接与硬约束",
        "## 正确连接\n\n"
        "`原生 H3 UNET → Dual-Clock(4步 / native_flow / video shift 6 / audio shift 3) "
        "→ LightX2V SLA → BasicGuider`。同时把 Dual-Clock 的同一组 `sigmas` 接入 SLA。"
        "SLA 节点自己加载固定 LoRA；必须删除或旁路旧 Turbo LoRA、KJ Sage、Sol-Attn、FETA、"
        "BlockCache、STG 和其他 attention wrapper。当前只开放 FL2VA，必须同时连接首帧与尾帧。",
    ),
    (
        "② 模式、模型与参数",
        "## 三种模式\n\n"
        "`apply_lightx2v_sla`：85% 动态块稀疏 + Sage2，是生成模式。"
        "`dense_lora_control`：同一个 SLA LoRA，但主 DiT 使用稠密 attention，供同 seed A/B。"
        "`disabled_identity`：完全旁路，不加载 LoRA。上游 LoRA 标注的正式基座是 BF16 FL2VA；"
        "当前 INT8 ConvRot 基座只能算兼容实验，`auto_detect_exp` 会明确记录，"
        "`official_bf16_only` 会拒绝量化基座。`max_router_workspace_mib` 只限制路由图临时空间，"
        "不是整套工作流显存上限。",
    ),
    (
        "③ 审计结果与科学边界",
        "## 运行后必须检查 Audit\n\n"
        "成功的稀疏运行应显示 `lightx2v_dynamic_sparse_verified`、4 次模型前向、每次 50 个主块、"
        "200 次稀疏内核、0 次 dense fallback、0 次 kernel failure。85% 是请求稀疏率；"
        "实际保留率按 key block 数量向下取整并至少保留 1 块，所以短序列可能不是精确 15%。"
        "本节点复现的是 LightX2V 发布版的 learned block router + Sage2 路径，不宣称包含通用 SLA "
        "论文的全部 sparse+linear 分支。256×256×22 的 INT8 兼容机械验证已通过；"
        "公开模板默认 736×416×124，仍需按自己的显卡验证质量、速度和显存。",
    ),
)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def build(api_prompt: dict, object_info: dict) -> dict:
    prompt = json.loads(json.dumps(api_prompt))
    prompt["7"]["inputs"].update({"width": 736, "height": 416, "length": 124})
    prompt["15"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3_SLA/lightx2v_sla_fl2va_736x416_124f"
    )
    # VideoHelperSuite exposes format-specific widgets dynamically.  They are
    # valid API inputs but are not part of its stable object_info input order,
    # so the frontend converter must let VHS recreate their defaults.
    for name in ("pix_fmt", "crf", "save_metadata", "trim_to_audio"):
        prompt["15"]["inputs"].pop(name, None)
    workflow = convert(
        prompt,
        object_info,
        "MiniMax H3 LightX2V SLA FL2VA 4-Step (Advanced EXP)",
    )
    next_id = int(workflow["last_node_id"]) + 1
    for index, (title, text) in enumerate(NOTES):
        workflow["nodes"].append(
            {
                "id": next_id + index,
                "type": "MarkdownNote",
                "title": title,
                "pos": [index * 1050, 1150],
                "size": [930, 430],
                "flags": {},
                "order": len(workflow["nodes"]),
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {
                    "Node name for S&R": "MarkdownNote",
                    "cnr_id": "comfy-core",
                },
                "widgets_values": text,
            }
        )
    workflow["last_node_id"] = next_id + len(NOTES) - 1
    workflow["groups"] = [
        {
            "id": 1,
            "title": "MiniMax H3 LightX2V SLA · FL2VA 4 NFE · 6V/3A",
            "bounding": [-80, -80, 3300, 1050],
            "color": "#6b4fa1",
            "font_size": 28,
            "flags": {},
        },
        {
            "id": 2,
            "title": "使用说明 / 参数 / 科学边界",
            "bounding": [-80, 1070, 3300, 560],
            "color": "#3f789e",
            "font_size": 26,
            "flags": {},
        },
    ]
    workflow["extra"]["ds"] = {"scale": 0.55, "offset": [120, 120]}
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("api_prompt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    prompt = json.loads(args.api_prompt.read_text(encoding="utf-8"))
    object_info = _get_json(f"{args.server.rstrip('/')}/object_info")
    missing = sorted({node["class_type"] for node in prompt.values()} - set(object_info))
    if missing:
        raise ValueError(f"server is missing nodes: {missing}")
    workflow = build(prompt, object_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
