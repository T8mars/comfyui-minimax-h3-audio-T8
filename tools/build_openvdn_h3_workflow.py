#!/usr/bin/env python3
"""Build the pinned OpenVDN MiniMax H3 DMD8 frontend workflow."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE_WORKFLOW_ID = "d42d6f0e-7e73-4e86-bc76-044ec9024404"
WORKFLOW_ID = "94462fd1-0d4e-4b94-a5c7-0cb44db12de8"
COMPOSER_TYPE = "MiniMaxH3VDNModelComposerT8Advanced"
EXECUTION_TYPE = "MiniMaxH3VDNExecutionPlanT8Advanced"
BASE_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VDN_ROOT = "OpenVDN/vdn-minimax-h3"

PROMPT = (
    "One continuous locked-off cinematic medium close-up of one adult East Asian "
    "woman facing the camera in a quiet, softly lit concert room. She remains the "
    "only visible and audible person. She says exactly once, clearly and naturally "
    "in Mandarin Chinese: <d>[Chinese] 你在哪里</d>. Her mouth movement precisely "
    "matches this one short sentence, then her lips relax. Clean classical chamber "
    "music plays continuously underneath: warm solo cello and soft acoustic piano "
    "performing a slow adagio, with the voice clear in the foreground and music at "
    "a lower level. Stable face, eyes and anatomy, realistic skin, one continuous "
    "shot. No additional words, repeated speech, singing, subtitles, cuts, camera "
    "movement, hiss, static, crackle, clipping, distortion, duplicate person or "
    "reflection."
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def build(source: dict) -> dict:
    """Convert the released FastH3 T2VA graph into the isolated OpenVDN graph."""

    workflow = copy.deepcopy(source)
    if workflow.get("id") != SOURCE_WORKFLOW_ID:
        raise ValueError("unexpected FastH3 source workflow identity")

    model_loader = _node(workflow, 1)
    model_loader["widgets_values"] = [BASE_MODEL, "default"]

    composer = _node(workflow, 2)
    composer.update(
        {
            "type": COMPOSER_TYPE,
            "title": "2. Compose pinned OpenVDN DMD8 branch + adapters",
            "size": [450, 166],
            "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [2]},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            "properties": {
                "cnr_id": "minimax-h3-audio-T8",
                "Node name for S&R": COMPOSER_TYPE,
            },
            "widgets_values": [VDN_ROOT, "stage_dmd_8nfe", True, True],
        }
    )

    conditioning = _node(workflow, 6)
    conditioning["title"] = (
        "3. Plain T2VA only · 960x512 · 73 frames · speech + classical music"
    )
    conditioning["widgets_values"] = [
        PROMPT,
        960,
        512,
        73,
        "T2VA",
        "native",
        1.0,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
    ]

    execution = _node(workflow, 7)
    execution.update(
        {
            "type": EXECUTION_TYPE,
            "title": "4. OpenVDN exact execution contract · DMD 8 NFE · 12V/3A",
            "size": [410, 142],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 2},
                {"name": "av_latent", "type": "LATENT", "link": 6},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [7]},
                {"name": "sampler", "type": "SAMPLER", "links": [11]},
                {"name": "sigmas", "type": "SIGMAS", "links": [12]},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            "properties": {
                "cnr_id": "minimax-h3-audio-T8",
                "Node name for S&R": EXECUTION_TYPE,
            },
            "widgets_values": [],
        }
    )

    _node(workflow, 9)["widgets_values"] = [2609032101, "fixed"]
    save = _node(workflow, 12)
    save["title"] = "Save OpenVDN DMD8 T2VA candidate"
    save["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3/OpenVDN_DMD8_T2VA/openvdn_dmd8_0p5mp_73f"
    )

    note = _node(workflow, 13)
    note["title"] = "OpenVDN MiniMax H3 v1 使用边界"
    note["widgets_values"] = (
        "## OpenVDN MiniMax H3 · 独立 Advanced EXP 路线\n\n"
        "权重固定到 `OpenVDN/vdn-minimax-h3` revision "
        "`18be6bcc4ee72585eee322ba28b5ccac2cf85ef0`，放在 "
        "`models/diffusion_models/OpenVDN/vdn-minimax-h3/`。默认 DMD 路线内部已经"
        "组合 default + turbo adapter，**不要再接 EMA_B、Turbo、SLA、VSA、"
        "Sol-Attn、BlockCache 或其他 MODEL/Attention 补丁**。\n\n"
        "v1 只允许普通 T2VA；任何首尾帧、图片、视频、音频参考或混合参考都会 "
        "fail closed。当前本机使用结构匹配的 INT8/ConvRot H3 底模，所以 Composer "
        "明确开启 `allow_structural_base`，报告不会把它冒充上游精确 BF16 base。"
        "若切换 Stage B，只改 Composer 的 stage 为 `stage_b_50nfe`；Execution Plan "
        "会自动改为 50 NFE，不能手工套用 8 步。\n\n"
        "权重遵循 MiniMax H3 Community License，其Applicable Territory排除欧盟、英国、"
        "韩国和美国；下载或运行前必须自行阅读模型目录中的 `NOTICE` 与完整许可。\n\n"
        "本机 960×512×73、DMD 8 NFE 已真实生成并通过 50 层分支、800个分支张量、"
        "104+259个 adapter target、音画解码及 512MiB 显存余量机械门。画质、"
        "声音和口型仍要看实际成片，不能从运行报告直接推断。"
    )

    workflow["id"] = WORKFLOW_ID
    workflow["revision"] = 0
    workflow["extra"]["workflow_name"] = (
        "MiniMax H3 OpenVDN DMD8 T2VA 0.5MP (Advanced EXP)"
    )
    workflow["groups"][0]["title"] = "H3 loaders + OpenVDN model composition"
    workflow["groups"][1]["title"] = "OpenVDN DMD · fixed 8 NFE · 12V/3A"
    workflow["groups"][2]["title"] = "Decode + synchronized MP4"
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
