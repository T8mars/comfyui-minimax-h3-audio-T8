#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "17-skin-finish"
    / "2026-08-24_H3_Skin_Finish_Two_Pass_Video_Stream_Advanced_EXP.json"
)


def _node(node_id, node_type, title, pos, size, inputs, outputs, widgets, *, core=False):
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {
            "cnr_id": "comfy-core" if core else "minimax-h3-audio-T8",
            "Node name for S&R": node_type,
        },
        "widgets_values": widgets,
    }


def _widget(name, value_type):
    return {"name": name, "type": value_type, "widget": {"name": name}, "link": None}


def _socket(name, value_type):
    return {"name": name, "type": value_type, "link": None}


def _output(name, value_type):
    return {"name": name, "type": value_type, "links": None}


def _note(node_id, title, text, pos, size=(900, 330)):
    return _node(node_id, "MarkdownNote", title, pos, size, [], [], [text], core=True)


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadVideo",
            "1. Final untrimmed 8-bit SDR VIDEO / 最终未裁切8位SDR视频",
            (0, 0),
            (430, 140),
            [_widget("file", "COMBO")],
            [_output("VIDEO", "VIDEO")],
            ["replace_with_final_untrimmed_24fps_video.mp4"],
            core=True,
        ),
        _node(
            2,
            "MiniMaxH3SkinFinishVideoStreamT8Advanced",
            "2. Two-pass bounded file stream / 两遍低内存肤质收尾",
            (560, 0),
            (610, 800),
            [
                _socket("source_video", "VIDEO"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("detection_threshold", "FLOAT"),
                _widget("minimum_face_height_px", "FLOAT"),
                _widget("minimum_detail", "FLOAT"),
                _widget("bbox_ema_alpha", "FLOAT"),
                _widget("scene_cut_threshold", "FLOAT"),
                _widget("maximum_faces", "INT"),
                _widget("mask_feather_px", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("chunk_frames", "INT"),
                _widget("filename_prefix", "STRING"),
                _widget("crf", "FLOAT"),
                _widget("accept_candidate", "BOOLEAN"),
            ],
            [
                _output("video", "VIDEO"),
                _output("saved_path", "STRING"),
                _output("report_json", "STRING"),
            ],
            [
                "subtle",
                0.35,
                0.90,
                0.35,
                0.45,
                24.0,
                0.010,
                0.55,
                0.28,
                4,
                3,
                640,
                4,
                "MiniMaxH3/SkinFinish/stream_skin_finish",
                18.0,
                False,
            ],
        ),
        _note(
            3,
            "START / 用途与接入位置",
            "## 文件级长视频 Skin Finish\n\n把Long Video `Compose Accepted`、Studio `Reel Delivery`或其他最终文件VIDEO接到本节点。推荐顺序：生成/放大/二采 → AV Decode → Face Refine/Motion Recovery → 最终文件合成 → 本节点 → 字幕/调色/Tape-FX。不要先经过`GetVideoComponents`，否则会失去本路线避免完整IMAGE常驻的意义。",
            (0, 930),
            (1000, 370),
        ),
        _note(
            4,
            "TWO PASSES / 两遍执行合同",
            "## 不持有整段IMAGE\n\n第一遍逐帧解码，只保留固定YuNet脸框、权重、切镜和16×16来源摘要；释放Detector后，第二遍最多按`chunk_frames`帧处理并立即编码H.264。两遍摘要不一致、文件中途变化、帧数或尺寸漂移都会拒绝发布。它仍需解码两次，因此目标是降低峰值内存，不是加速。",
            (1080, 930),
            (1050, 390),
        ),
        _note(
            5,
            "MASK + PEOPLE / 多人与遮罩边界",
            "## 保守脸内代理\n\n固定使用本机已校验SHA的OpenCV Zoo YuNet 2023mar，CPU执行、无下载。每帧最多4张脸，切镜重置，bbox EMA 0.55；只处理脸框内部并近似排除眼/鼻/嘴，所有人共享中性色参数。它不加载SAM、不做身份识别，也不是像素级皮肤/头发解析；遮挡、侧脸、小脸和模糊脸会降权或不处理。",
            (2210, 930),
            (1050, 390),
        ),
        _note(
            6,
            "PARAMETERS / 建议参数",
            "## 推荐起点\n\n`subtle / amount 0.35 / texture_keep 0.90 / shine 0.35`；threshold 0.45；最小脸高24px；detail 0.010；cut 0.28；feather 3；proxy 640；CPU chunk 4；CRF 18。小远景脸不是本节点目标：它不会锐化、去模糊、修五官、补身份或生成毛孔。",
            (3340, 930),
            (1000, 360),
        ),
        _note(
            7,
            "AUDIO + ACCEPT / 原音频与人工门",
            "## 默认完全不执行\n\n`accept_candidate=false`时直接返回源VIDEO，不分析、不写文件。需要生成审片候选时才打开；画面逐帧重编码，但AAC/MP3/ALAC/AC3/EAC3原压缩音频packet payload逐包复制并做SHA-256复核。裁切、旋转、HDR/10-bit、未知codec、包漂移或无可靠人脸会fail closed，源文件永不覆盖。",
            (4420, 930),
            (1050, 390),
        ),
    ]
    links = []
    node_map = {node["id"]: node for node in nodes}

    def connect(source, source_slot, target, target_slot, value_type):
        link_id = len(links) + 1
        links.append([link_id, source, source_slot, target, target_slot, value_type])
        output = node_map[source]["outputs"][source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
        node_map[target]["inputs"][target_slot]["link"] = link_id

    connect(1, 0, 2, 0, "VIDEO")
    return {
        "id": "f0c3c398-5f4a-4a02-8e31-cd3916bcb558",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish P1: true two-pass file stream without full IMAGE input",
                "bounding": [-50, -80, 1270, 960],
                "color": "#315a78",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.65, "offset": [120, 80]},
            "workflow_title": (
                "2026-08-24 H3 Skin Finish Two Pass Video Stream Advanced EXP"
            ),
            "t8_skin_finish": {
                "scope": "file-backed two-pass bounded CPU Skin Finish",
                "full_image_input": False,
                "default_selection": "source",
                "semantic_skin_parser": False,
                "hdr_supported": False,
            },
        },
        "version": 0.4,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
