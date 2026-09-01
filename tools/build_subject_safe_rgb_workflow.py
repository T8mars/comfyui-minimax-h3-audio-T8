from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "workflows" / "13-latent-upscale"
FILENAME = "2026-09-01_H3_Subject_Safe_RGB_Composite_v8_Advanced_EXP.json"


def _node(
    node_id: int,
    node_type: str,
    title: str,
    pos: list[int],
    size: list[int],
    order: int,
    inputs: list[dict],
    outputs: list[dict],
    widgets: list,
) -> dict:
    core_nodes = {
        "MarkdownNote",
        "LoadVideo",
        "GetVideoComponents",
        "ImageToMask",
        "MaskToImage",
        "PreviewImage",
        "PreviewAny",
        "CreateVideo",
        "SaveVideo",
    }
    cnr_id = "comfy-core" if node_type in core_nodes else "minimax-h3-audio-T8"
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"cnr_id": cnr_id, "Node name for S&R": node_type},
        "widgets_values": widgets,
    }


def build_workflow() -> dict:
    overview = (
        "## v8人物安全RGB合成（后处理）\n\n"
        "1. `D0`是学习型latent放大后、不做二次降噪的基底，负责全部背景。\n"
        "2. `T2`是二次低Sigma细化结果，只允许进入人物alpha。\n"
        "3. alpha必须是与视频同帧数、同尺寸的逐帧灰度无损视频；黑色保留D0，白色使用T2。\n"
        "4. 输出始终使用D0音频；节点不对音频加噪、重采样或降噪。\n"
        "5. 本模板已把accept_candidate打开以便生成候选，但仍必须完整人工观看。"
    )
    boundary = (
        "## 使用边界\n\n"
        "节点不会自动识别人、脸、字幕、遮挡或相机运动，也不会自动判断质量。alpha不完整、串人或覆盖背景，"
        "就会把错误带入结果。推荐先用SAM/RAFT获得轨迹，再人工检查alpha；脸、文字和身份保护区域应在输入alpha"
        "中预先清零，或连接节点的可选protect_mask。当前证据只有单人物固定镜头和一条自然跟拍平局，无多人交叉、"
        "甩镜、烟雾、玻璃或半透明衣物的通用保证。"
    )
    nodes = [
        _node(1, "MarkdownNote", "1 · 先看说明", [0, 0], [900, 330], 0, [], [], [overview]),
        _node(
            2,
            "LoadVideo",
            "载入D0基底（背景所有者）",
            [0, 390],
            [360, 124],
            1,
            [],
            [{"name": "VIDEO", "type": "VIDEO", "links": [1]}],
            ["replace_with_D0_base.mp4"],
        ),
        _node(
            3,
            "GetVideoComponents",
            "D0帧、音频和帧率",
            [410, 390],
            [380, 126],
            2,
            [{"name": "video", "type": "VIDEO", "link": 1}],
            [
                {"name": "images", "type": "IMAGE", "links": [4, 10]},
                {"name": "audio", "type": "AUDIO", "links": [7]},
                {"name": "fps", "type": "FLOAT", "links": None},
                {"name": "bit_depth", "type": "COMBO", "links": None},
                {"name": "color_space", "type": "COMBO", "links": None},
            ],
            [],
        ),
        _node(
            4,
            "LoadVideo",
            "载入T2低Sigma细化候选",
            [0, 580],
            [360, 124],
            3,
            [],
            [{"name": "VIDEO", "type": "VIDEO", "links": [2]}],
            ["replace_with_T2_refined.mp4"],
        ),
        _node(
            5,
            "GetVideoComponents",
            "T2帧",
            [410, 580],
            [380, 126],
            4,
            [{"name": "video", "type": "VIDEO", "link": 2}],
            [
                {"name": "images", "type": "IMAGE", "links": [5]},
                {"name": "audio", "type": "AUDIO", "links": None},
                {"name": "fps", "type": "FLOAT", "links": None},
                {"name": "bit_depth", "type": "COMBO", "links": None},
                {"name": "color_space", "type": "COMBO", "links": None},
            ],
            [],
        ),
        _node(
            6,
            "LoadVideo",
            "载入逐帧无损subject alpha",
            [0, 770],
            [360, 124],
            5,
            [],
            [{"name": "VIDEO", "type": "VIDEO", "links": [3]}],
            ["replace_with_lossless_subject_alpha.mkv"],
        ),
        _node(
            7,
            "GetVideoComponents",
            "取alpha帧",
            [410, 770],
            [380, 126],
            6,
            [{"name": "video", "type": "VIDEO", "link": 3}],
            [
                {"name": "images", "type": "IMAGE", "links": [6]},
                {"name": "audio", "type": "AUDIO", "links": None},
                {"name": "fps", "type": "FLOAT", "links": None},
                {"name": "bit_depth", "type": "COMBO", "links": None},
                {"name": "color_space", "type": "COMBO", "links": None},
            ],
            [],
        ),
        _node(
            8,
            "ImageToMask",
            "红通道转最终alpha",
            [840, 770],
            [330, 110],
            7,
            [{"name": "image", "type": "IMAGE", "link": 6}],
            [{"name": "MASK", "type": "MASK", "links": [8]}],
            ["red"],
        ),
        _node(
            9,
            "MiniMaxH3SubjectSafeRGBCompositeT8Advanced",
            "v8 · D0背景 + T2人物",
            [1240, 390],
            [520, 590],
            8,
            [
                {"name": "base_frames", "type": "IMAGE", "link": 4},
                {"name": "refined_frames", "type": "IMAGE", "link": 5},
                {"name": "subject_alpha", "type": "MASK", "link": 8},
                {"name": "protect_mask", "type": "MASK", "link": None, "shape": 7},
                {"name": "audio", "type": "AUDIO", "link": 7, "shape": 7},
            ],
            [
                {"name": "selected", "type": "IMAGE", "links": [9]},
                {"name": "candidate", "type": "IMAGE", "links": [11]},
                {"name": "source", "type": "IMAGE", "links": None},
                {"name": "used_alpha", "type": "MASK", "links": [12]},
                {"name": "audio", "type": "AUDIO", "links": [15]},
                {"name": "report_json", "type": "STRING", "links": [16]},
            ],
            [
                True,
                "input_alpha_exact",
                "strict_exact",
                0.002,
                0.45,
                0.08,
                "fallback_on_contract_failure",
                4,
            ],
        ),
        _node(
            10,
            "PreviewImage",
            "D0原版",
            [1830, 120],
            [340, 150],
            9,
            [{"name": "images", "type": "IMAGE", "link": 10}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            11,
            "PreviewImage",
            "v8候选",
            [1830, 320],
            [340, 150],
            10,
            [{"name": "images", "type": "IMAGE", "link": 11}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            12,
            "MaskToImage",
            "最终使用alpha",
            [1830, 520],
            [340, 110],
            11,
            [{"name": "mask", "type": "MASK", "link": 12}],
            [{"name": "IMAGE", "type": "IMAGE", "links": [13]}],
            [],
        ),
        _node(
            13,
            "PreviewImage",
            "alpha预览",
            [2220, 500],
            [340, 150],
            12,
            [{"name": "images", "type": "IMAGE", "link": 13}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            14,
            "CreateVideo",
            "24fps接回D0原音频",
            [1830, 710],
            [380, 264],
            13,
            [
                {"name": "images", "type": "IMAGE", "link": 9},
                {"name": "audio", "type": "AUDIO", "link": 15, "shape": 7},
            ],
            [{"name": "VIDEO", "type": "VIDEO", "links": [14]}],
            [24.0, 8, "sRGB"],
        ),
        _node(
            15,
            "SaveVideo",
            "保存v8候选",
            [2270, 710],
            [380, 238],
            14,
            [
                {"name": "video", "type": "VIDEO", "link": 14},
                {"name": "format", "type": "COMFY_DYNAMICCOMBO_V3", "link": None},
                {
                    "name": "codec",
                    "type": "COMFY_DYNAMICCOMBO_V3",
                    "link": None,
                    "shape": 7,
                },
            ],
            [{"name": "video", "type": "VIDEO", "links": None}],
            ["MiniMaxH3/SubjectSafe/v8_candidate"],
        ),
        _node(
            16,
            "PreviewAny",
            "合同报告",
            [1830, 1020],
            [820, 190],
            15,
            [{"name": "source", "type": "*", "link": 16}],
            [{"name": "output", "type": "STRING", "links": None}],
            [],
        ),
        _node(17, "MarkdownNote", "2 · 边界与回退", [0, 980], [1170, 330], 16, [], [], [boundary]),
    ]
    links = [
        [1, 2, 0, 3, 0, "VIDEO"],
        [2, 4, 0, 5, 0, "VIDEO"],
        [3, 6, 0, 7, 0, "VIDEO"],
        [4, 3, 0, 9, 0, "IMAGE"],
        [5, 5, 0, 9, 1, "IMAGE"],
        [6, 7, 0, 8, 0, "IMAGE"],
        [7, 3, 1, 9, 4, "AUDIO"],
        [8, 8, 0, 9, 2, "MASK"],
        [9, 9, 0, 14, 0, "IMAGE"],
        [10, 3, 0, 10, 0, "IMAGE"],
        [11, 9, 1, 11, 0, "IMAGE"],
        [12, 9, 3, 12, 0, "MASK"],
        [13, 12, 0, 13, 0, "IMAGE"],
        [14, 14, 0, 15, 0, "VIDEO"],
        [15, 9, 4, 14, 1, "AUDIO"],
        [16, 9, 5, 16, 0, "*"],
    ]
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "t8-h3-subject-safe-rgb-v8")),
        "revision": 0,
        "last_node_id": 17,
        "last_link_id": 16,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.75, "offset": [110, 80]},
            "frontendVersion": "1.24.3",
        },
        "version": 0.4,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / FILENAME
    path.write_text(
        json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
