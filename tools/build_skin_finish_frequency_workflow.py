#!/usr/bin/env python3
from __future__ import annotations

import json

from build_skin_finish_workflow import OUTPUT as P0_OUTPUT
from build_skin_finish_workflow import _node, _note, _output, _socket, _widget


OUTPUT = P0_OUTPUT.with_name(
    "2026-08-25_H3_Skin_Finish_Frequency_Split_Advanced_EXP.json"
)
PROJECT_ROOT = OUTPUT.parents[3]
USER_OUTPUT = (
    PROJECT_ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "17-skin-finish"
    / OUTPUT.name
)


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadImage",
            "1. Source frame or decoded IMAGE batch / 原片或解码帧批次",
            (0, 0),
            (390, 430),
            [_widget("image", "COMBO"), _widget("upload", "IMAGEUPLOAD")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_source_frame.png", "image"],
            core=True,
        ),
        _node(
            2,
            "LoadImage",
            "2. Reviewed semantic skin mask / 已审核语义皮肤遮罩",
            (0, 500),
            (390, 430),
            [_widget("image", "COMBO"), _widget("upload", "IMAGEUPLOAD")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_skin_mask_rgba.png", "image"],
            core=True,
        ),
        _node(
            3,
            "LoadAudio",
            "Optional original soundtrack / 可选原始音频",
            (0, 980),
            (390, 120),
            [_widget("audio", "COMBO")],
            [_output("AUDIO", "AUDIO")],
            ["replace_with_source_audio.wav"],
            core=True,
        ),
        _node(
            4,
            "MiniMaxH3SkinFinishAdvancedT8",
            "3. Build reviewed low-frequency candidate / 生成待审低频候选",
            (500, 100),
            (470, 650),
            [
                _socket("frames", "IMAGE"),
                _widget("mask_source", "COMBO"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("tone_adjust", "FLOAT"),
                _widget("execution_mode", "COMBO"),
                _widget("accept_candidate", "BOOLEAN"),
                _widget("protect_features", "BOOLEAN"),
                _widget("minimum_mask_area", "FLOAT"),
                _widget("maximum_mask_area", "FLOAT"),
                _widget("mask_feather_px", "INT"),
                _widget("temporal_mask_radius", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("chunk_frames", "INT"),
                _socket("skin_mask", "MASK"),
                _socket("face_plan", "H3_T8_FACE_REFINE_PLAN"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("used_skin_mask", "MASK"),
                _output("rejected_mask", "MASK"),
                _output("difference", "IMAGE"),
                _output("skin_finish_state", "H3_T8_SKIN_FINISH_STATE"),
                _output("report_json", "STRING"),
            ],
            [
                "external_exact",
                "subtle",
                0.35,
                0.9,
                0.35,
                0.0,
                "candidate_only",
                False,
                True,
                0.002,
                0.45,
                3,
                0,
                640,
                4,
            ],
        ),
        _node(
            5,
            "MiniMaxH3SkinFinishFrequencySplitT8Advanced",
            "4. Candidate low + source detail / 候选低频+来源纹理解耦",
            (1060, 80),
            (520, 650),
            [
                _socket("source_frames", "IMAGE"),
                _socket("candidate_frames", "IMAGE"),
                _socket("used_skin_mask", "MASK"),
                _widget("low_frequency_strength", "FLOAT"),
                _widget("source_detail_gain", "FLOAT"),
                _widget("separation_radius_percent", "FLOAT"),
                _widget("maximum_radius_px", "INT"),
                _widget("minimum_mask_area", "FLOAT"),
                _widget("maximum_mask_area", "FLOAT"),
                _widget("maximum_new_clipped_fraction", "FLOAT"),
                _widget("clipping_epsilon", "FLOAT"),
                _widget("chunk_frames", "INT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("frequency_split_candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("effective_mask", "MASK"),
                _output("rejected_mask", "MASK"),
                _output("difference", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [1.0, 1.0, 1.0, 32, 0.0001, 0.50, 0.0005, 1.0 / 255.0, 4, False],
        ),
        _node(
            6,
            "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "5. Required downstream texture/exposure guard / 下游纹理曝光硬门",
            (1640, 80),
            (520, 650),
            [
                _socket("source_frames", "IMAGE"),
                _socket("candidate_frames", "IMAGE"),
                _socket("used_skin_mask", "MASK"),
                _widget("shadow_protection", "FLOAT"),
                _widget("highlight_protection", "FLOAT"),
                _widget("transition_width", "FLOAT"),
                _widget("minimum_texture_ratio", "FLOAT"),
                _widget("minimum_reference_texture", "FLOAT"),
                _widget("maximum_new_clipped_fraction", "FLOAT"),
                _widget("clipping_epsilon", "FLOAT"),
                _widget("texture_radius", "INT"),
                _widget("chunk_frames", "INT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("guarded_candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("effective_mask", "MASK"),
                _output("rejected_mask", "MASK"),
                _output("difference", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [0.10, 0.94, 0.06, 0.78, 0.003, 0.0005, 1.0 / 255.0, 1, 4, False],
        ),
        _node(
            7,
            "PreviewImage",
            "P0 raw candidate / P0原候选",
            (2260, 0),
            (390, 100),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            8,
            "PreviewImage",
            "Frequency-split candidate / 频率解耦候选",
            (2260, 170),
            (390, 100),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            9,
            "PreviewImage",
            "Guarded frequency-split candidate / 硬门后候选",
            (2260, 340),
            (390, 100),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            10,
            "PreviewImage",
            "Selected remains source by default / 默认仍选择原片",
            (2260, 510),
            (390, 100),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            11,
            "PreviewImage",
            "Frequency-split absolute difference / 解耦绝对差异",
            (2720, 170),
            (390, 100),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _note(
            12,
            "START HERE / 正确接线顺序",
            "## 这是独立候选，不修改旧节点\n\n正确顺序：`Skin Finish Advanced -> Frequency Split -> Texture Guard -> Safety Audit/人工审片`。Frequency Split只把P0候选的低频肤色/亮度变化与来源高频纹理重新组合；两个`accept_candidate=false`，最终仍选择原片。",
            (0, 1220),
            (820, 340),
        ),
        _note(
            13,
            "THE MATH / 数学与三个核心参数",
            "## 候选低频 + 来源高频\n\n`low_frequency_strength=1.0`使用候选低频；降到0会逐位回到来源低频。`source_detail_gain=1.0`原样恢复来源已有高频，建议先保持1.0。`separation_radius_percent=1.0`按短边比例计算半径，`maximum_radius_px=32`限制CPU成本。不是把图锐化，也不会生成缺失毛孔。",
            (860, 1220),
            (820, 380),
        ),
        _note(
            14,
            "SOURCE DETAIL / 来源纹理的真实边界",
            "## 来源模糊时仍然模糊\n\n高频层严格来自原片：原片有真实毛孔时可减少磨皮损失；原片只有噪声、压缩块或锐化边缘时，也可能把这些保留下来；原片本来模糊时不会自动变清楚。它不是Face Refine、去模糊或生成式细节恢复。",
            (1720, 1220),
            (820, 360),
        ),
        _note(
            15,
            "FAIL CLOSED / 必须保留后续硬门",
            "## 频率解耦不替代Texture Guard\n\nmask面积异常或新增裁切超限时，本节点按帧回退原片。随后仍应使用Texture Guard检查极暗/极亮与来源相对高频下限；多人交叉、mask泄漏和时间处理突变继续交给Safety Audit。任何PASS都只是机械通过，不等于更美。",
            (2580, 1220),
            (820, 370),
        ),
        _note(
            16,
            "AUDIO + MEMORY / 音频与资源",
            "## 视觉节点不处理音频\n\nAUDIO以同一Python对象旁路，不重采样、EQ或重新编码。默认CPU `chunk_frames=4`，仅分块处理计算；输入与输出完整IMAGE批次仍需存在内存。长片不能据此宣传零内存，文件流路线仍需独立适配和验收。",
            (3440, 1220),
            (820, 330),
        ),
        _note(
            17,
            "SDR + HUMAN REVIEW / 色彩与人工验收",
            "## 目前只支持显示参照SDR RGB\n\n算法在ComfyUI 0..1显示参照RGB上工作，不是线性光；HDR、10-bit和广色域尚未验证。请比较原片、P0候选、Frequency Split和Guarded候选，重点看毛孔、蜡像感、halo、闪烁、眼唇口型与不同肤色；不要用单一高频RMS自动选片。",
            (4300, 1220),
            (850, 390),
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

    connect(1, 0, 4, 0, "IMAGE")
    connect(2, 1, 4, 16, "MASK")
    connect(3, 0, 4, 18, "AUDIO")
    connect(4, 1, 5, 0, "IMAGE")
    connect(4, 0, 5, 1, "IMAGE")
    connect(4, 4, 5, 2, "MASK")
    connect(4, 3, 5, 13, "AUDIO")
    connect(5, 1, 6, 0, "IMAGE")
    connect(5, 0, 6, 1, "IMAGE")
    connect(5, 4, 6, 2, "MASK")
    connect(5, 3, 6, 13, "AUDIO")
    connect(4, 0, 7, 0, "IMAGE")
    connect(5, 0, 8, 0, "IMAGE")
    connect(6, 0, 9, 0, "IMAGE")
    connect(6, 2, 10, 0, "IMAGE")
    connect(5, 6, 11, 0, "IMAGE")

    return {
        "id": "bc8aef1a-498b-4cf5-99dd-d45c36459a5d",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish P2: candidate low frequency + source detail",
                "bounding": [-40, -70, 3100, 900],
                "color": "#5e6f47",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.58, "offset": [100, 80]},
            "workflow_title": "2026-08-25 H3 Skin Finish Frequency Split Advanced EXP",
            "t8_skin_finish": {
                "scope": "P2 non-generative low/high frequency separation",
                "default_selection": "source",
                "automatic_quality_claim": False,
                "audio": "same-object passthrough",
                "recommended_order": [
                    "Skin Finish Advanced",
                    "Frequency Split",
                    "Texture Guard",
                    "Safety Audit or human review",
                ],
            },
        },
        "version": 0.4,
    }


def main() -> int:
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    for path in (OUTPUT, USER_OUTPUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
