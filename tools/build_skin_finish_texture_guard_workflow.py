#!/usr/bin/env python3
from __future__ import annotations

import json

from build_skin_finish_workflow import OUTPUT as P0_OUTPUT
from build_skin_finish_workflow import _node, _note, _output, _socket, _widget


OUTPUT = P0_OUTPUT.with_name(
    "2026-08-24_H3_Skin_Finish_Texture_Guard_Advanced_EXP.json"
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
            "2. Reviewed skin mask with alpha / 已审核皮肤遮罩",
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
            "3. Existing Skin Finish candidate / 现有肤质候选",
            (520, 120),
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
            "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "4. P2 mechanical texture/exposure guard / P2纹理与曝光硬门",
            (1100, 100),
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
            6,
            "PreviewImage",
            "Raw Skin Finish candidate / 原始候选",
            (1750, 0),
            (390, 110),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            7,
            "PreviewImage",
            "Guarded candidate / 护栏后候选",
            (1750, 190),
            (390, 110),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            8,
            "PreviewImage",
            "Selected remains source by default / 默认仍选择原片",
            (1750, 380),
            (390, 110),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            9,
            "PreviewImage",
            "Guarded absolute difference / 护栏后差异",
            (1750, 570),
            (390, 110),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _note(
            10,
            "START HERE / 使用顺序",
            "## P2机械护栏，不是自动美颜\n\n1. 先由现有Skin Finish Advanced产生`candidate/source/used_skin_mask`。\n2. 再接Texture Guard；它只减少风险，不证明画面更美。\n3. 默认`accept_candidate=false`，所以selected仍是原片。\n4. 看候选和report后再手动接受。旧Skin Finish节点与旧工作流均未修改。",
            (0, 1220),
            (850, 330),
        ),
        _note(
            11,
            "EXPOSURE PROTECTION / 曝光保护",
            "## 默认保护源片极端亮暗\n\n- `shadow_protection=0.10`：源片亮度低于该值的区域保留。\n- `highlight_protection=0.94`：接近饱和高光保留。\n- `transition_width=0.06`：用平滑过渡避免硬边。\n- 这是SDR显示值门禁，不等于线性光、HDR或广色域处理。Log/HDR素材先不要使用。",
            (900, 1220),
            (850, 330),
        ),
        _note(
            12,
            "TEXTURE + CLIPPING / 纹理与裁切",
            "## 两个逐帧硬失败条件\n\n- `minimum_texture_ratio=0.78`比较候选与源片遮罩内高通RMS；低于下限则整帧回退。它只能拦截过度磨皮，不能证明毛孔真实，也可能把噪声算作高频。\n- `maximum_new_clipped_fraction=0.0005`限制候选新增的黑/白裁切像素；超过就整帧回退。\n- 失败帧不会保留一半处理结果。",
            (1800, 1220),
            (850, 350),
        ),
        _note(
            13,
            "AUDIO + MEMORY / 音频与内存",
            "## 音频完全旁路\n\n- AUDIO以同一对象透传；视觉护栏不会重采样、EQ或重新编码音频。\n- 默认CPU `chunk_frames=4`，但输入IMAGE批次本身仍由上游持有，不能宣传为零内存。\n- 长文件优先用现有Two Pass Video Stream；本工作流用于已经解码并审核的候选。",
            (2700, 1220),
            (850, 320),
        ),
        _note(
            14,
            "KNOWN LIMITS / 不能解决",
            "## 明确边界\n\n- 不识别皮肤语义、不保护头发/眼唇语义、不做人脸身份判断。\n- 不去模糊、不锐化、不补毛孔、不修五官、不处理口型。\n- 纹理RMS和裁切率只是硬拦截器，不能作为自动选片分数。\n- 多人、遮挡、切镜与不同肤色仍需后续semantic parser和人工验收。",
            (3600, 1220),
            (850, 330),
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
    connect(4, 0, 6, 0, "IMAGE")
    connect(5, 0, 7, 0, "IMAGE")
    connect(5, 2, 8, 0, "IMAGE")
    connect(5, 6, 9, 0, "IMAGE")

    return {
        "id": "86d3b35e-d1c9-4e3c-96e0-d7e4eb385113",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish P2: source-relative texture and exposure guard",
                "bounding": [-40, -70, 2200, 900],
                "color": "#5e6f47",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.62, "offset": [100, 80]},
            "workflow_title": "2026-08-24 H3 Skin Finish Texture Guard Advanced EXP",
            "t8_skin_finish": {
                "scope": "P2 mechanical source-relative guard",
                "default_selection": "source",
                "automatic_quality_claim": False,
                "audio": "same-object passthrough",
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
