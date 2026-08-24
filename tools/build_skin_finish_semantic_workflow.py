#!/usr/bin/env python3
from __future__ import annotations

import json

from build_skin_finish_workflow import OUTPUT as P0_OUTPUT
from build_skin_finish_workflow import _node, _note, _output, _socket, _widget


OUTPUT = P0_OUTPUT.with_name(
    "2026-08-24_H3_Skin_Finish_Semantic_Mask_Advanced_EXP.json"
)
PARSER_MODEL = "facexlib_parsenet_v0.2.2_pinned"


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadImage",
            "1. Source IMAGE; replace with final decoded H3 frame batch",
            (0, 0),
            (400, 430),
            [_widget("image", "COMBO"), _widget("upload", "IMAGEUPLOAD")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_source_frame_or_connect_image_batch.png", "image"],
            core=True,
        ),
        _node(
            2,
            "MiniMaxH3FaceRefinePlanT8Advanced",
            "2. Source-bound YuNet face plan / 来源绑定人脸规划",
            (510, 0),
            (500, 690),
            [
                _socket("frames", "IMAGE"),
                _widget("fps", "FLOAT"),
                _widget("detector_mode", "COMBO"),
                _widget("detector_model", "COMBO"),
                _widget("detector_device", "COMBO"),
                _widget("confidence", "FLOAT"),
                _widget("manual_roi_x", "FLOAT"),
                _widget("manual_roi_y", "FLOAT"),
                _widget("manual_roi_width", "FLOAT"),
                _widget("manual_roi_height", "FLOAT"),
                _widget("scene_cut_threshold", "FLOAT"),
                _widget("max_track_jump", "FLOAT"),
                _widget("max_gap_frames", "INT"),
                _widget("smoothing_radius", "INT"),
                _widget("crop_context_scale", "FLOAT"),
                _widget("canvas_size", "COMBO"),
                _widget("require_h3_grid", "BOOLEAN"),
                _widget("analysis_chunk_frames", "INT"),
            ],
            [
                _output("face_plan", "H3_T8_FACE_REFINE_PLAN"),
                _output("crops", "IMAGE"),
                _output("preview", "IMAGE"),
                _output("report_json", "STRING"),
                _output("canvas_width", "INT"),
                _output("canvas_height", "INT"),
                _output("frame_count", "INT"),
            ],
            [
                24.0,
                "local_opencv_yunet",
                "face_detection/face_detection_yunet_2023mar.onnx",
                "cpu",
                0.35,
                0.30,
                0.10,
                0.40,
                0.55,
                0.28,
                0.18,
                4,
                2,
                3.0,
                "auto_512",
                False,
                8,
            ],
        ),
        _node(
            3,
            "MiniMaxH3SkinFinishSemanticMaskT8Advanced",
            "3. Pinned ParseNet semantic skin mask / 固定权重语义皮肤遮罩",
            (1120, 20),
            (540, 560),
            [
                _socket("frames", "IMAGE"),
                _socket("face_plan", "H3_T8_FACE_REFINE_PLAN"),
                _widget("parser_model", "COMBO"),
                _widget("include_neck", "BOOLEAN"),
                _widget("crop_expansion", "FLOAT"),
                _widget("minimum_face_weight", "FLOAT"),
                _widget("minimum_class_probability", "FLOAT"),
                _widget("feature_protection_px", "INT"),
                _widget("minimum_skin_area", "FLOAT"),
                _widget("maximum_skin_area", "FLOAT"),
                _widget("preview_count", "INT"),
            ],
            [
                _output("semantic_skin_mask", "MASK"),
                _output("mask_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [PARSER_MODEL, False, 1.45, 0.35, 0.55, 3, 0.0005, 0.25, 6],
        ),
        _node(
            4,
            "MiniMaxH3SkinFinishAdvancedT8",
            "4. Skin Finish using semantic mask / 使用语义遮罩润饰",
            (1770, 0),
            (510, 690),
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
                0.90,
                0.35,
                0.0,
                "candidate_only",
                False,
                True,
                0.0005,
                0.25,
                2,
                0,
                640,
                4,
            ],
        ),
        _node(
            5,
            "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "5. Source-relative Texture Guard / 源片相对纹理保护",
            (2400, 0),
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
        _node(6, "PreviewImage", "Face plan preview / 人脸框预览", (1120, 660), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(7, "PreviewImage", "Green skin; red protected features / 绿皮肤红五官", (1120, 850), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(8, "PreviewImage", "Raw candidate / 原始候选", (3060, 0), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(9, "PreviewImage", "Guarded candidate / 护栏后候选", (3060, 180), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(10, "PreviewImage", "Selected source by default / 默认选择原片", (3060, 360), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(11, "PreviewImage", "Guarded difference / 护栏后差异", (3060, 540), (390, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(
            12,
            "LoadAudio",
            "Optional original soundtrack / 可选原始音频",
            (0, 520),
            (400, 120),
            [_widget("audio", "COMBO")],
            [_output("AUDIO", "AUDIO")],
            ["replace_with_source_audio.wav"],
            core=True,
        ),
        _note(
            13,
            "START HERE / 接线顺序",
            "## 真正语义遮罩路线\n\n把最终AV Decode或Face Refine后的同一批IMAGE同时接到Face Plan、Semantic Mask和Skin Finish。Semantic Mask输出接Skin Finish的`skin_mask`，其`mask_source`保持`external_exact`。默认两个接受开关都是false，先看绿/红遮罩与候选，再手动接受。",
            (0, 1120),
            (900, 350),
        ),
        _note(
            14,
            "MODEL + SECURITY / 模型与安全",
            "## 固定本地权重，禁止自动下载\n\n文件必须放在`ComfyUI/models/facedetection/parsing_parsenet.pth`。节点检查85,331,193字节和SHA-256 `3d558d8d...20de2`，只使用`torch.load(weights_only=True)`。缺失、大小/hash不符、依赖缺失都会输出空MASK并ABSTAIN/REJECT，不会退化为全屏皮肤。权重不会随节点包分发。",
            (960, 1120),
            (900, 370),
        ),
        _note(
            15,
            "MASK COLOURS + PARAMETERS / 遮罩颜色与参数",
            "## 绿=可处理皮肤，红=受保护语义\n\n默认只选择`skin`，不处理颈部和耳朵；鼻、眼镜、眼、眉、嘴、上下唇、头发、帽子、耳环、项链和衣服均排除。建议`crop_expansion=1.45`、`minimum_class_probability=0.55`、`feature_protection_px=3`。不要为了得到更大面积盲目降低阈值。",
            (1920, 1120),
            (900, 370),
        ),
        _note(
            16,
            "ALIGNMENT + MULTI-PERSON LIMIT / 对齐与多人边界",
            "## 当前face plan只有脸框，没有五点关键点\n\nParseNet使用扩展正方形框，不是仿射对齐；大侧脸、旋转、遮挡和极小脸可能ABSTAIN。这个节点只消费单轨Face Refine Plan，不能冒充多人身份分配；多人需后续将相同parser接入SAM3.1逐人物track/shot计划。切镜和来源hash不匹配会失败关闭。",
            (2880, 1120),
            (900, 390),
        ),
        _note(
            17,
            "MEMORY + QUALITY BOUNDARY / 内存与效果边界",
            "## CPU逐帧解析并执行后卸载\n\n模型仅在CPU加载，逐帧512解析，finally释放且不做持久缓存、不触碰H3/CUDA模型。完整MASK仍占主存，长片要分段。语义遮罩只减少误涂，不证明更美、毛孔真实、去模糊或修复身份；Texture Guard也只是机械硬门，人工审核仍是最终门。AUDIO保持同一对象旁路。",
            (3840, 1120),
            (900, 390),
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

    connect(1, 0, 2, 0, "IMAGE")
    connect(1, 0, 3, 0, "IMAGE")
    connect(2, 0, 3, 1, "H3_T8_FACE_REFINE_PLAN")
    connect(1, 0, 4, 0, "IMAGE")
    connect(3, 0, 4, 16, "MASK")
    connect(12, 0, 4, 18, "AUDIO")
    connect(4, 1, 5, 0, "IMAGE")
    connect(4, 0, 5, 1, "IMAGE")
    connect(4, 4, 5, 2, "MASK")
    connect(4, 3, 5, 13, "AUDIO")
    connect(2, 2, 6, 0, "IMAGE")
    connect(3, 1, 7, 0, "IMAGE")
    connect(4, 0, 8, 0, "IMAGE")
    connect(5, 0, 9, 0, "IMAGE")
    connect(5, 2, 10, 0, "IMAGE")
    connect(5, 6, 11, 0, "IMAGE")

    return {
        "id": "1f6e6178-1841-4e95-8cb8-f95d646a25ba",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish: pinned ParseNet semantic mask -> source-safe candidate",
                "bounding": [-40, -70, 3520, 1050],
                "color": "#4c6b58",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.55, "offset": [100, 80]},
            "workflow_title": "2026-08-24 H3 Skin Finish Semantic Mask Advanced EXP",
            "t8_skin_finish": {
                "scope": "pinned ParseNet semantic skin mask plus Texture Guard",
                "default_selection": "source",
                "runtime_download": False,
                "model_persistent_cache": False,
                "multi_person_identity_claim": False,
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
