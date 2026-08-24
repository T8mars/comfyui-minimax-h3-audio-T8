#!/usr/bin/env python3
from __future__ import annotations

import json

from build_skin_finish_p1_workflow import OUTPUT as P1_OUTPUT
from build_skin_finish_p1_workflow import _node, _note, _output, _socket, _widget


OUTPUT = P1_OUTPUT.with_name(
    "2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json"
)
PARSER_MODEL = "facexlib_parsenet_v0.2.2_pinned"


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadVideo",
            "1. Final untrimmed 8-bit SDR VIDEO / 最终未裁切8位SDR视频",
            (0, 0),
            (420, 130),
            [_widget("file", "COMBO")],
            [_output("VIDEO", "VIDEO")],
            ["replace_with_clear_two_or_three_person_24fps_source.mp4"],
            core=True,
        ),
        _node(
            2,
            "GetVideoComponents",
            "2. Decode exact source frames and audio / 解码同一来源画面音频",
            (500, 0),
            (420, 120),
            [_socket("video", "VIDEO")],
            [
                _output("images", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("fps", "FLOAT"),
                _output("bit_depth", "INT"),
            ],
            [],
            core=True,
        ),
        _node(
            3,
            "CheckpointLoaderSimple",
            "Native ComfyUI SAM3.1 multiplex checkpoint / 原生SAM3.1",
            (0, 250),
            (420, 130),
            [_widget("ckpt_name", "COMBO")],
            [
                _output("MODEL", "MODEL"),
                _output("CLIP", "CLIP"),
                _output("VAE", "VAE"),
            ],
            ["sam3.1_multiplex_fp16.safetensors"],
            core=True,
        ),
        _node(
            4,
            "CLIPTextEncode",
            "Person prompt for SAM3.1 / SAM人物提示",
            (500, 220),
            (420, 180),
            [_widget("text", "STRING"), _socket("clip", "CLIP")],
            [_output("CONDITIONING", "CONDITIONING")],
            ["front-facing person with a visible face"],
            core=True,
        ),
        _node(
            5,
            "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            "3. Track per shot, then offload SAM / 逐镜追踪后卸载SAM",
            (1010, 0),
            (460, 530),
            [
                _socket("frames", "IMAGE"),
                _socket("model", "MODEL"),
                _socket("conditioning", "CONDITIONING"),
                _widget("fps", "FLOAT"),
                _widget("maximum_people", "INT"),
                _widget("detection_threshold", "FLOAT"),
                _widget("detect_interval", "INT"),
                _widget("scene_cut_threshold", "FLOAT"),
                _widget("analysis_max_side", "COMBO"),
                _widget("preview_stride", "INT"),
                _widget("release_policy", "COMBO"),
            ],
            [
                _output("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _output("colored_preview", "IMAGE"),
                _output("report_json", "STRING"),
                _output("shot_count", "INT"),
                _output("shot_local_track_count", "INT"),
            ],
            [24.0, 3, 0.50, 3, 0.28, 640, 8, "offload_sam31_after_track"],
        ),
        _node(
            6,
            "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
            "4. YuNet five-point + ParseNet per tracked face / 逐人五点语义皮肤",
            (1580, 0),
            (570, 810),
            [
                _socket("frames", "IMAGE"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _widget("parser_model", "COMBO"),
                _widget("detection_threshold", "FLOAT"),
                _widget("minimum_face_height_px", "FLOAT"),
                _widget("minimum_detail", "FLOAT"),
                _widget("minimum_person_overlap", "FLOAT"),
                _widget("minimum_track_quality", "FLOAT"),
                _widget("minimum_class_probability", "FLOAT"),
                _widget("feature_protection_px", "INT"),
                _widget("include_neck", "BOOLEAN"),
                _widget("minimum_skin_area_per_face", "FLOAT"),
                _widget("maximum_skin_area_per_frame", "FLOAT"),
                _widget("maximum_alignment_rms", "FLOAT"),
                _widget("minimum_ready_frame_fraction", "FLOAT"),
                _widget("preview_count", "INT"),
                # ComfyUI v1 object-info serializes optional sockets after every
                # required input, even when the v3 schema declares it earlier.
                _socket("identity_assignment", "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT"),
            ],
            [
                _output("semantic_skin_mask", "MASK"),
                _output("mask_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [
                PARSER_MODEL,
                0.45,
                32.0,
                0.010,
                0.20,
                0.10,
                0.55,
                3,
                False,
                0.00005,
                0.35,
                0.08,
                0.50,
                6,
            ],
        ),
        _node(
            7,
            "MiniMaxH3SkinFinishAdvancedT8",
            "5. Source-safe Skin Finish with exact semantic mask / 语义遮罩肤质候选",
            (2280, 0),
            (520, 700),
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
                0.00005,
                0.35,
                2,
                0,
                640,
                4,
            ],
        ),
        _node(
            8,
            "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "6. Source-relative Texture Guard / 源片相对纹理保护",
            (2920, 0),
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
            9,
            "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            "7. Explicit accept + source-audio packet copy / 人工接受后原音频封装",
            (3570, 0),
            (520, 330),
            [
                _socket("source_video", "VIDEO"),
                _socket("processed_frames", "IMAGE"),
                _widget("filename_prefix", "STRING"),
                _widget("crf", "FLOAT"),
                _widget("accept_candidate", "BOOLEAN"),
            ],
            [
                _output("video", "VIDEO"),
                _output("saved_path", "STRING"),
                _output("report_json", "STRING"),
            ],
            ["MiniMaxH3/SkinFinish/multiface_semantic_skin_finish", 18.0, False],
        ),
        _node(10, "PreviewImage", "SAM shot-local tracks / SAM逐镜人物轨迹", (1010, 640), (420, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(11, "PreviewImage", "Green skin; red protected features / 绿皮肤红五官", (1580, 900), (520, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(12, "PreviewImage", "Raw semantic candidate / 语义遮罩原始候选", (3570, 430), (500, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(13, "PreviewImage", "Texture-guarded candidate / 纹理护栏候选", (3570, 610), (500, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(14, "PreviewImage", "Selected source by default / 默认仍为原片", (3570, 790), (500, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(15, "MaskToImage", "Effective semantic skin mask / 实际语义皮肤MASK", (2920, 760), (300, 100), [_socket("mask", "MASK")], [_output("IMAGE", "IMAGE")], [], core=True),
        _node(16, "PreviewImage", "Mask after all gates / 所有门禁后的MASK", (3290, 790), (240, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _note(
            17,
            "START HERE / 正确顺序",
            "## 多人真实语义 Skin Finish\n\n使用最终解码且未裁切的同一条SDR视频：`SAM3.1逐镜追踪 -> YuNet五点对齐 -> 固定ParseNet逐脸解析 -> 与各自人物MASK相交 -> Skin Finish -> Texture Guard -> 人工接受后封装`。Face Refine、Motion Recovery和放大应在本链之前完成；字幕、调色和Tape-FX放在之后。",
            (0, 1180),
            (1050, 370),
        ),
        _note(
            18,
            "MODELS + MEMORY / 模型与内存",
            "## 三阶段不会同时常驻\n\nSAM3.1完成轨迹后按默认策略卸载；语义节点先用固定YuNet检测并释放Detector，再加载CPU ParseNet，执行后`finally`卸载且不持久缓存。ParseNet必须位于`models/facedetection/parsing_parsenet.pth`，大小85,331,193字节、SHA-256以`3d558d8d...20de2`开头，只允许`weights_only=True`安全加载，禁止运行时下载。完整IMAGE和MASK仍占主存，长片应先做已接受片段。",
            (1130, 1180),
            (1100, 410),
        ),
        _note(
            19,
            "ALIGNMENT + PARAMETERS / 对齐与参数",
            "## 每张可靠脸对齐到512解析画布\n\nYuNet五点经左右眼/嘴角x排序后，以LMEDS相似变换对齐到标准FFHQ 512模板；默认归一化残差上限0.08。建议阈值：YuNet 0.45、最小脸高32px、人物重叠0.20、track质量0.10、ParseNet概率0.55、五官保护3px、整帧皮肤上限0.35。不要为了强行出MASK而降低来源、对齐或面积门。",
            (2310, 1180),
            (1100, 410),
        ),
        _note(
            20,
            "IDENTITY + SHOTS / 身份与切镜",
            "## SAM轨迹身份只在单镜头内成立\n\n默认不需要参考图也能逐人物解析；切镜后track会重新编号。若需要在报告中跨镜显示Character_A/B，可把已有MultiFace流程的`identity_assignment`接到可选输入，该映射只是人工/SFace建议标签，不是身份真伪证明，也不会自动改人物肤色。缺关键点、遮挡、歧义、来源hash变化或覆盖不足都会空MASK并ABSTAIN，不传播上一帧脸。",
            (3490, 1180),
            (1100, 430),
        ),
        _note(
            21,
            "QUALITY + AUDIO + ACCEPT / 质量、音频与接受",
            "## 本节点不是去模糊或五官修复\n\n绿区才处理皮肤，红区保护鼻、眼镜、眼眉、嘴唇、头发、饰品和衣服；颈部默认关闭。它只能润饰已有肤色/油光/表面纹理，不能补毛孔、修崩脸、身份或口型。Skin Finish、Texture Guard和Video Finalize均默认不接受；先审SAM轨迹、绿红MASK和候选，再只打开最后所需接受开关。视觉处理不改音频；Finalize只在兼容时逐包复制原音频payload，否则拒绝。",
            (4670, 1180),
            (1120, 450),
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
    connect(3, 1, 4, 1, "CLIP")
    connect(2, 0, 5, 0, "IMAGE")
    connect(3, 0, 5, 1, "MODEL")
    connect(4, 0, 5, 2, "CONDITIONING")
    connect(5, 1, 10, 0, "IMAGE")
    connect(2, 0, 6, 0, "IMAGE")
    connect(5, 0, 6, 1, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(6, 1, 11, 0, "IMAGE")
    connect(2, 0, 7, 0, "IMAGE")
    connect(6, 0, 7, 16, "MASK")
    connect(2, 1, 7, 18, "AUDIO")
    connect(7, 1, 8, 0, "IMAGE")
    connect(7, 0, 8, 1, "IMAGE")
    connect(7, 4, 8, 2, "MASK")
    connect(7, 3, 8, 13, "AUDIO")
    connect(7, 0, 12, 0, "IMAGE")
    connect(8, 0, 13, 0, "IMAGE")
    connect(8, 2, 14, 0, "IMAGE")
    connect(8, 4, 15, 0, "MASK")
    connect(15, 0, 16, 0, "IMAGE")
    connect(1, 0, 9, 0, "VIDEO")
    connect(8, 0, 9, 1, "IMAGE")

    return {
        "id": "676cf762-26e1-4f60-9931-f55c5c12bd0d",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish: SAM3.1 shot tracks -> five-point ParseNet -> review",
                "bounding": [-50, -80, 4200, 1100],
                "color": "#3f6655",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.48, "offset": [110, 90]},
            "workflow_title": (
                "2026-08-24 H3 Skin Finish MultiPerson Semantic Mask Advanced EXP"
            ),
            "t8_skin_finish": {
                "scope": "shot-local SAM3.1 tracks plus per-face five-point ParseNet masks",
                "default_selection": "source",
                "runtime_download": False,
                "parser_persistent_cache": False,
                "identity_assignment_optional": True,
                "identity_is_proof": False,
            },
        },
        "version": 0.4,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
