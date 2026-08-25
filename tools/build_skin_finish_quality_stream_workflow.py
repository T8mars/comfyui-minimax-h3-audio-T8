#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json

from build_skin_finish_p1_workflow import (
    ROOT,
    _node,
    _note,
    _output,
    _socket,
    _widget,
)


OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "17-skin-finish"
    / "2026-08-25_H3_Skin_Finish_Quality_Stream_Advanced_EXP.json"
)
USER_OUTPUT = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "17-skin-finish"
    / OUTPUT.name
)
OIL_CONTROL_OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "17-skin-finish"
    / "2026-08-25_H3_Skin_Finish_Oil_Control_Stream_Advanced_EXP.json"
)
OIL_CONTROL_USER_OUTPUT = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "17-skin-finish"
    / OIL_CONTROL_OUTPUT.name
)


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadVideo",
            "1. Final untrimmed 8-bit SDR VIDEO / 最终未裁切8位SDR视频",
            (0, 0),
            (440, 140),
            [_widget("file", "COMBO")],
            [_output("VIDEO", "VIDEO")],
            ["replace_with_final_untrimmed_24fps_video.mp4"],
            core=True,
        ),
        _node(
            2,
            "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced",
            "2. ParseNet quality stream, bounded RAM / 语义低内存肤质收尾",
            (560, 0),
            (650, 980),
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
                _widget("crop_expansion", "FLOAT"),
                _widget("minimum_class_probability", "FLOAT"),
                _widget("feature_protection_px", "INT"),
                _widget("mask_feather_px", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("low_frequency_strength", "FLOAT"),
                _widget("source_detail_gain", "FLOAT"),
                _widget("separation_radius_percent", "FLOAT"),
                _widget("maximum_radius_px", "INT"),
                _widget("shadow_protection", "FLOAT"),
                _widget("highlight_protection", "FLOAT"),
                _widget("minimum_texture_ratio", "FLOAT"),
                _widget("maximum_temporal_effect_jump", "FLOAT"),
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
                0.30,
                0.95,
                0.25,
                0.45,
                32.0,
                0.010,
                0.55,
                0.28,
                4,
                1.45,
                0.55,
                4,
                0,
                640,
                1.0,
                1.0,
                1.0,
                32,
                0.10,
                0.94,
                0.78,
                0.04,
                2,
                "MiniMaxH3/SkinFinish/quality_stream",
                18.0,
                False,
            ],
        ),
        _note(
            3,
            "START / 只接最终文件",
            "## 最终交付文件专用\n\n把Long Video `Compose Accepted`或Studio `Reel Delivery`的最终、未裁切文件VIDEO直接接入。不要先经过`GetVideoComponents`，也不要对带上下文重叠的各分段分别处理。推荐顺序：AV Decode/Face Refine/Motion Recovery完成并合成最终文件后，再运行本节点。",
            (0, 1100),
            (920, 360),
        ),
        _note(
            4,
            "BOUNDED PIPELINE / 两遍有界处理",
            "## 不物化完整IMAGE或完整语义MASK\n\n第一遍逐帧仅保留YuNet小型脸框/切镜元数据和来源摘要；第二遍按`chunk_frames`加载固定CPU ParseNet，依次运行语义皮肤MASK、Skin Finish、Frequency Split、Texture Guard和跨chunk Safety Audit，并立即单线程H.264编码。仅额外保留上一帧用于检查chunk边界。",
            (980, 1100),
            (980, 410),
        ),
        _note(
            5,
            "SAFE DEFAULTS / 推荐起点",
            "## 先保持轻量参数\n\n推荐：`subtle / amount 0.30 / texture_keep 0.95 / shine 0.25 / crop 1.45 / class probability 0.55 / feather 0 / source detail 1.0 / chunk 2 / CRF 18`。不要为了扩大皮肤面积盲目降低检测或语义概率；远小脸、低细节、无可靠皮肤和硬门失败的帧会精确回退原片。",
            (2020, 1100),
            (980, 420),
        ),
        _note(
            6,
            "SOURCE BY DEFAULT / 默认不执行",
            "## `accept_candidate=false`完全不分析也不写文件\n\n只有显式改为true才生成供人工审片的候选；这不是自动接受开关。true会在加载ParseNet前检查系统可用内存：测得到且低于2048MiB时直接返回原VIDEO、`ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN`，不加载模型、不写文件；2048MiB来自32秒实测约1163MiB进程增量并保留约885MiB余量，不是任意机器安全保证。无可靠脸同样返回原VIDEO。任何异常都会删除partial文件，旧Two Pass Stream节点及旧工作流行为不变。",
            (3060, 1100),
            (940, 500),
        ),
        _note(
            7,
            "AUDIO + FORMAT / 音频和媒体边界",
            "## 兼容音频逐包复制\n\nAAC/MP3/ALAC/AC3/EAC3只复制原压缩packet payload并逐包复核SHA-256，不做解码、降噪或重编码。当前仅接受未旋转、未裁切的8-bit SDR MP4路径；HDR、10-bit、未知codec、几何/文件变化或严格解码失败都会fail closed。",
            (4060, 1100),
            (960, 390),
        ),
        _note(
            8,
            "LIMITS + REVIEW / 能力边界与审片",
            "## 这是非生成式肤质收尾\n\n它只能缓解可靠皮肤区域内的轻度肤色不均、油光和表面观感，并保留来源已有高频。不能把模糊脸变清楚，不能重建身份、五官、口型或缺失毛孔。输出必须检查首/中/尾帧、说话口型、眼唇、闪烁、halo和多人误涂；机械PASS不代表审美更好。",
            (5080, 1100),
            (980, 420),
        ),
    ]

    nodes[0]["outputs"][0]["links"] = [1]
    nodes[1]["inputs"][0]["link"] = 1
    links = [[1, 1, 0, 2, 0, "VIDEO"]]
    return {
        "id": "9c4f332a-93d4-49ab-bd8b-7e2a191a9b25",
        "revision": 0,
        "last_node_id": 8,
        "last_link_id": 1,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Bounded file VIDEO Skin Finish quality route",
                "bounding": [-40, -70, 1300, 1110],
                "color": "#596f56",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.58, "offset": [100, 80]},
            "workflow_title": "2026-08-25 H3 Skin Finish Quality Stream Advanced EXP",
            "t8_skin_finish": {
                "scope": "bounded final-file non-generative quality stream",
                "default_selection": "source",
                "automatic_quality_claim": False,
                "full_image_batch_materialized": False,
                "full_semantic_mask_batch_materialized": False,
                "audio": "verified source packet payload copy",
            },
        },
        "version": 0.4,
    }


def build_oil_control() -> dict:
    workflow = deepcopy(build())
    workflow["id"] = "d3eb092d-714f-4cad-96ec-ce39fd472674"
    quality = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced"
    )
    quality["title"] = "2. Dedicated oil control, bounded RAM / 专用去油低内存处理"
    quality["widgets_values"][0:4] = ["oil_control", 0.35, 0.90, 0.35]
    quality["widgets_values"][24] = "MiniMaxH3/SkinFinish/oil_control_stream"
    quality["widgets_values"][25] = 16.0

    safe_note = next(node for node in workflow["nodes"] if node["id"] == 5)
    safe_note["title"] = "OIL CONTROL / 已验证去油起点"
    safe_note["widgets_values"][0] = (
        "## 专用油光控制参数\n\n"
        "本工作流固定`oil_control / amount 0.35 / texture_keep 0.90 / shine 0.35 / "
        "chunk 2 / CRF 16`。该组合已在v1.0八步LoRA生成的960×544×124中文说话油感近景上完成"
        "一次机械验证：124/124帧处理、零Frequency/Texture/Safety拒绝、音频packet和PCM精确。"
        "这不是自动美颜结论；来源没有明显油光时应保持原片或选择平局/无法判断。"
    )
    limits_note = next(node for node in workflow["nodes"] if node["id"] == 8)
    limits_note["title"] = "REVIEW / 去油不等于磨皮"
    limits_note["widgets_values"][0] = (
        "## 必须观看完整候选后人工决定\n\n"
        "`oil_control`主要压低可靠皮肤区域内相对周围的局部高光，并保留来源已有高频。它不能把模糊脸"
        "变清楚，不能重建身份、五官、口型或缺失毛孔。重点检查额头、鼻梁、双颊的高光是否更自然，"
        "同时检查蜡像感、眼唇变化、闪烁、halo和多人误涂。`accept_candidate=true`只负责生成待审文件，"
        "不代表自动接受。"
    )
    workflow["groups"][0]["title"] = (
        "Bounded final-file Skin Finish dedicated oil-control route"
    )
    workflow["extra"]["workflow_title"] = (
        "2026-08-25 H3 Skin Finish Oil Control Stream Advanced EXP"
    )
    workflow["extra"]["t8_skin_finish"].update(
        {
            "scope": "bounded final-file dedicated oil-control candidate",
            "preset": "oil_control",
            "validated_parameters": {
                "amount": 0.35,
                "texture_keep": 0.90,
                "shine_control": 0.35,
                "chunk_frames": 2,
                "crf": 16.0,
            },
            "representative_review_id": "d4eb04003a44",
        }
    )
    return workflow


def main() -> int:
    publications = (
        (build(), (OUTPUT, USER_OUTPUT)),
        (
            build_oil_control(),
            (OIL_CONTROL_OUTPUT, OIL_CONTROL_USER_OUTPUT),
        ),
    )
    for workflow, paths in publications:
        payload = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
