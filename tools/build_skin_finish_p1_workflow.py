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
    / "2026-08-24_H3_Skin_Finish_MultiPerson_Video_Finalize_Advanced_EXP.json"
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
            "1. Untrimmed 8-bit SDR source VIDEO / 未裁切8位SDR原视频",
            (0, 0),
            (410, 130),
            [_widget("file", "COMBO")],
            [_output("VIDEO", "VIDEO")],
            ["replace_with_untrimmed_24fps_source.mp4"],
            core=True,
        ),
        _node(
            2,
            "GetVideoComponents",
            "2. Decode source frames; audio remains source-owned / 解码画面",
            (480, 0),
            (410, 120),
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
            (0, 260),
            (410, 130),
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
            (480, 220),
            (410, 180),
            [_widget("text", "STRING"), _socket("clip", "CLIP")],
            [_output("CONDITIONING", "CONDITIONING")],
            ["front-facing person with a visible face"],
            core=True,
        ),
        _node(
            5,
            "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            "3. Track once, then offload SAM / 只追踪一次并卸载SAM",
            (980, 0),
            (440, 530),
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
            "MiniMaxH3SkinFinishMultiPersonT8Advanced",
            "4. Shot/person-aware source-safe candidate / 多人分镜肤质候选",
            (1530, 0),
            (520, 860),
            [
                _socket("frames", "IMAGE"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _widget("absolute_start_frame", "INT"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("detection_threshold", "FLOAT"),
                _widget("minimum_face_height_px", "FLOAT"),
                _widget("minimum_detail", "FLOAT"),
                _widget("bbox_ema_alpha", "FLOAT"),
                _widget("max_missing_frames", "INT"),
                _widget("protect_features", "BOOLEAN"),
                _widget("include_neck", "BOOLEAN"),
                _widget("maximum_overlap_frames", "INT"),
                _widget("mask_feather_px", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("chunk_frames", "INT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("previous_state", "H3_T8_SKIN_FINISH_SEQUENCE_STATE"),
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
                _output("sequence_state", "H3_T8_SKIN_FINISH_SEQUENCE_STATE"),
                _output("report_json", "STRING"),
                _output("emitted_absolute_start", "INT"),
                _output("emitted_frame_count", "INT"),
            ],
            [
                0,
                "subtle",
                0.35,
                0.90,
                0.35,
                0.45,
                24.0,
                0.010,
                0.55,
                2,
                True,
                False,
                8,
                3,
                640,
                4,
                False,
            ],
        ),
        _node(
            7,
            "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            "5. Explicit accept + exact source-audio packet copy / 审核后原音频封装",
            (2180, 0),
            (500, 330),
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
            ["MiniMaxH3/SkinFinish/multiface_skin_finish", 18.0, False],
        ),
        _node(
            8,
            "PreviewImage",
            "SAM shot-local color tracks / SAM分镜分色轨迹",
            (980, 620),
            (420, 120),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            9,
            "PreviewImage",
            "Skin Finish candidate (not selected by default) / 肤质候选",
            (2180, 430),
            (500, 120),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            10,
            "MaskToImage",
            "Used face-skin proxy mask / 实际使用肤区代理",
            (2180, 620),
            (310, 110),
            [_socket("mask", "MASK")],
            [_output("IMAGE", "IMAGE")],
            [],
            core=True,
        ),
        _node(
            11,
            "PreviewImage",
            "Review mask before acceptance / 接受前检查遮罩",
            (2560, 620),
            (420, 120),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _note(
            12,
            "START HERE / 用途与顺序",
            "## 多人 Skin Finish P1\n\n`VIDEO -> 原生SAM3.1追踪一次 -> Multi-Person Skin Finish -> 人工审片 -> Video Finalize`。本节点只做非生成式低频肤色/油光收尾；不修五官、不去模糊、不锐化、不补身份、不修口型。Face Refine/Motion Recovery应在它之前完成。",
            (0, 980),
            (900, 310),
        ),
        _note(
            13,
            "TRACK + MASK / 追踪与遮罩",
            "## 不重复加载SAM\n\nMulti-Person节点直接消费上游`track_plan`里的压缩逐帧人物mask，SAM追踪后默认选择性卸载。每帧再用CPU YuNet定位脸部，并与人物mask相交。眼眉/睫毛、鼻孔、嘴唇/牙齿区域近似排除；`include_neck=false`默认不处理颈部。该mask是保守代理，不是像素级语义皮肤解析。",
            (980, 980),
            (980, 350),
        ),
        _note(
            14,
            "PARAMETERS / 安全参数",
            "## 推荐起点\n\n`subtle / amount 0.35 / texture_keep 0.90 / shine 0.35`；YuNet threshold 0.45；最小脸高24px；detail 0.010；bbox EMA 0.55；丢失最多2帧衰减；feather 3；proxy 640；CPU chunk 4。多人只共享同一套中性色参数，不自动按人物改色相/饱和度，`tone_adjust`在内部固定为0。",
            (2060, 980),
            (980, 350),
        ),
        _note(
            15,
            "LONG VIDEO STATE / 长视频续块",
            "## 分块合同\n\n完整批次保持`absolute_start_frame=0`且不接state。长视频可复制Multi-Person节点：首块从0开始，后块连接前块`sequence_state`并填写真实绝对起帧；允许最多8帧重叠。重叠来源代理必须匹配并会被丢弃，禁止跳帧空洞。状态只延续分镜内bbox EMA，不对RGB做时间平均。CPU chunk限制临时张量，但上游`GetVideoComponents`仍会持有完整IMAGE批次。",
            (3140, 980),
            (1050, 390),
        ),
        _note(
            16,
            "AUDIO + ACCEPT / 原音频与接受",
            "## 两道人工门禁\n\nMulti-Person与Video Finalize的`accept_candidate`默认都是false。先看候选、轨迹和mask；最终要保存候选时，Video Finalize应直接连接candidate并单独打开accept。它逐帧重编码画面，但兼容的AAC/MP3/ALAC/AC3/EAC3音频包payload逐包复制并做SHA-256复核；裁切VIDEO、旋转、10-bit/HDR、未知音频codec或包漂移都会拒绝，不会静默重编码音频。",
            (4310, 980),
            (1080, 390),
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
    connect(5, 1, 8, 0, "IMAGE")
    connect(2, 0, 6, 0, "IMAGE")
    connect(5, 0, 6, 1, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(2, 1, 6, 20, "AUDIO")
    connect(6, 0, 9, 0, "IMAGE")
    connect(6, 4, 10, 0, "MASK")
    connect(10, 0, 11, 0, "IMAGE")
    connect(1, 0, 7, 0, "VIDEO")
    connect(6, 0, 7, 1, "IMAGE")

    return {
        "id": "7e810af0-b0c5-4ef9-bf88-a4a1af99d496",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish P1: reuse SAM tracks, review, packet-copy source audio",
                "bounding": [-50, -80, 3100, 970],
                "color": "#315a78",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.55, "offset": [120, 100]},
            "workflow_title": "2026-08-24 H3 Skin Finish MultiPerson Video Finalize Advanced EXP",
            "t8_skin_finish": {
                "scope": "P1 tracked multi-person SDR candidate plus source-audio packet-copy finalization",
                "sam31_reloaded": False,
                "default_selection": "source",
                "hdr_supported": False,
            },
        },
        "version": 0.4,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
