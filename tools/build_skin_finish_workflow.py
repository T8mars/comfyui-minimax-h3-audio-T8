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
    / "2026-08-24_H3_Skin_Finish_External_Mask_Advanced_EXP.json"
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


def _note(node_id, title, text, pos, size=(720, 300)):
    return _node(
        node_id,
        "MarkdownNote",
        title,
        pos,
        size,
        [],
        [],
        [text],
        core=True,
    )


def build() -> dict:
    nodes = []
    nodes.append(
        _node(
            1,
            "LoadImage",
            "1. Source frame or replace with decoded IMAGE batch / 来源画面或视频帧批次",
            (0, 0),
            (390, 430),
            [_widget("image", "COMBO"), _widget("upload", "IMAGEUPLOAD")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_source_frame.png", "image"],
            core=True,
        )
    )
    nodes.append(
        _node(
            2,
            "LoadImage",
            "2. Skin mask with alpha / 带Alpha的肤质遮罩",
            (0, 500),
            (390, 430),
            [_widget("image", "COMBO"), _widget("upload", "IMAGEUPLOAD")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_skin_mask_rgba.png", "image"],
            core=True,
        )
    )
    nodes.append(
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
        )
    )
    nodes.append(
        _node(
            4,
            "MiniMaxH3SkinFinishT8",
            "Basic: conservative source-safe candidate / 基础肤质候选",
            (520, 0),
            (430, 440),
            [
                _socket("frames", "IMAGE"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("tone_adjust", "FLOAT"),
                _widget("execution_mode", "COMBO"),
                _widget("chunk_frames", "INT"),
                _socket("skin_mask", "MASK"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("used_skin_mask", "MASK"),
                _output("report_json", "STRING"),
            ],
            ["subtle", 0.35, 0.9, 0.35, 0.0, "candidate_only", 4],
        )
    )
    nodes.append(
        _node(
            5,
            "MiniMaxH3SkinFinishAdvancedT8",
            "Advanced: masks, gates, exact passthrough / 高级遮罩与硬门禁",
            (520, 520),
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
        )
    )
    nodes.append(
        _node(
            6,
            "MiniMaxH3SkinFinishPreviewAuditT8Advanced",
            "Preview/Audit: left source, right candidate / 预览审计：左原图右候选",
            (1120, 420),
            (500, 520),
            [
                _socket("source_frames", "IMAGE"),
                _socket("candidate_frames", "IMAGE"),
                _socket("used_mask", "MASK"),
                _socket("rejected_mask", "MASK"),
                _socket("skin_finish_state", "H3_T8_SKIN_FINISH_STATE"),
                _widget("gate_report_json", "STRING"),
                _widget("frame_index", "INT"),
                _widget("comparison_position", "FLOAT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("audio_source", "AUDIO"),
                _socket("audio_passthrough", "AUDIO"),
            ],
            [
                _output("selected", "IMAGE"),
                _output("split_comparison", "IMAGE"),
                _output("source_crop", "IMAGE"),
                _output("candidate_crop", "IMAGE"),
                _output("mask_preview", "IMAGE"),
                _output("difference_preview", "IMAGE"),
                _output("plus_minus_2_loop", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("review_report_json", "STRING"),
            ],
            ["", 0, 0.5, False],
        )
    )

    preview_titles = [
        "Basic candidate (not auto-accepted)",
        "Split comparison: source left / candidate right",
        "Full-resolution source crop",
        "Full-resolution candidate crop",
        "Mask audit: green used / red rejected",
        "Amplified absolute difference",
        "±2-frame split loop for flicker review",
    ]
    for index, title in enumerate(preview_titles, start=7):
        row = index - 7
        nodes.append(
            _node(
                index,
                "PreviewImage",
                title,
                (1740 + (row % 2) * 430, (row // 2) * 260),
                (390, 110),
                [_socket("images", "IMAGE")],
                [_output("images", "IMAGE")],
                [],
                core=True,
            )
        )

    nodes.extend(
        [
            _note(
                14,
                "START HERE / 从这里开始",
                "## Skin Finish P0：非生成式肤质收尾\n\n1. `frames`可接H3解码后的完整IMAGE批次；本示例LoadImage只是最小可导入入口。\n2. 默认只产生候选，`accept_candidate=false`时`selected`仍是原片。\n3. 先看分屏、全分辨率裁切、遮罩和±2帧循环，再决定是否接受。\n4. 它处理低频肤色不均和油光，不负责修脸、锐化模糊、补身份、补毛孔或修口型。",
                (0, 1210),
                (920, 330),
            ),
            _note(
                15,
                "MASK CONTRACT / 遮罩合同",
                "## 没有可靠遮罩就ABSTAIN\n\n- 当前P0支持`external_exact`和Advanced节点的`face_refine_plan`。\n- 本示例第二个LoadImage应使用带Alpha的RGBA皮肤遮罩；普通不带Alpha的JPG会得到空MASK并安全回退原片。\n- 使用Face Refine Plan时，把`mask_source`改为`face_refine_plan`并连接同一来源帧生成的plan。该路线只是保守脸区代理，不是语义皮肤解析。\n- 绿=实际使用，红=被面积门禁拒绝。遮罩外RGB以及Alpha/额外通道必须精确不变。",
                (960, 1210),
                (920, 350),
            ),
            _note(
                16,
                "PARAMETERS / 参数建议",
                "## 安全起点\n\n- `preset=subtle`、`amount=0.35`、`texture_keep=0.90`、`shine_control=0.35`。\n- 油光明显可先换`oil_control`，不要先把amount拉满。\n- `mask_feather_px=3`只在原遮罩内部软化，不会向外扩张。\n- `chunk_frames=4`、`proxy_long_side=640`是CPU有界默认值；它降低工作内存，不代表输入IMAGE批次本身是流式解码。\n- 124帧先检查首/中/尾以及±2帧循环，防止肤色呼吸和边缘闪烁。",
                (1920, 1210),
                (920, 350),
            ),
            _note(
                17,
                "AUDIO + ACCEPT / 音频与接受",
                "## 音频不参与处理\n\n- Advanced节点只原对象透传AUDIO；Preview/Audit同时比较输入与透传PCM合同。\n- 两侧音频缺一或PCM不一致时，即使打开`accept_candidate`也不会选择候选。\n- 要最终采用候选，请在Preview/Audit确认画面后手动打开其`accept_candidate=true`并重新排队。\n- Advanced节点自己的accept开关也默认为false，避免误把候选当原片。",
                (2880, 1210),
                (920, 330),
            ),
            _note(
                18,
                "KNOWN LIMITS / 已知边界",
                "## 目前不承诺\n\n- 不自动下载或捆绑人脸解析权重；缺少可靠mask就保持原片。\n- 不做多人语义归属、跨镜头人物对应、HDR/Log/广色域或生成式皮肤重建。\n- 对超暗、遮挡、极端色偏、强运动模糊和压缩破损必须人工复核。\n- 本节点是候选生成与审计工具，不是自动美颜开关。",
                (3840, 1210),
                (820, 310),
            ),
        ]
    )

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
    connect(2, 1, 4, 8, "MASK")
    connect(1, 0, 5, 0, "IMAGE")
    connect(2, 1, 5, 16, "MASK")
    connect(3, 0, 5, 18, "AUDIO")
    connect(4, 0, 7, 0, "IMAGE")
    connect(5, 1, 6, 0, "IMAGE")
    connect(5, 0, 6, 1, "IMAGE")
    connect(5, 4, 6, 2, "MASK")
    connect(5, 5, 6, 3, "MASK")
    connect(5, 7, 6, 4, "H3_T8_SKIN_FINISH_STATE")
    connect(5, 8, 6, 5, "STRING")
    connect(3, 0, 6, 9, "AUDIO")
    connect(5, 3, 6, 10, "AUDIO")
    for output_slot, preview_id in enumerate(range(8, 14), start=1):
        connect(6, output_slot, preview_id, 0, "IMAGE")

    return {
        "id": "4df0c414-90a8-4a8f-ac3e-b29c9fe83e8f",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish P0: source-safe candidate and human audit",
                "bounding": [-40, -70, 2250, 1260],
                "color": "#3f6f5f",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.55, "offset": [120, 100]},
            "workflow_title": "2026-08-24 H3 Skin Finish External Mask Advanced EXP",
            "t8_skin_finish": {
                "scope": "P0 non-generative SDR candidate",
                "default_selection": "source",
                "audio": "exact passthrough only",
                "representative_validation": "1088x544x124; one run only",
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
