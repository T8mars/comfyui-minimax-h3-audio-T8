#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json

from build_skin_finish_p1_workflow import _node, _note, _output, _socket, _widget
from build_skin_finish_per_person_workflow import (
    PARSER_MODEL,
    PROJECT_ROOT,
    USER_OUTPUT as PER_PERSON_USER_OUTPUT,
    build as build_per_person,
)


OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "workflows"
    / "17-skin-finish"
    / "2026-08-25_H3_Skin_Finish_Studio_Timeline_Advanced_EXP.json"
)
USER_OUTPUT = PER_PERSON_USER_OUTPUT.with_name(OUTPUT.name)


def _keyframe_node(
    node_id: int,
    title: str,
    position: tuple[int, int],
    *,
    shot: int,
    frame: int,
    preset: str,
    amount: float,
    texture: float,
    shine: float,
    tone: float,
) -> dict:
    return _node(
        node_id,
        "MiniMaxH3SkinFinishTimelineKeyframeT8Advanced",
        title,
        position,
        (530, 530),
        [
            _socket("studio_timeline", "H3_T8_STUDIO_TIMELINE"),
            _widget("selector_type", "COMBO"),
            _widget("selector", "STRING"),
            _widget("studio_shot_index", "INT"),
            _widget("frame_in_shot", "INT"),
            _widget("interpolation_to_next", "COMBO"),
            _widget("preset", "COMBO"),
            _widget("amount", "FLOAT"),
            _widget("texture_keep", "FLOAT"),
            _widget("shine_control", "FLOAT"),
            _widget("tone_adjust", "FLOAT"),
            _socket("previous_plan", "H3_T8_SKIN_FINISH_TIMELINE_PLAN"),
        ],
        [
            _output("timeline_plan", "H3_T8_SKIN_FINISH_TIMELINE_PLAN"),
            _output("report_json", "STRING"),
        ],
        ["global", "*", shot, frame, "smoothstep", preset, amount, texture, shine, tone],
    )


def build() -> dict:
    workflow = deepcopy(build_per_person())
    replaced = {14, 15, 16}
    nodes = [node for node in workflow["nodes"] if node["id"] not in replaced]
    notes = {node["id"]: node for node in nodes if node["id"] in {27, 28, 29, 30, 31, 32, 35}}
    for note in notes.values():
        nodes.remove(note)

    nodes.extend([
        _node(
            14,
            "MiniMaxH3StudioTimelineT8Advanced",
            "7. Exact creative shots and frame ranges / 精确创作镜头时间轴",
            (2020, 900),
            (650, 550),
            [
                _widget("project_id", "STRING"),
                _widget("shots_json", "STRING"),
                _widget("default_backend", "COMBO"),
                _widget("default_duration_seconds", "FLOAT"),
                _widget("default_aspect_ratio", "STRING"),
                _widget("base_seed", "INT"),
                _widget("seed_policy", "COMBO"),
                _widget("split_long_shots", "BOOLEAN"),
                _widget("strict_exact_dialogue", "BOOLEAN"),
                _socket("cast", "H3_T8_UNIFIED_CAST"),
                _socket("sound_canvas", "H3_T8_SOUND_CANVAS"),
            ],
            [
                _output("timeline", "H3_T8_STUDIO_TIMELINE"),
                _output("timeline_json", "STRING"),
            ],
            [
                "skin_finish_timeline_example",
                (
                    '[{"id":"shot_0","prompt":"First accepted shot",'
                    '"duration_seconds":0.9166666667},{"id":"shot_1",'
                    '"prompt":"Second accepted shot","duration_seconds":0.9166666667}]'
                ),
                "minimax_h3",
                0.9166666667,
                "16:9",
                42,
                "increment",
                True,
                True,
            ],
        ),
        _keyframe_node(
            15,
            "8A. Shot 0 frame 0: conservative start / 镜头0保守起点",
            (2740, 900),
            shot=0,
            frame=0,
            preset="subtle",
            amount=0.25,
            texture=0.95,
            shine=0.25,
            tone=0.0,
        ),
        _keyframe_node(
            16,
            "8B. Shot 0 frame 21: smooth stronger finish / 镜头0平滑加强",
            (3330, 900),
            shot=0,
            frame=21,
            preset="subtle",
            amount=0.40,
            texture=0.92,
            shine=0.40,
            tone=0.0,
        ),
        _keyframe_node(
            36,
            "8C. Shot 1 frame 0: new cut, no carry-over / 镜头1重新起算",
            (3920, 900),
            shot=1,
            frame=0,
            preset="oil_control",
            amount=0.38,
            texture=0.92,
            shine=0.45,
            tone=0.0,
        ),
        _keyframe_node(
            37,
            "8D. Shot 1 frame 21: ease back / 镜头1缓和收尾",
            (4510, 900),
            shot=1,
            frame=21,
            preset="subtle",
            amount=0.28,
            texture=0.94,
            shine=0.30,
            tone=0.0,
        ),
        _node(
            38,
            "MiniMaxH3SkinFinishTimelineT8Advanced",
            "9. Apply source-bound keys; never cross cuts / 按时间轴逐人物执行",
            (2740, 0),
            (650, 800),
            [
                _socket("frames", "IMAGE"),
                _socket("studio_timeline", "H3_T8_STUDIO_TIMELINE"),
                _socket("timeline_plan", "H3_T8_SKIN_FINISH_TIMELINE_PLAN"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _socket("semantic_skin_mask", "MASK"),
                _widget("semantic_report_json", "STRING"),
                _widget("execution_mode", "COMBO"),
                _widget("accept_candidate", "BOOLEAN"),
                _widget("chunk_frames", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("preview_count", "INT"),
                _socket("identity_assignment", "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("used_skin_mask", "MASK"),
                _output("rejected_skin_mask", "MASK"),
                _output("ownership_preview", "IMAGE"),
                _output("skin_finish_state", "H3_T8_SKIN_FINISH_STATE"),
                _output("report_json", "STRING"),
            ],
            ["", "candidate_only", False, 2, 640, 6],
        ),
        _note(
            27,
            "START HERE / 使用顺序",
            "## Skin Finish Studio Timeline\n\n这条路线用于最终解码并完成Face Refine/Motion Recovery后的SDR视频：`Studio Timeline -> Skin关键帧计划 -> SAM人物归属 -> Skin Finish -> Texture Guard -> Safety Audit -> 人工接受`。示例是两个22帧镜头，共44帧；替换视频后必须同步改shots_json，使Timeline总帧数与解码帧数完全一致，否则节点安全回原片。",
            (0, 1680),
            (1120, 430),
        ),
        _note(
            28,
            "TWO SHOT DOMAINS / 两套镜头编号",
            "## Studio shot不是SAM shot\n\n`studio_shot_index + frame_in_shot`只决定参数在哪个创作镜头、哪一帧变化；`shot_track=0:1`是SAM3.1的人物轨迹编号。两者刻意独立，不能因为数字相同就认为是一套编号。人物优先级固定为`SAM shot:track > character_id > global > source`。换镜后先审核SAM彩色轨迹和Character绑定。",
            (1200, 1680),
            (1180, 450),
        ),
        _note(
            29,
            "INTERPOLATION / 插值规则",
            "## 只在同一Studio镜头内插值\n\n`hold`保持左关键帧；`linear`匀速；`smoothstep`两端更缓，推荐用于克制变化。amount、texture_keep、shine_control、tone_adjust是连续值；preset是分类值，不做数值混合，在到达下一关键帧时才切换。首个关键帧之前保持首值，末键之后保持末值；绝不跨切镜延续。",
            (2460, 1680),
            (1180, 470),
        ),
        _note(
            30,
            "TARGETING / 人物与全局关键帧",
            "## global只是最低优先级\n\n示例用global让所有已识别人物共享时间曲线。需要逐人物时，把关键帧selector_type改成`character_id`并填已审核Character；某条SAM轨迹需要特殊处理时用`shot_track`。同一人物同一镜头可以有多个时间键，但同一selector同一帧不能重复。没有可用关键帧、人物重叠歧义或未绑定人物都保持bit-exact source。",
            (3720, 1680),
            (1180, 470),
        ),
        _note(
            31,
            "PARAMETERS / 参数建议",
            "## 避免时间泵动\n\n真人从`subtle, amount 0.25~0.40, texture_keep 0.90~0.95, shine 0.25~0.45, tone 0`开始。相邻关键帧差值宜小；高频快速来回变化会造成亮度/肤色呼吸，节点不会自动替你判美。明显油光才切oil_control，切preset最好放在镜头边界或动作遮挡点。它不修脸、不去模糊，也不生成毛孔。",
            (4980, 1680),
            (1120, 450),
        ),
        _note(
            32,
            "MEMORY + AUDIO + ACCEPT / 内存、音频与接受",
            "## 默认仍然选择原片\n\n执行器按CPU小chunk搬运，但为了逐帧准确插值会逐帧处理，长片仍可能较慢。AUDIO返回同一对象，视觉关键帧不改声音。Timeline执行器、Texture Guard、Safety Audit和Finalizer的接受开关默认都是false；先看整段口型、眼唇、侧脸、遮挡、切镜和肤色闪烁，再只打开最终需要的接受门。",
            (6180, 1680),
            (1160, 450),
        ),
        _note(
            35,
            "FINAL SAFETY AUDIT / 最终安全审计",
            "## 机械通过不等于更好看\n\nSafety Audit检查mask外变化、SAM人物越界、多人重叠归属、时间处理跳变和AUDIO PCM；失败时gated_candidate回到source。绿色只代表机械合同通过，红色表示硬失败。关键帧变化点仍必须人工观看，因为指标不能判断自然肤色、身份、嘴唇是否好看。HDR、10-bit和广色域不在本工作流合同内。",
            (7420, 1680),
            (1160, 460),
        ),
    ])

    node_map = {node["id"]: node for node in nodes}
    for node in nodes:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = None

    kept_links = [
        link
        for link in workflow["links"]
        if link[1] not in replaced and link[3] not in replaced
    ]
    links: list[list] = []

    def connect(source: int, source_slot: int, target: int, target_slot: int, value_type: str):
        link_id = len(links) + 1
        links.append([link_id, source, source_slot, target, target_slot, value_type])
        output = node_map[source]["outputs"][source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
        node_map[target]["inputs"][target_slot]["link"] = link_id

    for _old_id, source, source_slot, target, target_slot, value_type in kept_links:
        connect(source, source_slot, target, target_slot, value_type)

    connect(14, 0, 15, 0, "H3_T8_STUDIO_TIMELINE")
    connect(14, 0, 16, 0, "H3_T8_STUDIO_TIMELINE")
    connect(15, 0, 16, 11, "H3_T8_SKIN_FINISH_TIMELINE_PLAN")
    connect(14, 0, 36, 0, "H3_T8_STUDIO_TIMELINE")
    connect(16, 0, 36, 11, "H3_T8_SKIN_FINISH_TIMELINE_PLAN")
    connect(14, 0, 37, 0, "H3_T8_STUDIO_TIMELINE")
    connect(36, 0, 37, 11, "H3_T8_SKIN_FINISH_TIMELINE_PLAN")
    connect(2, 0, 38, 0, "IMAGE")
    connect(14, 0, 38, 1, "H3_T8_STUDIO_TIMELINE")
    connect(37, 0, 38, 2, "H3_T8_SKIN_FINISH_TIMELINE_PLAN")
    connect(5, 0, 38, 3, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(13, 0, 38, 4, "MASK")
    connect(13, 2, 38, 5, "STRING")
    connect(12, 0, 38, 11, "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
    connect(2, 1, 38, 12, "AUDIO")
    connect(38, 1, 17, 0, "IMAGE")
    connect(38, 0, 17, 1, "IMAGE")
    connect(38, 4, 17, 2, "MASK")
    connect(38, 3, 17, 13, "AUDIO")
    connect(38, 6, 22, 0, "IMAGE")
    connect(38, 0, 23, 0, "IMAGE")
    connect(38, 1, 33, 0, "IMAGE")

    workflow["nodes"] = sorted(nodes, key=lambda node: node["id"])
    workflow["links"] = links
    workflow["last_node_id"] = max(node_map)
    workflow["last_link_id"] = len(links)
    workflow["groups"] = [{
        "id": 1,
        "title": "Skin Finish: Studio Timeline keyframes -> per-person source-safe finish",
        "bounding": [-50, -80, 5250, 1680],
        "color": "#536d4d",
        "font_size": 24,
        "flags": {},
    }]
    workflow["extra"]["workflow_title"] = (
        "2026-08-25 H3 Skin Finish Studio Timeline Advanced EXP"
    )
    workflow["extra"]["t8_skin_finish"] = {
        "scope": "Studio-shot-local Skin Finish parameter keyframes with independent SAM identity routing",
        "timeline_frame_contract": "decoded frames must exactly equal Studio Timeline total_frames",
        "routing_precedence": "sam_shot_track_over_character_id_over_global_over_source",
        "no_cross_shot_interpolation": True,
        "default_selection": "source",
        "automatic_accept": False,
        "parser_model": PARSER_MODEL,
    }
    return workflow


def main() -> int:
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    for path in (OUTPUT, USER_OUTPUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(OUTPUT)
    print(USER_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
