from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-08-09_H3_Long_Video_Accepted_22F_EXP.json"
)
OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-08-22_H3_Enhance_A_Video_Long_Video_Accepted_22F_Stock20_Advanced_EXP.json"
)
INSTALLED = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "04-long-video"
    / OUTPUT.name
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == int(node_id))


def _remove_link(workflow: dict, link_id: int) -> None:
    link = next(value for value in workflow["links"] if int(value[0]) == int(link_id))
    _source_id, source_slot, target_id, target_slot = link[1:5]
    source = _node(workflow, link[1])
    target = _node(workflow, target_id)
    source_links = source["outputs"][source_slot].get("links") or []
    source["outputs"][source_slot]["links"] = [
        value for value in source_links if int(value) != int(link_id)
    ] or None
    target["inputs"][target_slot]["link"] = None
    workflow["links"] = [
        value for value in workflow["links"] if int(value[0]) != int(link_id)
    ]


def _append_link(
    workflow: dict,
    source: dict,
    source_slot: int,
    target: dict,
    target_slot: int,
    link_type: str,
) -> int:
    link_id = int(workflow["last_link_id"]) + 1
    workflow["last_link_id"] = link_id
    workflow["links"].append(
        [link_id, source["id"], source_slot, target["id"], target_slot, link_type]
    )
    source["outputs"][source_slot].setdefault("links", [])
    source["outputs"][source_slot]["links"] = source["outputs"][source_slot]["links"] or []
    source["outputs"][source_slot]["links"].append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    return link_id


def _note(node_id: int, title: str, text: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "title": title,
        "pos": pos,
        "size": [900, 320],
        "flags": {},
        "order": 0,
        "mode": 0,
        "color": "#2d3f66",
        "bgcolor": "#111827",
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": text,
    }


def build() -> dict:
    workflow = json.loads(BASE.read_text(encoding="utf-8"))
    unet = _node(workflow, 1)
    conditioning = _node(workflow, 8)
    dual = _node(workflow, 9)
    guider = _node(workflow, 10)
    sampler = _node(workflow, 12)
    decode = _node(workflow, 14)
    planner = _node(workflow, 6)

    # Remove the historical Turbo LoRA and preserve the native Stock20 model path.
    _remove_link(workflow, 1)
    _remove_link(workflow, 2)
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != 2]
    _append_link(workflow, unet, 0, conditioning, 0, "MODEL")
    dual["widgets_values"] = [20, 12.0, 3.0, "dual_clock_euler", "native_flow"]
    dual["title"] = "Native Stock20 dual clock · exact per-segment schedule"

    composer_id = int(workflow["last_node_id"]) + 1
    composer = {
        "id": composer_id,
        "type": "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced",
        "title": "Fresh EAV runtime bound to this Long Video segment",
        "pos": [1430, -120],
        "size": [600, 480],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "sigmas", "type": "SIGMAS", "link": None},
            {"name": "segment_index", "type": "INT", "link": None},
            {"name": "context_frames", "type": "INT", "link": None},
            {"name": "mode", "type": "COMBO", "widget": {"name": "mode"}, "link": None},
            {"name": "tau", "type": "FLOAT", "widget": {"name": "tau"}, "link": None},
            {"name": "start_video_progress", "type": "FLOAT", "widget": {"name": "start_video_progress"}, "link": None},
            {"name": "end_video_progress", "type": "FLOAT", "widget": {"name": "end_video_progress"}, "link": None},
            {"name": "max_workspace_mib", "type": "INT", "widget": {"name": "max_workspace_mib"}, "link": None},
            {"name": "g_hard_limit", "type": "FLOAT", "widget": {"name": "g_hard_limit"}, "link": None},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": []},
            {"name": "runtime", "type": "H3_T8_EAV_RUNTIME", "links": []},
            {"name": "report_json", "type": "STRING", "links": []},
        ],
        "properties": {
            "Node name for S&R": "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced"
        },
        "widgets_values": ["apply_exp", 4.0, 0.0, 1.0, 32, 1.5],
    }
    audit = {
        "id": composer_id + 1,
        "type": "MiniMaxH3EnhanceAVideoAuditT8Advanced",
        "title": "Required per-segment audit · 20 forwards × 50 blocks",
        "pos": [2500, -20],
        "size": [500, 220],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "av_latent", "type": "LATENT", "link": None},
            {"name": "runtime", "type": "H3_T8_EAV_RUNTIME", "link": None},
        ],
        "outputs": [
            {"name": "av_latent", "type": "LATENT", "links": []},
            {"name": "report_json", "type": "STRING", "links": []},
        ],
        "properties": {
            "Node name for S&R": "MiniMaxH3EnhanceAVideoAuditT8Advanced"
        },
        "widgets_values": [],
    }
    workflow["nodes"].extend([composer, audit])
    workflow["last_node_id"] = audit["id"]

    _remove_link(workflow, 14)
    _remove_link(workflow, 25)
    _append_link(workflow, dual, 0, composer, 0, "MODEL")
    _append_link(workflow, dual, 2, composer, 1, "SIGMAS")
    _append_link(workflow, planner, 1, composer, 2, "INT")
    _append_link(workflow, planner, 3, composer, 3, "INT")
    _append_link(workflow, composer, 0, guider, 0, "MODEL")
    _append_link(workflow, sampler, 0, audit, 0, "LATENT")
    _append_link(workflow, composer, 1, audit, 1, "H3_T8_EAV_RUNTIME")
    _append_link(workflow, audit, 0, decode, 0, "LATENT")

    note_id = int(workflow["last_node_id"]) + 1
    workflow["nodes"].extend(
        [
            _note(
                note_id,
                "① 正确连接与Stock20范围",
                "## Long Video Conditioning → DualClock → EAV+Long Video Composer\n\n"
                "本模板已移除旧Turbo LoRA，固定原生Stock20。Planner的segment_index和context_frames必须"
                "同时连接到Conditioning和Composer；不要再串普通EAV、Prompt Relay、STG、BlockCache或Sage。",
                [0, 900],
            ),
            _note(
                note_id + 1,
                "② 分段与断点恢复合同",
                "## 每段一个全新的Runtime Audit\n\nsegment 0必须使用context_frames=0；后续段只接受5/22/39帧，"
                "并核对上一已接受片段提供的运动keyframe偏移。每次改变segment_index都会创建新的EAV runtime；"
                "Audit被消费后不会跨段复用。候选保存、人工接受和manifest恢复仍由原Long Video节点负责。",
                [930, 900],
            ),
            _note(
                note_id + 2,
                "③ 参数与证据边界",
                "## tau=4只是候选值\n\n`disabled`保留原Long Video MODEL，`report_only`只测量，`apply_exp`启用增益。"
                "每个Stock20段必须独立通过20次forward、每个活跃forward 50次测量。当前只做低负载合同和工作流"
                "导入验证，不宣称接缝更好、音频非劣、提速、省显存或通用16GB安全。",
                [1860, 900],
            ),
        ]
    )
    workflow["last_node_id"] = note_id + 2
    workflow.setdefault("extra", {})["t8_enhance_a_video_long_video"] = {
        "scope": "native Stock20 per-segment Long Video EAV Advanced EXP",
        "composition_order": "long_video_extra_conds_then_per_segment_eav",
        "context_frames": [0, 5, 22, 39],
        "audit": "20 forwards and 50 active measurements per forward for every segment",
        "resume": "fresh execution-local runtime bound by segment_index and context_frames",
        "validation_status": "deterministic_low_load_contract_pass",
        "quality_claim": False,
        "audio_noninferiority_claim": False,
        "performance_claim": False,
        "memory_safe_claim": False,
    }

    priority = {
        "UNETLoader": 0,
        "CLIPLoader": 1,
        "VAELoader": 2,
        "MiniMaxH3LongVideoPlannerT8": 3,
        "MiniMaxH3LongVideoAcceptedContextLoadT8": 4,
        "MiniMaxH3LongVideoConditioningT8": 5,
        "MiniMaxH3DualClockSamplerT8": 6,
        "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced": 7,
        "BasicGuider": 8,
        "RandomNoise": 9,
        "SamplerCustomAdvanced": 10,
        "MiniMaxH3EnhanceAVideoAuditT8Advanced": 11,
        "MiniMaxH3AVDecodeT8": 12,
        "MiniMaxH3OutputTrimT8": 13,
        "MiniMaxH3LongVideoCandidateSaveT8": 14,
        "MiniMaxH3LongVideoAcceptCandidateT8": 15,
        "MarkdownNote": 16,
    }
    for order, node in enumerate(
        sorted(workflow["nodes"], key=lambda item: (priority.get(item["type"], 99), item["id"]))
    ):
        node["order"] = order
    return workflow


def main() -> None:
    workflow = build()
    payload = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    INSTALLED.parent.mkdir(parents=True, exist_ok=True)
    INSTALLED.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
