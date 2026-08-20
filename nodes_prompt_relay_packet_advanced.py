from __future__ import annotations

from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_packet_advanced import (
    PROMPT_RELAY_EVENTS_TYPE,
    build_prompt_relay_event,
    build_prompt_relay_plan_from_packet,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
PromptPacketIO = io.Custom("H3_T8_PROMPT_PACKET")
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)
PromptRelayEventsIO = io.Custom(PROMPT_RELAY_EVENTS_TYPE)


class MiniMaxH3PromptPacketRelayPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptPacketRelayPlanT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Packet → Relay Plan / 编译提示词转时间线 "
                "(Advanced)"
            ),
            description=(
                "Converts an authenticated MiniMax H3 Prompt Compiler packet into an "
                "auditable Prompt Relay timeline. The packet's compiled visual/audio "
                "prompt remains global; local events are never inferred or rewritten."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                PromptPacketIO.Input(
                    "prompt_packet",
                    tooltip=(
                        "直接连接 T8 Video Prompt Compiler；只接受 backend=minimax_h3 "
                        "且哈希完整的 packet。"
                    ),
                ),
                io.String.Input(
                    "events_json",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        '[\n  {"prompt":"The subject raises one hand."},\n'
                        '  {"prompt":"The subject turns and walks forward."},\n'
                        '  {"prompt":"The camera pulls back into a wide shot."}\n]'
                    ),
                    tooltip=(
                        "显式事件列表。auto_equal只写prompt；其他模式还要写start/end。"
                        "使用[]表示只保留全局Prompt Packet且不安装Relay补丁。"
                    ),
                ),
                io.Combo.Input(
                    "timing_mode",
                    options=["auto_equal", "frames", "seconds", "percent"],
                    default="auto_equal",
                    tooltip=(
                        "frames的end为包含端点；seconds/percent的end为时间边界。"
                    ),
                ),
                io.Combo.Input(
                    "math_profile",
                    options=["paper_v1", "legacy_repo_compat"],
                    default="paper_v1",
                    tooltip="默认使用Prompt Relay论文公式。",
                ),
                io.Float.Input(
                    "epsilon",
                    default=0.1,
                    min=0.000001,
                    max=0.999999,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("allow_gaps", default=False, advanced=True),
                io.Boolean.Input("allow_overlaps", default=False, advanced=True),
                PromptRelayEventsIO.Input(
                    "prompt_relay_events",
                    optional=True,
                    tooltip=(
                        "可选：连接链式 Event 节点后以它为权威事件源；未连接时继续读取 events_json。"
                    ),
                ),
            ],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.String.Output("compiled_prompt"),
                io.Int.Output("length"),
                io.String.Output("timeline_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_prompt_relay_plan_from_packet(**kwargs))


class MiniMaxH3PromptRelayEventT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayEventT8Advanced",
            display_name="MiniMax H3 Prompt Relay Event / 单事件串联 (Advanced)",
            description=(
                "Adds one explicit local action to a chainable Prompt Relay event list. "
                "Duplicate this node for each event; start/end are interpreted by the "
                "Packet bridge timing mode."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="The subject performs one clearly observable action.",
                    tooltip="只写这一时间段的局部动作，不重复全局人物/场景/声音设定。",
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "auto_equal时忽略；frames为整数起始帧，seconds为秒，percent为0..100；"
                        "多个Event必须按开始时间排序。"
                    ),
                ),
                io.Float.Input(
                    "end",
                    default=100.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "frames模式只接受整数并包含end帧；seconds/percent模式把end视为边界。"
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "关闭后不把本事件加入链；如果整条链都关闭，Plan会变成"
                        "全局提示直通，不需要拆线。"
                    ),
                ),
                PromptRelayEventsIO.Input(
                    "previous_events",
                    optional=True,
                    tooltip="连接上一个Event节点；第一个Event留空。",
                ),
            ],
            outputs=[
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("preview_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_prompt_relay_event(**kwargs))


PROMPT_RELAY_PACKET_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptPacketRelayPlanT8Advanced,
    MiniMaxH3PromptRelayEventT8Advanced,
]
