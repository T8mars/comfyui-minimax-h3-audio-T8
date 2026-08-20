from __future__ import annotations

import json

from comfy_api.latest import io

from .prompt_relay_advanced import (
    PROMPT_RELAY_PLAN_TYPE,
    build_prompt_relay_conditioning,
    build_prompt_relay_plan,
    configure_prompt_relay_query_route,
)
from .prompt_relay_events_advanced import (
    PROMPT_RELAY_EVENTS_TYPE,
    json_hash,
    prompt_relay_events_to_inputs,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
MAX_RESOLUTION = 16384
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)
PromptRelayEventsIO = io.Custom(PROMPT_RELAY_EVENTS_TYPE)


def build_prompt_relay_plan_with_optional_events(
    *,
    prompt_relay_events=None,
    **kwargs,
):
    if prompt_relay_events is None:
        return build_prompt_relay_plan(**kwargs)

    local_prompts, time_ranges, event_count, event_collection_hash = (
        prompt_relay_events_to_inputs(
            prompt_relay_events,
            str(kwargs["timing_mode"]),
        )
    )
    kwargs = dict(kwargs)
    kwargs["local_prompts"] = local_prompts
    kwargs["time_ranges"] = time_ranges
    plan, compiled_prompt, frame_count, timeline_json, report_json = (
        build_prompt_relay_plan(**kwargs)
    )
    plan = dict(plan)
    plan.pop("plan_hash", None)
    plan["source_events"] = {
        "source": "typed_event_chain",
        "events_hash": event_collection_hash,
        "event_count": event_count,
    }
    plan["plan_hash"] = json_hash(plan)

    report = json.loads(report_json)
    report.update(
        {
            "event_source": "typed_event_chain",
            "event_collection_hash": event_collection_hash,
            "event_count": event_count,
            "plan_hash": plan["plan_hash"],
        }
    )
    report["notes"] = list(report.get("notes", [])) + [
        "the connected Event chain overrides local_prompts and time_ranges",
    ]
    return (
        plan,
        compiled_prompt,
        frame_count,
        timeline_json,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


class MiniMaxH3PromptRelayPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayPlanT8Advanced",
            display_name="MiniMax H3 Prompt Relay Plan / 分段提示词时间线 (Advanced)",
            description=(
                "Compiles one global prompt plus line-separated local events into an "
                "auditable H3 timeline. paper_v1 uses the published epsilon/window/sigma "
                "equation; legacy_repo_compat is comparison-only."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "global_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        "Cinematic continuous shot, coherent subject identity, natural "
                        "motion, stable lighting and synchronized ambience."
                    ),
                    tooltip="始终可见的全局场景、人物、风格与声音约束。",
                ),
                io.String.Input(
                    "local_prompts",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        "The woman looks toward the camera and raises her hand.\n"
                        "She turns and walks rapidly into the illuminated street.\n"
                        "The camera pulls back while she becomes small in the distance."
                    ),
                    tooltip=(
                        "每行一个局部事件；也兼容使用 | 分隔。清空文本和time_ranges时"
                        "生成只含global_prompt的无补丁直通Plan。"
                    ),
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="24fps；会向上对齐到 MiniMax H3 的 17n+5 帧网格。",
                ),
                io.Combo.Input(
                    "timing_mode",
                    options=["auto_equal", "frames", "seconds", "percent"],
                    default="auto_equal",
                    tooltip="auto_equal 自动均分；其他模式读取 time_ranges。",
                ),
                io.String.Input(
                    "time_ranges",
                    multiline=True,
                    default="0-40\n41-81\n82-123",
                    tooltip=(
                        "每行一个范围，并按开始时间从早到晚排列。frames 只接受整数帧且"
                        "结束帧包含在范围内；seconds/percent 的结束值作为边界；percent "
                        "使用 0..100。"
                    ),
                ),
                io.Combo.Input(
                    "math_profile",
                    options=["paper_v1", "legacy_repo_compat"],
                    default="paper_v1",
                    tooltip=(
                        "paper_v1 是论文公式；legacy_repo_compat 只用于复现社区旧参数，"
                        "不代表论文实现。"
                    ),
                ),
                io.Float.Input(
                    "epsilon",
                    default=0.1,
                    min=0.000001,
                    max=0.999999,
                    step=0.01,
                    advanced=True,
                    tooltip="论文默认 0.1；paper_v1 会让事件边界权重回到 epsilon。",
                ),
                io.Boolean.Input(
                    "allow_gaps",
                    default=False,
                    advanced=True,
                    tooltip="关闭时，各局部事件必须覆盖完整目标时间线。",
                ),
                io.Boolean.Input(
                    "allow_overlaps",
                    default=False,
                    advanced=True,
                    tooltip="关闭时，局部事件时间范围不能重叠。",
                ),
                PromptRelayEventsIO.Input(
                    "prompt_relay_events",
                    optional=True,
                    tooltip=(
                        "可选：连接链式 Event 节点后以它为权威事件源；未连接时继续读取"
                        "local_prompts/time_ranges。"
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
        return io.NodeOutput(*build_prompt_relay_plan_with_optional_events(**kwargs))


class MiniMaxH3PromptRelayConditioningT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayConditioningT8Advanced",
            display_name="MiniMax H3 Prompt Relay Conditioning / 分段提示词条件 (Advanced)",
            description=(
                "Authoritatively encodes the Plan prompt through the native H3 Qwen chain, "
                "binds exact local token spans, and optionally patches a cloned H3 MODEL "
                "with a bounded-memory plan-selected query relay. Stable nodes remain unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="原生 MiniMax H3 MODEL；输出为局部克隆。"),
                io.Clip.Input("clip", tooltip="原生 MiniMax H3 Qwen3-VL CLIP。"),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE。"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE。"),
                PromptRelayPlanIO.Input(
                    "prompt_relay_plan",
                    tooltip="必须直接来自同一版本 Prompt Relay Plan 节点。",
                ),
                io.Int.Input("width", default=1056, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=608, min=32, max=MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "task_type",
                    options=["T2VA", "auto", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="T2VA",
                    tooltip=(
                        "支持 T2VA/I2VA/FL2VA/L2VA/Ref2VA/Hybrid 的目标视频时间路由；"
                        "显式接入 Query Route 后可实验性扩展到目标音频。"
                    ),
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["native", "lock_source", "remix_source", "reference_only"],
                    default="native",
                    tooltip=(
                        "支持原生生成、锁定源音频、重混源音频和仅参考音频。默认Plan只路由目标"
                        "视频；joint_av_exp会直接路由目标音频，不能与lock_source组合。"
                    ),
                ),
                io.Float.Input(
                    "audio_denoise_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("add_source_as_reference", default=True, advanced=True),
                io.Int.Input(
                    "prompt_primary_audio_ordinal",
                    default=1,
                    min=0,
                    max=9,
                    step=1,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    advanced=True,
                ),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.Combo.Input(
                    "execution_mode",
                    options=["report_only", "apply_exp"],
                    default="report_only",
                    tooltip=(
                        "先用 report_only 检查 token/layout/显存报告；确认后显式选择 "
                        "apply_exp 才安装注意力路由。零个或一个局部事件时会自动原样直通，"
                        "不安装补丁；单事件通常应合并进 global_prompt。"
                    ),
                ),
                io.Int.Input(
                    "query_chunk_rows",
                    default=256,
                    min=32,
                    max=2048,
                    step=32,
                    advanced=True,
                    tooltip=(
                        "目标视频/显式联合音频 query 分块。更小更省显存但更慢；不会创建完整 S×S mask。"
                    ),
                ),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="24fps IMAGE frame batch."),
                        prefix="ref_video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.Audio.Output("mux_audio"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("media_map_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_prompt_relay_conditioning(**kwargs))


class MiniMaxH3PromptRelayQueryRouteT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayQueryRouteT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Relay Query Route / 联合音画路由 (Advanced)"
            ),
            description=(
                "Optionally extends a Prompt Relay Plan from the paper-faithful target-video "
                "route to an experimental native-H3 target-audio-plus-video route. Connect "
                "this between Plan and Conditioning; old workflows bypass it unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                PromptRelayPlanIO.Input(
                    "prompt_relay_plan",
                    tooltip="来自 Prompt Relay Plan 的原始、已校验时间线。",
                ),
                io.Combo.Input(
                    "query_route",
                    options=["video_only_paper", "joint_av_exp"],
                    default="video_only_paper",
                    tooltip=(
                        "video_only_paper只路由目标视频并保持论文范围；joint_av_exp还会按"
                        "H3原生音频时间轴路由目标音频，属于未被论文验证的实验扩展。"
                    ),
                ),
            ],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*configure_prompt_relay_query_route(**kwargs))


PROMPT_RELAY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRelayPlanT8Advanced,
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayQueryRouteT8Advanced,
]
