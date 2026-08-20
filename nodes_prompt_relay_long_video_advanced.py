from __future__ import annotations

from comfy_api.latest import io

from .long_video import CONTEXT_TYPE_NAME
from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_long_video_advanced import (
    build_prompt_relay_long_video_conditioning,
    project_prompt_relay_plan_to_long_video_window,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
MAX_RESOLUTION = 16384
LongVideoContextIO = io.Custom(CONTEXT_TYPE_NAME)
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


class MiniMaxH3PromptRelayLongVideoPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayLongVideoPlanT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Relay Long Video Window / 长视频绝对时间窗口 (Advanced)"
            ),
            description=(
                "Projects one global Prompt Relay timeline into the exact rendered Long Video "
                "window. Continuation frame 0 maps to accepted-start minus context overlap, so "
                "events do not restart at each segment. Existing Plan and Long Video nodes stay unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                PromptRelayPlanIO.Input(
                    "prompt_relay_plan",
                    tooltip="整条长视频只创建一次的全局 Prompt Relay Plan。",
                ),
                io.Int.Input(
                    "segment_index",
                    default=0,
                    min=0,
                    max=99999,
                    force_input=True,
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    force_input=True,
                    tooltip="直接连接 Long Video Planner 的 length。",
                ),
                io.Int.Input(
                    "context_frames",
                    default=0,
                    min=0,
                    max=39,
                    force_input=True,
                    tooltip="直接连接 Long Video Planner 的 context_frames。",
                ),
                io.Float.Input(
                    "timeline_start_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                    force_input=True,
                ),
                io.Float.Input(
                    "timeline_end_seconds",
                    default=5.166667,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                    force_input=True,
                ),
            ],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.String.Output("compiled_prompt"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*project_prompt_relay_plan_to_long_video_window(**kwargs))


class MiniMaxH3PromptRelayLongVideoConditioningT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayLongVideoConditioningT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Relay Long Video Conditioning / 长视频分段路由条件 (Advanced)"
            ),
            description=(
                "Isolated composition of the existing Long Video motion-context payload repair "
                "and Prompt Relay attention routing. It requires the paired projected Plan and "
                "never changes the old Long Video Conditioning schema."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="未接其他补丁的原生 MiniMax H3 MODEL。"),
                io.Clip.Input("clip", tooltip="原生 MiniMax H3 Qwen3-VL CLIP。"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                LongVideoContextIO.Input("context"),
                PromptRelayPlanIO.Input(
                    "prompt_relay_plan",
                    tooltip="必须来自同段 Long Video Window 节点。",
                ),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
                io.Int.Input("context_frames", default=0, min=0, max=39, force_input=True),
                io.Combo.Input(
                    "context_audio",
                    options=["video_and_audio", "video_only"],
                    default="video_and_audio",
                ),
                io.Int.Input("width", default=1056, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=608, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, force_input=True),
                io.Combo.Input(
                    "task_type",
                    options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="auto",
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["native", "lock_source", "remix_source", "reference_only"],
                    default="native",
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
                    default=0,
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
                    tooltip="先 report_only；确认报告后再显式切到 apply_exp。",
                ),
                io.Int.Input(
                    "query_chunk_rows",
                    default=256,
                    min=32,
                    max=2048,
                    step=32,
                    advanced=True,
                ),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3
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
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3
                    ),
                ),
                io.Combo.Input(
                    "first_frame_reuse",
                    options=["segment0_only", "persistent_identity_reference"],
                    default="segment0_only",
                    optional=True,
                    advanced=True,
                ),
                io.Image.Input("persistent_identity_image", optional=True, advanced=True),
                io.Combo.Input(
                    "persistent_identity_strategy",
                    options=["single_reference", "scene_plus_identity"],
                    default="single_reference",
                    optional=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "persistent_identity_interval",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    optional=True,
                    advanced=True,
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
        return io.NodeOutput(*build_prompt_relay_long_video_conditioning(**kwargs))


PROMPT_RELAY_LONG_VIDEO_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
    MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
]
