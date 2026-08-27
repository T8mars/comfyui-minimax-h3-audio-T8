from __future__ import annotations

from comfy_api.latest import io

from .long_video_voice_context_advanced import (
    VOICE_CONTEXT_PLAN_TYPE,
    build_long_video_voice_context_plan,
    release_long_video_voice_context_plan,
)


CATEGORY = "T8/MiniMax H3/Long Video/Advanced"
VoiceContextPlanIO = io.Custom(VOICE_CONTEXT_PLAN_TYPE)


class MiniMaxH3LongVideoVoiceContextT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoVoiceContextT8Advanced",
            display_name="MiniMax H3 Long Video Voice Context / 长视频人物音色与句界 (Advanced EXP/T8)",
            description=(
                "Compiles Character-to-<Audio N> bindings, exact global/local audio pin frames, "
                "per-segment prompts and cross-boundary sentence diagnostics for the existing "
                "long-video runtime. It never reinjects or cuts waveform audio."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_voice_context"),
                io.Float.Input("total_duration_seconds", default=30.0, min=0.04, max=3600.0, step=0.01),
                io.Int.Input("render_window_frames", default=124, min=124, max=362, step=17),
                io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
                io.String.Input("global_prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "dialogue_timeline_json",
                    default='[{"character_id":"Character_A","text":"你好。","start_seconds":0,"end_seconds":1.5}]',
                    multiline=True,
                ),
                io.String.Input(
                    "voice_bindings_json",
                    default='{"Character_A":1}',
                    multiline=True,
                ),
                io.Combo.Input(
                    "cross_boundary_policy",
                    options=["abstain", "duplicate_exact_text_exp"],
                    default="abstain",
                ),
                io.Boolean.Input("first_shot_review_required", default=True),
            ],
            outputs=[
                VoiceContextPlanIO.Output("voice_context_plan"),
                io.String.Output("segment_prompts_json"),
                io.String.Output("audio_pin_frames_json"),
                io.Boolean.Output("ready"),
                io.Boolean.Output("first_shot_review_required"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_long_video_voice_context_plan(**kwargs))


class MiniMaxH3LongVideoVoiceReviewGateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoVoiceReviewGateT8Advanced",
            display_name="MiniMax H3 Voice Context Review Gate / 首段人工确认门 (Advanced EXP/T8)",
            description=(
                "Releases the compiled segment prompts only after the optional first-shot "
                "human review is accepted. This is a planning gate; a real pause between "
                "segments uses the existing Background/Accepted workflow."
            ),
            category=CATEGORY,
            inputs=[
                VoiceContextPlanIO.Input("voice_context_plan"),
                io.Boolean.Input("first_shot_approved", default=False),
            ],
            outputs=[
                io.String.Output("segment_prompts_json"),
                io.String.Output("audio_pin_frames_json"),
                io.Boolean.Output("released"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, voice_context_plan, first_shot_approved):
        return io.NodeOutput(
            *release_long_video_voice_context_plan(
                voice_context_plan, first_shot_approved
            )
        )


LONG_VIDEO_VOICE_CONTEXT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoVoiceContextT8Advanced,
    MiniMaxH3LongVideoVoiceReviewGateT8Advanced,
]
