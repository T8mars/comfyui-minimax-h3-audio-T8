from __future__ import annotations

from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_resource_estimate_advanced import (
    estimate_prompt_relay_resources,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


class MiniMaxH3PromptRelayResourceEstimateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayResourceEstimateT8Advanced",
            display_name="MiniMax H3 Prompt Relay Resource Estimate / 资源预估 (Advanced)",
            description=(
                "Model-free estimate of native H3 packed rows and Prompt Relay's bounded "
                "explicit chunk-bias allocation. It does not estimate total VRAM and is "
                "never a 16GB safety certificate. The Plan output is an unchanged pass-through."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                PromptRelayPlanIO.Input("prompt_relay_plan"),
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input(
                    "query_chunk_rows",
                    default=256,
                    min=32,
                    max=2048,
                    step=32,
                    tooltip="Must match the Prompt Relay Conditioning node for a useful estimate.",
                ),
                io.Combo.Input(
                    "precision",
                    options=["bf16_fp16", "fp32"],
                    default="bf16_fp16",
                    tooltip="Element width used only for the explicit bias-size estimate.",
                ),
                io.Int.Input(
                    "keyframe_stills",
                    default=0,
                    min=0,
                    max=16,
                    advanced=True,
                    tooltip="Count first/last/intermediate still guides at the target canvas size.",
                ),
                io.Int.Input(
                    "reference_images_match",
                    default=0,
                    min=0,
                    max=16,
                    advanced=True,
                    tooltip="Count only match-size reference images; use manual rows for max-size refs.",
                ),
                io.Int.Input(
                    "reference_video_count",
                    default=0,
                    min=0,
                    max=3,
                    advanced=True,
                ),
                io.Int.Input(
                    "reference_video_frames_each",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    advanced=True,
                    tooltip="Requested frames per reference video; the report shows H3 aligned-down frames.",
                ),
                io.Boolean.Input(
                    "reference_video_has_audio",
                    default=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "reference_video_audio_seconds_each",
                    default=5.0,
                    min=0.0,
                    max=900.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "standalone_reference_audio_count",
                    default=0,
                    min=0,
                    max=3,
                    advanced=True,
                ),
                io.Float.Input(
                    "standalone_reference_audio_seconds_each",
                    default=5.0,
                    min=0.0,
                    max=900.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "additional_text_rows",
                    default=256,
                    min=0,
                    max=1_000_000,
                    advanced=True,
                    tooltip="Conservative allowance for Qwen system/media tokens beyond UTF-8 prompt bytes.",
                ),
                io.Int.Input(
                    "manual_extra_packed_rows",
                    default=0,
                    min=0,
                    max=10_000_000,
                    advanced=True,
                    tooltip="Manual allowance for max-size refs or other conditioning not represented above.",
                ),
            ],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.Int.Output("estimated_seq_len"),
                io.Float.Output("peak_explicit_bias_mib"),
                io.String.Output("summary_text"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt_relay_plan,
        width,
        height,
        query_chunk_rows,
        precision,
        keyframe_stills,
        reference_images_match,
        reference_video_count,
        reference_video_frames_each,
        reference_video_has_audio,
        reference_video_audio_seconds_each,
        standalone_reference_audio_count,
        standalone_reference_audio_seconds_each,
        additional_text_rows,
        manual_extra_packed_rows,
    ):
        values = estimate_prompt_relay_resources(
            prompt_relay_plan,
            width,
            height,
            query_chunk_rows,
            precision,
            keyframe_stills,
            reference_images_match,
            reference_video_count,
            reference_video_frames_each,
            reference_video_has_audio,
            reference_video_audio_seconds_each,
            standalone_reference_audio_count,
            standalone_reference_audio_seconds_each,
            additional_text_rows,
            manual_extra_packed_rows,
        )
        return io.NodeOutput(*values, ui={"text": (values[3],)})


PROMPT_RELAY_RESOURCE_ESTIMATE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRelayResourceEstimateT8Advanced,
]
