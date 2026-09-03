from __future__ import annotations

from comfy_api.latest import io

from .long_video import CONTEXT_TYPE_NAME
from .native_masked_context_advanced import apply_native_masked_video_context


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
LongVideoContext = io.Custom(CONTEXT_TYPE_NAME)


class MiniMaxH3NativeMaskedVideoContextT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeMaskedVideoContextT8Advanced",
            display_name=(
                "MiniMax H3 Native Masked Video Context / "
                "原生画面硬续接 Plan B (Advanced EXP/T8)"
            ),
            description=(
                "Plan B for Long Video continuation. It copies the validated previous native "
                "video-latent tail into the current target and hard-locks only that video prefix. "
                "The current audio tensor and any Vocal Lock audio mask are reused unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip=(
                        "Connect the matching Long Video Conditioning latent before the sampler."
                    ),
                ),
                LongVideoContext.Input(
                    "context",
                    tooltip="Connect the same validated Previous Context used by Conditioning.",
                ),
                io.String.Input(
                    "planner_report_json",
                    force_input=True,
                    tooltip="Connect the matching Segment Planner report_json directly.",
                ),
                io.String.Input(
                    "conditioning_report_json",
                    force_input=True,
                    tooltip=(
                        "Connect the matching Long Video Conditioning report_json directly. "
                        "Conditioning must use context_audio=video_only."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Int.Output("trim_context_frames"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        context,
        planner_report_json,
        conditioning_report_json,
    ):
        return io.NodeOutput(
            *apply_native_masked_video_context(
                av_latent,
                context,
                planner_report_json,
                conditioning_report_json,
            )
        )


NATIVE_MASKED_CONTEXT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3NativeMaskedVideoContextT8Advanced,
]
