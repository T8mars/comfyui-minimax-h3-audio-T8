from __future__ import annotations

from comfy_api.latest import io

from .long_video_sampling_plan_advanced import (
    LONG_VIDEO_SAMPLING_PLAN_TYPE,
    PLAN_MODES,
    TAIL_SPACING,
    build_long_video_sampling_plan,
)


CATEGORY = "T8/MiniMax H3/Long Video/Advanced"
LongVideoSamplingPlanIO = io.Custom(LONG_VIDEO_SAMPLING_PLAN_TYPE)


class MiniMaxH3LongVideoSamplingPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoSamplingPlanT8Advanced",
            display_name="MiniMax H3 Long Video Tail/Second Pass Plan (Advanced EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Optional append-only sampling plan for the two in-node long-video runners. "
                "Disconnected/disabled preserves their old sampler exactly."
            ),
            inputs=[
                io.Combo.Input("mode", options=list(PLAN_MODES), default="disabled"),
                io.Int.Input("extra_tail_steps", default=1, min=0, max=8),
                io.Combo.Input(
                    "tail_spacing", options=list(TAIL_SPACING), default="video_sigma_linear"
                ),
                io.String.Input(
                    "manual_sigmas",
                    default="0.5, 0.412, 0.350, 0",
                    tooltip="Used only by manual_second_pass; must strictly descend to 0.",
                ),
            ],
            outputs=[
                LongVideoSamplingPlanIO.Output("sampling_plan"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, mode, extra_tail_steps, tail_spacing, manual_sigmas):
        return io.NodeOutput(
            *build_long_video_sampling_plan(
                mode, extra_tail_steps, tail_spacing, manual_sigmas
            )
        )


LONG_VIDEO_SAMPLING_PLAN_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoSamplingPlanT8Advanced,
]
