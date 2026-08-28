from __future__ import annotations

from comfy_api.latest import io

from .ays_schedule_advanced import (
    SCHEDULE_PROFILES,
    build_dual_clock_schedule_contract,
)
from .sampling import DEFAULT_SAMPLER_NAME, SAMPLER_OPTIONS


CATEGORY = "T8/MiniMax H3/Sampling/Experimental"


class MiniMaxH3DualClockAYSScheduleT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DualClockAYSScheduleT8Advanced",
            display_name="MiniMax H3 Dual-Clock AYS Schedule Contract (Advanced EXP)",
            description=(
                "Builds the native H3 schedule or imports H3-calibrated base-flow knots, "
                "then maps the same knots onto separate video/audio clocks. It does not "
                "claim that image-model AYS schedules are valid for MiniMax H3."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=8, min=1, max=1000),
                io.Float.Input(
                    "shift_video", default=12.0, min=0.01, max=100.0, step=0.01
                ),
                io.Float.Input(
                    "shift_audio", default=3.0, min=0.01, max=100.0, step=0.01
                ),
                io.Combo.Input(
                    "schedule_profile",
                    options=list(SCHEDULE_PROFILES),
                    default="native_flow_baseline",
                ),
                io.String.Input(
                    "manual_base_sigmas",
                    default="[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]",
                    multiline=True,
                    tooltip=(
                        "Used only by manual_h3_calibrated. Supply steps+1 strictly "
                        "descending base-flow sigmas from 1.0 to 0.0."
                    ),
                ),
                io.String.Input(
                    "schedule_label",
                    default="unvalidated H3 calibration",
                    tooltip="Human-readable provenance; a label is not validation evidence.",
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        av_latent,
        steps,
        shift_video,
        shift_audio,
        schedule_profile,
        manual_base_sigmas,
        schedule_label,
        sampler_name=DEFAULT_SAMPLER_NAME,
    ):
        return io.NodeOutput(
            *build_dual_clock_schedule_contract(
                model,
                av_latent,
                steps,
                shift_video,
                shift_audio,
                schedule_profile,
                manual_base_sigmas,
                schedule_label,
                sampler_name,
            )
        )


AYS_SCHEDULE_ADVANCED_NODE_CLASSES = [MiniMaxH3DualClockAYSScheduleT8Advanced]

