from __future__ import annotations

from comfy_api.latest import io

from .nodes_studio_advanced import StudioTimelineIO
from .reel_delivery_advanced import (
    PEAK_POLICIES,
    build_reel_delivery_plan,
    canonical_json,
    compose_reel_delivery,
)


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
ReelPlanIO = io.Custom("H3_T8_REEL_DELIVERY_PLAN")


class MiniMaxH3ReelDeliveryPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReelDeliveryPlanT8Advanced",
            display_name="MiniMax H3 Reel Delivery Plan / 成片交付计划 (Advanced)",
            description=(
                "Validates exact 24fps file clips, trims, bounded visual crossfades and "
                "dialogue/music/ambience/SFX lanes. Planning is read-only and fingerprints every "
                "source file."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("project_id", default="h3_reel"),
                io.String.Input(
                    "reel_json",
                    multiline=True,
                    default=(
                        '{"clips":[{"id":"shot_1","path":"output:shot_1.mp4",'
                        '"trim_in_seconds":0,"crossfade_to_next_seconds":0}],'
                        '"audio_lanes":[]}'
                    ),
                ),
                io.Int.Input(
                    "sample_rate",
                    default=48000,
                    min=32000,
                    max=48000,
                    step=1000,
                ),
                io.Float.Input(
                    "maximum_transition_seconds",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=1.0 / 24.0,
                ),
                io.Float.Input(
                    "maximum_transition_buffer_mib",
                    default=512.0,
                    min=16.0,
                    max=4096.0,
                    step=16.0,
                    advanced=True,
                ),
                StudioTimelineIO.Input("timeline", optional=True),
            ],
            outputs=[
                ReelPlanIO.Output("reel_plan"),
                io.Int.Output("total_frames"),
                io.Float.Output("total_duration_seconds"),
                io.String.Output("plan_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        project_id,
        reel_json,
        sample_rate,
        maximum_transition_seconds,
        maximum_transition_buffer_mib,
        timeline=None,
    ):
        plan = build_reel_delivery_plan(
            project_id,
            reel_json,
            sample_rate,
            maximum_transition_seconds,
            maximum_transition_buffer_mib,
            timeline,
        )
        return io.NodeOutput(
            plan,
            plan["total_frames"],
            plan["total_duration_seconds"],
            canonical_json(plan),
        )


class MiniMaxH3ReelDeliveryComposeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReelDeliveryComposeT8Advanced",
            display_name="MiniMax H3 Reel Delivery Compose / 成片合成 (Advanced)",
            description=(
                "Streams a validated file-level reel into an atomic MP4. It requires explicit "
                "confirmation, supports hash-verified phase resume, and never materializes the "
                "whole reel as a ComfyUI IMAGE/AUDIO batch."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                ReelPlanIO.Input("reel_plan"),
                io.Boolean.Input("confirm_compose", default=False),
                io.String.Input("filename_prefix", default="H3_Reel"),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.Combo.Input(
                    "peak_policy",
                    options=list(PEAK_POLICIES),
                    default="block_if_clipping",
                ),
            ],
            outputs=[
                io.String.Output("output_video_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        reel_plan,
        confirm_compose,
        filename_prefix,
        crf,
        peak_policy,
    ):
        path, report = compose_reel_delivery(
            reel_plan,
            confirm_compose,
            filename_prefix,
            crf,
            peak_policy,
        )
        return io.NodeOutput(path, canonical_json(report))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


REEL_DELIVERY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ReelDeliveryPlanT8Advanced,
    MiniMaxH3ReelDeliveryComposeT8Advanced,
]
