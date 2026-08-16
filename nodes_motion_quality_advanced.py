from __future__ import annotations

from comfy_api.latest import io

from .motion_quality_advanced import (
    TURBO_DUAL_CLOCK_TEST_STEPS,
    audit_motion_quality,
    build_av_sigma_same_nfe_schedule,
    build_av_sigma_tail_schedule,
    build_motion_repair_plan,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental"
StudioTimelineIO = io.Custom("H3_T8_STUDIO_TIMELINE")
RepairPlanIO = io.Custom("H3_T8_REPAIR_PLAN")


class MiniMaxH3AVSigmaTailSubdivisionT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVSigmaTailSubdivisionT8Advanced",
            display_name="MiniMax H3 AV Sigma Tail Subdivision / 音画尾段细分 (Advanced)",
            description=(
                "Default-off H3 schedule experiment. It preserves every input sigma knot, "
                "inserts optional points in base-flow time, and reports both video and audio "
                f"clocks. Turbo dual-clock validation uses {TURBO_DUAL_CLOCK_TEST_STEPS} steps."
            ),
            category=CATEGORY,
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "apply_exp"],
                    default="report_only",
                ),
                io.Int.Input(
                    "extra_substeps",
                    default=0,
                    min=0,
                    max=32,
                    tooltip=(
                        "Total additional full H3 audio-video DiT calls. Zero is an exact pass-through."
                    ),
                ),
                io.Combo.Input(
                    "range_mode",
                    options=["tail_intervals", "base_progress"],
                    default="tail_intervals",
                ),
                io.Int.Input(
                    "tail_intervals",
                    default=1,
                    min=1,
                    max=1000,
                    tooltip="Number of original intervals selected from the zero-sigma tail.",
                ),
                io.Float.Input(
                    "start_progress",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_progress",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "spacing",
                    options=["base_time_linear", "base_time_cosine"],
                    default="base_time_linear",
                ),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "profile",
                    options=[
                        "turbo_standard8",
                        "turbo_ema8",
                        "turbo_fl2v8",
                        "stock20",
                        "custom_strict",
                    ],
                    default="turbo_standard8",
                    tooltip="All project Turbo dual-clock tests use the 8-step baseline.",
                ),
                io.Combo.Input(
                    "sampling_route",
                    options=[
                        "dual_clock_euler",
                        "native_flow_av_unverified",
                        "multirate_exp_unsupported",
                        "unknown",
                    ],
                    default="dual_clock_euler",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "accept_turbo_schedule_ood",
                    default=False,
                    tooltip=(
                        "Required before adding untrained intermediate times to an 8-step Turbo schedule."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.Sigmas.Output("sigmas"),
                io.Int.Output("actual_nfe"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        sigmas,
        mode,
        extra_substeps,
        range_mode,
        tail_intervals,
        start_progress,
        end_progress,
        spacing,
        shift_video,
        shift_audio,
        profile,
        sampling_route,
        accept_turbo_schedule_ood,
    ):
        return io.NodeOutput(
            *build_av_sigma_tail_schedule(
                sigmas,
                mode,
                extra_substeps,
                range_mode,
                tail_intervals,
                start_progress,
                end_progress,
                spacing,
                shift_video,
                shift_audio,
                profile,
                sampling_route,
                accept_turbo_schedule_ood,
            )
        )


class MiniMaxH3AVSigmaSameNFERedistributionT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVSigmaSameNFERedistributionT8Advanced",
            display_name="MiniMax H3 AV Sigma Same-NFE / 音画同NFE重分布 (Advanced)",
            description=(
                "Default-off causal-control schedule. It keeps the exact joint AV model-call "
                "count and endpoints while optionally redistributing existing base-flow times. "
                "Turbo dual-clock validation uses eight steps and requires explicit OOD consent."
            ),
            category=CATEGORY,
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "apply_exp"],
                    default="report_only",
                ),
                io.Float.Input(
                    "start_progress",
                    default=0.5,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                ),
                io.Float.Input(
                    "tail_power",
                    default=1.6,
                    min=0.2,
                    max=5.0,
                    step=0.05,
                    tooltip="1.0 is exact identity; values above 1 move existing points toward the zero-sigma tail.",
                ),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "profile",
                    options=[
                        "turbo_standard8",
                        "turbo_ema8",
                        "turbo_fl2v8",
                        "stock20",
                        "custom_strict",
                    ],
                    default="turbo_standard8",
                ),
                io.Combo.Input(
                    "sampling_route",
                    options=[
                        "dual_clock_euler",
                        "native_flow_av_unverified",
                        "multirate_exp_unsupported",
                        "unknown",
                    ],
                    default="dual_clock_euler",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "accept_turbo_schedule_ood",
                    default=False,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Sigmas.Output("sigmas"),
                io.Int.Output("actual_nfe"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        sigmas,
        mode,
        start_progress,
        tail_power,
        shift_video,
        shift_audio,
        profile,
        sampling_route,
        accept_turbo_schedule_ood,
    ):
        return io.NodeOutput(
            *build_av_sigma_same_nfe_schedule(
                sigmas,
                mode,
                start_progress,
                tail_power,
                shift_video,
                shift_audio,
                profile,
                sampling_route,
                accept_turbo_schedule_ood,
            )
        )


class MiniMaxH3MotionQualityAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionQualityAuditT8Advanced",
            display_name="MiniMax H3 Motion Quality Audit / 高速动态质量审计 (Advanced)",
            description=(
                "Read-only, dependency-free temporal proxy audit. Use a reviewed face/subject mask "
                "or manual ROI for face-focused analysis. It does not claim face identity detection "
                "and never changes, uploads, or automatically accepts generated media."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input(
                    "roi_mode",
                    options=["full_frame", "manual_static_roi", "connected_mask"],
                    default="full_frame",
                ),
                io.Float.Input(
                    "roi_x", default=0.25, min=0.0, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "roi_y", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "roi_width", default=0.5, min=0.01, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "roi_height", default=0.5, min=0.01, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "sharpness_ratio_floor",
                    default=0.55,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "temporal_instability_multiplier",
                    default=2.5,
                    min=1.0,
                    max=100.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "high_motion_delta_floor",
                    default=0.03,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "freeze_delta_ceiling",
                    default=0.002,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                ),
                io.Int.Input(
                    "repair_context_frames",
                    default=12,
                    min=0,
                    max=3600,
                    advanced=True,
                ),
                io.Mask.Input("face_mask", optional=True),
            ],
            outputs=[
                io.Boolean.Output("risk_detected"),
                io.Int.Output("risk_range_count"),
                io.String.Output("risk_ranges_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        frames,
        fps,
        roi_mode,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        sharpness_ratio_floor,
        temporal_instability_multiplier,
        high_motion_delta_floor,
        freeze_delta_ceiling,
        repair_context_frames,
        face_mask=None,
    ):
        return io.NodeOutput(
            *audit_motion_quality(
                frames,
                fps,
                roi_mode,
                roi_x,
                roi_y,
                roi_width,
                roi_height,
                sharpness_ratio_floor,
                temporal_instability_multiplier,
                high_motion_delta_floor,
                freeze_delta_ceiling,
                repair_context_frames,
                face_mask,
            )
        )


class MiniMaxH3MotionRepairPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionRepairPlanT8Advanced",
            display_name="MiniMax H3 Motion Repair Plan / 动态风险重做计划 (Advanced)",
            description=(
                "Maps Motion Quality Audit proxy ranges into the existing non-destructive "
                "Selective Repair Plan. It never renders, overwrites, or automatically accepts media."
            ),
            category=CATEGORY,
            inputs=[
                StudioTimelineIO.Input("timeline"),
                io.String.Input("audit_report_json", multiline=True),
                io.Combo.Input(
                    "audit_scope",
                    options=["single_shot", "full_timeline"],
                    default="single_shot",
                ),
                io.Int.Input(
                    "single_shot_index",
                    default=0,
                    min=0,
                    max=100000,
                ),
                io.Combo.Input(
                    "mapping_basis",
                    options=["suggested_repair_window", "raw_risk_range"],
                    default="suggested_repair_window",
                ),
                io.Combo.Input(
                    "repair_mode",
                    options=[
                        "auto",
                        "seed_retry",
                        "prompt_tighten",
                        "reference_refresh",
                        "full_regenerate",
                    ],
                    default="auto",
                ),
                io.String.Input("prompt_addendum", multiline=True, default=""),
                io.Int.Input(
                    "seed_stride",
                    default=1009,
                    min=1,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
                io.Int.Input(
                    "context_before_frames",
                    default=22,
                    min=0,
                    max=362,
                ),
                io.Int.Input(
                    "context_after_frames",
                    default=22,
                    min=0,
                    max=362,
                ),
            ],
            outputs=[
                RepairPlanIO.Output("repair_plan"),
                io.Int.Output("repair_count"),
                io.String.Output("repair_plan_json"),
                io.String.Output("mapping_report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        timeline,
        audit_report_json,
        audit_scope,
        single_shot_index,
        mapping_basis,
        repair_mode,
        prompt_addendum,
        seed_stride,
        context_before_frames,
        context_after_frames,
    ):
        return io.NodeOutput(
            *build_motion_repair_plan(
                timeline,
                audit_report_json,
                audit_scope,
                single_shot_index,
                mapping_basis,
                repair_mode,
                prompt_addendum,
                seed_stride,
                context_before_frames,
                context_after_frames,
            )
        )


MOTION_QUALITY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AVSigmaTailSubdivisionT8Advanced,
    MiniMaxH3MotionQualityAuditT8Advanced,
    MiniMaxH3AVSigmaSameNFERedistributionT8Advanced,
    MiniMaxH3MotionRepairPlanT8Advanced,
]
