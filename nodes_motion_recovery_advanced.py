from __future__ import annotations

from comfy_api.latest import io

from .motion_recovery_advanced import (
    analyze_motion_overload,
    collect_motion_window,
    compose_motion_recovery,
    plan_motion_segment,
    prepare_motion_retiming,
    recover_motion_av,
    route_automatic_motion_recovery,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Motion Recovery"
MotionPlanIO = io.Custom("H3_T8_MOTION_RECOVERY_PLAN")


class MiniMaxH3MotionOverloadAnalyzeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionOverloadAnalyzeT8Advanced",
            display_name="MiniMax H3 Motion Overload Analyze / 动作过载分析 (Advanced)",
            description=(
                "Read-only H3 latent and decoded-frame analysis. It separates motion measurement "
                "from compute allocation, can abstain on calm clips, and never patches MODEL or sampling."
            ),
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Image.Input("frames"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "auto_conservative_exp", "manual_ranges"],
                    default="auto_conservative_exp",
                    tooltip=(
                        "report_only never expands frames. auto_conservative_exp applies only when "
                        "all abstain gates pass. manual_ranges uses the ranges field exactly."
                    ),
                ),
                io.String.Input(
                    "manual_ranges",
                    default="",
                    tooltip="Examples: 36-60:4 or 1.5s-2.4s:3; used only in manual_ranges mode.",
                ),
                io.Float.Input(
                    "overload_threshold", default=3.5, min=0.5, max=20.0, step=0.1
                ),
                io.Float.Input(
                    "minimum_profile_contrast",
                    default=2.0,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_residual_motion",
                    default=0.01,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Absolute decoded-frame residual gate after a small global-translation search. "
                        "This prevents a quantile-only rule from always selecting something."
                    ),
                ),
                io.Int.Input("max_hold", default=4, min=2, max=8),
                io.Int.Input("bridge_tokens", default=2, min=0, max=20, advanced=True),
                io.Int.Input("minimum_hot_frames", default=5, min=1, max=362, advanced=True),
                io.Float.Input("fps", default=24.0, min=24.0, max=24.0, step=1.0, advanced=True),
            ],
            outputs=[
                MotionPlanIO.Output("motion_plan"),
                io.Image.Output("heat_preview"),
                io.Boolean.Output("should_repair"),
                io.Int.Output("expanded_frames"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        frames,
        mode,
        manual_ranges,
        overload_threshold,
        minimum_profile_contrast,
        minimum_residual_motion,
        max_hold,
        bridge_tokens,
        minimum_hot_frames,
        fps,
    ):
        return io.NodeOutput(
            *analyze_motion_overload(
                av_latent,
                frames,
                mode,
                manual_ranges,
                overload_threshold,
                minimum_profile_contrast,
                minimum_residual_motion,
                max_hold,
                bridge_tokens,
                minimum_hot_frames,
                fps,
            )
        )


class MiniMaxH3MotionRetimingPrepareT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionRetimingPrepareT8Advanced",
            display_name="MiniMax H3 Motion Retiming Prepare / 动作扩时二采准备 (Advanced)",
            description=(
                "Expands only the frames selected by a signed motion plan, encodes the longer H3 "
                "video latent, and optionally seeds the shared AV Transformer with a phase-vocoder "
                "guide made from pass-1 audio. Keep final audio on pass1_original unless explicitly reviewed."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                MotionPlanIO.Input("motion_plan"),
                io.Vae.Input("video_vae"),
                io.Combo.Input(
                    "audio_seed_mode",
                    options=[
                        "follow_original_0p5",
                        "follow_original_0p7",
                        "lock_original_0p0",
                        "none_invent_exp",
                    ],
                    default="follow_original_0p5",
                ),
                io.Vae.Input("audio_vae", optional=True),
                io.Audio.Input("source_audio", optional=True),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Image.Output("smeared_frames"),
                io.Audio.Output("smeared_audio_seed"),
                MotionPlanIO.Output("motion_plan"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        frames,
        motion_plan,
        video_vae,
        audio_seed_mode,
        audio_vae=None,
        source_audio=None,
    ):
        return io.NodeOutput(
            *prepare_motion_retiming(
                frames,
                motion_plan,
                video_vae,
                audio_vae,
                source_audio,
                audio_seed_mode,
            )
        )


class MiniMaxH3MotionRecoveryComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionRecoveryComposerT8Advanced",
            display_name="MiniMax H3 Motion Recovery Composer / 动作二采编排 (Advanced)",
            description=(
                "Authenticates the prepared AV latent and signed plan, then selects the final "
                "fraction of an existing H3 sigma schedule for the second V2V pass. It installs "
                "no MODEL wrapper and never auto-stacks EAV, STG, Restart or BlockCache."
            ),
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Sigmas.Input("sigmas"),
                MotionPlanIO.Input("motion_plan"),
                io.Combo.Input(
                    "mode", options=["report_only", "apply_exp"], default="report_only"
                ),
                io.Float.Input(
                    "denoise_fraction",
                    default=0.48,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of the existing schedule retained for pass 2; not a sigma value.",
                ),
                io.Int.Input(
                    "minimum_second_pass_nfe", default=2, min=1, max=100, advanced=True
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Sigmas.Output("sigmas"),
                MotionPlanIO.Output("motion_plan"),
                io.Int.Output("actual_second_pass_nfe"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        sigmas,
        motion_plan,
        mode,
        denoise_fraction,
        minimum_second_pass_nfe,
    ):
        return io.NodeOutput(
            *compose_motion_recovery(
                av_latent,
                sigmas,
                motion_plan,
                mode,
                denoise_fraction,
                minimum_second_pass_nfe,
            )
        )


class MiniMaxH3MotionRecoverAVT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionRecoverAVT8Advanced",
            display_name="MiniMax H3 Motion Recover AV / 动作音画回时钟 (Advanced)",
            description=(
                "Returns the regenerated clip to its exact world-frame clock by selecting one real "
                "generated frame from every hold group. The safe default keeps pass-1 audio and "
                "does not phase-vocode the deliverable soundtrack."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("generated_frames"),
                io.Audio.Input("generated_audio"),
                io.Audio.Input("pass1_audio"),
                MotionPlanIO.Input("motion_plan"),
                io.Combo.Input(
                    "audio_mode",
                    options=["pass1_original", "pass2_recovered_exp", "blend_exp"],
                    default="pass1_original",
                    tooltip=(
                        "pass1_original is the stable exact-pass1 default. A full human review "
                        "rejected pass2_recovered_exp because its middle window suddenly sounded "
                        "distant before returning; keep it diagnostic-only. blend_exp sounded "
                        "normal in that one clip at pass1_mix=0.8 but remains opt-in."
                    ),
                ),
                io.Float.Input(
                    "pass1_mix",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip=(
                        "Used only by blend_exp; 1.0 is all pass-1 audio. One reviewed I2VA clip "
                        "passed at 0.8, which is not a general audio guarantee."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        generated_frames,
        generated_audio,
        pass1_audio,
        motion_plan,
        audio_mode,
        pass1_mix,
    ):
        return io.NodeOutput(
            *recover_motion_av(
                generated_frames,
                generated_audio,
                pass1_audio,
                motion_plan,
                audio_mode,
                pass1_mix,
            )
        )


class MiniMaxH3MotionSegmentPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionSegmentPlanT8Advanced",
            display_name="MiniMax H3 Motion Segment Plan / 动作分段窗口计划 (Advanced)",
            description=(
                "Turns one signed full-clip motion plan into budgeted sequential windows. It does "
                "not impose a global pixel/frame prohibition; it reports hot cuts so the user can "
                "raise the budget or lower max_hold before rendering."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                io.Audio.Input("baseline_audio"),
                MotionPlanIO.Input("motion_plan"),
                io.Int.Input(
                    "max_expanded_frames", default=209, min=39, max=10000, step=17
                ),
                io.Int.Input(
                    "window_index",
                    default=0,
                    min=0,
                    max=999,
                    control_after_generate=True,
                ),
                io.Int.Input("handle_frames", default=12, min=2, max=48),
                io.Combo.Input(
                    "coverage",
                    options=["hot_ranges_only", "full_clip"],
                    default="hot_ranges_only",
                ),
            ],
            outputs=[
                io.Image.Output("segment_frames"),
                MotionPlanIO.Output("window_plan"),
                io.Image.Output("first_frame"),
                io.Image.Output("last_frame"),
                io.Audio.Output("segment_audio"),
                io.Int.Output("window_count"),
                io.Int.Output("window_index"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        frames,
        baseline_audio,
        motion_plan,
        max_expanded_frames,
        window_index,
        handle_frames,
        coverage,
    ):
        return io.NodeOutput(
            *plan_motion_segment(
                frames,
                baseline_audio,
                motion_plan,
                max_expanded_frames,
                window_index,
                handle_frames,
                coverage,
            )
        )


class MiniMaxH3MotionWindowCollectT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionWindowCollectT8Advanced",
            display_name="MiniMax H3 Motion Window Collect / 动作窗口断点收集 (Advanced)",
            description=(
                "Banks one recovered window per queue item and deterministically assembles all "
                "signed windows when complete. Missing windows pass the baseline through, so a "
                "cancel, crash or OOM requires rerendering only the missing item."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("baseline_frames"),
                io.Audio.Input("baseline_audio"),
                io.Image.Input("recovered_segment"),
                io.Audio.Input("recovered_audio"),
                MotionPlanIO.Input("window_plan"),
                io.String.Input("run_name", default="motion_window_run"),
                io.String.Input(
                    "store_dir",
                    default="",
                    advanced=True,
                    tooltip="Empty uses ComfyUI/output/MiniMaxH3_Motion_Windows.",
                ),
                io.Boolean.Input("write_window", default=True),
                io.Combo.Input(
                    "store_dtype",
                    options=["float32_exact", "float16_half_disk"],
                    default="float32_exact",
                    advanced=True,
                    tooltip=(
                        "float32_exact preserves recovered frame values; float16_half_disk halves "
                        "the frame-bank storage but is not pixel-exact."
                    ),
                ),
                io.Int.Input("feather_frames", default=6, min=0, max=24),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.Boolean.Output("complete"),
                io.Int.Output("windows_on_disk"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        baseline_frames,
        baseline_audio,
        recovered_segment,
        recovered_audio,
        window_plan,
        run_name,
        store_dir,
        write_window,
        store_dtype,
        feather_frames,
    ):
        return io.NodeOutput(
            *collect_motion_window(
                baseline_frames,
                baseline_audio,
                recovered_segment,
                recovered_audio,
                window_plan,
                run_name,
                store_dir,
                write_window,
                store_dtype,
                feather_frames,
            )
        )


class MiniMaxH3MotionAutoGateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionAutoGateT8Advanced",
            display_name="MiniMax H3 Motion Auto Gate / 动作自动修复旁路 (Advanced)",
            description=(
                "Uses ComfyUI native lazy inputs. A calm clip marked ABSTAIN returns the exact "
                "pass-1 frames/audio without evaluating the connected second-pass graph; only "
                "should_repair=true requests the repaired inputs."
            ),
            category=CATEGORY,
            inputs=[
                io.Boolean.Input("should_repair"),
                io.Image.Input("baseline_frames"),
                io.Audio.Input("baseline_audio"),
                MotionPlanIO.Input("motion_plan"),
                io.Image.Input("repaired_frames", lazy=True),
                io.Audio.Input("repaired_audio", lazy=True),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.Boolean.Output("did_repair"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def check_lazy_status(
        cls,
        should_repair,
        baseline_frames,
        baseline_audio,
        motion_plan,
        repaired_frames=None,
        repaired_audio=None,
    ):
        del baseline_frames, baseline_audio, motion_plan
        if not bool(should_repair):
            return []
        missing = []
        if repaired_frames is None:
            missing.append("repaired_frames")
        if repaired_audio is None:
            missing.append("repaired_audio")
        return missing

    @classmethod
    def execute(
        cls,
        should_repair,
        baseline_frames,
        baseline_audio,
        motion_plan,
        repaired_frames=None,
        repaired_audio=None,
    ):
        return io.NodeOutput(
            *route_automatic_motion_recovery(
                should_repair,
                baseline_frames,
                baseline_audio,
                motion_plan,
                repaired_frames,
                repaired_audio,
            )
        )


MOTION_RECOVERY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3MotionOverloadAnalyzeT8Advanced,
    MiniMaxH3MotionRetimingPrepareT8Advanced,
    MiniMaxH3MotionRecoveryComposerT8Advanced,
    MiniMaxH3MotionRecoverAVT8Advanced,
    MiniMaxH3MotionSegmentPlanT8Advanced,
    MiniMaxH3MotionWindowCollectT8Advanced,
    MiniMaxH3MotionAutoGateT8Advanced,
]
