from __future__ import annotations

from comfy_api.latest import io

from .detail_sampling_advanced import (
    apply_h3_spatiotemporal_guidance,
    build_tail_detail_schedule,
    setup_model_time_bias_sampling,
    setup_rectified_flow_restart_sampling,
    temporal_detail_enhance,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental"
_PROFILES = [
    "turbo_standard8",
    "turbo_ema8",
    "turbo_fl2v8",
    "stock20",
    "custom_strict",
]


class MiniMaxH3AVTailDetailScheduleT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVTailDetailScheduleT8Advanced",
            display_name="MiniMax H3 AV Tail Detail Schedule / 音画尾段细化 (Advanced)",
            description=(
                "Subdivides only the final H3 dual-clock interval. The default adds one "
                "joint AV model call; three adds a gradual 75/50/25% descent to zero. "
                "This does not inject random noise."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Int.Input(
                    "extra_tail_steps",
                    default=1,
                    min=0,
                    max=8,
                    tooltip="Additional full H3 joint audio-video forwards. Zero is exact pass-through.",
                ),
                io.Combo.Input(
                    "spacing",
                    options=[
                        "video_sigma_linear",
                        "video_sigma_cosine",
                        "base_flow_linear",
                    ],
                    default="video_sigma_linear",
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
                io.Combo.Input("profile", options=_PROFILES, default="turbo_standard8"),
            ],
            outputs=[
                io.Sigmas.Output("sigmas"),
                io.Int.Output("actual_nfe"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_tail_detail_schedule(**kwargs))


class MiniMaxH3ModelTimeBiasSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ModelTimeBiasSamplerT8Advanced",
            display_name="MiniMax H3 Smooth Model-Time Bias / 平滑模型时间偏置 (Advanced)",
            description=(
                "A scientifically explicit Navyblue/Detail-Daemon-style experiment for "
                "H3. It biases only the sigma visible to the shared AV Transformer under "
                "a smooth tail window; integration sigmas and NFE stay unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input(
                    "bias",
                    default=-0.05,
                    min=-0.5,
                    max=0.0,
                    step=0.005,
                    tooltip="Negative values make the model see a slightly cleaner time. No random noise is added.",
                ),
                io.Float.Input("start_progress", default=0.70, min=0.0, max=0.99, step=0.01),
                io.Float.Input("end_progress", default=1.0, min=0.01, max=1.0, step=0.01),
                io.Combo.Input(
                    "bias_domain",
                    options=["video_sigma", "base_flow"],
                    default="video_sigma",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*setup_model_time_bias_sampling(**kwargs))


class MiniMaxH3RectifiedFlowRestartSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RectifiedFlowRestartSamplerT8Advanced",
            display_name="MiniMax H3 Joint AV RF Restart / 联合音画流重启 (Advanced)",
            description=(
                "Runs the normal dual-clock trajectory, re-noises the clean endpoint with "
                "the rectified-flow interpolation, then performs a second joint AV descent. "
                "This is true stochastic restart and is higher risk than tail subdivision."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input(
                    "restart_video_sigma",
                    default=0.15,
                    min=0.0,
                    max=0.5,
                    step=0.005,
                ),
                io.Int.Input("restart_steps", default=3, min=0, max=8),
                io.Int.Input(
                    "restart_seed",
                    default=2608183001,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*setup_rectified_flow_restart_sampling(**kwargs))


class MiniMaxH3SpatioTemporalGuidanceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpatioTemporalGuidanceT8Advanced",
            display_name="MiniMax H3 Spatio-Temporal Guidance / 时空引导 (Advanced)",
            description=(
                "H3-specific skip-layer spatio-temporal guidance. It adds one perturbed "
                "joint AV forward at every active step and refuses block-replacement conflicts."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=0.60, min=0.0, max=5.0, step=0.05),
                io.String.Input("double_blocks", default="25", advanced=True),
                io.Float.Input("start_progress", default=0.25, min=0.0, max=0.99, step=0.01),
                io.Float.Input("end_progress", default=0.85, min=0.01, max=1.0, step=0.01),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "rescale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Must remain 0; shared AV global-std rescale is intentionally rejected.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*apply_h3_spatiotemporal_guidance(**kwargs))


class MiniMaxH3TemporalDetailEnhanceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TemporalDetailEnhanceT8Advanced",
            display_name="MiniMax H3 Temporal Detail Enhance / 时序保护细节增强 (Advanced)",
            description=(
                "Motion-gated luma detail enhancement with optional multiple-of-32 bicubic "
                "upscaling. It reduces sharpening in moving regions to limit flicker, does "
                "not touch audio, and cannot reconstruct missing faces or geometry."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Float.Input("upscale_factor", default=1.0, min=1.0, max=4.0, step=0.05),
                io.Float.Input("strength", default=0.35, min=0.0, max=2.0, step=0.05),
                io.Int.Input("blur_radius", default=2, min=0, max=8),
                io.Float.Input("blur_sigma", default=1.2, min=0.1, max=8.0, step=0.1),
                io.Float.Input(
                    "motion_threshold",
                    default=0.04,
                    min=0.001,
                    max=1.0,
                    step=0.001,
                ),
                io.Float.Input("temporal_guard", default=0.85, min=0.0, max=1.0, step=0.05),
                io.Int.Input(
                    "frame_chunk_size",
                    default=8,
                    min=1,
                    max=64,
                    advanced=True,
                    tooltip="Processes frames with a one-frame temporal halo to bound temporary memory.",
                ),
                io.Float.Input(
                    "maximum_output_megapixels",
                    default=2.1,
                    min=0.1,
                    max=64.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Fail-closed output pixel budget; this does not include the input tensor.",
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*temporal_detail_enhance(**kwargs))


DETAIL_SAMPLING_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AVTailDetailScheduleT8Advanced,
    MiniMaxH3ModelTimeBiasSamplerT8Advanced,
    MiniMaxH3RectifiedFlowRestartSamplerT8Advanced,
    MiniMaxH3SpatioTemporalGuidanceT8Advanced,
    MiniMaxH3TemporalDetailEnhanceT8Advanced,
]
