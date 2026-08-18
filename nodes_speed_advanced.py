from __future__ import annotations

from comfy_api.latest import io

from .speed_advanced import (
    H3ModalityStableNoise,
    build_spectrum_profile,
    build_speed_plan,
    build_speed_source,
    execute_speed_sampling,
)


CATEGORY = "T8/MiniMax H3/SPEED/Experimental"
SpeedProfileIO = io.Custom("H3_T8_SPEED_PROFILE")
SpeedPlanIO = io.Custom("H3_T8_SPEED_PLAN")
SpeedSourceIO = io.Custom("H3_T8_SPEED_SOURCE")
MAX_RESOLUTION = 16384


class MiniMaxH3SPEEDModalityStableNoiseT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDModalityStableNoiseT8Advanced",
            display_name="MiniMax H3 Modality-Stable AV Noise / AV独立随机噪声 (Advanced)",
            description=(
                "Generates deterministic video and audio noise from separate derived seeds. "
                "Changing only the video canvas therefore cannot silently change the audio "
                "noise stream. Intended for controlled H3 SPEED baselines and diagnostics."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Int.Input(
                    "seed",
                    default=2608184001,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[io.Noise.Output("noise")],
        )

    @classmethod
    def execute(cls, seed):
        return io.NodeOutput(H3ModalityStableNoise(seed))


class MiniMaxH3SPEEDSpectrumHarvesterT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSpectrumHarvesterT8Advanced",
            display_name="MiniMax H3 SPEED Spectrum Harvester / 空间频谱标定 (Advanced)",
            description=(
                "Fits P(omega)=A|omega|^-beta from H3 VIDEO LATENT samples. It never "
                "substitutes WAN/FLUX constants. A single clip remains a research probe; "
                "dataset status requires at least 100 actual independent batch entries."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "video_latent",
                    tooltip="Separated MiniMax H3 video LATENT [B,24,T,H,W], not the joint AV latent.",
                ),
                io.String.Input("profile_name", default="h3_local_spectrum_probe"),
                io.Combo.Input(
                    "task_family",
                    options=["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="T2VA",
                ),
                io.String.Input(
                    "checkpoint_fingerprint",
                    default="unrecorded",
                    tooltip="Record a checkpoint SHA/header fingerprint; filenames alone are not proof.",
                ),
                io.String.Input("vae_fingerprint", default="unrecorded"),
                io.Int.Input(
                    "independent_clip_count",
                    default=1,
                    min=1,
                    max=1000000,
                    tooltip=(
                        "Must not exceed the actual latent batch count. Typing 100 for one clip "
                        "does not promote it; dataset status needs 100 actual batch entries whose "
                        "independence remains a dataset provenance assertion."
                    ),
                ),
                io.Float.Input(
                    "minimum_r_squared", default=0.80, min=0.0, max=1.0, step=0.01
                ),
                io.Int.Input(
                    "max_temporal_samples", default=32, min=1, max=512, advanced=True
                ),
            ],
            outputs=[
                SpeedProfileIO.Output("spectrum_profile"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video_latent,
        profile_name,
        task_family,
        checkpoint_fingerprint,
        vae_fingerprint,
        independent_clip_count,
        minimum_r_squared,
        max_temporal_samples,
    ):
        samples = video_latent.get("samples") if isinstance(video_latent, dict) else None
        return io.NodeOutput(
            *build_spectrum_profile(
                samples,
                profile_name=profile_name,
                task_family=task_family,
                checkpoint_fingerprint=checkpoint_fingerprint,
                vae_fingerprint=vae_fingerprint,
                independent_clip_count=independent_clip_count,
                minimum_r_squared=minimum_r_squared,
                max_temporal_samples=max_temporal_samples,
            )
        )


class MiniMaxH3SPEEDPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDPlanT8Advanced",
            display_name="MiniMax H3 SPEED Plan / 渐进分辨率计划 (Advanced)",
            description=(
                "Builds a multiple-of-32 H3 spatial progression with the official SPEED "
                "kappa and sigma-alignment equations. Manual sigmas are the safe default "
                "until an H3 dataset spectrum is validated. NFE is preserved exactly."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Int.Input("width", default=1056, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=608, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("steps", default=20, min=2, max=1000),
                io.String.Input(
                    "scales",
                    default="0.5,1.0",
                    tooltip="Strictly increasing spatial scales ending in 1.0.",
                ),
                io.Combo.Input(
                    "transition_mode",
                    options=["manual_sigmas", "delta_optimal"],
                    default="manual_sigmas",
                ),
                io.String.Input(
                    "manual_transition_sigmas",
                    default="0.85",
                    tooltip="One descending public H3 video sigma per resolution transition.",
                ),
                io.Float.Input("delta", default=0.01, min=0.0001, max=0.5, step=0.001),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input("transform", options=["dct"], default="dct"),
                io.Combo.Input(
                    "profile_policy",
                    options=["require_validated_profile", "allow_research_profile"],
                    default="require_validated_profile",
                    advanced=True,
                ),
                io.Combo.Input(
                    "fallback_policy",
                    options=["error", "full_resolution_passthrough"],
                    default="error",
                    advanced=True,
                ),
                SpeedProfileIO.Input("spectrum_profile", optional=True),
            ],
            outputs=[SpeedPlanIO.Output("speed_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_speed_plan(**kwargs))


class MiniMaxH3SPEEDSourceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSourceT8Advanced",
            display_name="MiniMax H3 SPEED Stage Source / 分阶段条件源 (Advanced)",
            description=(
                "Retains raw H3 text/media inputs so every SPEED stage can resize and "
                "re-encode keyframes/references for its own canvas. It does not load a "
                "second H3 model and does not alter the stable Conditioning node."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="24fps; the H3 builder snaps to the 17n+5 grid.",
                ),
                io.Combo.Input(
                    "task_type",
                    options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="auto",
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["lock_source", "remix_source", "reference_only", "native"],
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
                io.Boolean.Input("add_source_as_reference", default=True),
                io.Int.Input(
                    "prompt_primary_audio_ordinal",
                    default=1,
                    min=0,
                    max=9,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match", advanced=True
                ),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.String.Input(
                    "checkpoint_fingerprint",
                    default="unrecorded",
                    advanced=True,
                    tooltip=(
                        "SHA/header fingerprint used to bind a delta-optimal spectrum profile. "
                        "Manual-sigma plans do not require it."
                    ),
                ),
                io.String.Input(
                    "vae_fingerprint",
                    default="unrecorded",
                    advanced=True,
                    tooltip="Video-VAE fingerprint used to bind a delta-optimal spectrum profile.",
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
            ],
            outputs=[SpeedSourceIO.Output("speed_source"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_speed_source(**kwargs))


class MiniMaxH3SPEEDSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSamplerT8Advanced",
            display_name="MiniMax H3 SPEED Whole-Chain Sampler / 整链采样 (Advanced)",
            description=(
                "Runs native H3 Euler sampling over spatial stages, rebuilding AV layout and "
                "conditioning at every canvas. Video receives official DCT expansion. Audio "
                "stays spatially untouched and is reindexed on the shared public H3 flow; this "
                "H3-specific audio extension remains EXP until real GPU listening tests pass."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                SpeedPlanIO.Input("speed_plan"),
                SpeedSourceIO.Input("speed_source"),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Int.Input(
                    "seed",
                    default=2608184001,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Combo.Input(
                    "execution_scope",
                    options=[
                        "strict_t2va_stock20",
                        "multimodal_research_exp",
                        "turbo8_t2va_research_exp",
                    ],
                    default="strict_t2va_stock20",
                    tooltip=(
                        "Strict mode requires media-free native-audio T2VA, exactly 20 steps, shifts 12/3, "
                        "and an unpatched stock H3 model. Multimodal research mode "
                        "enables stage-rebuilt I/FL/L/Ref/Hybrid mechanics. Turbo8 research mode "
                        "requires media-free T2VA, exactly 8 steps and a compatible weight-patched MODEL; "
                        "the node cannot prove a LoRA file's identity from patch tensors."
                    ),
                ),
                io.Int.Input(
                    "dct_chunk_size",
                    default=64,
                    min=1,
                    max=1024,
                    advanced=True,
                    tooltip="Number of B*C*T spatial slices transformed per chunk.",
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Audio.Output("mux_audio"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("media_map_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*execute_speed_sampling(**kwargs))


SPEED_ADVANCED_NODE_CLASSES = [
    MiniMaxH3SPEEDSpectrumHarvesterT8Advanced,
    MiniMaxH3SPEEDPlanT8Advanced,
    MiniMaxH3SPEEDSourceT8Advanced,
    MiniMaxH3SPEEDSamplerT8Advanced,
    MiniMaxH3SPEEDModalityStableNoiseT8Advanced,
]
