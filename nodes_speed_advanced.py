from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .speed_advanced import (
    H3ModalityStableNoise,
    accumulate_spectrum_dataset,
    build_spectrum_profile,
    build_speed_plan,
    build_speed_source,
    canonical_json,
    execute_speed_sampling,
    finalize_spectrum_dataset,
    prepare_speed_calibration_window,
)
from .speed_spectrum_storage import (
    load_spectrum_dataset_file,
    save_spectrum_dataset_file,
    sha256_file,
    spectrum_dataset_file_fingerprint,
)


CATEGORY = "T8/MiniMax H3/SPEED/Experimental"
SpeedProfileIO = io.Custom("H3_T8_SPEED_PROFILE")
SpeedSpectrumDatasetIO = io.Custom("H3_T8_SPEED_SPECTRUM_DATASET")
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


class MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced",
            display_name=(
                "MiniMax H3 SPEED Spectrum Dataset Accumulate / "
                "频谱数据集累积 (Advanced)"
            ),
            description=(
                "Accumulates exact per-clip H3 spatial power statistics across ComfyUI "
                "executions without retaining source or CUDA latents. Duplicate batch IDs "
                "and exact clip-spectrum repeats fail closed; model, VAE, task and latent "
                "contracts must remain identical."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "video_latent",
                    tooltip="Separated H3 video LATENT [B,24,T,H,W], never joint AV latent.",
                ),
                io.String.Input(
                    "batch_id",
                    default="batch_001",
                    tooltip="A unique stable ID for this actual source batch; repeats are rejected.",
                ),
                io.Combo.Input(
                    "task_family",
                    options=["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="T2VA",
                ),
                io.String.Input(
                    "checkpoint_fingerprint",
                    default="sha256:replace_with_real_checkpoint_fingerprint",
                ),
                io.String.Input(
                    "vae_fingerprint",
                    default="sha256:replace_with_real_vae_fingerprint",
                ),
                io.Int.Input(
                    "max_temporal_samples", default=32, min=1, max=512, advanced=True
                ),
                SpeedSpectrumDatasetIO.Input("previous_dataset", optional=True),
                io.String.Input(
                    "dataset_provenance_json",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Optional reviewed natural-corpus provenance. Appended after the legacy "
                        "inputs so existing widget positions remain unchanged."
                    ),
                ),
                io.String.Input(
                    "source_entry_json",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Optional manifest-bound source and decoded-window hashes for this batch."
                    ),
                ),
            ],
            outputs=[
                SpeedSpectrumDatasetIO.Output("spectrum_dataset"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, **kwargs):
        samples = video_latent.get("samples") if isinstance(video_latent, dict) else None
        return io.NodeOutput(*accumulate_spectrum_dataset(samples, **kwargs))


class MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced",
            display_name=(
                "MiniMax H3 SPEED Spectrum Dataset Finalize / "
                "频谱数据集定稿 (Advanced)"
            ),
            description=(
                "Fits one task/model/VAE-bound H3 SPEED profile from accumulated sufficient "
                "statistics. Fewer than 100 unique clips or a weak fit remains a research "
                "probe and cannot authorize the validated delta-optimal Plan mode."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                SpeedSpectrumDatasetIO.Input("spectrum_dataset"),
                io.String.Input("profile_name", default="h3_dataset_spectrum_v1"),
                io.Float.Input(
                    "minimum_r_squared", default=0.80, min=0.0, max=1.0, step=0.01
                ),
                io.Int.Input(
                    "minimum_independent_clips",
                    default=100,
                    min=100,
                    max=1000000,
                    advanced=True,
                ),
            ],
            outputs=[
                SpeedProfileIO.Output("spectrum_profile"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, spectrum_dataset, **kwargs):
        return io.NodeOutput(*finalize_spectrum_dataset(spectrum_dataset, **kwargs))


class MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced",
            display_name=(
                "MiniMax H3 SPEED Spectrum Dataset File / "
                "频谱数据集文件 (Advanced)"
            ),
            description=(
                "Loads or explicitly saves one accumulated spectrum dataset under the "
                "ComfyUI output directory. Save is atomic, stores no source latent/video, "
                "requires confirmation and refuses silent overwrite or unsafe paths."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Combo.Input("mode", options=["load", "save"], default="load"),
                io.String.Input("dataset_name", default="h3_t2va_spectrum_dataset_v1"),
                io.Boolean.Input("overwrite", default=False),
                io.Boolean.Input("confirm_write", default=False),
                SpeedSpectrumDatasetIO.Input("spectrum_dataset", optional=True),
            ],
            outputs=[
                SpeedSpectrumDatasetIO.Output("spectrum_dataset"),
                io.String.Output("file_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mode,
        dataset_name,
        overwrite,
        confirm_write,
        spectrum_dataset=None,
    ):
        root = Path(folder_paths.get_output_directory()) / "h3_speed_spectrum_datasets"
        if mode == "load":
            return io.NodeOutput(
                *load_spectrum_dataset_file(root=root, dataset_name=dataset_name)
            )
        if mode != "save":
            raise ValueError("mode must be load or save")
        if not bool(confirm_write):
            raise ValueError("Saving a spectrum dataset requires confirm_write=true")
        if spectrum_dataset is None:
            raise ValueError("save mode requires spectrum_dataset")
        return io.NodeOutput(
            *save_spectrum_dataset_file(
                spectrum_dataset,
                root=root,
                dataset_name=dataset_name,
                overwrite=bool(overwrite),
            )
        )

    @classmethod
    def fingerprint_inputs(cls, mode, dataset_name, **_kwargs):
        if mode == "save":
            # Saving is an explicit side effect. Never allow ComfyUI to reuse an older save.
            return float("nan")
        if mode != "load":
            return f"unsupported-mode:{mode}"
        root = Path(folder_paths.get_output_directory()) / "h3_speed_spectrum_datasets"
        return spectrum_dataset_file_fingerprint(root=root, dataset_name=dataset_name)


class MiniMaxH3SPEEDModelVAEFingerprintT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDModelVAEFingerprintT8Advanced",
            display_name=(
                "MiniMax H3 SPEED Model + VAE Fingerprint / "
                "模型与VAE指纹 (Advanced)"
            ),
            description=(
                "Streams the complete selected diffusion checkpoint and video VAE files to "
                "produce binding SHA-256 fingerprints. It does not load model weights onto "
                "GPU; first execution can be disk-I/O heavy and ComfyUI may cache unchanged inputs."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "checkpoint_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
                io.Combo.Input(
                    "video_vae_name", options=folder_paths.get_filename_list("vae")
                ),
            ],
            outputs=[
                io.String.Output("checkpoint_fingerprint"),
                io.String.Output("vae_fingerprint"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, checkpoint_name, video_vae_name):
        checkpoint_path = Path(
            folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
        )
        vae_path = Path(folder_paths.get_full_path_or_raise("vae", video_vae_name))
        checkpoint_fingerprint = sha256_file(checkpoint_path)
        vae_fingerprint = sha256_file(vae_path)
        report = {
            "schema": "minimax_h3_speed_model_vae_fingerprint_t8_v1",
            "checkpoint": {
                "name": checkpoint_name,
                "bytes": checkpoint_path.stat().st_size,
                "fingerprint": checkpoint_fingerprint,
            },
            "video_vae": {
                "name": video_vae_name,
                "bytes": vae_path.stat().st_size,
                "fingerprint": vae_fingerprint,
            },
            "gpu_model_loaded": False,
            "hash_scope": "complete_file_bytes",
        }
        return io.NodeOutput(
            checkpoint_fingerprint, vae_fingerprint, canonical_json(report)
        )


class MiniMaxH3SPEEDCalibrationWindowT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPEEDCalibrationWindowT8Advanced",
            display_name=(
                "MiniMax H3 SPEED Calibration Window / "
                "画幅安全标定窗口 (Advanced)"
            ),
            description=(
                "Resamples one strict H3 24fps/17n+5 calibration window and uses "
                "aspect-preserving center-cover resize. It never stretches source geometry, "
                "never pads short clips and does not alter the existing Source Media Window."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Float.Input(
                    "source_fps", default=24.0, min=0.01, max=1000.0, step=0.001
                ),
                io.Int.Input("width", default=736, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=416, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Float.Input(
                    "start_seconds", default=0.0, min=0.0, max=86400.0, step=0.001
                ),
                io.Combo.Input(
                    "resize_mode", options=["center_cover"], default="center_cover"
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Int.Output("frame_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*prepare_speed_calibration_window(**kwargs))


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
    MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced,
    MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced,
    MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced,
    MiniMaxH3SPEEDModelVAEFingerprintT8Advanced,
    MiniMaxH3SPEEDCalibrationWindowT8Advanced,
]
