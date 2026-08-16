from __future__ import annotations

from comfy_api.latest import io

from .face_refine_advanced import (
    CANVAS_OPTIONS,
    build_face_refine_plan,
    inject_face_refine_video_latent,
    local_face_detector_options,
    setup_face_refine_sampling,
    stitch_face_refine_candidate,
)
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental"
FaceRefinePlanIO = io.Custom("H3_T8_FACE_REFINE_PLAN")


class MiniMaxH3FaceRefinePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        detector_options = local_face_detector_options()
        return io.Schema(
            node_id="MiniMaxH3FaceRefinePlanT8Advanced",
            display_name="MiniMax H3 Face Refine Plan / 远景脸修复规划 (Advanced)",
            description=(
                "Builds a shot-aware, source-bound crop plan for a second H3 pass. "
                "It never downloads models or accepts a candidate automatically. The default "
                "OpenCV YuNet route runs a local MIT-licensed detector on CPU and destroys its "
                "detector object after planning; OpenCV may retain process-global CPU allocator "
                "pages. Manual ROI and local Ultralytics remain explicit alternatives."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames", tooltip="Exact source IMAGE batch at the stated fps."),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input(
                    "detector_mode",
                    options=[
                        "local_opencv_yunet",
                        "local_anime_onnx_exp",
                        "manual_static_roi",
                        "local_ultralytics",
                    ],
                    default="local_opencv_yunet",
                    tooltip=(
                        "OpenCV YuNet is the recommended real-person CPU route. The isolated "
                        "anime ONNX route is EXP; manual ROI remains the fallback. No route "
                        "downloads a model during execution."
                    ),
                ),
                io.Combo.Input(
                    "detector_model",
                    options=detector_options,
                    default=detector_options[0],
                    advanced=True,
                ),
                io.Combo.Input(
                    "detector_device",
                    options=["cpu", "cuda_auto"],
                    default="cpu",
                    advanced=True,
                ),
                io.Float.Input(
                    "confidence", default=0.35, min=0.01, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input("manual_roi_x", default=0.30, min=0.0, max=1.0, step=0.01),
                io.Float.Input("manual_roi_y", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Float.Input("manual_roi_width", default=0.40, min=0.01, max=1.0, step=0.01),
                io.Float.Input("manual_roi_height", default=0.55, min=0.01, max=1.0, step=0.01),
                io.Float.Input(
                    "scene_cut_threshold",
                    default=0.28,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Low-resolution RGB mean absolute difference; smoothing resets at every cut.",
                ),
                io.Float.Input(
                    "max_track_jump",
                    default=0.18,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input("max_gap_frames", default=4, min=0, max=48, advanced=True),
                io.Int.Input("smoothing_radius", default=2, min=0, max=24, advanced=True),
                io.Float.Input(
                    "crop_context_scale", default=3.0, min=1.2, max=8.0, step=0.1
                ),
                io.Combo.Input("canvas_size", options=list(CANVAS_OPTIONS), default="auto_512"),
                io.Boolean.Input("require_h3_grid", default=True),
                io.Int.Input(
                    "analysis_chunk_frames", default=8, min=1, max=64, advanced=True
                ),
            ],
            outputs=[
                FaceRefinePlanIO.Output("face_plan"),
                io.Image.Output("crops"),
                io.Image.Output("preview"),
                io.String.Output("report_json"),
                io.Int.Output("canvas_width"),
                io.Int.Output("canvas_height"),
                io.Int.Output("frame_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames,
        fps,
        detector_mode,
        detector_model,
        detector_device,
        confidence,
        manual_roi_x,
        manual_roi_y,
        manual_roi_width,
        manual_roi_height,
        scene_cut_threshold,
        max_track_jump,
        max_gap_frames,
        smoothing_radius,
        crop_context_scale,
        canvas_size,
        require_h3_grid,
        analysis_chunk_frames,
    ):
        return io.NodeOutput(
            *build_face_refine_plan(
                frames,
                fps,
                detector_mode,
                detector_model,
                detector_device,
                confidence,
                manual_roi_x,
                manual_roi_y,
                manual_roi_width,
                manual_roi_height,
                scene_cut_threshold,
                max_track_jump,
                max_gap_frames,
                smoothing_radius,
                crop_context_scale,
                canvas_size,
                require_h3_grid,
                analysis_chunk_frames,
            )
        )


class MiniMaxH3FaceRefineConditioningT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineConditioningT8Advanced",
            display_name="MiniMax H3 Face Refine Latent / 脸部二次生成条件 (Advanced)",
            description=(
                "Strictly replaces only the video stream of an existing H3 AV latent with "
                "the VAE-encoded crop sequence. Audio latent and noise-mask objects are preserved; "
                "no temporal trim or padding is allowed. Use a native BasicScheduler denoise value "
                "for the second pass."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("av_latent"),
                io.Image.Input("crops"),
                io.Vae.Input("video_vae"),
                FaceRefinePlanIO.Input("face_plan"),
                io.Combo.Input(
                    "audio_policy",
                    options=["require_locked", "preserve_existing"],
                    default="require_locked",
                    tooltip="require_locked refuses a missing or nonzero audio noise mask.",
                ),
                io.Boolean.Input(
                    "allow_multi_shot_exp",
                    default=False,
                    tooltip="Default refuses hard cuts; split the source into shot-local H3 windows.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        av_latent,
        crops,
        video_vae,
        face_plan,
        audio_policy,
        allow_multi_shot_exp,
    ):
        return io.NodeOutput(
            *inject_face_refine_video_latent(
                positive,
                av_latent,
                crops,
                video_vae,
                face_plan,
                audio_policy,
                allow_multi_shot_exp,
            )
        )


class MiniMaxH3FaceRefineStitchAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineStitchAuditT8Advanced",
            display_name="MiniMax H3 Face Refine Stitch Audit / 脸部回贴审计 (Advanced)",
            description=(
                "Creates a review candidate with source-bound geometry, local color matching, "
                "per-frame fallback and a bit-exact outside-mask audit. It never overwrites or "
                "accepts the source and does not modify audio."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("refined_crops"),
                FaceRefinePlanIO.Input("face_plan"),
                io.Combo.Input(
                    "paste_region",
                    options=["ellipse", "rectangle", "full_crop_exp"],
                    default="ellipse",
                ),
                io.Float.Input(
                    "feather_source_px", default=12.0, min=0.5, max=256.0, step=0.5
                ),
                io.Float.Input("blend_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "color_match_strength", default=0.65, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "max_face_mean_abs_delta",
                    default=0.40,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Frames above this normalized RGB change fall back to the source.",
                ),
                io.Int.Input(
                    "fallback_neighbor_frames", default=1, min=0, max=12, advanced=True
                ),
                io.Combo.Input(
                    "processing_device",
                    options=["cpu_memory_safe", "cuda_if_available"],
                    default="cpu_memory_safe",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("candidate_frames"),
                io.Mask.Output("changed_mask"),
                io.Mask.Output("fallback_mask"),
                io.Int.Output("fallback_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_frames,
        refined_crops,
        face_plan,
        paste_region,
        feather_source_px,
        blend_strength,
        color_match_strength,
        max_face_mean_abs_delta,
        fallback_neighbor_frames,
        processing_device,
    ):
        return io.NodeOutput(
            *stitch_face_refine_candidate(
                base_frames,
                refined_crops,
                face_plan,
                paste_region,
                feather_source_px,
                blend_strength,
                color_match_strength,
                max_face_mean_abs_delta,
                fallback_neighbor_frames,
                processing_device,
            )
        )


class MiniMaxH3FaceRefineSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineSamplerT8Advanced",
            display_name="MiniMax H3 Face Refine Sampler / 低去噪双时钟采样 (Advanced)",
            description=(
                "Builds an isolated low-denoise dual-clock schedule for the face-refine second "
                "pass. It reuses the stable T8 sampler implementation without changing its source "
                "or any existing workflow. Denoise is experimental and not a calibrated strength."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=12, min=1, max=1000),
                io.Float.Input("denoise", default=0.45, min=0.01, max=1.0, step=0.01),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input(
                    "sampler_name", options=SAMPLER_OPTIONS, default=DEFAULT_SAMPLER_NAME
                ),
                io.Combo.Input(
                    "scheduler", options=SCHEDULER_OPTIONS, default=DEFAULT_SCHEDULER_NAME
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
    def execute(
        cls,
        model,
        av_latent,
        steps,
        denoise,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    ):
        return io.NodeOutput(
            *setup_face_refine_sampling(
                model,
                av_latent,
                steps,
                denoise,
                shift_video,
                shift_audio,
                sampler_name,
                scheduler,
            )
        )


FACE_REFINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceRefinePlanT8Advanced,
    MiniMaxH3FaceRefineConditioningT8Advanced,
    MiniMaxH3FaceRefineSamplerT8Advanced,
    MiniMaxH3FaceRefineStitchAuditT8Advanced,
]
