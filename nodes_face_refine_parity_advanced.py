from __future__ import annotations

from comfy_api.latest import io

from .face_refine_advanced import local_face_detector_options
from .face_refine_parity_advanced import (
    MANUAL512_RELATIVE_PROFILE,
    PARITY_CANVAS_MODES,
    apply_face_refine_per_frame_denoise,
    build_face_refine_parity_plan,
    gate_face_refine_parity_candidate,
    inject_face_refine_parity_video_latent,
    stitch_face_refine_parity_candidate,
    validate_face_refine_manual512_relative_baseline,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Face Refine Parity"
FaceRefineParityPlanIO = io.Custom("H3_T8_FACE_REFINE_PARITY_PLAN")


class MiniMaxH3FaceRefineParityPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        detector_options = local_face_detector_options()
        return io.Schema(
            node_id="MiniMaxH3FaceRefineParityPlanT8Advanced",
            display_name="MiniMax H3 Face Refine Parity Plan / 原版机制规划 (Advanced)",
            description=(
                "Builds the isolated upstream-parity crop contract: audited BGR YOLO input, "
                "separate 21/51-frame Gaussian center/size smoothing, per-frame float crops and "
                "the author's centred FaceDetailer-style mask geometry. The best source crop is "
                "only a convenient reference, not identity proof."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
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
                ),
                io.Combo.Input(
                    "detector_model",
                    options=detector_options,
                    default=detector_options[0],
                    advanced=True,
                ),
                io.Combo.Input(
                    "detector_device", options=["cpu", "cuda_auto"], default="cpu", advanced=True
                ),
                io.Float.Input("confidence", default=0.35, min=0.01, max=1.0, step=0.01),
                io.Float.Input("manual_roi_x", default=0.30, min=0.0, max=1.0, step=0.01),
                io.Float.Input("manual_roi_y", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Float.Input("manual_roi_width", default=0.40, min=0.01, max=1.0, step=0.01),
                io.Float.Input("manual_roi_height", default=0.55, min=0.01, max=1.0, step=0.01),
                io.Float.Input("scene_cut_threshold", default=0.28, min=0.01, max=1.0, step=0.01),
                io.Float.Input("max_track_jump", default=0.18, min=0.01, max=1.0, step=0.01),
                io.Int.Input("max_gap_frames", default=4, min=0, max=48),
                io.Int.Input("center_smooth_window", default=21, min=1, max=121, step=2),
                io.Int.Input("size_smooth_window", default=51, min=1, max=181, step=2),
                io.Float.Input("crop_factor", default=3.0, min=1.2, max=8.0, step=0.1),
                io.Combo.Input(
                    "canvas_mode", options=list(PARITY_CANVAS_MODES), default="auto_capped_768"
                ),
                io.Boolean.Input("require_h3_grid", default=True),
                io.Int.Input("analysis_chunk_frames", default=8, min=1, max=64, advanced=True),
            ],
            outputs=[
                FaceRefineParityPlanIO.Output("face_plan"),
                io.Image.Output("crops"),
                io.Image.Output("reference_crop"),
                io.Image.Output("preview"),
                io.String.Output("report_json"),
                io.Int.Output("canvas_width"),
                io.Int.Output("canvas_height"),
                io.Int.Output("frame_count"),
                io.Int.Output("reference_frame_index"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_face_refine_parity_plan(**kwargs))


class MiniMaxH3FaceRefineParityLatentT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineParityLatentT8Advanced",
            display_name="MiniMax H3 Face Refine Parity Latent / 原版机制Latent (Advanced)",
            description=(
                "Injects only the crop video latent into an existing MiniMax H3 AV latent. "
                "The audio latent and existing mask object are retained exactly. The real source "
                "frame batch is encoded directly; the VAE must naturally match the declared H3 "
                "latent time. Arbitrary pixel-tail duplication, trim, pad or resize is forbidden."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("av_latent"),
                io.Image.Input("crops"),
                io.Vae.Input("video_vae"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.Combo.Input(
                    "audio_policy",
                    options=["require_locked", "preserve_existing"],
                    default="require_locked",
                ),
                io.Boolean.Input("allow_multi_shot_exp", default=False, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*inject_face_refine_parity_video_latent(**kwargs))


class MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced",
            display_name="MiniMax H3 Face Refine Per-Frame Denoise / 逐帧去噪 (Advanced)",
            description=(
                "Applies the audited upstream face-size curve (crop height divided by crop "
                "factor) to the video noise mask only. "
                "Defaults map a 30px face to 0.8 and a 120px face to 0.35 with 9-frame "
                "smoothing. The nested audio mask stays exactly zero."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.Float.Input("strength_small_face", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Float.Input("strength_large_face", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "scale_mode", options=["absolute_px", "relative_to_clip"], default="absolute_px"
                ),
                io.Float.Input("face_px_small", default=30.0, min=1.0, max=2048.0, step=1.0),
                io.Float.Input("face_px_large", default=120.0, min=2.0, max=4096.0, step=1.0),
                io.Float.Input("gamma", default=1.0, min=0.05, max=8.0, step=0.05),
                io.Int.Input("smooth_frames", default=9, min=1, max=121, step=2),
                io.Combo.Input(
                    "video_mask_mode",
                    options=["replace_video_parity", "cap_existing"],
                    default="replace_video_parity",
                ),
                io.Boolean.Input("require_locked_audio", default=True),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*apply_face_refine_per_frame_denoise(**kwargs))


class MiniMaxH3FaceRefineParityStitchT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineParityStitchT8Advanced",
            display_name="MiniMax H3 Face Refine Parity Stitch / 原版机制回贴 (Advanced)",
            description=(
                "Creates a non-destructive review candidate with the audited upstream mechanics: "
                "centred face rectangle, chunk-midpoint 24px source feather and post-warp source-"
                "coordinate colour match. Pixels outside the mask are audited bit-exact; audio "
                "is not touched."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("refined_crops"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.Combo.Input(
                    "paste_region",
                    options=["face_only", "face_ellipse", "full_crop_exp"],
                    default="face_only",
                ),
                io.Int.Input("mask_dilation", default=24, min=0, max=256),
                io.Float.Input("feather_source_px", default=24.0, min=0.0, max=256.0, step=0.5),
                io.Float.Input("colour_match", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("blend", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "undetected_frames",
                    options=["fade_out", "skip", "composite_anyway"],
                    default="fade_out",
                ),
                io.Float.Input(
                    "max_face_mean_abs_delta",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
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
    def execute(cls, **kwargs):
        return io.NodeOutput(*stitch_face_refine_parity_candidate(**kwargs))


class MiniMaxH3FaceRefineQualityGateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineQualityGateT8Advanced",
            display_name="MiniMax H3 Face Refine Quality Gate / 候选质量门 (Advanced)",
            description=(
                "Rejects obvious Face Refine regressions before export. It accepts only "
                "continuous runs whose source-relative structure, face delta, measured "
                "sharpness and temporal residual all pass conservative proxy thresholds; "
                "rejected frames return to the original source. Passing is not identity or "
                "quality proof and the result still requires full-video human review."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("changed_mask"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.Float.Input(
                    "min_structure_ssim", default=0.82, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "min_sharpness_ratio", default=1.02, min=0.0, max=10.0, step=0.01
                ),
                io.Float.Input(
                    "max_sharpness_ratio", default=2.0, min=0.0, max=20.0, step=0.05
                ),
                io.Float.Input(
                    "max_face_mean_abs_delta", default=0.06, min=0.0, max=1.0, step=0.005
                ),
                io.Float.Input(
                    "max_residual_temporal_jitter",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                ),
                io.Int.Input("minimum_accept_run", default=3, min=1, max=121),
                io.Int.Input("edge_fade_frames", default=2, min=0, max=60),
            ],
            outputs=[
                io.Image.Output("safe_candidate_frames"),
                io.Mask.Output("accepted_change_mask"),
                io.Mask.Output("rejected_frame_mask"),
                io.Int.Output("accepted_frame_count"),
                io.Int.Output("rejected_frame_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*gate_face_refine_parity_candidate(**kwargs))


class MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced",
            display_name=(
                "MiniMax H3 Face Refine MANUAL512 REL Baseline / "
                "人工验收512相对模式基线 (Advanced)"
            ),
            description=(
                "Fail-closed pass-through for the user-selected MANUAL512 REL author-parity v2. "
                "It requires a 512x512 crop canvas, crop factor 2.5, 21/51 smoothing, "
                "BGR YOLO, crop-derived relative 0.8/0.35 video denoise, locked zero-mask audio "
                "and the author's centred-mask/source-space 24/24 stitch. Reports must agree, "
                "including any explicit one-frame H3 alignment tail. It does not use proxy "
                "scores to replace the candidate and is not a universal quality guarantee."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("candidate_frames"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.String.Input("latent_report_json", force_input=True),
                io.String.Input("denoise_report_json", force_input=True),
                io.String.Input("stitch_report_json", force_input=True),
                io.Combo.Input(
                    "profile",
                    options=[MANUAL512_RELATIVE_PROFILE],
                    default=MANUAL512_RELATIVE_PROFILE,
                ),
                io.Float.Input(
                    "minimum_crop_face_height_px",
                    default=200.0,
                    min=64.0,
                    max=512.0,
                    step=1.0,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("candidate_frames"),
                io.String.Output("baseline_report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*validate_face_refine_manual512_relative_baseline(**kwargs))


FACE_REFINE_PARITY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceRefineParityPlanT8Advanced,
    MiniMaxH3FaceRefineParityLatentT8Advanced,
    MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced,
    MiniMaxH3FaceRefineParityStitchT8Advanced,
    MiniMaxH3FaceRefineQualityGateT8Advanced,
    MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced,
]
