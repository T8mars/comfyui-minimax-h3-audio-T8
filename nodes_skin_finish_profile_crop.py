from __future__ import annotations

from comfy_api.latest import io

from .nodes_skin_finish_multiface_parser import (
    CATEGORY,
    IdentityAssignmentIO,
    TrackPlanIO,
)
from .skin_finish_multiface_parser import run_multiface_semantic_skin_mask
from .skin_finish_parser import PARSENET_MODEL_NAME


class MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=(
                "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
            ),
            display_name=(
                "MiniMax H3 Skin Finish Multi-Person Profile Semantic Mask / "
                "多人侧脸语义皮肤遮罩 (Advanced EXP)"
            ),
            description=(
                "Keeps the existing strict YuNet five-point FFHQ alignment first. Only when "
                "that alignment is rejected, it parses an expanded square crop in the "
                "original profile pose, projects the mask back, and intersects it with the "
                "exact SAM3.1 person track. It does not frontalize a face or loosen the "
                "five-point residual gate."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                TrackPlanIO.Input("track_plan"),
                IdentityAssignmentIO.Input("identity_assignment", optional=True),
                io.Combo.Input(
                    "parser_model",
                    options=[PARSENET_MODEL_NAME],
                    default=PARSENET_MODEL_NAME,
                ),
                io.Float.Input(
                    "detection_threshold", default=0.45, min=0.05, max=0.99, step=0.01
                ),
                io.Float.Input(
                    "minimum_face_height_px",
                    default=32.0,
                    min=8.0,
                    max=1024.0,
                    step=1.0,
                ),
                io.Float.Input(
                    "minimum_detail", default=0.010, min=0.0, max=0.20, step=0.001
                ),
                io.Float.Input(
                    "minimum_person_overlap", default=0.20, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "minimum_track_quality", default=0.10, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "minimum_class_probability",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input("feature_protection_px", default=3, min=0, max=32),
                io.Boolean.Input(
                    "include_neck",
                    default=False,
                    tooltip="Neck remains excluded by default; clothing is always protected.",
                ),
                io.Float.Input(
                    "minimum_skin_area_per_face",
                    default=0.00005,
                    min=0.0,
                    max=0.10,
                    step=0.00005,
                ),
                io.Float.Input(
                    "maximum_skin_area_per_frame",
                    default=0.35,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "maximum_alignment_rms",
                    default=0.08,
                    min=0.005,
                    max=0.25,
                    step=0.005,
                    tooltip=(
                        "Strict five-landmark residual. This node does not raise it; a "
                        "rejected pose may use the profile-crop fallback instead."
                    ),
                ),
                io.Float.Input(
                    "profile_crop_expansion",
                    default=1.45,
                    min=1.0,
                    max=3.0,
                    step=0.05,
                    tooltip=(
                        "Square face crop used only after strict five-point alignment rejects "
                        "the pose. 1.45 passed the bounded six-frame profile probe."
                    ),
                ),
                io.Float.Input(
                    "minimum_ready_frame_fraction",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "If fewer frames contain at least one reliable tracked face, the whole "
                        "batch returns an empty mask instead of intermittent processing."
                    ),
                ),
                io.Int.Input("preview_count", default=6, min=1, max=8),
            ],
            outputs=[
                io.Mask.Output("semantic_skin_mask"),
                io.Image.Output("mask_preview"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *run_multiface_semantic_skin_mask(
                **kwargs,
                alignment_policy="five_point_then_profile_crop",
            )
        )


SKIN_FINISH_PROFILE_CROP_NODE_CLASSES = [
    MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced
]
