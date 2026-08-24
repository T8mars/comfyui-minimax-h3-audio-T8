from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_parser import PARSENET_MODEL_NAME, run_semantic_skin_mask


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"
FaceRefinePlanIO = io.Custom("H3_T8_FACE_REFINE_PLAN")


class MiniMaxH3SkinFinishSemanticMaskT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishSemanticMaskT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Semantic Mask / 语义皮肤遮罩 (Advanced EXP)"
            ),
            description=(
                "Builds a source-bound semantic skin mask with the pinned local FaceXLib "
                "ParseNet checkpoint. Eyes, brows, nose, lips, mouth, hair and accessories are "
                "excluded by class. Missing, altered or unsafe weights fail closed to an empty "
                "mask; no network download or persistent model cache is used."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                FaceRefinePlanIO.Input("face_plan"),
                io.Combo.Input(
                    "parser_model",
                    options=[PARSENET_MODEL_NAME],
                    default=PARSENET_MODEL_NAME,
                ),
                io.Boolean.Input(
                    "include_neck",
                    default=False,
                    tooltip="Off by default so face finishing cannot spill onto neck or clothing.",
                ),
                io.Float.Input(
                    "crop_expansion",
                    default=1.45,
                    min=1.0,
                    max=3.0,
                    step=0.05,
                    tooltip=(
                        "Expanded upright square around each source-bound face box. Existing "
                        "face plans do not retain five-point landmarks, so this is not affine alignment."
                    ),
                ),
                io.Float.Input(
                    "minimum_face_weight", default=0.35, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "minimum_class_probability",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "feature_protection_px",
                    default=3,
                    min=0,
                    max=32,
                    tooltip="Protection dilation in the fixed 512x512 parser canvas.",
                ),
                io.Float.Input(
                    "minimum_skin_area", default=0.0005, min=0.0, max=0.25, step=0.0005
                ),
                io.Float.Input(
                    "maximum_skin_area", default=0.25, min=0.01, max=1.0, step=0.01
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
        return io.NodeOutput(*run_semantic_skin_mask(**kwargs))


SKIN_FINISH_PARSER_NODE_CLASSES = [MiniMaxH3SkinFinishSemanticMaskT8Advanced]
