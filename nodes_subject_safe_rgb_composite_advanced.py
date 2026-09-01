from __future__ import annotations

from comfy_api.latest import io

from .subject_safe_rgb_composite_advanced import compose_subject_safe_rgb


CATEGORY = "T8/MiniMax H3/Latent Upscale/Experimental"


class MiniMaxH3SubjectSafeRGBCompositeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectSafeRGBCompositeT8Advanced",
            display_name=(
                "MiniMax H3 Subject-Safe RGB Composite / 人物安全RGB合成 "
                "(v8 Advanced EXP)"
            ),
            description=(
                "Keeps the D0/base frame everywhere outside a reviewed per-frame subject "
                "alpha and blends the refined T2 candidate only inside that alpha. No "
                "automatic person, face, text, camera or quality decision is performed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("base_frames", tooltip="D0/base frames that own the background."),
                io.Image.Input("refined_frames", tooltip="T2/refined frames for the subject only."),
                io.Mask.Input(
                    "subject_alpha",
                    tooltip="Reviewed per-frame final alpha. Zero means exact D0 ownership.",
                ),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip="False keeps selected output on D0. Enable only for a reviewed candidate.",
                ),
                io.Combo.Input(
                    "mask_mode",
                    options=["input_alpha_exact", "threshold_binary"],
                    default="input_alpha_exact",
                ),
                io.Combo.Input(
                    "mask_frame_policy",
                    options=["strict_exact", "allow_single_broadcast_exp"],
                    default="strict_exact",
                    tooltip="Strict requires one alpha frame for every video frame.",
                ),
                io.Float.Input(
                    "minimum_subject_area", default=0.002, min=0.0, max=1.0, step=0.001
                ),
                io.Float.Input(
                    "maximum_subject_area", default=0.45, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "maximum_centroid_jump", default=0.08, min=0.0, max=1.0, step=0.005
                ),
                io.Combo.Input(
                    "strictness",
                    options=["fallback_on_contract_failure", "audit_only"],
                    default="fallback_on_contract_failure",
                ),
                io.Int.Input("chunk_frames", default=4, min=1, max=32),
                io.Mask.Input(
                    "protect_mask",
                    optional=True,
                    tooltip="Optional face/text/identity protection; one removes T2 ownership.",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="D0/source audio is returned as the exact same object.",
                ),
            ],
            outputs=[
                io.Image.Output("selected"),
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.Mask.Output("used_alpha"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*compose_subject_safe_rgb(**kwargs))


SUBJECT_SAFE_RGB_COMPOSITE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3SubjectSafeRGBCompositeT8Advanced,
]
