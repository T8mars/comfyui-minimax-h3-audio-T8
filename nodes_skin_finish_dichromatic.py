from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_dichromatic import attenuate_skin_specular_dichromatic


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishDichromaticT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishDichromaticT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Dichromatic Specular / 二色反射高光抑制 "
                "(Advanced EXP)"
            ),
            description=(
                "Experimental clean-room highlight attenuation for reliable semantic skin "
                "masks. It requires both a neutral-illuminant dichromatic specular estimate "
                "and local chroma dilution, so same-chromaticity bright skin is preserved. "
                "This is bounded SDR finishing, not physical inverse rendering. Source stays "
                "selected until full human review."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Float.Input(
                    "amount", default=0.80, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "specular_strength",
                    default=0.80,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "diffuse_radius_percent",
                    default=2.5,
                    min=0.10,
                    max=8.0,
                    step=0.10,
                ),
                io.Int.Input(
                    "maximum_radius_px", default=48, min=1, max=192, advanced=True
                ),
                io.Float.Input(
                    "specular_threshold_linear",
                    default=0.004,
                    min=0.0,
                    max=0.25,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "specular_softness_linear",
                    default=0.030,
                    min=0.001,
                    max=0.50,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "chroma_dilution_threshold",
                    default=0.0015,
                    min=0.0,
                    max=0.25,
                    step=0.0005,
                    advanced=True,
                ),
                io.Float.Input(
                    "chroma_dilution_softness",
                    default=0.020,
                    min=0.001,
                    max=0.50,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_diffuse_chroma",
                    default=0.008,
                    min=0.0,
                    max=0.25,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "diffuse_chroma_softness",
                    default=0.050,
                    min=0.001,
                    max=0.50,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_direction_cosine",
                    default=0.75,
                    min=-1.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_surface_delta",
                    default=0.10,
                    min=0.0,
                    max=0.25,
                    step=0.005,
                ),
                io.Float.Input(
                    "minimum_texture_ratio",
                    default=0.86,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "maximum_texture_ratio",
                    default=1.10,
                    min=1.0,
                    max=2.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "minimum_reference_texture",
                    default=0.003,
                    min=0.0,
                    max=0.10,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_mean_abs_change",
                    default=0.035,
                    min=0.0,
                    max=0.25,
                    step=0.005,
                ),
                io.Float.Input(
                    "maximum_peak_abs_change",
                    default=0.18,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "minimum_mask_area",
                    default=0.0001,
                    min=0.0,
                    max=0.25,
                    step=0.0001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_mask_area",
                    default=0.50,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_new_clipped_fraction",
                    default=0.0005,
                    min=0.0,
                    max=0.25,
                    step=0.0005,
                ),
                io.Float.Input(
                    "clipping_epsilon",
                    default=1.0 / 255.0,
                    min=0.0001,
                    max=0.05,
                    step=0.0001,
                    advanced=True,
                ),
                io.Int.Input(
                    "chunk_frames", default=2, min=1, max=16, advanced=True
                ),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "False keeps the exact source selected. Enable only after labelled "
                        "full-video review and downstream Texture/Safety gates."
                    ),
                ),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("specular_candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("effective_mask"),
                io.Mask.Output("rejected_mask"),
                io.Image.Output("difference"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*attenuate_skin_specular_dichromatic(**kwargs))


SKIN_FINISH_DICHROMATIC_NODE_CLASSES = [
    MiniMaxH3SkinFinishDichromaticT8Advanced
]
