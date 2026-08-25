from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_surface import finish_skin_surface


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishSurfaceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishSurfaceT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Guided Surface / 引导滤波肤质收尾 "
                "(Advanced EXP)"
            ),
            description=(
                "Clean-room guided-filter surface candidate for reliable semantic skin masks. "
                "It balances compact and broad highlights plus blemishes while explicitly "
                "retaining source texture. Broad highlights are measured against masked local "
                "skin illumination and treatment fades inside hard mask edges; this remains a "
                "display-referred finish, not physical specular separation. Source stays selected "
                "until full human review."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Float.Input("amount", default=0.65, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "surface_smoothing",
                    default=0.70,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "texture_keep", default=0.85, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "highlight_compression",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "broad_highlight_compression",
                    default=0.45,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "broad_highlight_start",
                    default=0.68,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "broad_highlight_end",
                    default=0.94,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "blemish_balance",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "surface_radius_percent",
                    default=2.0,
                    min=0.10,
                    max=5.0,
                    step=0.10,
                ),
                io.Int.Input(
                    "maximum_radius_px", default=32, min=1, max=128, advanced=True
                ),
                io.Float.Input(
                    "edge_epsilon",
                    default=0.0025,
                    min=0.000001,
                    max=0.10,
                    step=0.0001,
                    advanced=True,
                ),
                io.Float.Input(
                    "edge_protection_scale",
                    default=0.055,
                    min=0.001,
                    max=0.50,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "highlight_threshold",
                    default=0.006,
                    min=0.0,
                    max=0.10,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "blemish_threshold",
                    default=0.008,
                    min=0.0,
                    max=0.10,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_surface_delta",
                    default=0.08,
                    min=0.0,
                    max=0.25,
                    step=0.005,
                ),
                io.Float.Input(
                    "minimum_texture_ratio",
                    default=0.82,
                    min=0.0,
                    max=1.0,
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
                io.Image.Output("surface_candidate"),
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
        return io.NodeOutput(*finish_skin_surface(**kwargs))


SKIN_FINISH_SURFACE_NODE_CLASSES = [MiniMaxH3SkinFinishSurfaceT8Advanced]
