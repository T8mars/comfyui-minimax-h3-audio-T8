from __future__ import annotations

from comfy_api.latest import io

from .latent_upscale import (
    ALIGNMENT_POLICIES,
    UPSCALE_METHODS,
    upscale_latent_by_32,
)


LATENT_PIXEL_SCALE_OPTIONS = [
    "16 - MiniMax H3",
    "8 - SD/SDXL/other",
]


class MiniMaxH3LatentUpscaleBy32T8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentUpscaleBy32T8",
            display_name="Latent Upscale by 32 / 32整除潜空间放大 (T8)",
            description=(
                "Upscales a plain latent or the video part of a MiniMax H3 joint AV latent. "
                "The returned pixel width and height are always divisible by 32; best_aspect "
                "minimizes the unavoidable ratio error caused by the 32-pixel grid."
            ),
            category="T8/MiniMax H3/Latent",
            inputs=[
                io.Latent.Input("latent"),
                io.Combo.Input(
                    "upscale_method",
                    options=list(UPSCALE_METHODS),
                    default="bicubic",
                ),
                io.Float.Input("scale_by", default=1.5, min=1.0, max=8.0, step=0.01),
                io.Combo.Input(
                    "pixels_per_latent",
                    options=LATENT_PIXEL_SCALE_OPTIONS,
                    default="16 - MiniMax H3",
                    tooltip=(
                        "MiniMax H3 video latent uses 16 pixels per latent cell. "
                        "Choose 8 only for a plain SD/SDXL-style latent."
                    ),
                ),
                io.Combo.Input(
                    "alignment_policy",
                    options=list(ALIGNMENT_POLICIES),
                    default="best_aspect",
                    tooltip=(
                        "best_aspect checks the four closest legal size pairs and minimizes "
                        "aspect-ratio error. nearest/floor/ceil prioritize the requested size."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        upscale_method,
        scale_by,
        pixels_per_latent,
        alignment_policy,
    ):
        pixel_scale = int(str(pixels_per_latent).split(" ", 1)[0])
        return io.NodeOutput(
            *upscale_latent_by_32(
                latent,
                upscale_method,
                scale_by,
                pixel_scale,
                alignment_policy,
            )
        )


LATENT_UPSCALE_NODE_CLASSES = [MiniMaxH3LatentUpscaleBy32T8]
