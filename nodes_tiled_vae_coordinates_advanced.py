from __future__ import annotations

from comfy_api.latest import io

from .tiled_vae_coordinates_advanced import (
    build_tiled_vae_coordinate_compatibility,
)


CATEGORY = "T8/MiniMax H3/VAE/Advanced"


class MiniMaxH3GlobalCoordinateTiledVAET8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3GlobalCoordinateTiledVAET8Advanced",
            display_name="MiniMax H3 Tiled VAE Coordinate Audit (T8 Advanced EXP)",
            category=CATEGORY,
            description=(
                "Audit the upstream global-coordinate tiled-VAE proposal. A real "
                "fp16 H3 VAE check regressed grid artifacts, so the safe default "
                "reports only and returns the source VAE unchanged."
            ),
            is_experimental=True,
            inputs=[
                io.Vae.Input("video_vae"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "apply_global_coordinates_exp"],
                    default="report_only",
                ),
            ],
            outputs=[io.Vae.Output("video_vae"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, video_vae, mode) -> io.NodeOutput:
        return io.NodeOutput(
            *build_tiled_vae_coordinate_compatibility(video_vae, mode)
        )


TILED_VAE_COORDINATES_ADVANCED_NODE_CLASSES = [
    MiniMaxH3GlobalCoordinateTiledVAET8Advanced,
]
