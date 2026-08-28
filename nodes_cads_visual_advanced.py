from __future__ import annotations

from comfy_api.latest import io

from .cads_visual_advanced import NOISE_MODES, build_cads_visual_reference_model


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"


class MiniMaxH3CADSVisualReferenceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CADSVisualReferenceT8Advanced",
            display_name="MiniMax H3 CADS Visual Reference Annealing (Advanced EXP)",
            description=(
                "Applies the CADS paper formula to MiniMax H3 visual reference/keyframe "
                "latents at each denoising step. Audio conditioning and target audio are untouched."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "noise_scale", default=0.10, min=0.0, max=2.0, step=0.01
                ),
                io.Float.Input("tau1", default=0.60, min=0.0, max=1.0, step=0.01),
                io.Float.Input("tau2", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "rescale_mix", default=1.0, min=0.0, max=1.0, step=0.05
                ),
                io.Combo.Input(
                    "noise_mode",
                    options=list(NOISE_MODES),
                    default="paper_independent",
                ),
                io.Int.Input("seed", default=0, min=0, max=0x7FFF_FFFF_FFFF_FFFF),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        noise_scale,
        tau1,
        tau2,
        rescale_mix,
        noise_mode,
        seed,
    ):
        return io.NodeOutput(
            *build_cads_visual_reference_model(
                model,
                noise_scale,
                tau1,
                tau2,
                rescale_mix,
                noise_mode,
                seed,
            )
        )


CADS_VISUAL_ADVANCED_NODE_CLASSES = [MiniMaxH3CADSVisualReferenceT8Advanced]

