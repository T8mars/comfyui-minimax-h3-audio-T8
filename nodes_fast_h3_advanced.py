from __future__ import annotations

from comfy_api.latest import io

from .fast_h3_advanced import build_fast_h3_4step_setup


CATEGORY = "T8/MiniMax H3/Performance/Experimental"


class MiniMaxH3FastH34StepSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FastH34StepSetupT8Advanced",
            display_name="MiniMax H3 FastH3 4-Step Setup (Advanced EXP/T8)",
            description=(
                "Configures the published T2VA-only FastH3 4-step DMD2 joint-AV sampling "
                "contract. Apply the matching FastH3 LoRA first. Dense ComfyUI attention "
                "remains the default; the opt-in VSA profile applies the learned 90%-sparse "
                "tile-64 route only when the LoRA gate payload and current Comfy Kitchen "
                "VSA kernel contract are both available, otherwise it falls back to dense."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Combo.Input(
                    "task_family",
                    options=[
                        "t2va_only",
                        "t2va_fl2va",
                        "t2va_fl2va_legacy_untrained_exp",
                        "ref2va_untrained_exp",
                    ],
                    default="t2va_only",
                ),
                io.Combo.Input(
                    "attention_profile",
                    options=["dense_comfyui", "external_vsa_if_available"],
                    default="dense_comfyui",
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
        task_family="t2va_only",
        attention_profile="dense_comfyui",
    ):
        return io.NodeOutput(
            *build_fast_h3_4step_setup(
                model,
                av_latent,
                task_family=task_family,
                attention_profile=attention_profile,
            )
        )


FAST_H3_ADVANCED_NODE_CLASSES = [MiniMaxH3FastH34StepSetupT8Advanced]
