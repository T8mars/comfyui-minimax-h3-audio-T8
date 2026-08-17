from __future__ import annotations

from comfy_api.latest import io

from .dynamic_guidance_advanced import (
    build_dynamic_guidance_guider,
    finalize_dynamic_guidance_report,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental"
DynamicGuidanceRuntimeIO = io.Custom("H3_T8_DYNAMIC_GUIDANCE_RUNTIME")


class MiniMaxH3DynamicCFGGuiderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DynamicCFGGuiderT8Advanced",
            display_name="MiniMax H3 Dynamic Guidance / 动态引导 (Advanced)",
            description=(
                "Default-pass-through H3 guider. single_condition_gain_exp applies one "
                "device-side sigma-dependent gain to the BasicGuider route and is not true "
                "CFG. true_cfg_exp requires a layout-matched negative and explicit cost consent."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Sigmas.Input("sigmas"),
                io.Combo.Input(
                    "mode",
                    options=[
                        "passthrough_basic",
                        "single_condition_gain_exp",
                        "true_cfg_exp",
                    ],
                    default="passthrough_basic",
                ),
                io.Float.Input("early_scale", default=1.0, min=0.8, max=1.2, step=0.01),
                io.Float.Input("late_scale", default=1.0, min=0.8, max=1.2, step=0.01),
                io.Float.Input(
                    "start_progress", default=0.0, min=0.0, max=0.99, step=0.01
                ),
                io.Float.Input(
                    "end_progress", default=1.0, min=0.01, max=1.0, step=0.01
                ),
                io.Combo.Input("curve", options=["linear", "cosine"], default="linear"),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "profile",
                    options=[
                        "turbo_standard8",
                        "turbo_ema8",
                        "turbo_fl2v8",
                        "stock20",
                        "custom_strict",
                    ],
                    default="turbo_standard8",
                ),
                io.Boolean.Input(
                    "accept_true_cfg_cost",
                    default=False,
                    tooltip="Required for the two-branch true-CFG experiment.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "accept_turbo_guidance_ood",
                    default=False,
                    tooltip=(
                        "Required before applying a non-identity guidance curve to an 8-step "
                        "Turbo profile."
                    ),
                    advanced=True,
                ),
                io.Conditioning.Input("negative", optional=True),
            ],
            outputs=[
                io.Guider.Output("guider"),
                DynamicGuidanceRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_dynamic_guidance_guider(**kwargs))


class MiniMaxH3DynamicGuidanceAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DynamicGuidanceAuditT8Advanced",
            display_name="MiniMax H3 Dynamic Guidance Audit / 动态引导审计 (Advanced)",
            description=(
                "Place after sampling. It reports observed guider calls, physical model "
                "forwards and cond/uncond branch batching without modifying the AV latent."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                DynamicGuidanceRuntimeIO.Input("runtime"),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, av_latent, runtime):
        latent, report_json = finalize_dynamic_guidance_report(av_latent, runtime)
        return io.NodeOutput(
            latent,
            report_json,
            ui={"text": (report_json,)},
        )


DYNAMIC_GUIDANCE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3DynamicCFGGuiderT8Advanced,
    MiniMaxH3DynamicGuidanceAuditT8Advanced,
]
