from __future__ import annotations

from comfy_api.latest import io
import folder_paths

from .h3_fun_control_advanced import (
    CONTROL_TYPE,
    apply_h3_fun_control,
    load_h3_fun_control,
)


CATEGORY = "T8/MiniMax H3/Control/Advanced"
H3FunControlIO = io.Custom(CONTROL_TYPE)


class MiniMaxH3FunControlLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FunControlLoaderT8Advanced",
            display_name="MiniMax H3 Fun ControlNet Loader (T8 Advanced)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Load an H3 Fun ControlNet from models/controlnet. Uses the official ComfyUI "
                "implementation when available and a clone-scoped fallback on older cores."
            ),
            inputs=[
                io.Combo.Input(
                    "control_net_name",
                    options=folder_paths.get_filename_list("controlnet"),
                )
            ],
            outputs=[
                H3FunControlIO.Output("control_net"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, control_net_name):
        return io.NodeOutput(*load_h3_fun_control(control_net_name))


class MiniMaxH3FunControlApplyT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FunControlApplyT8Advanced",
            display_name="MiniMax H3 Fun ControlNet Apply (T8 Advanced)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Drive H3 target video rows with a preprocessed depth, pose or edge video. "
                "Returns both MODEL and CONDITIONING so it works on current and future native cores."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                H3FunControlIO.Input("control_net"),
                io.Vae.Input("vae"),
                io.Image.Input("control_video"),
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "control_kind",
                    options=["depth", "pose", "canny", "HED", "MLSD", "custom"],
                    default="depth",
                ),
                io.Combo.Input(
                    "fit_mode",
                    options=["exact", "center_crop", "stretch"],
                    default="exact",
                ),
                io.Float.Input("strength", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input(
                    "start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "end_percent", default=0.75, min=0.0, max=1.0, step=0.01, advanced=True
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        control_net,
        vae,
        control_video,
        width,
        height,
        length,
        control_kind,
        fit_mode,
        strength,
        start_percent,
        end_percent,
    ):
        return io.NodeOutput(
            *apply_h3_fun_control(
                model,
                positive,
                control_net,
                vae,
                control_video,
                width,
                height,
                length,
                control_kind,
                fit_mode,
                strength,
                start_percent,
                end_percent,
            )
        )


H3_FUN_CONTROL_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FunControlLoaderT8Advanced,
    MiniMaxH3FunControlApplyT8Advanced,
]
