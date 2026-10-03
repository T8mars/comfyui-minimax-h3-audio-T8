"""Two append-only explicit nodes; existing Fun sockets and defaults stay intact."""

from comfy_api.latest import io
from .h3_fun_control_advanced import _available_fun_control_names
from .h3_fun_union2 import CONTROL_TYPE, load_union2, apply_union2

UnionIO = io.Custom(CONTROL_TYPE)
CATEGORY = "T8/MiniMax H3/Control/Union 2 EXP"


class MiniMaxH3FunUnion2LoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FunUnion2LoaderEXPT8",
            display_name="H3 Fun Union 2.0 · 显式10×5 / post_norm",
            category=CATEGORY,
            is_experimental=True,
            description="Explicit 10-block 49-channel Union2. Converts raw VideoX-Fun names in memory; delegates Core. Not the legacy 5-block loader.",
            inputs=[
                io.Combo.Input(
                    "control_net_name", options=_available_fun_control_names()
                )
            ],
            outputs=[UnionIO.Output("union_control"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, control_net_name):
        return io.NodeOutput(*load_union2(control_net_name))


class MiniMaxH3FunUnion2ApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FunUnion2ApplyEXPT8",
            display_name="H3 Fun Union 2.0 · 源视频＋外置重绘MASK",
            category=CATEGORY,
            is_experimental=True,
            description="White MASK regenerates; native keep/visibility and post_norm. Align source/control/mask explicitly. Apply to each MODEL stage separately; EAV/Relay external. Joint audio not guaranteed unchanged.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                UnionIO.Input("union_control"),
                io.Vae.Input("vae"),
                io.Image.Input("source_video"),
                io.Mask.Input("regen_mask"),
                io.Int.Input("width", default=512, min=32, max=16384, step=32),
                io.Int.Input("height", default=288, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=None, step=17),
                io.Float.Input("strength", default=0.7, min=0, max=2, step=0.05),
                io.Float.Input("start_percent", default=0, min=0, max=1, step=0.01),
                io.Float.Input("end_percent", default=0.75, min=0, max=1, step=0.01),
                io.Boolean.Input("broadcast_single_mask", default=False),
                io.Image.Input("control_video", optional=True),
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
        union_control,
        vae,
        source_video,
        regen_mask,
        width,
        height,
        length,
        strength,
        start_percent,
        end_percent,
        broadcast_single_mask,
        control_video=None,
    ):
        return io.NodeOutput(
            *apply_union2(
                model,
                positive,
                union_control,
                vae,
                source_video,
                regen_mask,
                width,
                height,
                length,
                strength,
                start_percent,
                end_percent,
                control_video,
                broadcast_single_mask,
            )
        )


NODES = [MiniMaxH3FunUnion2LoaderEXPT8, MiniMaxH3FunUnion2ApplyEXPT8]
