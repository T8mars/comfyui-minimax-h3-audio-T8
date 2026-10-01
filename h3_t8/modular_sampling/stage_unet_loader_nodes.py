"""Append-only stage-ordered Core UNET loader for isolated HIGH branches."""

import folder_paths
from comfy_api.latest import io

from .stage_unet_loader import load_unet_after_stage


class MiniMaxH3StageUNETLoaderAfterEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StageUNETLoaderAfterEXPT8",
            display_name="H3 Stage · Independent UNET Load After Result (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental",
            is_experimental=True,
            description="Wait for a verified completed stage, then independently load this stage's "
                        "diffusion model through Core UNETLoader. Identical bare loader nodes may be "
                        "deduplicated by Core; a MODEL clone still shares the live network. This node "
                        "keeps HIGH MODEL/LoRA editable without hidden sampling, but may consume extra "
                        "CPU/RAM/VRAM until the earlier model is released. Connect a frozen StageLoad "
                        "result for HIGH-only resume.",
            inputs=[
                io.Custom("T8_STAGE_RESULT").Input("completed_stage"),
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                               default="default"),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, completed_stage, unet_name, weight_dtype="default"):
        return io.NodeOutput(*load_unet_after_stage(completed_stage, unet_name, weight_dtype))


NODES = (MiniMaxH3StageUNETLoaderAfterEXPT8,)
