from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .h3_lora_compat_advanced import load_minimax_h3_lora_model


CATEGORY = "T8/MiniMax H3/Model/Advanced"


def _lora_options() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    return names or ["place_h3_lora_in_models_loras.safetensors"]


class MiniMaxH3LoRACompatibilityLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoRACompatibilityLoaderT8Advanced",
            display_name="MiniMax H3 LoRA Compatibility Loader (Advanced/T8)",
            description=(
                "MODEL-only H3 LoRA loader supporting native ComfyUI plus direct "
                "DiffSynth-Studio/ModelScope module names. It uses structural key "
                "mapping only; filenames and sizes are display-only, with no hash "
                "scan or model-file execution gate."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=_lora_options()),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model=1.0):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(
            *load_minimax_h3_lora_model(model, path, strength_model)
        )


H3_LORA_COMPAT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LoRACompatibilityLoaderT8Advanced,
]
