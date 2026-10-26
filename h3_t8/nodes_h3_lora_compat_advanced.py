from __future__ import annotations

import json

import folder_paths
from comfy_api.latest import io

from .h3_lora_compat_advanced import FILE_ALPHA_POLICIES, PREFIX_POLICIES, load_minimax_h3_lora_model


CATEGORY = "T8/MiniMax H3/Model/Advanced"
DISABLED_LORA = "disabled"


def _lora_options() -> list[str]:
    names = [name for name in folder_paths.get_filename_list("loras") if name != DISABLED_LORA]
    return [DISABLED_LORA, *names]


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
                "scan or model-file execution gate. Select disabled to pass the "
                "MODEL through, which makes an optional per-stage external LoRA safe."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=_lora_options()),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    step=0.01,
                ),
                io.Combo.Input(
                    "file_alpha_policy", options=list(FILE_ALPHA_POLICIES),
                    default="legacy_tensor_only", optional=True, advanced=True,
                    tooltip=("Default keeps old tensor-only scaling. Explicit metadata_alpha_per_rank "
                             "uses literal ss_network_alpha / actual rank only for mapped linear pairs "
                             "without per-key alpha. Metadata is not proof of the training scale; "
                             "zero means zero here. User strength and order remain unchanged."),
                ),
                io.Combo.Input(
                    "prefix_policy", options=list(PREFIX_POLICIES), default="legacy", optional=True, advanced=True,
                    tooltip=("Default keeps existing Core mappings. Explicit unique_double_prefix adds an alias "
                             "only for a complete matching linear pair with exactly one known target. "
                             "No key rewriting, cross-namespace pair joining or collision overwriting."),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model=1.0, file_alpha_policy="legacy_tensor_only", prefix_policy="legacy"):
        if lora_name == DISABLED_LORA:
            report = {
                "schema": "t8.minimax_h3.lora_compat.v1",
                "status": "disabled",
                "selected_strength": float(strength_model),
                "model_class": model.model.__class__.__name__,
                "contract": "No LoRA file was opened and the input MODEL was returned unchanged.",
            }
            text = json.dumps(report, ensure_ascii=False, indent=2)
            return io.NodeOutput(model, text, ui={"t8_lora_loader": [
                {"selected_lora_name": lora_name, "report_json": text}]})
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        patched, text = load_minimax_h3_lora_model(model, path, strength_model,
            file_alpha_policy=file_alpha_policy, prefix_policy=prefix_policy)
        return io.NodeOutput(patched, text, ui={"t8_lora_loader": [
            {"selected_lora_name": lora_name, "report_json": text}]})


H3_LORA_COMPAT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LoRACompatibilityLoaderT8Advanced,
]
