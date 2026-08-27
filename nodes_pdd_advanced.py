from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .pdd_advanced import PDD_VARIANTS, build_pdd_8step_setup


CATEGORY = "T8/MiniMax H3/Performance/Experimental"
DEFAULT_FL2VA_PDD = "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors"
DEFAULT_REF2VA_PDD = "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors"


def _pdd_lora_options() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    defaults = [DEFAULT_FL2VA_PDD, DEFAULT_REF2VA_PDD]
    # Do not expose the user's entire LoRA library here.  A normal LoRA cannot
    # provide the four dynamic PDD head banks, and listing it would make the
    # dedicated loader look interchangeable with ComfyUI's generic loader.
    candidates = [
        name
        for name in names
        if "pdd" in name.lower()
        and name.lower().endswith((".safetensors", ".sft"))
    ]
    return [*defaults, *(name for name in candidates if name not in defaults)]


class MiniMaxH3PDD8StepSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PDD8StepSetupT8Advanced",
            display_name="MiniMax H3 PDD 8-Step Setup (T8 Advanced EXP)",
            description=(
                "Loads a converted Alibaba PAI MiniMax-H3 PDD adapter with its "
                "dynamic video/audio output heads. It applies the 258 backbone "
                "adapters as model-only bypass residuals and returns the required "
                "Euler/simple 8-step sampler and sigmas at shifts 12/3. Use only the "
                "matching full non-pruned FL2VA or Ref2VA base."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Matching native ComfyUI full/non-pruned H3 MODEL. Do not "
                        "connect an existing LoRA, pruned model or another PDD node."
                    ),
                ),
                io.Latent.Input(
                    "av_latent",
                    tooltip=(
                        "Connect av_latent from the existing T8 Conditioning node. "
                        "It is used to configure ComfyUI's native joint AV sampler."
                    ),
                ),
                io.Combo.Input(
                    "pdd_lora_name",
                    options=_pdd_lora_options(),
                    default=DEFAULT_FL2VA_PDD,
                    tooltip=(
                        "Converted *_comfyui_pdd.safetensors file. Ordinary Load "
                        "LoRA nodes drop the required dynamic PDD heads."
                    ),
                ),
                io.Combo.Input(
                    "base_variant",
                    options=list(PDD_VARIANTS),
                    default="FL2VA",
                    display_name="base variant / 基模类型",
                    tooltip=(
                        "Select how the PDD runtime should interpret the adapter. File metadata "
                        "and variant mismatches are reported but do not block loading."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Official released setting is 1.0. Lower values interpolate "
                        "both the dynamic LoRA residual and PDD heads and are unvalidated."
                    ),
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
    def execute(cls, model, av_latent, pdd_lora_name, base_variant, strength):
        path = folder_paths.get_full_path_or_raise("loras", pdd_lora_name)
        return io.NodeOutput(
            *build_pdd_8step_setup(
                model,
                av_latent,
                path,
                base_variant=base_variant,
                strength=strength,
            )
        )


PDD_ADVANCED_NODE_CLASSES = [MiniMaxH3PDD8StepSetupT8Advanced]
