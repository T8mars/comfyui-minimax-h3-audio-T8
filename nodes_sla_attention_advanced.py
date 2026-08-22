from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .sla_attention_advanced import (
    SLA_BASE_POLICIES,
    SLA_LORA_FILENAME,
    SLA_MODES,
    SLA_RUNTIME_TYPE,
    build_sla_model,
    finalize_sla_runtime,
)


CATEGORY = "T8/MiniMax H3/Performance/Experimental"
SLARuntimeIO = io.Custom(SLA_RUNTIME_TYPE)


def _lora_options() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    if SLA_LORA_FILENAME in names:
        names.remove(SLA_LORA_FILENAME)
    return [SLA_LORA_FILENAME, *names]


class MiniMaxH3LightX2VSLAT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lora_options = _lora_options()
        return io.Schema(
            node_id="MiniMaxH3LightX2VSLAT8Advanced",
            display_name="MiniMax H3 LightX2V SLA Loader + Attention (Advanced EXP)",
            description=(
                "Loads the authenticated 4-step FL2VA Turbo-SLA LoRA and owns the "
                "LightX2V 85%-sparse Sage2 attention path for all 50 main H3 blocks. "
                "Use it after the T8 Dual-Clock node at 4 steps, video shift 6, audio "
                "shift 3. External SageAttention/Sol-Attn/LoRA nodes must be bypassed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect the clean native H3 MODEL output from Dual-Clock. Do not "
                        "place KJ Sage, Sol-Attn, FETA, BlockCache, STG or another LoRA before it."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "Connect the same 4-step native_flow SIGMAS from Dual-Clock "
                        "(video shift 6.0, audio shift 3.0)."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=lora_options,
                    default=SLA_LORA_FILENAME,
                    tooltip=(
                        "Pinned LightX2V MiniMax H3 FL2V Turbo-SLA ComfyUI BF16 LoRA. "
                        "The file size, SHA-256, metadata and all 208 patches are verified."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(SLA_MODES),
                    default="apply_lightx2v_sla",
                    tooltip=(
                        "apply_lightx2v_sla runs the learned 85% block-sparse router. "
                        "dense_lora_control keeps the same SLA LoRA but uses dense attention "
                        "for a scientific A/B. disabled_identity changes nothing."
                    ),
                ),
                io.Combo.Input(
                    "base_policy",
                    options=list(SLA_BASE_POLICIES),
                    default="auto_detect_exp",
                    advanced=True,
                    tooltip=(
                        "The upstream LoRA names a BF16 FL2VA base. auto_detect_exp also "
                        "permits the current INT8 base but reports it as experimental; "
                        "official_bf16_only refuses a quantized base."
                    ),
                ),
                io.Int.Input(
                    "max_router_workspace_mib",
                    default=512,
                    min=32,
                    max=2048,
                    step=32,
                    advanced=True,
                    tooltip=(
                        "Fail-closed ceiling for estimated router score/map workspace. "
                        "This is not a whole-workflow VRAM limit."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                SLARuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        sigmas,
        lora_name,
        mode,
        base_policy,
        max_router_workspace_mib,
    ):
        if mode == "disabled_identity":
            lora_path = ""
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(
            *build_sla_model(
                model,
                sigmas,
                lora_path=lora_path,
                mode=mode,
                base_policy=base_policy,
                max_router_workspace_mib=max_router_workspace_mib,
            )
        )


class MiniMaxH3LightX2VSLAAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LightX2VSLAAuditT8Advanced",
            display_name="MiniMax H3 LightX2V SLA Runtime Audit (Advanced EXP)",
            description=(
                "Place after the sampler. It fails unless all four H3 forwards and all "
                "50 main blocks per forward used the selected dense-control or sparse path "
                "without a hidden kernel fallback."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                SLARuntimeIO.Input("runtime"),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, av_latent, runtime):
        latent, report_json = finalize_sla_runtime(av_latent, runtime)
        return io.NodeOutput(latent, report_json, ui={"text": (report_json,)})


class MiniMaxH3LightX2VSLAKJSageComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lora_options = _lora_options()
        return io.Schema(
            node_id="MiniMaxH3LightX2VSLAKJSageComposerT8Advanced",
            display_name="MiniMax H3 LightX2V SLA + KJ Sage Composer (Advanced EXP)",
            description=(
                "Compose an upstream KJNodes MiniMax H3 memory-efficient Sage patch "
                "with LightX2V SLA under one audited attention owner. SLA apply calls "
                "use block-sparse Sage2; dense-control and calls outside the SLA route "
                "retain KJ Sage. No attention call runs both kernels."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect Dual-Clock MODEL to KJNodes MiniMax H3 Mem Eff Sage "
                        "Attention Patch, then connect that MODEL here. Do not insert "
                        "ModelAttentionBackend, Sol-Attn or another attention node."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "Connect the same 4-step native_flow SIGMAS from Dual-Clock "
                        "(video shift 6.0, audio shift 3.0)."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=lora_options,
                    default=SLA_LORA_FILENAME,
                    tooltip=(
                        "Pinned LightX2V MiniMax H3 FL2V Turbo-SLA ComfyUI BF16 LoRA."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(SLA_MODES),
                    default="apply_lightx2v_sla",
                    tooltip=(
                        "apply_lightx2v_sla uses SLA block-sparse Sage2. "
                        "dense_lora_control keeps the same LoRA and routes all 50 main "
                        "blocks through the authenticated upstream KJ Sage forward."
                    ),
                ),
                io.Combo.Input(
                    "base_policy",
                    options=list(SLA_BASE_POLICIES),
                    default="auto_detect_exp",
                    advanced=True,
                    tooltip=(
                        "The official SLA base is BF16 FL2VA; quantized bases remain "
                        "explicit compatibility experiments."
                    ),
                ),
                io.Int.Input(
                    "max_router_workspace_mib",
                    default=512,
                    min=32,
                    max=2048,
                    step=32,
                    advanced=True,
                    tooltip="Fail-closed ceiling for SLA router score/map workspace.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                SLARuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        sigmas,
        lora_name,
        mode,
        base_policy,
        max_router_workspace_mib,
    ):
        if mode == "disabled_identity":
            lora_path = ""
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(
            *build_sla_model(
                model,
                sigmas,
                lora_path=lora_path,
                mode=mode,
                base_policy=base_policy,
                max_router_workspace_mib=max_router_workspace_mib,
                external_attention_policy="compose_kj_sage",
            )
        )


SLA_ATTENTION_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LightX2VSLAT8Advanced,
    MiniMaxH3LightX2VSLAAuditT8Advanced,
    MiniMaxH3LightX2VSLAKJSageComposerT8Advanced,
]
