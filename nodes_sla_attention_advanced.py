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
                "Loads a structurally valid H3 FL2VA Turbo-SLA LoRA and owns the "
                "LightX2V Sage2 attention path. The saved apply_lightx2v_sla mode remains "
                "loadable as a diagnostic policy: short sequences stay dense; eligible "
                "sequences use dense boundary forwards and prefix-protected sparse middle "
                "forwards. The exact all-sparse upstream route remains an explicit EXP mode. "
                "The upstream checkpoint defaults to 4 steps, video shift 6 and audio "
                "shift 3; other native_flow NFE values are accepted as experimental. "
                "External SageAttention/Sol-Attn/LoRA nodes must be bypassed."
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
                        "Connect the same native_flow SIGMAS from Dual-Clock. The included "
                        "upstream SLA checkpoint is officially 4-step/6V/3A; 8 steps and "
                        "other NFE values run under an explicit experimental schedule report."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=lora_options,
                    default=SLA_LORA_FILENAME,
                    tooltip=(
                        "Default LightX2V MiniMax H3 FL2V Turbo-SLA ComfyUI BF16 LoRA. "
                        "File name, SHA, size, metadata and tensor layout are diagnostic only. "
                        "The selected file is passed to the real ComfyUI loader."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(SLA_MODES),
                    default="apply_lightx2v_sla",
                    tooltip=(
                        "apply_lightx2v_sla is a diagnostic auto mode, not quality-safe: "
                        "under 50K packed "
                        "tokens it uses dense attention; longer sequences keep dense first/"
                        "last forwards and protect text/keyframe/audio keys during sparse "
                        "middle forwards. Both the short all-dense result and the earlier "
                        "fixed-sparse result failed full-duration human review. Use the new "
                        "Turbo / SLA Profile Router for the recommended Turbo8 fallback. "
                        "apply_lightx2v_sla_upstream_exact_exp reproduces "
                        "85% sparse attention on every forward and may collapse outside the "
                        "published long 768p profile. "
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
                        "Controls report labels only. Base dtype and quantization are recorded "
                        "but never used to reject a user-selected model."
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
                "Place after the sampler. It fails unless every scheduled H3 forward and all "
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
                "with LightX2V SLA under one audited attention owner. Diagnostic sparse "
                "forwards use block-sparse Sage2; planned dense forwards, dense-control "
                "and calls outside the SLA route retain KJ Sage. No attention call runs "
                "both kernels."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect Dual-Clock MODEL to KJNodes MiniMax H3 Mem Eff Sage "
                        "Attention Patch, then connect that MODEL here. A recognized built-in "
                        "ModelAttentionBackend is redundant and will be replaced by the SLA "
                        "owner; Sol-Attn and foreign attention overrides are still refused."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "Connect the same native_flow SIGMAS from Dual-Clock. The official "
                        "checkpoint defaults to 4-step/6V/3A; 8-step is allowed and audited "
                        "as an experimental user-selected NFE schedule."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=lora_options,
                    default=SLA_LORA_FILENAME,
                    tooltip=(
                        "Defaults to the LightX2V H3 FL2V Turbo-SLA LoRA. Compatible "
                        "repacked or alternate H3 SLA LoRAs use structural/full-mapping "
                        "validation instead of one fixed file hash."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(SLA_MODES),
                    default="apply_lightx2v_sla",
                    tooltip=(
                        "apply_lightx2v_sla uses a diagnostic sequence/forward policy that "
                        "failed full-duration human review and is not quality-safe. "
                        "Use the Turbo / SLA Profile Router for recommended Turbo8 output. "
                        "The upstream-exact EXP option keeps all forwards 85% sparse. "
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
