from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .sla_attention_advanced import (
    SLA_BASE_POLICIES,
    SLA_LORA_FILENAME,
    SLA_RUNTIME_TYPE,
)
from .sla_profile_router_advanced import (
    CONSUMER_TURBO_PROFILE,
    CORRECTED_TURBO_LORA_FILENAME,
    DISABLED_PROFILE,
    PROFILE_OPTIONS,
    SLA_EXACT_PROFILE,
    SLA_INT8_BYPASS_END_PERCENT,
    SLA_INT8_BYPASS_PROFILE,
    SLA_INT8_BYPASS_START_PERCENT,
    build_turbo_sla_profile_model,
)


CATEGORY = "T8/MiniMax H3/Performance/Experimental"
SLARuntimeIO = io.Custom(SLA_RUNTIME_TYPE)


def _lora_options(default: str) -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    if default in names:
        names.remove(default)
    return [default, *names]


class MiniMaxH3TurboSLAProfileRouterT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TurboSLAProfileRouterT8Advanced",
            display_name="MiniMax H3 Turbo / SLA Profile Router (Advanced EXP)",
            description=(
                "Separates the project-validated consumer Turbo8 route from the exact "
                "LightX2V SLA experiment. Consumer Turbo8 loads the corrected Alpha8 "
                "bypass LoRA and requires 8 NFE with video/audio shift 12/3. SLA exact "
                "loads the selected SLA LoRA and requires the published 4 NFE with "
                "shift 6/3. The released evidence used BF16/FP8, but other selected "
                "base families are reported rather than blocked. It "
                "never silently runs an SLA LoRA as a dense Turbo fallback. The explicit "
                "INT8 bypass experiment keeps the quantized base untouched and applies "
                "the SLA residual dynamically; it is not a validated quality preset."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect the Dual-Clock MODEL, optionally through the complete "
                        "KJNodes MiniMax H3 Sage patch. Do not connect an external LoRA."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "Connect SIGMAS from the same Dual-Clock node. Consumer Turbo8 "
                        "requires 8/12/3; SLA exact requires 4/6/3. A mismatch fails before "
                        "sampling instead of producing a misleading collapsed video."
                    ),
                ),
                io.Combo.Input(
                    "turbo_lora_name",
                    options=_lora_options(CORRECTED_TURBO_LORA_FILENAME),
                    default=CORRECTED_TURBO_LORA_FILENAME,
                    tooltip=(
                        "Regular Turbo LoRA for the recommended 8-step route. File identity, "
                        "metadata and structure are reported only and never block loading."
                    ),
                ),
                io.Combo.Input(
                    "sla_lora_name",
                    options=_lora_options(SLA_LORA_FILENAME),
                    default=SLA_LORA_FILENAME,
                    tooltip=(
                        "LightX2V SLA LoRA used only by the upstream-exact SLA profile. "
                        "Alternate files are passed to the real loader without an identity gate."
                    ),
                ),
                io.Combo.Input(
                    "profile",
                    options=list(PROFILE_OPTIONS),
                    default=CONSUMER_TURBO_PROFILE,
                    tooltip=(
                        "consumer_turbo8_recommended: corrected regular Turbo LoRA, 8/12/3, "
                        "dense or authenticated KJ Sage attention. "
                        "sla_4step_upstream_exact_exp: SLA LoRA, 4/6/3, exact released 85% "
                        "sparse experiment; base-family matching is diagnostic only. "
                        "sla_4step_int8_bypass_exp: 4/6/3 INT8 ConvRot research route using "
                        "dynamic model-only LoRA bypass to avoid base re-quantization. "
                        "disabled_identity: no LoRA or attention change."
                    ),
                ),
                io.Combo.Input(
                    "base_policy",
                    options=list(SLA_BASE_POLICIES),
                    default="auto_detect_exp",
                    advanced=True,
                    tooltip=(
                        "Records whether the base resembles the published BF16/FP8 evidence. "
                        "Unknown, quantized and INT8 bases still pass through to real execution."
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
                        "Fail-closed SLA router workspace ceiling. It affects only the SLA "
                        "profile and is not a whole-workflow VRAM limit."
                    ),
                ),
                io.Float.Input(
                    "sla_start_percent",
                    default=SLA_INT8_BYPASS_START_PERCENT,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "INT8 bypass only: begin SLA sparse attention at this ComfyUI "
                        "denoising progress. Default 0.15 keeps the first 4-step model "
                        "forward dense. Other profiles ignore this range."
                    ),
                ),
                io.Float.Input(
                    "sla_end_percent",
                    default=SLA_INT8_BYPASS_END_PERCENT,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "INT8 bypass only: stop SLA sparse attention after this denoising "
                        "progress. With 4 NFE, 0.15-0.90 maps to dense/sparse/sparse/sparse "
                        "because model calls begin at 0%, 25%, 50% and 75%."
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
        turbo_lora_name,
        sla_lora_name,
        profile,
        base_policy,
        max_router_workspace_mib,
        sla_start_percent=SLA_INT8_BYPASS_START_PERCENT,
        sla_end_percent=SLA_INT8_BYPASS_END_PERCENT,
    ):
        turbo_lora_path = ""
        sla_lora_path = ""
        if profile == CONSUMER_TURBO_PROFILE:
            turbo_lora_path = folder_paths.get_full_path_or_raise(
                "loras", turbo_lora_name
            )
        elif profile in {SLA_EXACT_PROFILE, SLA_INT8_BYPASS_PROFILE}:
            sla_lora_path = folder_paths.get_full_path_or_raise(
                "loras", sla_lora_name
            )
        elif profile != DISABLED_PROFILE:
            raise ValueError(f"Unknown H3 Turbo/SLA profile {profile!r}")
        return io.NodeOutput(
            *build_turbo_sla_profile_model(
                model,
                sigmas,
                turbo_lora_path=turbo_lora_path,
                sla_lora_path=sla_lora_path,
                profile=profile,
                base_policy=base_policy,
                max_router_workspace_mib=max_router_workspace_mib,
                sla_start_percent=sla_start_percent,
                sla_end_percent=sla_end_percent,
            )
        )


SLA_PROFILE_ROUTER_ADVANCED_NODE_CLASSES = [
    MiniMaxH3TurboSLAProfileRouterT8Advanced,
]
