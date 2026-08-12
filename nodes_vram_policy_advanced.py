from __future__ import annotations

import json

from comfy_api.latest import io

from .vram_policy import (
    VRAM_POLICY_MODES,
    VRAM_POLICY_TYPE,
    build_vram_policy,
    policy_input_fingerprint,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"
VRAMPolicyIO = io.Custom(VRAM_POLICY_TYPE)


class MiniMaxH3VRAMPolicyT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VRAMPolicyT8Advanced",
            display_name=(
                "MiniMax H3 VRAM Policy / VBAR显存预留策略 (Advanced)"
            ),
            description=(
                "Plans a guarded process-global ComfyUI reserve and DynamicVRAM/VBAR simple "
                "headroom. The default is report-only. Connect the policy to the Advanced "
                "Hybrid Loader to guarantee it is applied before model loading."
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=list(VRAM_POLICY_MODES),
                    default="report_only",
                    tooltip=(
                        "report_only has no side effects. fixed_total_reserved_exp sets a total "
                        "reserve. external_usage_plus_margin_exp requires global cleanup first."
                    ),
                ),
                io.Float.Input(
                    "fixed_total_reserved_gib",
                    default=4.0,
                    min=0.0,
                    max=16.0,
                    step=0.1,
                    tooltip=(
                        "4.0 GiB is the validated conservative starting point for the exact "
                        "RTX 4060 Ti 16 GiB, 736x416, 124-frame Hybrid Stock20 workflow. "
                        "It is not a universal safe value; re-measure other GPUs and workloads."
                    ),
                ),
                io.Float.Input(
                    "external_margin_gib",
                    default=1.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                ),
                io.Float.Input(
                    "maximum_reserved_gib",
                    default=8.0,
                    min=0.25,
                    max=16.0,
                    step=0.25,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "clean_before_load",
                    default=False,
                    tooltip=(
                        "Global side effect: unloads every ComfyUI model before measuring/applying. "
                        "Required by external_usage_plus_margin_exp."
                    ),
                ),
                io.Boolean.Input(
                    "require_dynamic_vram",
                    default=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_current_headroom_mib",
                    default=512.0,
                    min=0.0,
                    max=16384.0,
                    step=16.0,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_commit_headroom_gib",
                    default=16.0,
                    min=0.0,
                    max=512.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "block_when_commit_below_gate",
                    default=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "policy_epoch",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    tooltip="Increment to explicitly invalidate ComfyUI's cached policy output.",
                    advanced=True,
                ),
            ],
            outputs=[
                VRAMPolicyIO.Output("vram_policy"),
                io.Boolean.Output("current_gate_pass"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        mode,
        fixed_total_reserved_gib,
        external_margin_gib,
        maximum_reserved_gib,
        clean_before_load,
        require_dynamic_vram,
        minimum_current_headroom_mib,
        minimum_commit_headroom_gib,
        block_when_commit_below_gate,
        policy_epoch,
    ):
        values = {
            "mode": mode,
            "fixed_total_reserved_gib": fixed_total_reserved_gib,
            "external_margin_gib": external_margin_gib,
            "maximum_reserved_gib": maximum_reserved_gib,
            "clean_before_load": clean_before_load,
            "require_dynamic_vram": require_dynamic_vram,
            "minimum_current_headroom_mib": minimum_current_headroom_mib,
            "minimum_commit_headroom_gib": minimum_commit_headroom_gib,
            "block_when_commit_below_gate": block_when_commit_below_gate,
            "policy_epoch": policy_epoch,
        }
        policy, report = build_vram_policy(**values)
        return io.NodeOutput(
            policy,
            bool(report["current_gate_pass"]),
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        mode,
        fixed_total_reserved_gib,
        external_margin_gib,
        maximum_reserved_gib,
        clean_before_load,
        require_dynamic_vram,
        minimum_current_headroom_mib,
        minimum_commit_headroom_gib,
        block_when_commit_below_gate,
        policy_epoch,
    ):
        return policy_input_fingerprint(
            mode=mode,
            fixed_total_reserved_gib=fixed_total_reserved_gib,
            external_margin_gib=external_margin_gib,
            maximum_reserved_gib=maximum_reserved_gib,
            clean_before_load=clean_before_load,
            require_dynamic_vram=require_dynamic_vram,
            minimum_current_headroom_mib=minimum_current_headroom_mib,
            minimum_commit_headroom_gib=minimum_commit_headroom_gib,
            block_when_commit_below_gate=block_when_commit_below_gate,
            policy_epoch=policy_epoch,
        )


VRAM_POLICY_ADVANCED_NODE_CLASSES = [MiniMaxH3VRAMPolicyT8Advanced]
