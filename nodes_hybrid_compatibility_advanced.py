from __future__ import annotations

from comfy_api.latest import io

from .hybrid_compatibility import audit_hybrid_compatibility, hard_error_summary
from .hybrid_model import pretty_json


CATEGORY = "T8/MiniMax H3/Models/Experimental"


class MiniMaxH3HybridCompatibilityAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridCompatibilityAuditT8Advanced",
            display_name=(
                "MiniMax H3 Hybrid Compatibility Audit / 混合模型组合兼容审计 (Advanced)"
            ),
            description=(
                "Passes the exact same MODEL through while auditing Hybrid offset-set integrity, "
                "LoRA ordering/overlap, Block Cache, SageAttention, Long Video, MultiKeyframe, "
                "sampling, DynamicVRAM/VBAR policy provenance, current VRAM and host commit. "
                "Place it after every MODEL patch/sampler node and before BasicGuider."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "enforcement",
                    options=["report_only", "block_hard_conflicts"],
                    default="report_only",
                    tooltip=(
                        "report_only never blocks or mutates the MODEL. block_hard_conflicts "
                        "raises on mechanically proven patch/order/contract/memory-gate failures."
                    ),
                ),
                io.Boolean.Input(
                    "require_applied_vram_policy",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Require Loader provenance showing that a non-report-only T8 VRAM policy "
                        "was applied before the stock model load."
                    ),
                ),
                io.Float.Input(
                    "minimum_current_headroom_mib",
                    default=512.0,
                    min=0.0,
                    max=65536.0,
                    step=16.0,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_commit_headroom_gib",
                    default=16.0,
                    min=0.0,
                    max=1024.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Conditioning.Input(
                    "positive",
                    optional=True,
                    tooltip=(
                        "Optional final Conditioning lets the audit verify Long Video/"
                        "MultiKeyframe MODEL pairing and reference-modality coverage."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Boolean.Output("compatible"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        enforcement,
        require_applied_vram_policy,
        minimum_current_headroom_mib,
        minimum_commit_headroom_gib,
        positive=None,
    ):
        report = audit_hybrid_compatibility(
            model,
            positive,
            require_applied_vram_policy=require_applied_vram_policy,
            minimum_current_headroom_mib=minimum_current_headroom_mib,
            minimum_commit_headroom_gib=minimum_commit_headroom_gib,
        )
        if enforcement == "block_hard_conflicts" and not report["compatible"]:
            raise ValueError(
                "MiniMax H3 Hybrid compatibility audit blocked execution: "
                + hard_error_summary(report)
            )
        if enforcement not in {"report_only", "block_hard_conflicts"}:
            raise ValueError(f"unsupported Hybrid compatibility enforcement: {enforcement!r}")
        return io.NodeOutput(model, bool(report["compatible"]), pretty_json(report))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        # Runtime VRAM/commit and patch stacks can change without widget changes.
        return float("nan")


HYBRID_COMPATIBILITY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3HybridCompatibilityAuditT8Advanced,
]
