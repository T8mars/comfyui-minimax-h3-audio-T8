from __future__ import annotations

from comfy_api.latest import io

from .external_compatibility_advanced import (
    audit_clipproj_compatibility,
    audit_sol_attn_compatibility,
)


CATEGORY = "T8/MiniMax H3/System/Experimental"


class MiniMaxH3ClipProjCompatibilityAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
            display_name="MiniMax H3 ClipProj Compatibility Audit / 小编码器投影审计 (Advanced/T8)",
            description=(
                "Passes the connected CLIP through unchanged while auditing a separately installed "
                "ComfyUI-ClipProj wrapper, version, projection dimensions, Qwen3-VL declaration, "
                "load mode and visual-reference boundary. It never loads an encoder or matrix."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "encoder_family",
                    options=["auto", "4B", "8B", "32B"],
                    default="auto",
                ),
                io.Combo.Input(
                    "encoder_architecture",
                    options=["unknown", "qwen3_vl", "text_only_qwen3"],
                    default="unknown",
                ),
                io.Combo.Input(
                    "encoder_quantization",
                    options=["unknown", "bf16", "fp8", "nvfp4", "int8_convrot", "gguf", "int4"],
                    default="unknown",
                ),
                io.Combo.Input(
                    "load_mode",
                    options=["stock_pageable", "clipproj_dynamic", "clipproj_resident"],
                    default="stock_pageable",
                ),
                io.String.Input(
                    "projection_path",
                    default="",
                    tooltip=(
                        "Absolute path or filename under ComfyUI/models/clip_projections. "
                        "The header is read without loading tensors."
                    ),
                ),
                io.Boolean.Input("has_reference_images", default=False),
                io.Boolean.Input("has_reference_videos", default=False),
                io.Combo.Input(
                    "enforcement",
                    options=["report_only", "block_hard_conflicts"],
                    default="report_only",
                ),
            ],
            outputs=[
                io.Clip.Output("clip"),
                io.Boolean.Output("compatible"),
                io.String.Output("decision"),
                io.String.Output("plugin_version"),
                io.Int.Output("projection_input_dim"),
                io.Int.Output("projection_output_dim"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, enforcement, **kwargs):
        result = audit_clipproj_compatibility(**kwargs)
        if enforcement == "block_hard_conflicts" and not result[1]:
            raise ValueError(
                "MiniMax H3 ClipProj compatibility audit blocked execution: " + result[-1]
            )
        if enforcement not in {"report_only", "block_hard_conflicts"}:
            raise ValueError(f"unsupported ClipProj audit enforcement: {enforcement!r}")
        return io.NodeOutput(*result)

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3SolAttnCompatibilityAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolAttnCompatibilityAuditT8Advanced",
            display_name="MiniMax H3 Sol-Attn Compatibility Audit / 外部补丁所有权审计 (Advanced/T8)",
            description=(
                "Passes MODEL through unchanged while checking a separately installed Sol-Attn "
                "version, CUDA/BF16 architecture, complete H3 attention ownership, shadowing and "
                "unreviewed DiT/model-wrapper combinations. It does not import or run the kernel."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "intended_route",
                    options=["h3_memory_efficient", "h3_scheduled", "generic_sol", "audit_only"],
                    default="h3_scheduled",
                ),
                io.String.Input(
                    "expected_dense_blocks",
                    default="",
                    tooltip=(
                        "Must match the upstream Sol node's dense_blocks widget, e.g. 0-2,-1. "
                        "Those blocks intentionally have no Sol forward patch."
                    ),
                ),
                io.Boolean.Input(
                    "allow_unreviewed_composition",
                    default=False,
                    tooltip=(
                        "When false, DiT replacements, whole-model wrappers and unknown full-block "
                        "patches force ABSTAIN. True downgrades only that finding to a warning."
                    ),
                ),
                io.Combo.Input(
                    "enforcement",
                    options=["report_only", "block_hard_conflicts"],
                    default="report_only",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Boolean.Output("compatible"),
                io.String.Output("decision"),
                io.String.Output("plugin_version"),
                io.Int.Output("h3_sol_block_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, enforcement, **kwargs):
        result = audit_sol_attn_compatibility(**kwargs)
        if enforcement == "block_hard_conflicts" and not result[1]:
            raise ValueError(
                "MiniMax H3 Sol-Attn compatibility audit blocked execution: " + result[-1]
            )
        if enforcement not in {"report_only", "block_hard_conflicts"}:
            raise ValueError(f"unsupported Sol-Attn audit enforcement: {enforcement!r}")
        return io.NodeOutput(*result)

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


EXTERNAL_COMPATIBILITY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ClipProjCompatibilityAuditT8Advanced,
    MiniMaxH3SolAttnCompatibilityAuditT8Advanced,
]
