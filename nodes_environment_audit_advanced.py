from __future__ import annotations

from comfy_api.latest import io

from .environment_audit import audit_h3_environment, blocking_summary, canonical_json


CATEGORY = "T8/MiniMax H3/Models/Experimental"


class MiniMaxH3EnvironmentAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnvironmentAuditT8Advanced",
            display_name="MiniMax H3 Environment Audit / 环境兼容审计 (Advanced)",
            description=(
                "Read-only audit of the current ComfyUI H3 core, known fix ancestry, VAE "
                "chunking/tiled-decode contracts, wrapper ownership, DynamicVRAM and a requested "
                "workload. When Sage is selected it also preserves the exact package/core import, "
                "KJ-required symbols and CUDA-architecture probe evidence. It never patches, "
                "unloads, downloads, runs an attention kernel, or changes settings."
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "workload_profile",
                    options=[
                        "general",
                        "t2va",
                        "ref2va",
                        "hybrid",
                        "long_video",
                        "multikeyframe",
                        "speech",
                    ],
                    default="general",
                ),
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "model_family",
                    options=["auto_unknown", "fl2va", "ref2va"],
                    default="auto_unknown",
                ),
                io.Combo.Input(
                    "model_precision",
                    options=[
                        "auto_unknown",
                        "bf16_fp16",
                        "int8_convrot",
                        "fp8",
                        "nvfp4",
                        "other_quant",
                    ],
                    default="auto_unknown",
                ),
                io.Combo.Input(
                    "attention_backend",
                    options=["auto_detect", "stock", "sage_attention", "other_custom"],
                    default="auto_detect",
                ),
                io.Combo.Input(
                    "cache_backend",
                    options=[
                        "auto_detect",
                        "none",
                        "t8_h3_block_cache",
                        "step_cache",
                        "spectrum",
                        "other_custom",
                    ],
                    default="auto_detect",
                ),
                io.Combo.Input(
                    "decode_mode",
                    options=["regular", "tiled"],
                    default="regular",
                ),
                io.Combo.Input(
                    "dynamic_vram_mode",
                    options=["auto_detect", "enabled", "disabled"],
                    default="auto_detect",
                ),
                io.Int.Input(
                    "reference_media_count",
                    default=0,
                    min=0,
                    max=32,
                    advanced=True,
                ),
                io.Int.Input(
                    "middle_keyframe_count",
                    default=0,
                    min=0,
                    max=32,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_current_headroom_mib",
                    default=512.0,
                    min=0.0,
                    max=65536.0,
                    step=16.0,
                    advanced=True,
                ),
                io.Combo.Input(
                    "enforcement",
                    options=["report_only", "block_known_unsafe"],
                    default="report_only",
                    advanced=True,
                ),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
            ],
            outputs=[
                io.Boolean.Output("no_known_blocker"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        workload_profile,
        width,
        height,
        length,
        model_family,
        model_precision,
        attention_backend,
        cache_backend,
        decode_mode,
        dynamic_vram_mode,
        reference_media_count,
        middle_keyframe_count,
        minimum_current_headroom_mib,
        enforcement,
        model=None,
        positive=None,
    ):
        report = audit_h3_environment(
            workload_profile,
            width,
            height,
            length,
            model_family,
            model_precision,
            attention_backend,
            cache_backend,
            decode_mode,
            dynamic_vram_mode,
            reference_media_count,
            middle_keyframe_count,
            minimum_current_headroom_mib,
            model,
            positive,
        )
        if enforcement == "block_known_unsafe" and not report["no_known_blocker"]:
            raise ValueError(
                "MiniMax H3 environment audit blocked execution: "
                + blocking_summary(report)
            )
        if enforcement not in {"report_only", "block_known_unsafe"}:
            raise ValueError(f"unsupported environment audit enforcement: {enforcement!r}")
        return io.NodeOutput(
            bool(report["no_known_blocker"]),
            str(report["status"]),
            canonical_json(report, indent=2),
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        # Runtime modules, git state and GPU/host memory can change without widget changes.
        return float("nan")


ENVIRONMENT_AUDIT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3EnvironmentAuditT8Advanced,
]
