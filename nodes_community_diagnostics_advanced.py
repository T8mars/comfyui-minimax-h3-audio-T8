from __future__ import annotations

from comfy_api.latest import io

from .community_diagnostics_advanced import (
    diagnose_official_h3_risks,
    probe_generic_loop_capability,
    probe_taeh3_preview_capability,
)


CATEGORY = "T8/MiniMax H3/Diagnostics/Experimental"


class MiniMaxH3GenericLoopCapabilityT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3GenericLoopCapabilityT8Advanced",
            display_name="MiniMax H3 Generic Loop Capability / 通用循环能力探针 (Advanced EXP/T8)",
            description=(
                "Read-only detection of the draft ComfyUI Generic Loops scheduler contract. "
                "It never enables a draft backend or changes the released T8 long-video runtime."
            ),
            category=CATEGORY,
            inputs=[],
            outputs=[
                io.Boolean.Output("available"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(*probe_generic_loop_capability())

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3OfficialRiskDiagnosticT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3OfficialRiskDiagnosticT8Advanced",
            display_name="MiniMax H3 Official Risk Diagnostic / 官方问题证据诊断 (Advanced EXP/T8)",
            description=(
                "Read-only evidence classifier for VRAM/V-copy, reference load, Sage FP8, "
                "audio integrity, multi-speaker cross-talk and periodic dark flashes. Unknown "
                "remains unknown; no model fingerprint or canvas-size hard gate is used."
            ),
            category=CATEGORY,
            inputs=[
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=1, max=100000),
                io.Int.Input("reference_media_count", default=0, min=0, max=128),
                io.Int.Input("speaker_count", default=1, min=0, max=64),
                io.Int.Input("isolated_voice_reference_count", default=0, min=0, max=64),
                io.Combo.Input(
                    "attention_backend",
                    options=["unknown", "stock", "sage", "sage_fp8", "sol", "other"],
                    default="unknown",
                ),
                io.String.Input("runtime_report_json", default="", multiline=True, advanced=True),
                io.String.Input("audio_report_json", default="", multiline=True, advanced=True),
                io.String.Input("frame_report_json", default="", multiline=True, advanced=True),
            ],
            outputs=[
                io.String.Output("status"),
                io.Int.Output("risk_count"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*diagnose_official_h3_risks(**kwargs))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3TAEH3PreviewCapabilityT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TAEH3PreviewCapabilityT8Advanced",
            display_name="MiniMax H3 TAEH3 Preview Capability / 快速预览能力 (Advanced EXP/T8)",
            description=(
                "Read-only inspection of ComfyUI's native TAEH3 progress-preview path and the "
                "installed models/vae_approx asset. It never changes preview settings or sampling."
            ),
            category=CATEGORY,
            inputs=[],
            outputs=[
                io.Boolean.Output("available"),
                io.Boolean.Output("active"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(*probe_taeh3_preview_capability())

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


COMMUNITY_DIAGNOSTICS_ADVANCED_NODE_CLASSES = [
    MiniMaxH3GenericLoopCapabilityT8Advanced,
    MiniMaxH3OfficialRiskDiagnosticT8Advanced,
    MiniMaxH3TAEH3PreviewCapabilityT8Advanced,
]
