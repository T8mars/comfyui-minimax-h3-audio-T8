from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .flashvsr_advanced import (
    MEMORY_POLICIES,
    MODEL_TYPE,
    PIPELINE_MODES,
    PLAN_TYPE,
    PRECISIONS,
    QUALITY_PROFILES,
    RELEASE_POLICIES,
    SPATIAL_STRATEGIES,
    build_flashvsr_plan,
    load_flashvsr_model,
    restore_flashvsr,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental/FlashVSR"
FlashVSRModelIO = io.Custom(MODEL_TYPE)
FlashVSRPlanIO = io.Custom(PLAN_TYPE)

folder_paths.add_model_folder_path("flashvsr", str(Path(folder_paths.models_dir) / "FlashVSR"))
folder_paths.add_model_folder_path("flashvsr", str(Path(folder_paths.models_dir) / "FlashVSR-v1.1"))


def _model_directories() -> list[str]:
    root = Path(folder_paths.models_dir)
    names = [
        path.name
        for path in root.glob("FlashVSR*")
        if path.is_dir()
    ]
    return sorted(set(names), key=lambda name: ("v1.1" not in name.lower(), name.lower())) or [
        "FlashVSR-v1.1"
    ]


class MiniMaxH3FlashVSRModelT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FlashVSRModelT8Advanced",
            display_name="MiniMax H3 FlashVSR Model / 视频超分模型 (T8 Advanced EXP)",
            description=(
                "Loads the official FlashVSR/FlashVSR-v1.1 checkpoint set from ComfyUI/models. "
                "No hash, file-size or pixel limit is enforced; missing structural files remain actionable errors."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_directory", options=_model_directories(), default="FlashVSR-v1.1"),
                io.Combo.Input("mode", options=list(PIPELINE_MODES), default="tiny"),
                io.Combo.Input("precision", options=list(PRECISIONS), default="bf16"),
            ],
            outputs=[FlashVSRModelIO.Output("flashvsr_model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model_directory, mode, precision):
        model_dir = Path(folder_paths.models_dir) / model_directory
        return io.NodeOutput(
            *load_flashvsr_model(
                model_dir=model_dir,
                model_name=model_directory,
                mode=mode,
                precision=precision,
            )
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3FlashVSRExecutionPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FlashVSRExecutionPlanT8Advanced",
            display_name="MiniMax H3 FlashVSR Plan / 质量与显存计划 (T8 Advanced EXP)",
            description=(
                "Plans fixed quality, opt-in motion-aware budgets, or memory-safe tiling. "
                "quality_locked keeps the public 2.0/3.0/11 LCSA settings unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Combo.Input("quality_profile", options=list(QUALITY_PROFILES), default="quality_locked"),
                io.Combo.Input("spatial_strategy", options=list(SPATIAL_STRATEGIES), default="auto"),
                io.Combo.Input("memory_policy", options=list(MEMORY_POLICIES), default="auto"),
                io.Float.Input("base_attention_budget", default=2.0, min=1.5, max=2.0, step=0.05),
                io.Float.Input("kv_retention", default=3.0, min=1.0, max=3.0, step=0.5),
                io.Int.Input("local_radius", default=11, min=9, max=11, step=2),
                io.Int.Input("tile_size", default=256, min=128, max=1024, step=128, advanced=True),
                io.Int.Input("tile_overlap", default=24, min=0, max=128, step=8, advanced=True),
            ],
            outputs=[FlashVSRPlanIO.Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, frames, **kwargs):
        return io.NodeOutput(*build_flashvsr_plan(frames, **kwargs))


class MiniMaxH3FlashVSRRestoreT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FlashVSRRestoreT8Advanced",
            display_name="MiniMax H3 FlashVSR Restore / 视频超分执行 (T8 Advanced EXP)",
            description=(
                "Runs the selected FlashVSR plan, trims repeated tail padding and returns the exact original "
                "AUDIO object. No resolution ceiling is imposed; use memory_safe when resources are limited."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                FlashVSRModelIO.Input("flashvsr_model"),
                FlashVSRPlanIO.Input("plan"),
                io.Image.Input("frames"),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("scale", default=2, min=2, max=4, step=2),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("color_fix", default=True),
                io.Combo.Input("release_policy", options=list(RELEASE_POLICIES), default="offload_after"),
            ],
            outputs=[
                io.Image.Output("restored_frames"),
                io.Image.Output("source_frames"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, flashvsr_model, plan, frames, **kwargs):
        kwargs.setdefault("audio", None)
        return io.NodeOutput(
            *restore_flashvsr(flashvsr_model, plan, frames, **kwargs)
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


FLASHVSR_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FlashVSRModelT8Advanced,
    MiniMaxH3FlashVSRExecutionPlanT8Advanced,
    MiniMaxH3FlashVSRRestoreT8Advanced,
]
