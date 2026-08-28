from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .realbasicvsr_advanced import (
    CHECKPOINT_BRANCHES,
    OUTPUT_MODES,
    PRECISIONS,
    RELEASE_POLICIES,
    restore_realbasicvsr,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental/Temporal Restoration"


def _model_names() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list("upscale_models")
        if Path(name).suffix.lower() in {".pth", ".pt", ".safetensors"}
    ]


def _model_path(name: str) -> Path:
    return Path(folder_paths.get_full_path_or_raise("upscale_models", name))


class MiniMaxH3RealBasicVSRRestoreT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RealBasicVSRRestoreT8Advanced",
            display_name="MiniMax H3 RealBasicVSR Temporal Restore / 时序修复 (Advanced)",
            description=(
                "Optional post-process based on the official OpenMMLab RealBasicVSR x4 "
                "architecture. It restores temporal/detail consistency in small serial chunks, "
                "keeps AUDIO as the exact original object and never downloads a model at runtime."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Audio.Input("audio", optional=True),
                io.Combo.Input("model_name", options=_model_names()),
                io.Combo.Input("output_mode", options=list(OUTPUT_MODES), default="native_size_restore"),
                io.Float.Input("strength", default=0.65, min=0.0, max=1.0, step=0.01),
                io.Int.Input("chunk_frames", default=8, min=2, max=32),
                io.Int.Input("overlap_frames", default=2, min=0, max=16),
                io.Combo.Input("precision", options=list(PRECISIONS), default="auto"),
                io.Combo.Input(
                    "checkpoint_branch",
                    options=list(CHECKPOINT_BRANCHES),
                    default="prefer_ema",
                    advanced=True,
                ),
                io.Combo.Input(
                    "release_policy", options=list(RELEASE_POLICIES), default="offload_after"
                ),
            ],
            outputs=[
                io.Image.Output("restored_frames"),
                io.Image.Output("source_frames"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model_name, **kwargs):
        return io.NodeOutput(
            *restore_realbasicvsr(
                model_path=_model_path(model_name), model_name=model_name, **kwargs
            )
        )


REALBASICVSR_ADVANCED_NODE_CLASSES = [MiniMaxH3RealBasicVSRRestoreT8Advanced]

