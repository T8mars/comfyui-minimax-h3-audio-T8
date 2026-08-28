from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .optical_flow_advanced import (
    MODEL_TYPES,
    PRECISIONS,
    RELEASE_POLICIES,
    audit_optical_flow,
    propagate_keyframe_masks,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Optical Flow"


def _model_names() -> list[str]:
    return list(folder_paths.get_filename_list("optical_flow"))


def _model_path(name: str) -> Path:
    return Path(folder_paths.get_full_path_or_raise("optical_flow", name))


class MiniMaxH3RAFTMotionAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RAFTMotionAuditT8Advanced",
            display_name="MiniMax H3 RAFT Motion Audit / 光流运动审计 (Advanced)",
            description=(
                "Read-only RAFT optical-flow audit for real motion, sudden cuts and temporal "
                "collapse. It does not repair faces, sharpen frames or modify H3 sampling. "
                "Model selection has no filename/hash/byte-size allowlist."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("frames"),
                io.Combo.Input("model_name", options=_model_names()),
                io.Combo.Input("model_type", options=list(MODEL_TYPES), default="raft_small"),
                io.Combo.Input("precision", options=list(PRECISIONS), default="auto"),
                io.Combo.Input("analysis_max_side", options=[384, 512, 640, 768, 0], default=640),
                io.Int.Input("pair_batch_size", default=1, min=1, max=8, advanced=True),
                io.Boolean.Input("consistency_check", default=True, advanced=True),
                io.Float.Input(
                    "scene_cut_threshold", default=0.20, min=0.01, max=1.0, step=0.01
                ),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                    tooltip=(
                        "offload_after removes only RAFT weights from GPU; it never calls the global "
                        "ComfyUI unload-all-models path."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("flow_preview"),
                io.String.Output("report_json"),
                io.Float.Output("mean_motion_px"),
                io.Float.Output("p95_motion_px"),
                io.Int.Output("scene_cut_count"),
            ],
        )

    @classmethod
    def execute(cls, model_name, **kwargs):
        return io.NodeOutput(
            *audit_optical_flow(
                model_path=_model_path(model_name), model_name=model_name, **kwargs
            )
        )


class MiniMaxH3RAFTMaskPropagationT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RAFTMaskPropagationT8Advanced",
            display_name="MiniMax H3 RAFT Mask Propagation / 光流遮罩传播 (Advanced)",
            description=(
                "Propagates one reviewed person/object mask through a shot with bidirectional RAFT "
                "and forward-backward confidence. Run once per identity. It stops at detected cuts "
                "and never assigns identity automatically."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Mask.Input("keyframe_masks"),
                io.String.Input(
                    "keyframe_indices",
                    default="0",
                    tooltip="Comma-separated 0-based frame indices; count must equal the MASK batch.",
                ),
                io.Combo.Input("model_name", options=_model_names()),
                io.Combo.Input("model_type", options=list(MODEL_TYPES), default="raft_small"),
                io.Combo.Input("precision", options=list(PRECISIONS), default="auto"),
                io.Combo.Input("analysis_max_side", options=[384, 512, 640, 768, 0], default=640),
                io.Int.Input("pair_batch_size", default=1, min=1, max=8, advanced=True),
                io.Float.Input(
                    "scene_cut_threshold", default=0.20, min=0.01, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "consistency_threshold",
                    default=2.0,
                    min=0.1,
                    max=20.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_confidence",
                    default=0.08,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Boolean.Input("extend_edges", default=True),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                ),
            ],
            outputs=[
                io.Mask.Output("propagated_masks"),
                io.Mask.Output("confidence_masks"),
                io.Image.Output("preview"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model_name, **kwargs):
        return io.NodeOutput(
            *propagate_keyframe_masks(
                model_path=_model_path(model_name), model_name=model_name, **kwargs
            )
        )


OPTICAL_FLOW_ADVANCED_NODE_CLASSES = [
    MiniMaxH3RAFTMotionAuditT8Advanced,
    MiniMaxH3RAFTMaskPropagationT8Advanced,
]
