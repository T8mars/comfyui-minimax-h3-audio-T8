from __future__ import annotations

from comfy_api.latest import io

from .forward_sync_advanced import build_forward_sync_optimization


CATEGORY = "T8/MiniMax H3/Performance/Advanced"


class MiniMaxH3ForwardSyncOptimizationT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ForwardSyncOptimizationT8Advanced",
            display_name="MiniMax H3 Forward Sync Optimization (T8 Advanced)",
            category=CATEGORY,
            description=(
                "Reduce repeated H3 per-step device-to-host synchronization without "
                "changing schedules. Newer ComfyUI cores with the official behavior "
                "pass through natively."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(*build_forward_sync_optimization(model))


FORWARD_SYNC_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ForwardSyncOptimizationT8Advanced,
]
