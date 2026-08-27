from __future__ import annotations

from comfy_api.latest import io

from .creator_segment_cache_advanced import (
    CREATOR_SEGMENT_CACHE_PLAN_TYPE,
    compile_creator_segment_cache_plan,
)
from .nodes_creator_workspace_advanced import CreatorWorkspaceIO


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
CreatorSegmentCachePlanIO = io.Custom(CREATOR_SEGMENT_CACHE_PLAN_TYPE)


class MiniMaxH3CreatorSegmentCacheT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorSegmentCacheT8Advanced",
            display_name="MiniMax H3 Creator Segment Cache / 创作分段语义缓存计划 (Advanced EXP/T8)",
            description=(
                "Hashes source/model/LoRA/prompt/sampling/effect contracts per shot variant, "
                "reports hits and scoped invalidation, and proposes LRU quarantine under byte/count "
                "limits. It never opens, moves or deletes artifacts; accepted media stays protected."
            ),
            category=CATEGORY,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.String.Input("model_contract_json", default='{"model":"unknown"}', multiline=True),
                io.String.Input("lora_contract_json", default='{"loras":[]}', multiline=True),
                io.String.Input("sampling_contract_json", default='{"steps":8,"shift_video":12,"shift_audio":3}', multiline=True),
                io.String.Input("effect_plan_json", default='{"effects":[]}', multiline=True),
                io.String.Input("cache_index_json", default="", multiline=True, advanced=True),
                io.Float.Input("maximum_cache_gib", default=20.0, min=0.1, max=2048.0, step=0.1),
                io.Int.Input("maximum_entries", default=100, min=1, max=100000),
            ],
            outputs=[
                CreatorSegmentCachePlanIO.Output("cache_plan"),
                io.String.Output("status"),
                io.Int.Output("hit_count"),
                io.Int.Output("invalid_count"),
                io.Int.Output("proposed_quarantine_count"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*compile_creator_segment_cache_plan(**kwargs))


CREATOR_SEGMENT_CACHE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3CreatorSegmentCacheT8Advanced,
]
