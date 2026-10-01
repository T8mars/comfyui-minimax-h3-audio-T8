"""Non-output audit variants for a single-sink checkpoint delivery graph.

The existing audit logic and schemas remain unchanged for older workflows.
"""
from __future__ import annotations

from .motion_nodes import MiniMaxH3MotionStageAuditEXPT8
from .nodes import MiniMaxH3StageEAVAuditEXPT8


class MiniMaxH3MotionStageAuditPassThroughEXPT8(MiniMaxH3MotionStageAuditEXPT8):
    _OUTPUT_NODE = None

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3MotionStageAuditPassThroughEXPT8"
        schema.display_name = "H3 Motion Recovery · Audit Pass 2 In-line (T8 EXP)"
        schema.description = (
            "The same source-bound second-pass candidate audit, only as an "
            "in-line dependency of a single final output sink."
        )
        schema.is_output_node = False
        return schema


class MiniMaxH3StageEAVAuditPassThroughEXPT8(MiniMaxH3StageEAVAuditEXPT8):
    _OUTPUT_NODE = None

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3StageEAVAuditPassThroughEXPT8"
        schema.display_name = "H3 Stage EAV · Actual Calls Audit In-line (T8 EXP)"
        schema.description = (
            "The same actual-call EAV audit, only as an in-line dependency of "
            "a single final output sink."
        )
        schema.is_output_node = False
        return schema
