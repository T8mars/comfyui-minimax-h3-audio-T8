from __future__ import annotations

from comfy_api.latest import io

from .creator_workspace_advanced import (
    add_creator_shot_override,
    build_synchronized_comparison,
    compile_creator_workspace,
    select_creator_workspace_shot,
)
from .nodes_studio_advanced import PromptPacketIO, StudioTimelineIO


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
CreatorEditIO = io.Custom("H3_T8_CREATOR_EDIT_PLAN")
CreatorWorkspaceIO = io.Custom("H3_T8_CREATOR_WORKSPACE")


class MiniMaxH3CreatorShotOverrideT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorShotOverrideT8Advanced",
            display_name="MiniMax H3 Creator Shot Override / 可编辑镜头卡 (Advanced/T8)",
            description=(
                "Adds one non-destructive overlay to an existing Studio Timeline shot: prompt, "
                "seed variants, media roles, retention and hold metadata. Chain one node per shot."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                StudioTimelineIO.Input("timeline"),
                io.Int.Input("shot_index", default=0, min=0, max=100000),
                io.Boolean.Input("enabled", default=True),
                io.String.Input("compiled_prompt_override", multiline=True, default=""),
                io.Boolean.Input("use_seed_override", default=False),
                io.Int.Input("seed_override", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("variant_count", default=1, min=1, max=64),
                io.Int.Input("variant_seed_stride", default=1, min=1, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input(
                    "media_roles_json",
                    multiline=True,
                    default='{"picture_1":"first_frame","audio_1":"voice_reference"}',
                ),
                io.Combo.Input(
                    "retention_policy",
                    options=[
                        "keep_all",
                        "keep_winner_and_metadata",
                        "keep_accepted_only",
                        "metadata_only",
                    ],
                    default="keep_winner_and_metadata",
                ),
                io.Combo.Input(
                    "hold_policy",
                    options=["none", "hold_first", "hold_last", "hold_both", "custom_metadata"],
                    default="none",
                ),
                io.Int.Input("hold_frames", default=0, min=0, max=100000),
                CreatorEditIO.Input("previous_edits", optional=True),
            ],
            outputs=[CreatorEditIO.Output("edit_plan"), io.String.Output("edit_plan_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*add_creator_shot_override(**kwargs))


class MiniMaxH3CreatorWorkspaceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorWorkspaceT8Advanced",
            display_name="MiniMax H3 Creator Workspace / 创作运行窗口 (Advanced/T8)",
            description=(
                "Compiles run-from/run-to selection, variants, retention and hold-map sidecar. "
                "It never queues, loads a model, deletes media or changes sampler math."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                StudioTimelineIO.Input("timeline"),
                io.Int.Input("run_from_index", default=0, min=0, max=100000),
                io.Int.Input("run_to_index", default=-1, min=-1, max=100000),
                io.Boolean.Input("include_disabled_shots", default=False),
                io.String.Input("workspace_notes", multiline=True, default=""),
                CreatorEditIO.Input("edit_plan", optional=True),
            ],
            outputs=[
                CreatorWorkspaceIO.Output("workspace"),
                io.String.Output("run_summary"),
                io.String.Output("sidecar_json"),
                io.String.Output("workspace_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*compile_creator_workspace(**kwargs))


class MiniMaxH3CreatorWorkspaceShotSelectT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorWorkspaceShotSelectT8Advanced",
            display_name="MiniMax H3 Creator Workspace Shot Select / 工作区镜头选择 (Advanced/T8)",
            description=(
                "Selects one run position and one deterministic variant seed for existing H3 "
                "Conditioning/Sampler nodes. No hidden queue execution is performed."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.Int.Input("run_position", default=0, min=0, max=100000),
                io.Int.Input("variant_index", default=0, min=0, max=63),
            ],
            outputs=[
                PromptPacketIO.Output("prompt_packet"),
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.Int.Output("length"),
                io.Int.Output("seed"),
                io.String.Output("shot_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, workspace, run_position, variant_index):
        return io.NodeOutput(
            *select_creator_workspace_shot(workspace, run_position, variant_index)
        )


class MiniMaxH3CreatorSynchronizedCompareT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorSynchronizedCompareT8Advanced",
            display_name="MiniMax H3 Creator Synchronized A/B / 同步候选对比 (Advanced/T8)",
            description=(
                "Creates a labelled side-by-side IMAGE batch on CPU, preserving aspect ratio by "
                "center padding and trimming only to the shorter frame count. Audio is not compared."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("frames_a"),
                io.Image.Input("frames_b"),
                io.String.Input("label_a", default="candidate_a"),
                io.String.Input("label_b", default="candidate_b"),
                io.Int.Input("seed_a", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("seed_b", default=1, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input("winner", options=["ABSTAIN", "TIE", "A", "B"], default="ABSTAIN"),
                io.String.Input("reviewer_notes", multiline=True, default=""),
                io.Boolean.Input("require_equal_geometry", default=True),
            ],
            outputs=[
                io.Image.Output("comparison_frames"),
                io.String.Output("winner"),
                io.Int.Output("selected_seed"),
                io.String.Output("review_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_synchronized_comparison(**kwargs))


CREATOR_WORKSPACE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3CreatorShotOverrideT8Advanced,
    MiniMaxH3CreatorWorkspaceT8Advanced,
    MiniMaxH3CreatorWorkspaceShotSelectT8Advanced,
    MiniMaxH3CreatorSynchronizedCompareT8Advanced,
]
