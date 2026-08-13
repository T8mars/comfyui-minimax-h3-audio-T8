from __future__ import annotations

from comfy_api.latest import io

from .repair_compose_advanced import compose_repair_overlay
from .repair_execution_advanced import (
    accept_staged_repair,
    bind_repair_execution,
    canonical_json,
    stage_repair_candidate,
)
from .nodes_studio_advanced import RepairPlanIO


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
RepairExecutionIO = io.Custom("H3_T8_REPAIR_EXECUTION")
RepairStagedIO = io.Custom("H3_T8_REPAIR_STAGED")


class MiniMaxH3SelectiveRepairBindT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SelectiveRepairBindT8Advanced",
            display_name="MiniMax H3 Selective Repair Bind / 修复绑定 (Advanced)",
            description=(
                "Binds one repair-plan item to an immutable accepted-manifest revision and "
                "exposes the exact existing generation inputs. It does not load models, queue "
                "work or modify the accepted chain."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                RepairPlanIO.Input("repair_plan"),
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.Int.Input("repair_index", default=0, min=0, max=100000),
            ],
            outputs=[
                RepairExecutionIO.Output("repair_execution"),
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.Int.Output("length"),
                io.Int.Output("seed"),
                io.Int.Output("segment_index"),
                io.Float.Output("timeline_start_seconds"),
                io.String.Output("parent_candidate_id"),
                io.Int.Output("base_manifest_revision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, repair_plan, chain_id, repair_index):
        execution, report = bind_repair_execution(repair_plan, chain_id, repair_index)
        repair = execution["repair"]
        return io.NodeOutput(
            execution,
            repair["prompt"],
            repair["negative_prompt"],
            repair["frame_count"],
            repair["repair_seed"],
            repair["shot_index"],
            execution["timeline_start_seconds"],
            execution["parent_candidate_id"],
            execution["base_manifest"]["revision"],
            canonical_json(report),
        )

    @classmethod
    def fingerprint_inputs(cls, repair_plan, chain_id, repair_index):
        try:
            execution, _report = bind_repair_execution(
                repair_plan, chain_id, repair_index
            )
            return execution["execution_hash"]
        except (FileNotFoundError, TypeError, ValueError):
            return f"unresolved:{chain_id!r}:{repair_index!r}"


class MiniMaxH3SelectiveRepairStageT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SelectiveRepairStageT8Advanced",
            display_name="MiniMax H3 Selective Repair Stage / 修复暂存 (Advanced)",
            description=(
                "Verifies a generated candidate against the bound frame/sample/timeline and "
                "base-manifest contract. It remains preview-only and writes no repair overlay."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                RepairExecutionIO.Input("repair_execution"),
                io.String.Input("candidate_json_path", default="", force_input=True),
            ],
            outputs=[
                RepairStagedIO.Output("staged_repair"),
                io.String.Output("candidate_video_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, repair_execution, candidate_json_path):
        staged, video_path, report = stage_repair_candidate(
            repair_execution,
            candidate_json_path,
        )
        return io.NodeOutput(staged, video_path, canonical_json(report))


class MiniMaxH3SelectiveRepairAcceptT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SelectiveRepairAcceptT8Advanced",
            display_name="MiniMax H3 Selective Repair Accept / 修复接受 (Advanced)",
            description=(
                "Explicitly accepts a reviewed candidate into an atomic repair overlay. "
                "The original accepted manifest and every original segment stay unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                RepairStagedIO.Input("staged_repair"),
                io.Boolean.Input("accept_repair", default=False),
                io.Boolean.Input(
                    "replace_existing",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Only enable after reviewing a new take for a slot that already has "
                        "an accepted repair overlay."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("repair_manifest_path"),
                io.Boolean.Output("accepted"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, staged_repair, accept_repair, replace_existing):
        path, accepted, report = accept_staged_repair(
            staged_repair,
            accept_repair,
            replace_existing,
        )
        return io.NodeOutput(path, accepted, canonical_json(report))


class MiniMaxH3SelectiveRepairComposeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SelectiveRepairComposeT8Advanced",
            display_name="MiniMax H3 Selective Repair Compose / 修复合成 (Advanced)",
            description=(
                "Streams the immutable base segments plus accepted repair overlays into one "
                "bounded-memory MP4. base_rollback ignores the overlay without deleting anything."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.String.Input("repair_manifest_path", default="", force_input=True),
                io.Combo.Input(
                    "compose_mode",
                    options=["repair_overlay", "base_rollback"],
                    default="repair_overlay",
                ),
                io.String.Input("filename_prefix", default="H3_Selective_Repair"),
                io.Boolean.Input("require_final_segment", default=True),
                io.Combo.Input(
                    "audio_seam_policy",
                    options=["cosine_bridge", "none"],
                    default="cosine_bridge",
                ),
                io.Float.Input(
                    "bridge_ms",
                    default=5.0,
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
            ],
            outputs=[
                io.String.Output("output_video_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        chain_id,
        repair_manifest_path,
        compose_mode,
        filename_prefix,
        require_final_segment,
        audio_seam_policy,
        bridge_ms,
        crf,
    ):
        path, report = compose_repair_overlay(
            chain_id,
            repair_manifest_path,
            compose_mode,
            filename_prefix,
            require_final_segment,
            audio_seam_policy,
            bridge_ms,
            crf,
        )
        return io.NodeOutput(path, canonical_json(report))


REPAIR_EXECUTION_ADVANCED_NODE_CLASSES = [
    MiniMaxH3SelectiveRepairBindT8Advanced,
    MiniMaxH3SelectiveRepairStageT8Advanced,
    MiniMaxH3SelectiveRepairAcceptT8Advanced,
    MiniMaxH3SelectiveRepairComposeT8Advanced,
]
