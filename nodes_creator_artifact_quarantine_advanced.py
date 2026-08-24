from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .creator_artifact_quarantine_advanced import (
    QUARANTINE_ACTIONS,
    execute_creator_artifact_quarantine,
)


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
CreatorRetentionPlanIO = io.Custom("H3_T8_CREATOR_RETENTION_PLAN")
CreatorQuarantineManifestIO = io.Custom("H3_T8_CREATOR_QUARANTINE_MANIFEST")


def creator_quarantine_output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


class MiniMaxH3CreatorArtifactQuarantineT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorArtifactQuarantineT8Advanced",
            display_name=(
                "MiniMax H3 Creator Artifact Quarantine / "
                "候选文件可恢复隔离 (Advanced EXP/T8)"
            ),
            description=(
                "Append-only executor for a reviewed Creator Retention Plan. prepare_only hashes "
                "every proposed file without mutation. quarantine requires the retained manifest, "
                "exact plan hash, a new epoch and explicit confirmation, then atomically moves "
                "files into output/MiniMaxH3/creator_quarantine. restore/recover_to_source returns "
                "them to their original paths. It never permanently deletes files."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorRetentionPlanIO.Input("retention_plan"),
                io.Combo.Input(
                    "action",
                    options=list(QUARANTINE_ACTIONS),
                    default="prepare_only",
                    tooltip=(
                        "Always run prepare_only first and keep its manifest_json. quarantine "
                        "moves reviewed files; restore/recover_to_source moves them back."
                    ),
                ),
                io.String.Input(
                    "execution_manifest_json",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Empty for prepare_only. For quarantine, paste the exact prepared manifest. "
                        "For restore, the quarantine receipt or original manifest is accepted."
                    ),
                ),
                io.String.Input(
                    "expected_plan_hash",
                    default="",
                    tooltip=(
                        "Required for every mutating action and must exactly match the reviewed "
                        "Retention Plan plan_hash."
                    ),
                ),
                io.Int.Input(
                    "execution_epoch",
                    default=0,
                    min=0,
                    max=2_147_483_647,
                    step=1,
                    tooltip=(
                        "0 for prepare_only. Use a new positive integer for quarantine and reuse "
                        "the same value for restore/recovery."
                    ),
                ),
                io.Boolean.Input(
                    "confirm_action",
                    default=False,
                    tooltip=(
                        "False is non-mutating. Must be enabled explicitly for quarantine, restore "
                        "or recovery after reviewing paths and hashes."
                    ),
                ),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="Bounded CPU read chunk for SHA-256 verification.",
                ),
            ],
            outputs=[
                CreatorQuarantineManifestIO.Output("execution_manifest"),
                io.String.Output("status"),
                io.String.Output("manifest_json"),
                io.Int.Output("file_count"),
                io.Int.Output("total_bytes"),
                io.String.Output("receipt_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *execute_creator_artifact_quarantine(
                output_root=creator_quarantine_output_root(),
                **kwargs,
            )
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        retention_plan,
        action,
        execution_manifest_json,
        expected_plan_hash,
        execution_epoch,
        confirm_action,
        hash_chunk_megabytes,
    ):
        plan_hash = (
            retention_plan.get("plan_hash")
            if isinstance(retention_plan, dict)
            else "unresolved"
        )
        return (
            f"{plan_hash}:{action}:{execution_manifest_json}:{expected_plan_hash}:"
            f"{int(execution_epoch)}:{bool(confirm_action)}:{int(hash_chunk_megabytes)}"
        )


CREATOR_ARTIFACT_QUARANTINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3CreatorArtifactQuarantineT8Advanced,
]
