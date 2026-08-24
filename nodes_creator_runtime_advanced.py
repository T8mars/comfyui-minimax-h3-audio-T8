from __future__ import annotations

import json

from comfy_api.latest import io

from .creator_runtime_advanced import (
    CACHE_OBSERVATIONS,
    OUTCOMES,
    compile_creator_background_selection,
    compile_creator_retention_plan,
    compile_creator_resume_plan,
    creator_background_binding,
    record_creator_run_receipt,
)
from .long_video_background import (
    BACKGROUND_JOBS,
    BACKGROUND_SCHEMA,
    BACKGROUND_STATE_FORMAT,
    EXECUTION_MODES,
    RELEASE_POLICIES,
)
from .nodes_creator_workspace_advanced import CreatorWorkspaceIO
from .nodes_studio_advanced import PromptPacketIO


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
CreatorRunLedgerIO = io.Custom("H3_T8_CREATOR_RUN_LEDGER")
CreatorResumePlanIO = io.Custom("H3_T8_CREATOR_RESUME_PLAN")
CreatorRetentionPlanIO = io.Custom("H3_T8_CREATOR_RETENTION_PLAN")


class MiniMaxH3CreatorRunReceiptT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorRunReceiptT8Advanced",
            display_name=(
                "MiniMax H3 Creator Run Receipt / 候选运行回执 (Advanced/T8)"
            ),
            description=(
                "Records one explicitly observed candidate outcome, attempt, prompt ID, cache "
                "observation and artifact metadata. It never probes or mutates the queue/cache."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.Int.Input("run_position", default=0, min=0, max=100000),
                io.Int.Input("variant_index", default=0, min=0, max=63),
                io.Int.Input("attempt_number", default=1, min=1, max=100000),
                io.Combo.Input("outcome", options=list(OUTCOMES), default="completed"),
                io.String.Input("prompt_id", default=""),
                io.Combo.Input(
                    "cache_observation",
                    options=list(CACHE_OBSERVATIONS),
                    default="unknown",
                ),
                io.String.Input(
                    "artifact_manifest_json",
                    multiline=True,
                    default='{"video":{"path":"output/candidate.mp4"}}',
                ),
                io.String.Input("notes", multiline=True, default=""),
                CreatorRunLedgerIO.Input("previous_ledger", optional=True),
            ],
            outputs=[
                CreatorRunLedgerIO.Output("ledger"),
                io.String.Output("event_json"),
                io.String.Output("ledger_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*record_creator_run_receipt(**kwargs))


class MiniMaxH3CreatorResumePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorResumePlanT8Advanced",
            display_name=(
                "MiniMax H3 Creator Resume Plan / 下一候选与继续位置 (Advanced/T8)"
            ),
            description=(
                "Finds the next explicit render, review or retry from an immutable Creator "
                "Workspace and optional receipt ledger. It never queues, cancels or deletes."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                CreatorRunLedgerIO.Input("ledger", optional=True),
            ],
            outputs=[
                CreatorResumePlanIO.Output("resume_plan"),
                io.String.Output("action"),
                io.Int.Output("run_position"),
                io.Int.Output("variant_index"),
                io.Int.Output("attempt_number"),
                io.String.Output("summary"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, workspace, ledger=None):
        return io.NodeOutput(*compile_creator_resume_plan(workspace, ledger))

    @classmethod
    def fingerprint_inputs(cls, workspace, ledger=None):
        workspace_hash = (
            workspace.get("workspace_hash") if isinstance(workspace, dict) else "unresolved"
        )
        ledger_hash = ledger.get("ledger_hash") if isinstance(ledger, dict) else "empty"
        return f"{workspace_hash}:{ledger_hash}"


class MiniMaxH3CreatorRetentionPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorRetentionPlanT8Advanced",
            display_name=(
                "MiniMax H3 Creator Artifact Retention Plan / "
                "候选保留与拟删除清单 (Advanced/T8)"
            ),
            description=(
                "Compiles each shot's retention policy against explicit run receipts. It emits "
                "reviewable keep/proposed-delete manifests and never touches the filesystem."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.Boolean.Input("confirm_artifact_paths_reviewed", default=False),
                CreatorRunLedgerIO.Input("ledger", optional=True),
            ],
            outputs=[
                CreatorRetentionPlanIO.Output("retention_plan"),
                io.String.Output("status"),
                io.String.Output("keep_manifest_json"),
                io.String.Output("proposed_delete_manifest_json"),
                io.String.Output("summary"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, workspace, confirm_artifact_paths_reviewed, ledger=None):
        return io.NodeOutput(
            *compile_creator_retention_plan(
                workspace,
                ledger,
                confirm_artifact_paths_reviewed,
            )
        )

    @classmethod
    def fingerprint_inputs(cls, workspace, confirm_artifact_paths_reviewed, ledger=None):
        workspace_hash = (
            workspace.get("workspace_hash") if isinstance(workspace, dict) else "unresolved"
        )
        ledger_hash = ledger.get("ledger_hash") if isinstance(ledger, dict) else "empty"
        return f"{workspace_hash}:{ledger_hash}:{bool(confirm_artifact_paths_reviewed)}"


class MiniMaxH3CreatorBackgroundStartT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorBackgroundStartT8Advanced",
            display_name=(
                "MiniMax H3 Creator × Long Video Background Start / "
                "创作工作区后台绑定 (Advanced/T8)"
            ),
            description=(
                "Binds one Creator Workspace hash to the existing proven Long Video background "
                "queue/cancel/retry controller. Use only with the Long Video candidate/Auto "
                "Accept terminal; review_only remains non-mutating."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.String.Input("chain_id", default="creator_long_video"),
                io.Combo.Input(
                    "execution_mode",
                    options=list(EXECUTION_MODES),
                    default="review_only",
                ),
                io.Int.Input("max_retries", default=1, min=0, max=10),
                io.Float.Input(
                    "retry_delay_seconds", default=2.0, min=0.0, max=300.0, step=0.1
                ),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_execution_cache",
                ),
            ],
            outputs=[
                CreatorWorkspaceIO.Output("workspace"),
                io.String.Output("chain_id"),
                io.Boolean.Output("auto_accept"),
                io.String.Output("job_id"),
                io.String.Output("background_state_json"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        workspace,
        chain_id,
        execution_mode,
        max_retries,
        retry_delay_seconds,
        release_policy,
    ):
        binding = creator_background_binding(workspace)
        if execution_mode == "review_only":
            state = {
                "schema": BACKGROUND_SCHEMA,
                "format": BACKGROUND_STATE_FORMAT,
                "chain_id": str(chain_id),
                "state": "review_only",
                "accepted_count": 0,
                "retry_count": 0,
                "max_retries": int(max_retries),
                "binding_metadata": binding,
                "queue_mutated": False,
                "message": (
                    "No queue, cancellation, retry or release is active until "
                    "auto_accept_and_continue is explicitly selected."
                ),
            }
            return io.NodeOutput(
                workspace,
                str(chain_id),
                False,
                "",
                json.dumps(state, ensure_ascii=False, indent=2),
            )
        current = BACKGROUND_JOBS.status(chain_id)
        if int(current.get("accepted_count", 0)) > 0 and (
            current.get("binding_metadata") != binding
        ):
            raise ValueError(
                "This chain already has accepted segments without the same Creator Workspace "
                "binding; use a new chain_id"
            )
        if current.get("manifest_complete") or current.get("state") == "completed":
            values = (
                workspace,
                str(chain_id),
                True,
                str(current.get("job_id") or ""),
                json.dumps(current, ensure_ascii=False, indent=2),
            )
            return io.NodeOutput(
                *values,
                block_execution=f"Creator background chain '{chain_id}' is already complete.",
            )
        state = BACKGROUND_JOBS.attach_prompt(
            chain_id,
            cls.hidden.prompt,
            str(cls.hidden.unique_id or ""),
            max_retries,
            retry_delay_seconds,
            release_policy,
            binding_metadata=binding,
        )
        return io.NodeOutput(
            workspace,
            state["chain_id"],
            True,
            state["job_id"],
            json.dumps(state, ensure_ascii=False, indent=2),
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3CreatorBackgroundRunSelectT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CreatorBackgroundRunSelectT8Advanced",
            display_name=(
                "MiniMax H3 Creator × Long Video Run Select / "
                "按后台进度选择镜头 (Advanced/T8)"
            ),
            description=(
                "Uses the bound Long Video accepted_count and retry_count to choose the next "
                "Creator shot and deterministic seed. It never owns the queue; paused, failed, "
                "cancelled, detached or review-only states block downstream generation."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                CreatorWorkspaceIO.Input("workspace"),
                io.String.Input("background_state_json", force_input=True),
                io.Combo.Input(
                    "variant_policy",
                    options=["retry_as_variant_clamped", "fixed_first"],
                    default="retry_as_variant_clamped",
                ),
            ],
            outputs=[
                PromptPacketIO.Output("prompt_packet"),
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.Int.Output("length"),
                io.Int.Output("seed"),
                io.Int.Output("run_position"),
                io.Int.Output("variant_index"),
                io.Int.Output("attempt_number"),
                io.Boolean.Output("ready"),
                io.String.Output("action"),
                io.String.Output("shot_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, workspace, background_state_json, variant_policy):
        plan, selected, report = compile_creator_background_selection(
            workspace, background_state_json, variant_policy
        )
        packet, prompt, negative, length, seed, shot_json, _selection_report = selected
        values = (
            packet,
            prompt,
            negative,
            length,
            seed,
            plan["run_position"],
            plan["variant_index"],
            plan["attempt_number"],
            plan["ready"],
            plan["action"],
            shot_json,
            report,
        )
        if plan["ready"]:
            return io.NodeOutput(*values)
        return io.NodeOutput(
            *values,
            block_execution=(
                f"Creator background is {plan['background_state']}: {plan['action']}. "
                "Use the existing Long Video status/pause/resume/cancel controls."
            ),
        )


CREATOR_RUNTIME_ADVANCED_NODE_CLASSES = [
    MiniMaxH3CreatorRunReceiptT8Advanced,
    MiniMaxH3CreatorResumePlanT8Advanced,
    MiniMaxH3CreatorBackgroundStartT8Advanced,
    MiniMaxH3CreatorBackgroundRunSelectT8Advanced,
]

CREATOR_RETENTION_ADVANCED_NODE_CLASSES = [
    MiniMaxH3CreatorRetentionPlanT8Advanced,
]
