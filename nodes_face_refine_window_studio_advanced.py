from __future__ import annotations

import json

from comfy_api.latest import io

from .face_refine_window_studio_advanced import (
    STUDIO_SCHEMA,
    commit_face_refine_window_studio,
    compose_face_refine_window_studio,
    prepare_face_refine_window_studio,
)
from .long_video_background import BACKGROUND_JOBS, RELEASE_POLICIES
from .nodes_face_refine_window_advanced import (
    CATEGORY,
    FaceRefineWindowMappingIO,
    FaceRefineWindowPlanIO,
)


class MiniMaxH3FaceRefineWindowStudioStartT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineWindowStudioStartT8Advanced",
            display_name=(
                "MiniMax H3 Face Refine Window Studio Start / 多窗口串行启动 (Advanced)"
            ),
            description=(
                "Creates or resumes a source-bound multi-window manifest. review_only never "
                "queues work. Explicit accept-and-continue reuses the Long Video OS process "
                "lease, retry and cancellation controller and may queue only one next window."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                FaceRefineWindowPlanIO.Input("window_plan"),
                io.String.Input("studio_id", default="face_refine_project_01"),
                io.Combo.Input(
                    "execution_mode",
                    options=["review_only", "explicit_accept_and_continue"],
                    default="review_only",
                    tooltip=(
                        "Preview in review_only first. The second mode does not accept by "
                        "itself; Commit still requires an explicit accept/reject decision."
                    ),
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
                io.Int.Output("window_index"),
                io.String.Output("chain_id"),
                io.Boolean.Output("auto_continue"),
                io.String.Output("job_id"),
                io.String.Output("manifest_path"),
                io.Boolean.Output("complete"),
                io.String.Output("background_state_json"),
                io.String.Output("report_json"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        window_plan,
        studio_id,
        execution_mode,
        max_retries,
        retry_delay_seconds,
        release_policy,
    ):
        chain_id, index, path, complete, _manifest, report = (
            prepare_face_refine_window_studio(window_plan, studio_id)
        )
        if complete:
            status = BACKGROUND_JOBS.status(chain_id)
            return io.NodeOutput(
                index,
                chain_id,
                False,
                str(status.get("job_id") or ""),
                path,
                True,
                json.dumps(status, ensure_ascii=False, indent=2),
                report,
                block_execution=(
                    "All Face Refine windows are resolved. Run Studio Compose; no window will "
                    "be generated again."
                ),
            )
        if execution_mode == "review_only":
            status = {
                "schema": STUDIO_SCHEMA,
                "chain_id": chain_id,
                "state": "review_only",
                "current_window_index": index,
                "message": "No queue, retry or automatic acceptance is active.",
            }
            return io.NodeOutput(
                index,
                chain_id,
                False,
                "",
                path,
                False,
                json.dumps(status, ensure_ascii=False, indent=2),
                report,
            )
        state = BACKGROUND_JOBS.attach_prompt(
            chain_id,
            cls.hidden.prompt,
            str(cls.hidden.unique_id or ""),
            max_retries,
            retry_delay_seconds,
            release_policy,
            binding_metadata={
                "kind": STUDIO_SCHEMA,
                "studio_id": str(studio_id),
                "window_plan_sha256": window_plan["plan_sha256"],
            },
        )
        return io.NodeOutput(
            index,
            chain_id,
            True,
            state["job_id"],
            path,
            False,
            json.dumps(state, ensure_ascii=False, indent=2),
            report,
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3FaceRefineWindowStudioCommitT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineWindowStudioCommitT8Advanced",
            display_name=(
                "MiniMax H3 Face Refine Window Studio Commit / 人工决策提交 (Advanced)"
            ),
            description=(
                "Validates the current source-bound candidate again, then atomically records "
                "an explicit accept or reject. Accepted overlays are immutable and never "
                "overwrite source media. Only after that durable boundary may one next prompt "
                "be queued."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("candidate_window_frames"),
                io.Mask.Input("changed_mask"),
                FaceRefineWindowMappingIO.Input("window_mapping"),
                FaceRefineWindowPlanIO.Input("window_plan"),
                io.String.Input("studio_id", default="face_refine_project_01"),
                io.Combo.Input(
                    "decision",
                    options=["preview_only", "reject", "accept_selected"],
                    default="preview_only",
                ),
                io.String.Input("accepted_subranges", default="", multiline=False),
                io.Boolean.Input("confirm_accept", default=False),
                io.Int.Input("edge_fade_frames", default=2, min=0, max=60),
                io.String.Input("job_id", default="", force_input=True),
                io.Boolean.Input("auto_continue", default=False, force_input=True),
            ],
            outputs=[
                io.Image.Output("review_frames"),
                io.Image.Output("current_result_frames"),
                io.Mask.Output("accepted_change_mask"),
                io.Mask.Output("rejected_change_mask"),
                io.Boolean.Output("committed"),
                io.String.Output("manifest_path"),
                io.Int.Output("resolved_window_count"),
                io.Boolean.Output("complete"),
                io.String.Output("background_state_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_frames,
        candidate_window_frames,
        changed_mask,
        window_mapping,
        window_plan,
        studio_id,
        decision,
        accepted_subranges,
        confirm_accept,
        edge_fade_frames,
        job_id,
        auto_continue,
    ):
        if auto_continue and decision == "preview_only":
            raise ValueError(
                "Auto-continue requires an explicit accept_selected or reject decision. "
                "Run review_only first to inspect the candidate."
            )
        if auto_continue:
            BACKGROUND_JOBS.assert_accept_allowed(job_id)
        result = commit_face_refine_window_studio(
            base_frames,
            candidate_window_frames,
            changed_mask,
            window_mapping,
            window_plan,
            studio_id,
            decision,
            accepted_subranges,
            confirm_accept,
            edge_fade_frames,
        )
        state = {
            "schema": STUDIO_SCHEMA,
            "state": "review_only",
            "message": "No next prompt was queued.",
        }
        if auto_continue and result[4]:
            report = json.loads(result[-1])
            try:
                state = BACKGROUND_JOBS.segment_accepted(
                    job_id,
                    candidate_index=int(report["window_index"]),
                    candidate_json_path=result[5],
                    manifest_path=result[5],
                    accepted_count=int(result[6]),
                    is_final_segment=bool(result[7]),
                )
            except Exception as error:
                # The window decision is already durable. Do not retry or undo it; a fresh
                # controller run resumes from the next unresolved manifest entry.
                state = BACKGROUND_JOBS.fail_job(
                    job_id, f"Post-commit background transition failed: {error}"
                )
        return io.NodeOutput(
            *result[:-1],
            json.dumps(state, ensure_ascii=False, indent=2),
            result[-1],
        )


class MiniMaxH3FaceRefineWindowStudioComposeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineWindowStudioComposeT8Advanced",
            display_name=(
                "MiniMax H3 Face Refine Window Studio Compose / 已接受窗口合成 (Advanced)"
            ),
            description=(
                "Rebuilds the full IMAGE timeline from immutable source frames plus accepted "
                "Studio overlays. Rejected and pending windows remain exact source. Connect the "
                "original complete AUDIO separately when saving the final video."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("base_frames"),
                FaceRefineWindowPlanIO.Input("window_plan"),
                io.String.Input("studio_id", default="face_refine_project_01"),
                io.String.Input(
                    "commit_barrier",
                    default="",
                    optional=True,
                    force_input=True,
                    tooltip="Optional Commit report link used only to enforce graph order.",
                ),
            ],
            outputs=[
                io.Image.Output("result_frames"),
                io.Mask.Output("combined_accepted_mask"),
                io.Boolean.Output("complete"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, base_frames, window_plan, studio_id, commit_barrier=""):
        del commit_barrier
        return io.NodeOutput(
            *compose_face_refine_window_studio(base_frames, window_plan, studio_id)
        )


FACE_REFINE_WINDOW_STUDIO_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceRefineWindowStudioStartT8Advanced,
    MiniMaxH3FaceRefineWindowStudioCommitT8Advanced,
    MiniMaxH3FaceRefineWindowStudioComposeT8Advanced,
]
