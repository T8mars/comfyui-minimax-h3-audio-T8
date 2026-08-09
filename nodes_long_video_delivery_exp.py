from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .long_video import CONTEXT_TYPE_NAME
from .long_video_background import (
    BACKGROUND_JOBS,
    BACKGROUND_SCHEMA,
    EXECUTION_MODES,
    RELEASE_POLICIES,
)
from .long_video_delivery import (
    accept_long_video_candidate,
    compose_accepted_long_video,
    load_accepted_context,
    manifest_fingerprint,
    load_delivery_manifest,
    load_long_video_candidate_descriptor,
    save_long_video_candidate,
)
from .long_video_orchestration import resolve_long_video_orchestration
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
LongVideoContext = io.Custom(CONTEXT_TYPE_NAME)


def _preview_video(path_value: str):
    path = Path(path_value).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root not in path.parents:
        raise ValueError("Long-video preview is not inside the ComfyUI output directory")
    relative = path.relative_to(output_root)
    saved = ui.SavedResult(relative.name, relative.parent.as_posix(), io.FolderType.output)
    return InputImpl.VideoFromFile(str(path)), ui.PreviewVideo([saved])


class MiniMaxH3LongVideoOrchestratorT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoOrchestratorT8",
            display_name="MiniMax H3 Chain Orchestrator / 总时长自动分段 (EXP/T8)",
            description=(
                "Converts a total duration into a fixed-window H3 timeline and resumes at the "
                "first unaccepted manifest segment. A completed final manifest blocks downstream "
                "sampling instead of generating an accidental extra segment."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.Float.Input(
                    "total_duration_seconds",
                    default=60.0,
                    min=0.04,
                    max=3600.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "render_window_frames",
                    default=124,
                    min=124,
                    max=362,
                    step=17,
                    tooltip=(
                        "Fixed internal H3 window for every segment. Keep 124 for the current "
                        "bounded-memory baseline; larger windows require separate VRAM validation."
                    ),
                ),
                io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
                io.String.Input("global_prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "segment_prompts_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Optional list/object. Each item may contain prompt, seed and note. "
                        "Unspecified segments use global_prompt and the selected seed policy."
                    ),
                ),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input(
                    "seed_policy",
                    options=["increment", "fixed", "hash_chain_segment"],
                    default="increment",
                ),
                io.Int.Input("steps", default=4, min=1, max=1000),
                io.Float.Input(
                    "shift_video", default=12.0, min=0.01, max=100.0, step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio", default=3.0, min=0.01, max=100.0, step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                    display_name="sampler / 采样器",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                    display_name="scheduler / 调度器",
                ),
            ],
            outputs=[
                io.String.Output("chain_id"),
                io.Int.Output("segment_index"),
                io.Int.Output("length"),
                io.Int.Output("context_frames"),
                io.Float.Output("trim_start_seconds"),
                io.Float.Output("final_duration_seconds"),
                io.Float.Output("timeline_start_seconds"),
                io.Float.Output("timeline_end_seconds"),
                io.Boolean.Output("save_context"),
                io.Boolean.Output("is_final_segment"),
                io.String.Output("prompt"),
                io.Int.Output("seed"),
                io.Boolean.Output("has_next"),
                io.Float.Output("progress"),
                io.String.Output("plan_json"),
                io.String.Output("report_json"),
                io.Int.Output("steps"),
                io.Float.Output("shift_video"),
                io.Float.Output("shift_audio"),
                io.Combo.Output("sampler_name"),
                io.Combo.Output("scheduler"),
                io.String.Output("sampling_summary"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        chain_id,
        total_duration_seconds,
        render_window_frames,
        context_frames,
        global_prompt,
        segment_prompts_json,
        base_seed,
        seed_policy,
        steps=4,
        shift_video=12.0,
        shift_audio=3.0,
        sampler_name=DEFAULT_SAMPLER_NAME,
        scheduler=DEFAULT_SCHEDULER_NAME,
    ):
        result, manifest = resolve_long_video_orchestration(
            chain_id,
            total_duration_seconds,
            render_window_frames,
            context_frames,
            global_prompt,
            segment_prompts_json,
            base_seed,
            seed_policy,
            steps,
            shift_video,
            shift_audio,
            sampler_name,
            scheduler,
        )
        segment = result.next_segment or result.segments[-1]
        plan = segment.plan
        values = (
            result.chain_id,
            segment.index,
            plan.render_frames,
            plan.context_frames,
            plan.trim_start_seconds,
            plan.final_duration_seconds,
            plan.timeline_start_seconds,
            plan.timeline_end_seconds,
            plan.save_context,
            plan.is_final_segment,
            segment.prompt,
            segment.seed,
            not result.complete,
            result.progress,
            result.plan_json(manifest),
            result.report(),
            result.steps,
            result.shift_video,
            result.shift_audio,
            result.sampler_name,
            result.scheduler,
            result.sampling_summary,
        )
        if result.complete:
            return io.NodeOutput(
                *values,
                block_execution=(
                    f"MiniMax H3 chain '{result.chain_id}' is complete: "
                    f"{len(result.segments)} accepted segment(s), "
                    f"{result.quantized_total_duration_seconds:.3f}s."
                ),
            )
        return io.NodeOutput(*values)

    @classmethod
    def fingerprint_inputs(cls, chain_id, **_kwargs):
        try:
            return manifest_fingerprint(chain_id)
        except (TypeError, ValueError):
            return f"unresolved:{chain_id!r}"


class MiniMaxH3LongVideoCandidateSaveT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoCandidateSaveT8",
            display_name="MiniMax H3 Save Candidate / 保存候选片段 (EXP/T8)",
            description=(
                "Atomically writes only the current trimmed segment plus its bounded AV tail. "
                "It does not change accepted history; connect the descriptor to Review & Accept."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                io.Audio.Input("audio"),
                io.Latent.Input("av_latent"),
                io.String.Input("chain_id", default="my_h3_long_video", force_input=True),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
                io.Float.Input(
                    "timeline_start_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                    force_input=True,
                ),
                io.Boolean.Input("save_context", default=True, force_input=True),
                io.String.Input("parent_candidate_id", default="", force_input=True),
                io.Int.Input(
                    "parent_manifest_revision", default=0, min=0, max=999999, force_input=True
                ),
                io.String.Input(
                    "candidate_id",
                    default="",
                    advanced=True,
                    tooltip="Blank creates a unique id. Set a new id for every intentional re-roll.",
                ),
                io.String.Input("model_id", default="unknown", advanced=True),
                io.String.Input(
                    "sampling_summary",
                    default="dual_clock_euler/native_flow",
                    advanced=True,
                ),
                io.String.Input("prompt", default="", multiline=True, force_input=True),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, force_input=True
                ),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
            ],
            outputs=[
                io.String.Output("candidate_json_path"),
                io.String.Output("candidate_video_path"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        frames,
        audio,
        av_latent,
        chain_id,
        segment_index,
        timeline_start_seconds,
        save_context,
        parent_candidate_id,
        parent_manifest_revision,
        candidate_id,
        model_id,
        sampling_summary,
        prompt,
        seed,
        bit_depth,
        crf,
    ):
        return io.NodeOutput(*save_long_video_candidate(
            frames,
            audio,
            av_latent,
            chain_id,
            segment_index,
            timeline_start_seconds,
            save_context,
            parent_candidate_id,
            parent_manifest_revision,
            candidate_id,
            model_id,
            sampling_summary,
            prompt,
            seed,
            24,
            bit_depth,
            crf,
        ))


class MiniMaxH3LongVideoAcceptCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoAcceptCandidateT8",
            display_name="MiniMax H3 Review & Accept / 预览并接受候选 (EXP/T8)",
            description=(
                "Preview with accept_candidate=false. When true, atomically promotes the candidate "
                "and updates the manifest. Replacing a segment invalidates all dependent later segments."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("candidate_json_path", default="", force_input=True),
                io.Boolean.Input("accept_candidate", default=False),
                io.Combo.Input(
                    "replace_policy",
                    options=["reject_existing", "replace_and_invalidate_following"],
                    default="reject_existing",
                    advanced=True,
                ),
                io.Boolean.Input("strict_chain_identity", default=True, advanced=True),
            ],
            outputs=[
                io.Video.Output("video"),
                io.Boolean.Output("accepted"),
                io.String.Output("manifest_path"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        candidate_json_path,
        accept_candidate,
        replace_policy,
        strict_chain_identity,
    ):
        video_path, accepted, manifest_path, report = accept_long_video_candidate(
            candidate_json_path,
            accept_candidate,
            replace_policy,
            strict_chain_identity,
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(video, accepted, manifest_path, report, ui=preview)


class MiniMaxH3LongVideoAcceptedContextLoadT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoAcceptedContextLoadT8",
            display_name="MiniMax H3 Accepted Context / 读取已接受上下文 (EXP/T8)",
            description=(
                "Loads segment N-1 only from the accepted manifest. It also returns the exact parent "
                "candidate identity needed to prevent accepting a stale continuation."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video", force_input=True),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
            ],
            outputs=[
                LongVideoContext.Output("context"),
                io.Boolean.Output("has_context"),
                io.String.Output("accepted_candidate_id"),
                io.Int.Output("manifest_revision"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, chain_id, segment_index):
        return io.NodeOutput(*load_accepted_context(chain_id, segment_index))

    @classmethod
    def fingerprint_inputs(cls, chain_id, segment_index):
        try:
            return manifest_fingerprint(chain_id, segment_index)
        except (TypeError, ValueError):
            return f"unresolved:{chain_id!r}:{segment_index!r}"


class MiniMaxH3LongVideoComposeAcceptedT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoComposeAcceptedT8",
            display_name="MiniMax H3 Compose Accepted / 合成已接受片段 (EXP/T8)",
            description=(
                "Verifies every accepted file and streams them into one MP4. Memory is bounded to one "
                "video frame plus one segment of PCM; cosine_bridge preserves the exact sample count."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.String.Input("filename_prefix", default="H3_Long_Video"),
                io.Boolean.Input("require_final_segment", default=True),
                io.Combo.Input(
                    "audio_seam_policy",
                    options=["cosine_bridge", "none"],
                    default="cosine_bridge",
                ),
                io.Float.Input(
                    "bridge_ms", default=5.0, min=0.0, max=50.0, step=0.1, advanced=True
                ),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        chain_id,
        filename_prefix,
        require_final_segment,
        audio_seam_policy,
        bridge_ms,
        crf,
    ):
        video_path, report = compose_accepted_long_video(
            chain_id,
            filename_prefix,
            require_final_segment,
            audio_seam_policy,
            bridge_ms,
            crf,
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(video, video_path, report, ui=preview)


class MiniMaxH3LongVideoBackgroundStartT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoBackgroundStartT8",
            display_name="MiniMax H3 Background Start / 后台长视频启动 (EXP/T8)",
            description=(
                "Registers the current prompt before model execution so upstream OOM/errors can "
                "be retried. review_only is non-mutating; auto_accept_and_continue must be selected "
                "explicitly. Connect chain_id and controller outputs to the background workflow."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.Combo.Input(
                    "execution_mode",
                    options=list(EXECUTION_MODES),
                    default="review_only",
                    tooltip=(
                        "auto_accept_and_continue accepts every successfully rendered candidate "
                        "without human review and queues the next segment."
                    ),
                ),
                io.Int.Input(
                    "max_retries",
                    default=1,
                    min=0,
                    max=10,
                    tooltip="Additional attempts per failed segment; resolution is never reduced.",
                ),
                io.Float.Input(
                    "retry_delay_seconds", default=2.0, min=0.0, max=300.0, step=0.1
                ),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_execution_cache",
                    tooltip=(
                        "Applied after every accepted segment, including pause and final. "
                        "clear_execution_cache resets ComfyUI execution tensors and soft cache. "
                        "unload_all_models is stronger but globally unloads every ComfyUI model, "
                        "not only H3."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("chain_id"),
                io.Boolean.Output("auto_accept"),
                io.String.Output("job_id"),
                io.String.Output("background_state_json"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        chain_id,
        execution_mode,
        max_retries,
        retry_delay_seconds,
        release_policy,
    ):
        if execution_mode == "review_only":
            status = {
                "schema": BACKGROUND_SCHEMA,
                "chain_id": str(chain_id),
                "state": "review_only",
                "message": "No automatic acceptance, queueing, retry, or release is active.",
            }
            return io.NodeOutput(
                str(chain_id), False, "", json.dumps(status, ensure_ascii=False, indent=2)
            )
        current = BACKGROUND_JOBS.status(chain_id)
        if current.get("manifest_complete") or current.get("state") == "completed":
            values = (
                str(chain_id),
                True,
                str(current.get("job_id") or ""),
                json.dumps(current, ensure_ascii=False, indent=2),
            )
            return io.NodeOutput(
                *values,
                block_execution=f"MiniMax H3 chain '{chain_id}' is already complete.",
            )
        state = BACKGROUND_JOBS.attach_prompt(
            chain_id,
            cls.hidden.prompt,
            str(cls.hidden.unique_id or ""),
            max_retries,
            retry_delay_seconds,
            release_policy,
        )
        return io.NodeOutput(
            state["chain_id"],
            True,
            state["job_id"],
            json.dumps(state, ensure_ascii=False, indent=2),
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        # This node must attach every queued prompt before any expensive upstream work.
        return float("nan")


class MiniMaxH3LongVideoAutoQueueT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoAutoQueueT8",
            display_name="MiniMax H3 Auto Accept & Continue / 自动接受续跑 (EXP/T8)",
            description=(
                "Terminal node for the explicit background workflow. It accepts the current "
                "candidate, optionally composes a final MP4, requests the selected release policy "
                "even at pause/final boundaries, "
                "and queues exactly one next prompt. review_only remains non-mutating."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("candidate_json_path", default="", force_input=True),
                io.String.Input("job_id", default="", force_input=True),
                io.Boolean.Input("auto_accept", default=False, force_input=True),
                io.Combo.Input(
                    "replace_policy",
                    options=["reject_existing", "replace_and_invalidate_following"],
                    default="reject_existing",
                    advanced=True,
                ),
                io.Boolean.Input("strict_chain_identity", default=True, advanced=True),
                io.Boolean.Input("compose_when_complete", default=True),
                io.String.Input("filename_prefix", default="H3_Long_Video"),
                io.Combo.Input(
                    "audio_seam_policy",
                    options=["cosine_bridge", "none"],
                    default="cosine_bridge",
                ),
                io.Float.Input(
                    "bridge_ms", default=5.0, min=0.0, max=50.0, step=0.1, advanced=True
                ),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
            ],
            outputs=[
                io.Video.Output("video"),
                io.Boolean.Output("accepted"),
                io.String.Output("manifest_path"),
                io.String.Output("final_video_path"),
                io.String.Output("background_state_json"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        candidate_json_path,
        job_id,
        auto_accept,
        replace_policy,
        strict_chain_identity,
        compose_when_complete,
        filename_prefix,
        audio_seam_policy,
        bridge_ms,
        crf,
    ):
        candidate, candidate_video = load_long_video_candidate_descriptor(candidate_json_path)
        if not auto_accept:
            video_path, accepted, manifest_path, accept_report = accept_long_video_candidate(
                candidate_json_path, False, replace_policy, strict_chain_identity
            )
            status = {
                "schema": BACKGROUND_SCHEMA,
                "chain_id": candidate["chain_id"],
                "state": "review_only",
            }
            video, preview = _preview_video(video_path)
            return io.NodeOutput(
                video,
                accepted,
                manifest_path,
                "",
                json.dumps(status, ensure_ascii=False, indent=2),
                accept_report,
                ui=preview,
            )

        BACKGROUND_JOBS.assert_accept_allowed(job_id)
        video_path, accepted, manifest_path, accept_report = accept_long_video_candidate(
            candidate_json_path, True, replace_policy, strict_chain_identity
        )
        index = int(candidate["index"])
        manifest = None
        manifest_source = "unknown"
        is_final = bool(candidate.get("is_final_segment"))
        final_video_path = ""
        compose_report = ""
        post_accept_error = ""
        try:
            manifest, manifest_source = load_delivery_manifest(candidate["chain_id"])
            if index >= len(manifest["segments"]):
                raise RuntimeError("Accepted manifest did not contain the promoted candidate")
            accepted_entry = manifest["segments"][index]
            if accepted_entry.get("candidate_id") != candidate.get("candidate_id"):
                raise RuntimeError("Accepted manifest candidate identity changed unexpectedly")
            is_final = bool(accepted_entry.get("is_final_segment"))
            if is_final and compose_when_complete:
                final_video_path, compose_report = compose_accepted_long_video(
                    candidate["chain_id"],
                    filename_prefix,
                    True,
                    audio_seam_policy,
                    bridge_ms,
                    crf,
                )
            state = BACKGROUND_JOBS.segment_accepted(
                job_id,
                candidate_index=index,
                candidate_json_path=candidate_json_path,
                manifest_path=manifest_path,
                accepted_count=len(manifest["segments"]),
                is_final_segment=is_final,
                final_video_path=final_video_path,
            )
        except Exception as error:
            # Acceptance is already durable. Do not let the prompt monitor retry and silently
            # advance from the new manifest. Keep the job failed for explicit operator action.
            post_accept_error = f"Post-accept processing failed: {error}"
            state = BACKGROUND_JOBS.fail_job(job_id, post_accept_error)
        report = {
            "schema": BACKGROUND_SCHEMA,
            "candidate": candidate,
            "accept": json.loads(accept_report),
            "manifest_source": manifest_source,
            "manifest_revision": manifest["revision"] if manifest is not None else None,
            "is_final_segment": is_final,
            "compose": json.loads(compose_report) if compose_report else None,
            "post_accept_error": post_accept_error,
            "background": state,
        }
        display_path = final_video_path or video_path or candidate_video
        video, preview = _preview_video(display_path)
        return io.NodeOutput(
            video,
            accepted,
            manifest_path,
            final_video_path,
            json.dumps(state, ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=preview,
        )
