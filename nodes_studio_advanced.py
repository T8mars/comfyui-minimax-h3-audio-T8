from __future__ import annotations

from comfy_api.latest import io

from .studio_advanced import (
    BACKENDS,
    build_selective_repair_plan,
    build_sound_canvas,
    build_studio_timeline,
    build_unified_cast,
    canonical_json,
    compile_prompt_packet,
    select_repair_segment,
    select_studio_shot,
)


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
CastIO = io.Custom("H3_T8_UNIFIED_CAST")
SoundCanvasIO = io.Custom("H3_T8_SOUND_CANVAS")
PromptPacketIO = io.Custom("H3_T8_PROMPT_PACKET")
StudioTimelineIO = io.Custom("H3_T8_STUDIO_TIMELINE")
RepairPlanIO = io.Custom("H3_T8_REPAIR_PLAN")


class MiniMaxH3UnifiedCastT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UnifiedCastT8Advanced",
            display_name="MiniMax H3 Unified Cast / 统一角色表 (Advanced)",
            description=(
                "Builds one validated character-continuity contract for video prompts. "
                "It stores text metadata only and does not load a face, voice, or vision model."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "cast_json",
                    multiline=True,
                    default='[{"id":"hero","name":"Hero","visual_identity":"consistent facial identity and hairstyle"}]',
                ),
                io.Boolean.Input("strict_identity", default=True),
            ],
            outputs=[CastIO.Output("cast"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, cast_json, strict_identity):
        result = build_unified_cast(cast_json, strict_identity)
        return io.NodeOutput(result, canonical_json(result))


class MiniMaxH3SoundCanvasT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SoundCanvasT8Advanced",
            display_name="MiniMax H3 Sound Canvas / 声音画布 (Advanced)",
            description=(
                "Plans dialogue, music, ambience and SFX on one explicit timeline. "
                "The no-extra-speech guard preserves non-speech audio instead of trimming the master."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "events_json",
                    multiline=True,
                    default='[{"id":"room","role":"ambience","start_seconds":0,"end_seconds":5.167,"description":"quiet room tone"}]',
                ),
                io.Float.Input("total_duration_seconds", default=5.167, min=0.001, max=86400.0, step=0.001),
                io.Boolean.Input("no_unrequested_speech", default=True),
                io.Boolean.Input("allow_dialogue_overlap", default=False, advanced=True),
            ],
            outputs=[
                SoundCanvasIO.Output("sound_canvas"),
                io.String.Output("h3_audio_prompt"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, events_json, total_duration_seconds, no_unrequested_speech, allow_dialogue_overlap):
        result = build_sound_canvas(
            events_json,
            total_duration_seconds,
            no_unrequested_speech,
            allow_dialogue_overlap,
        )
        return io.NodeOutput(result, result["h3_audio_prompt"], canonical_json(result))


class MiniMaxH3PromptCompilerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptCompilerT8Advanced",
            display_name="T8 Video Prompt Compiler / 多后端提示词编译 (Advanced)",
            description=(
                "Compiles one structured visual/audio brief for MiniMax H3, Wan 2.2, "
                "LTX-Video or a generic cinematic backend. This is text compilation only."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("backend", options=list(BACKENDS), default="minimax_h3"),
                io.Float.Input("duration_seconds", default=5.167, min=0.001, max=86400.0, step=0.001),
                io.String.Input("aspect_ratio", default="16:9"),
                io.String.Input("dialogue", multiline=True, default="", optional=True),
                io.String.Input("negative_prompt", multiline=True, default="", optional=True),
                io.Boolean.Input("strict_exact_dialogue", default=True),
                CastIO.Input("cast", optional=True),
                SoundCanvasIO.Input("sound_canvas", optional=True),
                io.String.Input("cast_ids", default="", optional=True, advanced=True),
                io.String.Input("creative_brief_json", multiline=True, default="{}", optional=True, advanced=True),
            ],
            outputs=[
                PromptPacketIO.Output("prompt_packet"),
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.String.Output("audio_prompt"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        backend,
        duration_seconds,
        aspect_ratio,
        dialogue="",
        negative_prompt="",
        strict_exact_dialogue=True,
        cast=None,
        sound_canvas=None,
        cast_ids="",
        creative_brief_json="{}",
    ):
        import json

        try:
            brief = json.loads(creative_brief_json or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"creative_brief_json is invalid JSON: {error}") from error
        ids = [value.strip() for value in str(cast_ids).split(",") if value.strip()]
        packet = compile_prompt_packet(
            prompt,
            backend,
            duration_seconds,
            aspect_ratio,
            dialogue,
            negative_prompt,
            strict_exact_dialogue,
            cast,
            ids,
            sound_canvas,
            brief,
        )
        return io.NodeOutput(
            packet,
            packet["compiled_prompt"],
            packet["negative_prompt"],
            packet["audio_prompt"],
            canonical_json(packet),
        )


class MiniMaxH3StudioTimelineT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StudioTimelineT8Advanced",
            display_name="MiniMax H3 Studio Timeline / 创作时间轴 (Advanced)",
            description=(
                "Compiles many shots into deterministic H3 17n+5 windows. Long visual-only "
                "shots can be split explicitly; long dialogue must be split by the author."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input("project_id", default="h3_project"),
                io.String.Input(
                    "shots_json",
                    multiline=True,
                    default='[{"id":"shot_1","prompt":"A cinematic opening shot","duration_seconds":5.0}]',
                ),
                io.Combo.Input("default_backend", options=list(BACKENDS), default="minimax_h3"),
                io.Float.Input("default_duration_seconds", default=5.0, min=0.001, max=86400.0, step=0.001),
                io.String.Input("default_aspect_ratio", default="16:9"),
                io.Int.Input("base_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input(
                    "seed_policy",
                    options=["increment", "fixed", "hash_project_shot"],
                    default="increment",
                ),
                io.Boolean.Input("split_long_shots", default=True),
                io.Boolean.Input("strict_exact_dialogue", default=True),
                CastIO.Input("cast", optional=True),
                SoundCanvasIO.Input("sound_canvas", optional=True),
            ],
            outputs=[StudioTimelineIO.Output("timeline"), io.String.Output("timeline_json")],
        )

    @classmethod
    def execute(
        cls,
        project_id,
        shots_json,
        default_backend,
        default_duration_seconds,
        default_aspect_ratio,
        base_seed,
        seed_policy,
        split_long_shots,
        strict_exact_dialogue,
        cast=None,
        sound_canvas=None,
    ):
        timeline = build_studio_timeline(
            project_id,
            shots_json,
            default_backend,
            default_duration_seconds,
            default_aspect_ratio,
            base_seed,
            seed_policy,
            split_long_shots,
            strict_exact_dialogue,
            cast,
            sound_canvas,
        )
        preview = {
            "project_id": timeline["project_id"],
            "timeline_hash": timeline["timeline_hash"],
            "shot_count": timeline["shot_count"],
            "total_frames": timeline["total_frames"],
            "total_duration_seconds": timeline["total_duration_seconds"],
            "fps": timeline["fps"],
            "shots": [
                {
                    "index": shot["index"],
                    "id": shot["id"],
                    "start_seconds": shot["start_seconds"],
                    "end_seconds": shot["end_seconds"],
                    "frame_count": shot["frame_count"],
                    "seed": shot["seed"],
                    "backend": shot["prompt_packet"]["backend"],
                    "status": shot["status"],
                    "prompt": shot["prompt_packet"]["visual_prompt"][:240],
                }
                for shot in timeline["shots"]
            ],
        }
        return io.NodeOutput(
            timeline,
            canonical_json(timeline),
            ui={"t8_studio_timeline": [canonical_json(preview)]},
        )


class MiniMaxH3StudioShotSelectT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StudioShotSelectT8Advanced",
            display_name="MiniMax H3 Studio Shot Select / 镜头选择 (Advanced)",
            description=(
                "Selects one planned shot and exposes prompt, length and seed for existing "
                "Conditioning/Sampler nodes. It does not queue or load anything by itself."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                StudioTimelineIO.Input("timeline"),
                io.Int.Input("shot_index", default=0, min=0, max=100000),
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
    def execute(cls, timeline, shot_index):
        shot, report = select_studio_shot(timeline, shot_index)
        packet = shot["prompt_packet"]
        return io.NodeOutput(
            packet,
            packet["compiled_prompt"],
            packet["negative_prompt"],
            shot["frame_count"],
            shot["seed"],
            canonical_json(shot),
            canonical_json(report),
        )


class MiniMaxH3SelectiveSegmentRepairT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SelectiveSegmentRepairT8Advanced",
            display_name="MiniMax H3 Selective Segment Repair / 选择性分段重做 (Advanced)",
            description=(
                "Creates a non-destructive repair list from explicit indices or quality evidence. "
                "It never overwrites or automatically accepts existing media."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                StudioTimelineIO.Input("timeline"),
                io.Combo.Input(
                    "selection_policy",
                    options=["manual", "failed_status", "score_threshold", "failed_or_score"],
                    default="manual",
                ),
                io.String.Input("manual_indices", default="0"),
                io.String.Input("quality_report_json", multiline=True, default='{"segments":[]}'),
                io.String.Input("thresholds_json", multiline=True, default='{"identity":{"min":0.75}}'),
                io.Combo.Input(
                    "repair_mode",
                    options=["auto", "seed_retry", "prompt_tighten", "reference_refresh", "full_regenerate"],
                    default="auto",
                ),
                io.String.Input("prompt_addendum", multiline=True, default=""),
                io.Int.Input("seed_stride", default=1009, min=1, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("context_before_frames", default=22, min=0, max=362),
                io.Int.Input("context_after_frames", default=22, min=0, max=362),
            ],
            outputs=[RepairPlanIO.Output("repair_plan"), io.String.Output("repair_plan_json")],
        )

    @classmethod
    def execute(
        cls,
        timeline,
        selection_policy,
        manual_indices,
        quality_report_json,
        thresholds_json,
        repair_mode,
        prompt_addendum,
        seed_stride,
        context_before_frames,
        context_after_frames,
    ):
        plan = build_selective_repair_plan(
            timeline, quality_report_json, selection_policy, manual_indices,
            thresholds_json, repair_mode, prompt_addendum, seed_stride,
            context_before_frames, context_after_frames,
        )
        return io.NodeOutput(plan, canonical_json(plan))


class MiniMaxH3RepairSegmentSelectT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RepairSegmentSelectT8Advanced",
            display_name="MiniMax H3 Repair Segment Select / 重做段选择 (Advanced)",
            description=(
                "Selects one item from a non-destructive repair plan and exposes it to the "
                "existing generation graph. Acceptance remains a separate user action."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                RepairPlanIO.Input("repair_plan"),
                io.Int.Input("repair_index", default=0, min=0, max=100000),
            ],
            outputs=[
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.Int.Output("length"),
                io.Int.Output("seed"),
                io.String.Output("repair_mode"),
                io.String.Output("repair_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, repair_plan, repair_index):
        repair, report = select_repair_segment(repair_plan, repair_index)
        return io.NodeOutput(
            repair["prompt"],
            repair["negative_prompt"],
            repair["frame_count"],
            repair["repair_seed"],
            repair["mode"],
            canonical_json(repair),
            canonical_json(report),
        )


STUDIO_ADVANCED_NODE_CLASSES = [
    MiniMaxH3UnifiedCastT8Advanced,
    MiniMaxH3SoundCanvasT8Advanced,
    MiniMaxH3PromptCompilerT8Advanced,
    MiniMaxH3StudioTimelineT8Advanced,
    MiniMaxH3StudioShotSelectT8Advanced,
    MiniMaxH3SelectiveSegmentRepairT8Advanced,
    MiniMaxH3RepairSegmentSelectT8Advanced,
]
