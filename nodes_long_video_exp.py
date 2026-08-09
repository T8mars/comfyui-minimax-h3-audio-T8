from __future__ import annotations

from comfy_api.latest import io

from .long_video import (
    CONTEXT_TYPE_NAME,
    build_long_video_conditioning,
    context_fingerprint,
    load_context_state,
    make_long_video_plan,
    patch_long_video_model,
    save_context_state,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
LongVideoContext = io.Custom(CONTEXT_TYPE_NAME)


class MiniMaxH3LongVideoPlannerT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoPlannerT8",
            display_name="MiniMax H3 Segment Planner / 长视频分段规划 (EXP/T8)",
            description=(
                "Plans one retry-safe H3 chain segment. Segment 0 has no overlap; later "
                "segments render a 5/22/39-frame head context and expose exact AV trim metadata."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video"),
                io.Int.Input("segment_index", default=0, min=0, max=99999),
                io.Float.Input("new_duration_seconds", default=4.25, min=0.04, max=900.0, step=0.01),
                io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
                io.Int.Input(
                    "minimum_render_frames",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    advanced=True,
                    tooltip="Keep 124 for the current approximate H3 trained minimum.",
                ),
                io.Float.Input(
                    "timeline_start_seconds",
                    default=-1.0,
                    min=-1.0,
                    max=86400.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "-1 derives the fixed-settings timeline from the quantized first and "
                        "continuation segment durations; set it explicitly if earlier settings differ."
                    ),
                ),
                io.Boolean.Input(
                    "is_final_segment",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Only the final segment may trim a hidden tail for an exact requested duration. "
                        "Its continuation checkpoint is disabled automatically."
                    ),
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
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        chain_id,
        segment_index,
        new_duration_seconds,
        context_frames,
        minimum_render_frames,
        timeline_start_seconds,
        is_final_segment,
    ):
        plan = make_long_video_plan(
            chain_id,
            segment_index,
            new_duration_seconds,
            context_frames,
            minimum_render_frames,
            timeline_start_seconds,
            is_final_segment,
        )
        return io.NodeOutput(
            plan.chain_id,
            plan.segment_index,
            plan.render_frames,
            plan.context_frames,
            plan.trim_start_seconds,
            plan.final_duration_seconds,
            plan.timeline_start_seconds,
            plan.timeline_end_seconds,
            plan.save_context,
            plan.report(),
        )


class MiniMaxH3LongVideoContextLoadT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoContextLoadT8",
            display_name="MiniMax H3 Previous Context / 读取上一段上下文 (EXP/T8)",
            description=(
                "Segment 0 returns an empty passthrough context. Segment N loads only the "
                "validated AV tail saved for segment N-1; it never selects the newest file."
            ),
            category=CATEGORY,
            inputs=[
                io.String.Input("chain_id", default="my_h3_long_video", force_input=True),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
            ],
            outputs=[
                LongVideoContext.Output("context"),
                io.Boolean.Output("has_context"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, chain_id, segment_index):
        return io.NodeOutput(*load_context_state(chain_id, segment_index))

    @classmethod
    def fingerprint_inputs(cls, chain_id, segment_index):
        # Dynamic upstream outputs can be unresolved during ComfyUI's first
        # cache probe. Execution still validates them strictly; the fingerprint
        # must not emit a warning merely because scheduling has not resolved the
        # Planner yet.
        try:
            return context_fingerprint(chain_id, segment_index)
        except (TypeError, ValueError):
            return f"unresolved:{chain_id!r}:{segment_index!r}"


class MiniMaxH3LongVideoConditioningT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoConditioningT8",
            display_name="MiniMax H3 Long Video Conditioning / 长视频续写条件 (EXP/T8)",
            description=(
                "Independent long-video conditioning. It returns a cloned MODEL with a local "
                "H3 payload/layout patch; the input MODEL and all stable T8 nodes remain unchanged."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                LongVideoContext.Input("context"),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
                io.Int.Input("context_frames", default=0, min=0, max=39, force_input=True),
                io.Combo.Input(
                    "context_audio",
                    options=["video_and_audio", "video_only"],
                    default="video_and_audio",
                    tooltip=(
                        "video_and_audio continues the generated AV latent. video_only keeps "
                        "motion context but leaves audio to the selected native/source mode."
                    ),
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, force_input=True),
                io.Combo.Input(
                    "task_type",
                    options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="auto",
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["lock_source", "remix_source", "reference_only", "native"],
                    default="native",
                ),
                io.Float.Input(
                    "audio_denoise_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("add_source_as_reference", default=True),
                io.Int.Input(
                    "prompt_primary_audio_ordinal",
                    default=0,
                    min=0,
                    max=9,
                    step=1,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip=(
                        "Exact frame 0 for the first segment. Later segments normally ignore it; "
                        "persistent_identity_reference may reuse it as a compatibility fallback."
                    ),
                ),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3
                    ),
                ),
                io.Combo.Input(
                    "first_frame_reuse",
                    options=["segment0_only", "persistent_identity_reference"],
                    default="segment0_only",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "segment0_only preserves legacy behavior. persistent_identity_reference "
                        "adds one non-timeline image reference on continuation segments. A connected "
                        "persistent_identity_image is preferred; otherwise first_frame is reused. "
                        "This remains experimental, adds reference rows/VRAM, and is not identity lock."
                    ),
                ),
                io.Image.Input(
                    "persistent_identity_image",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Optional continuation-only identity crop. Prefer one clear face or upper-body "
                        "image. It is ignored on segment 0 and unless first_frame_reuse is set to "
                        "persistent_identity_reference; first_frame still owns exact frame 0."
                    ),
                ),
                io.Combo.Input(
                    "persistent_identity_strategy",
                    options=["single_reference", "scene_plus_identity"],
                    default="single_reference",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "single_reference uses persistent_identity_image when connected, otherwise "
                        "first_frame. scene_plus_identity supplies both images as separate references; "
                        "it costs more reference rows/VRAM and remains a gated experiment."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.Audio.Output("mux_audio"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("media_map_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        context,
        segment_index,
        context_frames,
        context_audio,
        prompt,
        width,
        height,
        length,
        task_type,
        audio_mode,
        audio_denoise_strength,
        add_source_as_reference,
        prompt_primary_audio_ordinal,
        strict_prompt_tags,
        ref_image_size,
        reference_video_policy,
        drive_audio=None,
        final_audio=None,
        first_frame=None,
        last_frame=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        first_frame_reuse="segment0_only",
        persistent_identity_image=None,
        persistent_identity_strategy="single_reference",
    ):
        outputs = build_long_video_conditioning(
            clip,
            video_vae,
            audio_vae,
            context,
            segment_index,
            context_frames,
            context_audio,
            prompt,
            width,
            height,
            length,
            task_type,
            audio_mode,
            audio_denoise_strength,
            add_source_as_reference,
            prompt_primary_audio_ordinal,
            strict_prompt_tags,
            ref_image_size,
            reference_video_policy,
            drive_audio,
            final_audio,
            first_frame,
            last_frame,
            ref_images,
            ref_videos,
            ref_video_audios,
            ref_audios,
            first_frame_reuse,
            persistent_identity_image,
            persistent_identity_strategy,
        )
        return io.NodeOutput(patch_long_video_model(model), *outputs)


class MiniMaxH3LongVideoContextSaveT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoContextSaveT8",
            display_name="MiniMax H3 Save AV Tail / 保存下一段上下文 (EXP/T8)",
            description=(
                "Atomically saves only the bounded H3 AV tail for this segment. Re-running the "
                "same segment replaces its own slot and never changes segment N-1."
            ),
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"),
                io.String.Input("chain_id", default="my_h3_long_video", force_input=True),
                io.Int.Input("segment_index", default=0, min=0, max=99999, force_input=True),
                io.Boolean.Input("save_context", default=True, force_input=True),
                io.String.Input("model_id", default="unknown", advanced=True),
                io.String.Input(
                    "sampling_summary",
                    default="dual_clock_euler/native_flow",
                    advanced=True,
                ),
            ],
            outputs=[io.String.Output("context_path"), io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, av_latent, chain_id, segment_index, save_context, model_id, sampling_summary):
        return io.NodeOutput(*save_context_state(
            av_latent,
            chain_id,
            segment_index,
            model_id,
            sampling_summary,
            save_context,
        ))
