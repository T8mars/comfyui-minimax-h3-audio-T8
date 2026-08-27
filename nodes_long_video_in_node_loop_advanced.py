from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .long_video_in_node_loop_advanced import run_long_video_in_node_loop
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


def _preview_video(path_value: str):
    path = Path(path_value).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root not in path.parents:
        raise ValueError("In-node long-video preview is not inside the ComfyUI output directory")
    relative = path.relative_to(output_root)
    saved = ui.SavedResult(relative.name, relative.parent.as_posix(), io.FolderType.output)
    return InputImpl.VideoFromFile(str(path)), ui.PreviewVideo([saved])


class MiniMaxH3LongVideoInNodeLoopT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoInNodeLoopT8Advanced",
            display_name=(
                "MiniMax H3 In-Node Long Video Loop / 节点内循环长视频 (Advanced EXP/T8)"
            ),
            description=(
                "One execution generates all H3 segments strictly in sequence, atomically "
                "accepts each completed segment, resumes from disk after interruption, and "
                "streams the accepted files into one final VIDEO. Existing long-video nodes "
                "and workflows are unchanged. This route intentionally has no per-segment "
                "human review gate; use the existing Background route when review is required."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.String.Input("chain_id", default="my_h3_in_node_long_video"),
                io.Float.Input(
                    "total_duration_seconds",
                    default=30.0,
                    min=0.04,
                    max=3600.0,
                    step=0.01,
                ),
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input(
                    "render_window_frames",
                    default=124,
                    min=124,
                    max=362,
                    step=17,
                    tooltip=(
                        "Each segment is sampled independently with this fixed H3 window. "
                        "124 is the bounded-memory baseline."
                    ),
                ),
                io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
                io.String.Input(
                    "global_prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "segment_prompts_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Optional list/object. Each segment may override prompt, seed and note."
                    ),
                ),
                io.Int.Input(
                    "base_seed",
                    default=123456789,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
                io.Combo.Input(
                    "seed_policy",
                    options=["increment", "fixed", "hash_chain_segment"],
                    default="increment",
                ),
                io.Int.Input("steps", default=4, min=1, max=1000),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
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
                io.Combo.Input(
                    "task_type",
                    options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="auto",
                ),
                io.Combo.Input(
                    "context_audio",
                    options=["video_and_audio", "video_only"],
                    default="video_and_audio",
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
                io.Boolean.Input("add_source_as_reference", default=True, advanced=True),
                io.Int.Input(
                    "prompt_primary_audio_ordinal",
                    default=0,
                    min=0,
                    max=9,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    advanced=True,
                ),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.Combo.Input(
                    "first_frame_reuse",
                    options=["segment0_only", "persistent_identity_reference"],
                    default="segment0_only",
                    advanced=True,
                ),
                io.Combo.Input(
                    "persistent_identity_strategy",
                    options=["single_reference", "scene_plus_identity"],
                    default="single_reference",
                    advanced=True,
                ),
                io.Int.Input(
                    "persistent_identity_interval",
                    default=1,
                    min=1,
                    max=32,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "resume_existing",
                    default=True,
                    tooltip=(
                        "Resume only when the saved job contract matches. Disable to require "
                        "an empty chain_id."
                    ),
                ),
                io.String.Input("filename_prefix", default="H3_In_Node_Long_Video"),
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
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input("model_id", default="unknown", advanced=True),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Image.Input("persistent_identity_image", optional=True, advanced=True),
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
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_segments"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        chain_id,
        total_duration_seconds,
        width,
        height,
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
        task_type,
        context_audio,
        audio_mode,
        audio_denoise_strength,
        add_source_as_reference,
        prompt_primary_audio_ordinal,
        strict_prompt_tags,
        ref_image_size,
        reference_video_policy,
        first_frame_reuse,
        persistent_identity_strategy,
        persistent_identity_interval,
        resume_existing,
        filename_prefix,
        audio_seam_policy,
        bridge_ms,
        bit_depth,
        crf,
        model_id,
        drive_audio=None,
        final_audio=None,
        first_frame=None,
        last_frame=None,
        persistent_identity_image=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):
        video_path, manifest_path, completed, status, report = (
            run_long_video_in_node_loop(
                model,
                clip,
                video_vae,
                audio_vae,
                chain_id=chain_id,
                total_duration_seconds=total_duration_seconds,
                render_window_frames=render_window_frames,
                context_frames=context_frames,
                global_prompt=global_prompt,
                segment_prompts_json=segment_prompts_json,
                base_seed=base_seed,
                seed_policy=seed_policy,
                steps=steps,
                shift_video=shift_video,
                shift_audio=shift_audio,
                sampler_name=sampler_name,
                scheduler=scheduler,
                width=width,
                height=height,
                task_type=task_type,
                context_audio=context_audio,
                audio_mode=audio_mode,
                audio_denoise_strength=audio_denoise_strength,
                add_source_as_reference=add_source_as_reference,
                prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
                strict_prompt_tags=strict_prompt_tags,
                ref_image_size=ref_image_size,
                reference_video_policy=reference_video_policy,
                first_frame_reuse=first_frame_reuse,
                persistent_identity_strategy=persistent_identity_strategy,
                persistent_identity_interval=persistent_identity_interval,
                resume_existing=resume_existing,
                filename_prefix=filename_prefix,
                audio_seam_policy=audio_seam_policy,
                bridge_ms=bridge_ms,
                bit_depth=bit_depth,
                crf=crf,
                model_id=model_id,
                drive_audio=drive_audio,
                final_audio=final_audio,
                first_frame=first_frame,
                last_frame=last_frame,
                persistent_identity_image=persistent_identity_image,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video,
            video_path,
            manifest_path,
            completed,
            status,
            report,
            ui=preview,
        )


LONG_VIDEO_IN_NODE_LOOP_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoInNodeLoopT8Advanced,
]
