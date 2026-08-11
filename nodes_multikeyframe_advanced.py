from __future__ import annotations

from comfy_api.latest import io

from .multikeyframe_advanced import (
    DEFAULT_VISUAL_NOISE_AUG,
    KEYFRAME_PLAN_TYPE,
    append_keyframe_plan,
    build_multikeyframe_conditioning,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
MAX_RESOLUTION = 16384
KeyframePlanIO = io.Custom(KEYFRAME_PLAN_TYPE)


class MiniMaxH3KeyframePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3KeyframePlanT8Advanced",
            display_name="MiniMax H3 Keyframe Plan / 中间关键帧计划 (Advanced)",
            description=(
                "Chains one middle timeline image with an explicit frame/second/percent "
                "position and a raw per-frame H3 visual noise-augmentation value."
            ),
            category=CATEGORY,
            inputs=[
                KeyframePlanIO.Input(
                    "previous_plan",
                    optional=True,
                    tooltip="Connect the previous Advanced Keyframe Plan to build a timeline.",
                ),
                io.Image.Input("image", tooltip="One middle timeline keyframe; batch item 0 is used."),
                io.Combo.Input(
                    "position_mode",
                    options=["frame", "seconds", "percent"],
                    default="frame",
                    tooltip="percent uses an explicit 0..100 scale; no 0..1 ambiguity.",
                ),
                io.Float.Input(
                    "position",
                    default=62.0,
                    min=0.0,
                    max=3600.0,
                    step=0.01,
                    tooltip=(
                        "Absolute 24fps frame, seconds, or 0..100 percent according to "
                        "position_mode. It must resolve strictly between first and last."
                    ),
                ),
                io.Float.Input(
                    "visual_noise_aug",
                    display_name="raw visual noise_aug / 原始视觉混噪值",
                    default=DEFAULT_VISUAL_NOISE_AUG,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Raw H3 conditioning value, not a calibrated linear strength. "
                        "Start with 0.999; values <=0.950 are aggressive."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["center_crop", "stretch"],
                    default="center_crop",
                ),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[KeyframePlanIO.Output(display_name="keyframe_plan")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        image,
        position_mode,
        position,
        visual_noise_aug,
        resize_mode,
        enabled,
        previous_plan=None,
    ):
        return io.NodeOutput(
            append_keyframe_plan(
                previous_plan,
                image,
                position_mode,
                position,
                visual_noise_aug,
                resize_mode,
                enabled,
            )
        )


class MiniMaxH3MultiKeyframeConditioningT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiKeyframeConditioningT8Advanced",
            display_name=(
                "MiniMax H3 Multi-Keyframe Conditioning / 多关键帧条件 (Advanced)"
            ),
            description=(
                "Opt-in Advanced FL2VA/Hybrid conditioning with first, last, and chained "
                "middle timeline keyframes. It clones MODEL and uses scoped H3 patches; "
                "the stable Conditioning node and saved workflows remain unchanged."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 MODEL; the output is a scoped clone."),
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="24fps; snapped up to the 17n+5 H3 grid.",
                ),
                io.Combo.Input(
                    "task_type",
                    options=["auto", "FL2VA", "Hybrid"],
                    default="auto",
                    tooltip=(
                        "Middle timeline keyframes require both first and last frames. "
                        "Use Hybrid when reference media is also connected."
                    ),
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["lock_source", "remix_source", "reference_only", "native"],
                    default="lock_source",
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
                    default=1,
                    min=0,
                    max=9,
                    step=1,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match", advanced=True
                ),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.Float.Input(
                    "first_frame_noise_aug",
                    display_name="first frame raw noise_aug / 首帧原始混噪值",
                    default=DEFAULT_VISUAL_NOISE_AUG,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Raw H3 value for the first timeline frame, not a calibrated "
                        "strength percentage. Keep 0.999 unless running a controlled A/B."
                    ),
                ),
                io.Float.Input(
                    "last_frame_noise_aug",
                    display_name="last frame raw noise_aug / 尾帧原始混噪值",
                    default=DEFAULT_VISUAL_NOISE_AUG,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Raw H3 value for the last timeline frame, not a calibrated "
                        "strength percentage. Keep 0.999 unless running a controlled A/B."
                    ),
                ),
                io.Float.Input(
                    "reference_visual_noise_aug",
                    display_name="reference raw noise_aug / 普通视觉参考全局值",
                    default=DEFAULT_VISUAL_NOISE_AUG,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "One raw H3 value shared by non-timeline image/video reference blocks; "
                        "not an independent or calibrated strength percentage."
                    ),
                ),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                KeyframePlanIO.Input("keyframe_plan", optional=True),
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
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(display_name="mux_audio"),
                io.String.Output(display_name="conditioned_prompt"),
                io.String.Output(display_name="media_map_json"),
                io.String.Output(display_name="keyframe_map_json"),
                io.String.Output(display_name="report_json"),
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
        first_frame_noise_aug,
        last_frame_noise_aug,
        reference_visual_noise_aug,
        drive_audio=None,
        final_audio=None,
        first_frame=None,
        last_frame=None,
        keyframe_plan=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):
        return io.NodeOutput(
            *build_multikeyframe_conditioning(
                model=model,
                clip=clip,
                video_vae=video_vae,
                audio_vae=audio_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                task_type=task_type,
                audio_mode=audio_mode,
                audio_denoise_strength=audio_denoise_strength,
                add_source_as_reference=add_source_as_reference,
                prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
                strict_prompt_tags=strict_prompt_tags,
                ref_image_size=ref_image_size,
                reference_video_policy=reference_video_policy,
                first_frame_noise_aug=first_frame_noise_aug,
                last_frame_noise_aug=last_frame_noise_aug,
                reference_visual_noise_aug=reference_visual_noise_aug,
                drive_audio=drive_audio,
                final_audio=final_audio,
                first_frame=first_frame,
                last_frame=last_frame,
                keyframe_plan=keyframe_plan,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
        )


MULTIKEYFRAME_ADVANCED_NODE_CLASSES = [
    MiniMaxH3KeyframePlanT8Advanced,
    MiniMaxH3MultiKeyframeConditioningT8Advanced,
]
