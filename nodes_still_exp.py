from __future__ import annotations

from comfy_api.latest import io

from .still_image import (
    build_still_image_edit_conditioning,
    decode_still_image,
    run_still_preflight,
)


CATEGORY = "T8/MiniMax H3/Still/Experimental"
MAX_RESOLUTION = 16384


class MiniMaxH3StillConditioningT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StillConditioningT8",
            display_name="MiniMax H3 Reference Image Edit (EXP/T8)",
            description=(
                "Experimental Ref2VA single/multi-image semantic editing. "
                "Picture 1 is the primary edit source; this is not mask-based pixel editing."
            ),
            category=CATEGORY,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "canvas_mode",
                    options=["from_edit_image", "custom"],
                    default="from_edit_image",
                ),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "target_mode",
                    options=[
                        "direct_1_frame",
                        "micro_video_5_frames",
                        "short_video_22_frames",
                        "trained_124_frames",
                    ],
                    default="direct_1_frame",
                    tooltip=(
                        "1-frame is cheapest and most experimental; 22-frame is the next native "
                        "17n+5 grid point; 124-frame is the approximate trained-range baseline."
                    ),
                ),
                io.Float.Input(
                    "reference_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="1 keeps reference latents clean; lower values add seeded reference noise.",
                ),
                io.Combo.Input(
                    "audio_target",
                    options=["generate_and_discard", "lock_silence"],
                    default="generate_and_discard",
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    advanced=True,
                ),
                io.Image.Input(
                    "edit_image",
                    tooltip="Primary source image, presented to Ref2VA as <Picture 1>.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=8,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.String.Output(display_name="conditioned_prompt"),
                io.String.Output(display_name="media_map_json"),
                io.String.Output(display_name="report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        clip,
        video_vae,
        prompt,
        canvas_mode,
        width,
        height,
        target_mode,
        reference_strength,
        audio_target,
        strict_prompt_tags,
        ref_image_size,
        edit_image,
        ref_images=None,
    ):
        return io.NodeOutput(*build_still_image_edit_conditioning(
            clip,
            video_vae,
            prompt,
            canvas_mode,
            width,
            height,
            target_mode,
            reference_strength,
            audio_target,
            strict_prompt_tags,
            ref_image_size,
            edit_image,
            ref_images,
        ))


class MiniMaxH3StillPreflightT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StillPreflightT8",
            display_name="MiniMax H3 Still Preflight (EXP/T8)",
            description="Checks the experimental Ref2VA still-image contract before sampling.",
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "canvas_mode",
                    options=["from_edit_image", "custom"],
                    default="from_edit_image",
                ),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "target_mode",
                    options=[
                        "direct_1_frame",
                        "micro_video_5_frames",
                        "short_video_22_frames",
                        "trained_124_frames",
                    ],
                    default="direct_1_frame",
                ),
                io.Float.Input(
                    "reference_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                ),
                io.Combo.Input(
                    "audio_target",
                    options=["generate_and_discard", "lock_silence"],
                    default="generate_and_discard",
                ),
                io.Model.Input("model", optional=True),
                io.Vae.Input("video_vae", optional=True),
                io.Image.Input("edit_image", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=8,
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output("ready"),
                io.Int.Output("warning_count"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        canvas_mode,
        width,
        height,
        target_mode,
        reference_strength,
        audio_target,
        model=None,
        video_vae=None,
        edit_image=None,
        ref_images=None,
    ):
        return io.NodeOutput(*run_still_preflight(
            canvas_mode,
            width,
            height,
            target_mode,
            reference_strength,
            audio_target,
            model,
            video_vae,
            edit_image,
            ref_images,
        ))


class MiniMaxH3StillDecodeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StillDecodeT8",
            display_name="MiniMax H3 Still Decode (EXP/T8)",
            description="Decodes only the H3 video latent and selects one still candidate.",
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Vae.Input("video_vae"),
                io.Combo.Input(
                    "frame_selection",
                    options=["middle", "first", "last", "index"],
                    default="middle",
                ),
                io.Int.Input("frame_index", default=0, min=0, max=3599, advanced=True),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Image.Output("candidate_frames"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, av_latent, video_vae, frame_selection, frame_index):
        return io.NodeOutput(*decode_still_image(
            av_latent,
            video_vae,
            frame_selection,
            frame_index,
        ))
