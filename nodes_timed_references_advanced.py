from __future__ import annotations

import math

from comfy_api.latest import io

from .timed_references_advanced import (
    DEFAULT_ANALYSIS_FPS,
    REFERENCE_SIZES,
    _resize_frames,
    append_timed_reference,
    normalize_prompt_tag,
    prepare_timed_video_frames,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Advanced"


class MiniMaxH3TimedImageReferenceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimedImageReferenceT8Advanced",
            display_name="MiniMax H3 Timed Image Reference (Advanced/T8)",
            description=(
                "Adds one Qwen-only semantic image at an exact target time. Chain on "
                "the H3 CLIP line and use #prompt_tag in core Ref2VA. It is not "
                "VAE-encoded, does not enter minimax_refs and consumes no native slot."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Image.Input("image"),
                io.String.Input("prompt_tag", default="anchor"),
                io.Float.Input(
                    "time_seconds", default=0.0, min=0.0, max=9999.0, step=0.001
                ),
                io.Combo.Input(
                    "image_size", options=list(REFERENCE_SIZES), default="512"
                ),
            ],
            outputs=[io.Clip.Output("clip")],
        )

    @classmethod
    def execute(cls, clip, image, prompt_tag, time_seconds, image_size):
        value = float(time_seconds)
        if not math.isfinite(value) or value < 0:
            raise ValueError("time_seconds must be finite and non-negative")
        frames = _resize_frames(image[:1], image_size).expand(2, -1, -1, -1)
        output = append_timed_reference(
            clip,
            {
                "prompt_tag": normalize_prompt_tag(prompt_tag),
                "frames": frames,
                "timestamps": [value, value],
            },
        )
        return io.NodeOutput(output)


class MiniMaxH3TimedVideoReferenceT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimedVideoReferenceT8Advanced",
            display_name="MiniMax H3 Timed Video Reference (Advanced EXP/T8)",
            description=(
                "Presents selected video frames to Qwen at their generated-timeline "
                "times. This is semantic-only conditioning: no native H3 reference "
                "slot, VAE latent or audio is added."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Image.Input("video_frames"),
                io.String.Input("prompt_tag", default="motion"),
                io.Float.Input(
                    "target_start_seconds",
                    default=0.0,
                    min=0.0,
                    max=9999.0,
                    step=0.001,
                ),
                io.Float.Input(
                    "source_fps", default=24.0, min=0.01, max=240.0, step=0.001
                ),
                io.Float.Input(
                    "analysis_fps",
                    default=DEFAULT_ANALYSIS_FPS,
                    min=0.5,
                    max=240.0,
                    step=0.5,
                ),
                io.Combo.Input(
                    "video_size", options=list(REFERENCE_SIZES), default="512"
                ),
            ],
            outputs=[io.Clip.Output("clip")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        video_frames,
        prompt_tag,
        target_start_seconds,
        source_fps,
        analysis_fps,
        video_size,
    ):
        frames, timestamps = prepare_timed_video_frames(
            video_frames,
            source_fps,
            target_start_seconds,
            video_size,
            analysis_fps,
        )
        output = append_timed_reference(
            clip,
            {
                "prompt_tag": normalize_prompt_tag(prompt_tag),
                "frames": frames,
                "timestamps": timestamps,
            },
        )
        return io.NodeOutput(output)


TIMED_REFERENCES_ADVANCED_NODE_CLASSES = [
    MiniMaxH3TimedImageReferenceT8Advanced,
    MiniMaxH3TimedVideoReferenceT8Advanced,
]
