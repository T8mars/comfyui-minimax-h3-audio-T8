from __future__ import annotations

from comfy_api.latest import io

from .long_video import CONTEXT_TYPE_NAME
from .long_video_color_match_advanced import process_long_video_color_match


CATEGORY = "T8/MiniMax H3/Long Video/Advanced"
LongVideoContext = io.Custom(CONTEXT_TYPE_NAME)


class MiniMaxH3LongVideoColorMatchT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoColorMatchT8Advanced",
            display_name="MiniMax H3 Long Video Color Match / 长视频接缝颜色统一 (Advanced EXP/T8)",
            description=(
                "Default-on optional seam color matching after Output Trim. Segment 0 stores its "
                "actual tail statistics; later segments use bounded global Lab distribution "
                "matching plus an 8x5 local RGB correction that fades out. Audio and native AV "
                "latent are never changed."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                LongVideoContext.Input("context"),
                io.String.Input("chain_id", default="my_h3_long_video", force_input=True),
                io.Int.Input(
                    "segment_index", default=0, min=0, max=99999, force_input=True
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Default on. Disable for exact source frames while still recording the "
                        "tail statistics for a later segment."
                    ),
                ),
                io.Int.Input("reference_frames", default=5, min=1, max=24, advanced=True),
                io.Int.Input("transition_frames", default=24, min=1, max=240, advanced=True),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True
                ),
                io.Float.Input(
                    "minimum_jump",
                    default=0.0005,
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_offset",
                    default=0.02,
                    min=0.0,
                    max=0.25,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "scene_cut_threshold",
                    default=0.18,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*process_long_video_color_match(**kwargs))


LONG_VIDEO_COLOR_MATCH_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoColorMatchT8Advanced,
]
