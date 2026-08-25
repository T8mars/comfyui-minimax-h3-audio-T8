from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_specular_frequency import (
    separate_skin_finish_specular_frequencies,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishSpecularFrequencyT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishSpecularFrequencyT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Specular-Aware Split / 高光感知纹理解耦 "
                "(Advanced EXP)"
            ),
            description=(
                "Runs the unchanged Frequency Split, then selectively restores only the darker "
                "highlight intent that split lost. The result stays between the frequency base "
                "and the input Skin Finish candidate; mask exterior, alpha and audio stay bound."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Float.Input("low_frequency_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("source_detail_gain", default=1.0, min=0.0, max=1.25, step=0.01),
                io.Float.Input(
                    "separation_radius_percent",
                    default=3.0,
                    min=0.10,
                    max=5.0,
                    step=0.05,
                ),
                io.Int.Input("maximum_radius_px", default=32, min=1, max=128, advanced=True),
                io.Float.Input(
                    "highlight_detail_suppression",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Restores the input candidate only where bright positive detail was lost. "
                        "The correction cannot pass beyond that candidate; 0 is exact ordinary split."
                    ),
                ),
                io.Float.Input("highlight_start", default=0.60, min=0.0, max=0.95, step=0.01),
                io.Float.Input("highlight_end", default=0.92, min=0.05, max=1.0, step=0.01),
                io.Float.Input(
                    "positive_detail_threshold",
                    default=0.004,
                    min=0.0,
                    max=0.05,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "treatment_intent_scale",
                    default=0.004,
                    min=0.0001,
                    max=0.10,
                    step=0.0005,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_specular_delta",
                    default=0.04,
                    min=0.0,
                    max=0.10,
                    step=0.005,
                    tooltip="Hard per-pixel luma correction cap before Texture Guard.",
                ),
                io.Float.Input(
                    "minimum_mask_area",
                    default=0.0001,
                    min=0.0,
                    max=0.25,
                    step=0.0001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_mask_area",
                    default=0.50,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_new_clipped_fraction",
                    default=0.0005,
                    min=0.0,
                    max=0.25,
                    step=0.0005,
                ),
                io.Float.Input(
                    "clipping_epsilon",
                    default=1.0 / 255.0,
                    min=0.0001,
                    max=0.05,
                    step=0.0001,
                    advanced=True,
                ),
                io.Int.Input("chunk_frames", default=4, min=1, max=32, advanced=True),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip="The source stays selected until the labelled candidate is reviewed.",
                ),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("specular_frequency_candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("effective_mask"),
                io.Mask.Output("rejected_mask"),
                io.Image.Output("difference"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*separate_skin_finish_specular_frequencies(**kwargs))


SKIN_FINISH_SPECULAR_FREQUENCY_NODE_CLASSES = [
    MiniMaxH3SkinFinishSpecularFrequencyT8Advanced
]
