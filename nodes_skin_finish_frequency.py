from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_frequency import separate_skin_finish_frequencies


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishFrequencySplitT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishFrequencySplitT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Frequency Split / 肤色纹理解耦 (Advanced EXP)"
            ),
            description=(
                "Rebuilds a Skin Finish candidate from its low-frequency tone/brightness layer "
                "and the source frame's existing high-frequency detail. It is non-generative: "
                "it cannot deblur, create pores, repair identity or decide that a result is better."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Float.Input(
                    "low_frequency_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "0 keeps the source low-frequency layer; 1 uses the candidate's "
                        "low-frequency tone/brightness correction."
                    ),
                ),
                io.Float.Input(
                    "source_detail_gain",
                    default=1.0,
                    min=0.0,
                    max=1.25,
                    step=0.01,
                    tooltip=(
                        "Gain for high-frequency detail already present in the source. "
                        "1.0 is neutral; this does not create missing detail."
                    ),
                ),
                io.Float.Input(
                    "separation_radius_percent",
                    default=1.0,
                    min=0.10,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Low-pass radius as a percentage of the shorter image side. "
                        "Resolution-relative scaling avoids one fixed pixel radius across sizes."
                    ),
                ),
                io.Int.Input(
                    "maximum_radius_px",
                    default=32,
                    min=1,
                    max=128,
                    advanced=True,
                    tooltip="Hard CPU-cost cap for the calculated separation radius.",
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
                    tooltip="Frames exceeding this new clipping fraction return to source.",
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
                    tooltip="Source remains selected until the recombined candidate is reviewed.",
                ),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("frequency_split_candidate"),
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
        return io.NodeOutput(*separate_skin_finish_frequencies(**kwargs))


SKIN_FINISH_FREQUENCY_NODE_CLASSES = [
    MiniMaxH3SkinFinishFrequencySplitT8Advanced
]
