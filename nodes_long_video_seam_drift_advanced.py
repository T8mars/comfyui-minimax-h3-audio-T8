from __future__ import annotations

from comfy_api.latest import io

from .long_video_seam_drift_advanced import process_long_video_seam_drift


CATEGORY = "T8/MiniMax H3/Long Video/Advanced"


class MiniMaxH3LongVideoSeamDriftT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoSeamDriftT8Advanced",
            display_name="MiniMax H3 Long Video Seam Drift / 长视频接缝色彩漂移审计 (Advanced EXP/T8)",
            description=(
                "Reports seam brightness, white-balance, chroma, texture and optional person-ROI "
                "changes. The opt-in candidate uses only bounded low-frequency gain/offset and "
                "rolls unsafe frames back to source. It never sharpens, invents detail or touches audio."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                io.String.Input("boundary_frames_json", default="[124]", multiline=True),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "bounded_candidate_exp"],
                    default="report_only",
                ),
                io.Combo.Input(
                    "color_contract",
                    options=["sdr_rec709_0_to_1", "unknown_or_hdr"],
                    default="sdr_rec709_0_to_1",
                ),
                io.Int.Input("analysis_window_frames", default=3, min=1, max=24),
                io.Int.Input("transition_frames", default=24, min=1, max=240),
                io.Float.Input("scene_cut_threshold", default=0.18, min=0.01, max=1.0, step=0.01),
                io.Float.Input("minimum_drift", default=0.008, min=0.0, max=0.25, step=0.001),
                io.Float.Input("maximum_gain", default=1.08, min=1.0, max=1.5, step=0.01),
                io.Float.Input("maximum_offset", default=0.04, min=0.0, max=0.25, step=0.005),
                io.Float.Input("maximum_frame_change", default=0.06, min=0.0, max=0.5, step=0.005),
                io.Float.Input(
                    "maximum_texture_ratio_deviation",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Mask.Input("person_roi", optional=True),
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
        return io.NodeOutput(*process_long_video_seam_drift(**kwargs))


LONG_VIDEO_SEAM_DRIFT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoSeamDriftT8Advanced,
]
