from __future__ import annotations

from comfy_api.latest import io

from .face_refine_window_advanced import (
    apply_face_refine_manual_review,
    build_face_refine_window_plan,
    extract_face_refine_window,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Face Refine Window"
FaceRefineWindowPlanIO = io.Custom("H3_T8_FACE_REFINE_WINDOW_PLAN")
FaceRefineWindowMappingIO = io.Custom("H3_T8_FACE_REFINE_WINDOW_MAPPING")


class MiniMaxH3FaceRefineWindowPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineWindowPlanT8Advanced",
            display_name="MiniMax H3 Face Refine Window Plan / 局部修脸窗口规划 (Advanced)",
            description=(
                "Plans explicit shot-local 24fps H3 windows around 0-based inclusive repair "
                "ranges. It binds the source, rejects hard-cut crossings and overlap ambiguity, "
                "and never loads a model or accepts a candidate automatically. Empty or disabled "
                "plans are exact no-op routes."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.String.Input(
                    "repair_ranges",
                    default="0-23",
                    multiline=False,
                    tooltip="0-based inclusive pairs, for example 0-23,50-60.",
                ),
                io.Combo.Input(
                    "range_mode",
                    options=["frames_inclusive", "seconds_inclusive"],
                    default="frames_inclusive",
                ),
                io.Int.Input("context_before_frames", default=24, min=0, max=361),
                io.Int.Input("context_after_frames", default=42, min=0, max=361),
                io.Int.Input("min_render_frames", default=90, min=22, max=362),
                io.Int.Input("max_render_frames", default=362, min=22, max=362),
                io.Float.Input(
                    "scene_cut_threshold", default=0.28, min=0.01, max=1.0, step=0.01
                ),
                io.Combo.Input(
                    "overlap_policy", options=["reject", "merge"], default="reject"
                ),
                io.Combo.Input(
                    "short_shot_policy",
                    options=["reject", "edge_hold_exp"],
                    default="reject",
                ),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[
                FaceRefineWindowPlanIO.Output("window_plan"),
                io.Mask.Output("repair_mask_preview"),
                io.Int.Output("window_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_face_refine_window_plan(**kwargs))


class MiniMaxH3FaceRefineWindowExtractT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineWindowExtractT8Advanced",
            display_name="MiniMax H3 Face Refine Window Extract / 合法窗口提取 (Advanced)",
            description=(
                "Extracts one contiguous 17n+5 IMAGE window without resizing. Explicit edge "
                "padding repeats only the boundary image while the corresponding AUDIO samples "
                "remain zero. Generated window audio is never authorized as the final soundtrack."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("base_frames"),
                FaceRefineWindowPlanIO.Input("window_plan"),
                io.Int.Input("window_index", default=0, min=0, max=1023),
                io.Combo.Input(
                    "pad_policy", options=["reject", "edge_hold_exp"], default="reject"
                ),
                io.Audio.Input("source_audio", optional=True),
            ],
            outputs=[
                io.Image.Output("render_frames"),
                io.Audio.Output("render_audio"),
                FaceRefineWindowMappingIO.Output("window_mapping"),
                io.Float.Output("source_start_seconds"),
                io.Float.Output("render_duration_seconds"),
                io.String.Output("accept_relative_ranges_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*extract_face_refine_window(**kwargs))


class MiniMaxH3FaceRefineManualReviewT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineManualReviewT8Advanced",
            display_name="MiniMax H3 Face Refine Manual Review / 人工选择回贴 (Advanced)",
            description=(
                "Shows the source window beside the candidate and keeps the complete source "
                "timeline by default. Acceptance requires an explicit confirmation and can only "
                "touch selected source repair frames inside the supplied changed mask. Context, "
                "padding and candidate audio are never accepted automatically."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("candidate_window_frames"),
                io.Mask.Input("changed_mask"),
                FaceRefineWindowMappingIO.Input("window_mapping"),
                io.Combo.Input(
                    "decision",
                    options=["preview_only", "reject", "accept_selected"],
                    default="preview_only",
                ),
                io.String.Input(
                    "accepted_subranges",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Optional 0-based inclusive source-frame ranges. Empty means the planned "
                        "repair range, but only after confirm_accept is enabled."
                    ),
                ),
                io.Boolean.Input("confirm_accept", default=False),
                io.Int.Input("edge_fade_frames", default=2, min=0, max=60),
            ],
            outputs=[
                io.Image.Output("review_frames"),
                io.Image.Output("result_frames"),
                io.Mask.Output("accepted_change_mask"),
                io.Mask.Output("rejected_change_mask"),
                io.Int.Output("accepted_frame_count"),
                io.Int.Output("rejected_frame_count"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*apply_face_refine_manual_review(**kwargs))


FACE_REFINE_WINDOW_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceRefineWindowPlanT8Advanced,
    MiniMaxH3FaceRefineWindowExtractT8Advanced,
    MiniMaxH3FaceRefineManualReviewT8Advanced,
]
