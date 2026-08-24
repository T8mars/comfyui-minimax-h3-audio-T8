from __future__ import annotations

from comfy_api.latest import io

from .skin_finish import PRESET_CONFIG, build_skin_finish_review, run_skin_finish


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"
SkinFinishStateIO = io.Custom("H3_T8_SKIN_FINISH_STATE")
FaceRefinePlanIO = io.Custom("H3_T8_FACE_REFINE_PLAN")


class MiniMaxH3SkinFinishT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishT8",
            display_name="MiniMax H3 Skin Finish / H3肤质收尾 (T8)",
            description=(
                "Creates a conservative non-generative skin-finishing candidate only inside "
                "an explicit MASK. Missing or unreliable masks ABSTAIN to the exact source. "
                "It cannot reconstruct identity, missing facial structure, blur or lip sync."
            ),
            category=CATEGORY,
            inputs=[
                io.Image.Input("frames"),
                io.Combo.Input(
                    "preset",
                    options=list(PRESET_CONFIG),
                    default="subtle",
                    tooltip=(
                        "subtle is the safe default; oil_control reduces bright oily patches; "
                        "tone_even favors low-frequency tone consistency."
                    ),
                ),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("texture_keep", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("shine_control", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("tone_adjust", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "execution_mode",
                    options=["candidate_only", "review_only", "bypass"],
                    default="candidate_only",
                ),
                io.Int.Input("chunk_frames", default=4, min=1, max=32, advanced=True),
                io.Mask.Input(
                    "skin_mask",
                    optional=True,
                    tooltip=(
                        "Required for processing. Without a reliable explicit mask the node "
                        "returns the source unchanged and reports ABSTAIN."
                    ),
                ),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.Audio.Output("audio"),
                io.Mask.Output("used_skin_mask"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames,
        preset,
        amount,
        texture_keep,
        shine_control,
        tone_adjust,
        execution_mode,
        chunk_frames,
        skin_mask=None,
        audio=None,
    ):
        values = run_skin_finish(
            frames,
            preset=preset,
            amount=amount,
            texture_keep=texture_keep,
            shine_control=shine_control,
            tone_adjust=tone_adjust,
            execution_mode=execution_mode,
            chunk_frames=chunk_frames,
            mask=skin_mask,
            audio=audio,
            mask_source="external_exact",
            accept_candidate=False,
        )
        candidate, source, _, used_mask, _, _, _, audio_out, report = values
        return io.NodeOutput(candidate, source, audio_out, used_mask, report)


class MiniMaxH3SkinFinishAdvancedT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishAdvancedT8",
            display_name="MiniMax H3 Skin Finish / H3肤质收尾 (Advanced)",
            description=(
                "Append-only advanced P0 route with explicit mask provenance, area gates, "
                "CPU-bounded chunks, exact outside-mask pixels, exact alpha/aux preservation, "
                "and source-by-default selection. face_refine_plan builds a conservative inner-"
                "face proxy mask; it is not a semantic face parser."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Combo.Input(
                    "mask_source",
                    options=["external_exact", "face_refine_plan"],
                    default="external_exact",
                ),
                io.Combo.Input("preset", options=list(PRESET_CONFIG), default="subtle"),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("texture_keep", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("shine_control", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("tone_adjust", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "execution_mode",
                    options=["candidate_only", "review_only", "bypass"],
                    default="candidate_only",
                ),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "False preserves the source on selected output. Acceptance is always "
                        "explicit and never automatic."
                    ),
                ),
                io.Boolean.Input("protect_features", default=True),
                io.Float.Input(
                    "minimum_mask_area", default=0.002, min=0.0, max=0.50, step=0.001
                ),
                io.Float.Input(
                    "maximum_mask_area", default=0.45, min=0.01, max=1.0, step=0.01
                ),
                io.Int.Input("mask_feather_px", default=3, min=0, max=64),
                io.Int.Input("temporal_mask_radius", default=0, min=0, max=8),
                io.Int.Input("proxy_long_side", default=640, min=128, max=1280, step=32),
                io.Int.Input("chunk_frames", default=4, min=1, max=32),
                io.Mask.Input("skin_mask", optional=True),
                FaceRefinePlanIO.Input("face_plan", optional=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("used_skin_mask"),
                io.Mask.Output("rejected_mask"),
                io.Image.Output("difference"),
                SkinFinishStateIO.Output("skin_finish_state"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        skin_mask = kwargs.pop("skin_mask", None)
        face_plan = kwargs.pop("face_plan", None)
        audio = kwargs.pop("audio", None)
        values = run_skin_finish(
            mask=skin_mask,
            face_plan=face_plan,
            audio=audio,
            **kwargs,
        )
        candidate, source, selected, used, rejected, difference, state, audio_out, report = values
        return io.NodeOutput(
            candidate,
            source,
            selected,
            audio_out,
            used,
            rejected,
            difference,
            state,
            report,
        )


class MiniMaxH3SkinFinishPreviewAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishPreviewAuditT8Advanced",
            display_name="MiniMax H3 Skin Finish Preview / 肤质收尾预览审计 (Advanced)",
            description=(
                "Builds a source/candidate split, full-resolution masked crop, used/rejected "
                "mask view, amplified difference and a ±2-frame loop. The selected output "
                "stays on source unless accept_candidate is explicitly enabled and audio "
                "passthrough remains PCM-exact."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("used_mask"),
                io.Mask.Input("rejected_mask"),
                SkinFinishStateIO.Input("skin_finish_state"),
                io.String.Input("gate_report_json", multiline=True),
                io.Int.Input("frame_index", default=0, min=0, max=65535),
                io.Float.Input(
                    "comparison_position",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Left side is source; right side is candidate.",
                ),
                io.Boolean.Input("accept_candidate", default=False),
                io.Audio.Input("audio_source", optional=True),
                io.Audio.Input("audio_passthrough", optional=True),
            ],
            outputs=[
                io.Image.Output("selected"),
                io.Image.Output("split_comparison"),
                io.Image.Output("source_crop"),
                io.Image.Output("candidate_crop"),
                io.Image.Output("mask_preview"),
                io.Image.Output("difference_preview"),
                io.Image.Output("plus_minus_2_loop"),
                io.Audio.Output("audio"),
                io.String.Output("review_report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = build_skin_finish_review(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[-1],)})


SKIN_FINISH_NODE_CLASSES = [
    MiniMaxH3SkinFinishT8,
    MiniMaxH3SkinFinishAdvancedT8,
    MiniMaxH3SkinFinishPreviewAuditT8Advanced,
]
