from __future__ import annotations

from comfy_api.latest import io

from .nodes_skin_finish_multiface_parser import CATEGORY, TrackPlanIO
from .skin_finish_safety_audit import (
    AUDIT_SCOPES,
    TEMPORAL_POLICIES,
    audit_skin_finish_candidate,
)


class MiniMaxH3SkinFinishSafetyAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishSafetyAuditT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Safety Audit / 肤质候选安全审计 (Advanced EXP)"
            ),
            description=(
                "Final fail-closed audit for a Skin Finish IMAGE candidate. It can reject "
                "mask spill, excessive changes, treatment jumps, multi-person track spill "
                "and audio mismatch. It cannot auto-score beauty or replace human review."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Combo.Input(
                    "audit_scope",
                    options=list(AUDIT_SCOPES),
                    default="mask_only",
                    tooltip=(
                        "mask_only works without SAM. track_union requires every edited pixel "
                        "inside a tracked person. unique_track_owner also rejects edits where "
                        "two person masks overlap."
                    ),
                ),
                io.Combo.Input(
                    "temporal_policy",
                    options=list(TEMPORAL_POLICIES),
                    default="report_only",
                    tooltip=(
                        "report_only records source-relative treatment jumps. hard_gate also "
                        "rejects jumps above the configured limit."
                    ),
                ),
                io.Float.Input(
                    "maximum_mean_abs_change",
                    default=0.08,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                ),
                io.Float.Input(
                    "maximum_peak_abs_change",
                    default=0.30,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "maximum_temporal_effect_jump",
                    default=0.04,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "Maximum adjacent-frame change in mask-weighted treatment statistics, "
                        "not raw video motion. Shot boundaries reset the comparison."
                    ),
                ),
                io.Float.Input(
                    "maximum_track_leak_fraction",
                    default=0.001,
                    min=0.0,
                    max=1.0,
                    step=0.0005,
                ),
                io.Int.Input(
                    "minimum_temporal_pixels",
                    default=64,
                    min=1,
                    max=1048576,
                    advanced=True,
                ),
                io.Float.Input(
                    "scene_cut_reset_threshold",
                    default=0.20,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Used only without a track plan. A source-frame mean absolute change "
                        "above this value resets the temporal comparison."
                    ),
                ),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "Even after hard gates pass, the source remains selected until a human "
                        "explicitly enables this switch."
                    ),
                ),
                TrackPlanIO.Input("track_plan", optional=True),
                io.Audio.Input("audio_source", optional=True),
                io.Audio.Input("audio_passthrough", optional=True),
            ],
            outputs=[
                io.Image.Output("selected"),
                io.Image.Output(
                    "gated_candidate",
                    tooltip=(
                        "The untouched candidate when all hard gates pass; exact source when "
                        "any hard gate fails. Finalizers should consume this output."
                    ),
                ),
                io.Image.Output("source"),
                io.Audio.Output("audio"),
                io.Boolean.Output("hard_gate_pass"),
                io.Int.Output("failed_frame_count"),
                io.Image.Output("failure_preview"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*audit_skin_finish_candidate(**kwargs))


SKIN_FINISH_SAFETY_AUDIT_NODE_CLASSES = [
    MiniMaxH3SkinFinishSafetyAuditT8Advanced
]
