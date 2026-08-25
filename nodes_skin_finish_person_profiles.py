from __future__ import annotations

from comfy_api.latest import io

from .skin_finish import PRESET_CONFIG
from .skin_finish_person_profiles import (
    build_skin_finish_person_profile,
    run_skin_finish_per_person,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"
TrackPlanIO = io.Custom("H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
IdentityAssignmentIO = io.Custom("H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
SkinFinishProfilesIO = io.Custom("H3_T8_SKIN_FINISH_PERSON_PROFILES")
SkinFinishStateIO = io.Custom("H3_T8_SKIN_FINISH_STATE")


class MiniMaxH3SkinFinishPersonProfileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishPersonProfileT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Person Profile / "
                "逐人物肤质参数 (Advanced EXP)"
            ),
            description=(
                "Adds one explicit character_id or shot:track Skin Finish profile to a small "
                "hashed in-memory stack. Chain the output into another profile node for more "
                "people. It never estimates skin tone or accepts a candidate automatically."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "selector_type",
                    options=["character_id", "shot_track"],
                    default="character_id",
                ),
                io.String.Input(
                    "selector",
                    default="Character_A",
                    tooltip=(
                        "Use the reviewed character ID, or switch selector_type to shot_track "
                        "and enter a shot-local key such as 0:1."
                    ),
                ),
                io.Combo.Input("preset", options=list(PRESET_CONFIG), default="subtle"),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("texture_keep", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("shine_control", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "tone_adjust",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Bounded midtone exposure-like adjustment; not automatic skin-tone "
                        "matching. Keep 0 unless reviewed on the actual character."
                    ),
                ),
                SkinFinishProfilesIO.Input("previous_profiles", optional=True),
            ],
            outputs=[
                SkinFinishProfilesIO.Output("profiles"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_skin_finish_person_profile(**kwargs))


class MiniMaxH3SkinFinishPerPersonT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishPerPersonT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Per-Person / "
                "逐人物逐镜头肤质收尾 (Advanced EXP)"
            ),
            description=(
                "Reassigns a source-bound combined ParseNet skin mask to exact SAM3.1 tracks, "
                "then applies reviewed per-character or per-shot parameters. Exact shot:track "
                "profiles override character profiles. Cross-person overlap and unmatched "
                "people remain bit-exact source pixels."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                TrackPlanIO.Input("track_plan"),
                io.Mask.Input("semantic_skin_mask"),
                io.String.Input(
                    "semantic_report_json",
                    multiline=True,
                    tooltip=(
                        "Connect report_json from the same Multi-Person Semantic Mask node. "
                        "The report binds source, track plan and mask."
                    ),
                ),
                io.Combo.Input(
                    "default_policy",
                    options=["source_unmatched", "default_profile"],
                    default="source_unmatched",
                    tooltip=(
                        "source_unmatched changes only explicitly profiled people. "
                        "default_profile applies the default controls to every remaining track."
                    ),
                ),
                io.Combo.Input(
                    "default_preset", options=list(PRESET_CONFIG), default="subtle"
                ),
                io.Float.Input("default_amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "default_texture_keep", default=0.90, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "default_shine_control", default=0.35, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "default_tone_adjust", default=0.0, min=-1.0, max=1.0, step=0.01
                ),
                io.Combo.Input(
                    "execution_mode",
                    options=["candidate_only", "review_only", "bypass"],
                    default="candidate_only",
                ),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip="False preserves source on selected output; acceptance is never automatic.",
                ),
                io.Int.Input("chunk_frames", default=2, min=1, max=32),
                io.Int.Input("proxy_long_side", default=640, min=128, max=1280, step=32),
                io.Int.Input("preview_count", default=6, min=1, max=8),
                SkinFinishProfilesIO.Input("profiles", optional=True),
                IdentityAssignmentIO.Input("identity_assignment", optional=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("used_skin_mask"),
                io.Mask.Output("rejected_skin_mask"),
                io.Image.Output("ownership_preview"),
                SkinFinishStateIO.Output("skin_finish_state"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*run_skin_finish_per_person(**kwargs))


SKIN_FINISH_PERSON_PROFILE_NODE_CLASSES = [
    MiniMaxH3SkinFinishPersonProfileT8Advanced,
    MiniMaxH3SkinFinishPerPersonT8Advanced,
]
