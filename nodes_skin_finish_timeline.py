from __future__ import annotations

from comfy_api.latest import io

from .skin_finish import PRESET_CONFIG
from .skin_finish_timeline import (
    build_skin_finish_timeline_keyframe,
    run_skin_finish_timeline,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"
StudioTimelineIO = io.Custom("H3_T8_STUDIO_TIMELINE")
TrackPlanIO = io.Custom("H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
IdentityAssignmentIO = io.Custom("H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
SkinFinishTimelinePlanIO = io.Custom("H3_T8_SKIN_FINISH_TIMELINE_PLAN")
SkinFinishStateIO = io.Custom("H3_T8_SKIN_FINISH_STATE")


class MiniMaxH3SkinFinishTimelineKeyframeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishTimelineKeyframeT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Timeline Keyframe / "
                "肤质时间关键帧 (Advanced EXP)"
            ),
            description=(
                "Adds one source-bound Skin Finish keyframe to a hashed Studio Timeline plan. "
                "Keyframes interpolate only inside their Studio shot; SAM shot:track identity "
                "remains an independent routing domain. Chain multiple nodes for more keys."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                StudioTimelineIO.Input("studio_timeline"),
                io.Combo.Input(
                    "selector_type",
                    options=["global", "character_id", "shot_track"],
                    default="global",
                ),
                io.String.Input(
                    "selector",
                    default="*",
                    tooltip=(
                        "global ignores this text; character_id uses a reviewed assignment; "
                        "shot_track uses the SAM-local key such as 0:1."
                    ),
                ),
                io.Int.Input("studio_shot_index", default=0, min=0, max=511),
                io.Int.Input(
                    "frame_in_shot",
                    default=0,
                    min=0,
                    max=361,
                    tooltip="Local frame inside the selected Studio Timeline shot.",
                ),
                io.Combo.Input(
                    "interpolation_to_next",
                    options=["hold", "linear", "smoothstep"],
                    default="smoothstep",
                    tooltip=(
                        "Controls continuous parameters until the next key in the same Studio "
                        "shot. Preset is categorical and switches only at the next key."
                    ),
                ),
                io.Combo.Input("preset", options=list(PRESET_CONFIG), default="subtle"),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("texture_keep", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("shine_control", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("tone_adjust", default=0.0, min=-1.0, max=1.0, step=0.01),
                SkinFinishTimelinePlanIO.Input("previous_plan", optional=True),
            ],
            outputs=[
                SkinFinishTimelinePlanIO.Output("timeline_plan"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_skin_finish_timeline_keyframe(**kwargs))


class MiniMaxH3SkinFinishTimelineT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishTimelineT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Studio Timeline / "
                "逐人物肤质时间执行 (Advanced EXP)"
            ),
            description=(
                "Applies reviewed Skin Finish keyframes to exact semantic skin owned by each "
                "SAM3.1 track. Routing precedence is SAM shot:track, character_id, global, then "
                "bit-exact source. It never blends across Studio cuts or accepts automatically."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                StudioTimelineIO.Input("studio_timeline"),
                SkinFinishTimelinePlanIO.Input("timeline_plan"),
                TrackPlanIO.Input("track_plan"),
                io.Mask.Input("semantic_skin_mask"),
                io.String.Input(
                    "semantic_report_json",
                    multiline=True,
                    tooltip=(
                        "Connect report_json from the same source-bound Multi-Person Semantic "
                        "Mask node."
                    ),
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
        return io.NodeOutput(*run_skin_finish_timeline(**kwargs))


SKIN_FINISH_TIMELINE_NODE_CLASSES = [
    MiniMaxH3SkinFinishTimelineKeyframeT8Advanced,
    MiniMaxH3SkinFinishTimelineT8Advanced,
]
