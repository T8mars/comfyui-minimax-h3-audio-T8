from __future__ import annotations

from comfy_api.latest import io

from .trajectory_control_advanced import (
    CLIP_POLICIES,
    EASING_MODES,
    RENDER_MODES,
    build_trajectory_control_plan,
    render_trajectory_control,
)


CATEGORY = "T8/MiniMax H3/Control/Experimental/Trajectory"
TrajectoryControlIO = io.Custom("H3_T8_TRAJECTORY_CONTROL_PLAN")


class MiniMaxH3TrajectoryControlPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrajectoryControlPlanT8Advanced",
            display_name="MiniMax H3 Trajectory Control Plan / 轨迹控制规划 (Advanced)",
            description=(
                "Builds normalized multi-object bbox trajectories from explicit keyframes. This is "
                "TrailBlazer-inspired creator interaction for H3 control-video conditioning, not a "
                "claim of reproducing TrailBlazer's U-Net attention edits."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "keyframes_json",
                    multiline=True,
                    default=(
                        '[{"frame":0,"object_id":"subject","x":0.10,"y":0.25,'
                        '"width":0.22,"height":0.48,"strength":1.0},'
                        '{"frame":123,"object_id":"subject","x":0.66,"y":0.20,'
                        '"width":0.22,"height":0.48,"strength":1.0}]'
                    ),
                ),
                io.Int.Input("width", default=1152, min=32, max=16384, step=32),
                io.Int.Input("height", default=640, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01),
                io.Combo.Input("easing", options=list(EASING_MODES), default="smoothstep"),
                io.Combo.Input(
                    "clip_policy", options=list(CLIP_POLICIES), default="clip_to_canvas"
                ),
            ],
            outputs=[
                TrajectoryControlIO.Output("trajectory_plan"),
                io.Image.Output("path_preview"),
                io.String.Output("report_json"),
                io.Int.Output("object_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_trajectory_control_plan(**kwargs))


class MiniMaxH3TrajectoryControlRenderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrajectoryControlRenderT8Advanced",
            display_name="MiniMax H3 Trajectory Control Render / 轨迹控制视频 (Advanced)",
            description=(
                "Renders the plan as a soft region, box path or reference sprite IMAGE batch for "
                "the existing H3 Fun Control Apply node. It creates no audio and owns no H3 attention."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                TrajectoryControlIO.Input("trajectory_plan"),
                io.Combo.Input("render_mode", options=list(RENDER_MODES), default="reference_sprite"),
                io.Float.Input("feather", default=0.01, min=0.0, max=0.2, step=0.001),
                io.Int.Input("line_width", default=6, min=1, max=128),
                io.Float.Input(
                    "background_level", default=0.0, min=0.0, max=1.0, step=0.01
                ),
                io.Image.Input("reference_images", optional=True),
                io.Mask.Input("reference_masks", optional=True),
            ],
            outputs=[
                io.Image.Output("control_video"),
                io.Mask.Output("trajectory_masks"),
                io.Image.Output("preview"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*render_trajectory_control(**kwargs))


TRAJECTORY_CONTROL_ADVANCED_NODE_CLASSES = [
    MiniMaxH3TrajectoryControlPlanT8Advanced,
    MiniMaxH3TrajectoryControlRenderT8Advanced,
]
