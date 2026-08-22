from __future__ import annotations

from comfy_api.latest import io

from .lanpaint_av_advanced import composite_lanpaint_av, prepare_lanpaint_av_latent


CATEGORY = "T8/MiniMax H3/Repair/Experimental"


class MiniMaxH3LanPaintAVPrepareT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LanPaintAVPrepareT8Advanced",
            display_name="MiniMax H3 LanPaint AV Prepare / 音画局部重绘准备 (Advanced)",
            description=(
                "Encodes source frames and audio once and attaches exact nested video/audio "
                "noise masks for the upstream LanPaint KSampler Advanced. The external "
                "GPL LanPaint sampler is intentionally not duplicated in this node pack."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Audio.Input("source_audio"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input(
                    "audio_intervals",
                    multiline=True,
                    default="[]",
                    tooltip='JSON，例如 [{"start":1.2,"end":2.0}]；仅这些时段重绘音频。',
                ),
                io.Combo.Input(
                    "frame_policy",
                    options=["strict", "trim_down"],
                    default="strict",
                    tooltip="strict要求17n+5；trim_down只裁掉尾部不足一个网格的帧。",
                ),
                io.Boolean.Input(
                    "require_lanpaint_sampler",
                    default=True,
                    tooltip="未安装上游LanPaint时提前报错，避免误用普通采样器。",
                ),
                io.Mask.Input("video_mask", optional=True),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Image.Output("source_frames"),
                io.Audio.Output("source_audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*prepare_lanpaint_av_latent(**kwargs))


class MiniMaxH3LanPaintAVCompositeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LanPaintAVCompositeT8Advanced",
            display_name="MiniMax H3 LanPaint AV Composite / 原片音画回贴 (Advanced)",
            description=(
                "Blends repaired pixels only inside the video mask and repaired audio only "
                "inside declared intervals, preserving the original soundtrack elsewhere."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("repaired_frames"),
                io.Audio.Input("source_audio"),
                io.Audio.Input("repaired_audio"),
                io.String.Input("audio_intervals", multiline=True, default="[]"),
                io.Int.Input("video_blend_pixels", default=11, min=1, max=101, step=2),
                io.Float.Input(
                    "audio_crossfade_seconds",
                    default=0.02,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                ),
                io.Mask.Input("video_mask", optional=True),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*composite_lanpaint_av(**kwargs))


LANPAINT_AV_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LanPaintAVPrepareT8Advanced,
    MiniMaxH3LanPaintAVCompositeT8Advanced,
]
