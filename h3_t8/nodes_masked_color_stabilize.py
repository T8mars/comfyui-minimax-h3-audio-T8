"""Opt-in, visible RGB postprocessing for masked colour flicker."""

from comfy_api.latest import io

from .masked_color_stabilize import stabilize_masked_color


class MiniMaxH3MaskedColorStabilizeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedColorStabilizeEXPT8",
            display_name="H3 Masked Color Stabilize / 局部颜色稳定（显式后处理 EXP/T8）",
            category="T8/MiniMax H3/Repair/Experimental",
            is_experimental=True,
            description="Explicit RGB colour correction after decoding, before optional Composite. "
            "Use the original paired source clip/mask and an authored target at source_reference_frame. "
            "Retains Lab detail residuals subject to gamut clipping; no motion warp, frame blending, "
            "sampling or audio changes. This does not certify the raw model output. Disconnect or set strength=0 to bypass.",
            inputs=[
                io.Image.Input("repaired_frames"),
                io.Image.Input("source_frames"),
                io.Mask.Input("video_mask"),
                io.Image.Input("target_reference"),
                io.Int.Input("source_reference_frame", default=0, min=0),
                io.Int.Input("target_reference_index", default=0, min=0),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("follow_source_lighting", default=True),
                io.Boolean.Input("stabilize_chroma", default=True),
                io.Combo.Input(
                    "detail_source",
                    options=["generated", "source_recolour"],
                    default="generated",
                    optional=True,
                    tooltip="generated keeps model texture and only corrects the whole-mask mean; local flicker can remain. "
                    "source_recolour explicitly keeps the paired original source texture and applies target colour. "
                    "Use only for colour changes, not replacement geometry, material or generated texture.",
                ),
            ],
            outputs=[io.Image.Output("frames"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*stabilize_masked_color(**kwargs))


NODES = [MiniMaxH3MaskedColorStabilizeEXPT8]
