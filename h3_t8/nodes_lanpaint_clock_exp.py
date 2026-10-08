"""Append-only, explicit external LanPaint H3 clock compatibility node."""
from comfy_api.latest import io

from .lanpaint_clock_exp import sample_lanpaint_clock


class MiniMaxH3LanPaintClockSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LanPaintClockSamplerEXPT8",
            display_name="H3 LanPaint Audio Clock / 音频时钟适配采样 (EXP/T8)",
            category="T8/MiniMax H3/Repair/Experimental", is_experimental=True,
            description="Uses the independently installed GPL LanPaint sampler in an isolated clock namespace. "
                        "Connect existing NOISE/GUIDER/SAMPLER/SIGMAS and LanPaint AV Prepare. "
                        "Does not modify Core, ordinary LanPaint, old workflows or source PCM; "
                        "raw edit and optional composite need separate human review.",
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Int.Input("inner_steps", default=5, min=0, max=100),
                    io.Float.Input("guidance_lambda", default=5.0, min=0.1, max=50.0, step=0.1),
                    io.Float.Input("step_size", default=0.2, min=0.0001, max=1.0, step=0.01),
                    io.Combo.Input("prompt_mode", options=["Image First", "Prompt First"], default="Image First")],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*sample_lanpaint_clock(**kwargs))


NODES = [MiniMaxH3LanPaintClockSamplerEXPT8]
