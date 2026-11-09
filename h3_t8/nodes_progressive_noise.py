"""Append-only paired noise preparation for existing modular samplers."""
from comfy_api.latest import io
from .progressive_noise import MODES, prepare_paired_noise


class MiniMaxH3ProgressivePairedNoiseEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Progressive · Paired Noise / 高低分关联噪声 (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Progressive Experimental",
            is_experimental=True,
            description="给现有 LOW Sampler 和 HIGH Handoff 提供噪声，不运行采样或放大。"
                        "independent 保留旧 LOW seed/HIGH seed+1；haar 让半尺寸 LOW 与 HIGH 共用空间噪声。"
                        "音频不缩放，但关联模式的随机序列与旧模式不同；不保证画质更好。",
            inputs=[io.Latent.Input("low_source"), io.Latent.Input("high_source"),
                    io.Int.Input("seed", default=1234, min=0, max=2**64-1, control_after_generate=True),
                    io.Combo.Input("mode", options=list(MODES), default="independent",
                                   tooltip="haar 要求 LOW 宽高严格各为 HIGH 一半；尺寸不符不会静默回退。")],
            outputs=[io.Noise.Output("low_noise"), io.Noise.Output("high_video_noise"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_source, high_source, seed, mode="independent"):
        return io.NodeOutput(*prepare_paired_noise(low_source, high_source, seed, mode))


NODES = [MiniMaxH3ProgressivePairedNoiseEXPT8]
