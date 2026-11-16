from comfy_api.latest import io

from .dmad_renoise import prepare_dmad_renoise


class MiniMaxH3DMADRenoiseSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DMADRenoiseSetupEXPT8",
            display_name="H3 DMAD Re-noise / DMAD声画重加噪 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental",
            is_experimental=True,
            description="Explicit flow re-noise recipe, video/audio shift 12/2. "
            "Load your selected adapter upstream. Connect BasicGuider and SamplerCustomAdvanced. "
            "Does not change existing DMAD8/simple examples.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=8, min=1, max=10000),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, av_latent, steps=8):
        return io.NodeOutput(*prepare_dmad_renoise(model, av_latent, steps))


NODES = [MiniMaxH3DMADRenoiseSetupEXPT8]
