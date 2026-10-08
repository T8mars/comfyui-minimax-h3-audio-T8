"""Append-only RES descriptor bridge; use existing external EAV/Relay nodes."""
from comfy_api.latest import io

from .res_effects_exp import bind_effects


class MiniMaxH3RESEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RESEffectsBindEXPT8",
            display_name="H3 RES External Effects / 外置效果接线 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description="Adds the exact pre-sampling RES context for existing external Stage EAV. "
                        "Connect paired external Relay MODEL before RES History and its CONDITIONING to BasicGuider. "
                        "Passes the same sampler/sigmas; no sampling, hidden model, or universal reuse approval.",
            inputs=[io.Model.Input("model"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("av_latent")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*bind_effects(**kwargs))


NODES = [MiniMaxH3RESEffectsBindEXPT8]
