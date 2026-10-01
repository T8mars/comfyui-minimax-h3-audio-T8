"""External LTX effects appended without changing existing node schemas."""
from comfy_api.latest import io

from .eav import CONFIG_TYPE
from .ltx_eav import RUNTIME_TYPE, apply_ltx_eav, audit_ltx_eav

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


class MiniMaxH3LTXEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="LTX Stage EAV · External Apply (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Connect after LTX Setup/LoRA, before Guider and Stage Bind. Reuse Stage EAV Config. "
                        "Delegates existing video attention, STG, masks, gates and backend; no audio patch. "
                        "Inspect downstream coverage. Not a sampler, quality or portable cache receipt.",
            inputs=[io.Model.Input("model"), io.Latent.Input("ltx_latent"), io.Sigmas.Input("sigmas"),
                    io.Custom(CONFIG_TYPE).Input("eav_config")],
            outputs=[io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, ltx_latent, sigmas, eav_config):
        return io.NodeOutput(*apply_ltx_eav(model, ltx_latent, sigmas, eav_config))


class MiniMaxH3LTXEAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="LTX Stage EAV · Observe Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Pass candidate LATENT through unchanged; report actual measured/applied blocks, "
                        "STG/window skips and bypassed hooks. Not proof of candidate sampling provenance.",
            inputs=[io.Model.Input("model"), io.Latent.Input("candidate_latent"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("candidate_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, candidate_latent, runtime):
        return io.NodeOutput(*audit_ltx_eav(model, candidate_latent, runtime))


NODES = [MiniMaxH3LTXEAVApplyEXPT8, MiniMaxH3LTXEAVAuditEXPT8]
