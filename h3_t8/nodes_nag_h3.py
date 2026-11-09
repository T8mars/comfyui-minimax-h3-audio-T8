import json
from comfy_api.latest import io

from .nag_h3 import apply_nag, NAG_WRAPPER_KEY


class MiniMaxH3NAGEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3NAGEXPT8", display_name="H3 NAG · Negative Attention / 负向注意力 (T8 EXP)",
            category="T8/MiniMax H3/Guidance/Experimental", is_experimental=True,
            description="Negative attention guidance works at CFG=1. Enabled NAG with an encoded empty prompt "
                        "is different from disabling it. Adds a dense negative attention pass in each active block. "
                        "The selected forward, head groups and positive attention override are preserved. "
                        "A custom producer must expose the optimized-attention/output-projection interface. "
                        "Existing model and LoRA selection is preserved; disabling this node leaves upstream patches intact.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("nag_negative"),
                    io.Boolean.Input("enabled", default=False),
                    io.Float.Input("scale", default=5., min=0., max=None),
                    io.Float.Input("tau", default=2.5, min=.001, max=None),
                    io.Float.Input("alpha", default=.25, min=0., max=1.),
                    io.Float.Input("sigma_end", default=0., min=0., max=None)],
            outputs=[io.Model.Output("model"), io.Custom("T8_NAG_RUNTIME").Output("runtime"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, nag_negative, enabled=False, scale=5., tau=2.5, alpha=.25, sigma_end=0.):
        selected = apply_nag(model, nag_negative, scale, tau, alpha, sigma_end) if enabled else model
        runtime = selected.get_attachment(NAG_WRAPPER_KEY)
        retained = selected is model and runtime is not None
        report = dict(status=("identity_bypass_existing_nag_retained" if retained else "identity_bypass")
                             if selected is model else "configured_not_sampled",
                      enabled=enabled, scale=scale, tau=tau, alpha=alpha, sigma_end=sigma_end,
                      upstream_nag_retained=retained,
                      original_model_modified=False, quality_verified=False)
        return io.NodeOutput(selected, runtime, json.dumps(report, ensure_ascii=False))


class MiniMaxH3NAGAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3NAGAuditEXPT8", display_name="H3 NAG · Actual Calls / 实际调用 (T8 EXP)",
            category="T8/MiniMax H3/Guidance/Experimental", is_experimental=True,
            inputs=[io.Latent.Input("av_latent"), io.Custom("T8_NAG_RUNTIME").Input("runtime")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, runtime):
        report = {"status":"identity_bypass", "guided_blocks":0} if runtime is None else runtime.snapshot()
        return io.NodeOutput(av_latent, json.dumps(report, ensure_ascii=False))


NODES = [MiniMaxH3NAGEXPT8, MiniMaxH3NAGAuditEXPT8]
