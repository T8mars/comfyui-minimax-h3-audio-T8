from comfy_api.latest import io

from .modular_sampling.nodes import _stage_store_root
from .modular_sampling.reuse import MODES
from .modular_sampling.automatic_reuse import sample, ui_report


class MiniMaxH3StageReuseEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageReuseEXPT8",
            display_name="H3 Stage · Sample or Reuse / 采样或复用 (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Opt-in verified reuse of a completed Stage. Checks actual MODEL, ordered LoRAs, "
                        "conditions, source latent, seed, sigma and implementation. Unknown user stacks still execute. "
                        "Reconnect to Decode/Save to retry delivery without resampling.",
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image"),
                    io.Combo.Input("mode", options=list(MODES), default="disabled"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context", optional=True),
                    io.Combo.Input("stage_role", options=["native_low", "native_high"], default="native_low", optional=True,
                                   advanced=True, tooltip="Only used without stage_context; does not change sampler math."),
                    io.Custom("H3_T8_CREATOR_WORKSPACE").Input("creator_workspace", optional=True, advanced=True),
                    io.Int.Input("run_position", default=0, min=0, optional=True, advanced=True),
                    io.Int.Input("variant_index", default=0, min=0, optional=True, advanced=True)],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, stage_context=None, mode="disabled",
                stage_role="native_low", creator_workspace=None, run_position=0, variant_index=0):
        values = sample(noise, guider, sampler, sigmas, latent_image, _stage_store_root(),
                        mode=mode, stage_context=stage_context, stage_role=stage_role,
                        creator_workspace=creator_workspace, run_position=run_position,
                        variant_index=variant_index)
        return io.NodeOutput(*values, ui={"t8_stage_reuse": [ui_report(values[3])]})

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Recheck external artifacts and actual inputs on every queue request.
        return float("nan")


NODES = [MiniMaxH3StageReuseEXPT8]
