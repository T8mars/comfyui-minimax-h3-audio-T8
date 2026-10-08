"""Append-only RES completed-result sampler/load nodes; legacy schemas unchanged."""
from comfy_api.latest import io

from .modular_sampling.nodes import _stage_store_root
from .modular_sampling.storage import fingerprint_stage, load_stage
from .res_stage_exp import STAGES, sample_res_stage


class MiniMaxH3RESStageSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RESStageSamplerEXPT8",
            display_name="H3 RES Complete Stage / RES完成态 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description="One actual native RES History sampler invocation. Keeps the original five inputs "
                        "and two AV outputs, adds observed completion for existing Stage Save. Only a "
                        "recognized exact execution can authorize explicit frozen-result reuse. "
                        "Unknown MODEL patches remain runnable, unqualified for persistent reuse. "
                        "Not partial x0, LOW/HIGH handoff, universal CUDA or quality approval.",
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image")],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.Custom("T8_STAGE_RESULT").Output("stage_result"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*sample_res_stage(**kwargs))


class MiniMaxH3RESStageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RESStageLoadEXPT8",
            display_name="H3 RES Complete Stage / 读取完成态 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description="Read-only explicit completed RES artifact by exact path/SHA. No MODEL load, "
                        "sampling, history resume or assertion that current settings match the frozen result.",
            inputs=[io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default=""),
                    io.Combo.Input("expected_stage", options=list(STAGES), default=STAGES[0])],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
                     io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256, expected_stage=STAGES[0]):
        return io.NodeOutput(*load_stage(_stage_store_root(), artifact_path, artifact_sha256, expected_stage))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, expected_stage=STAGES[0]):
        try:
            return fingerprint_stage(_stage_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3RESStageSamplerEXPT8, MiniMaxH3RESStageLoadEXPT8]
