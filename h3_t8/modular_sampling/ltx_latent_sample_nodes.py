"""Append-only LTX learned-refiner sample and completion-audit nodes."""

from comfy_api.latest import io

from .ltx_latent_nodes import BOUNDARY, CATEGORY
from .ltx_latent_sample_stage import (
    audit_completed_ltx_latent_stage, sample_ltx_latent_stage,
)

SAMPLE_PROOF = io.Custom("T8_LTX_LEARNED_SAMPLE_PROOF")


class MiniMaxH3LTXLearnedStageSampleEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXLearnedStageSampleEXPT8",
            display_name="H3→LTX Learned · Sample Separate Refiner (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Execute one native Core LTX sampler call with external noise, guider, "
                        "sampler and sigmas; return live-only evidence of three step callbacks.",
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("latent_image"), BOUNDARY.Input("stage_boundary")],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     SAMPLE_PROOF.Output("sample_proof"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, stage_boundary):
        return io.NodeOutput(*sample_ltx_latent_stage(
            noise, guider, sampler, sigmas, latent_image, stage_boundary))


class MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8",
            display_name="H3→LTX Learned · Audit Completed Refiner (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Recheck the old source-bound candidate contract and require the matching "
                        "live Stage Sample proof. No portable cache or quality claim.",
            inputs=[BOUNDARY.Input("stage_boundary"), io.Latent.Input("h3_latent"),
                    io.Latent.Input("original_h3_av"), io.Latent.Input("ltx_video_latent"),
                    io.String.Input("adapter_report_json"), io.Model.Input("model"),
                    io.Noise.Input("noise"), io.Guider.Input("guider"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.String.Input("setup_report_json"), io.Latent.Input("candidate_latent"),
                    SAMPLE_PROOF.Input("sample_proof")],
            outputs=[io.Latent.Output("candidate_latent"), io.Latent.Output("original_h3_av"),
                     io.Float.Output("duration_seconds"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_boundary, h3_latent, original_h3_av, ltx_video_latent,
                adapter_report_json, model, noise, guider, sampler, sigmas,
                setup_report_json, candidate_latent, sample_proof):
        return io.NodeOutput(*audit_completed_ltx_latent_stage(
            stage_boundary, h3_latent, original_h3_av, ltx_video_latent,
            adapter_report_json, model, noise, guider, sampler, sigmas,
            setup_report_json, candidate_latent, sample_proof))


NODES = [MiniMaxH3LTXLearnedStageSampleEXPT8,
         MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8]
