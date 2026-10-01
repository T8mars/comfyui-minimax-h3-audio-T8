"""Opt-in boundaries around the existing H3 Super RGB -> LTX sampler."""
from comfy_api.latest import io

from .ltx_rgb_stage import bind_ltx_rgb_stage, audit_ltx_rgb_stage

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
BOUNDARY = io.Custom("T8_LTX_RGB_STAGE_BOUNDARY")


class MiniMaxH3LTXRGBStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXRGBStageBindEXPT8",
            display_name="H3→LTX RGB · Bind Separate Refiner (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Bind H3 RGB preparation, original audio bypass, learned LTX lift and the "
                        "external refiner MODEL/NOISE/GUIDER/SAMPLER/SIGMAS. No sampling or cache claim.",
            inputs=[io.Image.Input("source_frames"), io.Audio.Input("source_audio"),
                    io.Image.Input("prepared_frames"), io.String.Input("prep_report_json"),
                    io.Latent.Input("ltx_latent"), io.Model.Input("model"),
                    io.Noise.Input("noise"), io.Guider.Input("guider"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.String.Input("setup_report_json")],
            outputs=[io.Noise.Output("noise"), io.Guider.Output("guider"),
                     io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Latent.Output("ltx_latent"), BOUNDARY.Output("stage_boundary"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, source_frames, source_audio, prepared_frames, prep_report_json,
                ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json):
        return io.NodeOutput(*bind_ltx_rgb_stage(source_frames, source_audio, prepared_frames,
            prep_report_json, ltx_latent, model, noise, guider, sampler, sigmas, setup_report_json))


class MiniMaxH3LTXRGBStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXRGBStageAuditEXPT8",
            display_name="H3→LTX RGB · Audit Refiner Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Recheck current H3 source/audio, RGB→LTX handoff and connected sampler controls "
                        "before separate TAEHV decode. Pass the original H3 AUDIO object through unchanged; "
                        "this is not a portable receipt or a quality decision.",
            inputs=[BOUNDARY.Input("stage_boundary"), io.Image.Input("source_frames"),
                    io.Audio.Input("source_audio"), io.Image.Input("prepared_frames"),
                    io.String.Input("prep_report_json"), io.Latent.Input("ltx_latent"),
                    io.Model.Input("model"), io.Noise.Input("noise"),
                    io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.String.Input("setup_report_json"),
                    io.Latent.Input("candidate_latent")],
            outputs=[io.Latent.Output("candidate_latent"), io.Audio.Output("source_audio"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_boundary, source_frames, source_audio, prepared_frames,
                prep_report_json, ltx_latent, model, noise, guider, sampler,
                sigmas, setup_report_json, candidate_latent):
        return io.NodeOutput(*audit_ltx_rgb_stage(stage_boundary, source_frames, source_audio,
            prepared_frames, prep_report_json, ltx_latent, model, noise, guider,
            sampler, sigmas, setup_report_json, candidate_latent))


NODES = [MiniMaxH3LTXRGBStageBindEXPT8, MiniMaxH3LTXRGBStageAuditEXPT8]
