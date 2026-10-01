"""Public opt-in learned H3 LATENT -> external LTX refiner boundaries."""

from comfy_api.latest import io

from .ltx_latent_stage import (
    audit_ltx_latent_stage, bind_ltx_latent_stage, decode_original_h3_audio,
)

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
BOUNDARY = io.Custom("T8_LTX_LEARNED_STAGE_BOUNDARY")


class MiniMaxH3LTXLearnedStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXLearnedStageBindEXPT8",
            display_name="H3→LTX Learned · Bind Separate Refiner (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Bind the actual learned H3 video-latent adapter output and original H3 AV "
                        "to an external LTX MODEL/NOISE/GUIDER/SAMPLER/SIGMAS. No RGB bridge, "
                        "second upscaler, sampling or portable cache claim.",
            inputs=[io.Latent.Input("h3_latent"), io.Latent.Input("original_h3_av"),
                    io.Latent.Input("ltx_video_latent"), io.String.Input("adapter_report_json"),
                    io.Model.Input("model"), io.Noise.Input("noise"),
                    io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.String.Input("setup_report_json")],
            outputs=[io.Noise.Output("noise"), io.Guider.Output("guider"),
                     io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Latent.Output("ltx_video_latent"), BOUNDARY.Output("stage_boundary"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, h3_latent, original_h3_av, ltx_video_latent, adapter_report_json,
                model, noise, guider, sampler, sigmas, setup_report_json):
        return io.NodeOutput(*bind_ltx_latent_stage(h3_latent, original_h3_av,
            ltx_video_latent, adapter_report_json, model, noise, guider, sampler,
            sigmas, setup_report_json))


class MiniMaxH3LTXLearnedStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXLearnedStageAuditEXPT8",
            display_name="H3→LTX Learned · Audit Refiner Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Recheck source H3 AV, learned adapter geometry and connected LTX sampler "
                        "before separate video/audio decode. Does not prove external sampler execution.",
            inputs=[BOUNDARY.Input("stage_boundary"), io.Latent.Input("h3_latent"),
                    io.Latent.Input("original_h3_av"), io.Latent.Input("ltx_video_latent"),
                    io.String.Input("adapter_report_json"), io.Model.Input("model"),
                    io.Noise.Input("noise"), io.Guider.Input("guider"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.String.Input("setup_report_json"), io.Latent.Input("candidate_latent")],
            outputs=[io.Latent.Output("candidate_latent"), io.Latent.Output("original_h3_av"),
                     io.Float.Output("duration_seconds"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_boundary, h3_latent, original_h3_av, ltx_video_latent,
                adapter_report_json, model, noise, guider, sampler, sigmas,
                setup_report_json, candidate_latent):
        return io.NodeOutput(*audit_ltx_latent_stage(stage_boundary, h3_latent,
            original_h3_av, ltx_video_latent, adapter_report_json, model, noise,
            guider, sampler, sigmas, setup_report_json, candidate_latent))


class MiniMaxH3LTXOriginalAudioDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXOriginalAudioDecodeEXPT8",
            display_name="H3→LTX Learned · Decode Original H3 Audio (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Decode only the original H3 joint-AV audio latent with the H3 Audio VAE. "
                        "Never convert it to LTX audio or decode the H3 video again. "
                        "Video-only H3 inputs fail explicitly.",
            inputs=[io.Latent.Input("original_h3_av"),
                    io.String.Input("adapter_report_json"), io.Vae.Input("audio_vae")],
            outputs=[io.Audio.Output("original_h3_audio"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, original_h3_av, adapter_report_json, audio_vae):
        return io.NodeOutput(*decode_original_h3_audio(original_h3_av, adapter_report_json, audio_vae))


NODES = [MiniMaxH3LTXLearnedStageBindEXPT8,
         MiniMaxH3LTXLearnedStageAuditEXPT8,
         MiniMaxH3LTXOriginalAudioDecodeEXPT8]
