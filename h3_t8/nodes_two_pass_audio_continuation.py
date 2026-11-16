from comfy_api.latest import io

from .two_pass_audio_continuation import continue_audio


class MiniMaxH3TwoPassAudioContinueEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Two-Pass Audio / 双采声音连续续采 (EXP/T8)",
            category="T8/MiniMax H3/Audio/Experimental", is_experimental=True,
            description="Explicit opt-in: use LOW sampler output socket0 (not denoised_output) "
            "for unfinished audio; retain HIGH visual latent and selected visual noise. "
            "No new initial audio noise or extra audio freeze/steps; sampler-internal random noise is unchanged. LOW/HIGH sigma boundary and "
            "audio clocks must match. Existing workflows remain unchanged. Not a quality guarantee.",
            inputs=[io.Latent.Input("coarse_output"), io.Latent.Input("high_latent"),
                    io.Noise.Input("visual_noise"), io.Sigmas.Input("low_sigmas"),
                    io.Sigmas.Input("high_sigmas"),
                    io.Sampler.Input("coarse_sampler", optional=True,
                        tooltip="Connect LOW's T8 Dual-Clock sampler to normalize its exported partial audio. Native FLOW_AV/Euler needs no correction.")],
            outputs=[io.Latent.Output("av_latent"), io.Noise.Output("noise"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, coarse_output, high_latent, visual_noise, low_sigmas, high_sigmas,
                coarse_sampler=None):
        return io.NodeOutput(*continue_audio(coarse_output, high_latent, visual_noise,
                                            low_sigmas, high_sigmas, coarse_sampler))


NODES = [MiniMaxH3TwoPassAudioContinueEXPT8]
