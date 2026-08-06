from __future__ import annotations

from comfy_api.latest import io

from .sampling_multirate_exp import setup_multirate_exp_sampling


class MiniMaxH3MultiRateSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiRateSamplerEXPT8",
            display_name="MiniMax H3 Multi-Rate Sampler (EXP/T8)",
            description=(
                "Experimental H3 Euler sampler: video uses macro steps while audio uses more "
                "micro steps. audio_steps is the actual number of full joint H3 DiT calls, so "
                "4/10 costs about 10 model evaluations, not 4. Start with 4/8."
            ),
            category="T8/MiniMax H3/Audio/Experimental",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input(
                    "video_steps",
                    default=4,
                    min=1,
                    max=1000,
                    tooltip="Number of committed video Euler macro updates.",
                ),
                io.Int.Input(
                    "audio_steps",
                    default=8,
                    min=1,
                    max=1000,
                    tooltip=(
                        "Number of audio micro updates and full joint H3 DiT calls. "
                        "Must be at least video_steps."
                    ),
                ),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="sigmas"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, av_latent, video_steps, audio_steps, shift_video, shift_audio):
        return io.NodeOutput(*setup_multirate_exp_sampling(
            model,
            av_latent,
            video_steps,
            audio_steps,
            shift_video,
            shift_audio,
        ))
