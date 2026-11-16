from comfy_api.latest import io

from .audio_tail_release import (
    carry_audio_prefix,
    setup_audio_tail_release,
    audit_audio_tail,
)


class MiniMaxH3AudioPrefixCarryEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Audio Carry / 生成尾音前缀接续 (EXP/T8)",
            category="T8/MiniMax H3/Audio/Experimental",
            is_experimental=True,
            description="Copies explicitly selected previous AV audio rows into the next target and pins its actual audio mask. "
            "0 rows is off; native H3 uses about40audio rows/second. Does not turn voice-reference tags into a latent pin or edit final PCM.",
            inputs=[
                io.Latent.Input("target_av_latent"),
                io.Latent.Input("previous_av_latent", optional=True),
                io.Int.Input("context_rows", default=0, min=0, max=None),
                io.Int.Input(
                    "previous_end_row",
                    default=0,
                    min=0,
                    max=None,
                    advanced=True,
                    tooltip="0 uses actual source end; otherwise select an explicit end row.",
                ),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(
        cls, target_av_latent, previous_av_latent=None, context_rows=0, previous_end_row=0
    ):
        return io.NodeOutput(
            *carry_audio_prefix(
                target_av_latent, previous_av_latent, context_rows, previous_end_row
            )
        )


class MiniMaxH3AudioTailReleaseEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Audio Tail / 后段释放尾音约束 (EXP/T8)",
            category="T8/MiniMax H3/Audio/Experimental",
            is_experimental=True,
            description="Opt-in release on actual inpaint/audio-condition masks and T8 dual-clock mask update. "
            "Disabled by default. Step is a zero-based actual sigma interval, not a fixed30% recipe. "
            "Joint sampling may also change generated pictures. Does not modify source PCM or promise better voice.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Combo.Input(
                    "mode",
                    options=["disabled", "release_after_step"],
                    default="disabled",
                ),
                io.Int.Input("prefix_rows", default=0, min=0, max=None),
                io.Int.Input("release_step", default=0, min=0, max=None),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, av_latent, mode="disabled", prefix_rows=0, release_step=0):
        return io.NodeOutput(
            *setup_audio_tail_release(model, av_latent, mode, prefix_rows, release_step)
        )


class MiniMaxH3AudioTailAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Audio Tail Audit / 尾音释放实际记录 (EXP/T8)",
            category="T8/MiniMax H3/Audio/Experimental",
            is_experimental=True,
            is_output_node=True,
            inputs=[io.Model.Input("model"), io.Latent.Input("after_sampling")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, after_sampling):
        latent, report = audit_audio_tail(model, after_sampling)
        return io.NodeOutput(latent, report, ui={"text": [report]})

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [
    MiniMaxH3AudioPrefixCarryEXPT8,
    MiniMaxH3AudioTailReleaseEXPT8,
    MiniMaxH3AudioTailAuditEXPT8,
]
