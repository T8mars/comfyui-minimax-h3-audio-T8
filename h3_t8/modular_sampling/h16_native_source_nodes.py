"""Opt-in verified native-checkpoint source for H16-only resume graphs."""

from comfy_api.latest import io

from .h16_native_source import verified_h16_native_source


CATEGORY = "T8/MiniMax H3/Modular Sampling/H16 Experimental"


class MiniMaxH3H16VerifiedNativeSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H16-3 · Verified Native Resume Source (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Require a native AV checkpoint's exact external manifest and file SHA "
                        "before removing only its volatile Load provenance for H16 window identity. "
                        "Never samples or decodes media.",
            inputs=[io.Latent.Input("av_latent"),
                    io.Boolean.Input("resume_verified"),
                    io.String.Input("checkpoint_id"),
                    io.String.Input("content_sha256"),
                    io.String.Input("file_sha256"),
                    io.String.Input("manifest_json"),
                    io.String.Input("report_json")],
            outputs=[io.Latent.Output("source_av_latent"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, av_latent, resume_verified, checkpoint_id,
                content_sha256, file_sha256, manifest_json, report_json):
        return io.NodeOutput(*verified_h16_native_source(
            av_latent, resume_verified, checkpoint_id, content_sha256,
            file_sha256, manifest_json, report_json,
        ))
