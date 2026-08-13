from __future__ import annotations

from comfy_api.latest import io

from .av_decode_safety_advanced import ENFORCEMENT, MODES, decode_av_safely


CATEGORY = "T8/MiniMax H3/Audio/Experimental"


class MiniMaxH3AVDecodeSafetyT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVDecodeSafetyT8Advanced",
            display_name="MiniMax H3 AV Decode Safety / 音视频安全解码 (T8 Advanced)",
            description=(
                "Isolated AV decode preflight with explicit H3 latent/VAE contracts, output-size "
                "estimates and current headroom gates. preflight_only is the default and performs "
                "no decode. Current H3 regular decode also uses internal 256-pixel spatial tiles "
                "on larger canvases; explicit tile controls may be ignored and remain experimental."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Combo.Input("mode", options=list(MODES), default="preflight_only"),
                io.Float.Input(
                    "minimum_current_headroom_mib",
                    default=512.0,
                    min=0.0,
                    max=65536.0,
                    step=16.0,
                ),
                io.Float.Input(
                    "maximum_estimated_output_mib",
                    default=8192.0,
                    min=1.0,
                    max=131072.0,
                    step=64.0,
                ),
                io.Combo.Input(
                    "enforcement",
                    options=list(ENFORCEMENT),
                    default="report_only",
                ),
                io.Int.Input(
                    "video_tile_size",
                    default=32,
                    min=2,
                    max=256,
                    step=1,
                    advanced=True,
                ),
                io.Int.Input(
                    "video_tile_overlap",
                    default=8,
                    min=0,
                    max=128,
                    step=1,
                    advanced=True,
                ),
                io.Int.Input(
                    "video_tile_temporal",
                    default=999,
                    min=2,
                    max=4096,
                    step=1,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("generated_audio"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        video_vae,
        audio_vae,
        mode,
        minimum_current_headroom_mib,
        maximum_estimated_output_mib,
        enforcement,
        video_tile_size,
        video_tile_overlap,
        video_tile_temporal,
    ):
        return io.NodeOutput(*decode_av_safely(
            av_latent,
            video_vae,
            audio_vae,
            mode,
            minimum_current_headroom_mib,
            maximum_estimated_output_mib,
            enforcement,
            video_tile_size,
            video_tile_overlap,
            video_tile_temporal,
        ))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


AV_DECODE_SAFETY_ADVANCED_NODE_CLASSES = [MiniMaxH3AVDecodeSafetyT8Advanced]
