from comfy_api.latest import io
import folder_paths

from .latent_upscale_2d import upscale_2d


class MiniMaxH3LatentUpscale2DT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentUpscale2DT8",
            display_name="H3 2D Latent Upscale / 六档潜空间放大 (EXP/T8)",
            description="Optional learned 2D visual resize. Preserves joint audio, masks and metadata. "
            "Sharpness tiers are model conditions; higher tiers do not guarantee better motion.",
            category="T8/MiniMax H3/Latent/Experimental",
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                ),
                io.Int.Input("sharpness", default=1, min=1, max=6),
                io.Combo.Input(
                    "size_mode",
                    options=["scale_by", "target_megapixels", "target_dimensions"],
                ),
                io.Float.Input("scale_by", default=2.0, min=0.01, max=None, step=0.01),
                io.Float.Input(
                    "target_megapixels", default=0.6, min=0.01, max=None, step=0.01
                ),
                io.Int.Input("target_width", default=1024, min=32, max=None, step=32),
                io.Int.Input("target_height", default=576, min=32, max=None, step=32),
                io.Combo.Input(
                    "aspect_policy", options=["preserve_source", "honor_dimensions_exp"]
                ),
                io.Float.Input(
                    "max_anisotropy", default=1.05, min=1.0, max=None, advanced=True
                ),
                io.Combo.Input("precision", options=["bf16", "fp16", "fp32"]),
                io.Int.Input(
                    "chunk_frames",
                    default=8,
                    min=0,
                    max=None,
                    advanced=True,
                    tooltip="Latent frames per chunk; 0 runs the full video. Full temporal halo is added.",
                ),
                io.Combo.Input(
                    "release_policy",
                    options=["offload_after", "clear_after", "keep_loaded"],
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, av_latent, **kwargs):
        return io.NodeOutput(*upscale_2d(av_latent, **kwargs))


NODES = [MiniMaxH3LatentUpscale2DT8]
