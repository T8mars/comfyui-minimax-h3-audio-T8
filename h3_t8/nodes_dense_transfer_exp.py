"""Separate opt-in physical transfer sockets; legacy upscaler widgets unchanged."""
import folder_paths
from comfy_api.latest import io

from .dense_transfer_exp import DenseTransferProvider, upscale_dense_av
from .learned_latent_upscale_advanced import PRECISIONS, RELEASE_POLICIES

PROVIDER = io.Custom("H3_LATENT_UPSCALER")
CATEGORY = "T8/MiniMax H3/Latent/Experimental"


class MiniMaxH3DenseTransferProviderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DenseTransferProviderEXPT8",
            display_name="MiniMax H3 Dense Transfer Provider / 连续坐标放大器 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="Explicit API2 physical patch-center provider using existing T8 learned weights. "
                        "Ordinary clean-video calls retain half-pixel interpolation. Does not install Flow or "
                        "change any old upscaler. Trained continuity is not yet qualified.",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
                    io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                    io.Combo.Input("release_policy", options=list(RELEASE_POLICIES), default="offload_after")],
            outputs=[PROVIDER.Output("learned_upscaler")])

    @classmethod
    def execute(cls, model_name, precision, release_policy):
        return io.NodeOutput(DenseTransferProvider(model_name, precision, release_policy))


class MiniMaxH3DenseLearnedUpscaleEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DenseLearnedUpscaleEXPT8",
            display_name="MiniMax H3 Dense Learned Upscale / 分离连续坐标放大 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="Physical H3 dense-cell transport between learned encoder/decoder; full temporal "
                        "network execution, original joint audio retained. Explicitly different coordinates "
                        "from the ordinary trained half-pixel path; not a universal quality fix.",
            inputs=[io.Latent.Input("av_latent"), PROVIDER.Input("learned_upscaler"),
                    io.Float.Input("scale_by", default=1.2, min=1.0, max=4.0, step=.01)],
            outputs=[io.Latent.Output("av_latent"), io.Int.Output("width"), io.Int.Output("height"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, learned_upscaler, scale_by):
        return io.NodeOutput(*upscale_dense_av(av_latent=av_latent, provider=learned_upscaler, scale_by=scale_by))


NODES = [MiniMaxH3DenseTransferProviderEXPT8, MiniMaxH3DenseLearnedUpscaleEXPT8]
