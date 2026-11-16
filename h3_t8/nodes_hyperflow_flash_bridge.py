from comfy_api.latest import io
import folder_paths

from .hyperflow_flash_bridge import load_flash, flash_schedule


class MiniMaxH3HyperFlowFlashLoaderT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3HyperFlowFlashLoaderT8",
                         display_name="H3 HyperFlow Flash Loader / Flash独立加载 (EXP/T8)",
                         category="T8/MiniMax H3/HyperFlow/Experimental", is_experimental=True,
                         description="Uses the official ComfyUI-HyperFlow-Flash checkpoint consumer. "
                                     "A separate dependency; preserves its factorized layers and two-time endpoints. "
                                     "The existing full-model HyperFlow LoRA loader is unchanged.",
                         inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                                 io.Combo.Input("weights_in", options=["auto", "ram", "disk"])],
                         outputs=[io.Model.Output("model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model_name, weights_in="auto"):
        return io.NodeOutput(*load_flash(model_name, weights_in))


class MiniMaxH3FlashCheckpointSigmasT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FlashCheckpointSigmasT8",
                         display_name="H3 Flash Checkpoint Sigmas / Flash权重内步表 (EXP/T8)",
                         category="T8/MiniMax H3/HyperFlow/Experimental", is_experimental=True,
                         description="Returns the loaded checkpoint's actual schedule without another shift. "
                                     "You may instead connect your own SIGMAS directly to the sampler.",
                         inputs=[io.Model.Input("model"),
                                 io.Combo.Input("schedule", options=["checkpoint", "4", "6", "8"],
                                                default="checkpoint", optional=True)],
                         outputs=[io.Sigmas.Output("sigmas")])

    @classmethod
    def execute(cls, model, schedule="checkpoint"):
        return io.NodeOutput(flash_schedule(model, schedule))


NODES = [MiniMaxH3HyperFlowFlashLoaderT8, MiniMaxH3FlashCheckpointSigmasT8]
