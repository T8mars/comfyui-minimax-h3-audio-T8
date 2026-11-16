from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .ic_draft_refine import ic_recipe, compose_ic_target, setup_ic_sampler


folder_paths.add_model_folder_path(
    "h3_ic", str(Path(folder_paths.models_dir) / "h3_ic")
)


class MiniMaxH3ICRecipeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ICRecipeEXPT8",
            display_name="H3 IC Recipe / 草稿参考细化配方 (EXP/T8)",
            category="T8/MiniMax H3/Latent/Experimental",
            is_experimental=True,
            description="IC prompt and already-shifted sigma presets. Connect the actual draft video as Video 1. "
            "Load your selected Ref2VA/IC weights upstream; no automatic model substitution.",
            inputs=[
                io.String.Input("caption", multiline=True),
                io.Combo.Input("preset", options=["up3", "up4", "hf8"]),
                io.String.Input("custom_sigmas", default="", advanced=True),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, caption, preset="up3", custom_sigmas=""):
        return io.NodeOutput(*ic_recipe(caption, preset, custom_sigmas))


class MiniMaxH3ICComposeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ICComposeEXPT8",
            display_name="H3 IC Target / IC静音条件合成 (EXP/T8)",
            category="T8/MiniMax H3/Latent/Experimental",
            is_experimental=True,
            description="Upscaled video + selected silence asset. Preserves the target metadata/video mask; "
            "fixed audio rows are conditioning, mux original draft audio separately. "
            "For source124f silence use a matching124f target; fitting is explicit experimental behavior.",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("reference_latent"),
                io.Combo.Input(
                    "silence_name", options=folder_paths.get_filename_list("h3_ic")
                ),
                io.Combo.Input("audio_fit", options=["strict", "trim_or_zero_pad_exp"]),
                io.Boolean.Input("pin_audio", default=True),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(
        cls,
        video_latent,
        reference_latent,
        silence_name,
        audio_fit="strict",
        pin_audio=True,
    ):
        path = folder_paths.get_full_path_or_raise("h3_ic", silence_name)
        return io.NodeOutput(
            *compose_ic_target(
                video_latent, reference_latent, path, audio_fit, pin_audio
            )
        )


class MiniMaxH3ICSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ICSamplerEXPT8",
            display_name="H3 IC Sampler / IC双时间细化采样 (EXP/T8)",
            category="T8/MiniMax H3/Latent/Experimental",
            is_experimental=True,
            description="Connect the dedicated HyperFlow MODEL with selected Sharp/IC patches and composed target. "
            "up3/up4 re-noise the clean upscaled draft at the source tail sigma; hf8 starts from noise. "
            "The actual two-time endpoint is retained, not replaced by an ordinary HF LoRA.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Combo.Input("preset", options=["up3", "up4", "hf8"], default="up3"),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, av_latent, preset="up3"):
        return io.NodeOutput(*setup_ic_sampler(model, av_latent, preset))


NODES = [MiniMaxH3ICRecipeEXPT8, MiniMaxH3ICComposeEXPT8]
SAMPLER_NODES = [MiniMaxH3ICSamplerEXPT8]
