from __future__ import annotations

from comfy_api.latest import io

from .raven_streaming_advanced import (
    audit_raven_streaming_request,
    build_raven_streaming_profile,
    load_raven_model_guarded,
)


CATEGORY = "T8/MiniMax H3/RAVEN Streaming/Experimental"


def _filenames(folder: str) -> list[str]:
    try:
        import folder_paths

        values = list(folder_paths.get_filename_list(folder))
    except Exception:  # noqa: BLE001 - an empty install must still expose the schema
        values = []
    return values or ["<missing>"]


def _preferred(values: list[str], tokens: tuple[str, ...]) -> str:
    for value in values:
        lowered = value.casefold()
        if all(token in lowered for token in tokens):
            return value
    return values[0]


class MiniMaxH3RavenStreamingProfileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RavenStreamingProfileT8Advanced",
            display_name=(
                "MiniMax H3 RAVEN Streaming Profile / 流式参数配置 " "(Advanced EXP/T8)"
            ),
            description=(
                "Emits one synchronized RAVEN parameter set for both the T8 audit and the "
                "external RAVEN Streaming Sampler. The default is the published 4-NFE profile."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=["published_preview_4nfe", "manual_experimental"],
                    default="published_preview_4nfe",
                    tooltip=(
                        "Published mode fixes 4 steps, shifts 12/3, sink/window 2/2 and "
                        "CPU-pinned KV. Manual mode is forwarded exactly but is unreviewed."
                    ),
                ),
                io.Int.Input(
                    "manual_steps",
                    default=4,
                    min=1,
                    max=100,
                    step=1,
                    advanced=True,
                ),
                io.Float.Input(
                    "manual_video_shift",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "manual_audio_shift",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "manual_sink",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    advanced=True,
                ),
                io.Int.Input(
                    "manual_window",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                ),
                io.Combo.Input(
                    "manual_kv_cache_storage",
                    options=["cpu_pinned", "cpu", "gpu"],
                    default="cpu_pinned",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Int.Output("steps"),
                io.Float.Output("video_shift"),
                io.Float.Output("audio_shift"),
                io.Int.Output("sink"),
                io.Int.Output("window"),
                io.Combo.Output("kv_cache_storage"),
                io.String.Output("profile_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_raven_streaming_profile(**kwargs))


class MiniMaxH3RavenGuardedLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        diffusion_models = _filenames("diffusion_models")
        loras = _filenames("loras")
        return io.Schema(
            node_id="MiniMaxH3RavenGuardedLoaderT8Advanced",
            display_name=(
                "MiniMax H3 RAVEN Guarded Loader / 受保护加载器 " "(Advanced EXP/T8)"
            ),
            description=(
                "Preflights the separately installed MIT RAVEN ComfyUI plugin, full BF16 H3 "
                "base, mandatory RAVEN LoRA and reviewed memory envelope before delegating to "
                "the external loader. No upstream runtime source is copied here."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "unet_name",
                    options=diffusion_models,
                    default=_preferred(diffusion_models, ("minimax", "h3")),
                    tooltip=(
                        "Must be the full, non-pruned, non-quantized BF16 H3 diffusion model. "
                        "INT8/FP8/NVFP4/ConvRot checkpoints are incompatible with v0.1."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=loras,
                    default=_preferred(loras, ("raven",)),
                    tooltip="Mandatory MiniMax H3 RAVEN Streaming LoRA; strength stays 1.0.",
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "bf16", "fp32"],
                    default="default",
                ),
                io.Combo.Input(
                    "enforcement",
                    options=[
                        "block_outside_reviewed_envelope",
                        "block_mechanical_conflicts",
                        "report_only",
                    ],
                    default="block_outside_reviewed_envelope",
                    tooltip=(
                        "Default blocks before loading unless plugin/model/CUDA/BF16 and the "
                        "reviewed ~24GiB GPU + 192GiB host-memory envelope pass. report_only "
                        "still loads the model and is explicitly high risk."
                    ),
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*load_raven_model_guarded(**kwargs))


class MiniMaxH3RavenRequestAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RavenRequestAuditT8Advanced",
            display_name=(
                "MiniMax H3 RAVEN Request Audit / 流式请求审计 " "(Advanced EXP/T8)"
            ),
            description=(
                "Delegates strict T2VA conditioning, empty-AV-latent and causal-MODEL checks "
                "to the external runtime, verifies the mandatory 266-module adapter and blocks "
                "unreviewed model/attention patches before the external streaming sampler."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Int.Input("steps", default=4, min=1, max=100, step=1),
                io.Float.Input(
                    "video_shift", default=12.0, min=0.01, max=100.0, step=0.01
                ),
                io.Float.Input(
                    "audio_shift", default=3.0, min=0.01, max=100.0, step=0.01
                ),
                io.Int.Input("sink", default=2, min=0, max=64, step=1),
                io.Int.Input("window", default=2, min=1, max=64, step=1),
                io.Combo.Input(
                    "kv_cache_storage",
                    options=["cpu_pinned", "cpu", "gpu"],
                    default="cpu_pinned",
                ),
                io.Boolean.Input(
                    "allow_experimental_over_192",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "RAVEN supports the 17k+5 grid to 362 frames, but requests above 192 "
                        "frames are explicitly experimental and require this acknowledgement."
                    ),
                ),
                io.Combo.Input(
                    "enforcement",
                    options=[
                        "block_outside_reviewed_envelope",
                        "block_mechanical_conflicts",
                        "report_only",
                    ],
                    default="block_outside_reviewed_envelope",
                    tooltip=(
                        "Mechanical conflicts include references/keyframes, non-empty latent, "
                        "masks, non-causal MODEL, missing adapter and external object patches."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
                io.Boolean.Output("compatible"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*audit_raven_streaming_request(**kwargs))


RAVEN_STREAMING_ADVANCED_NODE_CLASSES = [
    MiniMaxH3RavenStreamingProfileT8Advanced,
    MiniMaxH3RavenGuardedLoaderT8Advanced,
    MiniMaxH3RavenRequestAuditT8Advanced,
]
