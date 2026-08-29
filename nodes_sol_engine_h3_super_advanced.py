from __future__ import annotations

import comfy.samplers
import folder_paths
from comfy_api.latest import io

from .sol_engine_h3_super_advanced import (
    decode_ltx_latent_with_taehv,
    encode_h3_frames_with_taehv,
    load_taehv_wide,
    prepare_h3_draft_for_ltx_refiner,
    setup_ltx_identity_preserve_refiner,
    setup_ltx_stage2_refiner,
)


CATEGORY = "T8/MiniMax H3/Acceleration/Experimental"
TAEHV = io.Custom("T8_SOL_ENGINE_TAEHV")

folder_paths.add_model_folder_path("taehv", str(folder_paths.models_dir + "/taehv"))
folder_paths.add_model_folder_path("taehv", str(folder_paths.models_dir + "/vae"))


def _taehv_names():
    names = folder_paths.get_filename_list("taehv")
    return names or ["taeltx2_3_wide.pth"]


class MiniMaxH3SolEngineDraftToLTXT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineDraftToLTXT8Advanced",
            display_name="H3 Super: Draft to LTX Refiner / H3草稿转LTX二阶段 (Advanced/T8)",
            description=(
                "Prepares decoded H3 RGB frames for NVIDIA's H3 Super Acceleration Stage 2: "
                "trim to the LTX 8n+1 frame grid, aspect-preserving center-crop to half the "
                "requested output size, then encode it with ComfyUI's full LTX-2.5 video VAE "
                "before the official x2 latent upsampler. The refiner was trained on the full "
                "VAE latent distribution; TAEHV is only used for the final fast decode. H3 "
                "audio must bypass Stage 2 on a separate wire."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames", tooltip="Decoded H3 draft frames; AUDIO is intentionally not accepted."),
                io.Int.Input("target_width", default=1920, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=1088, min=32, max=16384, step=32),
                io.Combo.Input(
                    "frame_policy",
                    options=["trim_to_8n_plus_1", "preserve_all_exp"],
                    default="trim_to_8n_plus_1",
                    tooltip=(
                        "Official route trims without adding frames. preserve_all_exp is only "
                        "for runtimes that explicitly accept a non-8n+1 length."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="Source H3 frame rate; used only to calculate exact audio-bypass trim duration.",
                ),
            ],
            outputs=[
                io.Image.Output("ltx_encoder_frames"),
                io.Int.Output("encoder_width"),
                io.Int.Output("encoder_height"),
                io.Int.Output("kept_frames"),
                io.Int.Output("dropped_tail_frames"),
                io.Float.Output("output_duration_seconds"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, frames, target_width, target_height, frame_policy, fps):
        return io.NodeOutput(
            *prepare_h3_draft_for_ltx_refiner(
                frames,
                target_width,
                target_height,
                frame_policy,
                fps,
            )
        )


class MiniMaxH3SolEngineLTXRefinerSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineLTXRefinerSetupT8Advanced",
            display_name="H3 Super: LTX-2.5 3-Step Refiner Setup (Advanced/T8)",
            description=(
                "Configures the official LTX-2.5 Stage-2 Euler schedule "
                "[0.909375, 0.725, 0.421875, 0], keeps self-attention layer 0 dense, "
                "and routes layers 1-47 through the loaded ComfyUI-SolAttn_triton backend "
                "with tau 1.0/1.25/1.5. Missing Sol-Attn falls back to dense without blocking. "
                "Apply the official LTX-2.5 distilled LoRA at the returned 0.8 strength before this node."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="LTX-2.5 dev transformer with the distilled refiner LoRA applied at 0.8."),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "attention_backend",
                    options=["auto_sol_attn", "dense_reference"],
                    default="auto_sol_attn",
                    tooltip="auto uses an already-loaded ComfyUI-SolAttn_triton; otherwise it safely stays dense.",
                ),
                io.Int.Input("min_tokens", default=4096, min=0, max=1048576, step=512, advanced=True),
                io.Combo.Input(
                    "kernel_precision",
                    options=["bf16_official", "int8_experimental"],
                    default="bf16_official",
                    advanced=True,
                ),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.Float.Output("refiner_lora_strength"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled,
        attention_backend,
        min_tokens,
        kernel_precision,
        verbose,
    ):
        patched, sigmas, strength, report = setup_ltx_stage2_refiner(
            model,
            enabled,
            attention_backend,
            min_tokens,
            kernel_precision,
            verbose,
        )
        return io.NodeOutput(
            patched,
            comfy.samplers.sampler_object("euler"),
            sigmas,
            strength,
            report,
        )


class MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced",
            display_name=(
                "H3 Super: LTX-2.5 Low-Sigma Identity Refiner / "
                "LTX低Sigma保脸细化 (Advanced/T8)"
            ),
            description=(
                "Append-only identity-preserving Stage-2 route. The default schedule is "
                "[0.5, 0.412, 0.350, 0]: three Euler updates with complete terminal denoising. "
                "Dense attention is the validated-first default. Optional Sol-Attn uses a "
                "conservative constant tau 1.0 EXP policy instead of remapping custom knots to "
                "the official per-step tau schedule. H3 audio remains on the external bypass."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "LTX-2.5 dev transformer with the distilled refiner LoRA applied at 0.8."
                    ),
                ),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "schedule_mode",
                    options=["identity_preserve_0p5", "manual_exp"],
                    default="identity_preserve_0p5",
                    tooltip=(
                        "The preset always uses 0.5, 0.412, 0.350, 0. manual_exp parses the next field."
                    ),
                ),
                io.String.Input(
                    "manual_sigmas",
                    default="0.5, 0.412, 0.350, 0",
                    tooltip=(
                        "Comma-separated, finite, strictly descending values ending at zero. "
                        "Used only when schedule_mode=manual_exp."
                    ),
                ),
                io.Combo.Input(
                    "attention_backend",
                    options=[
                        "dense_reference",
                        "auto_sol_attn_conservative_exp",
                    ],
                    default="dense_reference",
                    tooltip=(
                        "Dense isolates the sigma change. The Sol route is optional and experimental."
                    ),
                ),
                io.Int.Input(
                    "min_tokens",
                    default=4096,
                    min=0,
                    max=1048576,
                    step=512,
                    advanced=True,
                ),
                io.Combo.Input(
                    "kernel_precision",
                    options=["bf16_official", "int8_experimental"],
                    default="bf16_official",
                    advanced=True,
                ),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.Float.Output("refiner_lora_strength"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled,
        schedule_mode,
        manual_sigmas,
        attention_backend,
        min_tokens,
        kernel_precision,
        verbose,
    ):
        patched, sigmas, strength, report = setup_ltx_identity_preserve_refiner(
            model,
            enabled,
            schedule_mode,
            manual_sigmas,
            attention_backend,
            min_tokens,
            kernel_precision,
            verbose,
        )
        return io.NodeOutput(
            patched,
            comfy.samplers.sampler_object("euler"),
            sigmas,
            strength,
            report,
        )


class MiniMaxH3SolEngineTAEHVLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineTAEHVLoaderT8Advanced",
            display_name="H3 Super: TAEHV Wide Loader / TAEHV宽模型加载 (Advanced/T8)",
            description=(
                "Loads the taeltx2_3_wide codec used by NVIDIA's published H3 Super "
                "Stage 2. The architecture is selected explicitly; execution is never "
                "blocked by a filename, hash, or byte-size allowlist."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_taehv_names(),
                    default="taeltx2_3_wide.pth",
                    tooltip="Place the official weight in ComfyUI/models/taehv or models/vae.",
                )
            ],
            outputs=[TAEHV.Output("taehv")],
        )

    @classmethod
    def execute(cls, model_name):
        path = folder_paths.get_full_path_or_raise("taehv", model_name)
        return io.NodeOutput(load_taehv_wide(path))


class MiniMaxH3SolEngineTAEHVEncodeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineTAEHVEncodeT8Advanced",
            display_name="TAEHV Encode (Legacy/Diagnostic) / TAEHV编码（旧兼容/诊断）",
            description=(
                "Legacy-compatible TAEHV round-trip encoder. Do not connect this output to "
                "the H3 Super LTX-2.5 refiner: NVIDIA keeps the original full LTX video VAE "
                "encoder because the refiner was trained on that latent distribution."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                TAEHV.Input("taehv"),
                io.Combo.Input(
                    "execution_mode",
                    options=["auto_official", "sequential_low_vram", "parallel_high_vram_exp"],
                    default="auto_official",
                ),
                io.Combo.Input(
                    "precision",
                    options=["bf16_official", "fp32_reference"],
                    default="bf16_official",
                ),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, frames, taehv, execution_mode, precision):
        return io.NodeOutput(
            *encode_h3_frames_with_taehv(frames, taehv, execution_mode, precision)
        )


class MiniMaxH3SolEngineTAEHVDecodeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineTAEHVDecodeT8Advanced",
            display_name="H3 Super: TAEHV Decode / TAEHV解码 (Advanced/T8)",
            description="Decodes the refined normalized LTX video latent with the published TAEHV wide path.",
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("latent"),
                TAEHV.Input("taehv"),
                io.Combo.Input(
                    "execution_mode",
                    options=["auto_official", "sequential_low_vram", "parallel_high_vram_exp"],
                    default="auto_official",
                ),
                io.Combo.Input(
                    "precision",
                    options=["bf16_official", "fp32_reference"],
                    default="bf16_official",
                ),
            ],
            outputs=[io.Image.Output("frames"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, latent, taehv, execution_mode, precision):
        return io.NodeOutput(
            *decode_ltx_latent_with_taehv(latent, taehv, execution_mode, precision)
        )


SOL_ENGINE_H3_SUPER_ADVANCED_NODE_CLASSES = [
    MiniMaxH3SolEngineDraftToLTXT8Advanced,
    MiniMaxH3SolEngineLTXRefinerSetupT8Advanced,
    MiniMaxH3SolEngineTAEHVLoaderT8Advanced,
    MiniMaxH3SolEngineTAEHVEncodeT8Advanced,
    MiniMaxH3SolEngineTAEHVDecodeT8Advanced,
    MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced,
]
