from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .learned_latent_upscale_advanced import (
    ASPECT_POLICIES,
    AUDIO_POLICIES,
    PRECISIONS,
    RELEASE_POLICIES,
    SIZE_MODES,
    build_two_pass_sigma_plan,
    learned_upscale_h3_av_latent,
    reconcile_two_pass_h3_latent,
)


CATEGORY = "T8/MiniMax H3/Latent/Experimental"


class MiniMaxH3LearnedLatentUpscaleT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LearnedLatentUpscaleT8Advanced",
            display_name=(
                "MiniMax H3 Learned Latent Upscale / H3学习型潜空间放大 (Advanced)"
            ),
            description=(
                "Runs the verified 24-channel learned 3D H3 latent-resizer checkpoint on "
                "video only while preserving the native joint audio latent. Output pixel "
                "dimensions are exactly divisible by 32. The default unloads this upscaler "
                "from the GPU after every execution without unloading the user's H3 models."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                ),
                io.Combo.Input("size_mode", options=list(SIZE_MODES), default="scale_by"),
                io.Float.Input("scale_by", default=1.5, min=1.0, max=4.0, step=0.01),
                io.Float.Input(
                    "target_megapixels", default=0.70, min=0.01, max=2.0, step=0.01
                ),
                io.Int.Input("target_width", default=1152, min=32, max=1920, step=32),
                io.Int.Input("target_height", default=640, min=32, max=1088, step=32),
                io.Combo.Input(
                    "aspect_policy",
                    options=list(ASPECT_POLICIES),
                    default="preserve_source",
                    tooltip=(
                        "preserve_source is the safe default. honor_dimensions_exp permits "
                        "different X/Y scales but still enforces max_anisotropy."
                    ),
                ),
                io.Float.Input(
                    "max_anisotropy",
                    default=1.05,
                    min=1.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                    tooltip=(
                        "offload_after keeps a CPU cache but releases GPU weights; clear_after "
                        "also removes the CPU cache; keep_loaded is opt-in and retains GPU memory."
                    ),
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
    def execute(cls, **kwargs):
        latent = kwargs.pop("av_latent")
        return io.NodeOutput(*learned_upscale_h3_av_latent(latent=latent, **kwargs))


class MiniMaxH3TwoPassLatentReconcileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TwoPassLatentReconcileT8Advanced",
            display_name=(
                "MiniMax H3 Two-Pass Latent Reconcile / H3二采潜空间协调 (Advanced)"
            ),
            description=(
                "Combines learned-upscaled first-pass values with a newly rebuilt high-"
                "resolution H3 Conditioning template. It fails closed on stale low-resolution "
                "keyframes, reference metadata mismatches, audio-clock mismatches, or NaN."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("learned_latent"),
                io.Latent.Input("highres_template"),
                io.Conditioning.Input("positive"),
                io.Combo.Input(
                    "audio_policy",
                    options=list(AUDIO_POLICIES),
                    default="auto",
                    tooltip=(
                        "auto keeps the high-resolution template audio for lock/remix masks, "
                        "otherwise it continues the first-pass audio."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Conditioning.Output("positive"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*reconcile_two_pass_h3_latent(**kwargs))


class MiniMaxH3TwoPassSigmaPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TwoPassSigmaPlanT8Advanced",
            display_name="MiniMax H3 Two-Pass Sigma Plan / H3二采Sigma计划 (Advanced)",
            description=(
                "Splits one native H3 rectified-flow trajectory in base-flow time, then "
                "maps both halves with the actual video/audio shifts carried by MODEL. "
                "Default 4+4 keeps eight total joint AV Transformer calls."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect the MODEL output of MiniMax H3 Dual-Clock Sampler so the "
                        "actual video/audio shifts are available."
                    ),
                ),
                io.Int.Input("coarse_steps", default=4, min=1, max=1000),
                io.Int.Input("refine_steps", default=4, min=1, max=1000),
                io.Float.Input(
                    "restart_base_noise", default=0.50, min=0.01, max=0.99, step=0.01
                ),
            ],
            outputs=[
                io.Sigmas.Output("coarse_sigmas"),
                io.Sigmas.Output("refine_sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_two_pass_sigma_plan(**kwargs))


LEARNED_LATENT_UPSCALE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LearnedLatentUpscaleT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
    MiniMaxH3TwoPassSigmaPlanT8Advanced,
]
