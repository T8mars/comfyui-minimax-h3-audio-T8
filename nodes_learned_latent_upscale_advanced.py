from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .learned_latent_upscale_advanced import (
    ASPECT_POLICIES,
    AUDIO_POLICIES,
    PRECISIONS,
    RELEASE_POLICIES,
    SECOND_PASS_AUDIO_SOURCES,
    SIZE_MODES,
    audit_two_pass_h3_audio,
    build_learned_two_pass_parity_plan,
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
                "dimensions are exactly divisible by 32. Resolutions above the official "
                "1920x1088 reference area are allowed and reported as a memory risk instead "
                "of being blocked. The default unloads this upscaler from the GPU after every "
                "execution without unloading the user's H3 models."
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
                io.Float.Input("scale_by", default=2.0, min=1.0, max=4.0, step=0.01),
                io.Float.Input(
                    "target_megapixels", default=0.70, min=0.01, max=8.0, step=0.01
                ),
                io.Int.Input("target_width", default=1152, min=32, max=4096, step=32),
                io.Int.Input("target_height", default=640, min=32, max=4096, step=32),
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
                io.Combo.Input(
                    "second_pass_audio_source",
                    options=list(SECOND_PASS_AUDIO_SOURCES),
                    default="legacy_policy",
                    optional=True,
                    tooltip=(
                        "legacy_policy preserves old saved-workflow behavior. first_pass is the "
                        "explicit audio-lock route: reuse pass-1 audio while pass 2 refines video. "
                        "For native author-parity generation, keep legacy_policy so pass 2 can "
                        "finish the joint audio trajectory."
                    ),
                ),
                io.Float.Input(
                    "second_pass_audio_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Pass-2 audio denoise only for an explicit non-legacy source. Use 0 to "
                        "lock the selected audio exactly; values above 0 deliberately regenerate "
                        "it and remain experimental."
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


class MiniMaxH3TwoPassAudioAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TwoPassAudioAuditT8Advanced",
            display_name=(
                "MiniMax H3 Two-Pass Audio Audit / H3二采音频硬校验 (Advanced)"
            ),
            description=(
                "Compares the audio latent immediately before and after pass 2. In the "
                "explicit strength-0 lock route it allows only bounded floating-point roundoff, "
                "then restores the exact input audio before decode. Larger changes fail closed, "
                "preventing a mechanically valid but sonically rewritten result from being saved. "
                "Do not insert this node into the native author-parity route: native pass 2 must "
                "complete audio as well as video."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input(
                    "second_pass_input",
                    tooltip="Connect the Reconcile AV latent that enters pass 2.",
                ),
                io.Latent.Input(
                    "second_pass_output",
                    tooltip="Connect pass-2 SamplerCustomAdvanced output.",
                ),
                io.Float.Input(
                    "expected_audio_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Must match Reconcile second_pass_audio_strength.",
                ),
                io.Boolean.Input(
                    "fail_on_locked_mismatch",
                    default=True,
                    tooltip="At strength 0, stop the workflow if audio differs after pass 2.",
                ),
                io.Float.Input(
                    "locked_atol",
                    default=1e-5,
                    min=0.0,
                    max=1.0,
                    step=1e-7,
                    tooltip=(
                        "Maximum sampler roundoff allowed before failure. After the check, "
                        "locked audio is replaced by the exact pass-2 input latent."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("verified_av_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = audit_two_pass_h3_audio(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[1],)})


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


class MiniMaxH3LearnedTwoPassParityPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LearnedTwoPassParityPlanT8Advanced",
            display_name=(
                "MiniMax H3 Learned Two-Pass Parity Plan / H3学习放大原版二采计划 "
                "(Advanced)"
            ),
            description=(
                "Reproduces the published LBH low-resolution simple-schedule split and "
                "its current 3/4/5-step high-resolution raw video-sigma sequences. "
                "The native H3 sampler derives the matching audio clock from the active "
                "video/audio shifts."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect a MiniMax H3 model carrying the same video/audio shifts "
                        "that both passes will use."
                    ),
                ),
                io.Int.Input("base_steps", default=8, min=2, max=1000),
                io.Int.Input("coarse_steps", default=4, min=1, max=999),
                io.Int.Input(
                    "refine_steps",
                    default=4,
                    min=3,
                    max=5,
                    tooltip=(
                        "Defaults to the published four-call high-resolution profile so the "
                        "4+4 graph executes eight total joint AV forwards. Saved workflows "
                        "that explicitly store 3 or 5 remain compatible."
                    ),
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
        return io.NodeOutput(*build_learned_two_pass_parity_plan(**kwargs))


LEARNED_LATENT_UPSCALE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LearnedLatentUpscaleT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
    MiniMaxH3TwoPassSigmaPlanT8Advanced,
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
]

LEARNED_LATENT_AUDIO_AUDIT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3TwoPassAudioAuditT8Advanced,
]
