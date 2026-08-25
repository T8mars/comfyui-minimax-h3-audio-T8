from __future__ import annotations

from comfy_api.latest import io

from .audio_refine_advanced import (
    AUDIO_REFINE_AUDIT_TYPE,
    AUDIO_REFINE_PLAN_TYPE,
    audit_audio_refine,
    plan_audio_refine,
    setup_audio_refine,
)


CATEGORY = "T8/MiniMax H3/Audio/Experimental"
AudioRefineAuditIO = io.Custom(AUDIO_REFINE_AUDIT_TYPE)
AudioRefinePlanIO = io.Custom(AUDIO_REFINE_PLAN_TYPE)


class MiniMaxH3AudioRefineAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineAuditT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Audit / "
                "音频精修前置审计 (T8 Advanced EXP)"
            ),
            description=(
                "Read-only audit for an exact uncached MiniMax H3 joint-AV audio "
                "refinement tail. ALLOW is only a mechanical precondition; protected "
                "audio, unvalidated patches, and insufficient telemetry ABSTAIN."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Latent.Input("av_latent"),
                io.String.Input("conditioned_prompt", multiline=True),
                io.String.Input("media_map_json", multiline=True),
                io.String.Input("conditioning_report", multiline=True),
                io.Audio.Input("protected_audio", optional=True),
                io.Int.Input(
                    "minimum_free_vram_mib",
                    default=512,
                    min=512,
                    max=65536,
                    step=128,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_commit_headroom_gib",
                    default=16.0,
                    min=16.0,
                    max=512.0,
                    step=1.0,
                    round=0.1,
                    advanced=True,
                ),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioRefineAuditIO.Output("audit"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*audit_audio_refine(**kwargs))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefinePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefinePlanT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Plan / "
                "音频精修尾段计划 (T8 Advanced EXP)"
            ),
            description=(
                "Builds a deterministic KSampler-equivalent partial-tail plan with "
                "fixed CFG 1, video/audio shifts 12/3, video mask 0, audio mask 1, "
                "dual_clock_euler and native_flow. It performs no sampling."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                AudioRefineAuditIO.Input("audit"),
                io.Int.Input("refine_steps", default=4, min=1, max=8, step=1),
                io.Float.Input(
                    "audio_denoise",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    round=0.01,
                ),
                io.Int.Input(
                    "refine_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
                io.Combo.Input(
                    "model_strategy",
                    options=["connected_model_explicit"],
                    default="connected_model_explicit",
                    advanced=True,
                ),
            ],
            outputs=[
                AudioRefinePlanIO.Output("plan"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audit,
        refine_steps,
        audio_denoise,
        refine_seed,
        model_strategy="connected_model_explicit",
    ):
        return io.NodeOutput(
            *plan_audio_refine(
                audit,
                refine_steps,
                audio_denoise,
                refine_seed,
                model_strategy=model_strategy,
            )
        )


class MiniMaxH3AudioRefineDualClockSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineDualClockSetupT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Dual-Clock Setup / "
                "音频精修双时钟装配 (T8 Advanced EXP)"
            ),
            description=(
                "Revalidates the signed Plan and assembles deterministic NOISE, a "
                "BasicGuider-equivalent GUIDER, the stable T8 dual-clock sampler, exact "
                "tail SIGMAS and a nested 0/1 mask. It does not execute the model."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                AudioRefinePlanIO.Input("plan"),
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Latent.Input("av_latent"),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Noise.Output("noise"),
                io.Guider.Output("guider"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.Latent.Output("latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, plan, model, positive, av_latent):
        result = setup_audio_refine(
            plan=plan,
            model=model,
            positive=positive,
            av_latent=av_latent,
        )
        return io.NodeOutput(
            result.model,
            result.noise,
            result.guider,
            result.sampler,
            result.sigmas,
            result.latent,
            result.report_json,
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


AUDIO_REFINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AudioRefineAuditT8Advanced,
    MiniMaxH3AudioRefinePlanT8Advanced,
    MiniMaxH3AudioRefineDualClockSetupT8Advanced,
]
