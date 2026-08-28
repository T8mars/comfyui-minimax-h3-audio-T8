from __future__ import annotations

from comfy_api.latest import io

from .audio_refine_advanced import (
    AUDIO_REFINE_AUDIT_TYPE,
    AUDIO_REFINE_COMPAT_PLAN_TYPE,
    AUDIO_REFINE_COMPAT_ROUTE_TYPE,
    AUDIO_REFINE_MODEL_ROUTE_TYPE,
    AUDIO_REFINE_PHASE2_PLAN_TYPE,
    AUDIO_REFINE_PLAN_TYPE,
    audit_audio_refine,
    gate_audio_refine_candidate,
    plan_audio_refine,
    plan_audio_refine_compatibility,
    plan_audio_refine_phase2,
    route_audio_refine_model,
    route_audio_refine_compatibility,
    setup_audio_refine_compatibility,
    setup_audio_refine,
    setup_audio_refine_dual_model,
    split_audio_refine_long_video_delivery,
)


CATEGORY = "T8/MiniMax H3/Audio/Experimental"
AudioRefineAuditIO = io.Custom(AUDIO_REFINE_AUDIT_TYPE)
AudioRefinePlanIO = io.Custom(AUDIO_REFINE_PLAN_TYPE)
AudioRefineModelRouteIO = io.Custom(AUDIO_REFINE_MODEL_ROUTE_TYPE)
AudioRefinePhase2PlanIO = io.Custom(AUDIO_REFINE_PHASE2_PLAN_TYPE)
AudioRefineCompatRouteIO = io.Custom(AUDIO_REFINE_COMPAT_ROUTE_TYPE)
AudioRefineCompatPlanIO = io.Custom(AUDIO_REFINE_COMPAT_PLAN_TYPE)


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


class MiniMaxH3AudioRefineModelRouteT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineModelRouteT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Model Route / "
                "音频精修双模型路由 (T8 Advanced EXP)"
            ),
            description=(
                "Binds the audited first-pass MODEL to an explicit same-Turbo or "
                "base-without-Turbo refine MODEL using runtime base/patch UUIDs, "
                "weight-patch structure and LoRA metadata. Unknown stacks fail closed."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefineAuditIO.Input("audit"),
                io.Model.Input("first_pass_model"),
                io.Model.Input("refine_model"),
                io.Combo.Input(
                    "route_strategy",
                    options=["same_turbo_stack", "base_without_turbo"],
                    default="same_turbo_stack",
                ),
                io.Int.Input(
                    "declared_first_pass_nfe",
                    default=4,
                    min=4,
                    max=4,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output("refine_model"),
                AudioRefineModelRouteIO.Output("route"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = route_audio_refine_model(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[-1],)})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefinePhase2PlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefinePhase2PlanT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Phase 2 Plan / "
                "音频精修二阶段计划 (T8 Advanced EXP)"
            ),
            description=(
                "Builds the signed four-NFE Phase 2 tail for a proven model route. "
                "Only the pre-registered 0.35 and 0.50 denoise points are accepted."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefineModelRouteIO.Input("route"),
                io.Int.Input("refine_steps", default=4, min=4, max=4),
                io.Float.Input(
                    "audio_denoise",
                    default=0.50,
                    min=0.35,
                    max=0.50,
                    step=0.15,
                    round=0.01,
                ),
                io.Int.Input(
                    "refine_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
            ],
            outputs=[
                AudioRefinePhase2PlanIO.Output("plan"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, route, refine_steps, audio_denoise, refine_seed):
        values = plan_audio_refine_phase2(
            route,
            refine_steps,
            audio_denoise,
            refine_seed,
        )
        return io.NodeOutput(*values, ui={"text": (values[-1],)})


class MiniMaxH3AudioRefineDualModelSetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineDualModelSetupT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Dual-Model Setup / "
                "音频精修双模型装配 (T8 Advanced EXP)"
            ),
            description=(
                "Revalidates the signed Audit/Route/Plan, refine MODEL, conditioning, "
                "latent and resource gates, then emits the exact uncached dual-clock "
                "partial tail for SamplerCustomAdvanced."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefinePhase2PlanIO.Input("plan"),
                io.Model.Input("refine_model"),
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
    def execute(cls, plan, refine_model, positive, av_latent):
        result = setup_audio_refine_dual_model(
            plan=plan,
            refine_model=refine_model,
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
            ui={"text": (result.report_json,)},
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefineCompatibilityRouteT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineCompatibilityRouteT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine 4/8-Step Compatibility Route / "
                "音频精修4/8步兼容路由 (T8 Advanced EXP)"
            ),
            description=(
                "Append-only route for final-latent Audio Refine after Turbo4/8, "
                "learned two-pass, PDD, EAV, Prompt Relay and Long Video. The MODEL "
                "connected here is the explicit refine MODEL; filenames, sizes and "
                "disk hashes are never execution gates."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefineAuditIO.Input("audit"),
                io.Model.Input("refine_model"),
                io.Conditioning.Input("positive"),
                io.Combo.Input(
                    "generation_profile",
                    options=[
                        "turbo4",
                        "turbo8",
                        "learned_latent_twopass_final8",
                        "pdd8",
                        "pdd4_plus4",
                        "eav_turbo8",
                        "prompt_relay_turbo8",
                        "long_video_turbo8",
                        "long_video_prompt_relay_turbo8",
                    ],
                    default="turbo8",
                ),
                io.Int.Input(
                    "declared_first_pass_nfe",
                    default=8,
                    min=4,
                    max=8,
                    step=4,
                ),
            ],
            outputs=[
                io.Model.Output("refine_model"),
                AudioRefineCompatRouteIO.Output("route"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = route_audio_refine_compatibility(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[-1],)})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefineCompatibilityPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineCompatibilityPlanT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Compatibility Plan / "
                "音频精修兼容计划 (T8 Advanced EXP)"
            ),
            description=(
                "Builds the fixed four-NFE final audio tail for a signed compatibility "
                "route. Total NFE is reported as cost only, never as a training-distribution "
                "equivalence claim."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefineCompatRouteIO.Input("route"),
                io.Int.Input("refine_steps", default=4, min=4, max=4),
                io.Float.Input(
                    "audio_denoise",
                    default=0.50,
                    min=0.35,
                    max=0.50,
                    step=0.15,
                    round=0.01,
                ),
                io.Int.Input(
                    "refine_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
            ],
            outputs=[
                AudioRefineCompatPlanIO.Output("plan"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, route, refine_steps, audio_denoise, refine_seed):
        values = plan_audio_refine_compatibility(
            route,
            refine_steps,
            audio_denoise,
            refine_seed,
        )
        return io.NodeOutput(*values, ui={"text": (values[-1],)})


class MiniMaxH3AudioRefineCompatibilitySetupT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineCompatibilitySetupT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Compatibility Setup / "
                "音频精修兼容装配 (T8 Advanced EXP)"
            ),
            description=(
                "Revalidates the compatibility route, Relay/Long Video ownership when "
                "present, conditioning, final AV latent and resources, then emits the "
                "uncached four-NFE dual-clock audio tail for SamplerCustomAdvanced."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioRefineCompatPlanIO.Input("plan"),
                io.Model.Input("refine_model"),
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
    def execute(cls, plan, refine_model, positive, av_latent):
        result = setup_audio_refine_compatibility(
            plan=plan,
            refine_model=refine_model,
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
            ui={"text": (result.report_json,)},
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Long Video Delivery Split / "
                "长视频精修交付分流 (T8 Advanced EXP)"
            ),
            description=(
                "Keeps the exact original segment AV latent for continuation and emits a "
                "separate reviewed delivery latent whose video is forcibly restored from "
                "the original. Never feed delivery_av_latent into the next segment."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input("original_continuation_av_latent"),
                io.Latent.Input("reviewed_delivery_av_latent"),
                io.Boolean.Input("candidate_selected", default=False),
                io.Int.Input("segment_index", default=0, min=0, max=1000000),
            ],
            outputs=[
                io.Latent.Output("continuation_av_latent"),
                io.Latent.Output("delivery_av_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = split_audio_refine_long_video_delivery(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[-1],)})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


class MiniMaxH3AudioRefineQualityGateT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineQualityGateT8Advanced",
            display_name=(
                "MiniMax H3 Audio Refine Quality Gate / "
                "音频精修人工质量门 (T8 Advanced EXP)"
            ),
            description=(
                "Defaults to the original result. It audits decoded candidate audio, "
                "rejects invalid shape/rate/duration/non-finite data, and only after explicit "
                "human acceptance splices the candidate AUDIO latent into the exact original "
                "VIDEO latent. Signal heuristics are review cues, not quality claims."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input("original_av_latent"),
                io.Latent.Input("candidate_av_latent"),
                io.Audio.Input("original_audio"),
                io.Audio.Input("candidate_audio"),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "Keep false until you have listened to the saved original/candidate pair."
                    ),
                ),
                io.Int.Input("video_frame_count", default=0, min=0, max=1000000),
                io.Float.Input("fps", default=24.0, min=0.001, max=1000.0, step=0.001),
                io.Float.Input(
                    "maximum_duration_delta_ms",
                    default=50.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "spectral_drift_threshold",
                    default=0.30,
                    min=0.01,
                    max=4.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "level_delta_threshold_db",
                    default=4.0,
                    min=0.1,
                    max=40.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Int.Input(
                    "persistent_window_count",
                    default=3,
                    min=1,
                    max=100,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Latent.Output("selected_av_latent"),
                io.Audio.Output("selected_audio"),
                io.Boolean.Output("candidate_selected"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        values = gate_audio_refine_candidate(**kwargs)
        return io.NodeOutput(*values, ui={"text": (values[-1],)})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


AUDIO_REFINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AudioRefineAuditT8Advanced,
    MiniMaxH3AudioRefinePlanT8Advanced,
    MiniMaxH3AudioRefineDualClockSetupT8Advanced,
    MiniMaxH3AudioRefineQualityGateT8Advanced,
    MiniMaxH3AudioRefineModelRouteT8Advanced,
    MiniMaxH3AudioRefinePhase2PlanT8Advanced,
    MiniMaxH3AudioRefineDualModelSetupT8Advanced,
]


AUDIO_REFINE_COMPAT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
    MiniMaxH3AudioRefineCompatibilityPlanT8Advanced,
    MiniMaxH3AudioRefineCompatibilitySetupT8Advanced,
    MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced,
]
