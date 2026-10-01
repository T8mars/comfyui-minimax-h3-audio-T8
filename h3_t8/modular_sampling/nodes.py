from comfy_api.latest import io

from .fast_h3_v2 import PROFILES, STAGES, build_stage
import json
from .eav import CONFIG_TYPE, RUNTIME_TYPE, EAVConfig, apply_stage_eav, audit_stage_eav
from .results import sample_stage
from .storage import save_stage, load_stage, fingerprint_stage
from . import native_dual, manual_pass, rf_stages, native_explicit, pdd_stages, vdn_stages, vdn_relay, noise as stage_noise


class MiniMaxH3VDNRelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VDNRelayApplyEXPT8",
            display_name="H3 VDN · External Prompt Relay Apply (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Paired Relay Conditioning -> VDN Stage -> this node -> optional Stage EAV. "
                        "Adds real temporal bias to original VDN window keys and beta-weighted nonlinear text seeds "
                        "to the original linear scan. Explicit experimental VDN extension, not paper-softmax or trained "
                        "quality equivalence. Independent stage config, no extra diffusion NFE or dense fallback. "
                        "Workspace is adapter working-set estimate, not total VRAM. Unsupported kernels fail normally.",
            inputs=[io.Model.Input("model"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context"),
                    io.Combo.Input("mode", options=["disabled", "apply_exp"], default="apply_exp"),
                    io.Int.Input("max_workspace_mib", default=64, min=4, max=1024)],
            outputs=[io.Model.Output("model"), io.Custom(vdn_relay.RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sigmas, av_latent, stage_context, mode="apply_exp", max_workspace_mib=64):
        return io.NodeOutput(*vdn_relay.apply(model, sigmas, av_latent, stage_context,
                                             vdn_relay.Config(mode, max_workspace_mib)))


class MiniMaxH3VDNRelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VDNRelayAuditEXPT8",
            display_name="H3 VDN · External Prompt Relay Audit (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Wire after this stage's sampler/save. Reports real window/linear calls and coverage, "
                        "not just the presence of a Relay plan. Unknown bypassed producers remain unverified.",
            inputs=[io.Latent.Input("av_latent"), io.Custom(vdn_relay.RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, runtime):
        return io.NodeOutput(*vdn_relay.audit(av_latent, runtime))


class MiniMaxH3VDNStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VDNStageSetupEXPT8",
            display_name="H3 VDN · Complete / Own-Grid Tail Stage (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Connect an existing OpenVDN Composer MODEL. Prepares only the full DMD8/B50 or its "
                        "own fresh-noise tail; does not sample or load weights. Keep learned upscale/reconcile external. "
                        "Independent model/LoRA/condition/noise. VDN Stage EAV has an actual producer adapter. "
                        "VDN temporal Relay needs the separate VDN Relay Apply adapter; low-VRAM side-model "
                        "persistent identity and full pretrained/GPU quality remain pending.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("stage", options=list(vdn_stages.STAGES), default=vdn_stages.STAGES[0]),
                    io.Int.Input("refine_steps", default=4, min=1, max=49),
                    io.Latent.Input("first_pass_latent", optional=True)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, stage="vdn_complete", refine_steps=4, first_pass_latent=None):
        return io.NodeOutput(*vdn_stages.build_stage(model, av_latent, stage, refine_steps, first_pass_latent))


class MiniMaxH3PDDStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PDDStageSetupEXPT8",
            display_name="H3 PDD · Separate Absolute LOW4 / HIGH4 (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Connect an existing PDD MODEL and its original full nine-point SIGMAS. Sets up only "
                        "LOW heads 0–3 or HIGH heads 4–7 at this stage's actual AV geometry; no sampler executes. "
                        "Keep learned upscale and joint-audio reconcile external. Independent models/LoRA/noise/conditions. "
                        "Native head banks and source-authenticated legacy dynamic injection support stage identity/resume. "
                        "Unknown executable additions remain unverified, not silently removed.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"), io.Sigmas.Input("full_sigmas"),
                    io.Combo.Input("stage", options=list(pdd_stages.STAGES), default=pdd_stages.STAGES[0])],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, full_sigmas, stage="pdd_low_0_4"):
        return io.NodeOutput(*pdd_stages.build_stage(model, av_latent, full_sigmas, stage))


class MiniMaxH3NativeStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3NativeStageBindEXPT8",
            display_name="H3 Native · Bind Explicit Stage (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Connect an existing native MODEL/SAMPLER/SIGMAS and stage latent. No sampling, schedule "
                        "replacement or hidden second pass. Adds stage effects/results to base-flow, LBH and complete-first "
                        "graphs. Keep the original upscaler/audio reconcile outside. Not a PDD/VDN/V2 qualification.",
            inputs=[io.Model.Input("model"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("av_latent"), io.Combo.Input("stage", options=list(native_explicit.STAGES),
                                                                  default=native_explicit.STAGES[0])],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sampler, sigmas, av_latent, stage="native_low"):
        return io.NodeOutput(*native_explicit.bind_stage(model, sampler, sigmas, av_latent, stage))


class MiniMaxH3RFBaseStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RFBaseStageSetupEXPT8",
            display_name="H3 RF · Base Descent Only (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="ONE native dual-clock descent with explicit full/partial/tail SIGMAS. No hidden restart. "
                        "Bias/STG/Relay/EAV stay external. Stage Sampler preserves the original template and exact "
                        "model-space endpoint before Core's float32 output boundary for frozen restart.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"), io.Sigmas.Input("sigmas"),
                    io.Float.Input("shift_video", default=12., min=.01, max=100.),
                    io.Float.Input("shift_audio", default=3., min=.01, max=100.)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, sigmas, shift_video=12., shift_audio=3.):
        return io.NodeOutput(*rf_stages.build_base_stage(model, av_latent, sigmas, shift_video, shift_audio))


class MiniMaxH3RFHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RFHandoffEXPT8",
            display_name="H3 RF · Endpoint + Original Template (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Bind completed OUTPUT and the original base template; no model call or random draw. "
                        "Original template can be omitted only when a Stage Sampler RF result already contains it. "
                        "Do not substitute denoised or the completed endpoint for the original inpaint/audio anchor. "
                        "A plain Core output lacks the exact pre-conversion state; base_sigmas preserves schedule dtype only.",
            inputs=[io.Latent.Input("completed_av"), io.Latent.Input("original_template", optional=True),
                    io.Sigmas.Input("base_sigmas", optional=True)],
            outputs=[io.Custom("T8_RF_HANDOFF").Output("rf_handoff"), io.Latent.Output("completed_av"),
                     io.String.Output("report_json"), io.Int.Output("width"), io.Int.Output("height")])

    @classmethod
    def execute(cls, completed_av, original_template=None, base_sigmas=None):
        value = rf_stages.handoff(completed_av, original_template, base_sigmas)
        video = completed_av["samples"].unbind()[0]
        return io.NodeOutput(value, value.completed_av, json.dumps({"bound": True, "model_calls": 0,
            "noise_generated": False, "anchor_identity": json.loads(value.template_identity)}),
            int(video.shape[-1]) * 16, int(video.shape[-2]) * 16)


class MiniMaxH3RFRestartStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RFRestartStageSetupEXPT8",
            display_name="H3 RF · Restart Descent Only (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Sets up only the second RF descent. Original device-local packed AV re-noise executes "
                        "inside its single-stage SAMPLER, never a base pass. Use the original base NOISE for inpaint. "
                        "Zero steps/sigma is explicit zero-NFE identity. Model, effects and conditions remain editable.",
            inputs=[io.Model.Input("model"), io.Custom("T8_RF_HANDOFF").Input("rf_handoff"),
                    io.Float.Input("shift_video", default=12., min=.01, max=100.),
                    io.Float.Input("shift_audio", default=3., min=.01, max=100.),
                    io.Float.Input("restart_video_sigma", default=.15, min=0., max=.5, step=.01),
                    io.Int.Input("restart_steps", default=3, min=0, max=8),
                    io.Int.Input("restart_seed", default=1234, min=0, max=0xffffffffffffffff)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Latent.Output("av_latent"), io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, rf_handoff, shift_video=12., shift_audio=3., restart_video_sigma=.15,
                restart_steps=3, restart_seed=1234):
        prepared, sampler, sigmas, context, report = rf_stages.build_restart_stage(model, rf_handoff,
            shift_video, shift_audio, restart_video_sigma, restart_steps, restart_seed)
        return io.NodeOutput(prepared, sampler, sigmas, rf_handoff.completed_av, context, report)


class MiniMaxH3FastH3V2StageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FastH3V2StageSetupEXPT8",
            display_name="FastH3 V2 · Separate LOW / HIGH Stage (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Sets up ONE exact DMD stage, never runs a hidden loop. Use two standard "
                        "SamplerCustomAdvanced nodes and the existing learned upscale/reconcile nodes. "
                        "Each stage accepts its own MODEL and LoRA chain. LOW uses absolute 0:4; "
                        "HIGH uses 4:8 with AV shifts 10/3. Split/reference use remains experimental.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("stage", options=list(STAGES), default=STAGES[0]),
                    io.Combo.Input("profile", options=list(PROFILES), default=PROFILES[0]),
                    io.Int.Input("min_tokens", default=12288, min=0, max=1048576)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, av_latent, stage="low_0_4", profile="trained_vsa_exp", min_tokens=12288):
        return io.NodeOutput(*build_stage(model, av_latent, stage, profile, min_tokens))


class MiniMaxH3StageEAVConfigEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageEAVConfigEXPT8",
            display_name="H3 Stage EAV · External Config (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Reusable immutable configuration. Connect only LOW, only HIGH, or separate configs to both. "
                        "Progress is absolute 1-video_sigma; it is not reset at the stage boundary. "
                        "disabled is exact bypass; tau=0 is not an off switch.",
            inputs=[io.Combo.Input("mode", options=["disabled", "report_only", "apply_exp"], default="report_only"),
                    io.Float.Input("tau", default=4., min=-32., max=32., step=.25),
                    io.Float.Input("start_video_progress", default=.15, min=0., max=.99, step=.01),
                    io.Float.Input("end_video_progress", default=.90, min=.01, max=1., step=.01),
                    io.Int.Input("max_workspace_mib", default=32, min=4, max=512),
                    io.Float.Input("g_hard_limit", default=1.5, min=1., max=3., step=.05)],
            outputs=[io.Custom(CONFIG_TYPE).Output("eav_config")])

    @classmethod
    def execute(cls, mode="report_only", tau=4., start_video_progress=.15, end_video_progress=.90,
                max_workspace_mib=32, g_hard_limit=1.5):
        return io.NodeOutput(EAVConfig(mode, tau, start_video_progress, end_video_progress, max_workspace_mib, g_hard_limit))


class MiniMaxH3StageEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageEAVApplyEXPT8",
            display_name="H3 Stage EAV · Apply to This Stage (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Bind one stage after its Setup/Bind. Supports V2, Native Dual, explicit native, PDD, Manual Pass and RF without changing legacy EAV. "
                        "Keeps the selected backend and authenticated Relay; does not silently replace VSA with Dense. "
                        "Native V2 sparse FETA uses bounded extra video projections and the original kernel; "
                        "CUDA qualification and sparse Relay bias remain pending. Foreign producers may bypass effects. "
                        "Connect matching stage SIGMAS and AV latent and inspect the downstream audit.",
            inputs=[io.Model.Input("model"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context"), io.Custom(CONFIG_TYPE).Input("eav_config")],
            outputs=[io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sigmas, av_latent, stage_context, eav_config):
        return io.NodeOutput(*apply_stage_eav(model, sigmas, av_latent, stage_context, eav_config))


class MiniMaxH3StageEAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageEAVAuditEXPT8",
            display_name="H3 Stage EAV · Actual Calls Audit (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True, is_output_node=True,
            description="Wire after the matching sampler. Reports actual absolute sigmas, FETA/Relay calls and "
                        "incomplete producer coverage. No GPU-quality claim or persisted-cache authorization.",
            inputs=[io.Latent.Input("av_latent"), io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, runtime):
        value, report = audit_stage_eav(av_latent, runtime)
        return io.NodeOutput(value, report, ui={"text": (report,)})


class MiniMaxH3StageSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageSamplerEXPT8",
            display_name="H3 Stage Sampler · ONE Stage + Result (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Runs exactly one native SamplerCustomAdvanced call. The five native inputs and first "
                        "two outputs are unchanged; adds typed stage evidence for optional saving. No loop, "
                        "second pass, retry or automatic cache. Binds V2, Native Dual, explicit native, Manual Pass and RF adapters.",
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context")],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, stage_context):
        return io.NodeOutput(*sample_stage(noise, guider, sampler, sigmas, latent_image, stage_context))


def _stage_store_root():
    from pathlib import Path
    import folder_paths
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "stage_artifacts"


class MiniMaxH3StageSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageSaveEXPT8",
            display_name="H3 Stage Result · Save Frozen Artifact (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True, is_output_node=True,
            description="Writes a new immutable native AV artifact under output/MiniMaxH3/stage_artifacts. "
                        "Outputs exact path and mandatory SHA for explicit restoration. Existing artifacts "
                        "are never overwritten. Incomplete or unknown sampler execution is rejected before "
                        "a completed artifact is written; inspect report_json.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), io.String.Input("prefix", default="stage")],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.String.Output("artifact_path"), io.String.Output("artifact_sha256"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, prefix="stage"):
        path, digest, report = save_stage(stage_result, _stage_store_root(), prefix)
        return io.NodeOutput(stage_result.output, stage_result.denoised_output, path, digest, report,
                             ui={"text": ("artifact_path: " + path, "artifact_sha256: " + digest, report)})


class MiniMaxH3StageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageLoadEXPT8",
            display_name="H3 Stage Result · Load Exact Frozen Artifact (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Explicitly selects a saved stage by relative manifest path and exact SHA. Never runs "
                        "LOW or loads a model. Keeps x_sigma and denoised_output distinct. This freezes the "
                        "selected prior result, not a claim that edited LOW settings still match. To rerun "
                        "only HIGH, use saved LOW denoised_output at the existing learned upscale handoff.",
            inputs=[io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default=""),
                    io.Combo.Input("expected_stage", options=list(STAGES) + list(native_dual.STAGES)
                                   + list(manual_pass.STAGES) + list(rf_stages.STAGES)
                                    + list(native_explicit.STAGES) + list(pdd_stages.STAGES)
                                    + list(vdn_stages.STAGES), default=STAGES[0])],
            outputs=[io.Latent.Output("output"), io.Latent.Output("denoised_output"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
                     io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256, expected_stage="low_0_4"):
        return io.NodeOutput(*load_stage(_stage_store_root(), artifact_path, artifact_sha256, expected_stage))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, expected_stage="low_0_4"):
        try:
            return fingerprint_stage(_stage_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            # Force execute to report the actual error, never keep a cached
            # latent after a missing/corrupt/busy external artifact.
            return float("nan")


class MiniMaxH3NativeDualStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3NativeDualStageSetupEXPT8",
            display_name="H3 Native Dual · Separate LOW4 / LOW20 / HIGH3-5 (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Sets up ONE original Dual stage: LOW4 is simple8 prefix, LOW20 full native flow, "
                        "HIGH3/4/5 published LBH tables with this MODEL's own AV shifts. No sampling or loop. "
                        "LOW denoised_output -> existing learned3D -> Native Dual Handoff -> independent HIGH.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("stage", options=list(native_dual.STAGES), default=native_dual.STAGES[0]),
                    io.Float.Input("shift_video", default=12., min=.01, max=100., step=.1),
                    io.Float.Input("shift_audio", default=3., min=.01, max=100., step=.1)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, stage="dual_low_4", shift_video=12., shift_audio=3.):
        return io.NodeOutput(*native_dual.build_stage(model, av_latent, stage, shift_video, shift_audio))


class MiniMaxH3NativeDualHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3NativeDualHandoffEXPT8",
            display_name="H3 Native Dual · Learned / Audio Handoff (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Original Dual audio reconciliation after the existing learned3D upscaler. "
                        "LOW4 auto continues unfinished joint audio; LOW20 auto preserves completed audio. "
                        "Keeps template-locked audio prefix and old first_pass/zero migration. No sampler or upscale.",
            inputs=[io.Latent.Input("learned_latent"), io.Latent.Input("highres_template"),
                    io.Conditioning.Input("positive"),
                    io.Combo.Input("first_pass_steps", options=["4", "20"], default="4"),
                    io.Combo.Input("second_audio_source", options=list(native_dual.AUDIO_SOURCES), default="auto"),
                    io.Float.Input("second_audio_strength", default=0., min=0., max=1., step=.01)],
            outputs=[io.Latent.Output("av_latent"), io.Conditioning.Output("positive"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, learned_latent, highres_template, positive, first_pass_steps="4",
                second_audio_source="auto", second_audio_strength=0.):
        return io.NodeOutput(*native_dual.reconcile(learned_latent, highres_template, positive,
            int(first_pass_steps), second_audio_source, second_audio_strength))


class MiniMaxH3ManualPassStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ..sampling import SAMPLER_OPTIONS, SCHEDULER_OPTIONS
        return io.Schema(node_id="MiniMaxH3ManualPassStageSetupEXPT8",
            display_name="H3 Manual Second Pass · Separate Stage Setup (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="One original full first stage OR independent manual second stage. Does not sample. "
                        "Wire first OUTPUT to second latent_image, not learned-upscale denoised_output. "
                        "Each stage has independent MODEL, conditions and fresh NOISE; same seed reproduces old loops. "
                        "dual_clock_euler/euler completion adapted; other Core samplers run without false certification.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("stage", options=list(manual_pass.STAGES), default=manual_pass.STAGES[0]),
                    io.Int.Input("first_steps", default=20, min=1, max=10000),
                    io.String.Input("manual_sigmas", default="0.5,0.412,0.35,0"),
                    io.Float.Input("shift_video", default=12., min=.01, max=100., step=.1),
                    io.Float.Input("shift_audio", default=3., min=.01, max=100., step=.1),
                    io.Combo.Input("sampler_name", options=SAMPLER_OPTIONS, default="dual_clock_euler"),
                    io.Combo.Input("scheduler", options=SCHEDULER_OPTIONS, default="native_flow")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, stage="manual_first", first_steps=20,
                manual_sigmas="0.5,0.412,0.35,0", shift_video=12., shift_audio=3.,
                sampler_name="dual_clock_euler", scheduler="native_flow"):
        return io.NodeOutput(*manual_pass.build_stage(model, av_latent, stage, first_steps, manual_sigmas,
            shift_video, shift_audio, sampler_name, scheduler))


class MiniMaxH3StageNoiseEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3StageNoiseEXPT8",
            display_name="H3 Stage Noise · External FreeNoise / Legacy Plan (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Wraps one NOISE generation; no diffusion. Reuses original FreeNoise video permutation/blend, "
                        "preserves source seed/batch_index and exact audio noise. Same segment_index for both passes "
                        "reproduces old loops. from_model_plan reads the optional MODEL's old FreeNoise configuration. "
                        "disabled or absent model plan is exact provider bypass; configuration report is not execution.",
            inputs=[io.Noise.Input("noise"), io.Combo.Input("mode", options=list(stage_noise.MODES), default="disabled"),
                    io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                    io.Float.Input("reuse_ratio", default=.65, min=0., max=1., step=.01),
                    io.Int.Input("segment_index", default=0, min=0, max=0x7FFFFFFF),
                    io.Model.Input("model", optional=True)],
            outputs=[io.Noise.Output("noise"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, noise, mode="disabled", base_seed=123456789, reuse_ratio=.65, segment_index=0, model=None):
        return io.NodeOutput(*stage_noise.build_noise(noise, mode, base_seed, reuse_ratio, segment_index, model))
