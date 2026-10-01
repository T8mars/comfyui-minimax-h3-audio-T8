"""Append-only public H16-3 split-plan and one-window PASS2 nodes."""
from comfy_api.latest import io

from .h16_stages import AUDIO_OUTPUTS, build_h16_plan, sample_h16_pass2
from .h16_effects import bind_h16_eav, audit_h16_eav
from .h16_relay import RUNTIME_TYPE as RELAY_RUNTIME_TYPE, project_h16_relay, audit_h16_relay
from .h16_storage import save_window, load_window
from .storage import fingerprint_stage
from .eav import CONFIG_TYPE, RUNTIME_TYPE
from .chunked_stage_nodes import CONTEXT, PLAN, RESULT, SPEC


CATEGORY = "T8/MiniMax H3/Modular Sampling/H16 Experimental"
H16_RESULT = "T8_H16_PASS2_RESULT"


class MiniMaxH3H16Pass2PlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Explicit PASS2 Plan (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Builds the old H16 v4 full-frame plan from the already completed HIGH input. "
                        "It does not sample. Connect the same source to Chunked Source/Prepare, then "
                        "one independent PASS2 node per temporal window.",
            inputs=[io.Latent.Input("first_pass_latent"),
                    io.Combo.Input("temporal_strategy", options=["guarded_overlap_exp", "full_clip_safe"],
                                   default="guarded_overlap_exp"),
                    io.Int.Input("temporal_chunk_frames", default=34, min=17, max=3600, step=17),
                    io.Int.Input("temporal_overlap_frames", default=17, min=0, max=1700, step=17),
                    io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001),
                    io.String.Input("model_name", default="minimax_h3_latent_upscaler_3d_fp16.safetensors")],
            outputs=[io.Custom(PLAN).Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, first_pass_latent, temporal_strategy="guarded_overlap_exp",
                temporal_chunk_frames=34, temporal_overlap_frames=17,
                anchor_strength=0.999,
                model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors"):
        return io.NodeOutput(*build_h16_plan(
            first_pass_latent, temporal_strategy, temporal_chunk_frames,
            temporal_overlap_frames, anchor_strength, model_name,
        ))


class MiniMaxH3H16Pass2WindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Sample ONE PASS2 Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Samples one explicit v4 H16 full-frame PASS2 window with external editable "
                        "MODEL/positive/NOISE/SAMPLER/SIGMAS. The previous window is a typed input. "
                        "At the final window, refined_exp applies the old absolute-time audio "
                        "crossfade/quiet-tail policy; merge failure retains first-pass audio. "
                        "The denoised_output of the old node was only an output alias, not x0.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Noise.Input("noise"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Combo.Input("audio_output", options=list(AUDIO_OUTPUTS),
                                   default="preserve_first_pass"),
                    io.Custom(H16_RESULT).Input("previous_result", optional=True),
                    io.Conditioning.Input("negative", optional=True),
                    io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.01)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(H16_RESULT).Output("window_result"),
                     io.Custom(RESULT).Output("core_segment_result"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, positive, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, noise, sampler, sigmas,
                audio_output="preserve_first_pass", previous_result=None,
                negative=None, cfg=1.0):
        return io.NodeOutput(*sample_h16_pass2(
            model, positive, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, noise, sampler, sigmas, previous_result,
            negative, cfg, audio_output,
        ))


class MiniMaxH3H16EAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · External Window EAV (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Bind a separately editable EAV config to one exact H16 PASS2 window, "
                        "including the inherited video mask and prior-window guarded overlap. "
                        "Connect the MODEL to that window and audit its actual calls. "
                        "External Relay composition requires a separate local projection.",
            inputs=[io.Model.Input("model"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Custom(CONFIG_TYPE).Input("eav_config"),
                    io.Combo.Input("audio_output", options=list(AUDIO_OUTPUTS),
                                   default="preserve_first_pass"),
                    io.Custom(H16_RESULT).Input("previous_result", optional=True),
                    io.Conditioning.Input("relay_positive", optional=True)],
            outputs=[io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context")],
        )

    @classmethod
    def execute(cls, model, sigmas, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, eav_config, audio_output="preserve_first_pass",
                previous_result=None, relay_positive=None):
        return io.NodeOutput(*bind_h16_eav(
            model, sigmas, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, previous_result, audio_output, eav_config,
            relay_positive,
        ))


class MiniMaxH3H16EAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Window EAV Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit actual EAV forward and attention calls after this exact H16 "
                        "window. Returns the H16-selected final audio, not the core video's "
                        "first-pass-audio intermediate. Connection alone proves no effect.",
            inputs=[io.Custom(H16_RESULT).Input("window_result"),
                    io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, segment_spec, runtime):
        latent, report = audit_h16_eav(window_result, segment_spec, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


class MiniMaxH3H16RelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Project External Relay to Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Authenticate a full-clip external Relay MODEL/CONDITIONING pair and "
                        "project its timeline/layout to one exact guarded H16 PASS2 window. "
                        "Pass both returned MODEL and positive CONDITIONING to that window.",
            inputs=[io.Model.Input("raw_high_model"), io.Model.Input("relay_model"),
                    io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_full_av_latent"),
                    io.Custom("H3_T8_PROMPT_RELAY_PLAN").Input("prompt_relay_plan"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan"),
                    io.Sigmas.Input("sigmas"),
                    io.Combo.Input("audio_output", options=list(AUDIO_OUTPUTS),
                                   default="preserve_first_pass"),
                    io.Custom(H16_RESULT).Input("previous_result", optional=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Custom(RELAY_RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, raw_high_model, relay_model, relay_positive, relay_full_av_latent,
                prompt_relay_plan, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, sigmas, audio_output="preserve_first_pass",
                previous_result=None):
        return io.NodeOutput(*project_h16_relay(
            raw_high_model, relay_model, relay_positive, relay_full_av_latent,
            prompt_relay_plan, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, sigmas, audio_output, previous_result,
        ))


class MiniMaxH3H16RelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Window Relay Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit routed attention and completed forwards for a standalone "
                        "projected H16 Relay window. For Relay+EAV use the EAV joint audit.",
            inputs=[io.Custom(H16_RESULT).Input("window_result"),
                    io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(RELAY_RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, segment_spec, runtime):
        latent, report = audit_h16_relay(window_result, segment_spec, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


def _h16_store_root():
    from pathlib import Path
    import folder_paths
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "h16_window_artifacts"


class MiniMaxH3H16WindowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Freeze ONE Window (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicitly save one completed H16 PASS2 window with its cumulative video and "
                        "audio history. Copy both returned manifest path and exact SHA. This freezes "
                        "the selected result; it is not automatic MODEL/conditioning cache reuse.",
            inputs=[io.Custom(H16_RESULT).Input("window_result"),
                    io.Latent.Input("source_segment"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan"),
                    io.Boolean.Input("confirm_save", default=False)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(H16_RESULT).Output("window_result"),
                     io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, source_segment, segment_spec, pass2_context, plan,
                confirm_save=False):
        if not confirm_save:
            from .h16_storage import verify_window
            from .results import canonical
            verify_window(window_result, source_segment, segment_spec, pass2_context, plan)
            report = canonical({"status": "not_saved_confirm_save_false",
                                "automatic_cache_reuse": False})
            return io.NodeOutput(window_result.output_latent, window_result, "", "", report,
                                 ui={"text": (report,)})
        output, result, path, digest, report = save_window(
            window_result, source_segment, segment_spec, pass2_context, plan,
            _h16_store_root(),
        )
        return io.NodeOutput(output, result, path, digest, report,
                             ui={"text": ("artifact_path: " + path,
                                          "artifact_sha256: " + digest, report)})


class MiniMaxH3H16WindowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H16-3 · Load Exact Frozen Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Load an explicitly selected completed H16 window by exact manifest SHA. "
                        "Rechecks the current source, plan and window number, then restores the "
                        "typed result for the next window without sampling prior windows. "
                        "Changed MODEL/conditioning is not treated as an automatic cache hit.",
            inputs=[io.Latent.Input("source_segment"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan"),
                    io.String.Input("artifact_path", default=""),
                    io.String.Input("artifact_sha256", default="")],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(H16_RESULT).Output("window_result"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, source_segment, segment_spec, pass2_context, plan,
                artifact_path, artifact_sha256):
        return io.NodeOutput(*load_window(
            source_segment, segment_spec, pass2_context, plan,
            _h16_store_root(), artifact_path, artifact_sha256,
        ))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_kwargs):
        try:
            return fingerprint_stage(_h16_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3H16Pass2PlanEXPT8, MiniMaxH3H16Pass2WindowEXPT8,
         MiniMaxH3H16EAVApplyEXPT8, MiniMaxH3H16EAVAuditEXPT8,
         MiniMaxH3H16RelayProjectEXPT8, MiniMaxH3H16RelayAuditEXPT8,
         MiniMaxH3H16WindowSaveEXPT8, MiniMaxH3H16WindowLoadEXPT8]
