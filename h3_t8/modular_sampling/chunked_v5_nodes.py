"""Append-only public stages for standard joint 4+4 Chunked v5."""
from comfy_api.latest import io

from .chunked_v5 import lift_standard_joint, prepare_standard_joint, sample_standard_window
from .chunked_v5_effects import bind_v5_eav, audit_v5_eav
from .chunked_v5_relay import RUNTIME_TYPE as RELAY_RUNTIME_TYPE, project_v5_relay, audit_v5_relay
from .eav import CONFIG_TYPE, RUNTIME_TYPE
from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE


CATEGORY = "T8/MiniMax H3/Modular Sampling/Chunked Experimental"
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
LIFT = "T8_CHUNKED_V5_GLOBAL_LIFT"
PREPARED = "T8_CHUNKED_V5_PREPARED"
RESULT = "T8_CHUNKED_V5_WINDOW_RESULT"


class MiniMaxH3ChunkedV5GlobalLiftEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Global Learned Lift (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Runs the original learned 3D upscaler exactly once on the complete partial4 "
                        "denoised_output AV. Does not sample PASS2 or create a portable cache receipt.",
            inputs=[io.Latent.Input("partial4_denoised_output"), io.Custom(PLAN).Input("plan")],
            outputs=[io.Latent.Output("lifted_full_av"), io.Custom(LIFT).Output("lift_receipt"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, partial4_denoised_output, plan):
        return io.NodeOutput(*lift_standard_joint(partial4_denoised_output, plan))


class MiniMaxH3ChunkedV5PrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Prepare Full AV Noise (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Draws global target video and audio noise once after the global learned lift. "
                        "All PASS2 windows must consume this same typed live preparation.",
            inputs=[io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(LIFT).Input("lift_receipt"),
                    io.Custom(PLAN).Input("plan"), io.Noise.Input("noise")],
            outputs=[io.Custom(PREPARED).Output("prepared"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, partial4_denoised_output, lifted_full_av, lift_receipt, plan, noise):
        return io.NodeOutput(*prepare_standard_joint(
            partial4_denoised_output, lifted_full_av, lift_receipt, plan, noise,
        ))


class MiniMaxH3ChunkedV5PASS2WindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · One Joint AV PASS2 Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Runs exactly one standard remaining4 refine window with external MODEL, "
                        "HIGH CONDITIONING, NOISE, SAMPLER and SIGMAS. Later windows require the "
                        "prior typed result; final audio is refined joint AV, not first-pass passthrough. "
                        "Cumulative intermediate outputs are not portable cache receipts.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(PREPARED).Input("prepared"),
                    io.Custom(PLAN).Input("plan"), io.Noise.Input("noise"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Int.Input("window_index", default=0, min=0, max=9999),
                    io.Custom(RESULT).Input("previous_result", optional=True),
                    io.Conditioning.Input("negative", optional=True),
                    io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("window_result"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, positive, partial4_denoised_output, lifted_full_av,
                prepared, plan, noise, sampler, sigmas, window_index,
                previous_result=None, negative=None, cfg=1.0):
        return io.NodeOutput(*sample_standard_window(
            model, positive, partial4_denoised_output, lifted_full_av,
            prepared, plan, noise, sampler, sigmas, window_index,
            previous_result, negative, cfg,
        ))


class MiniMaxH3ChunkedV5EAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · External Window EAV (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Apply a separately editable Stage EAV Config to the exact joint AV PASS2 "
                        "window, including locked video/audio overlap. Connect its MODEL to the matching "
                        "PASS2 window and runtime to the audit. Relay composition is not yet qualified.",
            inputs=[io.Model.Input("model"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(PREPARED).Input("prepared"),
                    io.Custom(PLAN).Input("plan"), io.Int.Input("window_index", default=0, min=0, max=9999),
                    io.Custom(CONFIG_TYPE).Input("eav_config"),
                    io.Custom(RESULT).Input("previous_result", optional=True),
                    io.Conditioning.Input("relay_positive", optional=True)],
            outputs=[io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context")],
        )

    @classmethod
    def execute(cls, model, sigmas, partial4_denoised_output, lifted_full_av, prepared,
                plan, window_index, eav_config, previous_result=None, relay_positive=None):
        return io.NodeOutput(*bind_v5_eav(
            model, sigmas, partial4_denoised_output, lifted_full_av, prepared,
            plan, window_index, previous_result, eav_config, relay_positive,
        ))


class MiniMaxH3ChunkedV5EAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Window EAV Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit actual EAV calls after one matching joint AV PASS2 window. "
                        "A connection alone does not certify effect coverage or image/audio quality.",
            inputs=[io.Custom(RESULT).Input("window_result"),
                    io.Custom(PREPARED).Input("prepared"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, prepared, runtime):
        latent, report = audit_v5_eav(window_result, prepared, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


class MiniMaxH3ChunkedV5RelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Project External Relay to Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Authenticate a full-clip external Prompt Relay Plan/Conditioning pair, then "
                        "project its event coordinates and packed AV layout to one actual v5 PASS2 "
                        "window. The 136-frame first window cannot use an unprojected 141-frame "
                        "aligned Relay layout. Connect both projected MODEL and CONDITIONING to PASS2.",
            inputs=[io.Model.Input("raw_high_model"), io.Model.Input("relay_model"),
                    io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_full_av_latent"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("prompt_relay_plan"),
                    io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(PREPARED).Input("prepared"),
                    io.Custom(PLAN).Input("plan"), io.Sigmas.Input("sigmas"),
                    io.Int.Input("window_index", default=0, min=0, max=9999),
                    io.Custom(RESULT).Input("previous_result", optional=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Custom(RELAY_RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, raw_high_model, relay_model, relay_positive, relay_full_av_latent,
                prompt_relay_plan, partial4_denoised_output, lifted_full_av,
                prepared, plan, sigmas, window_index, previous_result=None):
        return io.NodeOutput(*project_v5_relay(
            raw_high_model, relay_model, relay_positive, relay_full_av_latent,
            prompt_relay_plan, partial4_denoised_output, lifted_full_av,
            prepared, plan, sigmas, window_index, previous_result,
        ))


class MiniMaxH3ChunkedV5RelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Window Relay Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit actual routed Relay attention calls after the matching v5 PASS2 "
                        "window. This standalone audit does not certify Relay+EAV composition.",
            inputs=[io.Custom(RESULT).Input("window_result"),
                    io.Custom(PREPARED).Input("prepared"),
                    io.Custom(RELAY_RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, prepared, runtime):
        latent, report = audit_v5_relay(window_result, prepared, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


NODES = [MiniMaxH3ChunkedV5GlobalLiftEXPT8, MiniMaxH3ChunkedV5PrepareEXPT8,
         MiniMaxH3ChunkedV5PASS2WindowEXPT8, MiniMaxH3ChunkedV5EAVApplyEXPT8,
         MiniMaxH3ChunkedV5EAVAuditEXPT8, MiniMaxH3ChunkedV5RelayProjectEXPT8,
         MiniMaxH3ChunkedV5RelayAuditEXPT8]
