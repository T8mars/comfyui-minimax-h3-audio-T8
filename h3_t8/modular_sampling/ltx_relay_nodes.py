"""Opt-in LTX Prompt Relay planning; effect nodes follow this independent grid."""
from comfy_api.latest import io

from ..prompt_relay_events_advanced import PROMPT_RELAY_EVENTS_TYPE
from .ltx_relay_plan import PLAN_TYPE, build_ltx_relay_plan
from .ltx_relay_text import BINDING_TYPE, encode_ltx_relay_conditioning
from .ltx_relay_apply import RUNTIME_TYPE, apply_ltx_relay, audit_ltx_relay


class MiniMaxH3LTXPromptRelayPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="LTX Prompt Relay · 8n+1 Timeline (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Plan events from the actual LTX latent length; no H3 17n+5 alignment, "
                        "text encoding or attention change yet. LTX adaptation is experimental.",
            inputs=[io.Latent.Input("ltx_latent"),
                    io.String.Input("global_prompt", multiline=True, dynamic_prompts=True,
                                    default="Cinematic continuous shot, stable subject and lighting."),
                    io.String.Input("local_prompts", multiline=True, dynamic_prompts=True,
                                    default="The subject enters the street.\nThe subject turns toward the camera."),
                    io.Combo.Input("timing_mode", options=list(("auto_equal", "frames", "seconds", "percent")),
                                   default="auto_equal"),
                    io.String.Input("time_ranges", multiline=True, default=""),
                    io.Float.Input("fps", default=24., min=0.001, max=1000., step=0.001),
                    io.Float.Input("epsilon", default=.1, min=.000001, max=.999999, step=.01),
                    io.Boolean.Input("allow_gaps", default=False),
                    io.Boolean.Input("allow_overlaps", default=False),
                    io.Custom(PROMPT_RELAY_EVENTS_TYPE).Input("prompt_relay_events", optional=True)],
            outputs=[io.Custom(PLAN_TYPE).Output("ltx_relay_plan"),
                     io.String.Output("compiled_prompt"), io.Int.Output("frame_count"),
                     io.String.Output("timeline_json"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_ltx_relay_plan(**kwargs))


class MiniMaxH3LTXPromptRelayEncodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="LTX Prompt Relay · Encode Text Segments (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Encode global and local segments with the selected LTX CLIP. "
                        "Connect positive to LTXVConditioning; this node alone does not apply Relay.",
            inputs=[io.Clip.Input("clip"), io.Custom(PLAN_TYPE).Input("ltx_relay_plan"),
                    io.Int.Input("max_text_tokens", default=2048, min=1, max=8192)],
            outputs=[io.Conditioning.Output("positive"), io.Custom(BINDING_TYPE).Output("text_binding"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, clip, ltx_relay_plan, max_text_tokens=2048):
        return io.NodeOutput(*encode_ltx_relay_conditioning(clip, ltx_relay_plan, max_text_tokens))


class MiniMaxH3LTXPromptRelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="LTX Prompt Relay · External Apply (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="After LTX Setup/LoRA, route MODEL to Guider/Stage Bind and positive to "
                        "LTXVConditioning. Only video text cross-attention is patched; audit coverage.",
            inputs=[io.Model.Input("model"), io.Latent.Input("ltx_latent"),
                    io.Sigmas.Input("sigmas"), io.Conditioning.Input("positive"),
                    io.Custom(BINDING_TYPE).Input("text_binding"),
                    io.Combo.Input("mode", options=list(("disabled", "report_only", "apply_exp")),
                                   default="report_only"),
                    io.Int.Input("max_workspace_mib", default=32, min=1, max=512)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Custom(RUNTIME_TYPE).Output("runtime"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, ltx_latent, sigmas, positive, text_binding,
                mode="report_only", max_workspace_mib=32):
        return io.NodeOutput(*apply_ltx_relay(model, ltx_latent, sigmas, positive,
                                               text_binding, mode, max_workspace_mib))


class MiniMaxH3LTXPromptRelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="LTX Prompt Relay · Observe Candidate (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            is_output_node=True,
            description="Reports observed delegated video text attention and candidate geometry. "
                        "Not sampler provenance, complete media or quality proof.",
            inputs=[io.Model.Input("model"), io.Latent.Input("candidate_latent"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("candidate_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, candidate_latent, runtime):
        return io.NodeOutput(*audit_ltx_relay(model, candidate_latent, runtime))


NODES = [MiniMaxH3LTXPromptRelayPlanEXPT8, MiniMaxH3LTXPromptRelayEncodeEXPT8,
         MiniMaxH3LTXPromptRelayApplyEXPT8, MiniMaxH3LTXPromptRelayAuditEXPT8]
