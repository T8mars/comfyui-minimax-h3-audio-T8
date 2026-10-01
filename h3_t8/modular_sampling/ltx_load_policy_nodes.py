"""Append-only explicit LTX native load profile and observation nodes."""
from comfy_api.latest import io

from .ltx_load_policy import MODES, apply_ltx_load_policy, audit_ltx_load_policy

RUNTIME = io.Custom("T8_LTX_LOAD_POLICY_RUNTIME")
CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


class MiniMaxH3LTXLoadPolicyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="LTX · Native Load Policy (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Legacy default returns the identical MODEL. Explicit consistent_streaming_exp clones MODEL "
                        "and requests Core native force_offload before LTX sampling, retaining all LoRAs/wrappers. "
                        "Connect this MODEL to effects, Guider and Stage Bind/Audit. Changes residency/rounding versus "
                        "mixed placement; no sampler, schedule, quality, portable-cache or universal speed guarantee.",
            inputs=[io.Model.Input("model"), io.Combo.Input("mode", options=list(MODES), default=MODES[0])],
            outputs=[io.Model.Output("model"), RUNTIME.Output("runtime"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, mode="legacy_default"):
        return io.NodeOutput(*apply_ltx_load_policy(model, mode))


class MiniMaxH3LTXLoadPolicyAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="LTX · Observe Native Load Policy (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Pass candidate LATENT unchanged; observe native preparation delegations and missing wrappers. "
                        "Not proof of sampling completion, provenance, exact repeatability or quality.",
            inputs=[io.Model.Input("model"), io.Latent.Input("candidate_latent"), RUNTIME.Input("runtime")],
            outputs=[io.Latent.Output("candidate_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, candidate_latent, runtime):
        return io.NodeOutput(*audit_ltx_load_policy(model, candidate_latent, runtime))


NODES = [MiniMaxH3LTXLoadPolicyEXPT8, MiniMaxH3LTXLoadPolicyAuditEXPT8]
