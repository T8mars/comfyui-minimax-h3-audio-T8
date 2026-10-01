"""Append-only public nodes for per-segment Chunked v1 Relay projection."""
from comfy_api.latest import io

from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .chunked_stage_nodes import CATEGORY, CONTEXT, PLAN, RESULT, SPEC
from .chunked_v1_relay import (
    RUNTIME_TYPE, audit_v1_local_relay, project_v1_local_relay,
)


class MiniMaxH3ChunkedV1RelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v1 · Project Relay to ONE Segment (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Project a stock full-clip Prompt Relay Plan/Conditioning onto one native v1 "
                        "temporal segment, including the previous completed segment's anchor. "
                        "Requires full_frame_safe and locked input audio; spatial tiles have no "
                        "Relay certification. Connect a separate instance before each PASS2 segment.",
            inputs=[io.Model.Input("raw_model"), io.Model.Input("relay_model"),
                    io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_full_av_latent"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("prompt_relay_plan"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Sigmas.Input("sigmas"),
                    io.Custom(RESULT).Input("previous_result", optional=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Custom(RUNTIME_TYPE).Output("runtime"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, raw_model, relay_model, relay_positive, relay_full_av_latent,
                prompt_relay_plan, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, sigmas, previous_result=None):
        return io.NodeOutput(*project_v1_local_relay(
            raw_model, relay_model, relay_positive, relay_full_av_latent,
            prompt_relay_plan, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, sigmas, previous_result,
        ))


class MiniMaxH3ChunkedV1RelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v1 · Segment Relay Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit the paired local Relay MODEL's actual forwards/routed attention "
                        "after exactly one v1 PASS2 segment. Relay+EAV combined calls belong "
                        "to the EAV audit, not this standalone report.",
            inputs=[io.Custom(RESULT).Input("segment_result"),
                    io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, segment_result, segment_spec, runtime):
        latent, report = audit_v1_local_relay(segment_result, segment_spec, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


NODES = [MiniMaxH3ChunkedV1RelayProjectEXPT8, MiniMaxH3ChunkedV1RelayAuditEXPT8]
