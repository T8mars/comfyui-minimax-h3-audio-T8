"""Append-only explicit Chunked learned-lift and PASS2 stages."""
from comfy_api.latest import io

from .chunked_stages import prepare_chunked_pass2, lift_chunked_segment, sample_chunked_pass2
from .chunked_effects import bind_chunked_eav, audit_chunked_eav
from .chunked_relay import RUNTIME_TYPE as RELAY_RUNTIME_TYPE, bind_full_clip_relay, audit_full_clip_relay
from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .eav import CONFIG_TYPE, RUNTIME_TYPE


CATEGORY = "T8/MiniMax H3/Modular Sampling/Chunked Experimental"
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
SPEC = "T8_CHUNKED_SOURCE_SEGMENT"
CONTEXT = "T8_CHUNKED_PASS2_CONTEXT"
RESULT = "T8_CHUNKED_PASS2_RESULT"


class MiniMaxH3ChunkedPass2PrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · Prepare Global PASS2 State (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Prepares v1-v4 original full-source mask and once-only global noise before any PASS2 "
                        "chunk. This is not sampling or a portable result. Use the same NOISE seed at PASS2. "
                        "The source first pass must be separately completed; v5 has another contract.",
            inputs=[io.Latent.Input("first_pass_latent"), io.Custom(PLAN).Input("plan"),
                    io.Noise.Input("noise")],
            outputs=[io.Custom(CONTEXT).Output("pass2_context"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, first_pass_latent, plan, noise):
        return io.NodeOutput(*prepare_chunked_pass2(first_pass_latent, plan, noise))


class MiniMaxH3ChunkedLearnedLiftEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · Learned Lift ONE Segment (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Runs exactly the legacy learned 3D latent upscaler on one sliced first-pass "
                        "segment. The PASS2 model, conditions, noise and sampler remain external and independent. "
                        "Does not sample or create a cache receipt.",
            inputs=[io.Latent.Input("source_segment"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan")],
            outputs=[io.Latent.Output("lifted_segment"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, source_segment, segment_spec, pass2_context, plan):
        return io.NodeOutput(*lift_chunked_segment(source_segment, segment_spec, pass2_context, plan))


class MiniMaxH3ChunkedPass2SegmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · Sample ONE PASS2 Segment (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Samples only this v1-v4 upscaled time segment with independently wired PASS2 "
                        "MODEL/CONDITIONING/NOISE/SAMPLER/SIGMAS. Keeps the original spatial tile and temporal "
                        "merge math; pass the prior result for the next segment. Final segment output keeps "
                        "the exact complete first-pass audio tensor. Stage-specific effects must be wired "
                        "outside this node and actually audited; no automatic cache or portable result.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Noise.Input("noise"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Custom(RESULT).Input("previous_result", optional=True),
                    io.Conditioning.Input("negative", optional=True),
                    io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.01)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("segment_result"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, positive, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, noise, sampler, sigmas, previous_result=None,
                negative=None, cfg=1.0):
        return io.NodeOutput(*sample_chunked_pass2(
            model, positive, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, noise, sampler, sigmas, previous_result, negative, cfg,
        ))


class MiniMaxH3ChunkedPass2EAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · External PASS2 EAV (one tile, T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Bind an external Stage EAV Config to exactly one v1-v4 PASS2 segment. "
                        "Currently accepts only full_frame_safe single-tile plans; multi-tile producer "
                        "coverage is not certified. Connect the output MODEL to this segment's PASS2 "
                        "and the runtime to the matching Chunked EAV Audit. The old graph is untouched.",
            inputs=[io.Model.Input("model"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Custom(CONFIG_TYPE).Input("eav_config")],
            outputs=[io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"),
                     io.String.Output("report_json"), io.Custom("T8_STAGE_CONTEXT").Output("stage_context")],
        )

    @classmethod
    def execute(cls, model, sigmas, source_segment, lifted_segment, segment_spec,
                pass2_context, plan, eav_config):
        return io.NodeOutput(*bind_chunked_eav(
            model, sigmas, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, eav_config,
        ))


class MiniMaxH3ChunkedPass2EAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · PASS2 EAV Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit the matching full_frame_safe segment's actual EAV forward/attention "
                        "calls after PASS2. A graph connection alone is not proof of effect coverage.",
            inputs=[io.Custom(RESULT).Input("segment_result"),
                    io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, segment_result, segment_spec, runtime):
        latent, report = audit_chunked_eav(segment_result, segment_spec, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


class MiniMaxH3ChunkedPass2RelayBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · Bind External PASS2 Relay (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Pair the existing external Prompt Relay Plan/Conditioning MODEL, positive and "
                        "AV template with exactly one v2-v4 full-clip/full-frame PASS2. Validates its "
                        "native packed layout, model-condition hash and lifted target geometry. "
                        "v1 temporal chunks and spatial multi-tile need separate local-layout adapters.",
            inputs=[io.Model.Input("relay_model"), io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_av_latent"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("prompt_relay_plan"),
                    io.Latent.Input("source_segment"), io.Latent.Input("lifted_segment"),
                    io.Custom(SPEC).Input("segment_spec"), io.Custom(CONTEXT).Input("pass2_context"),
                    io.Custom(PLAN).Input("plan"), io.Sigmas.Input("sigmas")],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Custom(RELAY_RUNTIME_TYPE).Output("runtime"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, relay_model, relay_positive, relay_av_latent, prompt_relay_plan,
                source_segment, lifted_segment, segment_spec, pass2_context, plan, sigmas):
        return io.NodeOutput(*bind_full_clip_relay(
            relay_model, relay_positive, relay_av_latent, prompt_relay_plan,
            source_segment, lifted_segment, segment_spec, pass2_context, plan, sigmas,
        ))


class MiniMaxH3ChunkedPass2RelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · PASS2 Relay Actual Calls (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Audit actual standalone Relay forward and routed-attention calls after "
                        "this full-clip PASS2. Relay+EAV combined calls belong to the EAV audit.",
            inputs=[io.Custom(RESULT).Input("segment_result"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(RELAY_RUNTIME_TYPE).Input("runtime")],
            outputs=[io.Latent.Output("cumulative_av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, segment_result, segment_spec, runtime):
        latent, report = audit_full_clip_relay(segment_result, segment_spec, runtime)
        return io.NodeOutput(latent, report, ui={"text": (report,)})


NODES = [MiniMaxH3ChunkedPass2PrepareEXPT8, MiniMaxH3ChunkedLearnedLiftEXPT8,
         MiniMaxH3ChunkedPass2SegmentEXPT8, MiniMaxH3ChunkedPass2EAVApplyEXPT8,
         MiniMaxH3ChunkedPass2EAVAuditEXPT8, MiniMaxH3ChunkedPass2RelayBindEXPT8,
         MiniMaxH3ChunkedPass2RelayAuditEXPT8]
