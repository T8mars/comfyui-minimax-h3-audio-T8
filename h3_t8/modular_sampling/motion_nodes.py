"""Opt-in source-bound Motion Recovery stage nodes; legacy paths are unchanged."""
from comfy_api.latest import io

from .motion_stage import bind_motion_stage, audit_motion_stage
from .motion_effects import retimed_relay_length, bind_motion_relay

PLAN = io.Custom("H3_T8_MOTION_RECOVERY_PLAN")
CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


class MiniMaxH3MotionStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MotionStageBindEXPT8",
            display_name="H3 Motion Recovery · Bind Separate Pass 2 (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Bind the signed motion plan, original frames/audio, expanded frames/audio seed, "
                        "actual AV and external MODEL/SAMPLER/SIGMAS before one independent refinement pass. "
                        "Does not run a VAE, sampler, phase vocoder or delivery decision.",
            inputs=[PLAN.Input("motion_plan"), io.Image.Input("baseline_frames"),
                    io.Audio.Input("baseline_audio"), io.Image.Input("smeared_frames"),
                    io.Audio.Input("smeared_audio"), io.String.Input("prepare_report_json"),
                    io.Model.Input("model"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("av_latent"), PLAN.Input("parent_plan", optional=True),
                    io.Image.Input("parent_frames", optional=True),
                    io.Audio.Input("parent_audio", optional=True)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, motion_plan, baseline_frames, baseline_audio, smeared_frames,
                smeared_audio, prepare_report_json, model, sampler, sigmas, av_latent,
                parent_plan=None, parent_frames=None, parent_audio=None):
        return io.NodeOutput(*bind_motion_stage(motion_plan, baseline_frames, baseline_audio,
            smeared_frames, smeared_audio, prepare_report_json, model, sampler, sigmas, av_latent,
            parent_plan=parent_plan, parent_frames=parent_frames, parent_audio=parent_audio))


class MiniMaxH3MotionStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MotionStageAuditEXPT8",
            display_name="H3 Motion Recovery · Audit Pass 2 Candidate (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="Before Recover AV and optional Window Collect/Gate, recheck that the sampled candidate "
                        "belongs to the current motion plan and exact connected source/retimed AV. "
                        "Original pass-1 audio remains the safe delivery default; no quality approval.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), PLAN.Input("motion_plan"),
                    io.Image.Input("baseline_frames"), io.Audio.Input("baseline_audio"),
                    io.Image.Input("smeared_frames"), io.Audio.Input("smeared_audio"),
                    io.String.Input("prepare_report_json"), io.Latent.Input("av_latent"),
                    PLAN.Input("parent_plan", optional=True),
                    io.Image.Input("parent_frames", optional=True),
                    io.Audio.Input("parent_audio", optional=True)],
            outputs=[io.Latent.Output("candidate_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, motion_plan, baseline_frames, baseline_audio, smeared_frames,
                smeared_audio, prepare_report_json, av_latent,
                parent_plan=None, parent_frames=None, parent_audio=None):
        return io.NodeOutput(*audit_motion_stage(stage_result, motion_plan, baseline_frames,
            baseline_audio, smeared_frames, smeared_audio, prepare_report_json, av_latent,
            parent_plan=parent_plan, parent_frames=parent_frames, parent_audio=parent_audio))


class MiniMaxH3MotionRelayLengthEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MotionRelayLengthEXPT8",
            display_name="H3 Motion Recovery · Retimed Relay Length (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Expose the signed expanded H3 frame count to an independent Prompt Relay Plan. "
                        "Rejects abstain and unaligned timelines; no model call or sampling.",
            inputs=[PLAN.Input("motion_plan")],
            outputs=[io.Int.Output("length"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, motion_plan):
        return io.NodeOutput(*retimed_relay_length(motion_plan))


class MiniMaxH3MotionRelayBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MotionRelayBindEXPT8",
            display_name="H3 Motion Recovery · Bind External Prompt Relay (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Pair the stock external Relay MODEL/CONDITIONING with the signed retimed "
                        "pass-2 AV. Relay's generated AV is used only to authenticate the packed "
                        "layout, never to replace Motion's VAE/phase-vocoder source. Connect Stage "
                        "EAV Config→Apply→Audit separately to observe actual calls.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_av_latent"),
                    io.Custom("H3_T8_PROMPT_RELAY_PLAN").Input("relay_plan"),
                    PLAN.Input("motion_plan"), io.Latent.Input("stage_av_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context")],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, relay_positive, relay_av_latent, relay_plan,
                motion_plan, stage_av_latent, stage_context):
        return io.NodeOutput(*bind_motion_relay(model, relay_positive, relay_av_latent,
            relay_plan, motion_plan, stage_av_latent, stage_context))


NODES = [MiniMaxH3MotionStageBindEXPT8, MiniMaxH3MotionStageAuditEXPT8,
         MiniMaxH3MotionRelayLengthEXPT8, MiniMaxH3MotionRelayBindEXPT8]
