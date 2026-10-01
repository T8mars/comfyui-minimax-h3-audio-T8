"""Opt-in Face Refine variant binding and post-sample audit nodes."""
from comfy_api.latest import io

from .face_stage import (bind_standard_face_stage, audit_standard_face_stage,
                         bind_parity_face_stage, audit_parity_face_stage,
                         bind_multiface_face_stage, audit_multiface_face_stage,
                         bind_window_face_stage, audit_window_face_stage)
from .face_relay import bind_standard_face_relay
from .face_local_relay import bind_local_face_relay

PLAN = io.Custom("H3_T8_FACE_REFINE_PLAN")
PARITY_PLAN = io.Custom("H3_T8_FACE_REFINE_PARITY_PLAN")
WINDOW_PLAN = io.Custom("H3_T8_FACE_REFINE_WINDOW_PLAN")
WINDOW_MAPPING = io.Custom("H3_T8_FACE_REFINE_WINDOW_MAPPING")
CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


class MiniMaxH3FaceStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceStageBindEXPT8",
            display_name="H3 Face · Bind Separate Refinement Stage (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Bind existing Face Plan/source and isolated low-denoise MODEL/SAMPLER/SIGMAS to one "
                        "editable stage. No sampling, VAE encoding or stitching. Standard Face Plan only; "
                        "other Face variants require separate adapters.",
            inputs=[PLAN.Input("face_plan"), io.Image.Input("source_frames"), io.Model.Input("model"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Combo.Input("audio_policy", options=["require_locked", "preserve_existing"],
                                   default="require_locked")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, face_plan, source_frames, model, sampler, sigmas, av_latent,
                audio_policy="require_locked"):
        return io.NodeOutput(*bind_standard_face_stage(face_plan, source_frames, model, sampler, sigmas,
                                                      av_latent, audio_policy))


class MiniMaxH3FaceStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceStageAuditEXPT8",
            display_name="H3 Face · Source-Bound Stage Audit (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="After Stage Sampler/Load, check actual candidate receipt against current Face Plan, "
                        "source frames and input AV before the existing Stitch node. Input audio mask lock does "
                        "not guarantee bit-exact sampled audio; deliver the original source audio separately. "
                        "Does not automatically approve or replace the candidate.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), PLAN.Input("face_plan"),
                    io.Image.Input("source_frames"), io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("candidate_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, face_plan, source_frames, av_latent):
        return io.NodeOutput(*audit_standard_face_stage(stage_result, face_plan, source_frames, av_latent))


class MiniMaxH3FaceParityStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceParityStageBindEXPT8",
            display_name="H3 Face Parity · Bind Separate Stage (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Bind the existing Parity Plan and currently connected MODEL/SAMPLER/SIGMAS/AV. "
                        "Keeps per-frame denoise and optional sampler-mask patch external. Exact-source default "
                        "Core er_sde can certify completion; unknown samplers run without portable identity.",
            inputs=[PARITY_PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Model.Input("model"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("av_latent"),
                    io.Combo.Input("audio_policy", options=["require_locked", "preserve_existing"],
                                   default="require_locked")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, face_plan, source_frames, model, sampler, sigmas, av_latent,
                audio_policy="require_locked"):
        return io.NodeOutput(*bind_parity_face_stage(face_plan, source_frames, model, sampler,
                                                    sigmas, av_latent, audio_policy))


class MiniMaxH3FaceParityStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceParityStageAuditEXPT8",
            display_name="H3 Face Parity · Source-Bound Audit (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="Check a sampled Parity/Per-Frame/Sampler-Mask candidate against the current plan, "
                        "source frames and input AV. Keep original source audio for delivery. Unknown sampler "
                        "completion stays unverified; no automatic acceptance or quality claim.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), PARITY_PLAN.Input("face_plan"),
                    io.Image.Input("source_frames"), io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("candidate_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, face_plan, source_frames, av_latent):
        return io.NodeOutput(*audit_parity_face_stage(stage_result, face_plan, source_frames, av_latent))


class MiniMaxH3MultiFaceStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MultiFaceStageBindEXPT8",
            display_name="H3 Multi-Face · Bind One Character Stage (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Bind one multi-person repair job, its model window and unchanged parent source to "
                        "a separate sampled stage. Verify absolute window/padding and actual external MODEL, "
                        "SAMPLER, SIGMAS and AV; each character needs its own instance. Does not auto-accept.",
            inputs=[PARITY_PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Image.Input("parent_frames"), io.Model.Input("model"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Combo.Input("audio_policy", options=["require_locked", "preserve_existing"],
                                   default="require_locked")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, face_plan, source_frames, parent_frames, model, sampler, sigmas, av_latent,
                audio_policy="require_locked"):
        return io.NodeOutput(*bind_multiface_face_stage(face_plan, source_frames, parent_frames,
                                                        model, sampler, sigmas, av_latent, audio_policy))


class MiniMaxH3MultiFaceStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MultiFaceStageAuditEXPT8",
            display_name="H3 Multi-Face · Source-Bound Candidate Audit (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="Before Parity Stitch and sequential Composite, verify that this sampled character "
                        "candidate belongs to the same job window, unchanged parent and source AV. "
                        "Unknown sampler completion is not portable or automatically accepted.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), PARITY_PLAN.Input("face_plan"),
                    io.Image.Input("source_frames"), io.Image.Input("parent_frames"),
                    io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("candidate_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, face_plan, source_frames, parent_frames, av_latent):
        return io.NodeOutput(*audit_multiface_face_stage(stage_result, face_plan, source_frames,
                                                         parent_frames, av_latent))


class MiniMaxH3FaceWindowStageBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceWindowStageBindEXPT8",
            display_name="H3 Face Window · Bind Separate Stage (T8 EXP)", category=CATEGORY,
            is_experimental=True,
            description="Bind one extracted repair window to its parent frames, signed window plan/mapping "
                        "and original/window audio before a separate Parity sampler stage. The Manual and "
                        "Studio review/commit nodes remain external; no automatic acceptance or delivery.",
            inputs=[PARITY_PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Image.Input("parent_frames"), WINDOW_PLAN.Input("window_plan"),
                    WINDOW_MAPPING.Input("window_mapping"), io.Audio.Input("source_audio"),
                    io.Audio.Input("window_audio"), io.Model.Input("model"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
                    io.Combo.Input("audio_policy", options=["require_locked", "preserve_existing"],
                                   default="require_locked")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, face_plan, source_frames, parent_frames, window_plan, window_mapping,
                source_audio, window_audio, model, sampler, sigmas, av_latent,
                audio_policy="require_locked"):
        return io.NodeOutput(*bind_window_face_stage(face_plan, source_frames, parent_frames,
            window_plan, window_mapping, source_audio, window_audio, model, sampler, sigmas,
            av_latent, audio_policy))


class MiniMaxH3FaceWindowStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceWindowStageAuditEXPT8",
            display_name="H3 Face Window · Source-Bound Candidate Audit (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="Before crop decode and Manual/Studio review, verify the sampled candidate still belongs "
                        "to the current parent source, repair window, frame map and original/window audio. "
                        "The sampled audio latent is not the delivered source track; no quality approval.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result"), PARITY_PLAN.Input("face_plan"),
                    io.Image.Input("source_frames"), io.Image.Input("parent_frames"),
                    WINDOW_PLAN.Input("window_plan"), WINDOW_MAPPING.Input("window_mapping"),
                    io.Audio.Input("source_audio"), io.Audio.Input("window_audio"),
                    io.Latent.Input("av_latent")],
            outputs=[io.Latent.Output("candidate_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, face_plan, source_frames, parent_frames, window_plan,
                window_mapping, source_audio, window_audio, av_latent):
        return io.NodeOutput(*audit_window_face_stage(stage_result, face_plan, source_frames,
            parent_frames, window_plan, window_mapping, source_audio, window_audio, av_latent))


class MiniMaxH3FaceRelayBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceRelayBindEXPT8",
            display_name="H3 Face · Bind External Prompt Relay (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Pair a full-clip standard Face refinement Stage with the stock external "
                        "Relay MODEL/CONDITIONING. The Relay AV checks packed layout only; face-crop "
                        "AV and untouched source audio stay authoritative. Not for Parity or windows.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_av_latent"),
                    io.Custom("H3_T8_PROMPT_RELAY_PLAN").Input("relay_plan"),
                    PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Latent.Input("stage_av_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context")],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, relay_positive, relay_av_latent, relay_plan,
                face_plan, source_frames, stage_av_latent, stage_context):
        return io.NodeOutput(*bind_standard_face_relay(
            model, relay_positive, relay_av_latent, relay_plan,
            face_plan, source_frames, stage_av_latent, stage_context))


class MiniMaxH3FaceLocalRelayBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FaceLocalRelayBindEXPT8",
            display_name="H3 Face Parity/Window · Bind Local Relay (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Pair a local Ref2VA Relay Plan and CONDITIONING with one signed Face repair window. "
                        "Parent/window mapping remains source-bound. The Plan is local, not a silent slice of "
                        "global film events. Sampled audio is not the delivered source soundtrack.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("relay_positive"),
                    io.Latent.Input("relay_av_latent"),
                    io.Custom("H3_T8_PROMPT_RELAY_PLAN").Input("relay_plan"),
                    PARITY_PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Latent.Input("stage_av_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context"),
                    io.Image.Input("parent_frames", optional=True),
                    WINDOW_PLAN.Input("window_plan", optional=True),
                    WINDOW_MAPPING.Input("window_mapping", optional=True),
                    io.Audio.Input("source_audio", optional=True),
                    io.Audio.Input("window_audio", optional=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, relay_positive, relay_av_latent, relay_plan,
                face_plan, source_frames, stage_av_latent, stage_context,
                parent_frames=None, window_plan=None, window_mapping=None,
                source_audio=None, window_audio=None):
        return io.NodeOutput(*bind_local_face_relay(
            model, relay_positive, relay_av_latent, relay_plan, face_plan,
            source_frames, stage_av_latent, stage_context,
            parent_frames=parent_frames, window_plan=window_plan,
            window_mapping=window_mapping, source_audio=source_audio,
            window_audio=window_audio))


NODES = [MiniMaxH3FaceStageBindEXPT8, MiniMaxH3FaceStageAuditEXPT8,
         MiniMaxH3FaceParityStageBindEXPT8, MiniMaxH3FaceParityStageAuditEXPT8,
         MiniMaxH3MultiFaceStageBindEXPT8, MiniMaxH3MultiFaceStageAuditEXPT8,
         MiniMaxH3FaceWindowStageBindEXPT8, MiniMaxH3FaceWindowStageAuditEXPT8]
