"""Opt-in Audio Refine stage boundaries for all three existing Plan families."""
from comfy_api.latest import io

from .. import audio_refine_advanced as refine
from .audio_refine_stage import (
    bind_audio_refine_stage, audit_audio_refine_stage,
    audit_audio_refine_tail_delivery,
)

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
BOUNDARY = io.Custom("T8_AUDIO_REFINE_STAGE_BOUNDARY")


class _AudioRefineBind(io.ComfyNode):
    PLAN_TYPE = None
    VARIANT = None
    NODE_ID = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.NODE_ID,
            display_name=f"H3 Audio Refine {cls.VARIANT} · Bind Separate Tail (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Check signed Plan, existing Setup controls, original AV and exact 0-video/1-audio "
                        "mask without changing MODEL/NOISE/GUIDER/SAMPLER/SIGMAS. Empty-SIGMAS abstain "
                        "remains a no-sample path. No automatic cache or quality approval.",
            inputs=[io.Custom(cls.PLAN_TYPE).Input("plan"), io.Latent.Input("original_av_latent"),
                    io.Model.Input("model"), io.Noise.Input("noise"), io.Guider.Input("guider"),
                    io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("stage_latent"), io.String.Input("setup_report_json")],
            outputs=[io.Model.Output("model"), io.Noise.Output("noise"), io.Guider.Output("guider"),
                     io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Latent.Output("stage_latent"), BOUNDARY.Output("stage_boundary"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, plan, original_av_latent, model, noise, guider, sampler,
                sigmas, stage_latent, setup_report_json):
        return io.NodeOutput(*bind_audio_refine_stage(plan, original_av_latent, model, noise,
            guider, sampler, sigmas, stage_latent, setup_report_json,
            expected_variant=cls.VARIANT))


class MiniMaxH3AudioRefineDualClockStageBindEXPT8(_AudioRefineBind):
    PLAN_TYPE = refine.AUDIO_REFINE_PLAN_TYPE
    VARIANT = "dual_clock"
    NODE_ID = "MiniMaxH3AudioRefineDualClockStageBindEXPT8"


class MiniMaxH3AudioRefineDualModelStageBindEXPT8(_AudioRefineBind):
    PLAN_TYPE = refine.AUDIO_REFINE_PHASE2_PLAN_TYPE
    VARIANT = "dual_model"
    NODE_ID = "MiniMaxH3AudioRefineDualModelStageBindEXPT8"


class MiniMaxH3AudioRefineCompatStageBindEXPT8(_AudioRefineBind):
    PLAN_TYPE = refine.AUDIO_REFINE_COMPAT_PLAN_TYPE
    VARIANT = "compatibility"
    NODE_ID = "MiniMaxH3AudioRefineCompatStageBindEXPT8"


class _AudioRefineAudit(io.ComfyNode):
    PLAN_TYPE = None
    NODE_ID = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.NODE_ID,
            display_name="H3 Audio Refine · Audit Separate Tail Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Recheck source and signed stage boundary after the existing external sampler. "
                        "ABSTAIN must return original AV unchanged. Candidate audio is not accepted here; "
                        "the existing Quality Gate and human listening remain authoritative.",
            inputs=[BOUNDARY.Input("stage_boundary"), io.Custom(cls.PLAN_TYPE).Input("plan"),
                    io.Latent.Input("original_av_latent"), io.Latent.Input("stage_latent"),
                    io.String.Input("setup_report_json"), io.Latent.Input("candidate_av_latent")],
            outputs=[io.Latent.Output("candidate_av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_boundary, plan, original_av_latent, stage_latent,
                setup_report_json, candidate_av_latent):
        return io.NodeOutput(*audit_audio_refine_stage(stage_boundary, plan, original_av_latent,
            stage_latent, setup_report_json, candidate_av_latent))


class MiniMaxH3AudioRefineDualClockStageAuditEXPT8(_AudioRefineAudit):
    PLAN_TYPE = refine.AUDIO_REFINE_PLAN_TYPE
    NODE_ID = "MiniMaxH3AudioRefineDualClockStageAuditEXPT8"


class MiniMaxH3AudioRefineDualModelStageAuditEXPT8(_AudioRefineAudit):
    PLAN_TYPE = refine.AUDIO_REFINE_PHASE2_PLAN_TYPE
    NODE_ID = "MiniMaxH3AudioRefineDualModelStageAuditEXPT8"


class MiniMaxH3AudioRefineCompatStageAuditEXPT8(_AudioRefineAudit):
    PLAN_TYPE = refine.AUDIO_REFINE_COMPAT_PLAN_TYPE
    NODE_ID = "MiniMaxH3AudioRefineCompatStageAuditEXPT8"


class MiniMaxH3AudioRefineTailDeliveryAuditEXPT8(io.ComfyNode):
    NODE_ID = "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name="H3 Audio Refine · Audit Sample or Abstain (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="For sampled tails, run the original strict two-pass audio audit unchanged. "
                        "For signed ABSTAIN tails, require exact original AV and pass it through "
                        "without inventing a sampling mask or claiming an accepted candidate.",
            inputs=[io.Latent.Input("second_pass_input"),
                    io.Latent.Input("second_pass_output"),
                    io.Float.Input("expected_audio_strength", default=1.0, min=0.0, max=1.0),
                    io.Boolean.Input("fail_on_locked_mismatch", default=False),
                    io.Float.Input("locked_atol", default=0.0, min=0.0, max=1.0),
                    BOUNDARY.Input("stage_boundary"),
                    io.String.Input("stage_report_json")],
            outputs=[io.Latent.Output("verified_av_latent"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, second_pass_input, second_pass_output,
                expected_audio_strength, fail_on_locked_mismatch, locked_atol,
                stage_boundary, stage_report_json):
        values = audit_audio_refine_tail_delivery(
            stage_boundary, stage_report_json, second_pass_input,
            second_pass_output, expected_audio_strength,
            fail_on_locked_mismatch, locked_atol)
        return io.NodeOutput(*values, ui={"text": (values[1],)})


NODES = [MiniMaxH3AudioRefineDualClockStageBindEXPT8,
         MiniMaxH3AudioRefineDualClockStageAuditEXPT8,
         MiniMaxH3AudioRefineDualModelStageBindEXPT8,
         MiniMaxH3AudioRefineDualModelStageAuditEXPT8,
         MiniMaxH3AudioRefineCompatStageBindEXPT8,
         MiniMaxH3AudioRefineCompatStageAuditEXPT8,
         MiniMaxH3AudioRefineTailDeliveryAuditEXPT8]
