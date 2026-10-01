"""Explicit accepted-parent context and HIGH-prefix ports for FastH3 V2."""

from comfy_api.latest import io

from ..long_video import CONTEXT_TYPE_NAME
from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .fast_h3_v2_continuation import (
    accepted_phase_context, lock_accepted_high_prefix, plan_accepted_window,
    accepted_delivery_window, project_accepted_relay, reconcile_accepted_high,
)

CONTEXTS = "T8_CONTINUATION_STAGE_CONTEXTS"
CATEGORY = "T8/MiniMax H3/Modular Sampling/Continuation Experimental"


class MiniMaxH3FastH3V2AcceptedContextEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted LOW / HIGH Context Port (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Expose one authenticated accepted-parent context to separate native Long Video or "
                        "Prompt Relay conditioning. LOW is accepted RGB24 resized and VAE-encoded; HIGH is "
                        "the completed AV tail. No hidden sampler or HIGH dependency on LOW conditions.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"),
                    io.Combo.Input("phase", options=["low", "high"], default="low")],
            outputs=[io.Custom(CONTEXT_TYPE_NAME).Output("context"), io.Int.Output("segment_index"),
                     io.Int.Output("context_frames"), io.Int.Output("width"), io.Int.Output("height"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, phase="low"):
        return io.NodeOutput(*accepted_phase_context(contexts, phase))


class MiniMaxH3FastH3V2AcceptedWindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted Remainder Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Choose a compact 90-frame remainder or the old loop's fixed 124-frame "
                        "second render window. Both trim the same 22-context + 68-new delivery; "
                        "the fixed variant discards a 34-frame suffix. Geometry only; no sampler.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"),
                    io.Int.Input("total_accepted_frames", default=192, min=1, max=10000000),
                    io.Combo.Input("render_policy", options=["compact_remainder", "old_fixed_124"],
                                   default="compact_remainder", optional=True)],
            outputs=[io.Int.Output("length"), io.Int.Output("new_frames"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, total_accepted_frames=192, render_policy="compact_remainder"):
        return io.NodeOutput(*plan_accepted_window(contexts, total_accepted_frames, render_policy))


class MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted HIGH Prefix After Reconcile (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Apply the old V2 continuation HIGH video prefix and optional three-step mask ramp "
                        "only after separate learned upscale/reconcile. Keeps the same completed-context "
                        "source and the reconciled audio tensor; no sampling or delivery is hidden here.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"), io.Latent.Input("reconciled_av"),
                    io.Combo.Input("mode", options=["high_native_mask_exp", "high_native_mask_ramp_exp"],
                                   default="high_native_mask_ramp_exp")],
            outputs=[io.Latent.Output("high_av"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, reconciled_av, mode="high_native_mask_ramp_exp"):
        return io.NodeOutput(*lock_accepted_high_prefix(contexts, reconciled_av, mode))


class MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted Timeline Relay Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Project an external global Relay plan using the selected accepted parent's exact "
                        "timeline; output plugs into separate LOW/HIGH native Long Video Relay conditioning. "
                        "Requires Dense EXP when Relay is active on V2; trained VSA has no bias adapter yet.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("global_plan"),
                    io.Int.Input("length", default=124, min=5, step=17),
                    io.Int.Input("accepted_end_frame", default=0, min=0, max=10000000,
                                 optional=True)],
            outputs=[io.Custom(PROMPT_RELAY_PLAN_TYPE).Output("projected_plan"),
                     io.String.Output("compiled_prompt"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, global_plan, length=124, accepted_end_frame=0):
        return io.NodeOutput(*project_accepted_relay(contexts, global_plan, length, accepted_end_frame))


class MiniMaxH3FastH3V2AcceptedReconcileEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted Joint-Audio Reconcile (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Reproduce the old V2 4+4 continuation legacy-policy audio selection from the "
                        "learned LOW x0 and fresh HIGH template. Keep separate HIGH prefix node AFTER this "
                        "reconcile; no hidden sampler or audio-freeze shortcut.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"), io.Latent.Input("learned_latent"),
                    io.Latent.Input("highres_template"), io.Conditioning.Input("positive")],
            outputs=[io.Latent.Output("reconciled_av"), io.Conditioning.Output("positive"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, learned_latent, highres_template, positive):
        return io.NodeOutput(*reconcile_accepted_high(contexts, learned_latent, highres_template, positive))


class MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Accepted Remainder Delivery Port (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Expose the authenticated parent identity and exact final-window AV trim coordinates. "
                        "For 124+22+68, trim decoded frame 22 onward to 68 fresh frames. Does not trim, "
                        "save a candidate, accept it or certify the current edited job contract.",
            inputs=[io.Custom(CONTEXTS).Input("contexts"),
                    io.Int.Input("total_accepted_frames", default=192, min=1, max=10000000),
                    io.Combo.Input("render_policy", options=["compact_remainder", "old_fixed_124"],
                                   default="compact_remainder", optional=True)],
            outputs=[io.String.Output("chain_id"), io.Int.Output("segment_index"),
                     io.String.Output("parent_candidate_id"), io.Int.Output("parent_manifest_revision"),
                     io.Float.Output("timeline_start_seconds"), io.Float.Output("trim_start_seconds"),
                     io.Float.Output("final_duration_seconds"), io.Boolean.Output("save_context"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, total_accepted_frames=192, render_policy="compact_remainder"):
        return io.NodeOutput(*accepted_delivery_window(contexts, total_accepted_frames, render_policy))


NODES = (MiniMaxH3FastH3V2AcceptedContextEXPT8, MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8,
         MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8, MiniMaxH3FastH3V2AcceptedReconcileEXPT8,
         MiniMaxH3FastH3V2AcceptedWindowEXPT8, MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8)
