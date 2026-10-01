"""Append-only external Relay projection and paired per-phase application."""
from comfy_api.latest import io

from . import hyperflow_p7_effects as effects
from .eav import CONFIG_TYPE, RUNTIME_TYPE, apply_stage_eav
from .hyperflow_p7_nodes import CONTEXTS, PHASE, schema
from .continuation_nodes import PROMPT_RELAY_PLAN_TYPE


RELAY = "T8_HYPERFLOW_P7_RELAY_WINDOW"


class MiniMaxH3HyperFlowP7RelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External Relay Window", "Project an external global Prompt Relay Plan onto the "
            "authenticated P7 segment's exact frame window. Connect compiled_prompt to either phase's "
            "Conditioning prompt. LOW/HIGH may use separate Plans; no MODEL patch or sampling here.",
            [io.Custom(CONTEXTS).Input("contexts"),
             io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("global_plan"),
             io.Int.Input("length", default=124, min=5, step=17),
             io.Int.Input("accepted_end_frame", default=-1, min=-1, max=10000000)],
            [io.Custom(RELAY).Output("projected_relay"), io.String.Output("compiled_prompt"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, global_plan, length=124, accepted_end_frame=-1):
        value = effects.project_relay(contexts, global_plan, length,
            accepted_end_frame=None if accepted_end_frame == -1 else accepted_end_frame)
        return io.NodeOutput(value, value.projected["compiled_prompt"], value.contract_json)


class MiniMaxH3HyperFlowP7RelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Pair Relay to ONE Phase", "Bind this P7 phase's original motion guides and "
            "projected Relay events to both MODEL and CONDITIONING. Apply before LOW/HIGH Setup, then "
            "optionally apply external Stage EAV after Setup. The completed Stage Result records actual "
            "Relay/EAV calls; an Apply node alone is not proof of execution.",
            [io.Model.Input("model"), io.Custom(PHASE).Input("prepared_phase"), io.Clip.Input("clip"),
             io.Custom(RELAY).Input("projected_relay"),
             io.Int.Input("query_chunk_rows", default=256, min=32, max=2048),
             io.Combo.Input("mode", options=["disabled", "apply_exp"], default="apply_exp")],
            [io.Model.Output("model"), io.Conditioning.Output("positive"),
             io.Conditioning.Output("negative"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, prepared_phase, clip, projected_relay,
                query_chunk_rows=256, mode="apply_exp"):
        return io.NodeOutput(*effects.apply_relay(model, prepared_phase, clip, projected_relay,
            query_chunk_rows=query_chunk_rows, mode=mode))


class MiniMaxH3HyperFlowP7StageEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External Stage EAV for This Phase",
            "Apply external Stage EAV with this authenticated P7 LOW/HIGH phase. The phase binds "
            "the original Long Video motion-keyframe layout; generic Stage EAV remains unchanged. "
            "Connect after phase Setup, and inspect the completed stage audit for actual calls.",
            [io.Model.Input("model"), io.Sigmas.Input("sigmas"), io.Latent.Input("av_latent"),
             io.Custom("T8_STAGE_CONTEXT").Input("stage_context"),
             io.Custom(CONFIG_TYPE).Input("eav_config"), io.Custom(PHASE).Input("prepared_phase")],
            [io.Model.Output("model"), io.Custom(RUNTIME_TYPE).Output("runtime"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sigmas, av_latent, stage_context, eav_config, prepared_phase):
        return io.NodeOutput(*apply_stage_eav(model, sigmas, av_latent, stage_context,
            eav_config, p7_phase=prepared_phase))


NODES = [MiniMaxH3HyperFlowP7RelayProjectEXPT8, MiniMaxH3HyperFlowP7RelayApplyEXPT8,
         MiniMaxH3HyperFlowP7StageEAVApplyEXPT8]
