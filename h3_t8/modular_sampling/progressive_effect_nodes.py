"""External stage effect controls; old node schemas are untouched."""
from comfy_api.latest import io

from . import progressive_effects as effects
from .eav import CONFIG_TYPE
from .progressive_nodes import PLAN, RESTART, BOUNDARY, HIGH_RESULT, schema
from .results import canonical


class MiniMaxH3ProgressiveRelayStageApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External Relay + ONE Stage Conditioning",
            "Connect this phase's own paired Relay Conditioning MODEL/positive and original target-resolution guides. "
            "Rebinds actual LOW/HIGH packed layout and prepares stage guides, without sampling. Use instead of "
            "ONE Stage Conditioning, not after it. HIGH can use a different Relay Plan or be the only enabled phase. "
            "disabled removes only that authenticated Relay owner. Persistent effect identity is not yet certified.",
            [io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Custom(PLAN).Input("plan"), io.Combo.Input("phase", options=["low", "high"], default="low"),
             io.Combo.Input("mode", options=["disabled", "apply_exp"], default="apply_exp"),
             io.Combo.Input("guide_resize", options=["legacy_bilinear", "preserve_mean"], default="legacy_bilinear")],
            [io.Model.Output("model"), io.Conditioning.Output("positive"), io.Conditioning.Output("negative")])

    @classmethod
    def execute(cls, model, positive, negative, plan, phase="low", mode="apply_exp", guide_resize="legacy_bilinear"):
        return io.NodeOutput(*effects.apply_relay(model, positive, negative, plan, phase,
                                                  mode=mode, guide_resize=guide_resize))


class MiniMaxH3ProgressiveLowEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External LOW EAV Apply",
            "External Stage EAV Config -> this node -> LOW Sampler Only. Binds LOW's native mask and full absolute "
            "video clock, not local step progress. Optional Relay must be stage-bound first. No sampling here. "
            "CFG1; no learned/GPU/media quality or persistent effect identity qualification.",
            [io.Model.Input("model"), io.Custom(CONFIG_TYPE).Input("eav_config"),
             io.Custom(PLAN).Input("plan"), io.Latent.Input("low_source")], [io.Model.Output("model")])

    @classmethod
    def execute(cls, model, eav_config, plan, low_source):
        return io.NodeOutput(effects.apply_eav(model, eav_config, plan, "low", low_source))


class MiniMaxH3ProgressiveHighEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External HIGH EAV Apply",
            "External Stage EAV Config -> this node -> HIGH Sampler Only. Uses the exact HIGH restart plan and "
            "clean-source mask, not its noisy state as an anchor. LOW effects may be absent or different. "
            "Optional Relay must be stage-bound first. CFG1; persistent effect identity remains unqualified.",
            [io.Model.Input("model"), io.Custom(CONFIG_TYPE).Input("eav_config"), io.Custom(RESTART).Input("high_restart")],
            [io.Model.Output("model")])

    @classmethod
    def execute(cls, model, eav_config, high_restart):
        return io.NodeOutput(effects.apply_eav(model, eav_config, high_restart.plan, "high", high_restart))


class MiniMaxH3ProgressiveLowEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "LOW Effects Execution Audit",
            "Reads actual immutable LOW completion evidence. No sampling or mutable Apply-node counter lookup. "
            "Missing effects and incomplete user-stack coverage never count as verified application.",
            [io.Custom(BOUNDARY).Input("low_boundary")],
            [io.Custom(BOUNDARY).Output("low_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_boundary):
        return io.NodeOutput(low_boundary, canonical(effects.audit(low_boundary, "low")))


class MiniMaxH3ProgressiveHighEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "HIGH Effects Execution Audit",
            "Reads actual immutable completed HIGH evidence; does not run LOW or inspect an old cached Apply counter.",
            [io.Custom(HIGH_RESULT).Input("high_result")],
            [io.Custom(HIGH_RESULT).Output("high_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_result):
        return io.NodeOutput(high_result, canonical(effects.audit(high_result, "high")))


NODES = [MiniMaxH3ProgressiveRelayStageApplyEXPT8, MiniMaxH3ProgressiveLowEAVApplyEXPT8,
         MiniMaxH3ProgressiveHighEAVApplyEXPT8, MiniMaxH3ProgressiveLowEffectsAuditEXPT8,
         MiniMaxH3ProgressiveHighEffectsAuditEXPT8]
