"""Append-only continuous HyperFlow effect binding and immutable audits."""
from comfy_api.latest import io

from . import hyperflow_effects as effects
from .hyperflow_nodes import BOUNDARY, RESULT, CATEGORY
from .results import canonical


def _schema(cls, title, description, inputs, outputs):
    return io.Schema(node_id=cls.__name__, display_name=f"H3 HyperFlow · {title} (T8 EXP)",
        category=CATEGORY, is_experimental=True, description=description, inputs=inputs, outputs=outputs)


def _bound_outputs():
    return [io.Model.Output("model"), io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
        io.Latent.Output("stage_template"), io.Sigmas.Output("sigmas"),
        io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")]


class MiniMaxH3HyperFlowHeadEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Bind External HEAD Effects",
            "Dedicated HyperFlow Loader -> optional paired Prompt Relay Conditioning -> this binding -> "
            "optional external Stage EAV Apply -> HEAD Only sampler. No sampling here. Use the same split "
            "on this binding and HEAD. SIGMAS are for effect validation, not another sampler or Progressive plan. "
            "CFG1 for active effects. Keep the paired MODEL and CONDITIONING together. No GPU/quality claim.",
            [io.Model.Input("model"), io.Latent.Input("av_latent"), io.Conditioning.Input("positive"),
             io.Conditioning.Input("negative"), io.Int.Input("split_interval", default=4, min=1, max=7)],
            _bound_outputs())

    @classmethod
    def execute(cls, model, av_latent, positive, negative, split_interval=4):
        return io.NodeOutput(*effects.bind_head(model, av_latent, positive, negative, split_interval))


class MiniMaxH3HyperFlowTailEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Bind External TAIL Effects",
            "Bind an independent TAIL MODEL/LoRAs and paired Relay conditions to the actual HEAD boundary. "
            "Optional external Stage EAV Apply uses returned sigmas/template/context. Template is NOT the raw "
            "x_sigma: keep continuous_boundary connected to TAIL Only. No new noise, upscale or HEAD execution. "
            "Supports loaded HEAD with certified identity. CFG1 for active effects; no GPU/quality claim.",
            [io.Custom(BOUNDARY).Input("continuous_boundary"), io.Model.Input("model"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative")], _bound_outputs())

    @classmethod
    def execute(cls, continuous_boundary, model, positive, negative):
        return io.NodeOutput(*effects.bind_tail(continuous_boundary, model, positive, negative))


class MiniMaxH3HyperFlowHeadEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "HEAD Effects Execution Audit",
            "Read immutable sampled HEAD effect coverage, not mutable Apply-node counters. "
            "Passes the exact typed boundary through. Missing/bypassed effects are not verified application.",
            [io.Custom(BOUNDARY).Input("continuous_boundary")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, continuous_boundary):
        return io.NodeOutput(continuous_boundary, canonical(effects.audit(continuous_boundary, "head")))


class MiniMaxH3HyperFlowTailEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "TAIL Effects Execution Audit",
            "Read immutable completed TAIL effect coverage; no HEAD rerun or old cached Apply counter lookup. "
            "Returns the typed completed result and its final AV for storage and decode. Not quality acceptance.",
            [io.Custom(RESULT).Input("completed_result")],
            [io.Custom(RESULT).Output("completed_result"), io.Latent.Output("av_latent"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, completed_result):
        report = effects.audit(completed_result, "tail")
        return io.NodeOutput(completed_result, completed_result.output, canonical(report))


NODES = [MiniMaxH3HyperFlowHeadEffectsBindEXPT8, MiniMaxH3HyperFlowTailEffectsBindEXPT8,
         MiniMaxH3HyperFlowHeadEffectsAuditEXPT8, MiniMaxH3HyperFlowTailEffectsAuditEXPT8]
