"""Append-only standard-Core full8/partial4/fresh4 stage interfaces."""
from comfy_api.latest import io

from . import hyperflow_fresh as stages, nodes as common
from .results import StageResult, canonical
from .storage import load_stage, fingerprint_stage

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


def schema(cls, title, description, inputs, outputs):
    return io.Schema(node_id=cls.__name__, display_name=f"H3 HyperFlow · {title} (T8 EXP)",
        category=CATEGORY, is_experimental=True, description=description, inputs=inputs, outputs=outputs)


class MiniMaxH3HyperFlowFreshLoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ..nodes_hyperflow_advanced import _weight_options
        return schema(cls, "Fresh Route Loader / Original Endpoint",
            "Load the original HyperFlow adapter for independently scheduled fresh stages. Uses original "
            "base time tensors recovered read-only from Core backups even when a sibling is resident. "
            "Same two-time formulas, no shared-model unload or LoRA removal. Legacy Loader stays unchanged. "
            "Use a full native H3 base; content LoRAs may be independent on each branch.",
            [io.Model.Input("model"), io.Combo.Input("hyperflow_file", options=_weight_options())],
            [io.Model.Output("model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, hyperflow_file):
        from ..nodes_hyperflow_advanced import load_hyperflow_original, _resolve
        selected, _, report = stages.install(model, load_hyperflow_original(_resolve(hyperflow_file)))
        return io.NodeOutput(selected, canonical(report))


class MiniMaxH3HyperFlowFreshStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Separate Full8 / Partial4 / Fresh4 Setup",
            "Dedicated HyperFlow Loader -> optional paired Prompt Relay Conditioning -> this setup -> "
            "optional external Stage EAV -> standard Core sampler or ONE Stage Sampler + Result. "
            "No sampling here. Full8 LOW uses completed output for learned3D; partial4 LOW uses denoised_output. "
            "HIGH uses fresh external NOISE and original audio rebase, not continuous raw x_sigma. "
            "Models, content LoRAs, prompts, noise and effects are independent. Active effects require CFG1.",
            [io.Model.Input("model"), io.Latent.Input("av_latent"),
             io.Combo.Input("stage", options=list(stages.STAGES), default=stages.STAGES[0])],
            [io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
             io.Custom("T8_STAGE_CONTEXT").Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, stage=stages.STAGES[0]):
        return io.NodeOutput(*stages.build_stage(model, av_latent, stage))


class MiniMaxH3HyperFlowFreshLiftInputEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Select LOW Learned-Lift Input",
            "Read a verified LOW Stage Result (including explicitly loaded frozen LOW). Selects output for "
            "complete8 and denoised_output for partial4. Zero diffusion, noise or upscale calls. Connect the "
            "selected AV to the existing learned3D and external reconcile nodes; never to continuous TAIL.",
            [io.Custom("T8_STAGE_RESULT").Input("stage_result")],
            [io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result):
        return io.NodeOutput(*stages.lift_input(stage_result))


class MiniMaxH3HyperFlowFreshStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Fresh Stage Execution Audit",
            "Pass the exact Stage Result and both original Core outputs through. Reads immutable actual "
            "absolute step/EAV/Relay coverage, not stale Apply counters. Does not sample or certify quality.",
            [io.Custom("T8_STAGE_RESULT").Input("stage_result")],
            [io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.Latent.Output("output"),
             io.Latent.Output("denoised_output"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result):
        if type(stage_result) is not StageResult:
            raise ValueError("Connect an actual HyperFlow fresh Stage Result")
        receipt = stage_result.verify()
        if receipt["request"]["stage_context"]["recipe"] not in stages.RECIPES:
            raise ValueError("Not a HyperFlow fresh stage result")
        report = {"schema": stages.SCHEMA, "receipt_sha256": receipt["receipt_sha256"],
            "verified_recipe_completion": receipt["verified_recipe_completion"],
            "portable_identity": receipt["portable_identity"],
            "execution": receipt["execution"]["hyperflow_fresh"], "sampling_calls": 0}
        return io.NodeOutput(stage_result, stage_result.output, stage_result.denoised_output, canonical(report))


class MiniMaxH3HyperFlowFreshStageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load Exact Frozen Fresh Stage",
            "Read an exact Stage Save artifact by mandatory path and SHA; no sampling or model loading. "
            "LOW -> Lift Input -> original learned3D/reconcile -> independent HIGH with fresh NOISE. "
            "Completed HIGH can be decoded directly without any sampler. Does not match current LOW settings "
            "automatically. Existing generic Load schema and old workflows remain unchanged.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default=""),
             io.Combo.Input("expected_stage", options=list(stages.STAGES), default=stages.STAGES[0])],
            [io.Latent.Output("output"), io.Latent.Output("denoised_output"),
             io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
             io.Custom("T8_STAGE_RESULT").Output("stage_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256, expected_stage=stages.STAGES[0]):
        if expected_stage not in stages.STAGES:
            raise ValueError("Select a HyperFlow full8/partial4/fresh4 stage")
        values = load_stage(common._stage_store_root(), artifact_path, artifact_sha256, expected_stage)
        MiniMaxH3HyperFlowFreshStageAuditEXPT8.execute(values[3])
        return io.NodeOutput(*values)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, expected_stage=stages.STAGES[0]):
        try:
            return fingerprint_stage(common._stage_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3HyperFlowFreshStageSetupEXPT8, MiniMaxH3HyperFlowFreshLiftInputEXPT8,
         MiniMaxH3HyperFlowFreshStageAuditEXPT8, MiniMaxH3HyperFlowFreshStageLoadEXPT8,
         MiniMaxH3HyperFlowFreshLoaderEXPT8]
