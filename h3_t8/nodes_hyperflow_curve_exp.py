"""Opt-in pruned HyperFlow approximation: independent fit, stages and effects.

The full-structure HyperFlow IDs and implementations are not replaced. Loading
a fit does not certify a MODEL or turn its approximation into quality approval.
"""
from pathlib import Path
import json
import uuid

import folder_paths
from comfy_api.latest import io
from safetensors import safe_open

from . import hyperflow_curve_fit_exp as fitting
from . import hyperflow_curve_runtime_exp as runtime
from . import hyperflow_curve_loader_exp as loader
from .hyperflow_weights_advanced import load_hyperflow_original
from .nodes_hyperflow_advanced import _resolve, _weight_options
from .modular_sampling import hyperflow_curve as stages
from .modular_sampling import hyperflow_curve_effects as effects
from .modular_sampling import hyperflow_curve_storage as storage
from .modular_sampling.hyperflow_nodes import _progress
from .modular_sampling.progressive_nodes import _generate_noise, _noise_seed
from .modular_sampling.results import canonical

CATEGORY = "T8/MiniMax H3/Modular Sampling/HyperFlow Curves Experimental"
FIT = "T8_HYPERFLOW_CURVE_FIT_V1"
BOUNDARY = "T8_HYPERFLOW_CURVE_CONTINUOUS_BOUNDARY_V1"
RESULT = "T8_HYPERFLOW_CURVE_COMPLETED_AV_V1"
folder_paths.add_model_folder_path("hyperflow_curve_fits", str(Path(folder_paths.models_dir) / "hyperflow" / "curve_fits"))


def _schema(cls, title, description, inputs, outputs, *, output=False):
    return io.Schema(node_id=cls.__name__, display_name="H3 HyperFlow Curve · " + title + " (T8 EXP)",
        category=CATEGORY, is_experimental=True, description=description,
        inputs=inputs, outputs=outputs, is_output_node=output)


def _bases():
    return folder_paths.get_filename_list("diffusion_models") or ["missing_native_H3_checkpoint"]


def _base(name):
    return Path(folder_paths.get_full_path_or_raise("diffusion_models", name))


def _fit_path(name, absolute_path):
    if absolute_path.strip():
        path = Path(absolute_path.strip())
        if not path.is_absolute():
            raise ValueError("Advanced fit path must be absolute")
        return path.resolve(strict=True)
    return Path(folder_paths.get_full_path_or_raise("hyperflow_curve_fits", name))


def _bounded_fit_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("Curve fit must be an existing safetensors asset")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Curve fit exceeds 16MiB bound")
    return fitting._file(path)


def _load_asset(path, expected_sha256=""):
    # Bounded asset loading only. The selected base and actual MODEL are
    # independently verified in Model Apply; metadata is not a certificate.
    path = Path(path).resolve(strict=True)
    before = _bounded_fit_identity(path)
    if expected_sha256 and before["sha256"] != storage._digest(expected_sha256):
        raise ValueError("Selected curve fit SHA mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(metadata) != {"t8_curve_fit"} or len(metadata["t8_curve_fit"]) > 262144:
            raise ValueError("Not a bounded T8 curve-fit asset")
        data = json.loads(metadata["t8_curve_fit"], object_pairs_hook=storage._unique)
    result = fitting.load_fit(path, base_sha256=data["base"]["sha256"], adapter_sha256=data["adapter"]["sha256"])
    if _bounded_fit_identity(path) != before:
        raise ValueError("Selected curve fit changed during loading")
    return result


def _fit_outputs():
    return [io.Custom(FIT).Output("curve_fit"), io.String.Output("fit_path"),
        io.String.Output("fit_sha256"), io.String.Output("report_json")]


def _fit_report(fit):
    return canonical({"schema": fitting.SCHEMA, "fit_sha256": fit.sha256,
        "selected_source_sha256": {key: fit.metadata[key]["sha256"] for key in ("base", "teacher", "adapter")},
        "projections": 51, "approximation": True, "model_checked": False, "quality_accepted": False})


def _store_root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "curve_stage_artifacts"


def _fingerprint(path, kind):
    try:
        return storage.fingerprint(_store_root(), path, kind)
    except (OSError, ValueError, RuntimeError):
        return float("nan")


class MiniMaxH3HyperFlowCurveFitBuildEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Build Explicit Fit", "Explicit potentially expensive producer. Select the native pruned base, "
            "matching full-time teacher and ORIGINAL HyperFlow adapter. Builds all51 independent projections and pins, "
            "measures residuals, writes a NEW unique asset under output/MiniMaxH3/curve_fits. No sampling, conversion "
            "of the base, hidden download, overwrite, full-model equivalence or quality certificate.",
            [io.Combo.Input("base_file", options=_bases()), io.Combo.Input("teacher_file", options=_bases()),
             io.Combo.Input("hyperflow_file", options=_weight_options()),
             io.Combo.Input("device", options=["cpu", "cuda"], default="cpu")], _fit_outputs())

    @classmethod
    def execute(cls, base_file, teacher_file, hyperflow_file, device="cpu"):
        import comfy.model_management
        import comfy.utils
        if device not in {"cpu", "cuda"}:
            raise ValueError("Choose explicit CPU or CUDA fit device")
        root = storage._root(Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "curve_fits", create=True)
        output = storage._path(root, "curve-" + uuid.uuid4().hex + ".safetensors")
        progress = comfy.utils.ProgressBar(51)
        result = fitting.build_fit(_base(base_file), _base(teacher_file), _resolve(hyperflow_file), output,
            device=device, progress=lambda index, _error: progress.update_absolute(index, 51),
            cancel=comfy.model_management.throw_exception_if_processing_interrupted)
        return io.NodeOutput(result, str(result.path), result.sha256, _fit_report(result))

    @classmethod
    def fingerprint_inputs(cls, base_file, teacher_file, hyperflow_file, device="cpu"):
        return tuple(fitting._file(path)["sha256"] for path in (_base(base_file), _base(teacher_file), _resolve(hyperflow_file)))


class MiniMaxH3HyperFlowCurveFitLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Selected Fit", "Reads models/hyperflow/curve_fits, or an explicit absolute advanced path "
            "for an already built asset. Optional SHA pins exact bytes. Does not load or certify a MODEL. "
            "Model Apply separately checks original adapter and selected base/basis. Never creates a fit on a cache miss.",
            [io.Combo.Input("fit_file", options=folder_paths.get_filename_list("hyperflow_curve_fits") or ["missing_curve_fit"]),
             io.String.Input("absolute_path", default=""), io.String.Input("expected_sha256", default="")], _fit_outputs())

    @classmethod
    def execute(cls, fit_file, absolute_path="", expected_sha256=""):
        result = _load_asset(_fit_path(fit_file, absolute_path), expected_sha256)
        return io.NodeOutput(result, str(result.path), result.sha256, _fit_report(result))

    @classmethod
    def fingerprint_inputs(cls, fit_file, absolute_path="", expected_sha256=""):
        try:
            return _bounded_fit_identity(_fit_path(fit_file, absolute_path))["sha256"]
        except (OSError, ValueError, RuntimeError):
            return float("nan")


class MiniMaxH3HyperFlowCurveModelApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Apply to Pruned MODEL", "Independent approximate route: native50-block pruned H3 only. "
            "Exact fit/selected-base/original-adapter bytes, actual raw FP32 curve basis, all208 backbone targets and "
            "two-time recipe checked. Prior ordinary content LoRAs and user delegates remain. Selected base SHA is "
            "NOT proof every loaded MODEL parameter equals that file. Unknown owners execute without portable reuse. "
            "Keep full-structure HyperFlow on its original loader; no silent fallback.",
            [io.Model.Input("model"), io.Custom(FIT).Input("curve_fit"),
             io.Combo.Input("base_file", options=_bases()), io.Combo.Input("hyperflow_file", options=_weight_options())],
            [io.Model.Output("model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, curve_fit, base_file, hyperflow_file):
        if type(curve_fit) is not fitting.CurveFit:
            raise ValueError("Select a dedicated Curve Fit asset, not a full HyperFlow plan")
        patched, binding, _audit = runtime.install_curve(model, load_hyperflow_original(_resolve(hyperflow_file)), curve_fit, _base(base_file))
        return io.NodeOutput(patched, canonical({"recipe": stages.RECIPE, "fit_sha256": binding.fit_sha256,
            "base_sha256": binding.base_sha256, "teacher_sha256": binding.teacher_sha256,
            "adapter_sha256": binding.adapter_sha256, "backbone_targets": 208, "approximation": True,
            "full_backbone_file_identity_certified": False, "quality_accepted": False}))

    @classmethod
    def fingerprint_inputs(cls, model, curve_fit, base_file, hyperflow_file):
        return tuple(fitting._file(path)["sha256"] for path in (_base(base_file), _resolve(hyperflow_file)))


class MiniMaxH3HyperFlowCurveFullSamplerSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Full 8-Interval Setup", "Only the full trained8 trajectory for SamplerCustomAdvanced. "
            "No partial-start selector: use typed HEAD/TAIL for actual captured x_sigma continuation. "
            "This is an approximation, not full-time HyperFlow equivalence. For persisted external effects use the "
            "dedicated stage binding nodes, not this standalone setup.",
            [io.Model.Input("model"), io.Latent.Input("av_latent")],
            [io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent):
        plan = runtime.build_plan(model)
        return io.NodeOutput(*runtime.setup_sampler(model, av_latent, plan), canonical(plan.as_report()))


class MiniMaxH3HyperFlowCurveHeadStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Continuous HEAD Only", "Only0:split. Captures exact model-space x_sigma separately from Core "
            "scaffold. Independent MODEL/content LoRAs/conditioning/NOISE; no TAIL/upscale/restart mask. Dedicated "
            "curve boundary is NOT clean x0 or a full-structure HyperFlow boundary. External EAV/Relay bind upstream.",
            [io.Model.Input("model"), io.Latent.Input("av_latent"), io.Noise.Input("noise"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Int.Input("split_interval", default=4, min=1, max=7),
             io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
             io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, noise, positive, negative, split_interval=4, cfg=1., reserve_vram_mib=1024):
        result = stages.sample_head(model, av_latent, positive, negative, _generate_noise(noise, av_latent),
            seed=_noise_seed(noise), split=split_interval, cfg=cfg, reserve_vram_mib=reserve_vram_mib,
            callback=_progress(0, split_interval))
        return io.NodeOutput(result, result.receipt_json)


class MiniMaxH3HyperFlowCurveTailStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Continuous TAIL Only", "Only split:8 from exact typed HEAD. Independent MODEL/content LoRAs/conditions, "
            "same original base/adapter/fit and producer implementation. No new noise, HEAD rerun, resize or clean-x0 restart. "
            "Seed only controls MODEL execution. Unknown delegates are retained but cannot certify persistent reuse.",
            [io.Custom(BOUNDARY).Input("continuous_boundary"), io.Model.Input("model"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Int.Input("seed", default=26092301, min=0, max=2**64-1),
             io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
             io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            [io.Latent.Output("av_latent"), io.String.Output("report_json"), io.Custom(RESULT).Output("completed_result")])

    @classmethod
    def execute(cls, continuous_boundary, model, positive, negative, seed=26092301, cfg=1., reserve_vram_mib=1024):
        if type(continuous_boundary) is not stages.CurveBoundary:
            raise ValueError("TAIL requires the dedicated curve HEAD, not clean x0/full HyperFlow")
        start = continuous_boundary.verify()["request"]["plan"]["absolute_interval"][1]
        result, report = stages.sample_tail(continuous_boundary, model, positive, negative,
            seed=seed, cfg=cfg, reserve_vram_mib=reserve_vram_mib, callback=_progress(start, 8))
        return io.NodeOutput(result.output, report, result)


class MiniMaxH3HyperFlowCurveHeadSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Save Frozen HEAD", "Archive captured x_sigma/scaffold/actual51 execution receipt in a NEW unique "
            "directory under output/MiniMaxH3/curve_stage_artifacts. Save path AND SHA. Not clean x0/quality approval; "
            "only certified portable identity can be loaded for a later process.",
            [io.Custom(BOUNDARY).Input("continuous_boundary"), io.String.Input("prefix", default="curve-head")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("artifact_path"),
             io.String.Output("artifact_sha256"), io.String.Output("report_json")], output=True)

    @classmethod
    def execute(cls, continuous_boundary, prefix="curve-head"):
        path, digest, report = storage.save_boundary(continuous_boundary, _store_root(), prefix)
        return io.NodeOutput(continuous_boundary, path, digest, report, ui={"text": [path, digest, report]})


class MiniMaxH3HyperFlowCurveHeadLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Exact Frozen HEAD", "Explicit relative path AND whole-file SHA. CPU-only selected artifact read, "
            "zero HEAD sampling/no MODEL loading/no fallback regeneration. Connect only TAIL branches in a cold graph. "
            "TAIL verifies same actual base/adapter/fit and producer implementation.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        return io.NodeOutput(*storage.load_boundary(_store_root(), artifact_path, artifact_sha256))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        return _fingerprint(artifact_path, "head")


class MiniMaxH3HyperFlowCurveTailSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Save Completed TAIL", "Write completed joint AV and immutable actual TAIL receipt, no additional "
            "sampling or automatic cache. New unique artifact, no overwrite. Keep exact path AND SHA for cold delivery.",
            [io.Custom(RESULT).Input("completed_result"), io.String.Input("prefix", default="curve-tail")],
            [io.Latent.Output("av_latent"), io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
             io.String.Output("report_json")], output=True)

    @classmethod
    def execute(cls, completed_result, prefix="curve-tail"):
        path, digest, report = storage.save_result(completed_result, _store_root(), prefix)
        return io.NodeOutput(completed_result.output, path, digest, report, ui={"text": [path, digest, report]})


class MiniMaxH3HyperFlowCurveTailLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Exact Completed TAIL", "Selected completed joint AV read only. Decode without diffusion "
            "MODEL/CLIP/fit/HEAD/TAIL. Exact path AND SHA; not a claim changed current settings match this frozen artifact.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Latent.Output("av_latent"), io.Custom(RESULT).Output("completed_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        result, report = storage.load_result(_store_root(), artifact_path, artifact_sha256)
        return io.NodeOutput(result.output, result, report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        return _fingerprint(artifact_path, "tail")


def _bound_outputs():
    return [io.Model.Output("model"), io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
        io.Latent.Output("stage_template"), io.Sigmas.Output("sigmas"), io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
        io.String.Output("report_json")]


class MiniMaxH3HyperFlowCurveHeadEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Bind External HEAD Effects", "Curve Apply -> optional paired Relay Conditioning -> this binding "
            "-> optional Stage EAV Apply -> HEAD Only. Same split on binding/sampler. Returned SIGMAS/context/template "
            "validate effects, not a new sampler/restart plan. CFG1 with active effects. No sampling here.",
            [io.Model.Input("model"), io.Latent.Input("av_latent"), io.Conditioning.Input("positive"),
             io.Conditioning.Input("negative"), io.Int.Input("split_interval", default=4, min=1, max=7)], _bound_outputs())

    @classmethod
    def execute(cls, model, av_latent, positive, negative, split_interval=4):
        return io.NodeOutput(*effects.bind_head(model, av_latent, positive, negative, split_interval))


class MiniMaxH3HyperFlowCurveTailEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Bind External TAIL Effects", "Independent TAIL MODEL/content LoRAs and paired Relay on exact "
            "curve HEAD. Returned scaffold is NOT raw x_sigma: keep typed boundary connected to TAIL. Optional Stage EAV "
            "Apply uses this phase's sigmas/template/context. CFG1 with effects. No hidden HEAD/noise/upscale.",
            [io.Custom(BOUNDARY).Input("continuous_boundary"), io.Model.Input("model"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative")], _bound_outputs())

    @classmethod
    def execute(cls, continuous_boundary, model, positive, negative):
        return io.NodeOutput(*effects.bind_tail(continuous_boundary, model, positive, negative))


class MiniMaxH3HyperFlowCurveHeadEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "HEAD Effects Execution Audit", "Read immutable sampled curve HEAD effect evidence, never "
            "mutable Apply-node counters. Pass exact boundary through, no quality approval.",
            [io.Custom(BOUNDARY).Input("continuous_boundary")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, continuous_boundary):
        return io.NodeOutput(continuous_boundary, canonical(effects.audit(continuous_boundary, "head")))


class MiniMaxH3HyperFlowCurveTailEffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "TAIL Effects Execution Audit", "Read completed immutable curve TAIL effect coverage. "
            "Returns typed completed result/final AV for storage/decode, no sampling or quality approval.",
            [io.Custom(RESULT).Input("completed_result")],
            [io.Custom(RESULT).Output("completed_result"), io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, completed_result):
        return io.NodeOutput(completed_result, completed_result.output, canonical(effects.audit(completed_result, "tail")))


class MiniMaxH3HyperFlowCurveBaseLoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Native FP32 Curve Base", "Explicit fresh native pruned H3 loader. Preserves Core's "
            "declared FP32 AdaLN/input/output precision islands and its original quantized backbone operations. "
            "Use BEFORE branching into independent content LoRAs/Curve Apply; does not repair an already loaded "
            "MODEL or change ordinary UNET loading. No fit, sampling, download or quality approval.",
            [io.Combo.Input("base_file", options=_bases())],
            [io.Model.Output("model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, base_file):
        model, report = loader.load_curve_base(_base(base_file))
        return io.NodeOutput(model, canonical(report))

    @classmethod
    def fingerprint_inputs(cls, base_file):
        return fitting._file(_base(base_file))["sha256"]


NODES = [MiniMaxH3HyperFlowCurveFitBuildEXPT8, MiniMaxH3HyperFlowCurveFitLoadEXPT8,
    MiniMaxH3HyperFlowCurveModelApplyEXPT8, MiniMaxH3HyperFlowCurveFullSamplerSetupEXPT8,
    MiniMaxH3HyperFlowCurveHeadStageEXPT8, MiniMaxH3HyperFlowCurveTailStageEXPT8,
    MiniMaxH3HyperFlowCurveHeadSaveEXPT8, MiniMaxH3HyperFlowCurveHeadLoadEXPT8,
    MiniMaxH3HyperFlowCurveTailSaveEXPT8, MiniMaxH3HyperFlowCurveTailLoadEXPT8,
    MiniMaxH3HyperFlowCurveHeadEffectsBindEXPT8, MiniMaxH3HyperFlowCurveTailEffectsBindEXPT8,
    MiniMaxH3HyperFlowCurveHeadEffectsAuditEXPT8, MiniMaxH3HyperFlowCurveTailEffectsAuditEXPT8,
    MiniMaxH3HyperFlowCurveBaseLoaderEXPT8]
