from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .hybrid_model import (
    AUTO_PROFILE,
    HYBRID_ARTIFACT_TYPE,
    HYBRID_PLAN_TYPE,
    PROFILE_SPECS,
    artifact_output_root,
    artifact_path_for_plan,
    build_hybrid_artifact,
    file_stat_fingerprint,
    conditioning_reference_fingerprint,
    inspect_checkpoint_pair,
    load_hybrid_model,
    pair_report_text,
    pretty_json,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"
QUALITY_BASE_DEFAULT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REFERENCE_OVERLAY_DEFAULT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
HybridPlanIO = io.Custom(HYBRID_PLAN_TYPE)
HybridArtifactIO = io.Custom(HYBRID_ARTIFACT_TYPE)


def _diffusion_models() -> list[str]:
    return list(folder_paths.get_filename_list("diffusion_models"))


def _preferred_default(options: list[str], preferred: str) -> str | None:
    if preferred in options:
        return preferred
    return options[0] if options else None


def _resolve_diffusion_model(name: str) -> str:
    return folder_paths.get_full_path_or_raise("diffusion_models", name)


class MiniMaxH3HybridPairInspectorT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = _diffusion_models()
        return io.Schema(
            node_id="MiniMaxH3HybridPairInspectorT8Advanced",
            display_name="MiniMax H3 Hybrid Pair Inspector / 混合模型配对检查 (Advanced)",
            description=(
                "Strictly checks the exact P0 FL2VA-pruned quality base and Ref2VA-pruned "
                "reference overlay before any artifact can be built. Explicit profiles run "
                "independently; auto modality matching waits for the optional Conditioning input."
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "quality_base",
                    options=models,
                    default=_preferred_default(models, QUALITY_BASE_DEFAULT),
                    tooltip="Exact validated FL2VA pruned ConvRot checkpoint used as the base.",
                ),
                io.Combo.Input(
                    "reference_overlay",
                    options=models,
                    default=_preferred_default(models, REFERENCE_OVERLAY_DEFAULT),
                    tooltip="Exact validated Ref2VA pruned ConvRot checkpoint used as the overlay source.",
                ),
                io.Combo.Input(
                    "profile",
                    options=[*PROFILE_SPECS, AUTO_PROFILE],
                    default="blocks_25_49_video_audio_exp",
                    tooltip=(
                        "Neutral experimental recipes only. None is a proven quality winner; "
                        "static modality rows also affect target streams."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    optional=True,
                    tooltip=(
                        "Optional existing H3 Conditioning. With auto_match_reference_modalities_exp, "
                        "its real ref kinds select the minimal video/audio modality recipe."
                    ),
                ),
                io.Combo.Input(
                    "verification",
                    options=["full_sha256", "header_only_exp"],
                    default="full_sha256",
                    advanced=True,
                    tooltip=(
                        "header_only_exp is diagnostic and always blocks artifact construction. "
                        "Use full_sha256 for a buildable plan."
                    ),
                ),
            ],
            outputs=[
                HybridPlanIO.Output("hybrid_plan"),
                io.Boolean.Output("compatible"),
                io.String.Output("report"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, quality_base, reference_overlay, profile, verification, positive=None):
        plan = inspect_checkpoint_pair(
            _resolve_diffusion_model(quality_base),
            _resolve_diffusion_model(reference_overlay),
            profile,
            verification,
            positive,
        )
        return io.NodeOutput(
            plan,
            bool(plan["compatible"]),
            pair_report_text(plan),
            pretty_json(plan),
        )

    @classmethod
    def fingerprint_inputs(
        cls, quality_base, reference_overlay, profile, verification, positive=None
    ):
        try:
            paths = [
                _resolve_diffusion_model(quality_base),
                _resolve_diffusion_model(reference_overlay),
            ]
            files = file_stat_fingerprint(paths)
        except (OSError, ValueError) as exc:
            files = f"unresolved:{type(exc).__name__}:{exc}"
        if positive is None:
            reference_fingerprint = "none"
        else:
            try:
                reference_fingerprint = conditioning_reference_fingerprint(positive)
            except (TypeError, ValueError) as exc:
                reference_fingerprint = f"invalid:{type(exc).__name__}:{exc}"
        return f"{profile}:{verification}:{reference_fingerprint}:{files}"


class MiniMaxH3HybridArtifactBuilderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridArtifactBuilderT8Advanced",
            display_name="MiniMax H3 Hybrid Artifact Builder / 小型混合补丁构建 (Advanced)",
            description=(
                "Builds or reuses a content-addressed 13.8–83.1 MiB FP16 target-slice artifact "
                "under ComfyUI/models/h3_hybrid_artifacts. It never copies a complete model and "
                "never overwrites a mismatched artifact."
            ),
            category=CATEGORY,
            inputs=[HybridPlanIO.Input("hybrid_plan")],
            outputs=[
                HybridArtifactIO.Output("hybrid_artifact"),
                io.String.Output("artifact_path"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, hybrid_plan):
        artifact = build_hybrid_artifact(
            hybrid_plan,
            artifact_output_root(folder_paths.models_dir),
        )
        return io.NodeOutput(artifact, artifact["path"], pretty_json(artifact))

    @classmethod
    def fingerprint_inputs(cls, hybrid_plan):
        try:
            output_root = artifact_output_root(folder_paths.models_dir)
            artifact_path = artifact_path_for_plan(hybrid_plan, output_root)
            paths = [
                hybrid_plan["source"]["base_path"],
                hybrid_plan["source"]["overlay_path"],
                artifact_path,
                artifact_path.with_suffix(artifact_path.suffix + ".json"),
            ]
            files = file_stat_fingerprint(paths)
            return f"{hybrid_plan.get('plan_fingerprint')}:{files}"
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return f"unresolved:{type(exc).__name__}:{exc}"


class MiniMaxH3HybridModelLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = _diffusion_models()
        return io.Schema(
            node_id="MiniMaxH3HybridModelLoaderT8Advanced",
            display_name="MiniMax H3 Hybrid Model Loader / 混合模型加载 (Advanced)",
            description=(
                "Loads the quality base through ComfyUI's stock diffusion loader, preserving "
                "DynamicVRAM/VBAR file-backed behavior, then applies only the connected small "
                "artifact to a clone. Use Hybrid -> LoRA ordering."
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "quality_base",
                    options=models,
                    default=_preferred_default(models, QUALITY_BASE_DEFAULT),
                ),
                io.Combo.Input(
                    "mode",
                    options=["base_only", "apply_artifact_exp"],
                    default="base_only",
                    tooltip=(
                        "base_only is the stock-loader control. apply_artifact_exp requires the "
                        "exact artifact and remains experimental."
                    ),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    default="default",
                    advanced=True,
                ),
                HybridArtifactIO.Input("hybrid_artifact", optional=True),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, quality_base, mode, weight_dtype, hybrid_artifact=None):
        model, report = load_hybrid_model(
            _resolve_diffusion_model(quality_base),
            mode,
            weight_dtype,
            hybrid_artifact,
        )
        return io.NodeOutput(model, pretty_json(report))

    @classmethod
    def fingerprint_inputs(
        cls,
        quality_base,
        mode,
        weight_dtype,
        hybrid_artifact=None,
    ):
        try:
            paths: list[str | Path] = [_resolve_diffusion_model(quality_base)]
            artifact_sha = "none"
            if hybrid_artifact is not None:
                artifact_path = Path(hybrid_artifact["path"])
                paths.extend(
                    [
                        artifact_path,
                        artifact_path.with_suffix(artifact_path.suffix + ".json"),
                    ]
                )
                artifact_sha = str(hybrid_artifact.get("artifact_sha256", "unknown"))
            files = file_stat_fingerprint(paths)
            return f"{mode}:{weight_dtype}:{artifact_sha}:{files}"
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return f"unresolved:{type(exc).__name__}:{exc}"


HYBRID_MODEL_ADVANCED_NODE_CLASSES = [
    MiniMaxH3HybridPairInspectorT8Advanced,
    MiniMaxH3HybridArtifactBuilderT8Advanced,
    MiniMaxH3HybridModelLoaderT8Advanced,
]
