from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import comfy.model_patcher
import comfy.samplers
import torch

from .motion_quality_advanced import (
    _inverse_shift_sigma,
    _schedule_sha,
    _validate_h3_sigmas,
    canonical_json,
)
from .sampling import time_shift_sigma


DYNAMIC_GUIDANCE_RUNTIME_SCHEMA = "minimax_h3_dynamic_guidance_runtime_t8_v1"
_DYNAMIC_MODES = {
    "passthrough_basic",
    "single_condition_gain_exp",
    "true_cfg_exp",
}
_CURVES = {"linear", "cosine"}
_CONFLICT_KEYS = {
    "sampler_cfg_function",
    "sampler_pre_cfg_function",
    "sampler_post_cfg_function",
    "sampler_calc_cond_batch_function",
    "model_function_wrapper",
}


def _structural_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
            "nested": bool(value.is_nested),
        }
    if isinstance(value, dict):
        return {
            str(key): _structural_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).startswith("minimax_")
            or str(key).startswith("t8_")
            or str(key)
            in {
                "kind",
                "resolved_frame_index",
                "frame_index",
                "video_latent_t",
                "audio_latent_t",
                "ref_video_t",
                "ref_audio_t",
                "latent",
                "video_latent",
                "audio_latent",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_structural_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"python_type": type(value).__name__}


def conditioning_layout_contract(conditioning: Any) -> dict[str, Any]:
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise TypeError("conditioning must be a non-empty ComfyUI CONDITIONING value")
    entries = []
    for index, item in enumerate(conditioning):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise TypeError(f"conditioning entry {index} is not a [tensor, metadata] pair")
        tensor, metadata = item[0], item[1]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"conditioning entry {index} has no tensor payload")
        if not isinstance(metadata, dict):
            raise TypeError(f"conditioning entry {index} metadata must be a dict")
        layout_metadata = {
            str(key): _structural_value(value)
            for key, value in sorted(metadata.items(), key=lambda pair: str(pair[0]))
            if str(key).startswith("minimax_") or str(key).startswith("t8_")
        }
        entries.append(
            {
                "embedding_shape": list(tensor.shape),
                "embedding_dtype": str(tensor.dtype),
                "layout_metadata": layout_metadata,
            }
        )
    canonical = canonical_json(entries)
    return {
        "entries": entries,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _validate_curve_parameters(
    early_scale: float,
    late_scale: float,
    start_progress: float,
    end_progress: float,
    curve: str,
    shift_video: float,
) -> None:
    for name, value in (("early_scale", early_scale), ("late_scale", late_scale)):
        if not math.isfinite(value) or not 0.8 <= value <= 1.2:
            raise ValueError(f"{name} must be finite and between 0.8 and 1.2")
    if not (
        math.isfinite(start_progress)
        and math.isfinite(end_progress)
        and 0.0 <= start_progress < end_progress <= 1.0
    ):
        raise ValueError("progress range must satisfy 0 <= start < end <= 1")
    if curve not in _CURVES:
        raise ValueError(f"unsupported guidance curve: {curve!r}")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")


def dynamic_guidance_scale(
    sigma: torch.Tensor,
    *,
    early_scale: float,
    late_scale: float,
    start_progress: float,
    end_progress: float,
    curve: str,
    shift_video: float,
) -> torch.Tensor:
    """Return a device-side scale without synchronising sigma back to the CPU."""
    _validate_curve_parameters(
        early_scale,
        late_scale,
        start_progress,
        end_progress,
        curve,
        shift_video,
    )
    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a torch.Tensor")
    denominator = shift_video + sigma * (1.0 - shift_video)
    base_sigma = sigma / denominator
    progress = 1.0 - base_sigma
    blend = ((progress - start_progress) / (end_progress - start_progress)).clamp(0.0, 1.0)
    if curve == "cosine":
        blend = (1.0 - torch.cos(blend * math.pi)) * 0.5
    return early_scale + (late_scale - early_scale) * blend


def _reshape_scale(scale: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=prediction.device, dtype=prediction.dtype)
    if scale.numel() == 1:
        return scale.reshape([1] * prediction.ndim)
    if scale.ndim == 1 and scale.shape[0] == prediction.shape[0]:
        return scale.reshape(scale.shape[0], *([1] * (prediction.ndim - 1)))
    while scale.ndim < prediction.ndim:
        scale = scale.unsqueeze(-1)
    return scale


@dataclass
class DynamicGuidanceRuntime:
    static_report: dict[str, Any]
    predict_noise_calls: int = 0
    cfg_callback_calls: int = 0
    physical_model_forward_calls: int = 0
    forward_branch_batches: list[list[int]] = field(default_factory=list)

    def record_predict_noise(self) -> None:
        self.predict_noise_calls += 1

    def cfg_function(self, args: dict[str, Any]) -> torch.Tensor:
        self.cfg_callback_calls += 1
        cond = args["cond"]
        uncond = args["uncond"]
        if not isinstance(cond, torch.Tensor) or not isinstance(uncond, torch.Tensor):
            raise TypeError("dynamic guidance requires tensor cond/uncond predictions")
        scale = dynamic_guidance_scale(
            args["sigma"],
            early_scale=float(self.static_report["early_scale"]),
            late_scale=float(self.static_report["late_scale"]),
            start_progress=float(self.static_report["start_progress"]),
            end_progress=float(self.static_report["end_progress"]),
            curve=str(self.static_report["curve"]),
            shift_video=float(self.static_report["shift_video"]),
        )
        scale = _reshape_scale(scale, cond)
        return uncond + (cond - uncond) * scale

    def model_function_wrapper(self, apply_model, options: dict[str, Any]):
        self.physical_model_forward_calls += 1
        branches = [int(value) for value in options.get("cond_or_uncond", [])]
        self.forward_branch_batches.append(branches)
        return apply_model(options["input"], options["timestep"], **options["c"])

    def final_report(self) -> str:
        branch_zero = sum(batch.count(0) for batch in self.forward_branch_batches)
        branch_one = sum(batch.count(1) for batch in self.forward_branch_batches)
        report = dict(self.static_report)
        report.update(
            {
                "runtime_observed": self.predict_noise_calls > 0,
                "actual_predict_noise_calls": self.predict_noise_calls,
                "actual_cfg_callback_calls": self.cfg_callback_calls,
                "actual_physical_model_forward_calls": self.physical_model_forward_calls,
                "actual_cond_branch_evaluations": branch_zero,
                "actual_uncond_branch_evaluations": branch_one,
                "actual_forward_branch_batches": self.forward_branch_batches,
                "true_cfg_validated": False,
                "quality_validated": False,
                "memory_safe_claim": False,
            }
        )
        return canonical_json(report, indent=2)


class DynamicGuidanceGuider(comfy.samplers.CFGGuider):
    def __init__(
        self,
        model,
        positive,
        runtime: DynamicGuidanceRuntime,
        *,
        negative=None,
        dynamic: bool,
        true_cfg: bool,
    ):
        super().__init__(model)
        self.runtime = runtime
        if true_cfg:
            self.inner_set_conds({"positive": positive, "negative": negative})
        else:
            self.inner_set_conds({"positive": positive})
        self.cfg = 1.0
        if dynamic:
            self.model_options = comfy.model_patcher.create_model_options_clone(
                model.model_options
            )
            self.model_options["sampler_cfg_function"] = runtime.cfg_function
            self.model_options["model_function_wrapper"] = runtime.model_function_wrapper
            if true_cfg:
                self.model_options["disable_cfg1_optimization"] = True

    def predict_noise(self, *args, **kwargs):
        self.runtime.record_predict_noise()
        return super().predict_noise(*args, **kwargs)


def _schedule_report(
    sigmas: torch.Tensor,
    *,
    profile: str,
    early_scale: float,
    late_scale: float,
    start_progress: float,
    end_progress: float,
    curve: str,
    shift_video: float,
    shift_audio: float,
) -> dict[str, Any]:
    values = _validate_h3_sigmas(sigmas, profile)
    scales = dynamic_guidance_scale(
        values,
        early_scale=early_scale,
        late_scale=late_scale,
        start_progress=start_progress,
        end_progress=end_progress,
        curve=curve,
        shift_video=shift_video,
    )
    base = _inverse_shift_sigma(values, shift_video)
    audio = time_shift_sigma(values, shift_video, shift_audio)
    return {
        "expected_nfe": int(values.numel() - 1),
        "input_schedule_sha256": _schedule_sha(values),
        "video_sigmas": [float(value) for value in values],
        "audio_sigmas": [float(value) for value in audio],
        "base_progress": [float(1.0 - value) for value in base],
        "expected_scales": [float(value) for value in scales],
    }


def build_dynamic_guidance_guider(
    model,
    positive,
    sigmas: torch.Tensor,
    mode: str,
    early_scale: float,
    late_scale: float,
    start_progress: float,
    end_progress: float,
    curve: str,
    shift_video: float,
    shift_audio: float,
    profile: str,
    accept_true_cfg_cost: bool,
    accept_turbo_guidance_ood: bool,
    negative=None,
) -> tuple[DynamicGuidanceGuider, DynamicGuidanceRuntime, str]:
    if mode not in _DYNAMIC_MODES:
        raise ValueError(f"unsupported dynamic guidance mode: {mode!r}")
    _validate_curve_parameters(
        early_scale,
        late_scale,
        start_progress,
        end_progress,
        curve,
        shift_video,
    )
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise ValueError("shift_audio must be finite and greater than zero")
    schedule = _schedule_report(
        sigmas,
        profile=profile,
        early_scale=early_scale,
        late_scale=late_scale,
        start_progress=start_progress,
        end_progress=end_progress,
        curve=curve,
        shift_video=shift_video,
        shift_audio=shift_audio,
    )
    identity_curve = math.isclose(early_scale, 1.0, abs_tol=1e-12) and math.isclose(
        late_scale, 1.0, abs_tol=1e-12
    )
    dynamic = mode != "passthrough_basic" and not identity_curve
    true_cfg = mode == "true_cfg_exp" and dynamic
    if mode == "single_condition_gain_exp" and negative is not None:
        raise ValueError("single_condition_gain_exp does not accept a negative conditioning")
    if true_cfg and negative is None:
        raise ValueError("true_cfg_exp requires negative conditioning")
    if true_cfg and not accept_true_cfg_cost:
        raise ValueError("true_cfg_exp requires accept_true_cfg_cost=true")
    if dynamic and profile.startswith("turbo_") and not accept_turbo_guidance_ood:
        raise ValueError(
            "dynamic guidance on an 8-step Turbo profile requires "
            "accept_turbo_guidance_ood=true"
        )

    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise TypeError("model must be a ComfyUI MODEL with model_options")
    conflicts = sorted(key for key in _CONFLICT_KEYS if key in model_options)
    if dynamic and conflicts:
        raise ValueError(
            "dynamic guidance refuses existing sampler/model wrappers: " + ", ".join(conflicts)
        )

    positive_contract = conditioning_layout_contract(positive)
    negative_contract = None
    if true_cfg:
        negative_contract = conditioning_layout_contract(negative)
        if positive_contract["sha256"] != negative_contract["sha256"]:
            raise ValueError(
                "true_cfg_exp requires positive and negative conditioning with identical "
                "H3 embedding shape and keyframe/reference/audio layout"
            )

    effective_mode = (
        "passthrough_basic"
        if not dynamic
        else "true_cfg_exp"
        if true_cfg
        else "single_condition_gain_exp"
    )
    report = {
        "schema": DYNAMIC_GUIDANCE_RUNTIME_SCHEMA,
        "status": "ready",
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "is_exact_basic_passthrough": not dynamic,
        "single_condition_gain_not_true_cfg": effective_mode
        == "single_condition_gain_exp",
        "early_scale": float(early_scale),
        "late_scale": float(late_scale),
        "start_progress": float(start_progress),
        "end_progress": float(end_progress),
        "curve": curve,
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "profile": profile,
        "accept_true_cfg_cost": bool(accept_true_cfg_cost),
        "accept_turbo_guidance_ood": bool(accept_turbo_guidance_ood),
        "positive_layout_sha256": positive_contract["sha256"],
        "negative_layout_sha256": (
            negative_contract["sha256"] if negative_contract is not None else None
        ),
        "existing_wrapper_conflicts": conflicts,
        "expected_condition_branches_per_step": 2 if true_cfg else 1,
        "generation_time_prevention_not_postprocess": True,
        "video_audio_share_one_guidance_scale": True,
        "true_cfg_validated": False,
        "quality_validated": False,
        "memory_safe_claim": False,
        **schedule,
    }
    runtime = DynamicGuidanceRuntime(report)
    guider = DynamicGuidanceGuider(
        model,
        positive,
        runtime,
        negative=negative,
        dynamic=dynamic,
        true_cfg=true_cfg,
    )
    return guider, runtime, canonical_json(report, indent=2)


def finalize_dynamic_guidance_report(
    av_latent: dict,
    runtime: DynamicGuidanceRuntime,
) -> tuple[dict, str]:
    if not isinstance(runtime, DynamicGuidanceRuntime):
        raise TypeError("runtime must come from MiniMaxH3DynamicCFGGuiderT8Advanced")
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise TypeError("av_latent must be a ComfyUI LATENT value")
    return av_latent, runtime.final_report()
