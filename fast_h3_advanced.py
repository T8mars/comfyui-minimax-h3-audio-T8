from __future__ import annotations

import json
from typing import Any

import torch

from .sampling import native_flow_sigmas, setup_dual_clock_sampling
from .fast_h3_vsa_advanced import apply_fast_h3_vsa, probe_comfy_kitchen_vsa


FAST_H3_STEPS = 4
FAST_H3_SHIFT_VIDEO = 12.0
FAST_H3_SHIFT_AUDIO = 3.0


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def probe_fast_h3_vsa() -> dict[str, Any]:
    capability = probe_comfy_kitchen_vsa()
    # Retain these two legacy report keys so archived tooling keeps parsing.
    capability["fastvideo_python_available"] = False
    capability["triton_available"] = capability[
        "external_vsa_executor_available"
    ]
    return capability


def build_fast_h3_4step_setup(
    model,
    av_latent,
    task_family: str = "t2va_only",
    attention_profile: str = "dense_comfyui",
):
    if task_family not in {
        "t2va_only",
        "t2va_fl2va",
        "t2va_fl2va_legacy_untrained_exp",
        "ref2va_untrained_exp",
    }:
        raise ValueError("unknown FastH3 task_family")
    if attention_profile not in {"dense_comfyui", "external_vsa_if_available"}:
        raise ValueError("unknown FastH3 attention_profile")

    vsa = probe_fast_h3_vsa()
    requested_vsa = attention_profile == "external_vsa_if_available"
    effective_attention = "dense_comfyui"
    vsa_runtime = None
    warnings: list[str] = []
    if task_family != "t2va_only":
        warnings.append(
            "FastH3 Preview v1 was trained for T2VA only. FL2VA and Ref2VA are "
            "untrained experimental routes and can collapse; this legacy choice is "
            "retained only so existing workflows still load."
        )
    working_model = model
    if requested_vsa:
        working_model, vsa_runtime, fallback_reason = apply_fast_h3_vsa(model)
        if vsa_runtime is None:
            warnings.append(
                "FastH3 VSA was not applied; using the valid dense 4-step route: "
                f"{fallback_reason}."
            )
        else:
            effective_attention = "comfy_kitchen_vsa_h3_90pct_tile64"

    patched, sampler, sigmas = setup_dual_clock_sampling(
        working_model,
        av_latent,
        FAST_H3_STEPS,
        FAST_H3_SHIFT_VIDEO,
        FAST_H3_SHIFT_AUDIO,
        "dual_clock_euler",
        "native_flow",
    )
    expected = native_flow_sigmas(FAST_H3_STEPS, FAST_H3_SHIFT_VIDEO)
    max_error = float(
        torch.max(
            torch.abs(
                torch.as_tensor(sigmas, dtype=torch.float64)
                - expected.to(dtype=torch.float64)
            )
        )
    )
    report = {
        "schema": "t8.minimax_h3.fast_h3_4step.v1",
        "status": "configured_with_warnings" if warnings else "configured",
        "task_family": task_family,
        "trained_contract": {
            "student": "FastH3 4-step Preview DMD2",
            "supported_family": "T2VA only",
            "steps_nfe": FAST_H3_STEPS,
            "cfg": 1.0,
            "sampler": "Euler",
            "base_sigma_ladder": [1.0, 0.75, 0.5, 0.25, 0.0],
            "shift_video": FAST_H3_SHIFT_VIDEO,
            "shift_audio": FAST_H3_SHIFT_AUDIO,
        },
        "attention_profile_requested": attention_profile,
        "attention_profile_effective": effective_attention,
        "vsa_capability": vsa,
        "vsa_runtime": vsa_runtime,
        "native_flow_schedule_max_abs_error": max_error,
        "warnings": warnings,
        "model_identity_policy": "user_selected_no_filename_size_or_hash_gate",
        "boundary": (
            "This node configures the published T2VA-only joint AV 4-step schedule. "
            "Apply the matching FastH3 LoRA with the H3 LoRA Compatibility Loader "
            "before this node. external_vsa_if_available owns the learned-gate, "
            "90%-sparse tile-64 Comfy Kitchen VSA route when every structural "
            "capability is present; otherwise it preserves the dense route."
        ),
    }
    return patched, sampler, sigmas, _json(report)
