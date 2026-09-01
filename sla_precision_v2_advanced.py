from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from .sla_attention_advanced import (
    _apply_authenticated_lora,
    _validate_sigmas,
)


SCHEMA = "t8.minimax_h3.sla_precision_v2.v1"
RUNTIME_TYPE = "MINIMAX_H3_SLA_PRECISION_V2_RUNTIME"
RUNTIME_ATTACHMENT_KEY = "t8_h3_sla_precision_v2_runtime_v1"
UPSTREAM_REPOSITORY = "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes"
UPSTREAM_COMMIT = "066ada9eb2378f392cc815663f63c4eef1060b4a"
RECOMMENDED_SCHEDULE = "recommended_8nfe_12v_3a"
USER_SELECTED_SCHEDULE = "user_selected_nfe_exp"
SCHEDULE_POLICIES = (RECOMMENDED_SCHEDULE, USER_SELECTED_SCHEDULE)
EXPECTED_H3_BLOCKS = 50


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _dual_clock_shifts(model) -> tuple[float, float]:
    transformer = dict(
        getattr(model, "model_options", {}).get("transformer_options", {})
    )
    return (
        float(transformer.get("minimax_h3_sigma_shift_video", float("nan"))),
        float(transformer.get("minimax_h3_sigma_shift_audio", float("nan"))),
    )


def _validate_schedule(model, sigmas: torch.Tensor, policy: str) -> dict[str, Any]:
    if policy not in SCHEDULE_POLICIES:
        raise ValueError(f"Unknown SLA Precision V2 schedule policy {policy!r}")
    shift_video, shift_audio = _dual_clock_shifts(model)
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise RuntimeError("SLA Precision V2 requires a positive Dual-Clock video shift")
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise RuntimeError("SLA Precision V2 requires a positive Dual-Clock audio shift")
    report = _validate_sigmas(sigmas, shift_video=shift_video)
    nfe = int(report["nfe"])
    if policy == RECOMMENDED_SCHEDULE:
        valid = (
            nfe == 8
            and math.isclose(shift_video, 12.0, abs_tol=1.0e-7)
            and math.isclose(shift_audio, 3.0, abs_tol=1.0e-7)
        )
        if not valid:
            raise RuntimeError(
                "SLA Precision V2 recommended route requires exactly 8 NFE and "
                "Dual-Clock video/audio shifts 12/3. Use user_selected_nfe_exp only "
                "for an explicitly experimental schedule."
            )
        status = "recommended_8nfe_12v_3a_valid"
    else:
        if not 1 <= nfe <= 64:
            raise RuntimeError("SLA Precision V2 experimental schedule requires 1-64 NFE")
        status = "user_selected_schedule_exp"
    return {
        **report,
        "policy": policy,
        "status": status,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
    }


def _parse_dense_steps(spec: str, nfe: int) -> list[int]:
    values: set[int] = set()
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid dense_steps token {token!r}") from exc
            values.update(range(min(start, end), max(start, end) + 1))
        else:
            try:
                values.add(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid dense_steps token {token!r}") from exc
    return sorted(value for value in values if 0 <= value < int(nfe))


def apply_sla_dynamic_lora_bypass(model, lora_path: str | Path):
    """Apply the SLA LoRA dynamically without mutating/re-quantizing the base."""

    patched, contract = _apply_authenticated_lora(
        model,
        lora_path,
        application_policy="bypass_model_only",
    )
    mapped = int(contract.get("mapped_patch_count") or 0)
    hooks = int(contract.get("bypass_hook_count") or 0)
    if mapped <= 0 or hooks != mapped:
        raise RuntimeError(
            "SLA Precision V2 requires every mapped SLA LoRA target to receive a "
            f"dynamic bypass hook; mapped={mapped}, hooks={hooks}"
        )
    if bool(getattr(patched, "patches", {})):
        raise RuntimeError(
            "SLA Precision V2 dynamic loader unexpectedly created standard weight patches"
        )
    report = {
        "schema": SCHEMA,
        "status": "sla_dynamic_lora_bypass_applied",
        "application": "quantized_base_plus_dynamic_lora_residual",
        "base_weight_mutation": False,
        "lora": contract,
    }
    return patched, _json(report)


@dataclass
class SLAPrecisionV2Runtime:
    config: dict[str, Any]
    state: dict[str, Any]
    finalized: bool = False


def patch_sla_precision_v2(
    model,
    sigmas: torch.Tensor,
    *,
    schedule_policy: str = RECOMMENDED_SCHEDULE,
    sparsity_ratio: float = 0.90,
    block_size: int = 32,
    min_seq_len: int = 8192,
    dense_last_steps: int = 1,
    dense_steps: str = "0",
    dense_backend: str = "comfy_kitchen",
    protect_audio: bool = True,
    disable_fp16_accum: bool = True,
    stabilize_motion: bool = False,
    reference_protection: bool = False,
):
    """Patch an already LoRA-prepared MODEL with the pinned precision SLA path."""

    schedule = _validate_schedule(model, sigmas, str(schedule_policy))
    nfe = int(schedule["nfe"])
    sparsity_ratio = float(sparsity_ratio)
    if not math.isfinite(sparsity_ratio) or not 0.60 <= sparsity_ratio <= 0.95:
        raise ValueError("SLA Precision V2 sparsity_ratio must be within [0.60, 0.95]")
    block_size = int(block_size)
    if block_size not in {32, 64, 128}:
        raise ValueError("SLA Precision V2 block_size must be 32, 64 or 128")
    min_seq_len = int(min_seq_len)
    if min_seq_len < 0:
        raise ValueError("SLA Precision V2 min_seq_len must be non-negative")
    dense_last_steps = int(dense_last_steps)
    if not 0 <= dense_last_steps <= nfe:
        raise ValueError("dense_last_steps must be between 0 and the connected NFE")
    explicit_dense = _parse_dense_steps(dense_steps, nfe)
    tail_dense = list(range(max(0, nfe - dense_last_steps), nfe))
    dense_indices = sorted(set(explicit_dense) | set(tail_dense))
    sparse_indices = [index for index in range(nfe) if index not in dense_indices]
    if not sparse_indices:
        raise ValueError("SLA Precision V2 configuration leaves no sparse sampling step")

    from .sla_precision_v2_vendor.patch import patch_h3_sla

    patched, state = patch_h3_sla(
        model,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
        min_seq_len=min_seq_len,
        dense_last_steps=dense_last_steps,
        dense_steps=dense_steps,
        dense_backend=str(dense_backend),
        disable_fp16_accum=bool(disable_fp16_accum),
        protect_audio=bool(protect_audio),
        stabilize_motion=bool(stabilize_motion),
        reference_protection="Light" if reference_protection else "Off",
        return_state=True,
    )
    config = {
        "schema": SCHEMA,
        "status": "ready_for_runtime_audit",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "license": "MIT",
            "vendor_scope": "block_map_kernel_patch",
        },
        "schedule": schedule,
        "attention": {
            "sparsity_ratio": sparsity_ratio,
            "block_size_q": block_size,
            "block_size_k": 64 if block_size == 128 else block_size,
            "min_seq_len": min_seq_len,
            "dense_last_steps": dense_last_steps,
            "dense_steps": explicit_dense,
            "dense_step_indices": dense_indices,
            "sparse_step_indices": sparse_indices,
            "dense_backend": str(dense_backend),
            "protect_audio": bool(protect_audio),
            "disable_fp16_accum": bool(disable_fp16_accum),
            "stabilize_motion": bool(stabilize_motion),
            "reference_protection": "Light" if reference_protection else "Off",
            "router_precision": "fp32_pool_and_fp32_scores",
            "sparse_kernel": "direct_triton_fp32_online_softmax",
        },
        "contract": (
            "Attention-only patch. Connect a MODEL already prepared with the SLA LoRA; "
            "the recommended workflow uses T8 dynamic model-only bypass so quantized "
            "base weights are not merged and re-quantized."
        ),
    }
    runtime = SLAPrecisionV2Runtime(config=config, state=state)
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(RUNTIME_ATTACHMENT_KEY, runtime)
    return patched, runtime, _json(config)


def finalize_sla_precision_v2_runtime(av_latent, runtime: SLAPrecisionV2Runtime):
    if not isinstance(runtime, SLAPrecisionV2Runtime):
        raise TypeError("SLA Precision V2 Audit requires its matching runtime object")
    if runtime.finalized:
        raise RuntimeError("SLA Precision V2 runtime was already finalized")
    runtime.finalized = True
    state = runtime.state
    schedule = runtime.config["schedule"]
    attention = runtime.config["attention"]
    nfe = int(schedule["nfe"])
    sparse_indices = list(attention["sparse_step_indices"])
    dense_indices = list(attention["dense_step_indices"])
    expected_sparse_calls = len(sparse_indices) * EXPECTED_H3_BLOCKS
    expected_dense_main_calls = len(dense_indices) * EXPECTED_H3_BLOCKS
    observed_sparse_calls = int(state.get("calls") or 0)
    observed_dense_calls = int(state.get("dense") or 0)
    failure = state.get("failed")
    raw_step_records = dict(state.get("step_records") or {})
    step_records = {
        int(index): dict(record) for index, record in raw_step_records.items()
    }
    observed_step_indices = sorted(step_records)
    sparse_step_checks = {
        index: (
            int(step_records.get(index, {}).get("sparse_calls") or 0)
            == EXPECTED_H3_BLOCKS
            and int(step_records.get(index, {}).get("kernel_fallbacks") or 0) == 0
            and step_records.get(index, {}).get("expected_dense") is False
        )
        for index in sparse_indices
    }
    dense_step_checks = {
        index: (
            int(step_records.get(index, {}).get("dense_calls") or 0)
            >= EXPECTED_H3_BLOCKS
            and int(step_records.get(index, {}).get("sparse_calls") or 0) == 0
            and int(step_records.get(index, {}).get("kernel_fallbacks") or 0) == 0
            and step_records.get(index, {}).get("expected_dense") is True
        )
        for index in dense_indices
    }
    checks = {
        "logical_nfe_matches_connected_sigmas": int(state.get("n_steps") or 0) == nfe,
        "last_logical_step_reached": int(state.get("last_step_index") or -1) == nfe - 1,
        "run_summary_completed": bool(state.get("summarized")),
        "all_logical_steps_observed_once_or_more": observed_step_indices
        == list(range(nfe))
        and all(
            int(step_records[index].get("wrapper_calls") or 0) >= 1
            for index in range(nfe)
        ),
        "each_sparse_step_routes_exactly_50_h3_blocks": all(
            sparse_step_checks.values()
        ),
        "each_dense_step_routes_at_least_50_h3_blocks": all(
            dense_step_checks.values()
        ),
        "sparse_calls_match_50_h3_blocks_per_sparse_step": (
            observed_sparse_calls == expected_sparse_calls
        ),
        "dense_calls_cover_50_h3_blocks_per_dense_step": (
            observed_dense_calls >= expected_dense_main_calls
        ),
        "no_sparse_kernel_failure_or_hidden_fallback": failure is None,
        "sequence_reached_sparse_threshold": int(state.get("seq") or 0)
        >= int(attention["min_seq_len"]),
        "at_least_one_key_block_retained": int(state.get("kept") or 0) > 0,
        "audio_or_language_blocks_protected": (
            not bool(attention["protect_audio"])
            or int(state.get("pinned") or 0) > 0
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": "precision_v2_mechanically_verified" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected": {
            "nfe": nfe,
            "sparse_step_indices": sparse_indices,
            "dense_step_indices": dense_indices,
            "sparse_main_attention_calls": expected_sparse_calls,
            "minimum_dense_main_attention_calls": expected_dense_main_calls,
        },
        "observed": {
            "sparse_calls": observed_sparse_calls,
            "dense_calls": observed_dense_calls,
            "logical_step_1_based": int(state.get("step") or 0),
            "logical_step_0_based": state.get("last_step_index"),
            "n_steps": int(state.get("n_steps") or 0),
            "sequence_tokens": int(state.get("seq") or 0),
            "selected_key_blocks": int(state.get("kept") or 0),
            "total_key_blocks": int(state.get("blocks") or 0),
            "pinned_key_blocks": int(state.get("pinned") or 0),
            "dense_backend": state.get("dense_backend"),
            "displaced_backend": state.get("backend"),
            "first_kernel_failure": failure,
            "per_logical_step": {
                str(index): step_records[index] for index in observed_step_indices
            },
        },
        "per_step_checks": {
            "sparse": {str(index): passed for index, passed in sparse_step_checks.items()},
            "dense": {str(index): passed for index, passed in dense_step_checks.items()},
        },
        "config": runtime.config,
        "quality_claim": (
            "Mechanical execution only. Full-speed visual review and clear-dialogue "
            "listening remain mandatory before recommendation."
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("SLA Precision V2 runtime audit failed: " + ", ".join(failed))
    return av_latent, _json(report)


__all__ = [
    "RECOMMENDED_SCHEDULE",
    "RUNTIME_TYPE",
    "SCHEDULE_POLICIES",
    "SLAPrecisionV2Runtime",
    "USER_SELECTED_SCHEDULE",
    "apply_sla_dynamic_lora_bypass",
    "finalize_sla_precision_v2_runtime",
    "patch_sla_precision_v2",
]
