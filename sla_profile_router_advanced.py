from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

from .sla_attention_advanced import (
    SLA_CONSUMER_TURBO_MODE,
    SLA_EXPECTED_NFE,
    SLA_FULL_RANGE_END_PERCENT,
    SLA_FULL_RANGE_START_PERCENT,
    SLA_SHIFT_AUDIO,
    SLA_SHIFT_VIDEO,
    SLARuntime,
    _assert_core_contract,
    _inspect_kj_sage_contract,
    _runtime_route,
    _validate_sla_percent_window,
    _validate_lora_header,
    _validate_sigmas,
    build_sla_model,
)


PROFILE_ROUTER_SCHEMA = 1
PROFILE_WRAPPER_KEY = "t8_h3_turbo_sla_profile_router_v1"
CONSUMER_TURBO_PROFILE = SLA_CONSUMER_TURBO_MODE
SLA_EXACT_PROFILE = "sla_4step_upstream_exact_exp"
DISABLED_PROFILE = "disabled_identity"
SLA_INT8_BYPASS_PROFILE = "sla_4step_int8_bypass_exp"
PROFILE_OPTIONS = (
    CONSUMER_TURBO_PROFILE,
    SLA_EXACT_PROFILE,
    DISABLED_PROFILE,
    SLA_INT8_BYPASS_PROFILE,
)
CORRECTED_TURBO_LORA_FILENAME = (
    "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors"
)
CONSUMER_TURBO_NFE = 8
CONSUMER_TURBO_SHIFT_VIDEO = 12.0
CONSUMER_TURBO_SHIFT_AUDIO = 3.0
SLA_INT8_BYPASS_START_PERCENT = 0.15
SLA_INT8_BYPASS_END_PERCENT = 0.90


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _classify_published_sla_base(base_contract: dict) -> str:
    """Return the base families covered by the released SLA evidence.

    The checkpoint itself is BF16.  LightX2V's published RTX 5090 recipe runs
    an FP8 DiT.  Neither source validates the ComfyUI INT8 ConvRot base.  Keep
    the old SLA nodes available for explicit diagnostics, but do not let the
    new quality-oriented profile call an INT8 experiment "upstream exact".
    """
    if bool(base_contract.get("official_bf16_base_observed")):
        return "bf16_checkpoint_family"
    quant_format = str(base_contract.get("observed_quant_format") or "").lower()
    weight_type = str(base_contract.get("observed_weight_type") or "").lower()
    if "fp8" in quant_format or "float8" in quant_format or "float8" in weight_type:
        return "lightx2v_fp8_recipe_family"
    return (
        "user_selected_unvalidated_base:"
        f"quant_format={quant_format or 'none'};weight_type={weight_type or 'unknown'}"
    )


def _classify_int8_sla_bypass_base(base_contract: dict) -> str:
    """Require the exact local INT8 ConvRot family for the bypass experiment."""
    quant_format = str(base_contract.get("observed_quant_format") or "").lower()
    weight_type = str(base_contract.get("observed_weight_type") or "").lower()
    convrot = bool(base_contract.get("observed_convrot"))
    targets = dict(base_contract.get("lora_target_quantization") or {})
    target_contract = (
        int(targets.get("main_target_count") or 0) == 200
        and int(targets.get("main_int8_convrot_count") or 0) == 200
        and int(targets.get("main_unquantized_count") or 0) == 0
        and int(targets.get("token_refiner_target_count") or 0) == 8
        and int(targets.get("token_refiner_int8_convrot_count") or 0) == 0
        and int(targets.get("token_refiner_unquantized_count") or 0) == 8
    )
    if (
        quant_format == "int8_tensorwise"
        and weight_type == "quantizedtensor"
        and convrot
        and target_contract
    ):
        return "comfyui_int8_convrot_bypass_experiment"
    return (
        "user_selected_unvalidated_bypass_base:"
        f"quant_format={quant_format or 'none'};"
        f"weight_type={weight_type or 'unknown'};convrot={convrot};"
        f"target_contract_match={target_contract}"
    )


def _validate_profile_schedule(
    sigmas: torch.Tensor,
    *,
    profile: str,
    shift_video: float,
    shift_audio: float,
) -> dict:
    profile = str(profile)
    shift_video = float(shift_video)
    shift_audio = float(shift_audio)
    report = _validate_sigmas(sigmas, shift_video=shift_video)
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise RuntimeError("H3 Turbo/SLA profile requires a finite positive audio shift")
    if profile == CONSUMER_TURBO_PROFILE:
        valid = (
            int(report["nfe"]) == CONSUMER_TURBO_NFE
            and math.isclose(
                shift_video, CONSUMER_TURBO_SHIFT_VIDEO, abs_tol=1.0e-7
            )
            and math.isclose(
                shift_audio, CONSUMER_TURBO_SHIFT_AUDIO, abs_tol=1.0e-7
            )
        )
        if not valid:
            raise RuntimeError(
                "H3 consumer Turbo route requires the project-validated 8 NFE, "
                "video/audio shift 12/3 profile. Change Dual-Clock to 8 / 12 / 3; "
                "do not reuse the SLA 4 / 6 / 3 schedule."
            )
        report["profile_schedule_status"] = "consumer_turbo8_12v_3a_validated"
    elif profile in {SLA_EXACT_PROFILE, SLA_INT8_BYPASS_PROFILE}:
        valid = (
            int(report["nfe"]) == SLA_EXPECTED_NFE
            and math.isclose(shift_video, SLA_SHIFT_VIDEO, abs_tol=1.0e-7)
            and math.isclose(shift_audio, SLA_SHIFT_AUDIO, abs_tol=1.0e-7)
        )
        if not valid:
            raise RuntimeError(
                "H3 SLA upstream-exact route requires 4 NFE, video/audio shift 6/3. "
                "Change Dual-Clock to 4 / 6 / 3; 8-step SLA is not the published profile."
            )
        report["profile_schedule_status"] = (
            "sla_upstream_4nfe_6v_3a_exact_exp"
            if profile == SLA_EXACT_PROFILE
            else "sla_int8_bypass_4nfe_6v_3a_exp"
        )
    else:
        raise ValueError(f"Unknown H3 Turbo/SLA profile {profile!r}")
    report["shift_audio"] = shift_audio
    return report


def _validate_consumer_turbo_lora_header(path: str | Path) -> dict:
    contract = _validate_lora_header(path)
    metadata = dict(contract.get("metadata") or {})
    try:
        alpha = float(metadata.get("peft_lora_alpha", "nan"))
        effective_scale = float(metadata.get("effective_lora_scale", "nan"))
        sampler_steps = int(metadata.get("sampler_steps", "-1"))
    except (TypeError, ValueError):
        alpha = float("nan")
        effective_scale = float("nan")
        sampler_steps = -1
    loader = str(metadata.get("comfyui_loader", "")).lower()
    reference_match = (
        math.isclose(alpha, 8.0, abs_tol=1.0e-7)
        and math.isclose(effective_scale, 0.0625, abs_tol=1.0e-9)
        and sampler_steps == 4
        and "bypass" in loader
        and "model only" in loader
    )
    contract["application_mode"] = "comfyui_bypass_model_only"
    contract["corrected_alpha"] = alpha
    contract["effective_lora_scale"] = effective_scale
    contract["sampler_steps_metadata"] = sampler_steps
    contract["corrected_alpha8_reference_match"] = reference_match
    contract["model_identity_policy"] = "diagnostic_only_not_a_load_gate"
    return contract


def _apply_corrected_turbo_bypass_lora(
    model,
    path: str | Path,
) -> tuple[object, dict]:
    contract = _validate_consumer_turbo_lora_header(path)
    state, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    converted = comfy.lora_convert.convert_lora(state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
    patched = model.clone()
    manager = comfy.weight_adapter.BypassInjectionManager()
    for key, adapter in loaded.items():
        manager.add_adapter(key, adapter, strength=1.0)
    injections = manager.create_injections(patched.model)
    hook_count = int(manager.get_hook_count())
    patched.set_injections("bypass_lora", injections)
    if metadata and hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_corrected_alpha8_metadata", dict(metadata))
    contract["mapped_patch_count"] = len(loaded)
    contract["bypass_hook_count"] = hook_count
    contract["strength_model"] = 1.0
    return patched, contract


def _external_attention_policy(model) -> tuple[str, str]:
    object_keys = sorted(
        key for key in getattr(model, "object_patches", {}) if key != "model_sampling"
    )
    if not object_keys:
        return "reject", "native_or_builtin_dense"
    contract = _inspect_kj_sage_contract(model)
    return "compose_kj_sage", f"authenticated_kj_sage_{contract['patch_count']}_blocks"


def _dual_clock_shifts(model) -> tuple[float, float]:
    transformer = dict(getattr(model, "model_options", {}).get("transformer_options", {}))
    return (
        float(transformer.get("minimax_h3_sigma_shift_video", float("nan"))),
        float(transformer.get("minimax_h3_sigma_shift_audio", float("nan"))),
    )


def _bind_consumer_runtime(model, runtime: SLARuntime):
    def _diffusion_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        if len(executor.wrappers) != 1:
            raise RuntimeError(
                "H3 consumer Turbo profile detected another diffusion wrapper after binding"
            )
        try:
            route = _runtime_route(
                x=x,
                context=context,
                payload=kwargs.get("minimax_payload"),
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
            )
            runtime.begin_forward(route)
            return executor(
                x,
                timestep,
                context,
                transformer_options,
                **kwargs,
            )
        except BaseException as exc:
            runtime.abort(exc)
            raise

    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        PROFILE_WRAPPER_KEY,
        _diffusion_wrapper,
    )
    return model


def build_turbo_sla_profile_model(
    model,
    sigmas: torch.Tensor,
    *,
    turbo_lora_path: str,
    sla_lora_path: str,
    profile: str,
    base_policy: str,
    max_router_workspace_mib: int,
    sla_start_percent: float = SLA_INT8_BYPASS_START_PERCENT,
    sla_end_percent: float = SLA_INT8_BYPASS_END_PERCENT,
):
    profile = str(profile)
    if profile not in PROFILE_OPTIONS:
        raise ValueError(f"Unknown H3 Turbo/SLA profile {profile!r}")
    requested_percent_window = _validate_sla_percent_window(
        sla_start_percent, sla_end_percent
    )
    if profile == DISABLED_PROFILE:
        return build_sla_model(
            model,
            sigmas,
            lora_path="",
            mode="disabled_identity",
            base_policy=base_policy,
            max_router_workspace_mib=max_router_workspace_mib,
        )

    external_policy, attention_backend = _external_attention_policy(model)
    shift_video, shift_audio = _dual_clock_shifts(model)
    schedule = _validate_profile_schedule(
        sigmas,
        profile=profile,
        shift_video=shift_video,
        shift_audio=shift_audio,
    )
    if profile in {SLA_EXACT_PROFILE, SLA_INT8_BYPASS_PROFILE}:
        published_core = _assert_core_contract(
            model,
            base_policy=base_policy,
            external_attention_policy=external_policy,
        )
        base_contract = dict(published_core.get("base") or {})
        if profile == SLA_EXACT_PROFILE:
            base_family = _classify_published_sla_base(base_contract)
            lora_application_policy = "standard_patch"
            status = "sla_upstream_exact_ready_for_runtime_audit"
            quality_claim = (
                "upstream-exact SLA experiment only; BF16/FP8 evidence matching is "
                "reported diagnostically and does not reject other user-selected bases"
            )
        else:
            base_family = _classify_int8_sla_bypass_base(base_contract)
            lora_application_policy = "bypass_model_only"
            status = "sla_int8_bypass_percent_window_ready_for_runtime_audit"
            quality_claim = (
                "INT8 ConvRot dynamic-LoRA bypass experiment only; it avoids merging "
                "the SLA residual into and re-quantizing the base. SLA is active only "
                "inside the requested denoising-percent window; this has no quality "
                "claim until the same-scale full-duration human gate passes"
            )
        effective_percent_window = (
            requested_percent_window
            if profile == SLA_INT8_BYPASS_PROFILE
            else _validate_sla_percent_window(
                SLA_FULL_RANGE_START_PERCENT, SLA_FULL_RANGE_END_PERCENT
            )
        )
        patched, runtime, sla_report_json = build_sla_model(
            model,
            sigmas,
            lora_path=sla_lora_path,
            mode="apply_lightx2v_sla_upstream_exact_exp",
            base_policy=base_policy,
            max_router_workspace_mib=max_router_workspace_mib,
            external_attention_policy=external_policy,
            lora_application_policy=lora_application_policy,
            sla_start_percent=effective_percent_window["start_percent"],
            sla_end_percent=effective_percent_window["end_percent"],
        )
        report = {
            "schema": PROFILE_ROUTER_SCHEMA,
            "profile": profile,
            "status": status,
            "schedule": schedule,
            "attention_backend": attention_backend,
            "base_family": base_family,
            "published_base_family": (
                base_family if profile == SLA_EXACT_PROFILE else None
            ),
            "selected_lora_role": "sla",
            "lora_application_policy": lora_application_policy,
            "requested_sla_percent_window": requested_percent_window,
            "effective_sla_percent_window": effective_percent_window,
            "percent_window_applied": profile == SLA_INT8_BYPASS_PROFILE,
            "sla_loader": json.loads(sla_report_json),
            "quality_claim": quality_claim,
        }
        return patched, runtime, _json(report)

    core = _assert_core_contract(
        model,
        base_policy=base_policy,
        external_attention_policy=external_policy,
    )
    patched, lora_contract = _apply_corrected_turbo_bypass_lora(
        model, turbo_lora_path
    )
    config = {
        "schema": PROFILE_ROUTER_SCHEMA,
        "mode": CONSUMER_TURBO_PROFILE,
        "profile": CONSUMER_TURBO_PROFILE,
        "status": "consumer_turbo8_ready_for_runtime_audit",
        "sigma_contract": schedule,
        "core_contract": core,
        "profile_contract": {
            "attention_backend": attention_backend,
            "lora_application": "corrected_alpha8_bypass_model_only",
            "expected_nfe": CONSUMER_TURBO_NFE,
            "shift_video": CONSUMER_TURBO_SHIFT_VIDEO,
            "shift_audio": CONSUMER_TURBO_SHIFT_AUDIO,
            "sla_lora_loaded": False,
            "sparse_attention_loaded": False,
            "requested_sla_percent_window": requested_percent_window,
            "percent_window_applied": False,
        },
        "lora_contract": lora_contract,
        "quality_claim": (
            "project-validated mechanical Turbo8 profile; full visual/audio quality "
            "still requires human review"
        ),
    }
    runtime = SLARuntime(config)
    patched = _bind_consumer_runtime(patched, runtime)
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(PROFILE_WRAPPER_KEY, dict(config))
    return patched, runtime, _json(config)
