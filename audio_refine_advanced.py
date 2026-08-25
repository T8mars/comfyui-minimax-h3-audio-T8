from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import torch

from .core import nested_av_parts, split_noise_masks
from .nfe_run_contract_advanced import compile_nfe_run_contract
from .sampling import setup_dual_clock_sampling
from .vram_policy import runtime_snapshot


AUDIO_REFINE_AUDIT_TYPE = "H3_T8_AUDIO_REFINE_AUDIT"
AUDIO_REFINE_PLAN_TYPE = "H3_T8_AUDIO_REFINE_PLAN"
AUDIO_REFINE_AUDIT_SCHEMA = "t8.minimax_h3.audio_refine.audit.v1"
AUDIO_REFINE_PLAN_SCHEMA = "t8.minimax_h3.audio_refine.plan.v1"

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_UNVALIDATED_PATCH_MARKERS = (
    "activation_chunk",
    "block_cache",
    "eav",
    "enhance_a_video",
    "long_video",
    "multirate",
    "prompt_relay",
    "sla",
    "stg",
)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _qualified_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _descriptor_payload_sha256(descriptor: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in descriptor.items() if key != "payload_sha256"}
    return _sha256_text(canonical_json(payload))


def _finish_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    descriptor["payload_sha256"] = _descriptor_payload_sha256(descriptor)
    return descriptor


def _model_sampling(model: Any) -> Any:
    getter = getattr(model, "get_model_object", None)
    if callable(getter):
        try:
            value = getter("model_sampling")
        except Exception:
            value = None
        if value is not None:
            return value
    value = getattr(model, "model_sampling", None)
    if value is not None:
        return value
    return getattr(getattr(model, "model", None), "model_sampling", None)


def _model_manifest(model: Any) -> tuple[dict[str, Any], bool, bool]:
    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    sampling = _model_sampling(model)
    class_names = [_qualified_name(value) for value in (model, base, diffusion) if value]
    normalized = " ".join(class_names).lower().replace("_", "")
    is_h3 = "minimaxh3" in normalized

    model_options = _mapping(getattr(model, "model_options", {}))
    transformer = _mapping(model_options.get("transformer_options", {}))
    patches_replace = _mapping(transformer.get("patches_replace", {}))
    wrappers = _mapping(transformer.get("wrappers", {}))
    object_patches = _mapping(getattr(model, "object_patches", {}))
    attachments = _mapping(getattr(model, "attachments", {}))
    weight_patches = _mapping(getattr(model, "patches", {}))

    patch_replace_groups = {
        str(group): len(entries) if isinstance(entries, Mapping) else int(bool(entries))
        for group, entries in patches_replace.items()
        if bool(entries)
    }
    wrapper_groups = {
        str(group): len(entries) if hasattr(entries, "__len__") else 1
        for group, entries in wrappers.items()
        if bool(entries)
    }
    structure_keys = sorted(
        {
            *(str(key) for key in transformer),
            *(str(key) for key in attachments),
            *(str(key) for key in object_patches),
        }
    )
    marker_hits = sorted(
        marker
        for marker in _UNVALIDATED_PATCH_MARKERS
        if any(marker in key.lower() for key in structure_keys)
    )
    shift_video = transformer.get("minimax_h3_sigma_shift_video")
    shift_audio = transformer.get("minimax_h3_sigma_shift_audio")
    dual_clock_values_are_valid = all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in (shift_video, shift_audio)
    ) and (float(shift_video), float(shift_audio)) == (12.0, 3.0)
    t8_dual_clock_patch_validated = (
        set(object_patches) == {"model_sampling"}
        and object_patches["model_sampling"] is sampling
        and dual_clock_values_are_valid
    )
    has_dual_clock_patch_evidence = bool(object_patches) or any(
        key in transformer
        for key in (
            "minimax_h3_sigma_shift_video",
            "minimax_h3_sigma_shift_audio",
        )
    )
    patch_stack_unvalidated = bool(
        patch_replace_groups
        or wrapper_groups
        or marker_hits
        or (has_dual_clock_patch_evidence and not t8_dual_clock_patch_validated)
    )
    manifest = {
        "patcher_class": _qualified_name(model),
        "base_model_class": None if base is None else _qualified_name(base),
        "diffusion_model_class": (
            None if diffusion is None else _qualified_name(diffusion)
        ),
        "model_sampling_class": (
            None if sampling is None else _qualified_name(sampling)
        ),
        "weight_patch_key_count": len(weight_patches),
        "weight_patch_keys": sorted(str(key) for key in weight_patches),
        "object_patch_keys": sorted(str(key) for key in object_patches),
        "attachment_keys": sorted(str(key) for key in attachments),
        "transformer_option_keys": sorted(str(key) for key in transformer),
        "patch_replace_groups": patch_replace_groups,
        "wrapper_groups": wrapper_groups,
        "unvalidated_marker_hits": marker_hits,
        "t8_dual_clock_patch": (
            "validated_12_3"
            if t8_dual_clock_patch_validated
            else "unvalidated"
            if has_dual_clock_patch_evidence
            else "absent"
        ),
    }
    manifest["structure_sha256"] = _sha256_text(canonical_json(manifest))
    return manifest, is_h3 and sampling is not None, patch_stack_unvalidated


def _parse_audio_mode(report: Any) -> tuple[str | None, str | None]:
    if not isinstance(report, str):
        return None, "conditioning_report must be a string"
    matches = re.findall(r"(?m)^audio_mode=([^\r\n]+)$", report)
    if len(matches) != 1:
        return None, "conditioning_report must contain exactly one audio_mode= line"
    mode = matches[0].strip()
    if not mode:
        return None, "audio_mode cannot be empty"
    return mode, None


def _mask_manifest(
    av_latent: dict,
    video: torch.Tensor,
    audio: torch.Tensor,
) -> tuple[dict[str, Any], str | None, str | None]:
    try:
        video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    except ValueError as error:
        return {"layout": "invalid", "error": str(error)}, None, "invalid"
    if video_mask is None and audio_mask is None:
        return {"layout": "absent", "audio": "absent"}, None, None
    if audio_mask is None:
        return {
            "layout": "legacy_video_only",
            "audio": "unknown",
            "video_shape": [int(value) for value in video_mask.shape],
        }, "legacy", None
    if tuple(video_mask.shape) != tuple(video.shape) or tuple(audio_mask.shape) != tuple(
        audio.shape
    ):
        return {
            "layout": "nested",
            "audio": "invalid_shape",
            "video_shape": [int(value) for value in video_mask.shape],
            "audio_shape": [int(value) for value in audio_mask.shape],
        }, None, "invalid"
    if not bool(torch.isfinite(audio_mask).all().item()):
        state = "invalid"
    elif bool(torch.all(audio_mask == 1).item()):
        state = "full"
    elif bool(torch.all(audio_mask == 0).item()):
        state = "locked"
    else:
        state = "fractional"
    return {
        "layout": "nested",
        "audio": state,
        "video_shape": [int(value) for value in video_mask.shape],
        "audio_shape": [int(value) for value in audio_mask.shape],
        "video_dtype": str(video_mask.dtype),
        "audio_dtype": str(audio_mask.dtype),
    }, None, state


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resource_manifest(
    runtime: Any,
    *,
    node_owned_incremental_bytes: int,
    minimum_free_vram_mib: float,
    minimum_commit_headroom_gib: float,
) -> tuple[dict[str, Any], str | None]:
    snapshot = runtime if isinstance(runtime, Mapping) else {}
    gpu = _mapping(snapshot.get("gpu", {}))
    host = _mapping(snapshot.get("host", {}))
    free_mib = _finite_number(gpu.get("whole_device_free_mib"))
    commit_gib = _finite_number(host.get("commit_headroom_gib"))
    ram_gib = _finite_number(host.get("ram_available_gib"))
    required_ram_gib = (
        1.5 * float(node_owned_incremental_bytes) + 512.0 * _MIB
    ) / _GIB
    manifest = {
        "whole_device_free_mib": free_mib,
        "commit_headroom_gib": commit_gib,
        "ram_available_gib": ram_gib,
        "required_ram_available_gib": required_ram_gib,
        "node_owned_incremental_bytes": int(node_owned_incremental_bytes),
    }
    if free_mib is None or commit_gib is None or ram_gib is None:
        return manifest, "ABSTAIN_RESOURCE_TELEMETRY_UNKNOWN"
    if (
        free_mib < minimum_free_vram_mib
        or commit_gib < minimum_commit_headroom_gib
        or ram_gib < required_ram_gib
    ):
        return manifest, "ABSTAIN_INSUFFICIENT_HEADROOM"
    return manifest, None


def _add_finding(
    findings: list[dict[str, str]],
    code: str,
    severity: str,
    message: str,
) -> None:
    if any(item["code"] == code for item in findings):
        return
    findings.append({"code": code, "severity": severity, "message": message})


def _decision(findings: list[dict[str, str]]) -> str:
    severities = {item["severity"] for item in findings}
    if "REJECT" in severities:
        return "REJECT"
    if "ABSTAIN" in severities:
        return "ABSTAIN"
    return "ALLOW"


def _tensor_manifest(
    tensor: torch.Tensor,
    *,
    name: str,
    hash_chunk_megabytes: int,
) -> dict[str, Any]:
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use a dense strided tensor")
    if tensor.is_quantized:
        raise ValueError(f"{name} must not be quantized")
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must contain materialized data")
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Infinity")

    chunk_bytes = int(hash_chunk_megabytes) * 1024 * 1024
    if chunk_bytes <= 0:
        raise ValueError("hash_chunk_megabytes must be positive")
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), chunk_bytes):
        chunk = raw[start : start + chunk_bytes].to(device="cpu")
        digest.update(chunk.numpy().tobytes())

    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "byte_count": int(tensor.numel() * tensor.element_size()),
        "content_sha256": digest.hexdigest().upper(),
    }


def classify_audio_refine_latent(
    av_latent: dict,
    *,
    hash_chunk_megabytes: int = 8,
) -> dict[str, Any]:
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24:
        raise ValueError(
            "MiniMax H3 video latent must have 24 channels, "
            f"got {video.shape[1]}"
        )
    if audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "MiniMax H3 audio latent must have 32 channels and stereo axis 2, "
            f"got {tuple(audio.shape)}"
        )

    manifest = {
        "schema": "t8.minimax_h3.audio_refine.latent_manifest.v1",
        "video": _tensor_manifest(
            video,
            name="video latent",
            hash_chunk_megabytes=hash_chunk_megabytes,
        ),
        "audio": _tensor_manifest(
            audio,
            name="audio latent",
            hash_chunk_megabytes=hash_chunk_megabytes,
        ),
    }
    return {
        "manifest": manifest,
        "manifest_sha256": _sha256_text(canonical_json(manifest)),
    }


def audit_audio_refine(
    *,
    model: Any,
    positive: Any,
    av_latent: dict,
    conditioned_prompt: str,
    media_map_json: str,
    conditioning_report: str,
    protected_audio: Any = None,
    minimum_free_vram_mib: int = 512,
    minimum_commit_headroom_gib: float = 16.0,
    hash_chunk_megabytes: int = 8,
    runtime_snapshot_fn=runtime_snapshot,
) -> tuple[dict[str, Any], str, str]:
    findings: list[dict[str, str]] = []
    minimum_free = max(512.0, float(minimum_free_vram_mib))
    minimum_commit = max(16.0, float(minimum_commit_headroom_gib))
    chunk_megabytes = int(hash_chunk_megabytes)

    audio_mode, audio_mode_error = _parse_audio_mode(conditioning_report)
    if audio_mode_error:
        _add_finding(
            findings,
            "REJECT_AUDIO_MODE_AMBIGUOUS",
            "REJECT",
            audio_mode_error,
        )
    elif audio_mode == "lock_source":
        _add_finding(
            findings,
            "ABSTAIN_AUDIO_LOCKED",
            "ABSTAIN",
            "lock_source audio is authoritative and must not be regenerated",
        )
    elif audio_mode == "remix_source":
        _add_finding(
            findings,
            "ABSTAIN_REMIX_SOURCE_NOT_VALIDATED",
            "ABSTAIN",
            "remix_source fractional preservation is outside the exact first version",
        )
    elif audio_mode not in {None, "native", "reference_only"}:
        _add_finding(
            findings,
            "REJECT_AUDIO_MODE_UNSUPPORTED",
            "REJECT",
            f"unsupported audio_mode={audio_mode!r}",
        )

    run_contract_json = None
    run_contract_sha256 = None
    run_contract_summary = None
    try:
        (
            run_contract_json,
            run_contract_sha256,
            run_contract_report,
        ) = compile_nfe_run_contract(
            positive=positive,
            conditioned_prompt=conditioned_prompt,
            media_map_json=media_map_json,
            conditioning_report=conditioning_report,
            hash_chunk_megabytes=chunk_megabytes,
        )
        run_contract_summary = json.loads(run_contract_report)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        _add_finding(
            findings,
            "REJECT_CONDITIONING_CONTRACT_INVALID",
            "REJECT",
            str(error),
        )

    latent_result = None
    video = None
    audio = None
    mask_manifest: dict[str, Any] = {"layout": "unavailable"}
    node_owned_incremental_bytes = 0
    try:
        latent_result = classify_audio_refine_latent(
            av_latent,
            hash_chunk_megabytes=chunk_megabytes,
        )
        video, audio = nested_av_parts(av_latent)
        sample_bytes = int(
            video.numel() * video.element_size()
            + audio.numel() * audio.element_size()
        )
        mask_bytes = int((video.numel() + audio.numel()) * 4)
        node_owned_incremental_bytes = 2 * sample_bytes + mask_bytes
        mask_manifest, legacy_mask, audio_mask_state = _mask_manifest(
            av_latent, video, audio
        )
        if legacy_mask:
            if audio_mode in {"native", "reference_only"}:
                _add_finding(
                    findings,
                    "WARN_LEGACY_VIDEO_ONLY_MASK",
                    "WARN",
                    "legacy video-only mask accepted because audio_mode is explicit",
                )
            else:
                _add_finding(
                    findings,
                    "ABSTAIN_AUDIO_MASK_PROVENANCE_UNKNOWN",
                    "ABSTAIN",
                    "legacy video-only mask has no explicit supported audio provenance",
                )
        elif audio_mask_state == "locked":
            _add_finding(
                findings,
                "ABSTAIN_AUDIO_LOCKED",
                "ABSTAIN",
                "the input audio noise mask is fully locked",
            )
        elif audio_mask_state in {"fractional", "invalid"}:
            _add_finding(
                findings,
                "ABSTAIN_PARTIAL_AUDIO_MASK_UNSUPPORTED",
                "ABSTAIN",
                "fractional, mixed, non-finite, or invalid audio masks are unsupported",
            )
    except (TypeError, ValueError, RuntimeError) as error:
        _add_finding(
            findings,
            "REJECT_INVALID_AV_LATENT",
            "REJECT",
            str(error),
        )

    model_manifest, is_h3_model, patch_stack_unvalidated = _model_manifest(model)
    if not is_h3_model:
        _add_finding(
            findings,
            "REJECT_NOT_MINIMAX_H3_MODEL",
            "REJECT",
            "the connected MODEL is not an identifiable MiniMax H3 sampling model",
        )
    if patch_stack_unvalidated:
        _add_finding(
            findings,
            "ABSTAIN_PATCH_STACK_UNVALIDATED",
            "ABSTAIN",
            "transformer wrappers, block replacements, or scoped runtime patches are unvalidated",
        )

    if protected_audio is not None:
        _add_finding(
            findings,
            "ABSTAIN_PROTECTED_FINAL_AUDIO",
            "ABSTAIN",
            "a protected final AUDIO value is connected",
        )

    try:
        snapshot = runtime_snapshot_fn()
    except Exception as error:
        snapshot = {"inspection_error": f"{type(error).__name__}: {error}"}
    resource_manifest, resource_reason = _resource_manifest(
        snapshot,
        node_owned_incremental_bytes=node_owned_incremental_bytes,
        minimum_free_vram_mib=minimum_free,
        minimum_commit_headroom_gib=minimum_commit,
    )
    if resource_reason:
        _add_finding(
            findings,
            resource_reason,
            "ABSTAIN",
            "required whole-device VRAM, host RAM, or commit telemetry is unavailable or below the fixed floor",
        )

    final_decision = _decision(findings)
    reason_codes = [
        item["code"] for item in findings if item["severity"] in {"REJECT", "ABSTAIN"}
    ]
    warning_codes = [
        item["code"] for item in findings if item["severity"] == "WARN"
    ]
    descriptor = _finish_descriptor(
        {
            "schema": AUDIO_REFINE_AUDIT_SCHEMA,
            "decision": final_decision,
            "reason_codes": reason_codes,
            "warning_codes": warning_codes,
            "findings": findings,
            "audio_mode": audio_mode,
            "bindings": {
                "model_object_id": int(id(model)),
                "model_structure_sha256": model_manifest["structure_sha256"],
                "run_contract_sha256": run_contract_sha256,
                "positive_conditioning_sha256": (
                    None
                    if run_contract_summary is None
                    else run_contract_summary.get("positive_conditioning_sha256")
                ),
                "latent_manifest_sha256": (
                    None
                    if latent_result is None
                    else latent_result["manifest_sha256"]
                ),
            },
            "model_manifest": model_manifest,
            "run_contract_json": run_contract_json,
            "latent_manifest": (
                None if latent_result is None else latent_result["manifest"]
            ),
            "noise_mask": mask_manifest,
            "resource_gates": {
                "minimum_free_vram_mib": minimum_free,
                "minimum_commit_headroom_gib": minimum_commit,
                "minimum_ram_formula": "1.5 * node_owned_incremental_bytes + 512 MiB",
            },
            "resource_snapshot": resource_manifest,
            "hash_chunk_megabytes": chunk_megabytes,
            "quality_claim": "none; ALLOW is only a mechanical precondition",
        }
    )
    return descriptor, final_decision, canonical_json(descriptor, indent=2)


def _validate_signed_descriptor(
    descriptor: Any,
    *,
    schema: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict) or descriptor.get("schema") != schema:
        raise ValueError(
            f"REJECT_DESCRIPTOR_TAMPERED: {label} schema is missing or unsupported"
        )
    supplied = descriptor.get("payload_sha256")
    expected = _descriptor_payload_sha256(descriptor)
    if not isinstance(supplied, str) or supplied != expected:
        raise ValueError(
            f"REJECT_DESCRIPTOR_TAMPERED: {label} payload SHA-256 does not match"
        )
    return descriptor


def _shift_sigma(base_sigma: float, shift: float) -> float:
    value = float(base_sigma)
    if value == 0.0:
        return 0.0
    return float(shift * value / (1.0 + (shift - 1.0) * value))


def plan_audio_refine(
    audit: dict[str, Any],
    refine_steps: int,
    audio_denoise: float,
    refine_seed: int,
    model_strategy: str = "connected_model_explicit",
) -> tuple[dict[str, Any], str, str]:
    audit = _validate_signed_descriptor(
        audit,
        schema=AUDIO_REFINE_AUDIT_SCHEMA,
        label="audio refine audit",
    )
    if isinstance(refine_steps, bool) or not isinstance(refine_steps, int):
        raise ValueError("refine_steps must be an integer")
    if not 1 <= refine_steps <= 8:
        raise ValueError("refine_steps must be between 1 and 8")
    if isinstance(audio_denoise, bool):
        raise ValueError("audio_denoise must be a finite float")
    denoise = float(audio_denoise)
    if not math.isfinite(denoise) or not 0.01 <= denoise <= 1.0:
        raise ValueError("audio_denoise must be between 0.01 and 1.0")
    if isinstance(refine_seed, bool) or not isinstance(refine_seed, int):
        raise ValueError("refine_seed must be an integer")
    if not 0 <= refine_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("refine_seed must be between 0 and 2^64-1")
    if model_strategy != "connected_model_explicit":
        raise ValueError(
            "model_strategy must be connected_model_explicit; implicit model or LoRA switching is forbidden"
        )

    full_steps = int(refine_steps / denoise)
    if full_steps < refine_steps:
        raise ValueError("partial-tail full step count cannot be below refine_steps")
    base_sigmas = [
        float((refine_steps - index) / full_steps)
        for index in range(refine_steps + 1)
    ]
    video_sigmas = [_shift_sigma(value, 12.0) for value in base_sigmas]
    audio_sigmas = [_shift_sigma(value, 3.0) for value in base_sigmas]
    decision = str(audit.get("decision"))
    if decision not in {"ALLOW", "ABSTAIN", "REJECT"}:
        raise ValueError(
            "REJECT_DESCRIPTOR_TAMPERED: audit decision is not recognized"
        )

    descriptor = _finish_descriptor(
        {
            "schema": AUDIO_REFINE_PLAN_SCHEMA,
            "decision": decision,
            "reason_codes": list(audit.get("reason_codes", [])),
            "warning_codes": list(audit.get("warning_codes", [])),
            "audit_payload_sha256": audit["payload_sha256"],
            "audit": audit,
            "actual_refine_nfe": int(refine_steps),
            "full_steps": int(full_steps),
            "requested_audio_denoise": denoise,
            "effective_audio_denoise": float(refine_steps / full_steps),
            "refine_seed": int(refine_seed),
            "model_strategy": model_strategy,
            "base_sigmas": base_sigmas,
            "video_sigmas": video_sigmas,
            "audio_sigmas": audio_sigmas,
            "fixed_contract": {
                "cfg": 1.0,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
                "video_noise_mask": 0.0,
                "audio_noise_mask": 1.0,
                "cache": "disabled_exact",
            },
            "quality_claim": "none; the plan has not sampled or evaluated audio",
        }
    )
    return descriptor, decision, canonical_json(descriptor, indent=2)


class AudioRefineRandomNoise:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def generate_noise(self, input_latent: dict):
        return comfy.sample.prepare_noise(
            input_latent["samples"],
            self.seed,
            input_latent.get("batch_index"),
        )


class AudioRefineBypassNoise:
    seed = 0

    def generate_noise(self, input_latent: dict):
        return input_latent["samples"]


class AudioRefineBasicGuider(comfy.samplers.CFGGuider):
    def __init__(self, model: Any, positive: Any):
        super().__init__(model)
        self.inner_set_conds({"positive": positive})
        self.set_cfg(1.0)


def _never_sample(*args, **kwargs):
    raise RuntimeError("Audio Refine bypass sampler must never be invoked")


@dataclass(frozen=True)
class AudioRefineSetupResult:
    model: Any
    noise: Any
    guider: Any
    sampler: Any
    sigmas: torch.Tensor
    latent: dict
    report_json: str


def _recompile_bound_run_contract(audit: Mapping[str, Any], positive: Any) -> str:
    raw_contract = audit.get("run_contract_json")
    if not isinstance(raw_contract, str):
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: Audit has no valid conditioning run contract"
        )
    try:
        payload = json.loads(raw_contract)
        conditioned_prompt = payload["conditioned_prompt"]
        media_map_json = canonical_json(payload["media_map"])
        conditioning_report = payload["conditioning_report"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: stored conditioning run contract is invalid"
        ) from error
    _, contract_sha256, _ = compile_nfe_run_contract(
        positive=positive,
        conditioned_prompt=conditioned_prompt,
        media_map_json=media_map_json,
        conditioning_report=conditioning_report,
        hash_chunk_megabytes=int(audit.get("hash_chunk_megabytes", 8)),
    )
    return contract_sha256


def _verify_setup_bindings(
    plan: Mapping[str, Any],
    model: Any,
    positive: Any,
    av_latent: dict,
) -> dict[str, Any]:
    audit = _validate_signed_descriptor(
        plan.get("audit"),
        schema=AUDIO_REFINE_AUDIT_SCHEMA,
        label="embedded audio refine audit",
    )
    if plan.get("audit_payload_sha256") != audit.get("payload_sha256"):
        raise ValueError(
            "REJECT_DESCRIPTOR_TAMPERED: Plan does not bind the embedded Audit"
        )
    bindings = _mapping(audit.get("bindings", {}))
    model_manifest, is_h3_model, _ = _model_manifest(model)
    mismatches: list[str] = []
    if not is_h3_model:
        mismatches.append("model_type")
    if int(bindings.get("model_object_id", -1)) != id(model):
        mismatches.append("model_object")
    if bindings.get("model_structure_sha256") != model_manifest.get(
        "structure_sha256"
    ):
        mismatches.append("model_structure")

    try:
        run_contract_sha256 = _recompile_bound_run_contract(audit, positive)
    except ValueError:
        mismatches.append("conditioning")
    else:
        if bindings.get("run_contract_sha256") != run_contract_sha256:
            mismatches.append("conditioning")

    try:
        latent_result = classify_audio_refine_latent(
            av_latent,
            hash_chunk_megabytes=int(audit.get("hash_chunk_megabytes", 8)),
        )
    except (TypeError, ValueError, RuntimeError):
        mismatches.append("latent")
    else:
        if bindings.get("latent_manifest_sha256") != latent_result.get(
            "manifest_sha256"
        ):
            mismatches.append("latent")

    if mismatches:
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: changed " + ", ".join(sorted(set(mismatches)))
        )
    return audit


def _bypass_setup_result(
    *,
    plan: Mapping[str, Any],
    model: Any,
    positive: Any,
    av_latent: dict,
    reason_codes: list[str],
    resource_snapshot: Mapping[str, Any],
) -> AudioRefineSetupResult:
    report = {
        "schema": "t8.minimax_h3.audio_refine.setup.v1",
        "decision": "ABSTAIN",
        "reason_codes": reason_codes,
        "bypassed": True,
        "sigmas": [],
        "resource_snapshot": dict(resource_snapshot),
        "plan_payload_sha256": plan["payload_sha256"],
        "quality_claim": "none; original latent returned without model sampling",
    }
    return AudioRefineSetupResult(
        model=model,
        noise=AudioRefineBypassNoise(),
        guider=AudioRefineBasicGuider(model, positive),
        sampler=comfy.samplers.KSAMPLER(_never_sample),
        sigmas=torch.empty((0,), dtype=torch.float32),
        latent=av_latent,
        report_json=canonical_json(report, indent=2),
    )


def setup_audio_refine(
    *,
    plan: dict[str, Any],
    model: Any,
    positive: Any,
    av_latent: dict,
    setup_sampling_fn=setup_dual_clock_sampling,
    runtime_snapshot_fn=runtime_snapshot,
) -> AudioRefineSetupResult:
    plan = _validate_signed_descriptor(
        plan,
        schema=AUDIO_REFINE_PLAN_SCHEMA,
        label="audio refine plan",
    )
    audit = _verify_setup_bindings(plan, model, positive, av_latent)
    plan_decision = plan.get("decision")
    if plan_decision == "REJECT":
        raise ValueError(
            "REJECT_AUDIO_REFINE_PLAN: " + ", ".join(plan.get("reason_codes", []))
        )
    if plan_decision not in {"ALLOW", "ABSTAIN"}:
        raise ValueError("REJECT_DESCRIPTOR_TAMPERED: Plan decision is unsupported")

    try:
        snapshot = runtime_snapshot_fn()
    except Exception as error:
        snapshot = {"inspection_error": f"{type(error).__name__}: {error}"}
    gates = _mapping(audit.get("resource_gates", {}))
    original_resource = _mapping(audit.get("resource_snapshot", {}))
    resource_manifest, resource_reason = _resource_manifest(
        snapshot,
        node_owned_incremental_bytes=int(
            original_resource.get("node_owned_incremental_bytes", 0)
        ),
        minimum_free_vram_mib=max(
            512.0, float(gates.get("minimum_free_vram_mib", 512.0))
        ),
        minimum_commit_headroom_gib=max(
            16.0, float(gates.get("minimum_commit_headroom_gib", 16.0))
        ),
    )
    reason_codes = list(plan.get("reason_codes", []))
    if resource_reason and resource_reason not in reason_codes:
        reason_codes.append(resource_reason)
    if plan_decision == "ABSTAIN" or resource_reason:
        return _bypass_setup_result(
            plan=plan,
            model=model,
            positive=positive,
            av_latent=av_latent,
            reason_codes=reason_codes,
            resource_snapshot=resource_manifest,
        )

    full_steps = int(plan["full_steps"])
    actual_nfe = int(plan["actual_refine_nfe"])
    patched_model, sampler, full_sigmas = setup_sampling_fn(
        model,
        av_latent,
        full_steps,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    )
    if not isinstance(full_sigmas, torch.Tensor) or full_sigmas.ndim != 1:
        raise ValueError("Audio Refine setup returned an invalid sigma tensor")
    if full_sigmas.numel() != full_steps + 1:
        raise ValueError("Audio Refine full sigma schedule has the wrong length")
    sigmas = full_sigmas[-(actual_nfe + 1) :].detach().to(
        device="cpu", dtype=torch.float32
    )
    expected_sigmas = torch.tensor(plan["video_sigmas"], dtype=torch.float32)
    if not torch.allclose(sigmas, expected_sigmas, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("Audio Refine sigma tail differs from the signed Plan")

    video, audio = nested_av_parts(av_latent)
    refined_latent = av_latent.copy()
    refined_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (
            torch.zeros(video.shape, dtype=torch.float32, device=video.device),
            torch.ones(audio.shape, dtype=torch.float32, device=audio.device),
        )
    )
    report = {
        "schema": "t8.minimax_h3.audio_refine.setup.v1",
        "decision": "ALLOW",
        "reason_codes": reason_codes,
        "bypassed": False,
        "actual_refine_nfe": actual_nfe,
        "full_steps": full_steps,
        "sigmas": [float(value) for value in sigmas.tolist()],
        "video_noise_mask": 0.0,
        "audio_noise_mask": 1.0,
        "cfg": 1.0,
        "resource_snapshot": resource_manifest,
        "plan_payload_sha256": plan["payload_sha256"],
        "quality_claim": "none; sampling and human listening have not occurred",
    }
    return AudioRefineSetupResult(
        model=patched_model,
        noise=AudioRefineRandomNoise(int(plan["refine_seed"])),
        guider=AudioRefineBasicGuider(patched_model, positive),
        sampler=sampler,
        sigmas=sigmas,
        latent=refined_latent,
        report_json=canonical_json(report, indent=2),
    )
