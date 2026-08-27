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

from .audio_integrity_advanced import (
    analyze_audio_integrity,
    analyze_audio_perceptual_drift,
)
from .core import nested_av_parts, split_noise_masks, validate_audio
from .nfe_run_contract_advanced import compile_nfe_run_contract
from .sampling import setup_dual_clock_sampling
from .vram_policy import runtime_snapshot


AUDIO_REFINE_AUDIT_TYPE = "H3_T8_AUDIO_REFINE_AUDIT"
AUDIO_REFINE_PLAN_TYPE = "H3_T8_AUDIO_REFINE_PLAN"
AUDIO_REFINE_MODEL_ROUTE_TYPE = "H3_T8_AUDIO_REFINE_MODEL_ROUTE"
AUDIO_REFINE_PHASE2_PLAN_TYPE = "H3_T8_AUDIO_REFINE_PHASE2_PLAN"
AUDIO_REFINE_AUDIT_SCHEMA = "t8.minimax_h3.audio_refine.audit.v1"
AUDIO_REFINE_PLAN_SCHEMA = "t8.minimax_h3.audio_refine.plan.v1"
AUDIO_REFINE_MODEL_ROUTE_SCHEMA = "t8.minimax_h3.audio_refine.model_route.v1"
AUDIO_REFINE_PHASE2_PLAN_SCHEMA = "t8.minimax_h3.audio_refine.phase2_plan.v1"

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


def _safe_signature_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _qualified_name(value)
    if isinstance(value, torch.Tensor):
        return {
            "type": _qualified_name(value),
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _safe_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_signature_value(item) for item in value]
    return {"type": _qualified_name(value)}


def _patch_payload_kind(payload: Any) -> str:
    if isinstance(payload, (list, tuple)) and payload:
        head = payload[0]
        if isinstance(head, str):
            return head.lower()
    name = _qualified_name(payload).lower()
    if "loraadapter" in name:
        return "lora_adapter"
    return name


def _patch_entry_signature(entry: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        return {"valid": False, "type": _qualified_name(entry)}, False
    strength_patch = _finite_number(entry[0])
    strength_model = _finite_number(entry[2])
    offset = entry[3] if len(entry) > 3 else None
    function = entry[4] if len(entry) > 4 else None
    valid = strength_patch is not None and strength_model is not None
    return {
        "valid": valid,
        "strength_patch": strength_patch,
        "strength_model": strength_model,
        "payload_kind": _patch_payload_kind(entry[1]),
        "payload_type": _qualified_name(entry[1]),
        "offset": _safe_signature_value(offset),
        "function": None if function is None else _qualified_name(function),
    }, valid


def _runtime_weight_stack_manifest(model: Any) -> dict[str, Any]:
    model_manifest, is_h3_model, patch_stack_unvalidated = _model_manifest(model)
    patches = _mapping(getattr(model, "patches", {}))
    attachments = _mapping(getattr(model, "attachments", {}))
    metadata = attachments.get("lora_metadata")
    metadata_signature = (
        _safe_signature_value(metadata) if isinstance(metadata, Mapping) else None
    )
    key_records: list[dict[str, Any]] = []
    strengths_patch: set[float] = set()
    strengths_model: set[float] = set()
    payload_kinds: set[str] = set()
    entry_count = 0
    maximum_entries_per_key = 0
    invalid_entry_count = 0
    offsets_present = False
    functions_present = False
    for key in sorted(patches, key=str):
        raw_entries = patches[key]
        entries = (
            list(raw_entries)
            if isinstance(raw_entries, (list, tuple))
            else [raw_entries]
        )
        maximum_entries_per_key = max(maximum_entries_per_key, len(entries))
        entry_signatures = []
        for entry in entries:
            signature, valid = _patch_entry_signature(entry)
            entry_signatures.append(signature)
            entry_count += 1
            if not valid:
                invalid_entry_count += 1
            if signature.get("strength_patch") is not None:
                strengths_patch.add(float(signature["strength_patch"]))
            if signature.get("strength_model") is not None:
                strengths_model.add(float(signature["strength_model"]))
            payload_kinds.add(str(signature.get("payload_kind")))
            offsets_present = offsets_present or signature.get("offset") is not None
            functions_present = functions_present or signature.get("function") is not None
        key_records.append({"key": str(key), "entries": entry_signatures})

    metadata_base = "" if metadata_signature is None else str(
        metadata_signature.get("base_model", "")
    )
    metadata_steps = "" if metadata_signature is None else str(
        metadata_signature.get("sampler_steps", "")
    )
    metadata_source_sha = "" if metadata_signature is None else str(
        metadata_signature.get("conversion_source_sha256", "")
    )
    metadata_is_turbo4 = (
        metadata_base.lower().replace("_", "-") == "minimax-h3"
        and metadata_steps == "4"
        and re.fullmatch(r"[0-9a-fA-F]{64}", metadata_source_sha) is not None
    )
    single_lora_structure = (
        entry_count > 0
        and maximum_entries_per_key == 1
        and invalid_entry_count == 0
        and strengths_patch == {1.0}
        and strengths_model == {1.0}
        and payload_kinds <= {"lora", "lora_adapter"}
        and not offsets_present
        and not functions_present
    )
    portable = {
        "weight_patches": key_records,
        "attachment_keys": sorted(str(key) for key in attachments),
        "lora_metadata": metadata_signature,
    }
    manifest = {
        "model_object_id": int(id(model)),
        "base_object_id": int(id(getattr(model, "model", None))),
        "clone_base_uuid": str(getattr(model, "clone_base_uuid", "")),
        "patches_uuid": str(getattr(model, "patches_uuid", "")),
        "is_h3_model": bool(is_h3_model),
        "patch_stack_unvalidated": bool(patch_stack_unvalidated),
        "model_structure_sha256": model_manifest["structure_sha256"],
        "weight_patch_key_count": len(patches),
        "weight_patch_entry_count": entry_count,
        "maximum_entries_per_key": maximum_entries_per_key,
        "invalid_entry_count": invalid_entry_count,
        "strength_patch_values": sorted(strengths_patch),
        "strength_model_values": sorted(strengths_model),
        "payload_kinds": sorted(payload_kinds),
        "offsets_present": offsets_present,
        "functions_present": functions_present,
        "attachment_keys": sorted(str(key) for key in attachments),
        "lora_metadata": metadata_signature,
        "lora_metadata_sha256": (
            None
            if metadata_signature is None
            else _sha256_text(canonical_json(metadata_signature))
        ),
        "weight_stack_sha256": _sha256_text(canonical_json(portable)),
        "turbo4_single_stack": bool(metadata_is_turbo4 and single_lora_structure),
    }
    runtime_payload = {
        **manifest,
        "runtime_stack_sha256": None,
    }
    runtime_payload.pop("runtime_stack_sha256")
    manifest["runtime_stack_sha256"] = _sha256_text(canonical_json(runtime_payload))
    return manifest


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


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def route_audio_refine_model(
    *,
    audit: dict[str, Any],
    first_pass_model: Any,
    refine_model: Any,
    route_strategy: str,
    declared_first_pass_nfe: int = 4,
) -> tuple[Any, dict[str, Any], str, str]:
    audit = _validate_signed_descriptor(
        audit,
        schema=AUDIO_REFINE_AUDIT_SCHEMA,
        label="audio refine audit",
    )
    if route_strategy not in {"same_turbo_stack", "base_without_turbo"}:
        raise ValueError(
            "route_strategy must be same_turbo_stack or base_without_turbo"
        )
    if isinstance(declared_first_pass_nfe, bool) or declared_first_pass_nfe != 4:
        raise ValueError("Phase 2 currently requires declared_first_pass_nfe=4")

    audit_bindings = _mapping(audit.get("bindings", {}))
    first_model_manifest, _, _ = _model_manifest(first_pass_model)
    if int(audit_bindings.get("model_object_id", -1)) != id(first_pass_model):
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: first_pass_model is not the audited MODEL object"
        )
    if audit_bindings.get("model_structure_sha256") != first_model_manifest.get(
        "structure_sha256"
    ):
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: first_pass_model structure changed after Audit"
        )

    first_stack = _runtime_weight_stack_manifest(first_pass_model)
    refine_stack = _runtime_weight_stack_manifest(refine_model)
    reason_codes = list(audit.get("reason_codes", []))
    audit_decision = str(audit.get("decision"))
    if audit_decision not in {"ALLOW", "ABSTAIN", "REJECT"}:
        raise ValueError("REJECT_DESCRIPTOR_TAMPERED: audit decision is unsupported")

    if first_stack["maximum_entries_per_key"] > 1:
        _append_reason(reason_codes, "ABSTAIN_REPEATED_OR_MIXED_LORA_STACK")
    elif not first_stack["turbo4_single_stack"]:
        _append_reason(reason_codes, "ABSTAIN_UNKNOWN_FIRST_PASS_STACK")
    if first_stack["attachment_keys"] != ["lora_metadata"]:
        _append_reason(reason_codes, "ABSTAIN_UNKNOWN_FIRST_PASS_ATTACHMENTS")
    if not refine_stack["is_h3_model"] or refine_stack["patch_stack_unvalidated"]:
        _append_reason(reason_codes, "ABSTAIN_UNKNOWN_REFINE_MODEL_STACK")

    same_base_object = first_stack["base_object_id"] == refine_stack["base_object_id"]
    same_weight_stack = (
        first_stack["weight_stack_sha256"] == refine_stack["weight_stack_sha256"]
        and first_stack["patches_uuid"] == refine_stack["patches_uuid"]
        and first_stack["lora_metadata_sha256"]
        == refine_stack["lora_metadata_sha256"]
    )
    same_weights_api: bool | None = None
    checker = getattr(first_pass_model, "clone_has_same_weights", None)
    if first_pass_model is refine_model:
        same_weights_api = True
    elif callable(checker):
        try:
            same_weights_api = bool(checker(refine_model))
        except Exception:
            same_weights_api = False

    if route_strategy == "same_turbo_stack":
        if not refine_stack["turbo4_single_stack"]:
            _append_reason(reason_codes, "ABSTAIN_REFINE_STACK_NOT_TURBO4")
        if not same_base_object or not same_weight_stack:
            _append_reason(reason_codes, "ABSTAIN_SAME_STACK_RELATION_UNPROVEN")
        if same_weights_api is False:
            _append_reason(reason_codes, "ABSTAIN_SAME_WEIGHTS_API_REJECTED")
    else:
        if not same_base_object:
            _append_reason(reason_codes, "ABSTAIN_BASE_OBJECT_RELATION_UNPROVEN")
        if (
            refine_stack["weight_patch_entry_count"] != 0
            or refine_stack["lora_metadata"] is not None
        ):
            _append_reason(reason_codes, "ABSTAIN_REFINE_MODEL_STILL_PATCHED")
        if refine_stack["attachment_keys"]:
            _append_reason(reason_codes, "ABSTAIN_UNKNOWN_REFINE_MODEL_ATTACHMENTS")

    new_route_reasons = [
        code for code in reason_codes if code not in audit.get("reason_codes", [])
    ]
    if audit_decision == "REJECT":
        decision = "REJECT"
    elif audit_decision == "ABSTAIN" or new_route_reasons:
        decision = "ABSTAIN"
    else:
        decision = "ALLOW"
    relationship = {
        "same_base_object": same_base_object,
        "same_weight_stack": same_weight_stack,
        "clone_has_same_weights": same_weights_api,
    }
    descriptor = _finish_descriptor(
        {
            "schema": AUDIO_REFINE_MODEL_ROUTE_SCHEMA,
            "decision": decision,
            "reason_codes": reason_codes,
            "audit_payload_sha256": audit["payload_sha256"],
            "audit": audit,
            "route_strategy": route_strategy,
            "declared_first_pass_nfe": int(declared_first_pass_nfe),
            "first_pass_stack": first_stack,
            "refine_stack": refine_stack,
            "relationship": relationship,
            "bindings": {
                "first_pass_model_object_id": int(id(first_pass_model)),
                "refine_model_object_id": int(id(refine_model)),
                "first_pass_runtime_stack_sha256": first_stack[
                    "runtime_stack_sha256"
                ],
                "refine_runtime_stack_sha256": refine_stack["runtime_stack_sha256"],
            },
            "identity_boundary": (
                "Runtime stack identity only; disk asset paths are recorded by the "
                "validation runner, not inferred from MODEL."
            ),
            "quality_claim": "none; model routing is only a mechanical contract",
        }
    )
    return refine_model, descriptor, decision, canonical_json(descriptor, indent=2)


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


def plan_audio_refine_phase2(
    route: dict[str, Any],
    refine_steps: int,
    audio_denoise: float,
    refine_seed: int,
) -> tuple[dict[str, Any], str, str]:
    route = _validate_signed_descriptor(
        route,
        schema=AUDIO_REFINE_MODEL_ROUTE_SCHEMA,
        label="audio refine model route",
    )
    if isinstance(refine_steps, bool) or refine_steps != 4:
        raise ValueError("Phase 2 currently requires refine_steps=4")
    if isinstance(audio_denoise, bool):
        raise ValueError("audio_denoise must be 0.35 or 0.50")
    denoise = float(audio_denoise)
    registered = next(
        (
            value
            for value in (0.35, 0.50)
            if math.isclose(denoise, value, rel_tol=0.0, abs_tol=1.0e-9)
        ),
        None,
    )
    if registered is None:
        raise ValueError(
            "Phase 2 audio_denoise must be a pre-registered point: 0.35 or 0.50"
        )
    if isinstance(refine_seed, bool) or not isinstance(refine_seed, int):
        raise ValueError("refine_seed must be an integer")
    if not 0 <= refine_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("refine_seed must be between 0 and 2^64-1")
    decision = str(route.get("decision"))
    if decision not in {"ALLOW", "ABSTAIN", "REJECT"}:
        raise ValueError("REJECT_DESCRIPTOR_TAMPERED: route decision is unsupported")

    full_steps = int(refine_steps / registered)
    base_sigmas = [
        float((refine_steps - index) / full_steps)
        for index in range(refine_steps + 1)
    ]
    video_sigmas = [_shift_sigma(value, 12.0) for value in base_sigmas]
    audio_sigmas = [_shift_sigma(value, 3.0) for value in base_sigmas]
    first_pass_nfe = int(route["declared_first_pass_nfe"])
    descriptor = _finish_descriptor(
        {
            "schema": AUDIO_REFINE_PHASE2_PLAN_SCHEMA,
            "decision": decision,
            "reason_codes": list(route.get("reason_codes", [])),
            "route_payload_sha256": route["payload_sha256"],
            "route": route,
            "route_strategy": route["route_strategy"],
            "declared_first_pass_nfe": first_pass_nfe,
            "actual_refine_nfe": int(refine_steps),
            "declared_total_nfe": first_pass_nfe + int(refine_steps),
            "full_steps": int(full_steps),
            "requested_audio_denoise": registered,
            "effective_audio_denoise": float(refine_steps / full_steps),
            "refine_seed": int(refine_seed),
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
            "training_distribution_equivalence_claim": False,
            "quality_claim": "none; equal total NFE does not imply equal training distribution",
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


def _verify_phase2_setup_bindings(
    plan: Mapping[str, Any],
    refine_model: Any,
    positive: Any,
    av_latent: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    route = _validate_signed_descriptor(
        plan.get("route"),
        schema=AUDIO_REFINE_MODEL_ROUTE_SCHEMA,
        label="embedded audio refine model route",
    )
    if plan.get("route_payload_sha256") != route.get("payload_sha256"):
        raise ValueError(
            "REJECT_DESCRIPTOR_TAMPERED: Phase 2 Plan does not bind the Route"
        )
    audit = _validate_signed_descriptor(
        route.get("audit"),
        schema=AUDIO_REFINE_AUDIT_SCHEMA,
        label="embedded audio refine audit",
    )
    if route.get("audit_payload_sha256") != audit.get("payload_sha256"):
        raise ValueError(
            "REJECT_DESCRIPTOR_TAMPERED: Model Route does not bind the Audit"
        )

    mismatches: list[str] = []
    route_bindings = _mapping(route.get("bindings", {}))
    refine_stack = _runtime_weight_stack_manifest(refine_model)
    if int(route_bindings.get("refine_model_object_id", -1)) != id(refine_model):
        mismatches.append("refine_model_object")
    if route_bindings.get("refine_runtime_stack_sha256") != refine_stack.get(
        "runtime_stack_sha256"
    ):
        mismatches.append("refine_model_stack")

    audit_bindings = _mapping(audit.get("bindings", {}))
    try:
        run_contract_sha256 = _recompile_bound_run_contract(audit, positive)
    except ValueError:
        mismatches.append("conditioning")
    else:
        if audit_bindings.get("run_contract_sha256") != run_contract_sha256:
            mismatches.append("conditioning")
    try:
        latent_result = classify_audio_refine_latent(
            av_latent,
            hash_chunk_megabytes=int(audit.get("hash_chunk_megabytes", 8)),
        )
    except (TypeError, ValueError, RuntimeError):
        mismatches.append("latent")
    else:
        if audit_bindings.get("latent_manifest_sha256") != latent_result.get(
            "manifest_sha256"
        ):
            mismatches.append("latent")
    if mismatches:
        raise ValueError(
            "REJECT_CONTRACT_MISMATCH: changed " + ", ".join(sorted(set(mismatches)))
        )
    return route, audit


def setup_audio_refine_dual_model(
    *,
    plan: dict[str, Any],
    refine_model: Any,
    positive: Any,
    av_latent: dict,
    setup_sampling_fn=setup_dual_clock_sampling,
    runtime_snapshot_fn=runtime_snapshot,
) -> AudioRefineSetupResult:
    plan = _validate_signed_descriptor(
        plan,
        schema=AUDIO_REFINE_PHASE2_PLAN_SCHEMA,
        label="audio refine Phase 2 plan",
    )
    route, audit = _verify_phase2_setup_bindings(
        plan, refine_model, positive, av_latent
    )
    plan_decision = str(plan.get("decision"))
    if plan_decision == "REJECT":
        raise ValueError(
            "REJECT_AUDIO_REFINE_PHASE2_PLAN: "
            + ", ".join(plan.get("reason_codes", []))
        )
    if plan_decision not in {"ALLOW", "ABSTAIN"}:
        raise ValueError("REJECT_DESCRIPTOR_TAMPERED: Phase 2 decision is unsupported")

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
    if resource_reason:
        _append_reason(reason_codes, resource_reason)
    if plan_decision == "ABSTAIN" or resource_reason:
        return _bypass_setup_result(
            plan=plan,
            model=refine_model,
            positive=positive,
            av_latent=av_latent,
            reason_codes=reason_codes,
            resource_snapshot=resource_manifest,
        )

    full_steps = int(plan["full_steps"])
    actual_nfe = int(plan["actual_refine_nfe"])
    patched_model, sampler, full_sigmas = setup_sampling_fn(
        refine_model,
        av_latent,
        full_steps,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    )
    if not isinstance(full_sigmas, torch.Tensor) or full_sigmas.ndim != 1:
        raise ValueError("Audio Refine Phase 2 returned an invalid sigma tensor")
    if full_sigmas.numel() != full_steps + 1:
        raise ValueError("Audio Refine Phase 2 full sigma schedule has the wrong length")
    sigmas = full_sigmas[-(actual_nfe + 1) :].detach().to(
        device="cpu", dtype=torch.float32
    )
    expected_sigmas = torch.tensor(plan["video_sigmas"], dtype=torch.float32)
    if not torch.allclose(sigmas, expected_sigmas, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("Audio Refine Phase 2 sigma tail differs from the signed Plan")

    video, audio = nested_av_parts(av_latent)
    refined_latent = av_latent.copy()
    refined_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (
            torch.zeros(video.shape, dtype=torch.float32, device=video.device),
            torch.ones(audio.shape, dtype=torch.float32, device=audio.device),
        )
    )
    report = {
        "schema": "t8.minimax_h3.audio_refine.phase2_setup.v1",
        "decision": "ALLOW",
        "reason_codes": reason_codes,
        "bypassed": False,
        "route_strategy": route["route_strategy"],
        "declared_first_pass_nfe": plan["declared_first_pass_nfe"],
        "actual_refine_nfe": actual_nfe,
        "declared_total_nfe": plan["declared_total_nfe"],
        "full_steps": full_steps,
        "sigmas": [float(value) for value in sigmas.tolist()],
        "video_noise_mask": 0.0,
        "audio_noise_mask": 1.0,
        "cfg": 1.0,
        "resource_snapshot": resource_manifest,
        "plan_payload_sha256": plan["payload_sha256"],
        "refine_runtime_stack_sha256": route["bindings"][
            "refine_runtime_stack_sha256"
        ],
        "training_distribution_equivalence_claim": False,
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


def _chunked_max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("tensor shapes differ")
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    maximum = 0.0
    for start in range(0, int(left_flat.numel()), 262_144):
        delta = (
            right_flat[start : start + 262_144].to(dtype=torch.float32)
            - left_flat[start : start + 262_144].to(dtype=torch.float32)
        )
        maximum = max(maximum, float(torch.amax(torch.abs(delta)).item()))
    return maximum


def gate_audio_refine_candidate(
    *,
    original_av_latent: dict,
    candidate_av_latent: dict,
    original_audio: Mapping,
    candidate_audio: Mapping,
    accept_candidate: bool = False,
    video_frame_count: int = 0,
    fps: float = 24.0,
    maximum_duration_delta_ms: float = 50.0,
    spectral_drift_threshold: float = 0.30,
    level_delta_threshold_db: float = 4.0,
    persistent_window_count: int = 3,
) -> tuple[dict, Mapping, bool, str, str]:
    """Fail closed, then splice only the reviewed candidate audio into original video."""

    original_video, original_audio_latent = nested_av_parts(original_av_latent)
    if not bool(torch.isfinite(original_video).all().item()) or not bool(
        torch.isfinite(original_audio_latent).all().item()
    ):
        raise ValueError("original AV latent contains NaN or Infinity")
    original_waveform, original_rate = validate_audio(original_audio, "original_audio")
    if not bool(torch.isfinite(original_waveform).all().item()):
        raise ValueError("original decoded audio contains NaN or Infinity")

    hard_reason_codes: list[str] = []
    review_reason_codes: list[str] = []
    candidate_video = candidate_audio_latent = None
    candidate_video_changed = None
    candidate_video_max_abs = None
    try:
        candidate_video, candidate_audio_latent = nested_av_parts(candidate_av_latent)
        if tuple(candidate_video.shape) != tuple(original_video.shape) or tuple(
            candidate_audio_latent.shape
        ) != tuple(original_audio_latent.shape):
            hard_reason_codes.append("REJECT_CANDIDATE_LATENT_SHAPE_MISMATCH")
        if not bool(torch.isfinite(candidate_video).all().item()) or not bool(
            torch.isfinite(candidate_audio_latent).all().item()
        ):
            hard_reason_codes.append("REJECT_CANDIDATE_LATENT_INVALID")
        if tuple(candidate_video.shape) == tuple(original_video.shape):
            candidate_video_changed = not bool(
                torch.equal(original_video, candidate_video)
            )
            candidate_video_max_abs = _chunked_max_abs_delta(
                original_video, candidate_video
            )
    except (KeyError, TypeError, ValueError, RuntimeError):
        hard_reason_codes.append("REJECT_CANDIDATE_LATENT_INVALID")

    integrity_report = None
    drift_report = None
    try:
        candidate_waveform, candidate_rate = validate_audio(
            candidate_audio, "candidate_audio"
        )
        if not bool(torch.isfinite(candidate_waveform).all().item()):
            hard_reason_codes.append("REJECT_CANDIDATE_AUDIO_NONFINITE")
        if int(candidate_rate) != int(original_rate):
            hard_reason_codes.append("REJECT_CANDIDATE_SAMPLE_RATE_MISMATCH")
        if tuple(candidate_waveform.shape[:-1]) != tuple(original_waveform.shape[:-1]):
            hard_reason_codes.append("REJECT_CANDIDATE_CHANNEL_SHAPE_MISMATCH")
        duration_delta_ms = (
            int(candidate_waveform.shape[-1]) * 1000.0 / int(candidate_rate)
            - int(original_waveform.shape[-1]) * 1000.0 / int(original_rate)
        )
        if abs(duration_delta_ms) > float(maximum_duration_delta_ms):
            hard_reason_codes.append("REJECT_CANDIDATE_DURATION_MISMATCH")

        integrity_values = analyze_audio_integrity(
            candidate_audio,
            video_frame_count=int(video_frame_count),
            fps=float(fps),
            max_av_delta_ms=float(maximum_duration_delta_ms),
        )
        integrity_report = json.loads(integrity_values[-1])
        review_reason_codes.extend(
            f"INTEGRITY_{item.get('code', 'UNKNOWN').upper()}"
            for item in integrity_report.get("findings", [])
        )

        drift_values = analyze_audio_perceptual_drift(
            original_audio,
            candidate_audio,
            spectral_drift_threshold=float(spectral_drift_threshold),
            level_delta_threshold_db=float(level_delta_threshold_db),
            persistent_window_count=int(persistent_window_count),
            max_duration_delta_ms=float(maximum_duration_delta_ms),
        )
        drift_report = json.loads(drift_values[-1])
        for item in drift_report.get("findings", []):
            code = str(item.get("code", "unknown"))
            if code in {"sample_rate_mismatch", "duration_mismatch", "nonfinite_samples"}:
                hard_reason_codes.append(f"REJECT_DRIFT_{code.upper()}")
            else:
                review_reason_codes.append(f"DRIFT_{code.upper()}")
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        hard_reason_codes.append("REJECT_CANDIDATE_AUDIO_INVALID")

    hard_reason_codes = list(dict.fromkeys(hard_reason_codes))
    review_reason_codes = list(dict.fromkeys(review_reason_codes))
    eligible = not hard_reason_codes
    selected = False
    selected_latent = original_av_latent
    selected_audio = original_audio
    if eligible and bool(accept_candidate):
        assert candidate_audio_latent is not None
        selected_latent = original_av_latent.copy()
        selected_latent["samples"] = comfy.nested_tensor.NestedTensor(
            (original_video, candidate_audio_latent)
        )
        selected_audio = candidate_audio
        selected = True
        decision = "ACCEPT_CANDIDATE"
    elif not eligible:
        decision = "REJECT_CANDIDATE"
    else:
        decision = "ABSTAIN_HUMAN_REVIEW_REQUIRED"

    report = {
        "schema": "t8.minimax_h3.audio_refine.quality_gate.v1",
        "decision": decision,
        "accept_candidate_requested": bool(accept_candidate),
        "candidate_mechanically_eligible": eligible,
        "candidate_selected": selected,
        "hard_reason_codes": hard_reason_codes,
        "review_reason_codes": review_reason_codes,
        "candidate_video_changed_during_sampling": candidate_video_changed,
        "candidate_video_max_abs": candidate_video_max_abs,
        "output_video_exact_original": True,
        "selected_video_relocked_exact": bool(selected),
        "integrity_audit": integrity_report,
        "perceptual_drift_audit": drift_report,
        "fallback": "original AV latent and original decoded audio",
        "quality_claim": "none; human listening remains authoritative",
    }
    return (
        selected_latent,
        selected_audio,
        selected,
        decision,
        canonical_json(report, indent=2),
    )
