from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .hybrid_model import (
    ALGORITHM,
    ARTIFACT_SCHEMA,
    ATTACHMENT_KEY,
    KNOWN_QUALITY_BASE_SHA256,
    KNOWN_QUALITY_CURVE_SHA256,
    KNOWN_REFERENCE_CURVE_SHA256,
    KNOWN_REFERENCE_OVERLAY_SHA256,
    VRAM_POLICY_ATTACHMENT_KEY,
    _expected_operations,
    audit_conditioning_references,
    canonical_json,
    sha256_bytes,
)
from .vram_policy import runtime_snapshot


COMPATIBILITY_SCHEMA = "t8.minimax_h3.hybrid_compatibility_audit.v1"
BLOCK_CACHE_KEY = "minimax_h3_block_cache_t8"
BLOCK_CACHE_WRAPPER_KEY = "minimax_h3_block_cache_t8"
LONG_VIDEO_CONDITIONING_KEY = "t8_long_video_schema"
MULTIKEYFRAME_CONDITIONING_KEY = "t8_multikeyframe_schema"
PATCH_VERSION = 1


def _issue(target: list[dict[str, Any]], code: str, message: str, **evidence: Any) -> None:
    value: dict[str, Any] = {"code": code, "message": message}
    if evidence:
        value["evidence"] = evidence
    target.append(value)


def _attachment(model: Any, key: str) -> Any:
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    attachments = getattr(model, "attachments", {})
    return attachments.get(key) if isinstance(attachments, Mapping) else None


def _offset(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _patch_kind(entry: Any) -> str | None:
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 2:
        return None
    patch = entry[1]
    if not isinstance(patch, Sequence) or isinstance(patch, (str, bytes)) or not patch:
        return None
    return str(patch[0])


def _patch_target_shape(entry: Any) -> tuple[int, ...] | None:
    try:
        arguments = entry[1][1]
        target = arguments[0]
    except (IndexError, KeyError, TypeError):
        return None
    shape = getattr(target, "shape", None)
    if shape is None:
        getter = getattr(target, "get_shape", None)
        shape = getter() if callable(getter) else None
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _entry_offset(entry: Any) -> tuple[int, int, int] | None:
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 4:
        return None
    return _offset(entry[3])


def _offsets_overlap(candidate: tuple[int, int, int] | None, expected: tuple[int, int, int]) -> bool:
    if candidate is None:
        return True
    candidate_dim, candidate_start, candidate_length = candidate
    expected_dim, expected_start, expected_length = expected
    if candidate_dim != expected_dim:
        return True
    candidate_end = candidate_start + candidate_length
    expected_end = expected_start + expected_length
    return candidate_start < expected_end and expected_start < candidate_end


def _validate_hybrid_attachment(
    attachment: Any,
    hard: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(attachment, Mapping):
        _issue(hard, "hybrid_attachment_missing", "MODEL does not carry the validated T8 Hybrid artifact attachment.")
        return None, []
    identity = attachment.get("identity")
    if not isinstance(identity, Mapping):
        _issue(hard, "hybrid_identity_missing", "Hybrid attachment identity is missing or malformed.")
        return None, []
    expected_contract_fields = {
        "schema": ARTIFACT_SCHEMA,
        "algorithm": ALGORITHM,
    }
    for key, expected in expected_contract_fields.items():
        if identity.get(key) != expected:
            _issue(
                hard,
                "hybrid_contract_mismatch",
                f"Hybrid attachment contract field {key!r} is unsupported.",
                field=key,
                actual=identity.get(key),
                expected=expected,
            )
    reference_identity_fields = {
        "base_sha256": KNOWN_QUALITY_BASE_SHA256,
        "overlay_sha256": KNOWN_REFERENCE_OVERLAY_SHA256,
        "base_curve_sha256": KNOWN_QUALITY_CURVE_SHA256,
        "overlay_curve_sha256": KNOWN_REFERENCE_CURVE_SHA256,
    }
    for key, expected in reference_identity_fields.items():
        if identity.get(key) != expected:
            _issue(
                warnings,
                "hybrid_reference_identity_mismatch",
                f"Hybrid attachment identity field {key!r} differs from the original reference pair; user-selected model identity is diagnostic only.",
                field=key,
                actual=identity.get(key),
                expected=expected,
            )
    fingerprint = sha256_bytes(canonical_json(dict(identity)).encode("utf-8"))
    if attachment.get("fingerprint") != fingerprint:
        _issue(hard, "hybrid_fingerprint_mismatch", "Hybrid attachment fingerprint does not match its identity.")
    try:
        operations = _expected_operations(dict(identity))
    except (KeyError, TypeError, ValueError) as error:
        _issue(hard, "hybrid_recipe_invalid", f"Hybrid recipe is not canonical: {error}")
        return dict(identity), []
    if int(attachment.get("operation_count", -1)) != len(operations):
        _issue(
            hard,
            "hybrid_operation_count_mismatch",
            "Hybrid attachment operation count does not match the canonical recipe.",
            actual=attachment.get("operation_count"),
            expected=len(operations),
        )
    expected_payload = sum(
        2 * _product(operation["shape"])
        for operation in operations
    )
    if int(attachment.get("payload_bytes", -1)) != expected_payload:
        _issue(
            hard,
            "hybrid_payload_size_mismatch",
            "Hybrid attachment payload byte count does not match the canonical recipe.",
            actual=attachment.get("payload_bytes"),
            expected=expected_payload,
        )
    return dict(identity), operations


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _audit_weight_patches(model: Any, operations: list[dict[str, Any]], hard: list[dict[str, Any]]) -> dict[str, Any]:
    patches = getattr(model, "patches", {})
    if not isinstance(patches, Mapping):
        _issue(hard, "model_patches_invalid", "MODEL patches storage is not a mapping.")
        patches = {}

    expected_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        expected_by_key[str(operation["model_key"])].append(operation)

    own_indices_by_key: dict[str, set[int]] = defaultdict(set)
    match_index_by_operation: dict[tuple[str, tuple[int, int, int]], int] = {}
    found = 0
    for key, key_operations in expected_by_key.items():
        entries = patches.get(key, [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            entries = []
        for operation in key_operations:
            expected_offset = _offset(operation["offset"])
            expected_shape = tuple(int(item) for item in operation["shape"])
            matches = [
                index
                for index, entry in enumerate(entries)
                if _patch_kind(entry) == "set"
                and _entry_offset(entry) == expected_offset
                and _patch_target_shape(entry) == expected_shape
            ]
            if len(matches) != 1:
                _issue(
                    hard,
                    "hybrid_set_patch_missing" if not matches else "hybrid_set_patch_duplicate",
                    "Canonical Hybrid offset-set patch is missing or duplicated.",
                    model_key=key,
                    offset=list(expected_offset or ()),
                    matches=len(matches),
                )
                continue
            found += 1
            index = matches[0]
            own_indices_by_key[key].add(index)
            match_index_by_operation[(key, expected_offset)] = index

    overlapping: list[dict[str, Any]] = []
    nonoverlapping_external = 0
    patch_entry_total = 0
    for key, raw_entries in patches.items():
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            continue
        entries = list(raw_entries)
        patch_entry_total += len(entries)
        key_operations = expected_by_key.get(str(key), [])
        for index, entry in enumerate(entries):
            if index in own_indices_by_key.get(str(key), set()):
                continue
            overlaps = [
                operation
                for operation in key_operations
                if _offsets_overlap(_entry_offset(entry), _offset(operation["offset"]))
            ]
            if not overlaps:
                nonoverlapping_external += 1
                continue
            own_positions = [
                match_index_by_operation.get((str(key), _offset(operation["offset"])))
                for operation in overlaps
            ]
            own_positions = [value for value in own_positions if value is not None]
            if own_positions and index < min(own_positions):
                relation = "before_hybrid"
                code = "patch_precedes_hybrid_set"
                message = "An overlapping weight patch precedes the Hybrid set operation; required Hybrid -> LoRA ordering is violated."
            elif own_positions and index > max(own_positions):
                relation = "after_hybrid"
                code = "adaln_patch_overlaps_hybrid"
                message = "A later patch overlaps Hybrid-selected AdaLN rows; this combination is uncalibrated and blocked by the strict audit."
            else:
                relation = "interleaved_or_unresolved"
                code = "hybrid_patch_order_ambiguous"
                message = "Overlapping Hybrid and external weight patches are interleaved or cannot be ordered safely."
            evidence = {
                "model_key": str(key),
                "entry_index": index,
                "patch_kind": _patch_kind(entry),
                "offset": None if _entry_offset(entry) is None else list(_entry_offset(entry)),
                "relation": relation,
                "overlap_count": len(overlaps),
            }
            overlapping.append(evidence)
            _issue(hard, code, message, **evidence)

    return {
        "patch_key_count": len(patches),
        "patch_entry_count": patch_entry_total,
        "hybrid_expected_operations": len(operations),
        "hybrid_entries_found": found,
        "nonoverlapping_external_entries": nonoverlapping_external,
        "overlapping_external_entries": overlapping,
        "required_order": "Hybrid Loader -> LoRA",
    }


def _total_blocks(model: Any) -> int | None:
    getter = getattr(model, "get_model_object", None)
    try:
        diffusion = getter("diffusion_model") if callable(getter) else model.model.diffusion_model
        blocks = getattr(diffusion, "blocks", None)
        return len(blocks) if blocks is not None else None
    except Exception:
        return None


def _audit_block_cache(model: Any, total_blocks: int | None, hard: list[dict[str, Any]]) -> dict[str, Any]:
    model_options = getattr(model, "model_options", {})
    transformer = model_options.get("transformer_options", {}) if isinstance(model_options, Mapping) else {}
    cache = transformer.get(BLOCK_CACHE_KEY) if isinstance(transformer, Mapping) else None
    replacements = {}
    if isinstance(transformer, Mapping):
        patches_replace = transformer.get("patches_replace", {})
        if isinstance(patches_replace, Mapping):
            value = patches_replace.get("dit", {})
            if isinstance(value, Mapping):
                replacements = value
    wrappers = getattr(model, "wrappers", {})
    wrapper_groups = []
    if isinstance(wrappers, Mapping):
        for group, values in wrappers.items():
            if isinstance(values, Mapping) and values.get(BLOCK_CACHE_WRAPPER_KEY):
                wrapper_groups.append(str(group))
    marker_present = cache is not None or bool(wrapper_groups)
    if not marker_present:
        return {"present": False, "complete": True}
    expected_replacements = []
    if total_blocks is not None and total_blocks >= 2:
        expected_replacements = [("double_block", 0), ("double_block", total_blocks - 1)]
    missing = [str(value) for value in expected_replacements if value not in replacements]
    extra_double = [
        str(value)
        for value in replacements
        if isinstance(value, tuple) and value and value[0] == "double_block" and value not in expected_replacements
    ]
    complete = cache is not None and total_blocks is not None and not missing and len(wrapper_groups) >= 2 and not extra_double
    if not complete:
        _issue(
            hard,
            "block_cache_contract_incomplete",
            "MiniMax H3 Block Cache markers are incomplete or conflict with additional DiT replacements.",
            total_blocks=total_blocks,
            missing_replacements=missing,
            extra_double_replacements=extra_double,
            wrapper_groups=wrapper_groups,
        )
    if isinstance(transformer, Mapping) and ("easycache" in transformer or "lazycache" in transformer):
        _issue(hard, "block_cache_conflicting_cache", "MiniMax H3 Block Cache is combined with EasyCache/LazyCache.")
    config = getattr(cache, "config", None)
    config_report = {
        key: getattr(config, key)
        for key in (
            "residual_diff_threshold",
            "start_percent",
            "end_percent",
            "max_consecutive_hits",
            "cache_device",
            "metric_stride",
        )
        if config is not None and hasattr(config, key)
    }
    return {
        "present": True,
        "complete": complete,
        "wrapper_groups": wrapper_groups,
        "boundary_replacements": [str(value) for value in expected_replacements if value in replacements],
        "config": config_report,
    }


def _callable_name(value: Any) -> str:
    function = getattr(value, "__func__", value)
    return str(getattr(function, "__name__", type(function).__name__))


def _audit_sage(model: Any, total_blocks: int | None, hard: list[dict[str, Any]]) -> dict[str, Any]:
    object_patches = getattr(model, "object_patches", {})
    values: dict[int, str] = {}
    if isinstance(object_patches, Mapping):
        prefix = "diffusion_model.blocks."
        suffix = ".attn.forward"
        for key, value in object_patches.items():
            if not isinstance(key, str) or not key.startswith(prefix) or not key.endswith(suffix):
                continue
            raw_index = key[len(prefix) : -len(suffix)]
            try:
                values[int(raw_index)] = _callable_name(value)
            except ValueError:
                _issue(hard, "attention_patch_index_invalid", "An attention forward patch has an invalid block index.", key=key)
    if not values:
        return {"present": False, "complete": True, "patched_blocks": 0}
    expected = {index for index, name in values.items() if name == "minimax_sageattn_forward"}
    unknown = {index: name for index, name in values.items() if name != "minimax_sageattn_forward"}
    complete = total_blocks is not None and expected == set(range(total_blocks)) and not unknown
    if unknown:
        _issue(hard, "unknown_attention_forward_patch", "Unknown H3 attention forward replacement is present.", patches=unknown)
    if expected and not complete:
        _issue(
            hard,
            "sage_attention_contract_incomplete",
            "MiniMax H3 SageAttention must patch every DiT block or remain absent.",
            patched_blocks=len(expected),
            total_blocks=total_blocks,
            missing=sorted(set(range(total_blocks or 0)) - expected),
        )
    return {
        "present": bool(expected),
        "complete": complete,
        "patched_blocks": len(expected),
        "total_blocks": total_blocks,
        "unknown_patches": unknown,
    }


def _conditioning_markers(positive: Any) -> tuple[dict[str, Any] | None, dict[str, int]]:
    if positive is None:
        return None, {"long_video": 0, "multikeyframe": 0}
    reference = audit_conditioning_references(positive)
    markers = {"long_video": 0, "multikeyframe": 0}
    for entry in positive:
        metadata = entry[1]
        if LONG_VIDEO_CONDITIONING_KEY in metadata:
            markers["long_video"] += 1
        if MULTIKEYFRAME_CONDITIONING_KEY in metadata:
            markers["multikeyframe"] += 1
    return reference, markers


def _audit_scoped_model_patches(model: Any, positive: Any, hard: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    object_patches = getattr(model, "object_patches", {})
    object_patches = object_patches if isinstance(object_patches, Mapping) else {}
    extra = object_patches.get("extra_conds")
    forward = object_patches.get("diffusion_model._forward")
    long_version = getattr(extra, "_t8_long_video_patch_version", None)
    multi_extra_version = getattr(extra, "_t8_multikeyframe_patch_version", None)
    multi_forward_version = getattr(forward, "_t8_multikeyframe_patch_version", None)
    long_present = long_version is not None
    multi_present = multi_extra_version is not None or multi_forward_version is not None
    if long_present and multi_present:
        _issue(hard, "long_video_multikeyframe_conflict", "Long Video and MultiKeyframe scoped MODEL patches cannot be stacked.")
    if long_present and long_version != PATCH_VERSION:
        _issue(hard, "long_video_patch_version_unknown", "Long Video patch version is unsupported.", version=long_version)
    if multi_present and (multi_extra_version != PATCH_VERSION or multi_forward_version != PATCH_VERSION):
        _issue(
            hard,
            "multikeyframe_patch_contract_incomplete",
            "MultiKeyframe requires matching version-1 extra_conds and _forward patches.",
            extra_conds_version=multi_extra_version,
            forward_version=multi_forward_version,
        )
    try:
        reference, markers = _conditioning_markers(positive)
    except (TypeError, ValueError) as error:
        _issue(hard, "conditioning_contract_invalid", f"Connected Conditioning could not be audited: {error}")
        reference, markers = None, {"long_video": 0, "multikeyframe": 0}
    if positive is not None:
        if markers["long_video"] and not long_present:
            _issue(hard, "long_video_conditioning_model_mismatch", "Long Video Conditioning is connected without its matching scoped MODEL patch.")
        if markers["multikeyframe"] and not multi_present:
            _issue(hard, "multikeyframe_conditioning_model_mismatch", "MultiKeyframe Conditioning is connected without its matching scoped MODEL patches.")
        if long_present and not markers["long_video"]:
            _issue(warnings, "long_video_model_without_conditioning_marker", "Long Video MODEL patch is present but connected Conditioning has no Long Video marker.")
        if multi_present and not markers["multikeyframe"]:
            _issue(warnings, "multikeyframe_model_without_conditioning_marker", "MultiKeyframe MODEL patch is present but connected Conditioning has no MultiKeyframe marker.")
    return {
        "long_video": {"present": long_present, "version": long_version, "conditioning_entries": markers["long_video"]},
        "multikeyframe": {
            "present": multi_present,
            "extra_conds_version": multi_extra_version,
            "forward_version": multi_forward_version,
            "conditioning_entries": markers["multikeyframe"],
        },
    }, reference


def _audit_sampling(model: Any, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    object_patches = getattr(model, "object_patches", {})
    object_patches = object_patches if isinstance(object_patches, Mapping) else {}
    sampling = object_patches.get("model_sampling")
    if sampling is None:
        route = "stock_or_unpatched"
        class_name = None
    else:
        class_name = type(sampling).__name__
        if class_name in {"MiniMaxH3FlowSampling", "MiniMaxH3NativeAVSampling"}:
            route = "stable_dual_clock_or_native_av"
        elif class_name == "MiniMaxH3MultiRateFlowSamplingEXP":
            route = "experimental_multirate"
        else:
            route = "unknown_custom_sampling"
            _issue(warnings, "unknown_sampling_patch", "A custom model_sampling patch is present but is not a known T8 route.", class_name=class_name)
    model_options = getattr(model, "model_options", {})
    transformer = model_options.get("transformer_options", {}) if isinstance(model_options, Mapping) else {}
    return {
        "route": route,
        "class_name": class_name,
        "shift_video": transformer.get("minimax_h3_sigma_shift_video") if isinstance(transformer, Mapping) else None,
        "shift_audio": transformer.get("minimax_h3_sigma_shift_audio") if isinstance(transformer, Mapping) else None,
    }


def _audit_memory(model: Any, require_policy: bool, minimum_current_headroom_mib: float, minimum_commit_headroom_gib: float, hard: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _attachment(model, VRAM_POLICY_ATTACHMENT_KEY)
    if policy is not None and not isinstance(policy, Mapping):
        _issue(hard, "vram_policy_attachment_invalid", "VRAM policy provenance attachment is malformed.")
        policy = None
    if isinstance(policy, Mapping) and policy.get("schema") != "t8.minimax_h3.vram_policy_apply_report.v1":
        _issue(hard, "vram_policy_attachment_schema_unknown", "VRAM policy provenance attachment schema is unsupported.", schema=policy.get("schema"))
    if require_policy and not isinstance(policy, Mapping):
        _issue(hard, "vram_policy_required_missing", "An applied T8 VRAM policy is required, but the MODEL has no provenance attachment.")
    if require_policy and isinstance(policy, Mapping) and not bool(policy.get("applied")):
        _issue(hard, "vram_policy_required_not_applied", "The connected policy was report-only and did not reserve VRAM.")
    try:
        runtime = runtime_snapshot()
    except Exception as error:
        runtime = {"inspection_error": f"{type(error).__name__}: {error}"}
    gpu = runtime.get("gpu", {}) if isinstance(runtime, Mapping) else {}
    host = runtime.get("host", {}) if isinstance(runtime, Mapping) else {}
    comfy = runtime.get("comfy", {}) if isinstance(runtime, Mapping) else {}
    free_mib = gpu.get("whole_device_free_mib") if isinstance(gpu, Mapping) else None
    commit_gib = host.get("commit_headroom_gib") if isinstance(host, Mapping) else None
    if free_mib is None:
        _issue(warnings, "gpu_headroom_unavailable", "Whole-device CUDA headroom is unavailable; current memory gate could not be proven.")
    elif float(free_mib) < float(minimum_current_headroom_mib):
        _issue(
            hard,
            "current_gpu_headroom_below_gate",
            "Current whole-device free VRAM is below the configured compatibility gate.",
            actual_mib=float(free_mib),
            minimum_mib=float(minimum_current_headroom_mib),
        )
    if commit_gib is None:
        _issue(warnings, "host_commit_headroom_unavailable", "Host commit headroom is unavailable; VBAR backing capacity could not be proven.")
    elif float(commit_gib) < float(minimum_commit_headroom_gib):
        _issue(
            hard,
            "host_commit_headroom_below_gate",
            "Host commit headroom is below the configured compatibility gate.",
            actual_gib=float(commit_gib),
            minimum_gib=float(minimum_commit_headroom_gib),
        )
    dynamic = comfy.get("dynamic_vram_enabled") if isinstance(comfy, Mapping) else None
    if dynamic is not True:
        _issue(warnings, "dynamic_vram_not_confirmed", "DynamicVRAM/VBAR is not confirmed active in the current runtime snapshot.")
    _issue(
        warnings,
        "memory_gate_is_not_peak_proof",
        "Current free VRAM and host commit are preflight observations, not a prediction of peak activations, attention workspaces, VAE/CLIP allocations, CUDA context, or other processes.",
    )
    return dict(policy) if isinstance(policy, Mapping) else {"present": False}, dict(runtime)


def audit_hybrid_compatibility(
    model: Any,
    positive: Any = None,
    *,
    require_applied_vram_policy: bool = False,
    minimum_current_headroom_mib: float = 512.0,
    minimum_commit_headroom_gib: float = 16.0,
) -> dict[str, Any]:
    hard: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    hybrid_attachment = _attachment(model, ATTACHMENT_KEY)
    identity, operations = _validate_hybrid_attachment(
        hybrid_attachment,
        hard,
        warnings,
    )
    weight_patches = _audit_weight_patches(model, operations, hard)
    total_blocks = _total_blocks(model)
    block_cache = _audit_block_cache(model, total_blocks, hard)
    sage = _audit_sage(model, total_blocks, hard)
    scoped, reference = _audit_scoped_model_patches(model, positive, hard, warnings)
    sampling = _audit_sampling(model, warnings)
    policy, runtime = _audit_memory(
        model,
        bool(require_applied_vram_policy),
        float(minimum_current_headroom_mib),
        float(minimum_commit_headroom_gib),
        hard,
        warnings,
    )
    dynamic_method = getattr(model, "is_dynamic", None)
    try:
        model_dynamic = bool(dynamic_method()) if callable(dynamic_method) else None
    except Exception as error:
        model_dynamic = None
        _issue(warnings, "model_dynamic_state_unavailable", f"MODEL DynamicVRAM state could not be read: {error}")

    recipe = identity.get("recipe") if isinstance(identity, Mapping) else None
    if reference is not None and isinstance(recipe, Mapping):
        modalities = set(recipe.get("modalities", []))
        if reference.get("has_visual_references") and "video" not in modalities:
            _issue(warnings, "visual_reference_modality_not_patched", "Connected visual references are not covered by the selected Hybrid video rows.")
        if reference.get("has_audio_references") and "audio" not in modalities:
            _issue(warnings, "audio_reference_modality_not_patched", "Connected audio references are not covered by the selected Hybrid audio rows.")

    _issue(
        warnings,
        "hybrid_quality_remains_experimental",
        "Static AdaLN fusion affects target and reference rows together; loading successfully does not prove a quality winner or reference-only routing.",
    )
    compatible = not hard
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "compatible": compatible,
        "status": "incompatible" if hard else "conditional",
        "memory_safe_claim": False,
        "quality_validated": False,
        "model_passthrough": True,
        "hybrid": {
            "present": isinstance(hybrid_attachment, Mapping),
            "identity": identity,
            "recipe": recipe,
            "attachment_operation_count": hybrid_attachment.get("operation_count") if isinstance(hybrid_attachment, Mapping) else None,
        },
        "weight_patches": weight_patches,
        "components": {
            "sampling": sampling,
            "block_cache": block_cache,
            "sage_attention": sage,
            "long_video": scoped["long_video"],
            "multikeyframe": scoped["multikeyframe"],
            "dynamic_model_patcher": model_dynamic,
            "vram_policy": policy,
        },
        "conditioning": reference,
        "runtime_memory": runtime,
        "hard_errors": hard,
        "warnings": warnings,
    }


def hard_error_summary(report: Mapping[str, Any]) -> str:
    errors = report.get("hard_errors", [])
    return "; ".join(
        f"{item.get('code', 'unknown')}: {item.get('message', '')}"
        for item in errors
        if isinstance(item, Mapping)
    )
