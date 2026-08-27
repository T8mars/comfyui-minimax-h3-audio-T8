from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .vram_policy import apply_vram_policy


HYBRID_PLAN_TYPE = "H3_T8_HYBRID_PLAN"
HYBRID_ARTIFACT_TYPE = "H3_T8_HYBRID_ARTIFACT"
ARTIFACT_SCHEMA = "t8.minimax_h3.hybrid_artifact.v1"
PLAN_SCHEMA = "t8.minimax_h3.hybrid_plan.v1"
ALGORITHM = "curve_affine_rebase_fp64_target_slice_set_v1"
ATTACHMENT_KEY = "t8_minimax_h3_hybrid_recipe_v1"
VRAM_POLICY_ATTACHMENT_KEY = "t8_minimax_h3_vram_policy_apply_v1"
ARTIFACT_MAINTENANCE_SCHEMA = "t8.minimax_h3.hybrid_artifact_maintenance.v1"
ARTIFACT_MAINTENANCE_ACTIONS = (
    "inspect_only",
    "quarantine_artifact_exp",
    "restore_quarantined_exp",
    "quarantine_stale_build_residue_exp",
    "recover_interrupted_exp",
)

BLOCK_COUNT = 50
CURVE_SHAPE = (1025, 8)
HIDDEN_SIZE = 5376
EXPAND = 6
MODALITY_ROWS = EXPAND * HIDDEN_SIZE
BLOCK_OUTPUT_ROWS = MODALITY_ROWS * 3
FINAL_OUTPUT_ROWS = HIDDEN_SIZE * 2
MODALITY_ORDER = ("video", "text", "audio")
MODALITY_INDEX = {name: index for index, name in enumerate(MODALITY_ORDER)}
AUTO_PROFILE = "auto_match_reference_modalities_exp"
VISUAL_REFERENCE_KINDS = frozenset({"image", "video", "video_audio"})
AUDIO_REFERENCE_KINDS = frozenset({"audio", "video_audio"})
IGNORED_REFERENCE_KINDS = frozenset({"t8_keyframe_latent"})

KNOWN_QUALITY_BASE_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
KNOWN_REFERENCE_OVERLAY_SHA256 = (
    "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
)
KNOWN_QUALITY_CURVE_SHA256 = (
    "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e"
)
KNOWN_REFERENCE_CURVE_SHA256 = (
    "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
)

PROFILE_SPECS = {
    "blocks_25_49_video_audio_exp": {
        "blocks": (25, 49),
        "modalities": ("video", "audio"),
    },
    "blocks_25_49_all_modalities_exp": {
        "blocks": (25, 49),
        "modalities": MODALITY_ORDER,
    },
    "blocks_25_49_video_exp": {
        "blocks": (25, 49),
        "modalities": ("video",),
    },
    "blocks_25_49_audio_exp": {
        "blocks": (25, 49),
        "modalities": ("audio",),
    },
    "blocks_0_49_video_audio_exp": {
        "blocks": (0, 49),
        "modalities": ("video", "audio"),
    },
    "blocks_0_49_all_modalities_exp": {
        "blocks": (0, 49),
        "modalities": MODALITY_ORDER,
    },
}

_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_CACHE_LOCK = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, use_cache: bool = True) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    cache_key = (str(resolved).casefold(), int(stat.st_size), int(stat.st_mtime_ns))
    if use_cache:
        with _HASH_CACHE_LOCK:
            cached = _HASH_CACHE.get(cache_key)
        if cached is not None:
            return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    if use_cache:
        with _HASH_CACHE_LOCK:
            _HASH_CACHE[cache_key] = value
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return sha256_bytes(value.numpy().tobytes())


def _slice_descriptor(handle, key: str) -> dict[str, Any]:
    view = handle.get_slice(key)
    return {
        "shape": [int(value) for value in view.get_shape()],
        "dtype": str(view.get_dtype()),
    }


def _checkpoint_header(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        descriptors = {key: _slice_descriptor(handle, key) for key in keys}
        if "adaln_t_table" not in descriptors:
            raise ValueError("checkpoint is not an H3 pruned-curve model: adaln_t_table is missing")
        curve = handle.get_tensor("adaln_t_table")
        curve_hash = _tensor_sha256(curve)
        metadata = dict(handle.metadata() or {})
    signature = sha256_bytes(canonical_json(descriptors).encode("utf-8"))
    return {
        "file_name": path.name,
        "size_bytes": int(path.stat().st_size),
        "key_count": len(keys),
        "keys": keys,
        "descriptors": descriptors,
        "header_signature_sha256": signature,
        "curve_sha256": curve_hash,
        "metadata": metadata,
    }


def _validate_pruned_curve_header(header: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    descriptors = header["descriptors"]
    if header["key_count"] != 932:
        errors.append(f"{label}: expected 932 tensors for the validated pruned family, got {header['key_count']}")
    table = descriptors.get("adaln_t_table")
    if table != {"shape": list(CURVE_SHAPE), "dtype": "F32"}:
        errors.append(f"{label}: adaln_t_table must be F32 {CURVE_SHAPE}, got {table}")
    for block in range(BLOCK_COUNT):
        weight_key = f"blocks.{block}.adaln_proj.linear.weight"
        bias_key = f"blocks.{block}.adaln_proj.linear.bias"
        expected_weight = {"shape": [BLOCK_OUTPUT_ROWS, CURVE_SHAPE[1]], "dtype": "F16"}
        expected_bias = {"shape": [BLOCK_OUTPUT_ROWS], "dtype": "F16"}
        if descriptors.get(weight_key) != expected_weight:
            errors.append(f"{label}: invalid {weight_key}: {descriptors.get(weight_key)}")
        if descriptors.get(bias_key) != expected_bias:
            errors.append(f"{label}: invalid {bias_key}: {descriptors.get(bias_key)}")
    final_weight = descriptors.get("final_layer.adaln_proj.linear.weight")
    final_bias = descriptors.get("final_layer.adaln_proj.linear.bias")
    if final_weight != {"shape": [FINAL_OUTPUT_ROWS, CURVE_SHAPE[1]], "dtype": "F16"}:
        errors.append(f"{label}: invalid final AdaLN weight: {final_weight}")
    if final_bias != {"shape": [FINAL_OUTPUT_ROWS], "dtype": "F16"}:
        errors.append(f"{label}: invalid final AdaLN bias: {final_bias}")
    block_quant_keys = [
        key
        for key in descriptors
        if ".adaln_proj.linear." in key and (key.endswith(".comfy_quant") or key.endswith("_scale"))
    ]
    if block_quant_keys:
        errors.append(
            f"{label}: quantized AdaLN companions are not supported by P0: {block_quant_keys[:3]}"
        )
    return errors


def _checkpoint_role(header: dict[str, Any], file_sha256: str | None) -> str:
    curve_hash = header["curve_sha256"]
    if curve_hash == KNOWN_QUALITY_CURVE_SHA256:
        if file_sha256 is None or file_sha256 == KNOWN_QUALITY_BASE_SHA256:
            return "quality_base_fl2va_pruned_curve"
        return "quality_curve_unknown_file"
    if curve_hash == KNOWN_REFERENCE_CURVE_SHA256:
        if file_sha256 is None or file_sha256 == KNOWN_REFERENCE_OVERLAY_SHA256:
            return "reference_overlay_ref2va_pruned_curve"
        return "reference_curve_unknown_file"
    return "unknown"


def recipe_spec(profile: str) -> dict[str, Any]:
    try:
        value = PROFILE_SPECS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown H3 hybrid profile: {profile!r}") from exc
    return {
        "profile": profile,
        "block_start": int(value["blocks"][0]),
        "block_end": int(value["blocks"][1]),
        "modalities": list(value["modalities"]),
    }


def audit_conditioning_references(positive: Any) -> dict[str, Any]:
    """Describe the real MiniMax H3 reference rows without reading tensor payloads.

    The current core shares video/audio AdaLN tags between target and reference rows. This audit
    therefore selects only a mechanically relevant *modality subset*; it is never a quality or
    reference-strength recommendation.
    """
    if not isinstance(positive, Sequence) or isinstance(positive, (str, bytes)) or not positive:
        raise ValueError("positive must be a non-empty ComfyUI CONDITIONING value")

    kinds: set[str] = set()
    unknown_kinds: set[str] = set()
    keyframe_count = 0
    reference_count = 0
    for index, entry in enumerate(positive):
        if not isinstance(entry, Sequence) or len(entry) < 2 or not isinstance(entry[1], Mapping):
            raise ValueError(
                f"positive conditioning entry {index} does not contain a metadata mapping"
            )
        metadata = entry[1]
        refs = metadata.get("minimax_refs") or []
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError("minimax_refs must be a sequence when present")
        for ref in refs:
            if not isinstance(ref, Mapping):
                raise ValueError("every minimax_refs entry must be a mapping")
            kind = str(ref.get("kind", ""))
            if kind in IGNORED_REFERENCE_KINDS:
                continue
            reference_count += 1
            if kind in VISUAL_REFERENCE_KINDS or kind in AUDIO_REFERENCE_KINDS:
                kinds.add(kind)
            else:
                unknown_kinds.add(kind or "<missing>")

        keyframes = metadata.get("minimax_keyframes") or []
        if not isinstance(keyframes, Sequence) or isinstance(keyframes, (str, bytes)):
            raise ValueError("minimax_keyframes must be a sequence when present")
        keyframe_count = max(keyframe_count, len(keyframes))

    has_visual_references = bool(kinds & VISUAL_REFERENCE_KINDS)
    has_audio_references = bool(kinds & AUDIO_REFERENCE_KINDS)
    if unknown_kinds:
        resolved_profile = None
        status = "unknown_reference_kind"
    elif has_visual_references and has_audio_references:
        resolved_profile = "blocks_25_49_video_audio_exp"
        status = "reference_modalities_detected"
    elif has_visual_references:
        resolved_profile = "blocks_25_49_video_exp"
        status = "reference_modalities_detected"
    elif has_audio_references:
        resolved_profile = "blocks_25_49_audio_exp"
        status = "reference_modalities_detected"
    else:
        resolved_profile = None
        status = "no_extra_references"

    return {
        "status": status,
        "reference_count": reference_count,
        "reference_kinds": sorted(kinds),
        "unknown_reference_kinds": sorted(unknown_kinds),
        "has_visual_references": has_visual_references,
        "has_audio_references": has_audio_references,
        "keyframe_count": keyframe_count,
        "resolved_profile": resolved_profile,
        "scope": "extra image/video/video-audio/audio reference rows only",
        "quality_recommendation": False,
    }


def conditioning_reference_fingerprint(positive: Any) -> str:
    audit = audit_conditioning_references(positive)
    payload = {
        key: audit[key]
        for key in (
            "reference_count",
            "reference_kinds",
            "unknown_reference_kinds",
            "has_visual_references",
            "has_audio_references",
            "keyframe_count",
            "resolved_profile",
        )
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def selected_slice_bytes(profile: str) -> int:
    spec = recipe_spec(profile)
    block_count = spec["block_end"] - spec["block_start"] + 1
    rows = block_count * len(spec["modalities"]) * MODALITY_ROWS
    fp16_bytes = torch.empty((), dtype=torch.float16).element_size()
    return int(rows * (CURVE_SHAPE[1] + 1) * fp16_bytes)


def inspect_checkpoint_pair(
    base_path: str | os.PathLike[str],
    overlay_path: str | os.PathLike[str],
    profile: str,
    verification: str = "full_sha256",
    positive: Any = None,
) -> dict[str, Any]:
    base = Path(base_path).resolve()
    overlay = Path(overlay_path).resolve()
    errors: list[str] = []
    warnings: list[str] = [
        "Experimental checkpoint fusion: static modality rows also affect target streams; this is not reference-only routing.",
        "No profile is a proven quality winner until controlled video/audio A/B and blind review pass.",
    ]
    requested_profile = profile
    reference_audit = None
    if profile == AUTO_PROFILE:
        try:
            reference_audit = audit_conditioning_references(positive)
            profile = reference_audit["resolved_profile"]
            if reference_audit["unknown_reference_kinds"]:
                errors.append(
                    "auto modality matching found unsupported reference kind(s): "
                    + ", ".join(reference_audit["unknown_reference_kinds"])
                )
            elif profile is None:
                errors.append(
                    "auto modality matching requires extra image/video/audio references; "
                    "first/last keyframes alone must use the stock FL2VA base"
                )
            else:
                warnings.append(
                    f"Auto modality matching resolved to {profile}; this minimizes patched "
                    "modality rows but does not select a proven quality winner."
                )
        except (TypeError, ValueError) as exc:
            errors.append(f"auto modality matching failed: {exc}")
            profile = None
    elif positive is not None:
        try:
            reference_audit = audit_conditioning_references(positive)
        except (TypeError, ValueError) as exc:
            errors.append(f"connected conditioning audit failed: {exc}")

    resolved_profile = profile if isinstance(profile, str) else "blocks_25_49_video_audio_exp"
    if verification not in {"full_sha256", "header_only_exp"}:
        raise ValueError(f"unsupported verification mode: {verification!r}")
    try:
        if base == overlay:
            warnings.append("quality base and reference overlay point to the same file")
        base_header = _checkpoint_header(base)
        overlay_header = _checkpoint_header(overlay)
        warnings.extend(_validate_pruned_curve_header(base_header, "quality_base"))
        warnings.extend(_validate_pruned_curve_header(overlay_header, "reference_overlay"))
        if base_header["descriptors"] != overlay_header["descriptors"]:
            warnings.append(
                "base and overlay tensor key/shape/dtype contracts are not identical; "
                "artifact construction may fail"
            )
        base_sha = sha256_file(base) if verification == "full_sha256" else None
        overlay_sha = sha256_file(overlay) if verification == "full_sha256" else None
        base_role = _checkpoint_role(base_header, base_sha)
        overlay_role = _checkpoint_role(overlay_header, overlay_sha)
        if verification != "full_sha256":
            warnings.append(
                "header_only_exp skips full-file identity reporting; model identity is not a load gate"
            )
        else:
            if base_sha != KNOWN_QUALITY_BASE_SHA256:
                warnings.append(
                    "quality base SHA-256 differs from the reference FL2VA file; continuing with the user-selected model"
                )
            if overlay_sha != KNOWN_REFERENCE_OVERLAY_SHA256:
                warnings.append(
                    "reference overlay SHA-256 differs from the reference Ref2VA file; continuing with the user-selected model"
                )
        if base_role != "quality_base_fl2va_pruned_curve":
            warnings.append(f"quality base reference-role diagnostic: {base_role}")
        if overlay_role != "reference_overlay_ref2va_pruned_curve":
            warnings.append(f"reference overlay reference-role diagnostic: {overlay_role}")
        spec = recipe_spec(resolved_profile)
        source = {
            "base_path": str(base),
            "overlay_path": str(overlay),
            "base_file_name": base.name,
            "overlay_file_name": overlay.name,
            "base_sha256": base_sha,
            "overlay_sha256": overlay_sha,
            "base_size_bytes": base_header["size_bytes"],
            "overlay_size_bytes": overlay_header["size_bytes"],
            "base_curve_sha256": base_header["curve_sha256"],
            "overlay_curve_sha256": overlay_header["curve_sha256"],
            "header_signature_sha256": base_header["header_signature_sha256"],
            "model_identity_policy": "diagnostic_only_not_a_build_gate",
        }
    except Exception as exc:
        errors.append(f"checkpoint inspection failed: {type(exc).__name__}: {exc}")
        spec = recipe_spec(resolved_profile)
        source = {
            "base_path": str(base),
            "overlay_path": str(overlay),
            "base_file_name": base.name,
            "overlay_file_name": overlay.name,
        }
    compatible = not errors
    identity_payload = {
        "schema": PLAN_SCHEMA,
        "algorithm": ALGORITHM,
        "source": {key: value for key, value in source.items() if not key.endswith("_path")},
        "recipe": spec,
    }
    plan_fingerprint = sha256_bytes(canonical_json(identity_payload).encode("utf-8"))
    return {
        "schema": PLAN_SCHEMA,
        "algorithm": ALGORITHM,
        "compatible": compatible,
        "verification": verification,
        "requested_profile": requested_profile,
        "reference_audit": reference_audit,
        "plan_fingerprint": plan_fingerprint,
        "recipe": spec,
        "selected_slice_bytes": selected_slice_bytes(resolved_profile),
        "source": source,
        "errors": errors,
        "warnings": warnings,
    }


def pair_report_text(plan: dict[str, Any]) -> str:
    status = "COMPATIBLE / 可构建" if plan.get("compatible") else "REJECTED / 已拒绝"
    recipe = plan.get("recipe", {})
    lines = [
        f"Status: {status}",
        f"Requested profile: {plan.get('requested_profile', recipe.get('profile', 'unknown'))}",
        f"Profile: {recipe.get('profile', 'unknown')}",
        f"Blocks: {recipe.get('block_start', '?')}..{recipe.get('block_end', '?')}",
        f"Modalities: {', '.join(recipe.get('modalities', [])) or 'none'}",
        f"Artifact payload: {plan.get('selected_slice_bytes', 0) / 1024 / 1024:.2f} MiB",
    ]
    lines.extend(f"ERROR: {value}" for value in plan.get("errors", []))
    lines.extend(f"WARNING: {value}" for value in plan.get("warnings", []))
    audit = plan.get("reference_audit")
    if isinstance(audit, dict):
        lines.append(
            "Reference audit: "
            f"kinds={','.join(audit.get('reference_kinds', [])) or 'none'}, "
            f"keyframes={audit.get('keyframe_count', 0)}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class CurveRebase:
    base_table: torch.Tensor
    overlay_table: torch.Tensor
    matrix: torch.Tensor
    offset: torch.Tensor
    rank: int
    condition_number: float
    table_relative_error: float
    table_max_abs_error: float


def fit_curve_rebase(base_table: torch.Tensor, overlay_table: torch.Tensor) -> CurveRebase:
    base = base_table.detach().cpu().to(torch.float64)
    overlay = overlay_table.detach().cpu().to(torch.float64)
    if tuple(base.shape) != tuple(overlay.shape):
        raise ValueError(f"curve table shape mismatch: {tuple(base.shape)} != {tuple(overlay.shape)}")
    if base.ndim != 2:
        raise ValueError(f"curve tables must be rank 2, got rank {base.ndim}")
    ones = torch.ones((base.shape[0], 1), dtype=torch.float64)
    augmented = torch.cat([base, ones], dim=1)
    rank = int(torch.linalg.matrix_rank(augmented).item())
    expected_rank = int(augmented.shape[1])
    if rank != expected_rank:
        raise ValueError(f"base curve affine basis is rank deficient: {rank} != {expected_rank}")
    condition_number = float(torch.linalg.cond(augmented).item())
    if not torch.isfinite(torch.tensor(condition_number)) or condition_number > 1.0e8:
        raise ValueError(f"base curve affine basis is ill-conditioned: {condition_number:.6g}")
    solution = torch.linalg.lstsq(augmented, overlay).solution
    matrix = solution[:-1, :]
    offset = solution[-1, :]
    fitted = base @ matrix + offset
    residual = fitted - overlay
    relative = float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(overlay))
    maximum = float(residual.abs().max().item())
    if not torch.isfinite(fitted).all():
        raise ValueError("curve rebase produced NaN or Inf")
    if relative > 1.0e-4:
        raise ValueError(f"curve rebase relative error exceeds 1e-4: {relative:.8g}")
    return CurveRebase(
        base_table=base,
        overlay_table=overlay,
        matrix=matrix,
        offset=offset,
        rank=rank,
        condition_number=condition_number,
        table_relative_error=relative,
        table_max_abs_error=maximum,
    )


def rebase_adaln_slice(
    overlay_weight: torch.Tensor,
    overlay_bias: torch.Tensor,
    rebase: CurveRebase,
    row_start: int,
    row_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if overlay_weight.ndim != 2 or overlay_bias.ndim != 1:
        raise ValueError("AdaLN weight/bias ranks must be 2/1")
    if overlay_weight.shape[0] != overlay_bias.shape[0]:
        raise ValueError("AdaLN weight and bias output rows do not match")
    if overlay_weight.shape[1] != rebase.matrix.shape[0]:
        raise ValueError("AdaLN input width does not match curve basis")
    stop = int(row_start) + int(row_length)
    if row_start < 0 or row_length <= 0 or stop > overlay_weight.shape[0]:
        raise ValueError(f"invalid AdaLN row slice: start={row_start} length={row_length}")
    source_weight = overlay_weight[row_start:stop].detach().cpu().to(torch.float64)
    source_bias = overlay_bias[row_start:stop].detach().cpu().to(torch.float64)
    target_weight_f64 = source_weight @ rebase.matrix.T
    target_bias_f64 = source_bias + source_weight @ rebase.offset
    target_weight = target_weight_f64.to(torch.float16).contiguous()
    target_bias = target_bias_f64.to(torch.float16).contiguous()
    if not torch.isfinite(target_weight).all() or not torch.isfinite(target_bias).all():
        raise ValueError("FP16 target slice contains NaN or Inf")

    base_aug = torch.cat(
        [rebase.base_table, torch.ones((rebase.base_table.shape[0], 1), dtype=torch.float64)],
        dim=1,
    )
    overlay_aug = torch.cat(
        [rebase.overlay_table, torch.ones((rebase.overlay_table.shape[0], 1), dtype=torch.float64)],
        dim=1,
    )
    candidate = torch.cat([target_weight.to(torch.float64), target_bias[:, None].to(torch.float64)], dim=1)
    native = torch.cat([source_weight, source_bias[:, None]], dim=1)
    base_gram = base_aug.T @ base_aug
    overlay_gram = overlay_aug.T @ overlay_aug
    cross_gram = base_aug.T @ overlay_aug
    candidate_energy = ((candidate @ base_gram) * candidate).sum()
    native_energy = ((native @ overlay_gram) * native).sum()
    cross_energy = ((candidate @ cross_gram) * native).sum()
    error_sq = torch.clamp(candidate_energy + native_energy - 2.0 * cross_energy, min=0.0)
    relative_error = float(torch.sqrt(error_sq / native_energy.clamp(min=1.0e-30)).item())
    return target_weight, target_bias, {
        "effective_function_relative_rms": relative_error,
        "target_weight_max_abs": float(target_weight.abs().max().item()),
        "target_bias_max_abs": float(target_bias.abs().max().item()),
    }


def _artifact_identity(plan: dict[str, Any]) -> dict[str, Any]:
    source = plan["source"]
    return {
        "schema": ARTIFACT_SCHEMA,
        "algorithm": ALGORITHM,
        "base_file_name": source["base_file_name"],
        "overlay_file_name": source["overlay_file_name"],
        "base_sha256": source["base_sha256"],
        "overlay_sha256": source["overlay_sha256"],
        "base_curve_sha256": source["base_curve_sha256"],
        "overlay_curve_sha256": source["overlay_curve_sha256"],
        "header_signature_sha256": source["header_signature_sha256"],
        "recipe": plan["recipe"],
    }


@contextmanager
def _artifact_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    maintenance_lock_path = path.with_suffix(path.suffix + ".maintenance.lock")
    if maintenance_lock_path.exists():
        raise RuntimeError(f"hybrid artifact maintenance is active: {maintenance_lock_path}")
    payload = canonical_json({"pid": os.getpid(), "created_unix": time.time(), "target": path.name})
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age > 3600.0:
            raise RuntimeError(
                f"stale hybrid artifact lock is older than one hour; inspect and remove it explicitly: {lock_path}"
            ) from exc
        raise RuntimeError(f"hybrid artifact is already being built: {lock_path}") from exc
    try:
        if maintenance_lock_path.exists():
            os.close(descriptor)
            descriptor = -1
            lock_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"hybrid artifact maintenance began before the build lock settled: {maintenance_lock_path}"
            )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _load_sidecar(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("hybrid artifact sidecar must contain a JSON object")
    return value


def _expected_operations(identity: dict[str, Any]) -> list[dict[str, Any]]:
    recipe = identity.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("hybrid artifact identity recipe is missing")
    canonical_recipe = recipe_spec(str(recipe.get("profile", "")))
    if recipe != canonical_recipe:
        raise ValueError("hybrid artifact recipe differs from its canonical profile")
    operations: list[dict[str, Any]] = []
    operation_index = 0
    for block in range(recipe["block_start"], recipe["block_end"] + 1):
        for modality in recipe["modalities"]:
            row_start = MODALITY_INDEX[modality] * MODALITY_ROWS
            for parameter, shape in (
                ("weight", [MODALITY_ROWS, CURVE_SHAPE[1]]),
                ("bias", [MODALITY_ROWS]),
            ):
                source_key = f"blocks.{block}.adaln_proj.linear.{parameter}"
                operations.append(
                    {
                        "artifact_key": f"patch_{operation_index:04d}_{parameter}",
                        "model_key": f"diffusion_model.{source_key}",
                        "source_key": source_key,
                        "block": block,
                        "modality": modality,
                        "parameter": parameter,
                        "offset": [0, row_start, MODALITY_ROWS],
                        "shape": shape,
                        "dtype": "F16",
                        "operation": "set",
                    }
                )
            operation_index += 1
    return operations


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("hybrid artifact manifest schema is unsupported")
    if manifest.get("algorithm") != ALGORITHM:
        raise ValueError("hybrid artifact algorithm is unsupported")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("hybrid artifact identity is missing")
    expected_identity_fields = {
        "schema": ARTIFACT_SCHEMA,
        "algorithm": ALGORITHM,
    }
    for key, expected in expected_identity_fields.items():
        if identity.get(key) != expected:
            raise ValueError(f"hybrid artifact identity {key} is unsupported")
    expected_fingerprint = sha256_bytes(canonical_json(identity).encode("utf-8"))
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("hybrid artifact identity fingerprint mismatch")
    if manifest.get("storage") != "fp16_target_slices_with_offset_set":
        raise ValueError("hybrid artifact storage mode is unsupported")
    expected_operations = _expected_operations(identity)
    if manifest.get("operations") != expected_operations:
        raise ValueError("hybrid artifact operations differ from the canonical recipe")
    expected_bytes = sum(
        int(torch.tensor(operation["shape"]).prod().item()) * 2
        for operation in expected_operations
    )
    if manifest.get("payload_bytes") != expected_bytes:
        raise ValueError("hybrid artifact payload byte count is inconsistent")
    curve_fit = manifest.get("curve_fit")
    if not isinstance(curve_fit, dict):
        raise ValueError("hybrid artifact curve-fit report is missing")
    for key in ("table_relative_error", "maximum_effective_function_relative_rms"):
        value = curve_fit.get(key)
        if not isinstance(value, (int, float)) or not float(value) < 1.0e-4:
            raise ValueError(f"hybrid artifact curve-fit metric {key} failed the 1e-4 gate")


def _validate_existing_artifact(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = path.with_suffix(path.suffix + ".json")
    artifact_exists = path.is_file()
    sidecar_exists = sidecar_path.is_file()
    if not artifact_exists and not sidecar_exists:
        raise FileNotFoundError("artifact or sidecar is missing")
    if artifact_exists != sidecar_exists:
        raise ValueError(
            "incomplete hybrid artifact publication detected; inspect and remove the orphan "
            f"explicitly instead of overwriting it: {path}"
        )
    sidecar = _load_sidecar(sidecar_path)
    artifact_sha = sha256_file(path, use_cache=False)
    if sidecar.get("artifact_sha256") != artifact_sha:
        raise ValueError("artifact SHA-256 does not match its sidecar")
    manifest = sidecar.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("artifact sidecar manifest is missing")
    _validate_manifest_contract(manifest)
    if manifest.get("identity") != expected_identity:
        raise ValueError("existing artifact identity does not match the requested plan")
    with safe_open(path, framework="pt", device="cpu") as handle:
        embedded = (handle.metadata() or {}).get("h3_t8_manifest")
        if embedded is None or json.loads(embedded) != manifest:
            raise ValueError("embedded artifact manifest does not match the sidecar")
        keys = set(handle.keys())
    expected_keys = {operation["artifact_key"] for operation in manifest.get("operations", [])}
    if keys != expected_keys:
        raise ValueError("artifact tensor keys do not match its operation manifest")
    return {
        "schema": HYBRID_ARTIFACT_TYPE,
        "path": str(path),
        "sidecar_path": str(sidecar_path),
        "artifact_sha256": artifact_sha,
        "manifest": manifest,
        "cache_hit": True,
    }


def build_hybrid_artifact(
    plan: dict[str, Any],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("hybrid_plan is not a MiniMax H3 T8 plan")
    if not plan.get("compatible"):
        raise ValueError("hybrid plan is not compatible: " + "; ".join(plan.get("errors", [])))
    if plan.get("verification") != "full_sha256":
        raise ValueError("artifact building requires full SHA-256 verification")
    source = plan["source"]
    base_path = Path(source["base_path"]).resolve()
    overlay_path = Path(source["overlay_path"]).resolve()
    if sha256_file(base_path) != source["base_sha256"]:
        raise ValueError("quality base changed after pair inspection")
    if sha256_file(overlay_path) != source["overlay_sha256"]:
        raise ValueError("reference overlay changed after pair inspection")

    identity = _artifact_identity(plan)
    fingerprint = sha256_bytes(canonical_json(identity).encode("utf-8"))
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_path_for_plan(plan, root)
    try:
        return _validate_existing_artifact(artifact_path, identity)
    except FileNotFoundError:
        pass

    with _artifact_lock(artifact_path):
        try:
            return _validate_existing_artifact(artifact_path, identity)
        except FileNotFoundError:
            pass

        tensors: dict[str, torch.Tensor] = {}
        operations: list[dict[str, Any]] = []
        slice_metrics: list[dict[str, Any]] = []
        with safe_open(base_path, framework="pt", device="cpu") as base_handle, safe_open(
            overlay_path, framework="pt", device="cpu"
        ) as overlay_handle:
            base_curve = base_handle.get_tensor("adaln_t_table")
            overlay_curve = overlay_handle.get_tensor("adaln_t_table")
            if _tensor_sha256(base_curve) != source["base_curve_sha256"]:
                raise ValueError("quality base curve hash changed after inspection")
            if _tensor_sha256(overlay_curve) != source["overlay_curve_sha256"]:
                raise ValueError("reference overlay curve hash changed after inspection")
            rebase = fit_curve_rebase(base_curve, overlay_curve)
            recipe = plan["recipe"]
            operation_index = 0
            for block in range(recipe["block_start"], recipe["block_end"] + 1):
                weight_source_key = f"blocks.{block}.adaln_proj.linear.weight"
                bias_source_key = f"blocks.{block}.adaln_proj.linear.bias"
                overlay_weight = overlay_handle.get_tensor(weight_source_key)
                overlay_bias = overlay_handle.get_tensor(bias_source_key)
                for modality in recipe["modalities"]:
                    row_start = MODALITY_INDEX[modality] * MODALITY_ROWS
                    weight, bias, metrics = rebase_adaln_slice(
                        overlay_weight,
                        overlay_bias,
                        rebase,
                        row_start,
                        MODALITY_ROWS,
                    )
                    weight_artifact_key = f"patch_{operation_index:04d}_weight"
                    bias_artifact_key = f"patch_{operation_index:04d}_bias"
                    tensors[weight_artifact_key] = weight
                    tensors[bias_artifact_key] = bias
                    for artifact_key, source_key, value in (
                        (weight_artifact_key, weight_source_key, weight),
                        (bias_artifact_key, bias_source_key, bias),
                    ):
                        operations.append(
                            {
                                "artifact_key": artifact_key,
                                "model_key": f"diffusion_model.{source_key}",
                                "source_key": source_key,
                                "block": block,
                                "modality": modality,
                                "parameter": "weight" if value.ndim == 2 else "bias",
                                "offset": [0, row_start, MODALITY_ROWS],
                                "shape": [int(item) for item in value.shape],
                                "dtype": "F16",
                                "operation": "set",
                            }
                        )
                    slice_metrics.append({"block": block, "modality": modality, **metrics})
                    operation_index += 1
                del overlay_weight, overlay_bias

        maximum_function_error = max(
            item["effective_function_relative_rms"] for item in slice_metrics
        )
        if maximum_function_error > 1.0e-4:
            raise ValueError(
                f"curve-aware AdaLN reconstruction exceeds 1e-4: {maximum_function_error:.8g}"
            )
        manifest = {
            "schema": ARTIFACT_SCHEMA,
            "algorithm": ALGORITHM,
            "identity": identity,
            "fingerprint": fingerprint,
            "storage": "fp16_target_slices_with_offset_set",
            "curve_fit": {
                "rank": rebase.rank,
                "condition_number": rebase.condition_number,
                "table_relative_error": rebase.table_relative_error,
                "table_max_abs_error": rebase.table_max_abs_error,
                "maximum_effective_function_relative_rms": maximum_function_error,
            },
            "operations": operations,
            "slice_metrics": slice_metrics,
            "payload_bytes": int(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())),
            "limitations": [
                "experimental_static_modality_overlay_not_reference_only",
                "no_proven_quality_winner",
                "validated_pruned_curve_pair_only",
                "final_adaln_and_output_heads_remain_on_quality_base",
            ],
        }
        temp_path = artifact_path.with_name(
            artifact_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
        )
        sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
        try:
            save_file(tensors, str(temp_path), metadata={"h3_t8_manifest": canonical_json(manifest)})
            # Windows rejects fsync on a read-only descriptor; r+b does not
            # mutate the completed safetensors file and makes the durability
            # barrier portable.
            with temp_path.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_path, artifact_path)
            artifact_sha = sha256_file(artifact_path, use_cache=False)
            sidecar = {"artifact_sha256": artifact_sha, "manifest": manifest}
            _write_json_atomic(sidecar_path, sidecar)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        return {
            "schema": HYBRID_ARTIFACT_TYPE,
            "path": str(artifact_path),
            "sidecar_path": str(sidecar_path),
            "artifact_sha256": artifact_sha,
            "manifest": manifest,
            "cache_hit": False,
        }


def validate_artifact_descriptor(artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, dict) or artifact.get("schema") != HYBRID_ARTIFACT_TYPE:
        raise ValueError("artifact input is not an H3 T8 hybrid artifact descriptor")
    path = Path(artifact.get("path", "")).resolve()
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = _load_sidecar(sidecar_path)
    manifest = sidecar.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("hybrid artifact manifest is missing")
    _validate_manifest_contract(manifest)
    artifact_sha = sha256_file(path)
    if artifact_sha != sidecar.get("artifact_sha256"):
        raise ValueError("hybrid artifact SHA-256 mismatch")
    if artifact.get("artifact_sha256") not in {None, artifact_sha}:
        raise ValueError("connected hybrid artifact descriptor is stale")
    with safe_open(path, framework="pt", device="cpu") as handle:
        embedded_raw = (handle.metadata() or {}).get("h3_t8_manifest")
        embedded = json.loads(embedded_raw) if embedded_raw else None
        if embedded != manifest:
            raise ValueError("embedded hybrid artifact manifest differs from the sidecar")
        actual_keys = set(handle.keys())
        for operation in manifest.get("operations", []):
            key = operation["artifact_key"]
            if key not in actual_keys:
                raise ValueError(f"hybrid artifact tensor is missing: {key}")
            descriptor = _slice_descriptor(handle, key)
            if descriptor != {"shape": operation["shape"], "dtype": operation["dtype"]}:
                raise ValueError(f"hybrid artifact tensor contract mismatch for {key}: {descriptor}")
        expected_keys = {operation["artifact_key"] for operation in manifest.get("operations", [])}
        if actual_keys != expected_keys:
            raise ValueError("hybrid artifact contains tensors outside its operation manifest")
    return {
        "schema": HYBRID_ARTIFACT_TYPE,
        "path": str(path),
        "sidecar_path": str(sidecar_path),
        "artifact_sha256": artifact_sha,
        "manifest": manifest,
        "cache_hit": bool(artifact.get("cache_hit", False)),
    }


def _maintenance_plan_context(
    plan: dict[str, Any],
    output_root: str | os.PathLike[str],
    operation_epoch: int,
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("hybrid_plan is not a MiniMax H3 T8 plan")
    if not plan.get("compatible"):
        raise ValueError("hybrid plan is not compatible: " + "; ".join(plan.get("errors", [])))
    if plan.get("verification") != "full_sha256":
        raise ValueError("hybrid artifact maintenance requires full SHA-256 verification")
    epoch = int(operation_epoch)
    if epoch < 0:
        raise ValueError("operation_epoch must be non-negative")
    identity = _artifact_identity(plan)
    fingerprint = sha256_bytes(canonical_json(identity).encode("utf-8"))
    root = Path(output_root).resolve()
    raw_artifact_path = artifact_path_for_plan(plan, root)
    raw_sidecar_path = raw_artifact_path.with_suffix(raw_artifact_path.suffix + ".json")
    if raw_artifact_path.exists() and raw_artifact_path.is_symlink():
        raise ValueError("hybrid artifact maintenance refuses an artifact symbolic link")
    if raw_sidecar_path.exists() and raw_sidecar_path.is_symlink():
        raise ValueError("hybrid artifact maintenance refuses a sidecar symbolic link")
    artifact_path = raw_artifact_path.resolve()
    if artifact_path.parent != root:
        raise ValueError("hybrid artifact path escaped the standard artifact directory")
    if artifact_path.suffix != ".safetensors" or not artifact_path.name.startswith("h3_t8_"):
        raise ValueError("hybrid artifact path does not match the content-addressed naming contract")
    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    operation_key = sha256_bytes(
        canonical_json(
            {
                "schema": ARTIFACT_MAINTENANCE_SCHEMA,
                "fingerprint": fingerprint,
                "operation_epoch": epoch,
            }
        ).encode("utf-8")
    )[:24]
    raw_transaction_root = root / "_maintenance_transactions"
    raw_recycle_root = root / "_recycle"
    raw_recycle_fingerprint_root = raw_recycle_root / fingerprint[:16]
    raw_recycle_dir = raw_recycle_fingerprint_root / operation_key
    for label, internal in (
        ("transaction", raw_transaction_root),
        ("recycle", raw_recycle_root),
        ("recycle fingerprint", raw_recycle_fingerprint_root),
        ("recycle transaction", raw_recycle_dir),
    ):
        if internal.exists() and internal.is_symlink():
            raise ValueError(
                f"hybrid artifact {label} directory may not be a symbolic link"
            )
    transaction_root = raw_transaction_root.resolve()
    recycle_root = raw_recycle_root.resolve()
    recycle_dir = raw_recycle_dir.resolve()
    for label, internal in (
        ("transaction", transaction_root),
        ("recycle", recycle_root),
        ("recycle transaction", recycle_dir),
    ):
        try:
            internal.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"hybrid artifact {label} path escaped its standard root") from exc
    return {
        "root": root,
        "artifact_path": artifact_path,
        "sidecar_path": sidecar_path,
        "build_lock_path": artifact_path.with_suffix(artifact_path.suffix + ".lock"),
        "maintenance_lock_path": artifact_path.with_suffix(
            artifact_path.suffix + ".maintenance.lock"
        ),
        "identity": identity,
        "fingerprint": fingerprint,
        "operation_epoch": epoch,
        "operation_key": operation_key,
        "transaction_root": transaction_root,
        "transaction_path": transaction_root / f"{operation_key}.json",
        "recycle_root": recycle_root,
        "recycle_dir": recycle_dir,
    }


def _assert_safe_internal_directory(path: Path, root: Path, *, create: bool) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"hybrid artifact internal directory may not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"hybrid artifact internal path escaped its standard root: {path}") from exc


def _assert_safe_maintenance_file(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise ValueError(f"hybrid artifact maintenance source escaped its standard root: {path}")
    if path.exists() and path.is_symlink():
        raise ValueError(f"hybrid artifact maintenance refuses symbolic links: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"hybrid artifact maintenance source is not a regular file: {path}")


def _file_fact(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_unix": float(stat.st_mtime),
        "age_seconds": max(0.0, time.time() - float(stat.st_mtime)),
        "is_file": path.is_file(),
        "is_symlink": path.is_symlink(),
    }


def _load_maintenance_journal(context: dict[str, Any]) -> dict[str, Any] | None:
    path = context["transaction_path"]
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "hybrid artifact maintenance journal must be a regular non-symlink file"
        )
    value = _load_sidecar(path)
    if value.get("schema") != ARTIFACT_MAINTENANCE_SCHEMA:
        raise ValueError("hybrid artifact maintenance journal schema is unsupported")
    for key in ("fingerprint", "operation_epoch", "operation_key"):
        if value.get(key) != context[key]:
            raise ValueError(f"hybrid artifact maintenance journal {key} mismatch")
    action = value.get("action")
    if action not in {
        "quarantine_artifact_exp",
        "quarantine_stale_build_residue_exp",
    }:
        raise ValueError("hybrid artifact maintenance journal action is unsupported")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("hybrid artifact maintenance journal has no items")
    allowed_phases = {
        "prepared",
        "moving_to_recycle",
        "quarantined",
        "restoring_to_active",
        "restored",
        "recovered_active",
    }
    if value.get("phase") not in allowed_phases:
        raise ValueError("hybrid artifact maintenance journal phase is unsupported")
    moved_count = value.get("moved_count")
    if not isinstance(moved_count, int) or not 0 <= moved_count <= len(items):
        raise ValueError("hybrid artifact maintenance journal moved_count is invalid")
    phase = value["phase"]
    phase_count_is_valid = (
        (phase == "prepared" and moved_count == 0)
        or (phase in {"moving_to_recycle", "restoring_to_active"} and 1 <= moved_count <= len(items))
        or (
            phase in {"quarantined", "restored", "recovered_active"}
            and moved_count == len(items)
        )
    )
    if not phase_count_is_valid:
        raise ValueError("hybrid artifact maintenance journal phase/count mismatch")
    allowed_exact = {
        context["artifact_path"].name,
        context["sidecar_path"].name,
        context["build_lock_path"].name,
    }
    allowed_prefixes = (
        context["artifact_path"].name + ".tmp-",
        context["sidecar_path"].name + ".tmp-",
    )
    seen_sources: set[str] = set()
    seen_recycle: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("hybrid artifact maintenance journal item is not an object")
        source = Path(str(item.get("source", ""))).resolve()
        recycle = Path(str(item.get("recycle", ""))).resolve()
        if source.parent != context["root"]:
            raise ValueError("hybrid artifact maintenance journal source escaped the artifact root")
        if source.name not in allowed_exact and not source.name.startswith(allowed_prefixes):
            raise ValueError("hybrid artifact maintenance journal source name is outside the contract")
        if recycle.parent != context["recycle_dir"] or recycle.name != source.name:
            raise ValueError("hybrid artifact maintenance recycle path is outside the transaction")
        if str(source).casefold() in seen_sources or str(recycle).casefold() in seen_recycle:
            raise ValueError("hybrid artifact maintenance journal contains duplicate paths")
        seen_sources.add(str(source).casefold())
        seen_recycle.add(str(recycle).casefold())
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("hybrid artifact maintenance journal item SHA-256 is invalid")
        if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
            raise ValueError("hybrid artifact maintenance journal item size is invalid")
    if action == "quarantine_artifact_exp":
        expected_sources = {
            str(context["artifact_path"]).casefold(),
            str(context["sidecar_path"]).casefold(),
        }
        if seen_sources != expected_sources:
            raise ValueError(
                "hybrid artifact-pair quarantine journal must contain exactly the "
                "artifact and sidecar"
            )
    return value


def _write_maintenance_journal(context: dict[str, Any], journal: dict[str, Any]) -> None:
    _assert_safe_internal_directory(
        context["transaction_root"], context["root"], create=True
    )
    _write_json_atomic(context["transaction_path"], journal)


def _journal_item(path: Path, recycle_dir: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"hybrid artifact maintenance requires a regular source file: {path}")
    return {
        "source": str(path.resolve()),
        "recycle": str((recycle_dir / path.name).resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path, use_cache=False),
    }


def _verify_journal_item(item: dict[str, Any], *, at_recycle: bool) -> None:
    path = Path(item["recycle"] if at_recycle else item["source"])
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"hybrid artifact maintenance transaction file is missing or unsafe: {path}")
    if int(path.stat().st_size) != item["size_bytes"]:
        raise ValueError(f"hybrid artifact maintenance transaction size mismatch: {path}")
    if sha256_file(path, use_cache=False) != item["sha256"]:
        raise ValueError(f"hybrid artifact maintenance transaction SHA-256 mismatch: {path}")


def _pid_is_running(pid: Any) -> bool | None:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = open_process(process_query_limited_information, False, value)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no process with this PID.
                return False
            if error == 5:  # ERROR_ACCESS_DENIED: a live protected process may own it.
                return True
            return None
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return None
            return int(exit_code.value) == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _lock_owner_state(path: Path) -> dict[str, Any]:
    fact = _file_fact(path)
    if not fact["exists"]:
        return {**fact, "owner_pid": None, "owner_running": None}
    if fact["is_symlink"] or not fact["is_file"]:
        return {
            **fact,
            "owner_pid": None,
            "owner_running": None,
            "payload_error": "maintenance lock must be a regular non-symlink file",
        }
    try:
        value = _load_sidecar(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            **fact,
            "owner_pid": None,
            "owner_running": None,
            "payload_error": f"{type(exc).__name__}: {exc}",
        }
    pid = value.get("pid")
    return {**fact, "owner_pid": pid, "owner_running": _pid_is_running(pid)}


@contextmanager
def _maintenance_lock(
    context: dict[str, Any],
    *,
    allow_stale_build_lock: bool,
    stale_after_seconds: float,
    recover_stale_maintenance_lock: bool,
):
    lock_path = context["maintenance_lock_path"]
    archived_stale_lock: str | None = None
    payload = canonical_json(
        {
            "pid": os.getpid(),
            "created_unix": time.time(),
            "operation_key": context["operation_key"],
        }
    )
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        state = _lock_owner_state(lock_path)
        recoverable = (
            recover_stale_maintenance_lock
            and not state.get("is_symlink", False)
            and state.get("age_seconds", 0.0) >= stale_after_seconds
            and state.get("owner_running") is not True
        )
        if not recoverable:
            raise RuntimeError(f"hybrid artifact maintenance is already active: {lock_path}") from exc
        _assert_safe_internal_directory(
            context["transaction_root"], context["root"], create=True
        )
        stale_path = context["transaction_path"].with_suffix(".stale-maintenance-lock")
        if stale_path.exists():
            raise RuntimeError(f"stale maintenance-lock archive already exists: {stale_path}") from exc
        os.replace(lock_path, stale_path)
        archived_stale_lock = str(stale_path)
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        build_lock_state = _lock_owner_state(context["build_lock_path"])
        if build_lock_state["exists"]:
            allowed = (
                allow_stale_build_lock
                and not build_lock_state.get("is_symlink", False)
                and build_lock_state.get("age_seconds", 0.0) >= stale_after_seconds
                and build_lock_state.get("owner_running") is not True
            )
            if not allowed:
                raise RuntimeError(
                    f"hybrid artifact build lock is active or not proven stale: {context['build_lock_path']}"
                )
        yield archived_stale_lock
    finally:
        lock_path.unlink(missing_ok=True)


def _new_maintenance_journal(
    context: dict[str, Any], action: str, paths: Sequence[Path]
) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_MAINTENANCE_SCHEMA,
        "fingerprint": context["fingerprint"],
        "operation_epoch": context["operation_epoch"],
        "operation_key": context["operation_key"],
        "action": action,
        "created_unix": time.time(),
        "updated_unix": time.time(),
        "phase": "prepared",
        "moved_count": 0,
        "items": [_journal_item(path, context["recycle_dir"]) for path in paths],
    }


def _move_journal_to_recycle(
    context: dict[str, Any], journal: dict[str, Any]
) -> None:
    _assert_safe_internal_directory(context["recycle_dir"], context["root"], create=True)
    for index, item in enumerate(journal["items"]):
        source = Path(item["source"])
        recycle = Path(item["recycle"])
        if source.exists() and recycle.exists():
            raise ValueError("hybrid artifact maintenance found both active and recycled copies")
        if source.exists():
            _verify_journal_item(item, at_recycle=False)
            os.replace(source, recycle)
        elif not recycle.exists():
            raise ValueError("hybrid artifact maintenance transaction lost a source file")
        _verify_journal_item(item, at_recycle=True)
        journal["phase"] = "moving_to_recycle"
        journal["moved_count"] = index + 1
        journal["updated_unix"] = time.time()
        _write_maintenance_journal(context, journal)
    journal["phase"] = "quarantined"
    journal["updated_unix"] = time.time()
    _write_maintenance_journal(context, journal)


def _move_journal_to_active(
    context: dict[str, Any], journal: dict[str, Any], *, terminal_phase: str
) -> None:
    for index, item in enumerate(journal["items"]):
        source = Path(item["source"])
        recycle = Path(item["recycle"])
        if source.exists() and recycle.exists():
            raise ValueError("hybrid artifact maintenance found both active and recycled copies")
        if recycle.exists():
            _verify_journal_item(item, at_recycle=True)
            os.replace(recycle, source)
        elif not source.exists():
            raise ValueError("hybrid artifact maintenance transaction lost both file copies")
        _verify_journal_item(item, at_recycle=False)
        journal["phase"] = "restoring_to_active"
        journal["moved_count"] = index + 1
        journal["updated_unix"] = time.time()
        _write_maintenance_journal(context, journal)
    journal["phase"] = terminal_phase
    journal["updated_unix"] = time.time()
    _write_maintenance_journal(context, journal)
    try:
        context["recycle_dir"].rmdir()
    except OSError:
        pass


def _active_artifact_validation(context: dict[str, Any]) -> dict[str, Any]:
    descriptor = validate_artifact_descriptor(
        {
            "schema": HYBRID_ARTIFACT_TYPE,
            "path": str(context["artifact_path"]),
        }
    )
    if descriptor["manifest"].get("identity") != context["identity"]:
        raise ValueError("active hybrid artifact identity does not match the exact plan")
    return descriptor


def inspect_hybrid_artifact_maintenance(
    plan: dict[str, Any],
    output_root: str | os.PathLike[str],
    operation_epoch: int = 0,
) -> dict[str, Any]:
    context = _maintenance_plan_context(plan, output_root, operation_epoch)
    artifact_exists = context["artifact_path"].is_file()
    sidecar_exists = context["sidecar_path"].is_file()
    validation: dict[str, Any]
    if artifact_exists and sidecar_exists:
        try:
            descriptor = _active_artifact_validation(context)
            validation = {
                "state": "valid_active_pair",
                "artifact_sha256": descriptor["artifact_sha256"],
                "payload_bytes": descriptor["manifest"]["payload_bytes"],
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            validation = {
                "state": "invalid_active_pair",
                "error": f"{type(exc).__name__}: {exc}",
            }
    elif artifact_exists or sidecar_exists:
        validation = {"state": "incomplete_active_pair"}
    else:
        validation = {"state": "missing_active_pair"}
    temp_paths = sorted(
        {
            *context["root"].glob(context["artifact_path"].name + ".tmp-*"),
            *context["root"].glob(context["sidecar_path"].name + ".tmp-*"),
        },
        key=lambda value: value.name,
    ) if context["root"].is_dir() else []
    try:
        journal = _load_maintenance_journal(context)
        journal_report = None if journal is None else {
            "action": journal["action"],
            "phase": journal["phase"],
            "moved_count": journal["moved_count"],
            "item_count": len(journal["items"]),
            "path": str(context["transaction_path"]),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        journal_report = {"error": f"{type(exc).__name__}: {exc}"}
    build_lock = _lock_owner_state(context["build_lock_path"])
    maintenance_lock = _lock_owner_state(context["maintenance_lock_path"])
    return {
        "schema": ARTIFACT_MAINTENANCE_SCHEMA,
        "mode": "inspect_only",
        "mutation_performed": False,
        "memory_safe_claim": False,
        "artifact_path": str(context["artifact_path"]),
        "sidecar_path": str(context["sidecar_path"]),
        "fingerprint": context["fingerprint"],
        "operation_epoch": context["operation_epoch"],
        "operation_key": context["operation_key"],
        "validation": validation,
        "build_lock": build_lock,
        "maintenance_lock": maintenance_lock,
        "temporary_files": [_file_fact(path) for path in temp_paths],
        "transaction": journal_report,
        "warnings": [
            "Inspection is side-effect free; quarantine/restore/recovery require explicit confirmation and a positive epoch.",
            "Quarantine is recoverable file movement, not secure erasure and not a model-cache unload.",
        ],
    }


def _stale_build_residue_paths(
    context: dict[str, Any], stale_after_seconds: float
) -> list[Path]:
    candidates: list[Path] = []
    artifact_exists = context["artifact_path"].is_file()
    sidecar_exists = context["sidecar_path"].is_file()
    if artifact_exists != sidecar_exists:
        candidates.append(context["artifact_path"] if artifact_exists else context["sidecar_path"])
    if context["build_lock_path"].is_file():
        candidates.append(context["build_lock_path"])
    if context["root"].is_dir():
        candidates.extend(context["root"].glob(context["artifact_path"].name + ".tmp-*"))
        candidates.extend(context["root"].glob(context["sidecar_path"].name + ".tmp-*"))
    unique = sorted({path.resolve() for path in candidates}, key=lambda value: value.name)
    if not unique:
        raise ValueError("no incomplete hybrid artifact build residue exists for this exact plan")
    for path in unique:
        _assert_safe_maintenance_file(path, context["root"])
        age = time.time() - path.stat().st_mtime
        if age < stale_after_seconds:
            raise ValueError(
                f"hybrid artifact build residue is not stale enough ({age:.1f}s < {stale_after_seconds:.1f}s): {path}"
            )
    lock_state = _lock_owner_state(context["build_lock_path"])
    if lock_state["exists"] and lock_state.get("owner_running") is True:
        raise ValueError("hybrid artifact build-lock owner is still running")
    return unique


def maintain_hybrid_artifact(
    plan: dict[str, Any],
    output_root: str | os.PathLike[str],
    action: str,
    confirm_action: bool,
    operation_epoch: int,
    stale_after_minutes: float = 60.0,
) -> dict[str, Any]:
    if action not in ARTIFACT_MAINTENANCE_ACTIONS:
        raise ValueError(f"unsupported hybrid artifact maintenance action: {action!r}")
    context = _maintenance_plan_context(plan, output_root, operation_epoch)
    if action == "inspect_only":
        return inspect_hybrid_artifact_maintenance(plan, output_root, operation_epoch)
    if not bool(confirm_action):
        raise ValueError("mutating hybrid artifact maintenance requires confirm_action=true")
    if context["operation_epoch"] <= 0:
        raise ValueError("mutating hybrid artifact maintenance requires operation_epoch > 0")
    stale_after_seconds = max(60.0, float(stale_after_minutes) * 60.0)
    recover_lock = action in {
        "quarantine_stale_build_residue_exp",
        "recover_interrupted_exp",
    }
    allow_build_lock = action == "quarantine_stale_build_residue_exp"
    with _maintenance_lock(
        context,
        allow_stale_build_lock=allow_build_lock,
        stale_after_seconds=stale_after_seconds,
        recover_stale_maintenance_lock=recover_lock,
    ) as archived_stale_lock:
        journal = _load_maintenance_journal(context)
        if action == "quarantine_artifact_exp":
            if journal is not None:
                if journal["action"] != action:
                    raise ValueError("operation_epoch was already used by another maintenance action")
                if journal["phase"] == "quarantined":
                    for item in journal["items"]:
                        _verify_journal_item(item, at_recycle=True)
                    performed = False
                elif journal["phase"] in {"restored", "recovered_active"}:
                    raise ValueError("operation_epoch is already complete; increment it for a new quarantine")
                else:
                    raise ValueError("maintenance transaction is incomplete; run recover_interrupted_exp")
            else:
                descriptor = _active_artifact_validation(context)
                journal = _new_maintenance_journal(
                    context,
                    action,
                    [context["artifact_path"], context["sidecar_path"]],
                )
                if journal["items"][0]["sha256"] != descriptor["artifact_sha256"]:
                    raise ValueError("active artifact changed between validation and transaction preparation")
                _write_maintenance_journal(context, journal)
                _move_journal_to_recycle(context, journal)
                performed = True
        elif action == "quarantine_stale_build_residue_exp":
            if journal is not None:
                if journal["action"] != action:
                    raise ValueError("operation_epoch was already used by another maintenance action")
                if journal["phase"] == "quarantined":
                    for item in journal["items"]:
                        _verify_journal_item(item, at_recycle=True)
                    performed = False
                elif journal["phase"] == "recovered_active":
                    raise ValueError("operation_epoch is already recovered; increment it to quarantine again")
                else:
                    raise ValueError("maintenance transaction is incomplete; run recover_interrupted_exp")
            else:
                paths = _stale_build_residue_paths(context, stale_after_seconds)
                journal = _new_maintenance_journal(context, action, paths)
                _write_maintenance_journal(context, journal)
                _move_journal_to_recycle(context, journal)
                performed = True
        elif action == "restore_quarantined_exp":
            if journal is None or journal.get("action") != "quarantine_artifact_exp":
                raise ValueError("no exact quarantined artifact transaction exists for this epoch")
            if journal["phase"] == "restored":
                for item in journal["items"]:
                    _verify_journal_item(item, at_recycle=False)
                performed = False
            elif journal["phase"] != "quarantined":
                raise ValueError("maintenance transaction is incomplete; run recover_interrupted_exp")
            else:
                if context["artifact_path"].exists() or context["sidecar_path"].exists():
                    raise ValueError("active artifact path is occupied; refusing to overwrite during restore")
                _move_journal_to_active(context, journal, terminal_phase="restored")
                _active_artifact_validation(context)
                performed = True
        else:
            if journal is None:
                raise ValueError("no exact maintenance transaction exists for recovery")
            if journal["phase"] in {"quarantined", "restored", "recovered_active"}:
                raise ValueError("maintenance transaction is already terminal and needs no recovery")
            _move_journal_to_active(context, journal, terminal_phase="recovered_active")
            if journal["action"] == "quarantine_artifact_exp":
                _active_artifact_validation(context)
            performed = True

    after = inspect_hybrid_artifact_maintenance(plan, output_root, operation_epoch)
    return {
        **after,
        "mode": action,
        "mutation_performed": performed,
        "archived_stale_maintenance_lock": archived_stale_lock,
        "transaction": {
            "action": journal["action"],
            "phase": journal["phase"],
            "moved_count": journal["moved_count"],
            "item_count": len(journal["items"]),
            "path": str(context["transaction_path"]),
        },
    }


def artifact_maintenance_fingerprint(
    plan: dict[str, Any],
    output_root: str | os.PathLike[str],
    action: str,
    operation_epoch: int,
) -> str:
    try:
        context = _maintenance_plan_context(plan, output_root, operation_epoch)
        paths = [
            context["artifact_path"],
            context["sidecar_path"],
            context["build_lock_path"],
            context["maintenance_lock_path"],
            context["transaction_path"],
        ]
        if context["root"].is_dir():
            paths.extend(context["root"].glob(context["artifact_path"].name + ".tmp-*"))
            paths.extend(context["root"].glob(context["sidecar_path"].name + ".tmp-*"))
        return sha256_bytes(
            canonical_json(
                {
                    "action": action,
                    "operation_epoch": int(operation_epoch),
                    "plan_fingerprint": plan.get("plan_fingerprint"),
                    "files": file_stat_fingerprint(paths),
                }
            ).encode("utf-8")
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return f"unresolved:{type(exc).__name__}:{exc}"


def _weight_model_options(weight_dtype: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if weight_dtype == "default":
        return options
    if weight_dtype == "fp8_e4m3fn":
        options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        options["dtype"] = torch.float8_e4m3fn
        options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    else:
        raise ValueError(f"unsupported weight_dtype: {weight_dtype!r}")
    return options


def _load_artifact_tensors(path: Path) -> dict[str, torch.Tensor]:
    import comfy.utils

    value = comfy.utils.load_torch_file(str(path))
    if not isinstance(value, dict):
        raise ValueError("ComfyUI did not return a state dictionary for the hybrid artifact")
    return value


def apply_artifact_to_model(model, artifact: dict[str, Any]):
    descriptor = validate_artifact_descriptor(artifact)
    manifest = descriptor["manifest"]
    tensor_dict = _load_artifact_tensors(Path(descriptor["path"]))
    patched = model.clone()
    patches: dict[Any, Any] = {}
    existing_keys = set(getattr(patched, "patches", {}))
    for operation in manifest["operations"]:
        if operation.get("operation") != "set":
            raise ValueError(f"unsupported hybrid artifact operation: {operation.get('operation')!r}")
        model_key = operation["model_key"]
        if model_key in existing_keys:
            raise ValueError(f"hybrid artifact conflicts with an existing whole-tensor patch: {model_key}")
        offset = tuple(int(value) for value in operation["offset"])
        patch_key = (model_key, offset)
        if patch_key in existing_keys or patch_key in patches:
            raise ValueError(f"hybrid artifact has a duplicate/conflicting slice: {patch_key}")
        target = tensor_dict[operation["artifact_key"]]
        patches[patch_key] = ("set", (target,))
    accepted = patched.add_patches(patches, strength_patch=1.0, strength_model=1.0)
    if set(accepted) != set(patches):
        missing = sorted(str(value) for value in set(patches) - set(accepted))
        raise ValueError(
            "current ComfyUI MODEL does not expose all expected H3 AdaLN keys: " + ", ".join(missing[:5])
        )
    attachment = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_sha256": descriptor["artifact_sha256"],
        "fingerprint": manifest["fingerprint"],
        "identity": manifest["identity"],
        "operation_count": len(manifest["operations"]),
        "payload_bytes": manifest["payload_bytes"],
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(ATTACHMENT_KEY, attachment)
    else:
        patched.attachments[ATTACHMENT_KEY] = attachment
    return patched, attachment


def _attach_vram_policy_provenance(model: Any, policy_report: dict[str, Any] | None) -> bool:
    """Attach only the immutable application facts needed by downstream audits."""
    if policy_report is None:
        return False
    keys = (
        "schema",
        "policy_fingerprint",
        "mode",
        "applied",
        "cleanup_performed",
        "target_reserved_gib",
        "dynamic_vram_route",
        "model_management_route",
        "current_gate_pass",
        "commit_gate_pass",
        "memory_safe_claim",
    )
    attachment = {key: policy_report.get(key) for key in keys if key in policy_report}
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter(VRAM_POLICY_ATTACHMENT_KEY, attachment)
        return True
    attachments = getattr(model, "attachments", None)
    if isinstance(attachments, dict):
        attachments[VRAM_POLICY_ATTACHMENT_KEY] = attachment
        return True
    return False


def load_hybrid_model(
    base_path: str | os.PathLike[str],
    mode: str,
    weight_dtype: str,
    artifact: dict[str, Any] | None = None,
    vram_policy: dict[str, Any] | None = None,
):
    started = time.perf_counter()
    base = Path(base_path).resolve()
    model_options = _weight_model_options(weight_dtype)
    policy_report = (
        None if vram_policy is None else apply_vram_policy(vram_policy)
    )
    import comfy.sd

    if mode == "base_only":
        model = comfy.sd.load_diffusion_model(str(base), model_options=model_options)
        policy_attachment_written = _attach_vram_policy_provenance(model, policy_report)
        report = {
            "mode": mode,
            "base_file_name": base.name,
            "weight_dtype": weight_dtype,
            "artifact_applied": False,
            "load_seconds": time.perf_counter() - started,
            "warnings": ["Base-only control: no reference-capability fusion was applied."],
        }
        if policy_report is not None:
            report["vram_policy"] = policy_report
            report["vram_policy_attachment_written"] = policy_attachment_written
        return model, report
    if mode != "apply_artifact_exp":
        raise ValueError(f"unsupported hybrid loader mode: {mode!r}")
    if artifact is None:
        raise ValueError("apply_artifact_exp requires a connected hybrid artifact")
    descriptor = validate_artifact_descriptor(artifact)
    manifest = descriptor["manifest"]
    identity = manifest["identity"]
    base_sha = sha256_file(base)
    base_identity_match = base_sha == identity["base_sha256"]
    if base.name != identity["base_file_name"]:
        name_warning = (
            f"Selected base name {base.name!r} differs from manifest name "
            f"{identity['base_file_name']!r}."
        )
    else:
        name_warning = None
    model = comfy.sd.load_diffusion_model(str(base), model_options=model_options)
    patched, attachment = apply_artifact_to_model(model, descriptor)
    policy_attachment_written = _attach_vram_policy_provenance(patched, policy_report)
    warnings = [
        "Experimental static AdaLN fusion; it is not reference-only routing and has no proven quality-winning preset.",
        "Apply Turbo LoRA only after this node; AdaLN-overlapping patches require an explicit compatibility check.",
    ]
    if name_warning:
        warnings.append(name_warning)
    if not base_identity_match:
        warnings.append(
            "Selected base SHA-256 differs from the artifact build manifest; "
            "continuing with the user-selected model without an identity gate."
        )
    report = {
        "mode": mode,
        "base_file_name": base.name,
        "base_sha256": base_sha,
        "artifact_base_sha256_match": base_identity_match,
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
        "weight_dtype": weight_dtype,
        "artifact_applied": True,
        "artifact_path": descriptor["path"],
        "artifact_sha256": descriptor["artifact_sha256"],
        "recipe": identity["recipe"],
        "attachment": attachment,
        "load_seconds": time.perf_counter() - started,
        "warnings": warnings,
    }
    if policy_report is not None:
        report["vram_policy"] = policy_report
        report["vram_policy_attachment_written"] = policy_attachment_written
    return patched, report


def artifact_output_root(models_dir: str | os.PathLike[str]) -> Path:
    return Path(models_dir).resolve() / "h3_hybrid_artifacts"


def artifact_path_for_plan(
    plan: dict[str, Any], output_root: str | os.PathLike[str]
) -> Path:
    identity = _artifact_identity(plan)
    fingerprint = sha256_bytes(canonical_json(identity).encode("utf-8"))
    profile = plan["recipe"]["profile"]
    return Path(output_root).resolve() / f"h3_t8_{profile}_{fingerprint[:16]}.safetensors"


def file_stat_fingerprint(paths: Iterable[str | os.PathLike[str]]) -> str:
    values = []
    for path in paths:
        resolved = Path(path).resolve()
        try:
            stat = resolved.stat()
            values.append((str(resolved).casefold(), int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError as exc:
            values.append((str(resolved).casefold(), type(exc).__name__, str(exc)))
    return sha256_bytes(canonical_json(values).encode("utf-8"))
