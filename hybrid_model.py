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
            errors.append("quality base and reference overlay must be different files")
        base_header = _checkpoint_header(base)
        overlay_header = _checkpoint_header(overlay)
        errors.extend(_validate_pruned_curve_header(base_header, "quality_base"))
        errors.extend(_validate_pruned_curve_header(overlay_header, "reference_overlay"))
        if base_header["descriptors"] != overlay_header["descriptors"]:
            errors.append("base and overlay tensor key/shape/dtype contracts are not identical")
        base_sha = sha256_file(base) if verification == "full_sha256" else None
        overlay_sha = sha256_file(overlay) if verification == "full_sha256" else None
        base_role = _checkpoint_role(base_header, base_sha)
        overlay_role = _checkpoint_role(overlay_header, overlay_sha)
        if verification != "full_sha256":
            errors.append("header_only_exp is diagnostic only; artifact construction requires full SHA-256 verification")
        else:
            if base_sha != KNOWN_QUALITY_BASE_SHA256:
                errors.append("quality base SHA-256 is outside the exact P0 validated FL2VA pruned pair")
            if overlay_sha != KNOWN_REFERENCE_OVERLAY_SHA256:
                errors.append("reference overlay SHA-256 is outside the exact P0 validated Ref2VA pruned pair")
        if base_role != "quality_base_fl2va_pruned_curve":
            errors.append(f"quality base role is not validated FL2VA pruned curve: {base_role}")
        if overlay_role != "reference_overlay_ref2va_pruned_curve":
            errors.append(f"reference overlay role is not validated Ref2VA pruned curve: {overlay_role}")
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
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
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
        "base_sha256": KNOWN_QUALITY_BASE_SHA256,
        "overlay_sha256": KNOWN_REFERENCE_OVERLAY_SHA256,
        "base_curve_sha256": KNOWN_QUALITY_CURVE_SHA256,
        "overlay_curve_sha256": KNOWN_REFERENCE_CURVE_SHA256,
    }
    for key, expected in expected_identity_fields.items():
        if identity.get(key) != expected:
            raise ValueError(f"hybrid artifact identity {key} is outside the P0 contract")
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
        return model, report
    if mode != "apply_artifact_exp":
        raise ValueError(f"unsupported hybrid loader mode: {mode!r}")
    if artifact is None:
        raise ValueError("apply_artifact_exp requires a connected hybrid artifact")
    descriptor = validate_artifact_descriptor(artifact)
    manifest = descriptor["manifest"]
    identity = manifest["identity"]
    base_sha = sha256_file(base)
    if base_sha != identity["base_sha256"]:
        raise ValueError("selected quality base does not match the artifact base SHA-256")
    if base.name != identity["base_file_name"]:
        # A renamed but bit-identical file is safe; retain a transparent warning.
        name_warning = (
            f"Selected base name {base.name!r} differs from manifest name "
            f"{identity['base_file_name']!r}, but SHA-256 matches."
        )
    else:
        name_warning = None
    model = comfy.sd.load_diffusion_model(str(base), model_options=model_options)
    patched, attachment = apply_artifact_to_model(model, descriptor)
    warnings = [
        "Experimental static AdaLN fusion; it is not reference-only routing and has no proven quality-winning preset.",
        "Apply Turbo LoRA only after this node; AdaLN-overlapping patches require an explicit compatibility check.",
    ]
    if name_warning:
        warnings.append(name_warning)
    report = {
        "mode": mode,
        "base_file_name": base.name,
        "base_sha256": base_sha,
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
