from __future__ import annotations

import base64
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
import unicodedata
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch

import comfy.nested_tensor

from .native_latent_timeline_advanced import (
    RESUME_MANIFEST_SCHEMA,
    audit_native_h3_av_latent_resume_manifest,
)


CHECKPOINT_SCHEMA = "t8.minimax_h3.native_latent_checkpoint.v1"
CHECKPOINT_METADATA_KEY = "t8_native_latent_checkpoint_json"
CHECKPOINT_EXTENSION = ".h3latent.safetensors"
MAX_METADATA_JSON_BYTES = 4 * 1024 * 1024
_VOLATILE_CHECKPOINT_KEY = "t8_native_latent_checkpoint"
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized and not re.fullmatch(r"[0-9A-F]{64}", normalized):
        raise ValueError(f"{label} must be an empty value or a 64-character SHA-256")
    return normalized


def _validate_path_part(value: str, label: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"{label} contains an empty or traversal path component")
    if normalized != normalized.rstrip(" ."):
        raise ValueError(f"{label} path components cannot end with a space or dot")
    if re.search(r'[<>:"|?*\x00-\x1f]', normalized):
        raise ValueError(f"{label} contains characters unsupported by the checkpoint store")
    if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{label} uses a reserved Windows filename")
    return normalized


def _relative_parts(value: str, label: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or "").strip()).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or re.match(r"^[A-Za-z]:", text):
        raise ValueError(f"{label} must be a non-empty path relative to the checkpoint root")
    return tuple(_validate_path_part(part, label) for part in path.parts)


def _resolved_checkpoint_root(storage_root: str | Path, *, create: bool) -> Path:
    unresolved = Path(storage_root).expanduser()
    if unresolved.exists() and unresolved.is_symlink():
        raise ValueError("checkpoint storage root cannot be a symbolic link")
    root = unresolved.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint storage root does not exist: {root}")
    return root


def _reject_symlink_components(root: Path, relative_parts: tuple[str, ...]) -> None:
    current = root
    for part in relative_parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("checkpoint paths cannot traverse symbolic links")


def _new_checkpoint_path(storage_root: str | Path, filename_prefix: str) -> tuple[Path, str]:
    root = _resolved_checkpoint_root(storage_root, create=True)
    parts = _relative_parts(filename_prefix, "filename_prefix")
    _reject_symlink_components(root, parts[:-1])
    parent = root.joinpath(*parts[:-1])
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError("filename_prefix escaped the checkpoint storage root")
    stem = parts[-1]
    for _attempt in range(8):
        suffix = uuid.uuid4().hex[:12]
        target = resolved_parent / f"{stem}_{suffix}{CHECKPOINT_EXTENSION}"
        if not target.exists():
            relative = target.relative_to(root).as_posix()
            return target, relative
    raise RuntimeError("could not allocate a unique checkpoint filename")


def resolve_native_h3_checkpoint_path(
    storage_root: str | Path,
    checkpoint_path: str,
) -> tuple[Path, str]:
    root = _resolved_checkpoint_root(storage_root, create=False)
    parts = _relative_parts(checkpoint_path, "checkpoint_path")
    if not parts[-1].endswith(CHECKPOINT_EXTENSION):
        raise ValueError(f"checkpoint_path must end with {CHECKPOINT_EXTENSION}")
    _reject_symlink_components(root, parts)
    raw = root.joinpath(*parts)
    resolved = raw.resolve()
    if root not in resolved.parents or resolved.is_symlink():
        raise ValueError("checkpoint_path escaped the checkpoint storage root")
    if not resolved.is_file():
        raise FileNotFoundError(f"native H3 checkpoint does not exist: {resolved}")
    return resolved, resolved.relative_to(root).as_posix()


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.layout != torch.strided:
        raise ValueError(f"checkpoint cannot serialize non-strided tensor layout {value.layout}")
    return value.detach().to(device="cpu").contiguous().clone()


def _encode_metadata_value(
    value: Any,
    tensors: dict[str, torch.Tensor],
    *,
    path: str,
) -> dict[str, Any]:
    if torch.is_tensor(value):
        key = f"metadata_tensor_{len(tensors):05d}"
        tensor = _cpu_tensor(value)
        tensors[key] = tensor
        return {
            "type": "tensor",
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if getattr(value, "is_nested", False) and hasattr(value, "unbind"):
        return {
            "type": "nested_tensor",
            "items": [
                _encode_metadata_value(item, tensors, path=f"{path}[{index}]")
                for index, item in enumerate(value.unbind())
            ],
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"checkpoint metadata contains a non-finite float at {path}")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"checkpoint metadata requires string mapping keys at {path}")
        return {
            "type": "mapping",
            "items": [
                [
                    key,
                    _encode_metadata_value(value[key], tensors, path=f"{path}.{key}"),
                ]
                for key in sorted(value)
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _encode_metadata_value(item, tensors, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    raise ValueError(
        f"checkpoint cannot safely serialize metadata at {path}: "
        f"unsupported type {type(value).__name__}"
    )


def _decode_metadata_value(
    descriptor: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    used_tensor_keys: set[str],
    *,
    path: str,
) -> Any:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"checkpoint metadata descriptor is invalid at {path}")
    kind = descriptor.get("type")
    if kind == "tensor":
        key = descriptor.get("key")
        if not isinstance(key, str) or key not in tensors or key in used_tensor_keys:
            raise ValueError(f"checkpoint metadata tensor reference is invalid at {path}")
        tensor = tensors[key]
        if list(tensor.shape) != descriptor.get("shape") or str(tensor.dtype) != descriptor.get(
            "dtype"
        ):
            raise ValueError(f"checkpoint metadata tensor contract changed at {path}")
        used_tensor_keys.add(key)
        return tensor
    if kind == "nested_tensor":
        items = descriptor.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"checkpoint nested metadata is invalid at {path}")
        return comfy.nested_tensor.NestedTensor(
            tuple(
                _decode_metadata_value(
                    item,
                    tensors,
                    used_tensor_keys,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(items)
            )
        )
    if kind == "none":
        return None
    if kind == "bool":
        value = descriptor.get("value")
        if not isinstance(value, bool):
            raise ValueError(f"checkpoint bool metadata is invalid at {path}")
        return value
    if kind == "int":
        value = descriptor.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"checkpoint int metadata is invalid at {path}")
        return value
    if kind == "float":
        value = descriptor.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"checkpoint float metadata is invalid at {path}")
        return float(value)
    if kind == "str":
        value = descriptor.get("value")
        if not isinstance(value, str):
            raise ValueError(f"checkpoint string metadata is invalid at {path}")
        return value
    if kind == "bytes":
        value = descriptor.get("base64")
        if not isinstance(value, str):
            raise ValueError(f"checkpoint bytes metadata is invalid at {path}")
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError(f"checkpoint bytes metadata is invalid at {path}") from exc
    if kind == "mapping":
        items = descriptor.get("items")
        if not isinstance(items, list):
            raise ValueError(f"checkpoint mapping metadata is invalid at {path}")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError(f"checkpoint mapping entry is invalid at {path}")
            key = item[0]
            if key in result:
                raise ValueError(f"checkpoint mapping has duplicate key {key!r} at {path}")
            result[key] = _decode_metadata_value(
                item[1],
                tensors,
                used_tensor_keys,
                path=f"{path}.{key}",
            )
        return result
    if kind in {"list", "tuple"}:
        items = descriptor.get("items")
        if not isinstance(items, list):
            raise ValueError(f"checkpoint sequence metadata is invalid at {path}")
        result = [
            _decode_metadata_value(
                item,
                tensors,
                used_tensor_keys,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(items)
        ]
        return tuple(result) if kind == "tuple" else result
    raise ValueError(f"checkpoint metadata descriptor has unsupported type {kind!r} at {path}")


def _checkpoint_payload(
    av_latent: Mapping[str, Any],
    checkpoint_id: str,
    manifest_json: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    parts = tuple(av_latent["samples"].unbind())
    if len(parts) != 2:
        raise ValueError("native H3 checkpoint requires exactly video and audio sample streams")
    tensors = {
        "samples_video": _cpu_tensor(parts[0]),
        "samples_audio": _cpu_tensor(parts[1]),
    }
    masks = av_latent.get("noise_mask")
    if masks is not None:
        if not getattr(masks, "is_nested", False):
            raise ValueError("native H3 checkpoint requires a nested AV noise_mask")
        mask_parts = tuple(masks.unbind())
        if len(mask_parts) != 2:
            raise ValueError("native H3 checkpoint requires exactly two noise_mask streams")
        tensors["noise_mask_video"] = _cpu_tensor(mask_parts[0])
        tensors["noise_mask_audio"] = _cpu_tensor(mask_parts[1])

    metadata_source = {
        key: value
        for key, value in av_latent.items()
        if key not in {"samples", "noise_mask", _VOLATILE_CHECKPOINT_KEY}
    }
    descriptor = _encode_metadata_value(metadata_source, tensors, path="metadata")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "manifest": json.loads(manifest_json),
        "metadata_descriptor": descriptor,
        "has_noise_mask": masks is not None,
        "tensor_keys": sorted(tensors),
        "pickle_used": False,
    }
    encoded = _json(payload).encode("utf-8")
    if len(encoded) > MAX_METADATA_JSON_BYTES:
        raise ValueError(
            f"native H3 checkpoint metadata is {len(encoded)} bytes, above the "
            f"{MAX_METADATA_JSON_BYTES}-byte safety limit"
        )
    return tensors, payload


def _read_checkpoint_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        raw_payload = metadata.get(CHECKPOINT_METADATA_KEY)
        if not raw_payload or len(raw_payload.encode("utf-8")) > MAX_METADATA_JSON_BYTES:
            raise ValueError("native H3 checkpoint metadata is missing or oversized")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError("native H3 checkpoint metadata is invalid JSON") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"native H3 checkpoint must use schema {CHECKPOINT_SCHEMA}")
        if payload.get("pickle_used") is not False:
            raise ValueError("native H3 checkpoint must declare pickle_used=false")
        keys = set(handle.keys())
        declared = payload.get("tensor_keys")
        if not isinstance(declared, list) or keys != set(declared) or len(declared) != len(keys):
            raise ValueError("native H3 checkpoint tensor keys do not match its descriptor")
        # Own the CPU storage independently of the mmap/file handle. This makes a loaded
        # checkpoint a normal ComfyUI latent even if the source file is later archived.
        tensors = {key: _cpu_tensor(handle.get_tensor(key)) for key in keys}

    required = {"samples_video", "samples_audio"}
    if not required <= tensors.keys():
        raise ValueError("native H3 checkpoint is missing video or audio samples")
    has_mask = payload.get("has_noise_mask")
    mask_keys = {"noise_mask_video", "noise_mask_audio"}
    if not isinstance(has_mask, bool) or (mask_keys <= tensors.keys()) != has_mask:
        raise ValueError("native H3 checkpoint noise_mask declaration is inconsistent")
    if bool(mask_keys & tensors.keys()) != has_mask:
        raise ValueError("native H3 checkpoint contains only part of the AV noise_mask")

    used = set(required)
    latent: dict[str, Any] = _decode_metadata_value(
        payload.get("metadata_descriptor"),
        tensors,
        used,
        path="metadata",
    )
    if not isinstance(latent, dict) or "samples" in latent or "noise_mask" in latent:
        raise ValueError("native H3 checkpoint top-level metadata is invalid")
    latent["samples"] = comfy.nested_tensor.NestedTensor(
        (tensors["samples_video"], tensors["samples_audio"])
    )
    if has_mask:
        used.update(mask_keys)
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (tensors["noise_mask_video"], tensors["noise_mask_audio"])
        )
    if used != tensors.keys():
        raise ValueError(
            "native H3 checkpoint has unreferenced tensor keys: "
            + ", ".join(sorted(tensors.keys() - used))
        )
    return latent, dict(payload)


def save_native_h3_av_checkpoint(
    av_latent: Mapping[str, Any],
    storage_root: str | Path,
    filename_prefix: str = "h3_native_latent",
    checkpoint_id: str = "timeline_checkpoint",
    confirm_save: bool = False,
    verify_after_write: bool = True,
    hash_chunk_megabytes: int = 8,
) -> tuple[Mapping[str, Any], str, str, str, str, str]:
    status, _verified, content_sha256, manifest_json = (
        audit_native_h3_av_latent_resume_manifest(
            av_latent,
            checkpoint_id=checkpoint_id,
            hash_chunk_megabytes=hash_chunk_megabytes,
        )
    )
    if status != "BASELINE_CREATED":
        raise RuntimeError("native H3 checkpoint baseline creation returned an unexpected status")
    baseline_manifest = json.loads(manifest_json)
    checkpoint_id = baseline_manifest["checkpoint_id"]
    if not confirm_save:
        report = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "NOT_SAVED",
            "checkpoint_id": checkpoint_id,
            "content_sha256": content_sha256,
            "confirm_save": False,
            "files_written": False,
            "source_latent_mutated": False,
        }
        return av_latent, "NOT_SAVED", "", "", manifest_json, _json(report)

    target, relative = _new_checkpoint_path(storage_root, filename_prefix)
    tensors, payload = _checkpoint_payload(av_latent, checkpoint_id, manifest_json)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary), metadata={CHECKPOINT_METADATA_KEY: _json(payload)})
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        file_sha256 = _sha256_file(temporary)
        if verify_after_write:
            loaded, loaded_payload = _read_checkpoint_file(temporary)
            audit_native_h3_av_latent_resume_manifest(
                loaded,
                checkpoint_id=checkpoint_id,
                expected_manifest_json=manifest_json,
                mismatch_policy="error",
                hash_chunk_megabytes=hash_chunk_megabytes,
            )
            if loaded_payload.get("manifest") != baseline_manifest:
                raise ValueError("native H3 checkpoint embedded manifest changed during write")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite native H3 checkpoint: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    final_sha256 = _sha256_file(target)
    if final_sha256 != file_sha256:
        raise RuntimeError("native H3 checkpoint file hash changed during atomic placement")
    final_status = "SAVED_VERIFIED" if verify_after_write else "SAVED_UNVERIFIED"
    report = {
        "schema": CHECKPOINT_SCHEMA,
        "status": final_status,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": relative,
        "absolute_path": str(target),
        "file_sha256": final_sha256,
        "content_sha256": content_sha256,
        "file_bytes": target.stat().st_size,
        "tensor_count": len(tensors),
        "verify_after_write": bool(verify_after_write),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": False,
        "pickle_used": False,
        "files_written": True,
        "source_latent_mutated": False,
        "scientific_boundary": (
            "This persists a complete denoised/native H3 AV latent checkpoint, not the internal "
            "state of an interrupted diffusion step. A process crash before atomic placement may "
            "leave an ignored .tmp file but cannot expose it as a loadable checkpoint."
        ),
    }
    return av_latent, final_status, relative, final_sha256, manifest_json, _json(report)


def fingerprint_native_h3_checkpoint_file(
    storage_root: str | Path,
    checkpoint_path: str,
) -> str:
    """Return a content fingerprint without loading or deserializing checkpoint tensors."""
    path, _relative = resolve_native_h3_checkpoint_path(storage_root, checkpoint_path)
    return _sha256_file(path)


def load_native_h3_av_checkpoint(
    storage_root: str | Path,
    checkpoint_path: str,
    expected_manifest_json: str = "",
    expected_file_sha256: str = "",
    hash_chunk_megabytes: int = 8,
) -> tuple[dict[str, Any], str, bool, str, str, str, str, str]:
    path, relative = resolve_native_h3_checkpoint_path(storage_root, checkpoint_path)
    file_sha256 = _sha256_file(path)
    expected_file = _validate_sha256(expected_file_sha256, "expected_file_sha256")
    if expected_file and file_sha256 != expected_file:
        raise ValueError(
            "native H3 checkpoint file SHA-256 mismatch: "
            f"expected {expected_file}, actual {file_sha256}"
        )
    latent, payload = _read_checkpoint_file(path)
    checkpoint_id = payload.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("native H3 checkpoint checkpoint_id is missing")
    embedded_manifest = payload.get("manifest")
    if not isinstance(embedded_manifest, Mapping) or embedded_manifest.get("schema") != RESUME_MANIFEST_SCHEMA:
        raise ValueError("native H3 checkpoint embedded resume manifest is invalid")
    embedded_json = _json(embedded_manifest)
    _status, embedded_verified, content_sha256, _manifest = (
        audit_native_h3_av_latent_resume_manifest(
            latent,
            checkpoint_id=checkpoint_id,
            expected_manifest_json=embedded_json,
            mismatch_policy="error",
            hash_chunk_megabytes=hash_chunk_megabytes,
        )
    )
    if not embedded_verified:
        raise RuntimeError("native H3 checkpoint embedded content verification did not pass")

    external_text = str(expected_manifest_json or "").strip()
    external_verified = False
    authoritative_manifest = embedded_json
    if external_text:
        _status, external_verified, external_content_sha256, _manifest = (
            audit_native_h3_av_latent_resume_manifest(
                latent,
                checkpoint_id=checkpoint_id,
                expected_manifest_json=external_text,
                mismatch_policy="error",
                hash_chunk_megabytes=hash_chunk_megabytes,
            )
        )
        if not external_verified or external_content_sha256 != content_sha256:
            raise RuntimeError("native H3 checkpoint external manifest verification did not pass")
        authoritative_manifest = _json(json.loads(external_text))

    status = "MATCH_EXTERNAL" if external_verified else "SELF_VERIFIED"
    report = {
        "schema": CHECKPOINT_SCHEMA,
        "status": status,
        "resume_verified": True,
        "verification_source": "external_manifest" if external_verified else "embedded_manifest",
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": relative,
        "absolute_path": str(path),
        "file_sha256": file_sha256,
        "expected_file_sha256_supplied": bool(expected_file),
        "content_sha256": content_sha256,
        "embedded_manifest_verified": True,
        "external_manifest_verified": external_verified,
        "pickle_used": False,
        "loaded_device": "cpu",
        "sampling_executed": False,
        "vae_decode_executed": False,
        "scientific_boundary": (
            "This proves exact persisted latent identity across a completed save/load boundary. "
            "It does not resume an interrupted diffusion iteration or prove perceptual continuity."
        ),
    }
    latent[_VOLATILE_CHECKPOINT_KEY] = report
    return (
        latent,
        status,
        True,
        checkpoint_id,
        content_sha256,
        file_sha256,
        authoritative_manifest,
        _json(report),
    )
