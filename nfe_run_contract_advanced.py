from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import torch

from .native_latent_checkpoint_advanced import MAX_METADATA_JSON_BYTES


NFE_RUN_CONTRACT_SCHEMA = "t8.minimax_h3.nfe_run_contract.v1"
MAX_PROMPT_UTF8_BYTES = 400_000
MAX_MEDIA_MAP_UTF8_BYTES = 65_536
MAX_REPORT_UTF8_BYTES = 65_536


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _required_text(value: Any, name: str, maximum_bytes: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    encoded = value.encode("utf-8")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{name} exceeds the {maximum_bytes}-byte UTF-8 safety limit")
    return value


def _parse_media_map(value: Any) -> tuple[dict[str, Any], str]:
    text = _required_text(value, "media_map_json", MAX_MEDIA_MAP_UTF8_BYTES)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "media_map_json is invalid JSON at "
            f"line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("media_map_json root must be a JSON object")
    for key in ("pictures", "videos", "audios"):
        if key not in parsed or not isinstance(parsed[key], dict):
            raise ValueError(f"media_map_json.{key} must be a JSON object")
    source_ordinal = parsed.get("source_audio_ordinal")
    if source_ordinal is not None and (
        isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int)
    ):
        raise ValueError("media_map_json.source_audio_ordinal must be an integer or null")
    canonical = _canonical_json(parsed)
    return parsed, canonical


def _feed(hasher: Any, tag: str, payload: bytes = b"") -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(len(tag_bytes).to_bytes(4, "big"))
    hasher.update(tag_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _tensor_bytes_digest(tensor: torch.Tensor, chunk_bytes: int) -> tuple[str, int]:
    if tensor.is_sparse or tensor.is_quantized or tensor.is_meta:
        raise ValueError(
            "positive conditioning contains a sparse, quantized or meta tensor that cannot "
            "be content-hashed safely"
        )
    if getattr(tensor, "is_nested", False):
        raise ValueError("positive conditioning contains an unsupported nested tensor")
    try:
        raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"positive conditioning tensor dtype {tensor.dtype} cannot be content-hashed"
        ) from error

    hasher = hashlib.sha256()
    total_bytes = int(raw.numel())
    for start in range(0, total_bytes, chunk_bytes):
        chunk = raw[start : start + chunk_bytes]
        if chunk.device.type != "cpu":
            chunk = chunk.to(device="cpu", non_blocking=False)
        hasher.update(chunk.contiguous().numpy().tobytes())
    return hasher.hexdigest().upper(), total_bytes


class _ConditioningDigester:
    def __init__(self, chunk_bytes: int):
        self.chunk_bytes = chunk_bytes
        self.tensor_manifest: list[dict[str, Any]] = []
        self.total_tensor_bytes = 0
        self._active_containers: set[int] = set()

    def _enter(self, value: Any, path: str) -> int:
        identity = id(value)
        if identity in self._active_containers:
            raise ValueError(f"positive conditioning contains a cycle at {path}")
        self._active_containers.add(identity)
        return identity

    def _digest(self, value: Any, hasher: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            content_sha256, byte_count = _tensor_bytes_digest(value, self.chunk_bytes)
            shape = [int(dimension) for dimension in value.shape]
            metadata = _canonical_json(
                {
                    "dtype": str(value.dtype),
                    "shape": shape,
                    "content_sha256": content_sha256,
                    "byte_count": byte_count,
                }
            ).encode("utf-8")
            _feed(hasher, "tensor", metadata)
            self.tensor_manifest.append(
                {
                    "path": path,
                    "shape": shape,
                    "dtype": str(value.dtype),
                    "byte_count": byte_count,
                    "content_sha256": content_sha256,
                }
            )
            self.total_tensor_bytes += byte_count
            return

        if value is None:
            _feed(hasher, "none")
            return
        if isinstance(value, bool):
            _feed(hasher, "bool", b"true" if value else b"false")
            return
        if isinstance(value, int):
            _feed(hasher, "int", str(value).encode("ascii"))
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"positive conditioning contains a non-finite float at {path}")
            _feed(hasher, "float", value.hex().encode("ascii"))
            return
        if isinstance(value, str):
            _feed(hasher, "string", value.encode("utf-8"))
            return
        if isinstance(value, bytes):
            _feed(hasher, "bytes", value)
            return

        value_type = type(value)
        if (
            value_type.__module__ == "comfy.conds"
            and value_type.__qualname__ == "CONDConstant"
            and hasattr(value, "cond")
        ):
            # Prompt Relay authenticates MODEL/CONDITIONING ownership with a
            # scalar CONDConstant(binding_hash).  Bind its actual payload into
            # the contract instead of hashing object identity or accepting
            # arbitrary runtime objects.
            _feed(hasher, "comfy-cond-constant-start")
            self._digest(value.cond, hasher, f"{path}.cond")
            _feed(hasher, "comfy-cond-constant-end")
            return

        if isinstance(value, Mapping):
            identity = self._enter(value, path)
            try:
                keys = list(value.keys())
                if any(not isinstance(key, str) for key in keys):
                    raise ValueError(
                        f"positive conditioning mapping keys must be strings at {path}"
                    )
                _feed(hasher, "mapping-start", str(len(keys)).encode("ascii"))
                for key in sorted(keys):
                    _feed(hasher, "mapping-key", key.encode("utf-8"))
                    self._digest(value[key], hasher, f"{path}.{key}")
                _feed(hasher, "mapping-end")
            finally:
                self._active_containers.remove(identity)
            return

        if isinstance(value, Sequence):
            identity = self._enter(value, path)
            try:
                sequence_tag = "tuple" if isinstance(value, tuple) else "list"
                _feed(hasher, f"{sequence_tag}-start", str(len(value)).encode("ascii"))
                for index, item in enumerate(value):
                    self._digest(item, hasher, f"{path}[{index}]")
                _feed(hasher, f"{sequence_tag}-end")
            finally:
                self._active_containers.remove(identity)
            return

        raise ValueError(
            "positive conditioning contains unsupported runtime object "
            f"{value_type.__module__}.{value_type.__qualname__} at {path}"
        )

    def digest(self, positive: Any) -> str:
        if not isinstance(positive, (list, tuple)) or not positive:
            raise ValueError("positive must contain at least one conditioning entry")
        hasher = hashlib.sha256()
        self._digest(positive, hasher, "$")
        return hasher.hexdigest().upper()


def compile_nfe_run_contract(
    *,
    positive: Any,
    conditioned_prompt: str,
    media_map_json: str,
    conditioning_report: str,
    hash_chunk_megabytes: int = 8,
) -> tuple[str, str, str]:
    """Compile an immutable JSON run contract from real H3 conditioning content."""

    chunk_megabytes = int(hash_chunk_megabytes)
    if not 1 <= chunk_megabytes <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")
    prompt = _required_text(
        conditioned_prompt, "conditioned_prompt", MAX_PROMPT_UTF8_BYTES
    )
    report = _required_text(
        conditioning_report, "conditioning_report", MAX_REPORT_UTF8_BYTES
    )
    media_map, canonical_media_map = _parse_media_map(media_map_json)

    digester = _ConditioningDigester(chunk_megabytes * 1024 * 1024)
    conditioning_sha256 = digester.digest(positive)
    payload = {
        "schema": NFE_RUN_CONTRACT_SCHEMA,
        "conditioned_prompt": prompt,
        "conditioned_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "media_map": media_map,
        "media_map_sha256": _sha256_bytes(canonical_media_map.encode("utf-8")),
        "conditioning_report": report,
        "conditioning_report_sha256": _sha256_bytes(report.encode("utf-8")),
        "positive_conditioning": {
            "root_sha256": conditioning_sha256,
            "tensor_count": len(digester.tensor_manifest),
            "tensor_bytes": digester.total_tensor_bytes,
            "tensors": digester.tensor_manifest,
        },
    }
    contract_json = _canonical_json(payload)
    encoded_contract = contract_json.encode("utf-8")
    if len(encoded_contract) > MAX_METADATA_JSON_BYTES:
        raise ValueError(
            "compiled run contract exceeds the checkpoint metadata safety limit; "
            "reduce conditioning metadata instead of truncating the contract"
        )
    contract_sha256 = _sha256_bytes(encoded_contract)
    report_json = _canonical_json(
        {
            "schema": NFE_RUN_CONTRACT_SCHEMA,
            "contract_sha256": contract_sha256,
            "contract_utf8_bytes": len(encoded_contract),
            "conditioned_prompt_sha256": payload["conditioned_prompt_sha256"],
            "media_map_sha256": payload["media_map_sha256"],
            "conditioning_report_sha256": payload["conditioning_report_sha256"],
            "positive_conditioning_sha256": conditioning_sha256,
            "tensor_count": len(digester.tensor_manifest),
            "tensor_bytes": digester.total_tensor_bytes,
            "hash_chunk_megabytes": chunk_megabytes,
        },
        indent=2,
    )
    return contract_json, contract_sha256, report_json
