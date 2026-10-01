"""Explicit typed Progressive LOW artifacts, never ordinary completed LATENTs.

Unique directories, OS leases and exact caller-selected SHA. The atomic rename
of the self-contained safetensors file is the sole completion point. Unknown
executable identities may be archived, but not certified for persistent reuse.
"""
import json
import os
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from .progressive import ProgressiveBoundary, plan_from_dict
from .results import canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _lease, _path, _root, file_sha

SCHEMA = "t8.modular-sampling.frozen-progressive-low.v1"
FILENAME = "progressive-low.safetensors"


def _payload(text):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate Progressive artifact key")
            output[key] = value
        return output
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON:
        raise ValueError("Progressive metadata exceeds its size limit")
    value = json.loads(text, object_pairs_hook=unique)
    if (type(value) is not dict or set(value) != {"schema", "receipt_json"}
            or value["schema"] != SCHEMA or type(value["receipt_json"]) is not str):
        raise ValueError("Unknown Progressive artifact schema")
    return value


def _check_file(path):
    if not path.is_file() or not 8 < path.stat().st_size <= MAX_FILE:
        raise ValueError("Missing or oversized Progressive LOW artifact")
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
    if not 0 < count <= MAX_JSON or count > path.stat().st_size - 8:
        raise ValueError("Invalid Progressive safetensors header size")


def save_boundary(boundary, storage_root, prefix="progressive-low"):
    if type(boundary) is not ProgressiveBoundary:
        raise ValueError("Save requires a typed ProgressiveBoundary, not a LATENT")
    receipt = boundary.verify()
    plan = plan_from_dict(receipt["request"]["plan"])
    if receipt["execution"]["actual_apply_calls"] < plan.low_evaluations:
        raise ValueError("Incomplete LOW forward evidence cannot be saved as completed")
    root = _root(storage_root, create=True)
    base = _path(root, prefix)
    directory = base.with_name(base.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Exclusive new namespace, no overwrite or resume guess.
    with _lease(directory):
        tensors = {key: value.detach().to(device="cpu", copy=True).contiguous()
                   for key, value in boundary.tensors.items()}
        ProgressiveBoundary(tensors, boundary.receipt_json).verify()
        payload = canonical({"schema": SCHEMA, "receipt_json": boundary.receipt_json})
        _payload(payload)
        if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("Progressive tensors exceed the size limit")
        partial, final = directory / (FILENAME + ".partial"), directory / FILENAME
        save_file(tensors, str(partial), metadata={"progressive_json": payload})
        _check_file(partial)
        with partial.open("rb+") as handle:
            os.fsync(handle.fileno())
        boundary.verify()
        os.replace(partial, final)
        digest = file_sha(final)
    report = {"schema": SCHEMA, "receipt_sha256": receipt["receipt_sha256"],
              "request_sha256": receipt["request_sha256"], "portable_identity": receipt["portable_identity"],
              "automatic_cache_reuse": False, "artifact_sha256": digest,
              "status": "saved_typed_low" if receipt["portable_identity"] else "archived_unverified_identity"}
    return final.relative_to(root).as_posix(), digest, canonical(report)


def load_boundary(storage_root, artifact_path, artifact_sha256):
    root = _root(storage_root)
    path = _path(root, artifact_path)
    digest = _digest(artifact_sha256)
    if path.name != FILENAME:
        raise ValueError("Select the completed Progressive LOW artifact, not a partial or ordinary stage")
    if not path.parent.is_dir():
        raise FileNotFoundError("Progressive artifact directory does not exist")
    with _lease(path.parent):
        _check_file(path)
        if file_sha(path) != digest:
            raise ValueError("Progressive LOW artifact SHA mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {"progressive_json"}:
                raise ValueError("Unknown Progressive tensor metadata")
            payload = _payload(metadata["progressive_json"])
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        result = ProgressiveBoundary(tensors, payload["receipt_json"])
        receipt = result.verify()
        if receipt["portable_identity"] is not True:
            raise ValueError("Unverified executable identity: persistent Progressive reuse needs its adapter")
        if file_sha(path) != digest:
            raise ValueError("Progressive artifact changed during loading")
    report = {"status": "explicit_frozen_progressive_low_loaded", "artifact_sha256": digest,
              "receipt_sha256": receipt["receipt_sha256"], "automatic_cache_reuse": False,
              "low_executed": False, "source_implementation": receipt["request"]["implementation"],
              "boundary": "Selected completed clean_video/audio_next/anchor_audio_noise, not a final AV latent; "
                          "does not assert today's LOW configuration matches the selected artifact."}
    return result, canonical(report)


def fingerprint_boundary(storage_root, artifact_path):
    root = _root(storage_root)
    path = _path(root, artifact_path)
    if path.name != FILENAME:
        raise ValueError("Expected the completed Progressive LOW artifact")
    if not path.parent.is_dir():
        raise FileNotFoundError("Progressive artifact directory does not exist")
    with _lease(path.parent):
        _check_file(path)
        return file_sha(path)
