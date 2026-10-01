"""Explicit frozen continuous HyperFlow state; never pickle or hidden sampling.

The single self-contained safetensors file is committed by atomic rename under
an OS lease. HEAD stores exact raw x_sigma and its separate Core scaffold;
completed TAIL stores AV for delivery. Neither is a Progressive/ordinary stage.
"""
import json
import os
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from .hyperflow import ContinuousBoundary, ContinuousResult
from .progressive import portable
from .results import canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _lease, _path, _root, file_sha

SCHEMA = "t8.modular-sampling.frozen-hyperflow-continuous.v1"
KINDS = {"head": (ContinuousBoundary, "hyperflow-head.safetensors"),
         "tail": (ContinuousResult, "hyperflow-tail.safetensors")}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate HyperFlow artifact field")
        result[key] = value
    return result


def _payload(text):
    if type(text) is not str or len(text.encode("utf-8")) > MAX_JSON:
        raise ValueError("HyperFlow metadata exceeds its limit")
    value = json.loads(text, object_pairs_hook=_unique)
    if (type(value) is not dict or set(value) != {"schema", "kind", "receipt_json", "state"}
            or value["schema"] != SCHEMA or value["kind"] not in KINDS
            or type(value["receipt_json"]) is not str):
        raise ValueError("Unknown typed HyperFlow artifact schema")
    json.loads(value["receipt_json"], object_pairs_hook=_unique)
    return value


def _check_file(path):
    if not path.is_file() or not 8 < path.stat().st_size <= MAX_FILE:
        raise ValueError("Missing or oversized HyperFlow artifact")
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
    if not 0 < count <= MAX_JSON or count > path.stat().st_size - 8:
        raise ValueError("Invalid HyperFlow safetensors header size")


def _portable(result, receipt):
    if type(result) is ContinuousBoundary:
        execution, output = receipt["execution"], receipt["outputs"]
    else:
        execution, output = receipt["sampling"]["execution"], receipt["output"]
        if receipt["sampling"].get("source_portable_identity") is not True:
            return False
    return bool(execution["portable_identity"] and portable(output))


def _decode(payload, tensors):
    used = set()
    state = _decode_metadata_value(payload["state"], tensors, used, path="hyperflow.state")
    if used != set(tensors) or type(state) is not dict:
        raise ValueError("HyperFlow artifact tensor inventory differs from its typed state")
    kind = payload["kind"]
    if kind == "head":
        if set(state) != {"x_sigma", "scaffold"}:
            raise ValueError("HyperFlow HEAD requires raw state and a separate scaffold")
        result = ContinuousBoundary(state["x_sigma"], state["scaffold"], payload["receipt_json"])
    else:
        if set(state) != {"output"}:
            raise ValueError("HyperFlow TAIL requires completed AV")
        result = ContinuousResult(state["output"], payload["receipt_json"])
    result.verify()
    return result


def _save(result, storage_root, prefix, kind):
    expected, filename = KINDS[kind]
    if type(result) is not expected:
        raise ValueError(f"HyperFlow {kind} Save requires its dedicated typed result")
    receipt = result.verify()
    execution = receipt["execution"] if kind == "head" else receipt["sampling"]["execution"]
    if execution.get("actual_forward_coverage") is not True:
        raise ValueError("HyperFlow artifact needs actual completed stage forward evidence")
    state = ({"x_sigma": result.x_sigma, "scaffold": result.scaffold} if kind == "head"
             else {"output": result.output})
    tensors = {}
    encoded = _encode_metadata_value(state, tensors, path="hyperflow.state")
    payload = canonical({"schema": SCHEMA, "kind": kind, "receipt_json": result.receipt_json, "state": encoded})
    _decode(_payload(payload), tensors)
    if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
        raise ValueError("HyperFlow tensors exceed the size limit")
    root = _root(storage_root, create=True)
    base = _path(root, prefix)
    directory = base.with_name(base.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        partial, final = directory / (filename + ".partial"), directory / filename
        save_file(tensors, str(partial), metadata={"hyperflow_json": payload})
        _check_file(partial)
        with partial.open("rb+") as handle:
            os.fsync(handle.fileno())
        result.verify()
        os.replace(partial, final)
        digest = file_sha(final)
    verified = _portable(result, receipt)
    return final.relative_to(root).as_posix(), digest, canonical({
        "schema": SCHEMA, "kind": kind, "artifact_sha256": digest,
        "receipt_sha256": receipt["receipt_sha256"], "portable_identity": verified,
        "automatic_cache_reuse": False,
        "status": "saved_typed_stage" if verified else "archived_unverified_identity"})


def _selected(storage_root, artifact_path, kind):
    path = _path(_root(storage_root), artifact_path)
    if path.name != KINDS[kind][1]:
        raise ValueError("Select the expected completed HyperFlow stage, not another kind or partial")
    if not path.parent.is_dir():
        raise FileNotFoundError("HyperFlow artifact directory does not exist")
    return path


def _load(storage_root, artifact_path, artifact_sha256, kind):
    path = _selected(storage_root, artifact_path, kind)
    digest = _digest(artifact_sha256)
    with _lease(path.parent):
        _check_file(path)
        if file_sha(path) != digest:
            raise ValueError("HyperFlow artifact SHA mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {"hyperflow_json"}:
                raise ValueError("Unknown HyperFlow tensor metadata")
            payload = _payload(metadata["hyperflow_json"])
            if payload["kind"] != kind:
                raise ValueError("HyperFlow artifact contains another stage kind")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        result = _decode(payload, tensors)
        receipt = result.verify()
        if not _portable(result, receipt):
            raise ValueError("Unverified executable identity: persistent HyperFlow reuse needs its adapter")
        if file_sha(path) != digest:
            raise ValueError("HyperFlow artifact changed during loading")
    return result, canonical({"status": "explicit_frozen_hyperflow_stage_loaded", "kind": kind,
        "artifact_sha256": digest, "receipt_sha256": receipt["receipt_sha256"], "sampling_calls": 0,
        "automatic_cache_reuse": False,
        "boundary": "Exact selected frozen stage; HEAD is raw x_sigma, not clean x0. No source stage execution. "
                    "Does not assert current HEAD settings match this artifact or certify media quality."})


def fingerprint(storage_root, artifact_path, kind):
    if kind not in KINDS:
        raise ValueError("Unknown HyperFlow stage kind")
    path = _selected(storage_root, artifact_path, kind)
    with _lease(path.parent):
        _check_file(path)
        return file_sha(path)


def save_boundary(boundary, storage_root, prefix="hyperflow-head"):
    return _save(boundary, storage_root, prefix, "head")


def load_boundary(storage_root, artifact_path, artifact_sha256):
    return _load(storage_root, artifact_path, artifact_sha256, "head")


def save_result(result, storage_root, prefix="hyperflow-tail"):
    return _save(result, storage_root, prefix, "tail")


def load_result(storage_root, artifact_path, artifact_sha256):
    return _load(storage_root, artifact_path, artifact_sha256, "tail")
