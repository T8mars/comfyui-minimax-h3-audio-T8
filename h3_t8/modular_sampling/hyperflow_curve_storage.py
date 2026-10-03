"""Explicit curve-stage safetensors store; separate types from full HyperFlow.

No MODEL/pickle serialization, hidden sampling or filename-based cache lookup.
The caller selects a completed artifact and its whole-file SHA deliberately.
"""
import json
import os
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from .hyperflow_curve import CurveBoundary, CurveResult
from .hyperflow_storage import _check_file
from .progressive import portable
from .results import canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _lease, _path, _root, file_sha

SCHEMA = "t8.hyperflow.frozen_curve_stage.v1"
KINDS = {"head": (CurveBoundary, "curve-head.safetensors"), "tail": (CurveResult, "curve-tail.safetensors")}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate curve artifact metadata field")
        result[key] = value
    return result


def _payload(text):
    if type(text) is not str or len(text.encode("utf-8")) > MAX_JSON:
        raise ValueError("Curve artifact metadata exceeds its bound")
    value = json.loads(text, object_pairs_hook=_unique)
    if (type(value) is not dict or set(value) != {"schema", "kind", "receipt_json", "state"}
            or value["schema"] != SCHEMA or value["kind"] not in KINDS or type(value["receipt_json"]) is not str):
        raise ValueError("Unknown curve artifact schema/kind")
    json.loads(value["receipt_json"], object_pairs_hook=_unique)
    return value


def _portable(result, receipt):
    if type(result) is CurveBoundary:
        return bool(receipt["execution"]["portable_identity"] and portable(receipt["outputs"]))
    return bool(receipt["sampling"]["execution"]["portable_identity"]
                and receipt["sampling"]["source_portable_identity"] and portable(receipt["output"]))


def _decode(payload, tensors):
    used = set()
    state = _decode_metadata_value(payload["state"], tensors, used, path="curve.state")
    if type(state) is not dict or used != set(tensors):
        raise ValueError("Curve artifact state/tensor inventory changed")
    if payload["kind"] == "head":
        if set(state) != {"x_sigma", "scaffold"}:
            raise ValueError("Curve HEAD needs exact raw x_sigma and separate scaffold")
        result = CurveBoundary(state["x_sigma"], state["scaffold"], payload["receipt_json"])
    else:
        if set(state) != {"output"}:
            raise ValueError("Curve TAIL requires completed joint AV")
        result = CurveResult(state["output"], payload["receipt_json"])
    result.verify()
    return result


def _save(result, storage_root, prefix, kind):
    expected, filename = KINDS[kind]
    if type(result) is not expected:
        raise ValueError("Curve store requires its dedicated typed phase result")
    receipt = result.verify()
    execution = receipt["execution"] if kind == "head" else receipt["sampling"]["execution"]
    if execution.get("actual_51_coverage") is not True:
        raise ValueError("Curve archive needs completed actual51 native projection evidence")
    state = {"x_sigma": result.x_sigma, "scaffold": result.scaffold} if kind == "head" else {"output": result.output}
    tensors = {}
    encoded = _encode_metadata_value(state, tensors, path="curve.state")
    payload = canonical({"schema": SCHEMA, "kind": kind, "receipt_json": result.receipt_json, "state": encoded})
    _decode(_payload(payload), tensors)
    if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
        raise ValueError("Curve artifact tensors exceed the store bound")
    root = _root(storage_root, create=True)
    base = _path(root, prefix)
    directory = base.with_name(base.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        temporary, final = directory / (filename + ".partial"), directory / filename
        save_file(tensors, str(temporary), metadata={"t8_curve_stage": payload})
        _check_file(temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        result.verify()
        os.replace(temporary, final)
        digest = file_sha(final)
    verified = _portable(result, receipt)
    return final.relative_to(root).as_posix(), digest, canonical({"schema": SCHEMA, "kind": kind,
        "artifact_sha256": digest, "receipt_sha256": receipt["receipt_sha256"], "portable_identity": verified,
        "automatic_cache_reuse": False, "status": "saved_curve_stage" if verified else "archived_nonportable_curve_stage"})


def _selected(storage_root, path, kind):
    file = _path(_root(storage_root), path)
    if file.name != KINDS[kind][1] or not file.parent.is_dir():
        raise ValueError("Select the exact completed curve stage kind, not a partial or full-HyperFlow artifact")
    return file


def _load(storage_root, path, expected_sha, kind):
    file = _selected(storage_root, path, kind)
    digest = _digest(expected_sha)
    with _lease(file.parent):
        _check_file(file)
        if file_sha(file) != digest:
            raise ValueError("Curve artifact SHA mismatch")
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {"t8_curve_stage"}:
                raise ValueError("Unknown curve artifact metadata owner")
            payload = _payload(metadata["t8_curve_stage"])
            if payload["kind"] != kind:
                raise ValueError("Curve artifact contains another phase")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        result = _decode(payload, tensors)
        receipt = result.verify()
        if not _portable(result, receipt):
            raise ValueError("Unknown executable owner: persistent curve reuse needs a portable identity adapter")
        if file_sha(file) != digest:
            raise ValueError("Curve artifact changed while loading")
    return result, canonical({"status": "explicit_frozen_curve_stage_loaded", "kind": kind,
        "artifact_sha256": digest, "receipt_sha256": receipt["receipt_sha256"], "sampling_calls": 0,
        "automatic_cache_reuse": False, "quality_accepted": False})


def save_boundary(result, root, prefix="curve-head"):
    return _save(result, root, prefix, "head")


def load_boundary(root, path, sha256):
    return _load(root, path, sha256, "head")


def save_result(result, root, prefix="curve-tail"):
    return _save(result, root, prefix, "tail")


def load_result(root, path, sha256):
    return _load(root, path, sha256, "tail")


def fingerprint(root, path, kind):
    if kind not in KINDS:
        raise ValueError("Unknown curve artifact kind")
    file = _selected(root, path, kind)
    with _lease(file.parent):
        _check_file(file)
        return file_sha(file)
