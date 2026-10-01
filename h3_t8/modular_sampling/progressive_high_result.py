"""Completed HIGH output evidence and explicit storage; never runs LOW or lift."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from . import progressive as stages
from .progressive_storage import _check_file
from .results import canonical, sha
from .storage import MAX_FILE, MAX_JSON, _digest, _lease, _path, _root, file_sha

SCHEMA = "t8.modular-sampling.progressive-completed-high.v1"
STORAGE_SCHEMA = "t8.modular-sampling.frozen-progressive-high.v1"
FILENAME = "progressive-high.safetensors"


@dataclass(frozen=True)
class ProgressiveHighResult:
    output: dict
    receipt_json: str

    def verify(self):
        receipt = json.loads(self.receipt_json)
        if type(receipt) is not dict or set(receipt) != {
                "schema", "sampling", "output", "producer_sha256", "portable_identity", "receipt_sha256"}:
            raise ValueError("Unknown completed Progressive HIGH receipt")
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_sha256")
        if receipt["schema"] != SCHEMA or sha(unsigned) != digest:
            raise ValueError("Completed Progressive HIGH receipt changed")
        sampling = receipt["sampling"]
        if (type(sampling) is not dict or sampling.get("schema") != "t8.modular-sampling.progressive-high.v1"
                or sampling.get("low_executed") is not False):
            raise ValueError("Expected native HIGH-only completion evidence")
        plan = stages.plan_from_dict(sampling["restart"]["plan"])
        stages._geometry(self.output, plan, "high")
        execution = sampling["execution"]
        if (execution["phase"] != "high" or execution["callbacks"] != list(range(plan.high_evaluations))
                or type(execution["actual_apply_calls"]) is not int
                or execution["actual_apply_calls"] < plan.high_evaluations):
            raise ValueError("Incomplete Progressive HIGH execution")
        if receipt["output"] != stages.snapshot(self.output):
            raise ValueError("Completed Progressive HIGH output changed")
        portable = bool(sampling["portable_identity"] and stages.portable(receipt["output"]))
        if type(receipt["portable_identity"]) is not bool or receipt["portable_identity"] != portable:
            raise ValueError("Progressive HIGH portable identity differs")
        _digest(receipt["producer_sha256"])
        return receipt


def sample_high_result(restart, model, sampler, positive, negative, **options):
    output, report = stages.sample_high(restart, model, sampler, positive, negative, **options)
    sampling = json.loads(report)
    identity = stages.snapshot(output)
    receipt = {"schema": SCHEMA, "sampling": sampling, "output": identity,
               "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "portable_identity": bool(sampling["portable_identity"] and stages.portable(identity))}
    receipt["receipt_sha256"] = sha(receipt)
    result = ProgressiveHighResult(output, canonical(receipt))
    result.verify()
    return result, report


def _payload(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate Progressive HIGH metadata key")
            result[key] = value
        return result
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON:
        raise ValueError("Progressive HIGH metadata exceeds its limit")
    result = json.loads(text, object_pairs_hook=unique)
    if (type(result) is not dict or set(result) != {"schema", "receipt_json", "output"}
            or result["schema"] != STORAGE_SCHEMA or type(result["receipt_json"]) is not str):
        raise ValueError("Unknown Progressive HIGH artifact schema")
    return result


def _decode(payload, tensors):
    used = set()
    output = _decode_metadata_value(payload["output"], tensors, used, path="progressive.high")
    if used != set(tensors):
        raise ValueError("Unexpected unused Progressive HIGH tensors")
    result = ProgressiveHighResult(output, payload["receipt_json"])
    result.verify()
    return result


def save_result(result, storage_root, prefix="progressive-high"):
    if type(result) is not ProgressiveHighResult:
        raise ValueError("Save requires a completed typed ProgressiveHighResult")
    receipt = result.verify()
    tensors = {}
    encoded = _encode_metadata_value(result.output, tensors, path="progressive.high")
    payload = canonical({"schema": STORAGE_SCHEMA, "receipt_json": result.receipt_json, "output": encoded})
    _decode(_payload(payload), tensors)
    if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
        raise ValueError("Progressive HIGH tensors exceed their size limit")
    root = _root(storage_root, create=True)
    base = _path(root, prefix)
    directory = base.with_name(base.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        partial, final = directory / (FILENAME + ".partial"), directory / FILENAME
        save_file(tensors, str(partial), metadata={"progressive_high_json": payload})
        _check_file(partial)
        with partial.open("rb+") as handle:
            os.fsync(handle.fileno())
        result.verify()
        os.replace(partial, final)
        digest = file_sha(final)
    return final.relative_to(root).as_posix(), digest, canonical({
        "schema": STORAGE_SCHEMA, "receipt_sha256": receipt["receipt_sha256"], "artifact_sha256": digest,
        "portable_identity": receipt["portable_identity"], "automatic_cache_reuse": False,
        "status": "saved_completed_high" if receipt["portable_identity"] else "archived_unverified_identity"})


def _selected(storage_root, artifact_path):
    root = _root(storage_root)
    path = _path(root, artifact_path)
    if path.name != FILENAME:
        raise ValueError("Select completed Progressive HIGH, not LOW/partial or an ordinary LATENT")
    if not path.parent.is_dir():
        raise FileNotFoundError("Progressive HIGH artifact directory does not exist")
    return path


def load_result(storage_root, artifact_path, artifact_sha256):
    path = _selected(storage_root, artifact_path)
    digest = _digest(artifact_sha256)
    with _lease(path.parent):
        _check_file(path)
        if file_sha(path) != digest:
            raise ValueError("Progressive HIGH artifact SHA mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {"progressive_high_json"}:
                raise ValueError("Unknown Progressive HIGH tensor metadata")
            payload = _payload(metadata["progressive_high_json"])
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        result = _decode(payload, tensors)
        receipt = result.verify()
        if receipt["portable_identity"] is not True:
            raise ValueError("Unverified executable identity: persistent Progressive HIGH reuse needs its adapter")
        if file_sha(path) != digest:
            raise ValueError("Progressive HIGH artifact changed during loading")
    return result, canonical({"status": "explicit_frozen_progressive_high_loaded", "artifact_sha256": digest,
        "receipt_sha256": receipt["receipt_sha256"], "sampling_calls": 0, "automatic_cache_reuse": False,
        "boundary": "Selected completed HIGH AV; no LOW, lift or HIGH execution. "
                    "Does not certify current model/conditions match this frozen result or any media quality."})


def fingerprint_result(storage_root, artifact_path):
    path = _selected(storage_root, artifact_path)
    with _lease(path.parent):
        _check_file(path)
        return file_sha(path)
