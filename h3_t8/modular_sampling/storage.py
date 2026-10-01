"""Explicit frozen stage artifacts, not a hidden cache or MODEL serializer.

A completed manifest is committed last, under an OS-owned lease. Readers need
its exact SHA plus the expected stage; orphan tensors are never completion.
Selecting an artifact deliberately freezes that prior stage, not an assertion
that today's LOW settings still match it. Automatic recipe cache lookup is a
separate, not-yet-enabled path.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from ..long_video_delivery import _open_advisory_lock, _try_advisory_lock, _release_advisory_lock
from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value, _relative_parts
from .contracts import StageContext
from .results import StageResult, canonical

SCHEMA = "t8.modular-sampling.frozen-stage.v1"
MAX_JSON = 4 * 1024 * 1024
MAX_FILE = 20 * 1024 ** 3


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-fA-F]{64}", value) is None:
        raise ValueError("An exact nonempty SHA-256 is required for a frozen stage")
    return value.lower()


def _is_link(path):
    # Windows junctions are also reparse points; resolve-only containment would
    # silently follow an in-root link that can later be retargeted.
    if path.is_symlink():
        return True
    return path.exists() and bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _root(value, create=False):
    unresolved = Path(value).absolute()
    for part in (unresolved, *unresolved.parents):
        if _is_link(part):
            raise ValueError("Stage store root cannot traverse a link or junction")
    if create:
        unresolved.mkdir(parents=True, exist_ok=True)
    root = unresolved.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Stage store does not exist")
    return root


def _path(root, relative):
    parts = _relative_parts(relative, "stage artifact")
    current = root
    for part in parts:
        current = current / part
        if _is_link(current):
            raise ValueError("Stage artifact cannot traverse a link or junction")
    if not current.resolve().is_relative_to(root):
        raise ValueError("Stage artifact escapes its store")
    return current


@contextmanager
def _lease(directory):
    lock = directory / ".stage.lock"
    if _is_link(lock):
        raise ValueError("Stage lease cannot be a link")
    handle = _open_advisory_lock(lock)
    acquired = False
    try:
        acquired = _try_advisory_lock(handle)
        if not acquired:
            raise RuntimeError("Stage artifact is busy; no data was reused or overwritten")
        yield
    finally:
        if acquired:
            _release_advisory_lock(handle)
        handle.close()


def _json_file(path):
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON:
        raise ValueError("Missing or oversized completed stage manifest")
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate stage manifest key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def save_stage(result, storage_root, prefix="stage"):
    if type(result) is not StageResult:
        raise ValueError("Save needs a sampler-produced typed StageResult, not an arbitrary LATENT")
    receipt = result.verify()
    if receipt["verified_recipe_completion"] is not True:
        raise ValueError("An incomplete or unknown recipe execution cannot be saved as completed")
    root = _root(storage_root, create=True)
    prefix_path = _path(root, prefix)
    directory = prefix_path.with_name(prefix_path.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Exclusive namespace; never replace an existing stage.
    with _lease(directory):
        tensors = {}
        encoded = _encode_metadata_value({"output": result.output, "denoised_output": result.denoised_output},
                                         tensors, path="stage.outputs")
        payload = canonical({"schema": SCHEMA, "receipt_json": result.receipt_json, "state": encoded})
        if len(payload.encode("utf-8")) > MAX_JSON:
            raise ValueError("Stage metadata exceeds the 4 MiB limit")
        if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("Stage tensor payload exceeds the 20 GiB limit")
        snapshot = _decode_metadata_value(encoded, tensors, set(), path="stage.outputs")
        StageResult(snapshot["output"], snapshot["denoised_output"], result.receipt_json).verify()
        # Serialization snapshots tensor values, then revalidates the original:
        # a concurrent mutation must not acquire a completed manifest.
        result.verify()
        partial = directory / "state.safetensors.partial"
        state = directory / "state.safetensors"
        save_file(tensors, str(partial), metadata={"stage_json": payload})
        with partial.open("rb+") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_bytes <= MAX_JSON:
                raise ValueError("Stage tensor header exceeds its bound")
            os.fsync(handle.fileno())
        os.replace(partial, state)
        result.verify()
        manifest = {"schema": SCHEMA, "state_file": state.name, "state_sha256": file_sha(state),
                    "state_bytes": state.stat().st_size, "receipt_sha256": receipt["receipt_sha256"],
                    "request_sha256": receipt["request_sha256"], "stage_context": receipt["request"]["stage_context"],
                    "portable_identity": receipt["portable_identity"], "automatic_cache_reuse": False}
        manifest_path = directory / "manifest.json"
        pending = directory / "manifest.json.partial"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, manifest_path)  # The sole completion commit point.
    return manifest_path.relative_to(root).as_posix(), file_sha(manifest_path), canonical(manifest)


def load_stage(storage_root, artifact_path, artifact_sha256, expected_stage):
    root = _root(storage_root)
    manifest_path = _path(root, artifact_path)
    if manifest_path.name != "manifest.json":
        raise ValueError("Select a completed manifest.json, not a tensor or partial file")
    expected_sha = _digest(artifact_sha256)
    if not manifest_path.parent.is_dir():
        raise FileNotFoundError("Stage artifact directory does not exist")
    with _lease(manifest_path.parent):
        if (not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_JSON
                or file_sha(manifest_path) != expected_sha):
            raise ValueError("Frozen stage manifest SHA mismatch or missing completion")
        manifest = _json_file(manifest_path)
        fields = {"schema", "state_file", "state_sha256", "state_bytes", "receipt_sha256", "request_sha256",
                  "stage_context", "portable_identity", "automatic_cache_reuse"}
        if (set(manifest) != fields or manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors"
                or manifest["automatic_cache_reuse"] is not False):
            raise ValueError("Unknown or incomplete stage manifest contract")
        context = StageContext.from_dict(manifest["stage_context"])
        if context.stage != expected_stage:
            raise ValueError("Frozen artifact is not the expected stage")
        if manifest["portable_identity"] is not True:
            raise ValueError("This artifact has unverified executable identity; persistent stage reuse needs its adapter")
        state = _path(root, (manifest_path.parent / manifest["state_file"]).relative_to(root).as_posix())
        if (not state.is_file() or type(manifest["state_bytes"]) is not int
                or not 8 < manifest["state_bytes"] <= MAX_FILE or state.stat().st_size != manifest["state_bytes"]):
            raise ValueError("Stage tensor size changed or exceeds its limit")
        if file_sha(state) != _digest(manifest["state_sha256"]):
            raise ValueError("Stage tensor SHA mismatch")
        with state.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_bytes <= MAX_JSON or header_bytes > state.stat().st_size - 8:
            raise ValueError("Stage tensor header exceeds its bound")
        with safe_open(str(state), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or set(metadata) != {"stage_json"}:
                raise ValueError("Stage tensor metadata is missing or unknown")
            payload = json.loads(metadata["stage_json"])
            if set(payload) != {"schema", "receipt_json", "state"} or payload["schema"] != SCHEMA:
                raise ValueError("Unknown stage tensor payload")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        used = set()
        values = _decode_metadata_value(payload["state"], tensors, used, path="stage.outputs")
        if used != set(tensors) or not isinstance(values, dict) or set(values) != {"output", "denoised_output"}:
            raise ValueError("Stage tensor inventory differs from its typed descriptor")
        result = StageResult(values["output"], values["denoised_output"], payload["receipt_json"])
        receipt = result.verify()
        if (receipt["receipt_sha256"] != manifest["receipt_sha256"]
                or receipt["request_sha256"] != manifest["request_sha256"]
                or receipt["request"]["stage_context"] != context.to_dict()
                or receipt["verified_recipe_completion"] is not True or receipt["portable_identity"] is not True):
            raise ValueError("Stage result and completion manifest disagree")
        if file_sha(state) != manifest["state_sha256"] or file_sha(manifest_path) != expected_sha:
            raise ValueError("Stage artifact changed while being loaded")
    report = {"status": "explicit_frozen_stage_loaded", "stage": context.stage,
              "request_sha256": receipt["request_sha256"], "artifact_sha256": expected_sha,
              "automatic_cache_reuse": False, "source_implementation": receipt["request"]["implementation"],
              "boundary": "Exact selected completed artifact; no LOW execution and no assertion that current "
                          "LOW settings match the frozen source. Native x_sigma and denoised outputs stay distinct."}
    return result.output, result.denoised_output, context, result, canonical(report)


def fingerprint_stage(storage_root, artifact_path):
    """Stable file-content fingerprint; detects replacement despite UI caching."""
    root = _root(storage_root)
    manifest = _path(root, artifact_path)
    if not manifest.parent.is_dir():
        raise FileNotFoundError("Stage artifact directory does not exist")
    with _lease(manifest.parent):
        data = _json_file(manifest)
        if data.get("state_file") != "state.safetensors":
            raise ValueError("Unknown stage state filename")
        state = _path(root, (manifest.parent / "state.safetensors").relative_to(root).as_posix())
        if not 8 < state.stat().st_size <= MAX_FILE:
            raise ValueError("Invalid stage tensor size")
        return file_sha(manifest), file_sha(state)
