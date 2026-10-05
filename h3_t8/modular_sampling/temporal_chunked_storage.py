"""Scoped receipt sidecar around the unchanged literal v5 window format.

Explicit selection, not automatic cache admission or today's MODEL/provider
equivalence. The old reader and its exact source pins remain untouched.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import uuid

from . import chunked_v5_storage as base_storage
from . import temporal_chunked_v5 as scoped_runtime
from . import temporal_chunked_relay as scoped_relay
from . import temporal_chunked_driver as scoped_driver
from .storage import MAX_JSON, _digest, _json_file, _lease, _path, _root, file_sha, fingerprint_stage
from .. import temporal_dialogue_bank as banks
from .. import temporal_dialogue_encoding as encoding
from .. import temporal_dialogue_identity as identity
from .. import temporal_dialogue_scope as scope
from ..temporal_dialogue_bank import select_window_conditioning
from ..temporal_dialogue_scope import DialogueReceipt, WindowDescriptor, _validate_receipt
from .results import canonical
from .temporal_chunked_v5 import ScopedWindowResult, _audio_sha


SCHEMA = "t8.temporal-dialogue.frozen-v5-window.v1"


def _implementation():
    paths = [Path(__file__), *(Path(module.__file__) for module in
                              (scoped_runtime, scoped_relay, scoped_driver, banks, encoding, identity, scope))]
    return {path.name: file_sha(path) for path in paths}


def verify_scoped_window(result, source, lifted, prepared, plan, bank):
    if type(result) is not ScopedWindowResult or result.bank_sha256 != bank.sha256:
        raise ValueError("Scoped freeze requires a result from this actual bank/policy")
    binding = base_storage.verify_window(result.base, source, lifted, prepared, plan)
    encoded = select_window_conditioning(bank, source, plan, result.base.index)
    receipt = result.dialogue_receipt
    _validate_receipt(receipt)
    if (receipt.window != encoded.compiled.window
            or receipt.plan_sha256 != bank.dialogue_plan.sha256
            or receipt.text_sha256 != encoded.compiled.sha256
            or receipt.published_audio_sha256 != _audio_sha(result.base)
            or ((receipt.window.index == 0) != (receipt.previous_receipt_sha256 is None))):
        raise ValueError("Scoped receipt, actual published audio or encoded ownership differs")
    return {"base_binding": binding, "bank_sha256": bank.sha256, "receipt": asdict(receipt)}


def _typed_receipt(raw):
    if type(raw) is not dict or set(raw) != {item.name for item in fields(DialogueReceipt)}:
        raise ValueError("Unknown scoped receipt fields")
    window = raw.get("window")
    if type(window) is not dict or set(window) != {item.name for item in fields(WindowDescriptor)}:
        raise ValueError("Unknown scoped window fields")
    receipt = DialogueReceipt(**{**raw, "window": WindowDescriptor(**window)})
    _validate_receipt(receipt)
    return receipt


def save_scoped_window(result, source, lifted, prepared, plan, bank, storage_root):
    binding = verify_scoped_window(result, source, lifted, prepared, plan, bank)
    root = _root(storage_root, create=True)
    directory = root / ("scoped-window-" + uuid.uuid4().hex)
    directory.mkdir()
    with _lease(directory):
        _, _, base_path, base_sha, _ = base_storage.save_window(
            result.base, source, lifted, prepared, plan, root,
            prefix=directory.relative_to(root).as_posix() + "/base")
        verify_scoped_window(result, source, lifted, prepared, plan, bank)
        manifest = {"schema": SCHEMA, "base_artifact_path": base_path,
                    "base_artifact_sha256": base_sha, "binding": binding,
                    "implementation": _implementation(), "automatic_cache_reuse": False,
                    "provider_equivalence_certified": False, "quality_accepted": False}
        payload = canonical(manifest).encode("utf-8")
        if len(payload) > MAX_JSON:
            raise ValueError("Scoped window manifest exceeds metadata limit")
        pending, path = directory / "scope.json.partial", directory / "scope.json"
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    return (result.base.output_latent, result.base, result,
            path.relative_to(root).as_posix(), file_sha(path), canonical(manifest))


def load_scoped_window(source, lifted, prepared, plan, bank, storage_root,
                       artifact_path, artifact_sha256, expected_index):
    encoded = select_window_conditioning(bank, source, plan, expected_index)
    root, expected_sha = _root(storage_root), _digest(artifact_sha256)
    path = _path(root, artifact_path)
    if path.name != "scope.json" or not path.parent.is_dir():
        raise ValueError("Select a completed scoped scope.json, not an old unscoped window")
    with _lease(path.parent):
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON or file_sha(path) != expected_sha:
            raise ValueError("Scoped manifest SHA mismatch or missing completion")
        manifest = _json_file(path)
        required = {"schema", "base_artifact_path", "base_artifact_sha256", "binding",
                    "implementation", "automatic_cache_reuse", "provider_equivalence_certified",
                    "quality_accepted"}
        if (type(manifest) is not dict or set(manifest) != required or manifest["schema"] != SCHEMA
                or manifest["implementation"] != _implementation()
                or any(manifest[key] is not False for key in
                       ("automatic_cache_reuse", "provider_equivalence_certified", "quality_accepted"))):
            raise ValueError("Unknown or stale scoped manifest implementation/contract")
        binding = manifest["binding"]
        if (type(binding) is not dict or set(binding) != {"base_binding", "bank_sha256", "receipt"}
                or binding["bank_sha256"] != bank.sha256):
            raise ValueError("Scoped frozen bank/policy differs")
        receipt = _typed_receipt(binding["receipt"])
        if receipt.window != encoded.compiled.window:
            raise ValueError("Scoped frozen ownership or expected window differs")
        base_path = _path(root, manifest["base_artifact_path"])
        if base_path.name != "manifest.json" or base_path.parent.parent != path.parent:
            raise ValueError("Scoped base artifact must belong to its owned sidecar directory")
        output, base, _ = base_storage.load_window(
            source, lifted, prepared, plan, root, manifest["base_artifact_path"],
            manifest["base_artifact_sha256"], expected_index)
        result = ScopedWindowResult(base, bank.sha256, receipt)
        if verify_scoped_window(result, source, lifted, prepared, plan, bank) != binding:
            raise ValueError("Scoped literal window content or receipt differs")
        if file_sha(path) != expected_sha:
            raise ValueError("Scoped manifest changed while loading")
    report = {"schema": SCHEMA, "status": "explicit_scoped_literal_window_loaded",
              "window_index": expected_index, "bank_sha256": bank.sha256,
              "automatic_cache_reuse": False, "provider_equivalence_certified": False,
              "quality_accepted": False}
    return output, base, result, json.dumps(report, ensure_ascii=False, sort_keys=True)


def fingerprint_scoped_window(storage_root, artifact_path):
    root = _root(storage_root)
    path = _path(root, artifact_path)
    if path.name != 'scope.json':
        raise ValueError('Select scoped scope.json')
    with _lease(path.parent):
        manifest = _json_file(path)
        if manifest.get('schema') != SCHEMA:
            raise ValueError('Unknown scoped window schema')
        base = _path(root, manifest['base_artifact_path'])
        if base.name != 'manifest.json' or base.parent.parent != path.parent:
            raise ValueError('Scoped base artifact escaped its sidecar directory')
        return file_sha(path), fingerprint_stage(root, manifest['base_artifact_path'])
