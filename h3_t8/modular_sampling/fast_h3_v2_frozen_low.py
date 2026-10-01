"""Carry an attested current V2 LOW recipe across an explicit stage reload.

This sidecar supplements, but never changes, the generic immutable Stage
manifest. It replays no sampler and claims no identity for newly edited LOW
inputs. Frozen HIGH must still attest its own live MODEL, conditions and EAV.
"""

import hashlib
import json
import os
from pathlib import Path
import uuid
import zlib

from . import storage
from .fast_h3_v2_conditioning import FastH3V2ConditionReceipt
from .fast_h3_v2_conditioning import implementation_sha256 as condition_implementation
from .fast_h3_v2_job import FastH3V2CurrentRecipe
from .fast_h3_v2_stage_attest import FastH3V2StageAttestation
from .results import StageResult, canonical, implementation_identity


SCHEMA = "t8.modular-sampling.fasth3-v2-frozen-low-bundle.v1"
SIDECAR = "current-v2-low-bundle.json"
ENVELOPE_SCHEMA = "t8.modular-sampling.fasth3-v2-frozen-low-envelope.v1"
BLOB = "current-v2-low-bundle.zlib"
MAX_INLINE_JSON = storage.MAX_JSON
MAX_BUNDLE_JSON = 64 * 1024 * 1024
MAX_COMPRESSED = 16 * 1024 * 1024


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate frozen LOW bundle key")
        result[key] = value
    return result


def _read_payload(target):
    wrapper = storage._json_file(target)
    if wrapper.get("schema") != ENVELOPE_SCHEMA:
        return wrapper
    if (wrapper.get("encoding") != "zlib" or wrapper.get("blob") != BLOB
            or type(wrapper.get("raw_bytes")) is not int
            or not 0 < wrapper["raw_bytes"] <= MAX_BUNDLE_JSON
            or type(wrapper.get("compressed_bytes")) is not int
            or not 0 < wrapper["compressed_bytes"] <= MAX_COMPRESSED):
        raise ValueError("Frozen LOW compressed envelope is invalid")
    blob = target.with_name(BLOB)
    if (blob.is_symlink() or not blob.is_file()
            or blob.stat().st_size != wrapper["compressed_bytes"]
            or storage.file_sha(blob) != wrapper.get("blob_sha256")):
        raise ValueError("Frozen LOW compressed blob changed")
    compressed = blob.read_bytes()
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, MAX_BUNDLE_JSON + 1)
    if (len(raw) != wrapper["raw_bytes"] or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail
            or hashlib.sha256(raw).hexdigest() != wrapper.get("raw_sha256")):
        raise ValueError("Frozen LOW compressed payload changed or exceeded its bound")
    payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    if canonical(payload).encode("utf-8") != raw:
        raise ValueError("Frozen LOW payload is not canonical")
    return payload


def implementation_sha256():
    from . import fast_h3_v2_job, fast_h3_v2_stage_attest
    sources = {"frozen_low_bundle": __file__,
               "current_recipe": fast_h3_v2_job.__file__,
               "stage_attestation": fast_h3_v2_stage_attest.__file__}
    fingerprints = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in sources.items()}
    return hashlib.sha256(canonical(fingerprints).encode("utf-8")).hexdigest()


def _witnesses(recipe, result, attestation, condition):
    if (type(recipe) is not FastH3V2CurrentRecipe or type(result) is not StageResult
            or type(attestation) is not FastH3V2StageAttestation
            or type(condition) is not FastH3V2ConditionReceipt):
        raise ValueError("Frozen V2 LOW needs typed current recipe, stage and condition witnesses")
    recipe_value, stage_value = recipe.verify(), result.verify()
    attested, conditioned = attestation.verify(), condition.verify()
    context = stage_value["request"]["stage_context"]
    if (recipe_value.get("implementation") != implementation_identity()
            or stage_value.get("verified_recipe_completion") is not True
            or stage_value.get("portable_identity") is not True
            or context.get("stage") != "low_0_4"
            or attested.get("stage") != "low_0_4"
            or attested.get("segment_index") not in (0, 1)
            or attested.get("current_recipe_sha256") != recipe.sha256
            or attested.get("stage_receipt_sha256") != stage_value["receipt_sha256"]
            or attested.get("stage_request_sha256") != stage_value["request_sha256"]
            or attested.get("condition_receipt_sha256") != condition.sha256
            or conditioned.get("portable_identity") is not True
            or conditioned.get("implementation_sha256") != condition_implementation()):
        raise ValueError("Frozen V2 LOW witnesses are incomplete or from another execution")
    return recipe_value, stage_value, attested, conditioned


def _artifact(storage_root, artifact_path, artifact_sha256):
    root = storage._root(storage_root)
    path = storage._path(root, artifact_path)
    if path.name != "manifest.json":
        raise ValueError("Frozen V2 LOW must name a completed Stage manifest")
    digest = storage._digest(artifact_sha256)
    _output, _x0, _context, result, _report = storage.load_stage(
        root, artifact_path, digest, "low_0_4")
    return root, path, digest, result


def _sidecar_path(root, manifest_path):
    relative = (manifest_path.parent / SIDECAR).relative_to(root).as_posix()
    return storage._path(root, relative)


def save_current_low_bundle(current_recipe, low_stage_result, low_attestation,
                            low_condition_receipt, artifact_path, artifact_sha256,
                            storage_root):
    recipe, stage, attested, conditioned = _witnesses(
        current_recipe, low_stage_result, low_attestation, low_condition_receipt)
    root, manifest, digest, restored = _artifact(storage_root, artifact_path, artifact_sha256)
    if restored.verify() != stage:
        raise ValueError("Frozen V2 LOW artifact differs from the just-attested sampler result")
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
        "artifact_path": artifact_path, "artifact_sha256": digest,
        "state_sha256": storage.fingerprint_stage(root, artifact_path)[1],
        "stage_receipt_sha256": stage["receipt_sha256"],
        "current_recipe_sha256": current_recipe.sha256, "current_recipe": recipe,
        "low_attestation_sha256": low_attestation.sha256,
        "low_attestation": attested,
        "low_condition_receipt_sha256": low_condition_receipt.sha256,
        "low_condition_receipt": conditioned,
        "boundary": "Frozen LOW witnesses only; current HIGH and parent still need live checks"}
    encoded = canonical(payload).encode("utf-8")
    if len(encoded) > MAX_BUNDLE_JSON:
        raise ValueError("Frozen V2 LOW bundle exceeds the 64 MiB witness limit")
    compressed = None
    if len(encoded) > MAX_INLINE_JSON:
        compressed = zlib.compress(encoded, level=9)
        if len(compressed) > MAX_COMPRESSED:
            raise ValueError("Frozen V2 LOW compressed witness exceeds the 16 MiB limit")
        wrapper = {"schema": ENVELOPE_SCHEMA, "encoding": "zlib", "blob": BLOB,
                   "raw_bytes": len(encoded), "raw_sha256": hashlib.sha256(encoded).hexdigest(),
                   "compressed_bytes": len(compressed),
                   "blob_sha256": hashlib.sha256(compressed).hexdigest()}
        sidecar_bytes = canonical(wrapper).encode("utf-8")
    else:
        sidecar_bytes = encoded
    target = _sidecar_path(root, manifest)
    temporary = target.with_name(target.name + ".partial." + uuid.uuid4().hex)
    blob = target.with_name(BLOB)
    temporary_blob = blob.with_name(blob.name + ".partial." + uuid.uuid4().hex)
    with storage._lease(manifest.parent):
        if storage.file_sha(manifest) != digest:
            raise ValueError("Frozen V2 LOW manifest changed before bundle commit")
        if target.exists() or blob.exists() or blob.is_symlink():
            raise FileExistsError("This frozen LOW artifact already has a current recipe bundle")
        try:
            if compressed is not None:
                with temporary_blob.open("xb") as handle:
                    handle.write(compressed)
                    handle.flush()
                    os.fsync(handle.fileno())
            with temporary.open("xb") as handle:
                handle.write(sidecar_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if storage.file_sha(manifest) != digest:
                raise ValueError("Frozen V2 LOW manifest changed during bundle write")
            if compressed is not None:
                os.replace(temporary_blob, blob)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
            if temporary_blob.exists():
                temporary_blob.unlink()
            # The envelope is the commit marker. If it was not committed,
            # the blob was written by this transaction and is safe to remove.
            if compressed is not None and not target.exists() and blob.exists():
                blob.unlink()
    bundle_sha = storage.file_sha(target)
    report = {"schema": SCHEMA, "status": "frozen_low_bundle_saved",
              "artifact_path": artifact_path, "artifact_sha256": digest,
              "bundle_path": target.relative_to(root).as_posix(),
              "bundle_sha256": bundle_sha, "storage_format": (
                  "zlib-envelope" if compressed is not None else "inline-json"),
              "current_recipe_sha256": current_recipe.sha256,
              "boundary": payload["boundary"]}
    return report["bundle_path"], bundle_sha, canonical(report)


def load_current_low_bundle(artifact_path, artifact_sha256, low_stage_result, storage_root):
    if type(low_stage_result) is not StageResult:
        raise ValueError("Frozen V2 LOW bundle must accompany the actual StageLoad result")
    root, manifest, digest, restored = _artifact(storage_root, artifact_path, artifact_sha256)
    if low_stage_result.verify() != restored.verify():
        raise ValueError("Frozen V2 LOW StageLoad result differs from its selected artifact")
    target = _sidecar_path(root, manifest)
    with storage._lease(manifest.parent):
        payload = _read_payload(target)
        if (payload.get("schema") != SCHEMA
                or payload.get("implementation_sha256") != implementation_sha256()
                or payload.get("artifact_path") != artifact_path
                or payload.get("artifact_sha256") != digest
                or payload.get("state_sha256") != storage.file_sha(manifest.parent / "state.safetensors")
                or payload.get("stage_receipt_sha256") != restored.verify()["receipt_sha256"]
                or storage.file_sha(manifest) != digest):
            raise ValueError("Frozen V2 LOW bundle or selected artifact changed")
        bundle_sha = storage.file_sha(target)
    recipe = FastH3V2CurrentRecipe(canonical(payload["current_recipe"]),
                                   payload["current_recipe_sha256"])
    attestation = FastH3V2StageAttestation(canonical(payload["low_attestation"]),
                                          payload["low_attestation_sha256"])
    condition = FastH3V2ConditionReceipt(canonical(payload["low_condition_receipt"]),
                                         payload["low_condition_receipt_sha256"])
    recipe_value, _stage, _attested, _conditioned = _witnesses(
        recipe, restored, attestation, condition)
    model_id = recipe_value["model_pass1"]["sha256"][:16] + ":" + recipe_value["model_pass2"]["sha256"][:16]
    report = {"schema": SCHEMA, "status": "frozen_low_bundle_loaded",
              "artifact_path": artifact_path, "artifact_sha256": digest,
              "bundle_path": target.relative_to(root).as_posix(),
              "bundle_sha256": bundle_sha, "current_recipe_sha256": recipe.sha256,
              "boundary": payload["boundary"]}
    return recipe, recipe.sha256, model_id, canonical(report), attestation, condition, bundle_sha
