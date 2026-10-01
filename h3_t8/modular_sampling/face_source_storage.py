"""Explicit data-only Multi-Face inputs paired to an independently saved stage.

SAM inference and VAE encoding are not rerun by Load. Their original outputs
are retained, not normalized or granted a new completion/quality certificate.
The current complete parent RGB/audio and exact StageResult remain mandatory.
"""
import hashlib
import json
import os
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..core import validate_audio
from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from .face_stage import audit_multiface_face_stage
from .results import StageResult, _input_identity, canonical
from .storage import _root, _path, _lease, _digest, _json_file, file_sha, MAX_JSON, MAX_FILE

SCHEMA = "t8.modular-sampling.multiface-source.v1"
METADATA_KEY = "t8_multiface_source"
FIELDS = {"face_plan", "source_frames", "av_latent", "parent_identity", "audio_identity", "receipt_sha256"}


def validate_source(values, stage_result, parent_frames, source_audio):
    if type(values) is not dict or set(values) != FIELDS or type(stage_result) is not StageResult:
        raise ValueError("Multi-Face source needs exact input data and an independent StageResult")
    validate_audio(source_audio, "original source_audio")
    receipt = stage_result.verify()
    if (receipt["verified_recipe_completion"] is not True or receipt["portable_identity"] is not True
            or values["receipt_sha256"] != receipt["receipt_sha256"]):
        raise ValueError("Multi-Face source belongs to another or unverified completed stage")
    if (values["parent_identity"] != _input_identity(parent_frames)
            or values["audio_identity"] != _input_identity(source_audio)):
        raise ValueError("Current complete parent RGB or source audio differs from frozen input")
    audit_multiface_face_stage(stage_result, values["face_plan"], values["source_frames"],
                              parent_frames, values["av_latent"])
    return _input_identity(values)


def source_values(stage_result, face_plan, source_frames, parent_frames, av_latent, source_audio):
    if type(stage_result) is not StageResult:
        raise ValueError("Multi-Face source needs a sampler-produced StageResult")
    values = dict(face_plan=face_plan, source_frames=source_frames, av_latent=av_latent,
                  parent_identity=_input_identity(parent_frames), audio_identity=_input_identity(source_audio),
                  receipt_sha256=stage_result.verify()["receipt_sha256"])
    validate_source(values, stage_result, parent_frames, source_audio)
    return values


def _check_tensors(tensors):
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_FILE - MAX_JSON:
        raise ValueError("Multi-Face input exceeds the 20 GiB limit")
    if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError("Multi-Face input contains non-finite tensors")


def report(status, digest, path=""):
    return canonical(dict(schema=SCHEMA, status=status, artifact_path=path, artifact_sha256=digest, data_only=True,
        sampling_executed=False, automatic_cache_reuse=False, quality_accepted=False,
        boundary="Exact explicit original Face plan/source/AV only. Independent completed StageResult and "
                 "current complete parent RGB/audio are still verified; no SAM/VAE recomputation or approval."))


def save_source(values, stage_result, parent_frames, source_audio, storage_root,
                prefix="multiface_source", *, interrupt=lambda: None):
    interrupt()
    identity = validate_source(values, stage_result, parent_frames, source_audio)
    tensors = {}
    encoded = _encode_metadata_value(values, tensors, path="multiface_source")
    _check_tensors(tensors)
    used = set()
    snapshot = _decode_metadata_value(encoded, tensors, used, path="multiface_source")
    if (used != set(tensors) or validate_source(snapshot, stage_result, parent_frames, source_audio) != identity
            or validate_source(values, stage_result, parent_frames, source_audio) != identity):
        raise ValueError("Multi-Face input changed while snapshotting")
    payload = canonical(dict(schema=SCHEMA, state=encoded, identity=identity))
    if len(payload.encode("utf8")) > MAX_JSON:
        raise ValueError("Multi-Face source metadata exceeds the 4 MiB limit")
    root = _root(storage_root, create=True)
    selected = _path(root, prefix)
    directory = selected.with_name(selected.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        partial, state = directory / "state.safetensors.partial", directory / "state.safetensors"
        save_file(tensors, str(partial), metadata={METADATA_KEY: payload})
        with partial.open("rb+") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < size <= MAX_JSON:
                raise ValueError("Multi-Face tensor header exceeds its bound")
            os.fsync(handle.fileno())
        interrupt()
        if validate_source(values, stage_result, parent_frames, source_audio) != identity:
            raise ValueError("Multi-Face input changed during persistence")
        os.replace(partial, state)
        manifest = dict(schema=SCHEMA, state_file=state.name, state_bytes=state.stat().st_size,
                        state_sha256=file_sha(state), payload_sha256=hashlib.sha256(payload.encode("utf8")).hexdigest(),
                        data_only=True)
        pending, committed = directory / "manifest.json.partial", directory / "manifest.json"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf8"))
            handle.flush()
            os.fsync(handle.fileno())
        interrupt()
        if validate_source(values, stage_result, parent_frames, source_audio) != identity:
            raise ValueError("Multi-Face input changed before manifest commit")
        os.replace(pending, committed)
    digest = file_sha(committed)
    relative = committed.relative_to(root).as_posix()
    return relative, digest, report("explicit_multiface_source_saved", digest, relative)


def load_source(storage_root, artifact_path, artifact_sha256, stage_result, parent_frames,
                source_audio, *, interrupt=lambda: None):
    interrupt()
    root = _root(storage_root)
    path = _path(root, artifact_path)
    expected = _digest(artifact_sha256)
    if path.name != "manifest.json" or not path.parent.is_dir():
        raise ValueError("Select a committed Multi-Face source manifest.json")
    with _lease(path.parent):
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON or file_sha(path) != expected:
            raise ValueError("Multi-Face source manifest SHA mismatch or missing completion")
        manifest = _json_file(path)
        if (set(manifest) != {"schema", "state_file", "state_bytes", "state_sha256", "payload_sha256", "data_only"}
                or manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors"
                or manifest["data_only"] is not True or type(manifest["state_bytes"]) is not int
                or not 8 < manifest["state_bytes"] <= MAX_FILE):
            raise ValueError("Unknown Multi-Face source manifest contract")
        state = _path(root, (path.parent / manifest["state_file"]).relative_to(root).as_posix())
        if (not state.is_file() or state.stat().st_size != manifest["state_bytes"]
                or file_sha(state) != _digest(manifest["state_sha256"])):
            raise ValueError("Multi-Face source tensor size or SHA differs")
        with state.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size <= min(MAX_JSON, state.stat().st_size - 8):
            raise ValueError("Multi-Face source tensor header exceeds its bound")
        interrupt()
        with safe_open(str(state), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {METADATA_KEY}:
                raise ValueError("Unknown Multi-Face source metadata")
            raw = metadata[METADATA_KEY]
            if hashlib.sha256(raw.encode("utf8")).hexdigest() != _digest(manifest["payload_sha256"]):
                raise ValueError("Multi-Face source payload SHA differs")
            payload = json.loads(raw)
            if set(payload) != {"schema", "state", "identity"} or payload["schema"] != SCHEMA:
                raise ValueError("Unknown Multi-Face source payload contract")
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        _check_tensors(tensors)
        used = set()
        values = _decode_metadata_value(payload["state"], tensors, used, path="multiface_source")
        if (used != set(tensors)
                or validate_source(values, stage_result, parent_frames, source_audio) != payload["identity"]):
            raise ValueError("Multi-Face source inventory or content identity differs")
        interrupt()
        if file_sha(path) != expected or file_sha(state) != manifest["state_sha256"]:
            raise ValueError("Multi-Face source files changed while reading")
    return values, report("explicit_multiface_source_loaded", expected, artifact_path)
