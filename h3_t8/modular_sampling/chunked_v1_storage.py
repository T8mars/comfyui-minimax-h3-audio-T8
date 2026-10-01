"""Explicitly frozen Chunked v1 PASS2 segment, never a recipe cache hit.

The selected cumulative video is literal. Loading does not claim that the
current MODEL, CONDITIONING or stage effects reproduce the frozen segment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import uuid

import comfy.nested_tensor
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from ..native_latent_checkpoint_advanced import _decode_metadata_value, _encode_metadata_value
from . import chunked_effects, chunked_source, chunked_stages, chunked_v1_relay
from .chunked_stages import ChunkedPass2Result, _check_segment
from .results import _input_identity, canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _json_file, _lease, _path, _root, file_sha


SCHEMA = "t8.modular-sampling.chunked-v1-frozen-segment.v1"
NATIVE_CHECKPOINT_PROVENANCE = "t8_native_latent_checkpoint"


def _implementation_identity():
    modules = (legacy, chunked_source, chunked_stages, chunked_effects, chunked_v1_relay)
    files = (Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _same_source_content(saved, current):
    """Only native checkpoint provenance may differ; all sampled AV/masks stay exact."""
    if type(saved) is not dict or type(current) is not dict:
        return False
    saved_content = {key: value for key, value in saved.items()
                     if key != NATIVE_CHECKPOINT_PROVENANCE}
    current_content = {key: value for key, value in current.items()
                       if key != NATIVE_CHECKPOINT_PROVENANCE}
    return "samples" in saved_content and saved_content == current_content


def verify_segment(result, source_segment, spec, context, plan):
    _check_segment(source_segment, spec, context, plan)
    if plan["schema"] != legacy.PLAN_SCHEMA_V1:
        raise ValueError("Chunked v1 freeze requires the v1 per-segment contract")
    if type(result) is not ChunkedPass2Result:
        raise ValueError("Chunked v1 freeze needs a completed typed PASS2 result")
    if type(result.output_latent) is not dict or set(result.output_latent) != {"samples"}:
        raise ValueError("Chunked v1 frozen result has unsupported latent fields")
    if (result.plan_sha256 != spec.plan_sha256
            or result.source_identity != spec.source_identity
            or result.index != spec.index or result.count != spec.count
            or not 0 <= spec.index < spec.count
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError("Chunked v1 frozen segment plan, source, index or content differs")
    samples = result.output_latent["samples"]
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError("Chunked v1 frozen segment needs native cumulative AV")
    video, audio = samples.unbind()
    if (type(video) is not torch.Tensor or type(audio) is not torch.Tensor
            or tuple(video.shape) != (
                1, 24, spec.end_token,
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE,
            ) or audio is not context.original_audio
            or not torch.isfinite(video).all() or not torch.isfinite(audio).all()):
        raise ValueError("Chunked v1 frozen segment has invalid video or original audio")
    return {"plan_sha256": spec.plan_sha256, "source_identity": spec.source_identity,
            "slice_identity": spec.slice_identity, "index": spec.index, "count": spec.count,
            "output_identity": result.output_identity}


def _rebuild(video, binding, context):
    source_video, _ = context.source_latent["samples"].unbind()
    output = {"samples": comfy.nested_tensor.NestedTensor((
        video.to(device=source_video.device), context.original_audio,
    ))}
    return ChunkedPass2Result(binding["plan_sha256"], binding["source_identity"],
                              binding["index"], binding["count"], output,
                              binding["output_identity"])


def save_segment(result, source_segment, spec, context, plan, storage_root,
                 prefix="chunked-v1-segment"):
    binding = verify_segment(result, source_segment, spec, context, plan)
    root = _root(storage_root, create=True)
    prefix_path = _path(root, prefix)
    directory = prefix_path.with_name(prefix_path.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        video, _audio = result.output_latent["samples"].unbind()
        tensors = {}
        encoded = _encode_metadata_value(video, tensors, path="chunked.v1.video")
        payload = canonical({"schema": SCHEMA, "binding": binding, "video": encoded})
        if len(payload.encode("utf-8")) > MAX_JSON:
            raise ValueError("Chunked v1 frozen metadata exceeds its limit")
        if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("Chunked v1 frozen tensors exceed their limit")
        snapshot = _decode_metadata_value(encoded, tensors, set(), path="chunked.v1.video")
        verify_segment(_rebuild(snapshot, binding, context), source_segment, spec, context, plan)
        verify_segment(result, source_segment, spec, context, plan)
        partial = directory / "state.safetensors.partial"
        state = directory / "state.safetensors"
        save_file(tensors, str(partial), metadata={"chunked_v1_json": payload})
        with partial.open("rb+") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_bytes <= MAX_JSON:
                raise ValueError("Chunked v1 tensor header exceeds its limit")
            os.fsync(handle.fileno())
        os.replace(partial, state)
        verify_segment(result, source_segment, spec, context, plan)
        manifest = {"schema": SCHEMA, "state_file": state.name,
                    "state_sha256": file_sha(state), "state_bytes": state.stat().st_size,
                    "binding": binding, "implementation": _implementation_identity(),
                    "automatic_cache_reuse": False,
                    "execution_identity_certified": False}
        pending = directory / "manifest.json.partial"
        manifest_path = directory / "manifest.json"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, manifest_path)
    path = manifest_path.relative_to(root).as_posix()
    digest = file_sha(manifest_path)
    return result.output_latent, result, path, digest, canonical(manifest)


def load_segment(source_segment, spec, context, plan, storage_root,
                 artifact_path, artifact_sha256):
    _check_segment(source_segment, spec, context, plan)
    if plan["schema"] != legacy.PLAN_SCHEMA_V1:
        raise ValueError("Chunked v1 load requires the v1 per-segment contract")
    root = _root(storage_root)
    manifest_path = _path(root, artifact_path)
    if manifest_path.name != "manifest.json" or not manifest_path.parent.is_dir():
        raise ValueError("Select an existing completed Chunked v1 manifest.json")
    expected_sha = _digest(artifact_sha256)
    with _lease(manifest_path.parent):
        if (not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_JSON
                or file_sha(manifest_path) != expected_sha):
            raise ValueError("Chunked v1 frozen manifest SHA mismatch or missing completion")
        manifest = _json_file(manifest_path)
        if (type(manifest) is not dict or set(manifest) != {
                "schema", "state_file", "state_sha256", "state_bytes", "binding",
                "implementation", "automatic_cache_reuse", "execution_identity_certified"}
                or manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors"
                or manifest["implementation"] != _implementation_identity()
                or manifest["automatic_cache_reuse"] is not False
                or manifest["execution_identity_certified"] is not False):
            raise ValueError("Unknown or incomplete Chunked v1 frozen manifest")
        binding = manifest["binding"]
        if (type(binding) is not dict or set(binding) != {
                "plan_sha256", "source_identity", "slice_identity", "index", "count",
                "output_identity"}
                or binding["plan_sha256"] != spec.plan_sha256
                or not _same_source_content(binding["source_identity"], spec.source_identity)
                or binding["slice_identity"] != spec.slice_identity
                or binding["index"] != spec.index or binding["count"] != spec.count):
            raise ValueError("Chunked v1 frozen segment does not match current source, plan or index")
        state_path = _path(root, (manifest_path.parent / "state.safetensors").relative_to(root).as_posix())
        if (not state_path.is_file() or type(manifest["state_bytes"]) is not int
                or not 8 < manifest["state_bytes"] <= MAX_FILE
                or state_path.stat().st_size != manifest["state_bytes"]
                or file_sha(state_path) != _digest(manifest["state_sha256"])):
            raise ValueError("Chunked v1 frozen tensor size or SHA mismatch")
        with state_path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_bytes <= MAX_JSON or header_bytes > state_path.stat().st_size - 8:
            raise ValueError("Chunked v1 frozen tensor header exceeds its limit")
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or set(metadata) != {"chunked_v1_json"}:
                raise ValueError("Chunked v1 frozen tensor metadata is missing or unknown")
            payload = json.loads(metadata["chunked_v1_json"])
            if (type(payload) is not dict or set(payload) != {"schema", "binding", "video"}
                    or payload["schema"] != SCHEMA or payload["binding"] != binding):
                raise ValueError("Chunked v1 frozen tensor payload differs from manifest")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        used = set()
        video = _decode_metadata_value(payload["video"], tensors, used, path="chunked.v1.video")
        if used != set(tensors) or type(video) is not torch.Tensor:
            raise ValueError("Chunked v1 frozen tensor inventory differs from descriptor")
        current_binding = {**binding, "source_identity": spec.source_identity}
        result = _rebuild(video, current_binding, context)
        if verify_segment(result, source_segment, spec, context, plan) != current_binding:
            raise ValueError("Chunked v1 frozen result binding differs from manifest")
        if file_sha(state_path) != manifest["state_sha256"] or file_sha(manifest_path) != expected_sha:
            raise ValueError("Chunked v1 frozen artifact changed while loading")
    report = {"schema": SCHEMA, "status": "explicit_frozen_chunked_v1_segment_loaded",
              "index": spec.index, "artifact_sha256": expected_sha,
              "native_checkpoint_provenance_rebound": (
                  binding["source_identity"] != spec.source_identity),
              "automatic_cache_reuse": False, "execution_identity_certified": False,
              "boundary": "Exact selected prior result; current MODEL/conditioning equivalence is not claimed."}
    return result.output_latent, result, canonical(report)
