"""Explicitly selected H16 window freeze, never an automatic recipe cache.

The manifest's exact SHA selects a completed, source-bound window.  Loading it
does not claim that today's MODEL/conditioning would reproduce that window.
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
from .. import nodes_h16_chunked_pass2 as legacy_h16
from ..native_latent_checkpoint_advanced import _decode_metadata_value, _encode_metadata_value
from . import chunked_source, chunked_stages, h16_stages
from .chunked_stages import ChunkedPass2Result, _check_segment
from .h16_stages import AUDIO_OUTPUTS, H16Pass2Result
from .results import _input_identity, canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _json_file, _lease, _path, _root, file_sha


SCHEMA = "t8.modular-sampling.h16-frozen-window.v1"


def _implementation_identity():
    # The selected prior result is literal, but the contract that interprets
    # it at the next window must not silently change across a restart.
    modules = (legacy, legacy_h16, chunked_source, chunked_stages, h16_stages)
    files = (Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def verify_window(result, source_segment, spec, context, plan):
    _check_segment(source_segment, spec, context, plan)
    if type(result) is not H16Pass2Result or type(result.core_result) is not ChunkedPass2Result:
        raise ValueError("H16 freeze needs a completed typed window result")
    core = result.core_result
    if (result.audio_output not in AUDIO_OUTPUTS or core.plan_sha256 != spec.plan_sha256
            or core.source_identity != spec.source_identity or core.index != spec.index
            or core.count != spec.count or not 0 <= spec.index < spec.count):
        raise ValueError("H16 frozen window plan, source, index or audio policy differs")
    if (_input_identity(core.output_latent) != core.output_identity
            or _input_identity(result.output_latent) != result.output_identity
            or len(result.audio_chunks) != len(result.audio_chunk_identities)
            or any(_input_identity(chunk) != identity for chunk, identity in zip(
                result.audio_chunks, result.audio_chunk_identities, strict=True))):
        raise ValueError("H16 frozen window tensor content changed")
    core_video, core_audio = core.output_latent["samples"].unbind()
    output_video, output_audio = result.output_latent["samples"].unbind()
    if (core_audio is not context.original_audio
            or tuple(core_video.shape[:3]) != (1, 24, spec.end_token)
            or tuple(core_video.shape[-2:]) != (
                spec.target_height // legacy.VAE_DOWNSAMPLE,
                spec.target_width // legacy.VAE_DOWNSAMPLE)
            or not torch.equal(core_video, output_video)
            or tuple(output_audio.shape) != tuple(context.original_audio.shape)):
        raise ValueError("H16 frozen window has invalid AV handoff geometry or audio identity")
    if result.audio_output == "preserve_first_pass":
        if result.audio_chunks or output_audio is not context.original_audio:
            raise ValueError("H16 preserve policy cannot freeze refined audio")
    else:
        if (len(result.audio_chunks) != spec.index + 1
                or (spec.index + 1 < spec.count and output_audio is not context.original_audio)):
            raise ValueError("H16 refined audio history is incomplete or early")
        if any(type(chunk) is not torch.Tensor or chunk.ndim != 4
               or chunk.shape[:3] != context.original_audio.shape[:3]
               for chunk in result.audio_chunks):
            raise ValueError("H16 refined audio chunk has unexpected shape")
    return {"plan_sha256": spec.plan_sha256, "source_identity": spec.source_identity,
            "index": spec.index, "count": spec.count, "audio_output": result.audio_output,
            "core_identity": core.output_identity, "output_identity": result.output_identity,
            "audio_chunk_identities": list(result.audio_chunk_identities)}


def _state(result):
    core_video, _ = result.core_result.output_latent["samples"].unbind()
    _video, output_audio = result.output_latent["samples"].unbind()
    return {"core_video": core_video, "output_audio": output_audio,
            "audio_chunks": result.audio_chunks}


def _rebuild(state, binding, context):
    if type(state) is not dict or set(state) != {"core_video", "output_audio", "audio_chunks"}:
        raise ValueError("H16 frozen window has an unknown tensor inventory")
    source_video, _ = context.source_latent["samples"].unbind()
    video = state["core_video"].to(device=source_video.device)
    chunks = tuple(chunk.to(device=context.original_audio.device) for chunk in state["audio_chunks"])
    core_output = {"samples": comfy.nested_tensor.NestedTensor((
        video, context.original_audio))}
    if binding["audio_output"] == "preserve_first_pass" or binding["index"] + 1 < binding["count"]:
        if _input_identity(state["output_audio"]) != _input_identity(context.original_audio):
            raise ValueError("H16 frozen intermediate audio differs from its current source")
        output_audio = context.original_audio
    else:
        output_audio = state["output_audio"].to(device=context.original_audio.device)
    output = {"samples": comfy.nested_tensor.NestedTensor((video, output_audio))}
    core = ChunkedPass2Result(binding["plan_sha256"], binding["source_identity"],
                              binding["index"], binding["count"], core_output,
                              binding["core_identity"])
    return H16Pass2Result(core, binding["audio_output"], chunks,
                          tuple(binding["audio_chunk_identities"]), output,
                          binding["output_identity"])


def save_window(result, source_segment, spec, context, plan, storage_root,
                prefix="h16-window"):
    binding = verify_window(result, source_segment, spec, context, plan)
    root = _root(storage_root, create=True)
    prefix_path = _path(root, prefix)
    directory = prefix_path.with_name(prefix_path.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        tensors = {}
        encoded = _encode_metadata_value(_state(result), tensors, path="h16.window")
        payload = canonical({"schema": SCHEMA, "binding": binding, "state": encoded})
        if len(payload.encode("utf-8")) > MAX_JSON:
            raise ValueError("H16 frozen metadata exceeds its limit")
        if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("H16 frozen tensors exceed their limit")
        snapshot = _decode_metadata_value(encoded, tensors, set(), path="h16.window")
        verify_window(_rebuild(snapshot, binding, context), source_segment, spec, context, plan)
        verify_window(result, source_segment, spec, context, plan)
        partial = directory / "state.safetensors.partial"
        state = directory / "state.safetensors"
        save_file(tensors, str(partial), metadata={"h16_json": payload})
        with partial.open("rb+") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_bytes <= MAX_JSON:
                raise ValueError("H16 tensor header exceeds its limit")
            os.fsync(handle.fileno())
        os.replace(partial, state)
        verify_window(result, source_segment, spec, context, plan)
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


def load_window(source_segment, spec, context, plan, storage_root, artifact_path,
                artifact_sha256):
    _check_segment(source_segment, spec, context, plan)
    root = _root(storage_root)
    manifest_path = _path(root, artifact_path)
    if manifest_path.name != "manifest.json" or not manifest_path.parent.is_dir():
        raise ValueError("Select an existing completed H16 manifest.json")
    expected_sha = _digest(artifact_sha256)
    with _lease(manifest_path.parent):
        if (not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_JSON
                or file_sha(manifest_path) != expected_sha):
            raise ValueError("H16 frozen manifest SHA mismatch or missing completion")
        manifest = _json_file(manifest_path)
        if (type(manifest) is not dict or set(manifest) != {
                "schema", "state_file", "state_sha256", "state_bytes", "binding",
                "implementation", "automatic_cache_reuse", "execution_identity_certified"}
                or manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors"
                or manifest["implementation"] != _implementation_identity()
                or manifest["automatic_cache_reuse"] is not False
                or manifest["execution_identity_certified"] is not False):
            raise ValueError("Unknown or incomplete H16 frozen manifest")
        binding = manifest["binding"]
        if (type(binding) is not dict or set(binding) != {
                "plan_sha256", "source_identity", "index", "count", "audio_output",
                "core_identity", "output_identity", "audio_chunk_identities"}
                or binding["plan_sha256"] != spec.plan_sha256
                or binding["source_identity"] != spec.source_identity
                or binding["index"] != spec.index or binding["count"] != spec.count):
            raise ValueError("H16 frozen window does not match current source, plan or index")
        state_path = _path(root, (manifest_path.parent / "state.safetensors").relative_to(root).as_posix())
        if (not state_path.is_file() or type(manifest["state_bytes"]) is not int
                or not 8 < manifest["state_bytes"] <= MAX_FILE
                or state_path.stat().st_size != manifest["state_bytes"]
                or file_sha(state_path) != _digest(manifest["state_sha256"])):
            raise ValueError("H16 frozen tensor size or SHA mismatch")
        with state_path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_bytes <= MAX_JSON or header_bytes > state_path.stat().st_size - 8:
            raise ValueError("H16 frozen tensor header exceeds its limit")
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or set(metadata) != {"h16_json"}:
                raise ValueError("H16 frozen tensor metadata is missing or unknown")
            payload = json.loads(metadata["h16_json"])
            if (type(payload) is not dict or set(payload) != {"schema", "binding", "state"}
                    or payload["schema"] != SCHEMA or payload["binding"] != binding):
                raise ValueError("H16 frozen tensor payload differs from manifest")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        used = set()
        state = _decode_metadata_value(payload["state"], tensors, used, path="h16.window")
        if used != set(tensors):
            raise ValueError("H16 frozen tensor inventory differs from descriptor")
        result = _rebuild(state, binding, context)
        if verify_window(result, source_segment, spec, context, plan) != binding:
            raise ValueError("H16 frozen result binding differs from manifest")
        if file_sha(state_path) != manifest["state_sha256"] or file_sha(manifest_path) != expected_sha:
            raise ValueError("H16 frozen artifact changed while loading")
    report = {"schema": SCHEMA, "status": "explicit_frozen_h16_window_loaded",
              "index": spec.index, "artifact_sha256": expected_sha,
              "automatic_cache_reuse": False, "execution_identity_certified": False,
              "boundary": "Exact selected prior result; current MODEL/conditioning equivalence is not claimed."}
    return result.output_latent, result, canonical(report)
