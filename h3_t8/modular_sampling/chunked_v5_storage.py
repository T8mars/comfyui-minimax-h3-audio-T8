"""Explicit, source-bound Chunked v5 joint AV window freeze.

Loading selects a literal completed window. It never asserts that today's
MODEL or conditioning would reproduce that window or auto-hits a cache.
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

from .. import chunked_two_pass_parity as parity
from .. import chunked_two_pass_upscale_advanced as legacy
from ..native_latent_checkpoint_advanced import _decode_metadata_value, _encode_metadata_value
from . import chunked_v5, chunked_v5_effects, chunked_v5_relay
from . import chunked_v5_storage_compat as compat
from .chunked_v5 import StandardPrepared, StandardWindowResult, _check_lift
from .results import _input_identity, canonical
from .storage import MAX_FILE, MAX_JSON, _digest, _json_file, _lease, _path, _root, file_sha


SCHEMA = "t8.modular-sampling.chunked-v5-frozen-window.v1"


def _implementation_identity():
    modules = (parity, legacy, chunked_v5, chunked_v5_effects, chunked_v5_relay, compat)
    paths = (Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _stable_upscale_report(report):
    """Bind the lift contract, not per-process memory and model-cache telemetry."""
    if type(report) is not dict:
        raise ValueError("v5 learned lift report has an unknown shape")
    stable = {key: value for key, value in report.items() if key not in {
        "memory_before", "memory_after_release", "gpu_weights_released", "cpu_cache_cleared",
    }}
    model = stable.get("model")
    if type(model) is dict:
        stable["model"] = {key: value for key, value in model.items() if key != "cache_hit"}
    geometry = stable.get("geometry")
    if type(geometry) is dict:
        stable["geometry"] = {key: value for key, value in geometry.items() if key != "memory_warning"}
    return stable


def _prepared_sha(prepared):
    return hashlib.sha256(canonical({
        "video_noise": _input_identity(prepared.video_noise),
        "audio_noise": _input_identity(prepared.audio_noise),
        "noise_report": _input_identity(prepared.noise_report),
        "noise_seed": prepared.noise_seed,
        "upscale_report": _input_identity(_stable_upscale_report(prepared.lift.upscale_report)),
    }).encode("utf-8")).hexdigest()


def verify_window(result, source, lifted, prepared, plan):
    if type(prepared) is not StandardPrepared or type(result) is not StandardWindowResult:
        raise ValueError("v5 freeze needs a completed typed window and preparation")
    receipt = prepared.lift
    _check_lift(source, lifted, receipt, plan)
    if (result.plan_sha256 != receipt.plan_sha256 or
            result.source_identity != receipt.source_identity or
            result.lifted_identity != receipt.lifted_identity or
            result.count != len(receipt.segments) or
            type(result.index) is not int or not 0 <= result.index < result.count):
        raise ValueError("v5 frozen window source, plan or index differs")
    output = result.output_latent
    expected_keys = {"samples", *(key for key in ("batch_index", "type") if key in source)}
    if type(output) is not dict or set(output) != expected_keys:
        raise ValueError("v5 frozen window has unexpected AV metadata")
    for key in expected_keys - {"samples"}:
        if _input_identity(output[key]) != _input_identity(source[key]):
            raise ValueError("v5 frozen window source metadata differs")
    video, audio = parity._parts(output["samples"], "v5 frozen cumulative AV")
    high_video, high_audio = parity._parts(lifted["samples"], "v5 global lift")
    end_video = receipt.segments[result.index][2]
    end_audio = receipt.audio_bounds[result.index][1]
    if (tuple(video.shape) != (*high_video.shape[:2], end_video, *high_video.shape[-2:]) or
            tuple(audio.shape) != (*high_audio.shape[:-1], end_audio) or
            tuple(prepared.video_noise.shape) != tuple(high_video.shape) or
            tuple(prepared.audio_noise.shape) != tuple(high_audio.shape) or
            _input_identity(output) != result.output_identity):
        raise ValueError("v5 frozen window AV/noise geometry or content differs")
    return {"plan_sha256": receipt.plan_sha256,
            "source_identity": receipt.source_identity,
            "lifted_identity": receipt.lifted_identity,
            "prepared_sha256": _prepared_sha(prepared),
            "index": result.index, "count": result.count,
            "output_identity": result.output_identity}


def _state(result):
    video, audio = result.output_latent["samples"].unbind()
    return {"video": video, "audio": audio,
            "metadata": {key: result.output_latent[key]
                         for key in ("batch_index", "type") if key in result.output_latent}}


def _rebuild(state, binding, lifted):
    if (type(state) is not dict or set(state) != {"video", "audio", "metadata"} or
            type(state["metadata"]) is not dict or
            not set(state["metadata"]) <= {"batch_index", "type"} or
            type(state["video"]) is not torch.Tensor or
            type(state["audio"]) is not torch.Tensor):
        raise ValueError("v5 frozen window has an unknown tensor inventory")
    high_video, high_audio = parity._parts(lifted["samples"], "v5 global lift")
    output = dict(state["metadata"])
    output["samples"] = comfy.nested_tensor.NestedTensor((
        state["video"].to(device=high_video.device),
        state["audio"].to(device=high_audio.device),
    ))
    return StandardWindowResult(
        binding["plan_sha256"], binding["source_identity"],
        binding["lifted_identity"], binding["index"], binding["count"],
        output, binding["output_identity"],
    )


def save_window(result, source, lifted, prepared, plan, storage_root,
                prefix="v5-window"):
    binding = verify_window(result, source, lifted, prepared, plan)
    root = _root(storage_root, create=True)
    prefix_path = _path(root, prefix)
    directory = prefix_path.with_name(prefix_path.name + "-" + uuid.uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _lease(directory):
        tensors = {}
        encoded = _encode_metadata_value(_state(result), tensors, path="chunked_v5.window")
        payload = canonical({"schema": SCHEMA, "binding": binding, "state": encoded})
        if len(payload.encode("utf-8")) > MAX_JSON or sum(
                item.numel() * item.element_size() for item in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("v5 frozen window exceeds metadata or tensor size limit")
        snapshot = _decode_metadata_value(encoded, tensors, set(), path="chunked_v5.window")
        if verify_window(_rebuild(snapshot, binding, lifted), source, lifted, prepared, plan) != binding:
            raise ValueError("v5 frozen window serialization changed content")
        verify_window(result, source, lifted, prepared, plan)
        partial = directory / "state.safetensors.partial"
        state = directory / "state.safetensors"
        save_file(tensors, str(partial), metadata={"chunked_v5_json": payload})
        with partial.open("rb+") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_bytes <= MAX_JSON:
                raise ValueError("v5 frozen tensor header exceeds its limit")
            os.fsync(handle.fileno())
        os.replace(partial, state)
        verify_window(result, source, lifted, prepared, plan)
        manifest = {"schema": SCHEMA, "state_file": state.name,
                    "state_sha256": file_sha(state), "state_bytes": state.stat().st_size,
                    "binding": binding, "implementation": _implementation_identity(),
                    "automatic_cache_reuse": False, "execution_identity_certified": False}
        pending = directory / "manifest.json.partial"
        manifest_path = directory / "manifest.json"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, manifest_path)
    return (result.output_latent, result, manifest_path.relative_to(root).as_posix(),
            file_sha(manifest_path), canonical(manifest))


def load_window(source, lifted, prepared, plan, storage_root, artifact_path,
                artifact_sha256, expected_index):
    if type(prepared) is not StandardPrepared:
        raise ValueError("v5 load needs current global PASS2 preparation")
    _check_lift(source, lifted, prepared.lift, plan)
    if (type(expected_index) is not int or
            not 0 <= expected_index < len(prepared.lift.segments)):
        raise ValueError("Expected v5 frozen window index is outside the current plan")
    root = _root(storage_root)
    manifest_path = _path(root, artifact_path)
    if manifest_path.name != "manifest.json" or not manifest_path.parent.is_dir():
        raise ValueError("Select a completed v5 manifest.json")
    expected_sha = _digest(artifact_sha256)
    with _lease(manifest_path.parent):
        if (not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_JSON or
                file_sha(manifest_path) != expected_sha):
            raise ValueError("v5 frozen manifest SHA mismatch or missing completion")
        manifest = _json_file(manifest_path)
        fields = {"schema", "state_file", "state_sha256", "state_bytes", "binding",
                  "implementation", "automatic_cache_reuse", "execution_identity_certified"}
        if (type(manifest) is not dict or set(manifest) != fields or
                manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors" or
                manifest["automatic_cache_reuse"] is not False or
                manifest["execution_identity_certified"] is not False):
            raise ValueError("Unknown, stale or incomplete v5 frozen manifest")
        profile = compat.implementation_profile(manifest["implementation"], _implementation_identity())
        binding = manifest["binding"]
        expected_binding = {"plan_sha256": prepared.lift.plan_sha256,
                            "source_identity": prepared.lift.source_identity,
                            "lifted_identity": prepared.lift.lifted_identity,
                            "prepared_sha256": compat.prepared_sha(prepared, profile, _prepared_sha(prepared))}
        if (type(binding) is not dict or set(binding) != {
                *expected_binding, "index", "count", "output_identity"} or
                any(binding[key] != value for key, value in expected_binding.items()) or
                type(binding["index"]) is not int or
                type(binding["count"]) is not int or
                binding["count"] != len(prepared.lift.segments) or
                binding["index"] != expected_index):
            raise ValueError("v5 frozen window does not match current source, plan or noise")
        state_path = _path(root, (manifest_path.parent / "state.safetensors").relative_to(root).as_posix())
        if (not state_path.is_file() or type(manifest["state_bytes"]) is not int or
                not 8 < manifest["state_bytes"] <= MAX_FILE or
                state_path.stat().st_size != manifest["state_bytes"] or
                file_sha(state_path) != _digest(manifest["state_sha256"])):
            raise ValueError("v5 frozen tensor size or SHA mismatch")
        with state_path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_bytes <= MAX_JSON or header_bytes > state_path.stat().st_size - 8:
            raise ValueError("v5 frozen tensor header exceeds its limit")
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or set(metadata) != {"chunked_v5_json"}:
                raise ValueError("v5 frozen tensor metadata is missing or unknown")
            payload = json.loads(metadata["chunked_v5_json"])
            if (type(payload) is not dict or set(payload) != {"schema", "binding", "state"} or
                    payload["schema"] != SCHEMA or payload["binding"] != binding):
                raise ValueError("v5 frozen tensor payload differs from manifest")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        used = set()
        state = _decode_metadata_value(payload["state"], tensors, used, path="chunked_v5.window")
        if used != set(tensors):
            raise ValueError("v5 frozen tensor inventory differs from descriptor")
        result = _rebuild(state, binding, lifted)
        current_binding = {**binding, "prepared_sha256": _prepared_sha(prepared)}
        if verify_window(result, source, lifted, prepared, plan) != current_binding:
            raise ValueError("v5 frozen result binding differs from manifest")
        if file_sha(state_path) != manifest["state_sha256"] or file_sha(manifest_path) != expected_sha:
            raise ValueError("v5 frozen artifact changed while loading")
    report = {"schema": SCHEMA, "status": "explicit_frozen_chunked_v5_window_loaded",
              "index": result.index, "artifact_sha256": expected_sha,
              "automatic_cache_reuse": False, "execution_identity_certified": False,
              "implementation_profile": profile,
              "source_storage_sha256": manifest["implementation"][compat.STORAGE_SOURCE],
              "historical_literal_format": profile != compat.CURRENT_PROFILE,
              "boundary": "Exact selected prior result; current MODEL/conditioning equivalence is not claimed."}
    return result.output_latent, result, canonical(report)
