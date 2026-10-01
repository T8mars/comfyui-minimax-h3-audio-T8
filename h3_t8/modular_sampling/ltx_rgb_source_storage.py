"""Explicit data-only RGB/LTX handoff; never a completed sampler receipt.

Freeze selected source RGB/audio, prepared RGB and the lifted LTX input. Rebuild
MODEL, conditioning, noise and the old Stage Bind after loading. No executables,
automatic cache lookup, hidden encoder/upscaler or implicit sampler are stored.
"""
import hashlib
import json
import math
import os
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from .ltx_rgb_stage import PREP_PIPELINE, _tensor, _report
from .results import _input_identity, canonical
from .storage import _root, _path, _lease, _digest, _json_file, file_sha, MAX_JSON, MAX_FILE

SCHEMA = "t8.modular-sampling.ltx-rgb-source.v1"
FIELDS = {"source_frames", "source_audio", "prepared_frames", "prep_report_json", "ltx_latent", "bit_depth"}
METADATA_KEY = "t8_ltx_rgb_source"


def validate_source(values):
    if not isinstance(values, dict) or set(values) != FIELDS:
        raise ValueError("LTX RGB source requires the exact six data inputs")
    if type(values["bit_depth"]) is not int or values["bit_depth"] not in (8, 10):
        raise ValueError("LTX RGB source requires an explicit 8 or 10 bit video depth")
    source = _tensor(values["source_frames"], "H3 source frames", 4)
    prepared = _tensor(values["prepared_frames"], "LTX encoder frames", 4)
    if source.shape[-1] != 3 or prepared.shape[-1] != 3 or min(*source.shape, *prepared.shape) < 1:
        raise ValueError("LTX RGB source requires nonempty RGB tensors")
    prep = _report(values["prep_report_json"], "RGB preparation report")
    frames, height, width = map(int, prepared.shape[:3])
    if (prep.get("status") != "prepared" or prep.get("pipeline") != PREP_PIPELINE
            or prep.get("audio_policy") != "bypass_stage2_and_preserve_original_h3_audio_object"
            or prep.get("source") != {"width": int(source.shape[2]), "height": int(source.shape[1]),
                                      "frames": int(source.shape[0])}
            or prep.get("ltx_encoder_input") != {"width": width, "height": height, "frames": frames,
                                               "resize": "aspect_preserving_center_crop"}
            or prep.get("target") != {"width": 2 * width, "height": 2 * height, "frames": frames}
            or width % 16 or height % 16 or (frames - 1) % 8 or frames > source.shape[0]
            or prep.get("dropped_tail_frames") != source.shape[0] - frames):
        raise ValueError("RGB source geometry differs from its preparation report")
    policy = prep.get("frame_policy")
    if ((policy == "trim_to_8n_plus_1" and frames != ((source.shape[0] - 1) // 8) * 8 + 1)
            or (policy == "preserve_all_exp" and frames != source.shape[0])
            or policy not in {"trim_to_8n_plus_1", "preserve_all_exp"}):
        raise ValueError("RGB source frame policy differs")
    fps, duration = prep.get("fps"), prep.get("output_duration_seconds")
    if (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(duration) not in (int, float) or not math.isfinite(duration)
            or not math.isclose(duration, frames / fps, rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("RGB source frame rate or duration differs")
    latent = values["ltx_latent"]
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("RGB handoff needs a lifted LTX LATENT")
    samples = _tensor(latent["samples"], "LTX lifted latent", 5)
    if tuple(samples.shape) != (1, 128, (frames - 1) // 8 + 1, height // 16, width // 16):
        raise ValueError("LTX lifted latent differs from RGB geometry")
    audio = values["source_audio"]
    if audio is not None:
        if (not isinstance(audio, dict) or "waveform" not in audio
                or type(audio.get("sample_rate")) is not int or audio["sample_rate"] <= 0):
            raise ValueError("Original H3 audio must be an AUDIO object or None")
        waveform = _tensor(audio["waveform"], "H3 original waveform", 3)
        if min(waveform.shape) < 1:
            raise ValueError("Original H3 audio cannot be empty")
    return _input_identity(values)


def _check_tensors(tensors):
    if sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_FILE - MAX_JSON:
        raise ValueError("RGB handoff exceeds the 20 GiB limit")
    for tensor in tensors.values():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("RGB handoff contains non-finite tensor metadata")


def _report_result(status, digest):
    return canonical({"schema": SCHEMA, "status": status, "artifact_sha256": digest,
        "sampling_executed": False, "sampler_completion_verified": False,
        "automatic_cache_reuse": False, "executable_identity_certified": False,
        "boundary": "Explicit selected input data only. Rebuild current MODEL/conditions/Stage Bind. "
                    "No H3 or LTX sampling completion, quality or current upstream equivalence is certified."})


def save_source(values, storage_root, prefix="rgb_source", *, interrupt=lambda: None):
    interrupt()
    identity = validate_source(values)
    tensors = {}
    encoded = _encode_metadata_value(values, tensors, path="ltx_rgb_source")
    _check_tensors(tensors)
    used = set()
    snapshot = _decode_metadata_value(encoded, tensors, used, path="ltx_rgb_source")
    if used != set(tensors) or validate_source(snapshot) != identity or validate_source(values) != identity:
        raise ValueError("RGB source changed while snapshotting")
    payload = canonical({"schema": SCHEMA, "state": encoded, "identity": identity})
    if len(payload.encode("utf8")) > MAX_JSON:
        raise ValueError("RGB source metadata exceeds the 4 MiB limit")
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
                raise ValueError("RGB source tensor header exceeds its bound")
            os.fsync(handle.fileno())
        interrupt()
        if validate_source(values) != identity:
            raise ValueError("RGB source changed during persistence")
        os.replace(partial, state)
        manifest = {"schema": SCHEMA, "state_file": state.name, "state_bytes": state.stat().st_size,
                    "state_sha256": file_sha(state),
                    "payload_sha256": hashlib.sha256(payload.encode("utf8")).hexdigest(),
                    "data_only": True}
        pending, committed = directory / "manifest.json.partial", directory / "manifest.json"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf8"))
            handle.flush()
            os.fsync(handle.fileno())
        interrupt()
        if validate_source(values) != identity:
            raise ValueError("RGB source changed before manifest commit")
        os.replace(pending, committed)
    digest = file_sha(committed)
    return committed.relative_to(root).as_posix(), digest, _report_result("explicit_rgb_source_saved", digest)


def load_source(storage_root, artifact_path, artifact_sha256, *, interrupt=lambda: None):
    interrupt()
    root = _root(storage_root)
    path = _path(root, artifact_path)
    expected = _digest(artifact_sha256)
    if path.name != "manifest.json" or not path.parent.is_dir():
        raise ValueError("Select a committed RGB source manifest.json")
    with _lease(path.parent):
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON or file_sha(path) != expected:
            raise ValueError("RGB source manifest SHA mismatch or missing completion")
        manifest = _json_file(path)
        if (set(manifest) != {"schema", "state_file", "state_bytes", "state_sha256", "payload_sha256", "data_only"}
                or manifest["schema"] != SCHEMA or manifest["state_file"] != "state.safetensors"
                or manifest["data_only"] is not True or type(manifest["state_bytes"]) is not int
                or not 8 < manifest["state_bytes"] <= MAX_FILE):
            raise ValueError("Unknown RGB source manifest contract")
        state = _path(root, (path.parent / manifest["state_file"]).relative_to(root).as_posix())
        if (not state.is_file() or state.stat().st_size != manifest["state_bytes"]
                or file_sha(state) != _digest(manifest["state_sha256"])):
            raise ValueError("RGB source tensor size or SHA differs")
        with state.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size <= min(MAX_JSON, state.stat().st_size - 8):
            raise ValueError("RGB source tensor header exceeds its bound")
        interrupt()
        with safe_open(str(state), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or set(metadata) != {METADATA_KEY}:
                raise ValueError("Unknown RGB source metadata")
            raw = metadata[METADATA_KEY]
            if hashlib.sha256(raw.encode("utf8")).hexdigest() != _digest(manifest["payload_sha256"]):
                raise ValueError("RGB source payload SHA differs")
            payload = json.loads(raw)
            if set(payload) != {"schema", "state", "identity"} or payload["schema"] != SCHEMA:
                raise ValueError("Unknown RGB source payload contract")
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        _check_tensors(tensors)
        used = set()
        values = _decode_metadata_value(payload["state"], tensors, used, path="ltx_rgb_source")
        if used != set(tensors) or validate_source(values) != payload["identity"]:
            raise ValueError("RGB source tensor inventory or content identity differs")
        interrupt()
        if file_sha(path) != expected or file_sha(state) != manifest["state_sha256"]:
            raise ValueError("RGB source files changed while reading")
    return values, _report_result("explicit_rgb_source_loaded", expected)
