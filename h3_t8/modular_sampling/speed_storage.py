"""Explicit SPEED stage snapshots, never automatic cache matches.

The completed manifest is committed last.  A load requires its exact SHA,
today's next-stage plan/seed, and the same numerical implementation bytes.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import uuid

import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file

from ..native_latent_checkpoint_advanced import _decode_metadata_value, _encode_metadata_value
from ..speed_advanced import SPEED_PLAN_SCHEMA, StageConditioning
from .contracts import StageContext
from .results import _input_identity, canonical
from .speed_stages import (
    FrozenSpeedSpec, FrozenSpeedStageResult, SpeedStageResult,
    _implementation_identity, _plan_sha, _text_template,
)
from .storage import MAX_FILE, MAX_JSON, _digest, _lease, _path, _root, file_sha

SCHEMA = "t8.modular-sampling.frozen-speed-stage.v1"
FILENAME = "speed-stage.safetensors"


def stage_root(*, output_root=None):
    base = Path(folder_paths.get_output_directory() if output_root is None else output_root)
    return base / "MiniMaxH3" / "modular_speed_stages"


def _unique_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate SPEED artifact metadata key")
            value[key] = item
        return value

    if type(text) is not str or len(text.encode("utf-8")) > MAX_JSON:
        raise ValueError("SPEED artifact metadata exceeds its limit")
    return json.loads(text, object_pairs_hook=unique)


def _check_state(path):
    if not path.is_file() or not 8 < path.stat().st_size <= MAX_FILE:
        raise ValueError("Missing or oversized completed SPEED tensor file")
    with path.open("rb") as handle:
        header = struct.unpack("<Q", handle.read(8))[0]
    if not 0 < header <= MAX_JSON or header > path.stat().st_size - 8:
        raise ValueError("SPEED tensor header exceeds its limit")


def save_speed_stage(result, *, output_root=None):
    if type(result) is not SpeedStageResult:
        raise ValueError("SPEED save needs a live sampler-produced typed stage result")
    receipt = result.verify_live()
    if receipt["verified_recipe_completion"] is not True or receipt["portable_identity"] is not True:
        raise ValueError("SPEED stage lacks portable completed execution evidence")
    if receipt["request"]["implementation"] != _implementation_identity():
        raise ValueError("SPEED implementation changed after this stage sampled")
    root = _root(stage_root(output_root=output_root), create=True)
    directory = _path(root, f"stage-{result.spec.index}-{uuid.uuid4().hex}")
    directory.mkdir()
    with _lease(directory):
        tensors = {}
        descriptor = _encode_metadata_value(
            {"output": result.external_output, "template": _text_template(result.spec.stage)},
            tensors, path="speed.stage",
        )
        if sum(value.numel() * value.element_size() for value in tensors.values()) > MAX_FILE - MAX_JSON:
            raise ValueError("SPEED stage tensors exceed the size limit")
        snapshot = _decode_metadata_value(descriptor, tensors, set(), path="speed.stage")
        if (_input_identity(snapshot["output"]) != receipt["output"]
                or _input_identity(snapshot["template"]) != receipt["request"]["text_template"]):
            raise ValueError("SPEED serialized snapshot differs from the completed stage")
        payload = canonical({"schema": SCHEMA, "receipt_json": result.receipt_json,
                             "state": descriptor})
        _unique_json(payload)
        if len(payload.encode("utf-8")) > MAX_JSON:
            raise ValueError("SPEED stage metadata exceeds its limit")
        partial, state = directory / (FILENAME + ".partial"), directory / FILENAME
        save_file(tensors, str(partial), metadata={"speed_json": payload})
        _check_state(partial)
        with partial.open("rb+") as handle:
            os.fsync(handle.fileno())
        result.verify_live()
        os.replace(partial, state)
        manifest = {"schema": SCHEMA, "state_file": FILENAME, "state_sha256": file_sha(state),
                    "state_bytes": state.stat().st_size, "receipt_sha256": receipt["receipt_sha256"],
                    "plan_sha256": result.spec.plan_sha256, "stage_index": result.spec.index,
                    "portable_identity": True, "automatic_cache_reuse": False}
        pending, final = directory / "manifest.json.partial", directory / "manifest.json"
        with pending.open("xb") as handle:
            handle.write(canonical(manifest).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        result.verify_live()
        os.replace(pending, final)
        digest = file_sha(final)
    report = {"schema": SCHEMA, "status": "saved_explicit_speed_stage",
              "stage_index": result.spec.index, "receipt_sha256": receipt["receipt_sha256"],
              "artifact_sha256": digest, "automatic_cache_reuse": False}
    return result, final.relative_to(root).as_posix(), digest, canonical(report)


def load_speed_stage(plan, seed, shift_audio, next_stage_index, artifact_path, artifact_sha256,
                     *, output_root=None):
    if (type(plan) is not dict or plan.get("schema") != SPEED_PLAN_SCHEMA
            or type(next_stage_index) is not int or not 1 <= next_stage_index < len(plan["stages"])
            or type(seed) is not int or not 0 <= seed < 2**64
            or type(shift_audio) not in (int, float) or not math.isfinite(shift_audio)
            or shift_audio <= 0):
        raise ValueError("SPEED load needs a valid plan, seed and next-stage index")
    plan_sha256 = _plan_sha(plan)
    root = _root(stage_root(output_root=output_root))
    manifest_path = _path(root, artifact_path)
    if manifest_path.name != "manifest.json" or not manifest_path.parent.is_dir():
        raise ValueError("Select a completed SPEED manifest.json")
    expected_sha = _digest(artifact_sha256)
    with _lease(manifest_path.parent):
        if (not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_JSON
                or file_sha(manifest_path) != expected_sha):
            raise ValueError("SPEED manifest SHA mismatch or missing completion")
        manifest = _unique_json(manifest_path.read_text(encoding="utf-8"))
        if (type(manifest) is not dict or set(manifest) != {
            "schema", "state_file", "state_sha256", "state_bytes", "receipt_sha256",
            "plan_sha256", "stage_index", "portable_identity", "automatic_cache_reuse",
        } or manifest["schema"] != SCHEMA or manifest["state_file"] != FILENAME
                or manifest["portable_identity"] is not True
                or manifest["automatic_cache_reuse"] is not False
                or manifest["plan_sha256"] != plan_sha256
                or manifest["stage_index"] != next_stage_index - 1):
            raise ValueError("SPEED manifest does not belong to this next stage")
        state = _path(root, (manifest_path.parent / FILENAME).relative_to(root).as_posix())
        _check_state(state)
        if (type(manifest["state_bytes"]) is not int or state.stat().st_size != manifest["state_bytes"]
                or file_sha(state) != _digest(manifest["state_sha256"])):
            raise ValueError("SPEED tensor file SHA or size mismatch")
        with safe_open(str(state), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {"speed_json"}:
                raise ValueError("Unknown SPEED tensor metadata")
            payload = _unique_json(metadata["speed_json"])
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        if type(payload) is not dict or set(payload) != {"schema", "receipt_json", "state"} or payload["schema"] != SCHEMA:
            raise ValueError("Unknown SPEED tensor payload")
        used = set()
        state_value = _decode_metadata_value(payload["state"], tensors, used, path="speed.stage")
        if used != set(tensors) or type(state_value) is not dict or set(state_value) != {"output", "template"}:
            raise ValueError("SPEED tensor inventory differs from its descriptor")
        receipt = _unique_json(payload["receipt_json"])
        request = receipt["request"]
        if (receipt["receipt_sha256"] != manifest["receipt_sha256"]
                or receipt["verified_recipe_completion"] is not True
                or receipt["portable_identity"] is not True
                or request["implementation"] != _implementation_identity()
                or request["plan_sha256"] != plan_sha256
                or request["stage_index"] != next_stage_index - 1
                or request["seed"] != seed
                or request["shift_audio"] != float(shift_audio)):
            raise ValueError("SPEED receipt or current next-stage identity changed")
        template = state_value["template"]
        if type(template) is not dict or set(template) != {
            "positive", "mux_audio", "conditioned_prompt", "media_map", "report",
        }:
            raise ValueError("SPEED text template has unknown fields")
        stage = StageConditioning(template["positive"], {}, template["mux_audio"],
                                  template["conditioned_prompt"], template["media_map"], template["report"])
        frozen_spec = FrozenSpeedSpec(
            plan, plan_sha256, next_stage_index - 1,
            seed, float(shift_audio), request["source_prompt"], stage,
            StageContext.from_dict(request["stage_context"]),
            request["audio_scale"], request["noise_scale"],
        )
        frozen = FrozenSpeedStageResult(frozen_spec, state_value["output"], payload["receipt_json"])
        frozen.verify_live()
        if file_sha(state) != manifest["state_sha256"] or file_sha(manifest_path) != expected_sha:
            raise ValueError("SPEED artifact changed while loading")
    report = {"schema": SCHEMA, "status": "loaded_explicit_speed_stage",
              "stage_index": frozen_spec.index, "artifact_sha256": expected_sha,
              "sampling_calls": 0, "automatic_cache_reuse": False,
              "boundary": "Frozen selected previous stage, not proof that edited previous-stage settings still match."}
    return frozen, frozen_spec, canonical(report)


def fingerprint_speed_stage(artifact_path, *, output_root=None):
    root = _root(stage_root(output_root=output_root))
    path = _path(root, artifact_path)
    if path.name != "manifest.json" or not path.parent.is_dir():
        raise ValueError("Select the completed SPEED manifest")
    with _lease(path.parent):
        if not path.is_file():
            raise FileNotFoundError("SPEED manifest is missing")
        manifest = _unique_json(path.read_text(encoding="utf-8"))
        if type(manifest) is not dict or manifest.get("state_file") != FILENAME:
            raise ValueError("SPEED manifest does not name its completed tensor file")
        state = _path(root, (path.parent / FILENAME).relative_to(root).as_posix())
        if not state.is_file():
            raise FileNotFoundError("SPEED tensor file is missing")
        # Both files affect the cached Load output. A changed state must cause
        # the node to re-execute and fail SHA validation, not serve stale RAM.
        return file_sha(path) + ":" + file_sha(state)
