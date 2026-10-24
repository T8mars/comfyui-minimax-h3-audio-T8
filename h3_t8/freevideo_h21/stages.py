"""Distinct v4 completion objects; no old-family relabelling or unchecked resume."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .source import REVISION, canonical, digest
from .plans import STAGE_SCHEMA


def tensor_record(value):
    import torch
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("H21 completion needs actual finite floating point tensors")
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


@dataclass(frozen=True)
class Stage:
    video: object
    audio: object
    receipt_json: str
    receipt_sha256: str


def validate(stage):
    if not isinstance(stage, Stage) or hashlib.sha256(stage.receipt_json.encode()).hexdigest() != stage.receipt_sha256:
        raise ValueError("Expected a content-bound H21 completed stage, not another producer's LATENT")
    value = json.loads(stage.receipt_json)
    if value.get("schema") != STAGE_SCHEMA:
        raise ValueError("Wrong completed-stage producer")
    if value.get("video") != tensor_record(stage.video) or value.get("audio") != tensor_record(stage.audio):
        raise ValueError("H21 completed-stage tensor content changed")
    selected, actual = value["plan"], value["actual"]
    identity = hashlib.sha256(canonical({key: item for key, item in selected.items() if key != "identity"}).encode()).hexdigest()
    if selected.get("identity") != identity or selected.get("source_revision") != REVISION:
        raise ValueError("H21 completed-stage plan identity/source changed")
    if actual.get("plan_identity") != identity or actual.get("plan_sha256") != hashlib.sha256(canonical(selected).encode()).hexdigest():
        raise ValueError("H21 actual completion belongs to a different plan")
    expected = selected["expected_callbacks"]
    if type(actual.get("completed_callbacks")) is not int or actual["completed_callbacks"] != expected:
        raise ValueError("H21 completion lacks actual sampler callback coverage")
    passes = actual.get("passes", [])
    sampling = selected["sampling"]
    actual_roles = [row["role"] for row in passes]
    expected_roles = (["LOW", "HIGH"] if sampling["enabled"] and selected["role"] == "FULL" else
                      [selected["role"]] if sampling["enabled"] else ["SINGLE"])
    if actual_roles != expected_roles:
        raise ValueError("H21 actual pass roles/order disagree with the explicit plan")
    canvas = sampling["first"] if selected["role"] == "LOW" else selected["actual_canvas"]
    frames = canvas["frames"]
    expected_video = (1, 24, (frames - 5) // 17 * 5 + 2, canvas["height"] // 16, canvas["width"] // 16)
    expected_audio = (1, 32, 2, round(frames / 24 * 40))
    if tuple(stage.video.shape) != expected_video or tuple(stage.audio.shape) != expected_audio:
        raise ValueError("H21 actual AV latent axes do not match the completed physical canvas/clock")
    if sum(p["completed_callbacks"] for p in passes) != expected:
        raise ValueError("H21 completion pass/callback mismatch")
    for row in passes:
        expected_pass = sampling["refine_steps"] if row["role"] == "HIGH" else sampling["base_steps"]
        if type(row.get("completed_callbacks")) is not int or row["completed_callbacks"] != expected_pass:
            raise ValueError("H21 individual pass callback coverage disagrees with its actual schedule")
        seconds = row["sample"]["step_seconds"]
        if len(seconds) != row["completed_callbacks"] or any(type(s) not in (int, float) or not math.isfinite(s) or s < 0 for s in seconds):
            raise ValueError("H21 callback durations are incomplete or invalid")
    if selected["role"] == "HIGH":
        low_sha = value.get("low_receipt_sha256")
        if (not isinstance(low_sha, str) or len(low_sha) != 64 or any(c not in "0123456789abcdef" for c in low_sha)
                or value.get("low_audio") != tensor_record(stage.audio)):
            raise ValueError("H21 HIGH does not retain its completed LOW audio lineage")
    for field in ("config_sha256", "inventory_sha256", "request_sha256", "output_sha256"):
        sha = value.get(field)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("H21 completion missing exact " + field)
    return value


def make(video, audio, receipt):
    video, audio = video.detach().cpu().clone(), audio.detach().cpu().clone()
    value = dict(receipt, schema=STAGE_SCHEMA, video=tensor_record(video), audio=tensor_record(audio))
    text = canonical(value)
    stage = Stage(video, audio, text, hashlib.sha256(text.encode()).hexdigest())
    validate(stage)
    return stage


def save(stage, directory):
    from safetensors.torch import save_file
    value = validate(stage)
    target = Path(directory).resolve() / ("h21-stage-" + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    partial = target / "latents.partial"
    save_file(dict(video=stage.video.contiguous(), audio=stage.audio.contiguous()), str(partial))
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    final = target / "latents.safetensors"
    partial.rename(final)
    marker = target / "stage.json"
    with marker.open("x", encoding="utf8") as stream:
        stream.write(canonical(dict(schema="t8.freevideo.h21.saved.v4", receipt=value,
                                   tensors=final.name, tensor_sha256=digest(final))))
        stream.flush()
        os.fsync(stream.fileno())
    return str(marker), digest(marker)


def load(path, sha256):
    from safetensors.torch import load_file
    path = Path(path).resolve(strict=True)
    if digest(path) != sha256:
        raise ValueError("H21 saved-stage manifest SHA changed")
    saved = json.loads(path.read_text(encoding="utf8"))
    if saved.get("schema") != "t8.freevideo.h21.saved.v4" or saved.get("tensors") != "latents.safetensors":
        raise ValueError("Wrong H21 saved-stage layout")
    payload = path.parent / saved["tensors"]
    if payload.is_symlink() or payload.resolve().parent != path.parent or digest(payload) != saved["tensor_sha256"]:
        raise ValueError("H21 saved-stage tensor path/content changed")
    values = load_file(str(payload), device="cpu")
    if set(values) != {"video", "audio"}:
        raise ValueError("H21 saved-stage tensor keys changed")
    stage = make(values["video"], values["audio"], saved["receipt"])
    if json.loads(stage.receipt_json) != saved["receipt"]:
        raise ValueError("H21 saved-stage receipt disagrees with actual tensors")
    return stage
