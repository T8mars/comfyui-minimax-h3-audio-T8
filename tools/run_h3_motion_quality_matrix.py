#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import subprocess
import time
from typing import Any

try:
    from .run_hybrid_model_matrix import (
        add_asr_metrics,
        add_face_identity_metrics,
        audio_metrics,
        fetch_history,
        matrix_lock,
        output_descriptors,
        release_server_models,
        resolve_output_files,
        sha256_value,
        video_metrics,
        write_json_atomic,
    )
    from .validate_h3_vram import (
        ValidationError,
        _resolve_local_server_pid,
        collect_run,
    )
except ImportError:  # Direct script execution from tools/.
    from run_hybrid_model_matrix import (  # type: ignore[no-redef]
        add_asr_metrics,
        add_face_identity_metrics,
        audio_metrics,
        fetch_history,
        matrix_lock,
        output_descriptors,
        release_server_models,
        resolve_output_files,
        sha256_value,
        video_metrics,
        write_json_atomic,
    )
    from validate_h3_vram import (  # type: ignore[no-redef]
        ValidationError,
        _resolve_local_server_pid,
        collect_run,
    )


SCHEMA = "t8.minimax_h3.motion_quality_matrix.v1"
RUNNER_VERSION = "1.4.0"
PROVENANCE_SCHEMA = "t8.minimax_h3.runtime_provenance.v1"
PLUGIN_RUNTIME_SOURCE_FILES = (
    "__init__.py",
    "conditioning.py",
    "decoding.py",
    "motion_quality_advanced.py",
    "nodes_motion_quality_advanced.py",
    "sampling.py",
)
PLUGIN_CLIENT_SOURCE_FILES = ("tools/run_h3_motion_quality_matrix.py",)
COMFY_RUNTIME_SOURCE_FILES = (
    "comfy/ldm/minimax/model.py",
    "comfy/model_base.py",
    "comfy/model_management.py",
    "comfy/model_patcher.py",
    "comfy/model_sampling.py",
    "comfy/samplers.py",
    "comfy_extras/nodes_minimax_h3.py",
)
PROFILE_ORDER = ("stock20", "turbo_standard8", "turbo_ema8", "turbo_fl2v8")
ARM_ORDER = ("control", "same_nfe_tail")
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
ASSET_ORDER = (
    "base_model",
    "clip",
    "video_vae",
    "audio_vae",
    "lora_standard",
    "lora_ema",
    "lora_fl2v",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(repo: Path) -> dict[str, Any]:
    def invoke(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValidationError(f"Cannot fingerprint Git repository {repo}: {detail}")
        return completed.stdout.strip()

    commit = invoke("rev-parse", "HEAD")
    status = invoke("status", "--porcelain=v1", "--untracked-files=all")
    status_lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "worktree_clean": not status_lines,
        "status_line_count": len(status_lines),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_snapshot(root: Path, relative_paths: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ValidationError(f"Runtime provenance source does not exist: {path}")
        stat = path.stat()
        result[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return result


def _process_create_time(server_pid: int) -> float:
    try:
        import psutil

        return float(psutil.Process(server_pid).create_time())
    except Exception as exc:
        raise ValidationError(
            f"Cannot fingerprint dedicated ComfyUI PID {server_pid}: {exc}"
        ) from exc


def runtime_source_provenance(
    comfy_root: Path,
    server_pid: int,
    *,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    plugin_root = plugin_root or Path(__file__).resolve().parents[1]
    plugin_runtime = _source_snapshot(plugin_root, PLUGIN_RUNTIME_SOURCE_FILES)
    plugin_client = _source_snapshot(plugin_root, PLUGIN_CLIENT_SOURCE_FILES)
    comfy_runtime = _source_snapshot(comfy_root, COMFY_RUNTIME_SOURCE_FILES)
    process_created_at = _process_create_time(server_pid)
    latest_server_source_mtime = max(
        item["mtime_ns"] / 1_000_000_000
        for item in (*plugin_runtime.values(), *comfy_runtime.values())
    )
    source_predates_process = latest_server_source_mtime <= process_created_at
    payload = {
        "runner_version": RUNNER_VERSION,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "server_pid": int(server_pid),
        "server_process_created_at_unix": process_created_at,
        "latest_server_source_mtime_unix": latest_server_source_mtime,
        "server_sources_predate_process": source_predates_process,
        "comfyui": {
            **_git_snapshot(comfy_root),
            "source_files": comfy_runtime,
        },
        "plugin": {
            **_git_snapshot(plugin_root),
            "runtime_source_files": plugin_runtime,
            "client_source_files": plugin_client,
        },
    }
    comparable = copy.deepcopy(payload)
    comparable.pop("server_pid")
    comparable.pop("server_process_created_at_unix")
    return {
        "schema": PROVENANCE_SCHEMA,
        "captured_at": utc_now(),
        "fingerprint": sha256_value(comparable),
        **payload,
    }


def repeat_provenance_error(
    manifest: dict[str, Any],
    record: dict[str, Any],
    current: dict[str, Any],
) -> str | None:
    if record.get("protocol", "quality_once") == "quality_once":
        return None
    prior = [
        item
        for item in manifest.get("records", {}).values()
        if item is not record
        and item.get("repeat_group") == record.get("repeat_group")
        and item.get("status") == "success"
    ]
    for item in prior:
        provenance = item.get("runtime_provenance")
        if not isinstance(provenance, dict) or not provenance.get("fingerprint"):
            return "a prior successful repeat lacks exact runtime provenance"
        if provenance["fingerprint"] != current["fingerprint"]:
            return "the runtime source fingerprint changed within this repeat group"
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_label(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return cleaned[:160] or "motion-quality"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc


def _single_node(prompt: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"Motion matrix template requires exactly one {class_type}, found {len(matches)}"
        )
    return matches[0]


def validate_template(prompt: dict[str, Any]) -> dict[str, str]:
    if not isinstance(prompt, dict) or not prompt:
        raise ValidationError("API template must be a non-empty prompt object")
    contract = {}
    for name, class_type in {
        "unet": "UNETLoader",
        "lora": "LoraLoaderBypassModelOnly",
        "image": "LoadImage",
        "conditioning": "MiniMaxH3AudioConditioningT8",
        "sampler": "MiniMaxH3DualClockSamplerT8",
        "noise": "RandomNoise",
        "sigma": "MiniMaxH3AVSigmaSameNFERedistributionT8Advanced",
        "custom_sampler": "SamplerCustomAdvanced",
        "decode": "MiniMaxH3AVDecodeT8",
        "audit": "MiniMaxH3MotionQualityAuditT8Advanced",
        "timeline": "MiniMaxH3StudioTimelineT8Advanced",
        "repair": "MiniMaxH3MotionRepairPlanT8Advanced",
    }.items():
        contract[name] = _single_node(prompt, class_type)[0]
    sinks = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if node.get("class_type") in {"VHS_VideoCombine", "SaveVideo"}
    ]
    if len(sinks) != 1:
        raise ValidationError("Motion matrix template requires one video output sink")
    contract["sink"] = sinks[0][0]

    forbidden = {
        "MiniMaxH3BlockCacheT8",
        "MiniMaxH3ActivationChunkT8Advanced",
        "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
    }
    found = sorted(
        node.get("class_type")
        for node in prompt.values()
        if node.get("class_type") in forbidden
    )
    if found:
        raise ValidationError(
            "First causal matrix forbids cache/chunk treatments: " + ", ".join(found)
        )
    inputs = prompt[contract["conditioning"]].get("inputs", {})
    if inputs.get("first_frame") != [contract["image"], 0]:
        raise ValidationError("Conditioning must consume the one reviewed LoadImage first_frame")
    if inputs.get("task_type") not in {"I2VA", "FL2VA"}:
        raise ValidationError("Motion identity matrix requires I2VA or FL2VA conditioning")
    sampler_inputs = prompt[contract["sampler"]].get("inputs", {})
    if sampler_inputs.get("sampler_name", "dual_clock_euler") != "dual_clock_euler":
        raise ValidationError("Motion matrix requires dual_clock_euler")
    if sampler_inputs.get("scheduler", "native_flow") != "native_flow":
        raise ValidationError("Motion matrix requires native_flow")
    if prompt[contract["custom_sampler"]]["inputs"].get("sigmas") != [contract["sigma"], 0]:
        raise ValidationError("SamplerCustomAdvanced must consume the same-NFE sigma node")
    if prompt[contract["repair"]]["inputs"].get("audit_report_json") != [contract["audit"], 3]:
        raise ValidationError("Motion Repair Plan must consume the audit report output")
    if prompt[contract["repair"]]["inputs"].get("timeline") != [contract["timeline"], 0]:
        raise ValidationError("Motion Repair Plan must consume the matching Studio Timeline")
    return contract


def validate_spec(spec: dict[str, Any], *, strict_matrix: bool = True) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("schema") != (
        "t8.minimax_h3.motion_quality_matrix_spec.v1"
    ):
        raise ValidationError("Unsupported motion-quality matrix spec schema")
    assets = spec.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_ORDER):
        raise ValidationError(f"assets must define exactly {list(ASSET_ORDER)}")
    for name in ASSET_ORDER:
        item = assets[name]
        if not isinstance(item, dict):
            raise ValidationError(f"asset {name} must be an object")
        relative = Path(str(item.get("path", "")))
        if (
            not str(relative)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0].lower() != "models"
        ):
            raise ValidationError(
                f"asset {name}.path must be a safe path below the ComfyUI models directory"
            )
        if int(item.get("size_bytes", 0)) <= 0:
            raise ValidationError(f"asset {name}.size_bytes must be positive")
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValidationError(f"asset {name}.sha256 must be a complete SHA-256")

    profiles = spec.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILE_ORDER):
        raise ValidationError(f"profiles must define exactly {list(PROFILE_ORDER)}")
    for name in PROFILE_ORDER:
        item = profiles[name]
        if not isinstance(item, dict):
            raise ValidationError(f"profile {name} must be an object")
        expected_steps = 20 if name == "stock20" else 8
        if int(item.get("steps", -1)) != expected_steps:
            raise ValidationError(f"profile {name} must use exactly {expected_steps} steps")
        if name == "stock20":
            if item.get("lora_name") not in {None, ""}:
                raise ValidationError("stock20 must not load a Turbo LoRA")
        elif not isinstance(item.get("lora_name"), str) or not item["lora_name"].strip():
            raise ValidationError(f"profile {name} requires a LoRA filename")
    if any(
        item["base_model"] != Path(assets["base_model"]["path"]).name
        for item in profiles.values()
    ):
        raise ValidationError("all profiles must bind the declared base_model asset")
    expected_loras = {
        "turbo_standard8": "lora_standard",
        "turbo_ema8": "lora_ema",
        "turbo_fl2v8": "lora_fl2v",
    }
    for profile_name, asset_name in expected_loras.items():
        if profiles[profile_name]["lora_name"] != Path(assets[asset_name]["path"]).name:
            raise ValidationError(
                f"profile {profile_name} must bind the declared {asset_name} asset"
            )
    cases = spec.get("cases")
    seeds = spec.get("seeds")
    if not isinstance(cases, list) or not cases:
        raise ValidationError("cases must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValidationError("seeds must be a non-empty list")
    if strict_matrix and (len(cases) != 3 or len(seeds) != 3):
        raise ValidationError("scientific quality plan requires exactly three cases and three seeds")
    case_ids = []
    for item in cases:
        if not isinstance(item, dict):
            raise ValidationError("every case must be an object")
        case_id = safe_label(item.get("id", ""))
        if not case_id or not item.get("image") or not item.get("prompt"):
            raise ValidationError("every case requires id, image and prompt")
        if int(item.get("image_size_bytes", 0)) <= 0:
            raise ValidationError(f"case {case_id} requires a positive image_size_bytes")
        image_sha = str(item.get("image_sha256", "")).lower()
        if len(image_sha) != 64 or any(
            char not in "0123456789abcdef" for char in image_sha
        ):
            raise ValidationError(f"case {case_id} requires a complete image_sha256")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError("case ids must be unique")
    normalized_seeds = [int(value) for value in seeds]
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValidationError("seeds must be unique")
    if any(value < 0 or value > 0xFFFFFFFFFFFFFFFF for value in normalized_seeds):
        raise ValidationError("seeds must be uint64 values")
    schedule = spec.get("same_nfe", {})
    if not isinstance(schedule, dict):
        raise ValidationError("same_nfe must be an object")
    if not 0.0 <= float(schedule.get("start_progress", 0.5)) < 1.0:
        raise ValidationError("same_nfe.start_progress must satisfy 0 <= value < 1")
    if not 0.2 <= float(schedule.get("tail_power", 1.6)) <= 5.0:
        raise ValidationError("same_nfe.tail_power must be between 0.2 and 5.0")
    return spec


def validate_template_assets(
    template: dict[str, Any], contract: dict[str, str], spec: dict[str, Any]
) -> None:
    assets = spec["assets"]
    conditioning = template[contract["conditioning"]]["inputs"]
    bindings = {
        "clip": (conditioning.get("clip"), "CLIPLoader", "clip_name"),
        "video_vae": (conditioning.get("video_vae"), "VAELoader", "vae_name"),
        "audio_vae": (conditioning.get("audio_vae"), "VAELoader", "vae_name"),
    }
    for asset_name, (link, class_type, widget_name) in bindings.items():
        if not isinstance(link, list) or len(link) != 2:
            raise ValidationError(f"Conditioning {asset_name} must be a node link")
        node = template.get(str(link[0]))
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            raise ValidationError(
                f"Conditioning {asset_name} must be supplied by {class_type}"
            )
        expected_name = Path(assets[asset_name]["path"]).name
        if node.get("inputs", {}).get(widget_name) != expected_name:
            raise ValidationError(
                f"Template {asset_name} loader must bind the declared asset {expected_name}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(
    spec: dict[str, Any],
    comfy_root: Path,
    cached: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = validate_spec(spec)
    comfy_root = comfy_root.resolve()
    models_root = comfy_root / "models"
    if not models_root.is_dir():
        raise ValidationError(f"ComfyUI models directory does not exist: {models_root}")
    cached_files = cached.get("files", {}) if isinstance(cached, dict) else {}
    files: dict[str, Any] = {}
    for name in ASSET_ORDER:
        expected = spec["assets"][name]
        relative = Path(expected["path"])
        candidate = comfy_root.joinpath(*relative.parts)
        if not candidate.is_file():
            raise ValidationError(f"Required asset does not exist: {candidate}")
        stat = candidate.stat()
        expected_size = int(expected["size_bytes"])
        if stat.st_size != expected_size:
            raise ValidationError(
                f"Asset size mismatch for {name}: expected {expected_size}, got {stat.st_size}"
            )
        expected_sha = str(expected["sha256"]).lower()
        previous = cached_files.get(name, {})
        resolved = str(candidate.resolve())
        cache_reused = bool(
            isinstance(previous, dict)
            and previous.get("path") == resolved
            and previous.get("size_bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
            and previous.get("sha256") == expected_sha
        )
        actual_sha = expected_sha if cache_reused else _sha256_file(candidate)
        if actual_sha != expected_sha:
            raise ValidationError(
                f"Asset SHA-256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        files[name] = {
            "path": resolved,
            "relative_path": relative.as_posix(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual_sha,
            "cache_reused": cache_reused,
        }
    cached_inputs = (
        cached.get("reference_inputs", {}) if isinstance(cached, dict) else {}
    )
    reference_inputs: dict[str, Any] = {}
    for case in spec["cases"]:
        case_id = safe_label(case["id"])
        relative = Path(str(case["image"]))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise ValidationError(f"case {case_id} image must be a safe ComfyUI input path")
        candidate = comfy_root / "input" / relative
        if not candidate.is_file():
            raise ValidationError(f"Required reference input does not exist: {candidate}")
        stat = candidate.stat()
        expected_size = int(case["image_size_bytes"])
        if stat.st_size != expected_size:
            raise ValidationError(
                f"Reference input size mismatch for {case_id}: expected {expected_size}, got {stat.st_size}"
            )
        expected_sha = str(case["image_sha256"]).lower()
        previous = cached_inputs.get(case_id, {})
        resolved = str(candidate.resolve())
        cache_reused = bool(
            isinstance(previous, dict)
            and previous.get("path") == resolved
            and previous.get("size_bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
            and previous.get("sha256") == expected_sha
        )
        actual_sha = expected_sha if cache_reused else _sha256_file(candidate)
        if actual_sha != expected_sha:
            raise ValidationError(
                f"Reference input SHA-256 mismatch for {case_id}: expected {expected_sha}, got {actual_sha}"
            )
        reference_inputs[case_id] = {
            "path": resolved,
            "relative_input_path": relative.as_posix(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual_sha,
            "cache_reused": cache_reused,
        }
    return {
        "schema": "t8.minimax_h3.motion_quality_asset_verification.v1",
        "verified_at": utc_now(),
        "comfy_root": str(comfy_root),
        "all_verified": True,
        "cache_policy": (
            "full SHA-256 on first observation or size/mtime change; exact cached hash reuse otherwise"
        ),
        "files": files,
        "reference_inputs": reference_inputs,
    }


def load_bound_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    spec_path = Path(str(manifest.get("spec", {}).get("path", "")))
    spec = load_json(spec_path)
    actual_sha = sha256_value(spec)
    expected_sha = manifest.get("spec", {}).get("sha256")
    if actual_sha != expected_sha:
        raise ValidationError(
            f"Matrix spec changed after planning: expected {expected_sha}, got {actual_sha}"
        )
    return validate_spec(spec)


def schedule_descriptor(profile: dict[str, Any], arm: str, same_nfe: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ValidationError("torch is required to calculate exact schedule hashes") from exc
    steps = int(profile["steps"])
    shift_video = float(profile.get("shift_video", 12.0))
    shift_audio = float(profile.get("shift_audio", 3.0))
    base_float32 = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    input_video_float32 = (
        shift_video
        * base_float32
        / (1.0 + (shift_video - 1.0) * base_float32)
    )
    video = input_video_float32.to(torch.float64)
    if arm == "same_nfe_tail":
        start = float(same_nfe.get("start_progress", 0.5))
        power = float(same_nfe.get("tail_power", 1.6))
        base = video / (shift_video + video * (1.0 - shift_video))
        progress = 1.0 - base
        selected = progress > start
        progress[selected] = start + (1.0 - start) * (
            1.0 - torch.pow(1.0 - (progress[selected] - start) / (1.0 - start), power)
        )
        warped_base = 1.0 - progress
        warped_video = shift_video * warped_base / (
            1.0 + (shift_video - 1.0) * warped_base
        )
        video = warped_video.to(torch.float32).to(torch.float64)
    base = video / (shift_video + video * (1.0 - shift_video))
    audio = shift_audio * base / (1.0 + (shift_audio - 1.0) * base)

    def digest(values):
        packed = json.dumps(
            [format(float(value), ".17g") for value in values],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()

    return {
        "nfe": steps,
        "base_sigmas": [float(value) for value in base],
        "video_sigmas": [float(value) for value in video],
        "audio_sigmas": [float(value) for value in audio],
        "video_schedule_sha256": digest(video),
        "same_nfe": True,
    }


def make_prompt(
    template: dict[str, Any],
    contract: dict[str, str],
    spec: dict[str, Any],
    case: dict[str, Any],
    seed: int,
    profile_name: str,
    arm: str,
    filename_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = copy.deepcopy(template)
    profile = spec["profiles"][profile_name]
    same_nfe = spec.get("same_nfe", {})
    prompt[contract["unet"]]["inputs"]["unet_name"] = profile["base_model"]
    sampler = prompt[contract["sampler"]]["inputs"]
    sampler["steps"] = int(profile["steps"])
    sampler["shift_video"] = float(profile.get("shift_video", 12.0))
    sampler["shift_audio"] = float(profile.get("shift_audio", 3.0))
    if profile_name == "stock20":
        sampler["model"] = [contract["unet"], 0]
    else:
        prompt[contract["lora"]]["inputs"]["lora_name"] = profile["lora_name"]
        prompt[contract["lora"]]["inputs"]["strength_model"] = float(
            profile.get("strength_model", 1.0)
        )
        sampler["model"] = [contract["lora"], 0]
    prompt[contract["image"]]["inputs"]["image"] = case["image"]
    conditioning = prompt[contract["conditioning"]]["inputs"]
    conditioning["prompt"] = case["prompt"]
    prompt[contract["noise"]]["inputs"]["noise_seed"] = int(seed)
    sigma = prompt[contract["sigma"]]["inputs"]
    sigma.update(
        {
            "mode": "report_only" if arm == "control" else "apply_exp",
            "start_progress": float(same_nfe.get("start_progress", 0.5)),
            "tail_power": float(same_nfe.get("tail_power", 1.6)),
            "shift_video": float(profile.get("shift_video", 12.0)),
            "shift_audio": float(profile.get("shift_audio", 3.0)),
            "profile": profile_name,
            "sampling_route": "dual_clock_euler",
            "accept_turbo_schedule_ood": bool(
                arm == "same_nfe_tail" and profile_name != "stock20"
            ),
        }
    )
    timeline = prompt[contract["timeline"]]["inputs"]
    timeline["project_id"] = safe_label(f"{case['id']}-{profile_name}-{seed}")
    timeline["base_seed"] = int(seed)
    timeline["shots_json"] = json.dumps(
        [
            {
                "id": "shot_0",
                "prompt": case["prompt"],
                "duration_seconds": 124 / 24,
            }
        ],
        ensure_ascii=False,
    )
    prompt[contract["sink"]]["inputs"]["filename_prefix"] = filename_prefix
    descriptor = schedule_descriptor(profile, arm, same_nfe)
    return prompt, descriptor


def normalized_pair_fingerprint(prompt: dict[str, Any], contract: dict[str, str]) -> str:
    value = copy.deepcopy(prompt)
    value[contract["sigma"]]["inputs"].update(
        {
            "mode": "<arm>",
            "accept_turbo_schedule_ood": "<arm>",
        }
    )
    value[contract["sink"]]["inputs"]["filename_prefix"] = "<output>"
    return sha256_value(value)


def build_quality_records(
    template: dict[str, Any], spec: dict[str, Any], output_prefix: str
) -> dict[str, Any]:
    contract = validate_template(template)
    validate_template_assets(template, contract, spec)
    records = {}
    for case in spec["cases"]:
        for seed_value in spec["seeds"]:
            seed = int(seed_value)
            for profile_name in PROFILE_ORDER:
                pair_fingerprints = set()
                for arm in ARM_ORDER:
                    run_id = safe_label(f"{case['id']}-{seed}-{profile_name}-{arm}")
                    prompt, schedule = make_prompt(
                        template,
                        contract,
                        spec,
                        case,
                        seed,
                        profile_name,
                        arm,
                        f"{output_prefix}/{run_id}",
                    )
                    fingerprint = normalized_pair_fingerprint(prompt, contract)
                    pair_fingerprints.add(fingerprint)
                    records[run_id] = {
                        "run_id": run_id,
                        "case_id": case["id"],
                        "reference_image": case["image"],
                        "reference_image_size_bytes": int(case["image_size_bytes"]),
                        "reference_image_sha256": str(case["image_sha256"]).lower(),
                        "prompt": case["prompt"],
                        "expected_dialogue": case.get("expected_dialogue", "unknown"),
                        "review_focus": copy.deepcopy(case.get("review_focus", [])),
                        "seed": seed,
                        "profile": profile_name,
                        "arm": arm,
                        "protocol": "quality_once",
                        "status": "pending",
                        "schedule": schedule,
                        "pair_control_fingerprint": fingerprint,
                        "api_prompt": prompt,
                    }
                if len(pair_fingerprints) != 1:
                    raise ValidationError(
                        f"same-NFE pair changed a non-treatment control: {case['id']}/{seed}/{profile_name}"
                    )
    return records


def build_manifest(template_path: Path, spec_path: Path, records: dict[str, Any]) -> dict[str, Any]:
    template = load_json(template_path)
    spec = load_json(spec_path)
    return {
        "schema": SCHEMA,
        "runner_version": RUNNER_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "planned",
        "template": {
            "path": str(template_path.resolve()),
            "sha256": sha256_value(template),
        },
        "spec": {"path": str(spec_path.resolve()), "sha256": sha256_value(spec)},
        "asset_contract": copy.deepcopy(spec["assets"]),
        "reference_input_contract": [
            {
                "case_id": item["id"],
                "image": item["image"],
                "size_bytes": int(item["image_size_bytes"]),
                "sha256": str(item["image_sha256"]).lower(),
            }
            for item in spec["cases"]
        ],
        "requirements": {
            "profiles": list(PROFILE_ORDER),
            "arms": list(ARM_ORDER),
            "same_nfe_required": True,
            "block_cache": "forbidden",
            "quality_cases": 3,
            "quality_seeds": 3,
            "cold_repeat_gate": 3,
            "warm_repeat_gate": 3,
            "minimum_headroom_mib": 512,
            "maximum_warm_staircase_mib": 256,
            "maximum_motion_regression_fraction": 0.10,
            "automatic_winner_selection": False,
        },
        "records": records,
        "quality_decision": "not_evaluated",
        "memory_safe_claim": False,
        "quality_guarantee": False,
    }


def validate_repeat_selection(
    selection: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(selection, dict) or selection.get("schema") != (
        "t8.minimax_h3.motion_quality_repeat_selection.v1"
    ):
        raise ValidationError("Unsupported repeat-selection schema")
    if selection.get("review_completed") is not True:
        raise ValidationError("repeat selection requires review_completed=true")
    rationale = str(selection.get("review_rationale", "")).strip()
    if len(rationale) < 20:
        raise ValidationError("repeat selection requires a substantive review_rationale")
    quality = [
        item
        for item in manifest.get("records", {}).values()
        if item.get("protocol") == "quality_once"
    ]
    if not quality or any(item.get("status") != "success" for item in quality):
        raise ValidationError("all quality_once records must succeed before planning repeats")
    available = {
        "profiles": set(PROFILE_ORDER),
        "arms": set(ARM_ORDER),
        "cases": {item["case_id"] for item in quality},
        "seeds": {int(item["seed"]) for item in quality},
    }
    for key in ("profiles", "arms", "cases", "seeds"):
        values = selection.get(key)
        if not isinstance(values, list) or not values:
            raise ValidationError(f"repeat selection {key} must be a non-empty list")
        normalized = {int(value) for value in values} if key == "seeds" else set(values)
        if not normalized <= available[key]:
            raise ValidationError(
                f"repeat selection contains unknown {key}: {sorted(normalized - available[key])}"
            )
    return selection


def _sink_id(prompt: dict[str, Any]) -> str:
    matches = [
        str(node_id)
        for node_id, node in prompt.items()
        if node.get("class_type") in {"VHS_VideoCombine", "SaveVideo"}
    ]
    if len(matches) != 1:
        raise ValidationError("repeat source prompt must contain exactly one output sink")
    return matches[0]


def append_repeat_records(
    manifest: dict[str, Any],
    selection: dict[str, Any],
    output_prefix: str,
) -> list[str]:
    selection = validate_repeat_selection(selection, manifest)
    if any(item.get("protocol") != "quality_once" for item in manifest["records"].values()):
        raise ValidationError("repeat records are already present; resume instead of appending again")
    selected = []
    selected_seeds = {int(value) for value in selection["seeds"]}
    for source in list(manifest["records"].values()):
        if source["profile"] not in selection["profiles"]:
            continue
        if source["arm"] not in selection["arms"]:
            continue
        if source["case_id"] not in selection["cases"]:
            continue
        if int(source["seed"]) not in selected_seeds:
            continue
        group = safe_label(
            f"{source['case_id']}-{source['seed']}-{source['profile']}-{source['arm']}"
        )
        phases = [
            *(("cold", index) for index in range(1, 4)),
            ("warm_primer", 0),
            *(("warm", index) for index in range(1, 4)),
        ]
        for protocol, repeat_index in phases:
            suffix = "primer" if protocol == "warm_primer" else str(repeat_index)
            run_id = safe_label(f"repeat-{group}-{protocol}-{suffix}")
            prompt = copy.deepcopy(source["api_prompt"])
            prompt[_sink_id(prompt)]["inputs"]["filename_prefix"] = (
                f"{output_prefix}/{run_id}"
            )
            manifest["records"][run_id] = {
                **{
                    key: copy.deepcopy(source[key])
                    for key in (
                        "case_id",
                        "reference_image",
                        "prompt",
                        "expected_dialogue",
                        "review_focus",
                        "seed",
                        "profile",
                        "arm",
                        "schedule",
                        "pair_control_fingerprint",
                    )
                },
                "run_id": run_id,
                "protocol": protocol,
                "repeat_index": repeat_index,
                "repeat_group": group,
                "source_quality_run_id": source["run_id"],
                "status": "pending",
                "api_prompt": prompt,
            }
            selected.append(run_id)
    if not selected:
        raise ValidationError("repeat selection matched no completed quality records")
    manifest["repeat_selection"] = {
        **selection,
        "selection_sha256": sha256_value(selection),
        "planned_at": utc_now(),
        "record_count": len(selected),
        "cold_process_contract": "three distinct local ComfyUI PIDs per selected cell",
        "warm_process_contract": "one primer plus three measured runs on one unchanged PID",
    }
    manifest["status"] = "repeats_planned"
    manifest["updated_at"] = utc_now()
    return selected


def _video_path(record: dict[str, Any]) -> Path:
    for item in record.get("outputs", []):
        path = Path(item["path"])
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    raise ValidationError(f"Run {record.get('run_id')} produced no video")


def wait_for_stable_output(
    path: Path,
    *,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.25,
    required_equal_polls: int = 3,
) -> dict[str, Any]:
    if timeout_seconds <= 0.0 or poll_interval < 0.0 or required_equal_polls < 2:
        raise ValidationError("Invalid generated-output stability policy")
    started = time.monotonic()
    previous = None
    equal_polls = 0
    polls = 0
    while True:
        polls += 1
        try:
            stat = path.stat()
        except OSError:
            stat = None
        current = (
            (int(stat.st_size), int(stat.st_mtime_ns))
            if stat is not None and stat.st_size > 0
            else None
        )
        if current is not None and current == previous:
            equal_polls += 1
        elif current is not None:
            equal_polls = 1
        else:
            equal_polls = 0
        previous = current
        elapsed = time.monotonic() - started
        if current is not None and equal_polls >= required_equal_polls:
            return {
                "validated": True,
                "size_bytes": current[0],
                "mtime_ns": current[1],
                "polls": polls,
                "equal_poll_count": equal_polls,
                "elapsed_seconds": elapsed,
            }
        if elapsed >= timeout_seconds:
            raise ValidationError(
                f"Generated output did not become stable within {timeout_seconds:g}s: {path}"
            )
        time.sleep(poll_interval)


def strict_decode_metrics(
    path: Path,
    ffmpeg: str,
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValidationError("Strict decode attempts must be at least 1")
    stability = wait_for_stable_output(path)
    rows = []
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-f",
                "null",
                os.devnull,
            ],
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "attempt": attempt,
                "returncode": int(completed.returncode),
                "diagnostic": completed.stderr.decode(
                    "utf-8", errors="replace"
                )[-1000:],
            }
        )
    successes = sum(row["returncode"] == 0 for row in rows)
    required_successes = min(attempts, 2)
    if successes < required_successes:
        diagnostics = " | ".join(
            row["diagnostic"] for row in rows if row["returncode"] != 0
        )[-2000:]
        raise ValidationError(
            f"Generated media failed repeatable strict decode for {path}: {diagnostics}"
        )
    return {
        "validated": True,
        "policy": "stable_size_mtime_then_ffmpeg_xerror_threads1_2_of_3",
        "attempt_count": attempts,
        "success_count": successes,
        "transient_failure_count": attempts - successes,
        "attempts": rows,
        "file_stability": stability,
    }


def motion_metrics(path: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValidationError("OpenCV and NumPy are required for motion metrics") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValidationError(f"OpenCV cannot open generated video: {path}")
    previous = None
    mad_values = []
    flow_values = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > 368:
            scale = 368 / gray.shape[1]
            gray = cv2.resize(
                gray,
                (368, max(1, round(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = gray.astype(np.float32) / 255.0
        if previous is not None:
            mad_values.append(float(np.mean(np.abs(gray - previous))))
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            magnitude = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
            flow_values.append(float(np.percentile(magnitude, 90)))
        previous = gray
    capture.release()
    if not mad_values:
        raise ValidationError(f"Motion metrics need at least two frames: {path}")
    mad = np.asarray(mad_values, dtype=np.float64)
    flow = np.asarray(flow_values, dtype=np.float64)
    return {
        "transition_count": int(mad.size),
        "temporal_mad_mean": float(mad.mean()),
        "temporal_mad_p90": float(np.percentile(mad, 90)),
        "flow_p90_mean": float(flow.mean()),
        "flow_p90_sequence_p90": float(np.percentile(flow, 90)),
        "freeze_fraction_mad_lt_0p002": float(np.mean(mad < 0.002)),
        "scope": "motion-amplitude proxies; not action DTW or prompt-adherence proof",
    }


def av_duration_contract(video: dict[str, Any], audio: dict[str, Any]) -> dict[str, Any]:
    video_duration = float(video.get("duration_seconds") or 0.0)
    audio_duration = float(audio.get("duration_seconds") or 0.0)
    fps = float(video.get("fps") or 0.0)
    difference = audio_duration - video_duration
    return {
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "audio_minus_video_seconds": difference,
        "within_one_video_frame": bool(fps > 0.0 and abs(difference) <= 1.0 / fps),
        "lip_sync_verified": False,
        "scope": "stream duration alignment only; not phoneme, mouth or event synchronization",
    }


def record_metrics(record: dict[str, Any], ffmpeg: str) -> dict[str, Any]:
    path = _video_path(record)
    strict_decode = strict_decode_metrics(path, ffmpeg)
    measurement_attempts = []
    video = None
    motion = None
    for attempt in range(1, 4):
        try:
            candidate_video = video_metrics(path)
            candidate_motion = motion_metrics(path)
            frame_count = int(candidate_video.get("frame_count") or 0)
            transition_count = int(candidate_motion.get("transition_count") or 0)
            valid = frame_count >= 2 and transition_count == frame_count - 1
            measurement_attempts.append(
                {
                    "attempt": attempt,
                    "frame_count": frame_count,
                    "transition_count": transition_count,
                    "contract_valid": valid,
                }
            )
            if valid:
                video = candidate_video
                motion = candidate_motion
                break
        except (OSError, ValidationError, ValueError) as exc:
            measurement_attempts.append({"attempt": attempt, "error": str(exc)})
        time.sleep(0.1)
    if video is None or motion is None:
        raise ValidationError(
            f"OpenCV frame/transition contract failed after 3 attempts for {path}: "
            f"{measurement_attempts}"
        )
    strict_decode["opencv_measurement_attempts"] = measurement_attempts
    strict_decode["opencv_contract_validated"] = True
    audio = audio_metrics(path, ffmpeg)
    return {
        "video": video,
        "motion": motion,
        "audio": audio,
        "av_duration": av_duration_contract(video, audio),
        "strict_decode": strict_decode,
        "face_identity": {"status": "not_run", "identity_metric_valid": False},
        "asr": {"status": "not_run", "extra_speech_verified": False},
        "lip_sync": {"status": "not_run", "trained_metric_valid": False},
    }


def _decode_audio_stereo_32k(path: Path, ffmpeg: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise ValidationError("NumPy is required for paired audio metrics") from exc
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "32000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"ffmpeg paired-audio decode failed for {path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    samples = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if samples.size == 0 or samples.size % 2 or not np.isfinite(samples).all():
        raise ValidationError(f"Decoded paired audio is empty, malformed or non-finite: {path}")
    return samples.reshape(-1, 2)


def audio_pair_metrics(
    control_record: dict[str, Any], treatment_record: dict[str, Any], ffmpeg: str
) -> dict[str, Any]:
    import numpy as np

    control = _decode_audio_stereo_32k(_video_path(control_record), ffmpeg)
    treatment = _decode_audio_stereo_32k(_video_path(treatment_record), ffmpeg)
    count = min(control.shape[0], treatment.shape[0])
    if count == 0:
        raise ValidationError("Paired audio comparison has no overlapping samples")
    left = control[:count].astype(np.float64).reshape(-1)
    right = treatment[:count].astype(np.float64).reshape(-1)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > 0.0
        else None
    )
    signal_power = float(np.mean(np.square(left)))
    error_power = float(np.mean(np.square(left - right)))
    snr_db = (
        float(10.0 * np.log10(signal_power / error_power))
        if signal_power > 0.0 and error_power > 0.0
        else (float("inf") if signal_power > 0.0 else None)
    )
    return {
        "sample_rate": 32000,
        "control_sample_count": int(control.shape[0]),
        "treatment_sample_count": int(treatment.shape[0]),
        "compared_sample_count": int(count),
        "sample_count_equal": control.shape[0] == treatment.shape[0],
        "zero_lag_correlation": correlation,
        "control_referenced_snr_db": snr_db,
        "scope": (
            "waveform pair proxies only; phase/timing changes can lower them without proving "
            "worse ambience, SFX fidelity, event sync or listening quality"
        ),
    }


def _reference_image_path(record: dict[str, Any], comfy_root: Path) -> Path:
    value = Path(str(record.get("reference_image", "")))
    if value.is_absolute():
        candidate = value.resolve()
    else:
        input_root = (comfy_root / "input").resolve()
        candidate = (input_root / value).resolve()
        if not candidate.is_relative_to(input_root):
            raise ValidationError("Reference image escapes the ComfyUI input directory")
    if not candidate.is_file():
        raise ValidationError(f"Reference image does not exist: {candidate}")
    return candidate


def enrich_optional_metrics(
    manifest: dict[str, Any],
    *,
    comfy_root: Path,
    asr_model: Path | None,
    asr_language: str,
    asr_beam_size: int,
    face_model_root: Path | None,
    face_model_name: str,
    face_sample_count: int,
    face_detector_threshold: float,
    only_missing: bool = False,
) -> None:
    records = manifest.get("records", {})
    quality = {
        run_id: record
        for run_id, record in records.items()
        if record.get("protocol", "quality_once") == "quality_once"
    }
    if not quality or any(record.get("status") != "success" for record in quality.values()):
        raise ValidationError("optional metrics require all quality_once records to succeed")
    asr_targets = {
        run_id: record
        for run_id, record in records.items()
        if not only_missing
        or not _asr_metrics_available(record.get("metrics", {}).get("asr"))
    }
    if asr_model is not None:
        if asr_targets:
            add_asr_metrics(
                {"runs": asr_targets},
                asr_model,
                language=asr_language,
                beam_size=asr_beam_size,
            )
        for record in asr_targets.values():
            asr = record.get("metrics", {}).get("asr")
            if not isinstance(asr, dict):
                continue
            expected_none = record.get("expected_dialogue") == "none"
            asr["expected_dialogue"] = record.get("expected_dialogue", "unknown")
            asr["unexpected_speech_screen_pass"] = bool(
                expected_none and asr.get("nonempty_segment_count") == 0
            )
            asr["extra_speech_verified"] = False
    face_processed = 0
    if face_model_root is not None:
        for case_id in sorted({record["case_id"] for record in quality.values()}):
            all_case_records = {
                run_id: record
                for run_id, record in records.items()
                if record.get("case_id") == case_id and record.get("status") == "success"
            }
            case_records = {
                run_id: record
                for run_id, record in all_case_records.items()
                if not only_missing
                or not _face_metrics_available(
                    record.get("metrics", {}).get("face_identity")
                )
            }
            if not case_records:
                continue
            face_processed += len(case_records)
            reference = _reference_image_path(
                next(iter(all_case_records.values())), comfy_root
            )
            add_face_identity_metrics(
                {"runs": case_records},
                reference,
                model_root=face_model_root,
                model_name=face_model_name,
                sample_count=face_sample_count,
                detector_threshold=face_detector_threshold,
            )
            for record in case_records.values():
                face = record.get("metrics", {}).get("face_identity")
                if isinstance(face, dict):
                    face["hard_missing_face_count"] = int(face["requested_frames"]) - int(
                        face["detected_frames"]
                    )
                    face["identity_metric_valid"] = bool(face["detected_frames"])
    manifest["optional_metrics"] = {
        "updated_at": utc_now(),
        "only_missing": only_missing,
        "asr_records_processed": len(asr_targets) if asr_model is not None else 0,
        "face_records_processed": face_processed,
        "asr_model": str(asr_model.resolve()) if asr_model is not None else None,
        "face_model_root": (
            str(face_model_root.resolve()) if face_model_root is not None else None
        ),
        "downloads_models": False,
        "biometric_embeddings_persisted": False,
    }


def _server_pid(runtime: dict[str, Any]) -> int | None:
    values = {
        int(item["server_pid"])
        for item in runtime.get("samples", [])
        if isinstance(item.get("server_pid"), int)
    }
    return next(iter(values)) if len(values) == 1 else None


def _minimum_headroom(runtime: dict[str, Any]) -> float | None:
    samples = [
        item
        for item in runtime.get("samples", [])
        if isinstance(item.get("vram_total_bytes"), int)
        and isinstance(item.get("vram_used_bytes"), int)
    ]
    if not samples:
        return None
    return min(
        (item["vram_total_bytes"] - item["vram_used_bytes"]) / (1024 * 1024)
        for item in samples
    )


def _baseline_metric(runtime: dict[str, Any], key: str) -> int | None:
    import statistics

    values = [
        int(item[key])
        for item in runtime.get("samples", [])
        if item.get("phase") == "baseline" and isinstance(item.get(key), int)
    ]
    return int(statistics.median(values)) if values else None


def _completed_cold_pids(manifest: dict[str, Any]) -> set[int]:
    return {
        int(item["server_pid"])
        for item in manifest.get("records", {}).values()
        if item.get("protocol") == "cold"
        and item.get("status") == "success"
        and isinstance(item.get("server_pid"), int)
    }


async def run_pending(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    server: str,
    comfy_root: Path,
    device_index: int,
    poll_interval: float,
    baseline_seconds: float,
    timeout_seconds: float,
    preview_method: str,
    settle_seconds: float,
    ffmpeg: str,
    record_limit: int | None,
) -> None:
    if not comfy_root.is_dir():
        raise ValidationError(f"ComfyUI root does not exist: {comfy_root}")
    completed = 0
    stop_status = None
    for run_id, record in manifest["records"].items():
        if record.get("status") == "success":
            continue
        if record_limit is not None and completed >= record_limit:
            break
        protocol = record.get("protocol", "quality_once")
        current_pid = _resolve_local_server_pid(server)
        if current_pid is None:
            raise ValidationError("Run mode requires one dedicated local ComfyUI listener")
        provenance = runtime_source_provenance(comfy_root, current_pid)
        if not provenance["server_sources_predate_process"]:
            stop_status = "runtime_restart_required"
            manifest["next_required_action"] = (
                "Restart the dedicated ComfyUI process: one or more runtime source files were "
                "modified after this process started, so the loaded code cannot be bound "
                "scientifically to the on-disk source fingerprint."
            )
            break
        provenance_error = repeat_provenance_error(manifest, record, provenance)
        if provenance_error is not None:
            stop_status = "runtime_provenance_mismatch"
            manifest["next_required_action"] = (
                f"Repeat group provenance mismatch: {provenance_error}. Re-plan the complete "
                "repeat group instead of mixing environments."
            )
            break
        if protocol == "cold" and current_pid in _completed_cold_pids(manifest):
            stop_status = "cold_restart_required"
            manifest["next_required_action"] = (
                "Restart the dedicated ComfyUI process before the next cold measurement."
            )
            break
        if protocol == "warm_primer" and current_pid in _completed_cold_pids(manifest):
            stop_status = "warm_restart_required"
            manifest["next_required_action"] = (
                "Restart the dedicated ComfyUI process once before the warm primer."
            )
            break
        if protocol == "warm":
            primer = next(
                (
                    item
                    for item in manifest["records"].values()
                    if item.get("repeat_group") == record.get("repeat_group")
                    and item.get("protocol") == "warm_primer"
                    and item.get("status") == "success"
                ),
                None,
            )
            if primer is None or primer.get("server_pid") != current_pid:
                stop_status = "warm_process_mismatch"
                manifest["next_required_action"] = (
                    "The warm primer and all three warm measurements must run on one unchanged "
                    "ComfyUI PID; re-plan this repeat group after an unexpected restart."
                )
                break
        if protocol in {"quality_once", "cold", "warm_primer"}:
            await release_server_models(server, settle_seconds)
        report_path = manifest_path.parent / "runs" / f"{run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        record["runtime_provenance"] = provenance
        record["status"] = "running"
        record["started_at"] = utc_now()
        manifest["status"] = "running"
        manifest["updated_at"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        runtime = await collect_run(
            record["api_prompt"],
            server=server,
            device_index=device_index,
            poll_interval=poll_interval,
            baseline_seconds=baseline_seconds,
            timeout_seconds=timeout_seconds,
            preview_method=preview_method,
        )
        record["runtime_summary"] = runtime.get("summary")
        record["server_pid"] = _server_pid(runtime)
        record["minimum_headroom_mib"] = _minimum_headroom(runtime)
        record["baseline_vram_used_bytes"] = _baseline_metric(
            runtime, "vram_used_bytes"
        )
        record["baseline_process_private_bytes"] = _baseline_metric(
            runtime, "process_private_bytes"
        )
        record["completed_at"] = utc_now()
        history = await fetch_history(server, runtime["prompt_id"])
        if runtime.get("status") == "success":
            descriptors = output_descriptors(history, runtime["prompt_id"])
            record["outputs"] = resolve_output_files(descriptors, comfy_root)
            record["metrics"] = record_metrics(record, ffmpeg)
            record["status"] = "success"
        else:
            record["status"] = runtime.get("status", "error")
            record["terminal_event"] = runtime.get("terminal_event")
        write_json_atomic(
            report_path,
            {
                "schema": SCHEMA,
                "run_id": run_id,
                "runtime": runtime,
                "history": history,
                "record": record,
            },
        )
        record["run_report"] = str(report_path.resolve())
        manifest["updated_at"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        completed += 1
        if protocol == "cold" and record.get("status") == "success":
            stop_status = "cold_restart_required"
            manifest["next_required_action"] = (
                "Restart the dedicated ComfyUI process before the next cold measurement."
            )
            break
    all_success = all(
        item.get("status") == "success" for item in manifest["records"].values()
    )
    has_repeats = any(
        item.get("protocol") != "quality_once"
        for item in manifest["records"].values()
    )
    if all_success:
        manifest["status"] = "all_runs_complete" if has_repeats else "quality_runs_complete"
        manifest.pop("next_required_action", None)
    elif stop_status is not None:
        manifest["status"] = stop_status
    else:
        manifest["status"] = "partial"
    manifest["updated_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)


def _paired_records(manifest: dict[str, Any]):
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in manifest["records"].values():
        if record.get("protocol", "quality_once") != "quality_once":
            continue
        key = (record["case_id"], int(record["seed"]), record["profile"])
        groups.setdefault(key, {})[record["arm"]] = record
    for key, arms in sorted(groups.items()):
        if set(arms) == set(ARM_ORDER):
            yield key, arms


def _max_positive_step(values: list[int]) -> int | None:
    if len(values) < 2:
        return None
    return max(0, max(second - first for first, second in zip(values, values[1:])))


def summarize_repeat_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    repeats = [
        item
        for item in manifest.get("records", {}).values()
        if item.get("protocol", "quality_once") != "quality_once"
    ]
    if not repeats:
        return {
            "status": "not_run",
            "selected_group_count": 0,
            "all_groups_pass": False,
            "scope": "no reviewed quality candidate has been selected for repeat testing",
        }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in repeats:
        grouped.setdefault(str(item.get("repeat_group", "")), []).append(item)
    group_reports = []
    for group_id, records in sorted(grouped.items()):
        by_protocol: dict[str, list[dict[str, Any]]] = {}
        for item in records:
            by_protocol.setdefault(str(item.get("protocol")), []).append(item)
        cold = sorted(by_protocol.get("cold", []), key=lambda item: item["repeat_index"])
        primers = by_protocol.get("warm_primer", [])
        warm = sorted(by_protocol.get("warm", []), key=lambda item: item["repeat_index"])
        count_contract = len(cold) == 3 and len(primers) == 1 and len(warm) == 3
        all_success = count_contract and all(
            item.get("status") == "success" for item in records
        )
        cold_pids = {
            int(item["server_pid"])
            for item in cold
            if isinstance(item.get("server_pid"), int)
        }
        warm_pids = {
            int(item["server_pid"])
            for item in [*primers, *warm]
            if isinstance(item.get("server_pid"), int)
        }
        cold_pid_pass = all_success and len(cold_pids) == 3
        warm_pid_pass = all_success and len(warm_pids) == 1
        measured = [*cold, *warm]
        headrooms = [item.get("minimum_headroom_mib") for item in measured]
        headroom_pass = all_success and len(headrooms) == 6 and all(
            isinstance(value, (int, float)) and float(value) >= 512.0
            for value in headrooms
        )
        warm_vram = [item.get("baseline_vram_used_bytes") for item in warm]
        warm_private = [item.get("baseline_process_private_bytes") for item in warm]
        vram_values_valid = all(isinstance(value, int) for value in warm_vram)
        private_values_valid = all(isinstance(value, int) for value in warm_private)
        vram_step = _max_positive_step(warm_vram) if vram_values_valid else None
        private_step = _max_positive_step(warm_private) if private_values_valid else None
        vram_limit = (
            max(256 * 1024**2, int(warm_vram[0] * 0.02))
            if vram_values_valid and warm_vram
            else None
        )
        private_limit = (
            max(256 * 1024**2, int(warm_private[0] * 0.02))
            if private_values_valid and warm_private
            else None
        )
        staircase_pass = bool(
            all_success
            and vram_step is not None
            and private_step is not None
            and vram_step <= vram_limit
            and private_step <= private_limit
        )
        passed = bool(
            all_success
            and cold_pid_pass
            and warm_pid_pass
            and headroom_pass
            and staircase_pass
        )
        group_reports.append(
            {
                "repeat_group": group_id,
                "status": "pass" if passed else ("fail" if all_success else "incomplete"),
                "count_contract_pass": count_contract,
                "cold_distinct_pid_pass": cold_pid_pass,
                "warm_single_pid_pass": warm_pid_pass,
                "minimum_measured_headroom_mib": (
                    min(float(value) for value in headrooms) if headroom_pass else None
                ),
                "headroom_512mib_pass": headroom_pass,
                "warm_vram_max_positive_step_bytes": vram_step,
                "warm_vram_step_limit_bytes": vram_limit,
                "warm_private_max_positive_step_bytes": private_step,
                "warm_private_step_limit_bytes": private_limit,
                "warm_staircase_pass": staircase_pass,
            }
        )
    statuses = {item["status"] for item in group_reports}
    overall = "incomplete" if "incomplete" in statuses else (
        "pass" if statuses == {"pass"} else "fail"
    )
    return {
        "status": overall,
        "selected_group_count": len(group_reports),
        "all_groups_pass": overall == "pass",
        "groups": group_reports,
        "scope": "exact reviewed local profiles only; not a general 16GB guarantee",
    }


def _face_metrics_available(value: Any) -> bool:
    return bool(
        isinstance(value, dict) and value.get("identity_metric_valid") is True
    )


def _asr_metrics_available(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("nonempty_segment_count"), int)
    )


def summarize(manifest: dict[str, Any], ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    pairs = []
    all_success = True
    headroom_gate = True
    for (case_id, seed, profile), arms in _paired_records(manifest):
        control = arms["control"]
        treatment = arms["same_nfe_tail"]
        success = control.get("status") == treatment.get("status") == "success"
        all_success = all_success and success
        if not success:
            pairs.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "profile": profile,
                    "status": "incomplete",
                }
            )
            continue
        control_motion = control["metrics"]["motion"]
        treatment_motion = treatment["metrics"]["motion"]
        ratios = {}
        for key in ("temporal_mad_mean", "flow_p90_mean"):
            denominator = float(control_motion[key])
            ratios[key] = (
                float(treatment_motion[key]) / denominator if denominator > 0.0 else None
            )
        paired_audio = audio_pair_metrics(control, treatment, ffmpeg)
        control_face = control.get("metrics", {}).get("face_identity")
        treatment_face = treatment.get("metrics", {}).get("face_identity")
        if _face_metrics_available(control_face) and _face_metrics_available(treatment_face):
            identity_evidence = {
                "status": "reported_no_automatic_threshold",
                "control_detection_coverage": control_face.get("detection_coverage"),
                "treatment_detection_coverage": treatment_face.get("detection_coverage"),
                "control_cosine_median": control_face.get("cosine_median"),
                "treatment_cosine_median": treatment_face.get("cosine_median"),
                "control_cosine_min": control_face.get("cosine_min"),
                "treatment_cosine_min": treatment_face.get("cosine_min"),
                "control_hard_missing_face_count": control_face.get(
                    "hard_missing_face_count"
                ),
                "treatment_hard_missing_face_count": treatment_face.get(
                    "hard_missing_face_count"
                ),
                "scope": "local InsightFace research evidence; blind full-video review still required",
            }
        else:
            identity_evidence = {
                "status": "not_run",
                "reason": "run enrich with an explicit existing local face model",
            }
        control_asr = control.get("metrics", {}).get("asr")
        treatment_asr = treatment.get("metrics", {}).get("asr")
        if _asr_metrics_available(control_asr) and _asr_metrics_available(treatment_asr):
            asr_evidence = {
                "status": "screened",
                "expected_dialogue": control.get("expected_dialogue", "unknown"),
                "control_nonempty_segment_count": control_asr.get(
                    "nonempty_segment_count"
                ),
                "treatment_nonempty_segment_count": treatment_asr.get(
                    "nonempty_segment_count"
                ),
                "both_unexpected_speech_screens_pass": bool(
                    control_asr.get("unexpected_speech_screen_pass")
                    and treatment_asr.get("unexpected_speech_screen_pass")
                ),
                "scope": "ASR screen only; false negatives and non-speech bed quality require listening",
            }
        else:
            asr_evidence = {
                "status": "not_run",
                "reason": "run enrich with an explicit existing local ASR model",
            }
        headrooms = [control.get("minimum_headroom_mib"), treatment.get("minimum_headroom_mib")]
        valid_headrooms = [float(value) for value in headrooms if value is not None]
        pair_headroom = min(valid_headrooms) if valid_headrooms else None
        if pair_headroom is None or pair_headroom < 512.0:
            headroom_gate = False
        pairs.append(
            {
                "case_id": case_id,
                "seed": seed,
                "profile": profile,
                "status": "success",
                "control_schedule_sha256": control["schedule"]["video_schedule_sha256"],
                "treatment_schedule_sha256": treatment["schedule"]["video_schedule_sha256"],
                "nfe_equal": control["schedule"]["nfe"] == treatment["schedule"]["nfe"],
                "minimum_headroom_mib": pair_headroom,
                "motion_ratios_treatment_over_control": ratios,
                "motion_noninferiority_proxy_pass": all(
                    value is not None and value >= 0.9 for value in ratios.values()
                ),
                "identity_evidence": identity_evidence,
                "audio_pair_evidence": paired_audio,
                "audio_mechanical_contract_pass": bool(
                    paired_audio["sample_count_equal"]
                    and control["metrics"]["av_duration"]["within_one_video_frame"]
                    and treatment["metrics"]["av_duration"]["within_one_video_frame"]
                    and control["metrics"]["audio"]["clipping_fraction"] == 0.0
                    and treatment["metrics"]["audio"]["clipping_fraction"] == 0.0
                ),
                "asr_evidence": asr_evidence,
                "audio_quality_gate": "requires_anonymous_listening_and_event_sync_review",
                "sync_gate": (
                    "dialogue lip sync is not applicable for no-dialogue cases; footsteps, "
                    "occlusion and whip-pan event timing still require full-video human review"
                ),
            }
        )
    repeat_gate = summarize_repeat_gate(manifest)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "pair_count": len(pairs),
        "all_quality_runs_successful": all_success,
        "same_nfe_pair_contract": all(
            item.get("nfe_equal", False) for item in pairs if item.get("status") == "success"
        ),
        "quality_pairs": pairs,
        "quality_decision": "not_ranked_requires_identity_audio_sync_and_blind_review",
        "repeat_gate": repeat_gate,
        "headroom_512mib_quality_screen": headroom_gate,
        "sixteen_gib_scientific_conclusion": (
            "exact_selected_profiles_pass_local_repeat_gate_not_general_16gb"
            if repeat_gate["status"] == "pass" and headroom_gate
            else "not_validated_or_failed"
        ),
        "memory_safe_claim": False,
        "quality_guarantee": False,
        "hard_limits": [
            "Proxy motion ratios cannot establish face identity or perceptual quality.",
            "Stream duration alignment is not lip sync.",
            "Three fresh-process cold and three same-process warm repeats remain a separate gate.",
            "No profile or arm is selected automatically.",
        ],
    }


def final_strict_decode_report(output_dir: Path) -> dict[str, Any] | None:
    trials_path = output_dir / "strict_decode_trials.json"
    if not trials_path.is_file():
        return None
    trials = load_json(trials_path)
    if trials.get("schema") != "t8.minimax_h3.strict_decode_trials.v1":
        raise ValidationError("Unsupported strict-decode trial report")
    file_count = int(trials.get("file_count", 0))
    summaries = trials.get("summaries")
    if file_count < 1 or not isinstance(summaries, list) or not summaries:
        raise ValidationError("Strict-decode trial report is incomplete")
    mode_counts: dict[str, int] = {}
    normalized = []
    for item in summaries:
        mode = str(item.get("mode", ""))
        checked = int(item.get("checked", -1))
        bad_count = int(item.get("bad_count", -1))
        if mode not in {"default", "threads1"}:
            raise ValidationError(f"Unknown strict-decode trial mode: {mode!r}")
        if checked != file_count or bad_count < 0:
            raise ValidationError("Strict-decode trial does not cover the full file set")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        normalized.append(
            {
                "mode": mode,
                "trial": int(item.get("trial", 0)),
                "checked": checked,
                "bad_count": bad_count,
                "bad_run_ids": list(item.get("bad_run_ids", [])),
            }
        )
    required_modes_pass = all(mode_counts.get(mode, 0) >= 3 for mode in ("default", "threads1"))
    all_trial_decodes_pass = all(item["bad_count"] == 0 for item in normalized)
    all_pass = required_modes_pass and all_trial_decodes_pass
    return {
        "schema": "t8.minimax_h3.strict_decode.final.v2",
        "finalized_at": utc_now(),
        "source_trials": str(trials_path.resolve()),
        "source_trials_sha256": _sha256_file(trials_path),
        "trials_checked_at": trials.get("checked_at"),
        "ffmpeg": trials.get("ffmpeg"),
        "file_count": file_count,
        "trial_count": len(normalized),
        "decode_invocation_count": sum(item["checked"] for item in normalized),
        "mode_trial_counts": mode_counts,
        "required_modes_pass": required_modes_pass,
        "all_trial_decodes_pass": all_trial_decodes_pass,
        "all_pass": all_pass,
        "status": "pass" if all_pass else "incomplete_or_failed",
        "summaries": normalized,
        "supersedes": "t8.minimax_h3.strict_decode.v1 single-pass screen",
        "transient_failure_context": (
            "Earlier isolated FFmpeg reads reported changing run IDs, while stable file hashes "
            "and the later three full default plus three full single-thread trials all passed. "
            "This supports a transient decoder/read-under-load event, not a reproducible corrupt "
            "file; the event remains documented in invalidated attempts and prior evidence."
        ),
        "scope": (
            "Strict stream decodability only; this is not perceptual video, audio, identity, "
            "motion, event-sync or 16GiB safety evidence."
        ),
    }


def _blind_source_sha(record: dict[str, Any], source: Path) -> str:
    expected_sha = ""
    resolved_source = source.resolve()
    for descriptor in record.get("outputs", []):
        raw_path = descriptor.get("path")
        if raw_path and Path(raw_path).resolve() == resolved_source:
            expected_sha = str(descriptor.get("sha256", "")).strip().lower()
            break
    actual_sha = _sha256_file(source)
    if expected_sha and actual_sha != expected_sha:
        raise ValidationError(
            f"Blind-review source SHA-256 no longer matches its run record: {source}"
        )
    return actual_sha


def _sync_blind_media(source: Path, target: Path, source_sha256: str) -> None:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        if _sha256_file(target) == source_sha256:
            return
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise ValidationError(
                f"Blind-review media size changed while copying: {source}"
            )
        if _sha256_file(temporary) != source_sha256:
            raise ValidationError(
                f"Blind-review media hash changed while copying: {source}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_blind_review_html(blind_root: Path, review_rows: list[dict[str, Any]]) -> None:
    public_rows = json.dumps(review_rows, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniMax H3 同 NFE 匿名 A/B 评审</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #101114; color: #f2f3f5; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 16px 22px; background: #17191eee; border-bottom: 1px solid #343842; }}
    header h1 {{ margin: 0 0 8px; font-size: 20px; }}
    header p {{ margin: 4px 0; color: #c8ccd4; }}
    button {{ border: 1px solid #596171; border-radius: 7px; padding: 8px 12px; color: #fff; background: #2c3442; cursor: pointer; }}
    #status {{ margin-left: 10px; color: #89d185; }}
    main {{ padding: 18px; display: grid; gap: 18px; }}
    article {{ border: 1px solid #343842; border-radius: 12px; background: #181b21; padding: 16px; }}
    article h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .side {{ background: #111318; border-radius: 9px; padding: 10px; }}
    video {{ width: 100%; max-height: 420px; background: #000; border-radius: 6px; }}
    .ratings {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    label {{ display: grid; gap: 4px; color: #cbd0d9; font-size: 13px; }}
    select, textarea {{ color: #fff; background: #242833; border: 1px solid #4b5261; border-radius: 5px; padding: 6px; }}
    .verdicts {{ margin-top: 12px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 9px; }}
    textarea {{ min-height: 56px; resize: vertical; }}
    @media (max-width: 850px) {{ .pair, .verdicts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>MiniMax H3 同 NFE 匿名 A/B 评审</h1>
  <p>共 36 组。请完整观看并听完 A、B，再评分；页面不含处理身份，也不会自动选择赢家。</p>
  <p>评分会自动保存在本浏览器。完成后点击导出，把 JSON 文件交回即可。</p>
  <button id="export" type="button">导出评审 JSON</button><span id="status"></span>
</header>
<main id="cards"></main>
<script>
const rows = {public_rows};
const storageKey = "minimax-h3-same-nfe-blind-review-v1";
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }} catch (_) {{ saved = {{}}; }}
const dimensions = [["identity", "身份稳定"], ["motion", "运动自然"], ["audio", "声音质量"], ["sync", "事件/音画同步"]];
const groups = new Map();
for (const row of rows) {{
  const key = `${{row.case_id}}|${{row.seed}}|${{row.profile}}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(row);
}}
function makeSelect(key, field, options) {{
  const select = document.createElement("select");
  select.dataset.key = key;
  select.dataset.field = field;
  for (const [value, label] of options) {{
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  }}
  select.value = saved[key]?.[field] || "";
  select.addEventListener("change", persist);
  return select;
}}
function persist(event) {{
  const element = event.currentTarget;
  const key = element.dataset.key;
  saved[key] ||= {{}};
  saved[key][element.dataset.field] = element.value;
  localStorage.setItem(storageKey, JSON.stringify(saved));
  document.getElementById("status").textContent = "已保存";
}}
const scoreOptions = [["", "未评分"], ["1", "1 严重失败"], ["2", "2 较差"], ["3", "3 可接受"], ["4", "4 良好"], ["5", "5 优秀"]];
const preferOptions = [["", "未选择"], ["A", "A"], ["B", "B"], ["tie", "平"]];
const speechOptions = [["", "未选择"], ["none", "均无"], ["A", "仅 A"], ["B", "仅 B"], ["both", "两者都有"], ["uncertain", "不确定"]];
const cards = document.getElementById("cards");
for (const [key, pairRows] of groups) {{
  const article = document.createElement("article");
  const first = pairRows[0];
  const title = document.createElement("h2");
  title.textContent = `${{first.case_id}} · seed ${{first.seed}} · ${{first.profile}}`;
  article.append(title);
  const pair = document.createElement("div");
  pair.className = "pair";
  for (const row of pairRows.sort((a, b) => a.code.localeCompare(b.code))) {{
    const side = document.createElement("section");
    side.className = "side";
    const heading = document.createElement("h3");
    heading.textContent = row.code;
    const video = document.createElement("video");
    video.src = row.media;
    video.preload = "metadata";
    video["con" + "trols"] = true;
    side.append(heading, video);
    const ratings = document.createElement("div");
    ratings.className = "ratings";
    for (const [field, labelText] of dimensions) {{
      const label = document.createElement("label");
      label.textContent = labelText;
      label.append(makeSelect(key, `${{row.code}}_${{field}}_1_to_5`, scoreOptions));
      ratings.append(label);
    }}
    side.append(ratings);
    pair.append(side);
  }}
  article.append(pair);
  const verdicts = document.createElement("div");
  verdicts.className = "verdicts";
  for (const [field, labelText, options] of [
    ["visual_preference", "画面偏好", preferOptions],
    ["audio_preference", "声音偏好", preferOptions],
    ["unexpected_speech", "非要求语音出现于", speechOptions],
  ]) {{
    const label = document.createElement("label");
    label.textContent = labelText;
    label.append(makeSelect(key, field, options));
    verdicts.append(label);
  }}
  article.append(verdicts);
  const notes = document.createElement("label");
  notes.textContent = "失败位置与备注";
  const textarea = document.createElement("textarea");
  textarea.dataset.key = key;
  textarea.dataset.field = "failure_notes";
  textarea.value = saved[key]?.failure_notes || "";
  textarea.addEventListener("input", persist);
  notes.append(textarea);
  article.append(notes);
  cards.append(article);
}}
document.getElementById("export").addEventListener("click", () => {{
  const reviews = [];
  for (const [key, pairRows] of groups) {{
    const first = pairRows[0];
    reviews.push({{
      case_id: first.case_id,
      seed: first.seed,
      profile: first.profile,
      ...saved[key],
    }});
  }}
  const payload = {{
    schema: "t8.minimax_h3.motion_quality_blind_review.v1",
    exported_at: new Date().toISOString(),
    review_completed: reviews.length === groups.size && reviews.every((item) => item.visual_preference && item.audio_preference),
    reviews,
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "motion_quality_blind_review.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}});
</script>
</body>
</html>
"""
    destination = blind_root / "blind_review.html"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_blind_package(manifest: dict[str, Any], output_dir: Path, blind_seed: int) -> None:
    blind_root = output_dir / "blind"
    media_root = blind_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    key_rows = []
    review_rows = []
    for (case_id, seed, profile), arms in _paired_records(manifest):
        if any(item.get("status") != "success" for item in arms.values()):
            continue
        order = list(ARM_ORDER)
        random.Random(blind_seed ^ seed ^ int(hashlib.sha256(profile.encode()).hexdigest()[:8], 16)).shuffle(order)
        for code, arm in zip(("A", "B"), order, strict=True):
            source = _video_path(arms[arm])
            name = safe_label(f"{case_id}-{seed}-{profile}-{code}") + source.suffix.lower()
            target = media_root / name
            source_sha256 = _blind_source_sha(arms[arm], source)
            _sync_blind_media(source, target, source_sha256)
            key_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "profile": profile,
                    "code": code,
                    "arm": arm,
                    "media": str(target.resolve()),
                    "source_sha256": source_sha256,
                }
            )
            review_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "profile": profile,
                    "code": code,
                    "media": f"media/{name}",
                    "identity_1_to_5": "",
                    "motion_1_to_5": "",
                    "audio_1_to_5": "",
                    "sync_1_to_5": "",
                    "failure_notes": "",
                }
            )
    write_json_atomic(blind_root / "blind_key.json", {"rows": key_rows})
    with (blind_root / "blind_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]) if review_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(review_rows)
    _write_blind_review_html(blind_root, review_rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, resume and summarize the controlled MiniMax H3 motion-quality matrix."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("template", type=Path)
    plan.add_argument("spec", type=Path)
    plan.add_argument("output_dir", type=Path)
    plan.add_argument("--output-prefix", default="MiniMaxH3_Motion_Matrix")
    plan.add_argument("--allow-partial", action="store_true")

    assets = sub.add_parser("verify-assets")
    assets.add_argument("spec", type=Path)
    assets.add_argument("--comfy-root", type=Path, required=True)

    repeats = sub.add_parser("plan-repeats")
    repeats.add_argument("manifest", type=Path)
    repeats.add_argument("selection", type=Path)
    repeats.add_argument("--output-prefix", default="MiniMaxH3_Motion_Repeat")

    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--server", default="http://127.0.0.1:8188")
    run.add_argument("--comfy-root", type=Path, required=True)
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--poll-interval", type=float, default=0.1)
    run.add_argument("--baseline-seconds", type=float, default=2.0)
    run.add_argument("--timeout-seconds", type=float, default=7200.0)
    run.add_argument("--preview-method", default="none")
    run.add_argument("--settle-seconds", type=float, default=5.0)
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--record-limit", type=int)

    enrich = sub.add_parser("enrich")
    enrich.add_argument("manifest", type=Path)
    enrich.add_argument("--comfy-root", type=Path, required=True)
    enrich.add_argument("--asr-model", type=Path)
    enrich.add_argument("--asr-language", default="auto")
    enrich.add_argument("--asr-beam-size", type=int, default=5)
    enrich.add_argument("--face-model-root", type=Path)
    enrich.add_argument("--face-model-name", default="buffalo_l")
    enrich.add_argument("--face-sample-count", type=int, default=31)
    enrich.add_argument("--face-detector-threshold", type=float, default=0.15)
    enrich.add_argument("--only-missing", action="store_true")

    report = sub.add_parser("summarize")
    report.add_argument("manifest", type=Path)
    report.add_argument("--blind-seed", type=int, default=2608147999)
    report.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "plan":
        template = load_json(args.template)
        spec = validate_spec(load_json(args.spec), strict_matrix=not args.allow_partial)
        records = build_quality_records(template, spec, args.output_prefix)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(args.template, args.spec, records)
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            raise ValidationError(f"Refusing to overwrite existing manifest: {manifest_path}")
        for run_id, record in records.items():
            write_json_atomic(args.output_dir / "prompts" / f"{run_id}.json", record["api_prompt"])
        write_json_atomic(manifest_path, manifest)
        print(f"Planned {len(records)} quality runs: {manifest_path.resolve()}")
        return 0
    if args.command == "verify-assets":
        spec = validate_spec(load_json(args.spec))
        result = verify_assets(spec, args.comfy_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    manifest = load_json(args.manifest)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValidationError("Unsupported motion-quality manifest")
    with matrix_lock(args.manifest.parent):
        if args.command == "plan-repeats":
            selection = load_json(args.selection)
            run_ids = append_repeat_records(manifest, selection, args.output_prefix)
            for run_id in run_ids:
                write_json_atomic(
                    args.manifest.parent / "prompts" / f"{run_id}.json",
                    manifest["records"][run_id]["api_prompt"],
                )
            write_json_atomic(args.manifest, manifest)
            print(f"Planned {len(run_ids)} repeat runs: {args.manifest.resolve()}")
            return 0
        if args.command == "enrich":
            if args.asr_model is None and args.face_model_root is None:
                raise ValidationError(
                    "enrich requires --asr-model and/or --face-model-root; models are never downloaded"
                )
            enrich_optional_metrics(
                manifest,
                comfy_root=args.comfy_root.resolve(),
                asr_model=(args.asr_model.resolve() if args.asr_model else None),
                asr_language=args.asr_language,
                asr_beam_size=args.asr_beam_size,
                face_model_root=(
                    args.face_model_root.resolve() if args.face_model_root else None
                ),
                face_model_name=args.face_model_name,
                face_sample_count=args.face_sample_count,
                face_detector_threshold=args.face_detector_threshold,
                only_missing=args.only_missing,
            )
            manifest["updated_at"] = utc_now()
            write_json_atomic(args.manifest, manifest)
            print(f"Enriched: {args.manifest.resolve()}")
            return 0
        if args.command == "run":
            spec = load_bound_spec(manifest)
            manifest["asset_verification"] = verify_assets(
                spec,
                args.comfy_root,
                cached=manifest.get("asset_verification"),
            )
            manifest["updated_at"] = utc_now()
            write_json_atomic(args.manifest, manifest)
            current_pid = _resolve_local_server_pid(args.server)
            if current_pid is None:
                raise ValidationError("Run mode requires one dedicated local ComfyUI listener")
            asyncio.run(
                run_pending(
                    manifest,
                    args.manifest,
                    server=args.server,
                    comfy_root=args.comfy_root,
                    device_index=args.device_index,
                    poll_interval=args.poll_interval,
                    baseline_seconds=args.baseline_seconds,
                    timeout_seconds=args.timeout_seconds,
                    preview_method=args.preview_method,
                    settle_seconds=args.settle_seconds,
                    ffmpeg=args.ffmpeg,
                    record_limit=args.record_limit,
                )
            )
            print(f"Updated: {args.manifest.resolve()}")
            return 0
        summary = summarize(manifest, args.ffmpeg)
        strict_report = final_strict_decode_report(args.manifest.parent)
        if strict_report is not None:
            write_json_atomic(args.manifest.parent / "strict_decode.json", strict_report)
        write_json_atomic(args.manifest.parent / "summary.json", summary)
        build_blind_package(manifest, args.manifest.parent, args.blind_seed)
        manifest["summary"] = summary
        manifest["quality_decision"] = summary["quality_decision"]
        manifest["updated_at"] = utc_now()
        write_json_atomic(args.manifest, manifest)
        print(f"Summary: {(args.manifest.parent / 'summary.json').resolve()}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
