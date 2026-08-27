#!/usr/bin/env python3
"""Run one missing MiniMax H3 Audio Refine Phase 2 baseline arm.

The default invocation is a read-only preflight. ``--confirm-run`` permits one prompt
on one private loopback ComfyUI process. Only ``base_ordinary8`` and ``base_refine4``
are accepted because the reviewed Turbo4 original and same-Turbo Refine4 arms already
exist and must not be regenerated merely to fill a matrix. There is no retry, stress,
parallel, cross-GPU, Frozen Cache, or user-service mutation path in this tool.
"""

from __future__ import annotations

import argparse
from array import array
import asyncio
import copy
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

import build_audio_refine_phase2_workflows as workflow_source
import run_audio_refine_smoke as smoke


SCHEMA = "t8.minimax_h3.audio_refine.phase2.single_arm.v1"
MAX_PROMPTS_PER_INVOCATION = 1
ALLOWED_ARMS = ("base_ordinary8", "base_refine4")
PROMPT = workflow_source.PROMPT
MIXED_AUDIO_PROMPT = (
    "雨夜的咖啡馆里，一位女性面对镜头自然地说：‘你在干嘛呢，我在这里呀，看看效果如何。’ "
    "窗外持续有清晰但不过响的雨声，室内持续播放轻柔爵士乐；她说完后把陶瓷杯放到木桌上，"
    "只出现一次清楚的杯底碰桌瞬态声。对白、雨声、音乐和瞬态彼此可辨，没有额外台词。"
)
I2VA_SPEECH_PROMPT = (
    "保持参考图中人物的面貌、服装和室内光线。她看向镜头，自然清楚地说："
    "‘你在干嘛呢，我在这里呀，看看效果如何。’ 语速平稳，近距离收音，安静室内，"
    "没有背景音乐、字幕或额外台词。"
)
WIDTH = 1056
HEIGHT = 608
FRAMES = 124
FPS = 24.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bounded_file_fingerprint(
    path: Path, *, sample_bytes: int = 1024 * 1024
) -> dict[str, Any]:
    """Fingerprint bounded first/middle/last samples without claiming a full hash."""

    path = Path(path)
    if isinstance(sample_bytes, bool) or int(sample_bytes) <= 0:
        raise ValueError("sample_bytes must be a positive integer")
    sample_bytes = int(sample_bytes)
    before = path.stat()
    size = int(before.st_size)
    if size <= sample_bytes:
        offsets = [0]
    else:
        offsets = sorted(
            {
                0,
                max(0, (size - sample_bytes) // 2),
                max(0, size - sample_bytes),
            }
        )

    digest = hashlib.sha256()
    digest.update(b"T8_BOUNDED_FILE_IDENTITY_V1\0")
    digest.update(str(size).encode("ascii"))
    sampled = 0
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(min(sample_bytes, max(0, size - offset)))
            digest.update(str(offset).encode("ascii") + b"\0")
            digest.update(str(len(payload)).encode("ascii") + b"\0")
            digest.update(payload)
            sampled += len(payload)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"asset changed while fingerprinting: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "mtime_ns": int(before.st_mtime_ns),
        "sample_bytes_per_offset": sample_bytes,
        "sample_offsets": offsets,
        "bytes_sampled": sampled,
        "bounded_sample_sha256": digest.hexdigest().upper(),
        "full_file_sha256": None,
        "claim": "bounded identity sample; not a full-file hash",
    }


def _asset_identity(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    sample_bytes = int(args.fingerprint_sample_mib) * 1024 * 1024
    identity = {
        role: bounded_file_fingerprint(path, sample_bytes=sample_bytes)
        for role, path in sorted(smoke._model_paths(args.comfy_root.resolve()).items())
        if path.is_file()
    }
    reference_image = getattr(args, "reference_image", None)
    if reference_image is not None and Path(reference_image).is_file():
        identity["reference_image"] = bounded_file_fingerprint(
            Path(reference_image), sample_bytes=sample_bytes
        )
    return identity


def build_prompt(
    *,
    arm: str,
    run_id: str,
    seed: int,
    width: int = WIDTH,
    height: int = HEIGHT,
    frames: int = FRAMES,
    audio_denoise: float = 0.50,
    checkpointed: bool = False,
    prompt_text: str = PROMPT,
    task_type: str = "T2VA",
    first_image_name: str | None = None,
) -> dict[str, Any]:
    if arm not in ALLOWED_ARMS:
        raise ValueError(f"unsupported Phase 2 arm: {arm}")
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Phase 2 width and height must be positive multiples of 32")
    if frames != FRAMES:
        raise ValueError(f"Phase 2 review contract requires exactly {FRAMES} frames")
    if not any(
        math.isclose(float(audio_denoise), point, rel_tol=0.0, abs_tol=1.0e-9)
        for point in (0.35, 0.50)
    ):
        raise ValueError("Phase 2 audio_denoise must be 0.35 or 0.50")

    source_key = "base_ordinary8" if arm == "base_ordinary8" else "base_without_turbo"
    prompt = copy.deepcopy(workflow_source.build_prompts(seed)[source_key])
    prompt["8"]["inputs"].update(
        {
            "prompt": str(prompt_text),
            "width": int(width),
            "height": int(height),
            "length": int(frames),
            "task_type": str(task_type),
        }
    )
    if task_type == "I2VA":
        if not first_image_name:
            raise ValueError("I2VA Phase 2 scenario requires first_image_name")
        prompt["35"] = {
            "class_type": "LoadImage",
            "inputs": {"image": str(first_image_name)},
        }
        prompt["8"]["inputs"]["first_frame"] = ["35", 0]
    elif task_type != "T2VA":
        raise ValueError("Phase 2 scenario task_type must be T2VA or I2VA")
    prefix_root = "MiniMaxH3_AudioRefine_Phase2"
    if arm == "base_ordinary8":
        prompt["15"]["inputs"]["filename_prefix"] = (
            f"{prefix_root}/{run_id}_{arm}"
        )
    else:
        prompt["18"]["inputs"]["audio_denoise"] = float(audio_denoise)
        prompt["25"]["inputs"]["video_frame_count"] = int(frames)
        if checkpointed:
            latent_sources = {
                "original": ["12", 0],
                "candidate": ["21", 0],
                "selected": ["25", 0],
            }
            save_nodes = {"original": "15", "candidate": "24", "selected": "28"}
            report_nodes = {"original": "29", "candidate": "30", "selected": "31"}
            manifest_nodes = {"original": "32", "candidate": "33", "selected": "34"}
            for node_id in ("14", "23", "26", "27"):
                prompt.pop(node_id, None)
            for label, node_id in save_nodes.items():
                prompt[node_id] = {
                    "class_type": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
                    "inputs": {
                        "av_latent": latent_sources[label],
                        "filename_prefix": (
                            f"audio_refine_phase2/{run_id}_{arm}_{label}"
                        ),
                        "checkpoint_id": f"{run_id}:{arm}:{label}",
                        "confirm_save": True,
                        "verify_after_write": True,
                        "hash_chunk_megabytes": 8,
                    },
                }
                prompt[report_nodes[label]] = {
                    "class_type": "PreviewAny",
                    "inputs": {"source": [node_id, 5]},
                }
                prompt[manifest_nodes[label]] = {
                    "class_type": "PreviewAny",
                    "inputs": {"source": [node_id, 4]},
                }
        else:
            for node_id, label in (
                ("15", "original"),
                ("24", "candidate"),
                ("28", "selected"),
            ):
                prompt[node_id]["inputs"]["filename_prefix"] = (
                    f"{prefix_root}/{run_id}_{arm}_{label}"
                )
    return prompt


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = dict(smoke.preflight(args))
    report.update(
        {
            "schema": f"{SCHEMA}.preflight",
            "arm": args.arm,
            "fixed_generation_contract": {
                "scenario": args.scenario,
                "prompt": args.prompt_text,
                "task_type": args.task_type,
                "reference_image": (
                    str(args.reference_image) if args.reference_image else None
                ),
                "seed": int(args.seed),
                "width": WIDTH,
                "height": HEIGHT,
                "frames": FRAMES,
                "fps": FPS,
                "audio_denoise": (
                    None if args.arm == "base_ordinary8" else float(args.audio_denoise)
                ),
                "declared_nfe": 8,
                "isolated_comfy_reserve_vram_gib": float(args.reserve_vram_gib),
            },
            "maximum_prompts": MAX_PROMPTS_PER_INVOCATION,
            "existing_arms_intentionally_not_run": [
                "turbo4_original",
                "same_turbo_stack",
            ],
        }
    )
    if report.get("checks", {}).get("required_paths_present"):
        try:
            report["bounded_asset_identity"] = _asset_identity(args)
        except (OSError, RuntimeError, ValueError) as error:
            report["ready_for_real_run"] = False
            report["status"] = "ABSTAIN_ASSET_IDENTITY_UNAVAILABLE"
            report["bounded_asset_identity_error"] = (
                f"{type(error).__name__}: {error}"
            )
    if args.reference_image is not None and not args.reference_image.is_file():
        report["ready_for_real_run"] = False
        report["status"] = "ABSTAIN_REFERENCE_IMAGE_MISSING"
        report.setdefault("checks", {})["reference_image_present"] = False
    elif args.reference_image is not None:
        report.setdefault("checks", {})["reference_image_present"] = True
    return report


def _history_json(phase: Mapping[str, Any], node_id: str) -> dict[str, Any] | None:
    text = _history_text(phase, node_id)
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _history_text(phase: Mapping[str, Any], node_id: str) -> str | None:
    executed_outputs = phase.get("executed_outputs")
    history = phase.get("history")
    history_outputs = history.get("outputs") if isinstance(history, Mapping) else None
    candidates = (
        executed_outputs.get(node_id) if isinstance(executed_outputs, Mapping) else None,
        history_outputs.get(node_id) if isinstance(history_outputs, Mapping) else None,
    )
    text = None
    for output in candidates:
        if not isinstance(output, Mapping):
            continue
        for key in ("text", "report_json"):
            values = output.get(key)
            if isinstance(values, list) and values and isinstance(values[-1], str):
                text = values[-1]
                break
        if text:
            break
    return text


def _checkpoint_records(phase: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    report_nodes = {"original": "29", "candidate": "30", "selected": "31"}
    manifest_nodes = {"original": "32", "candidate": "33", "selected": "34"}
    records: dict[str, dict[str, Any]] = {}
    for label in ("original", "candidate", "selected"):
        report = _history_json(phase, report_nodes[label])
        manifest_json = _history_text(phase, manifest_nodes[label])
        if not report or report.get("status") != "SAVED_VERIFIED":
            raise RuntimeError(f"{label} latent checkpoint was not saved and verified")
        if not manifest_json:
            raise RuntimeError(f"{label} latent checkpoint manifest was not captured")
        manifest = json.loads(manifest_json)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{label} latent checkpoint manifest is not an object")
        records[label] = {
            **report,
            "manifest_json": manifest_json,
            "manifest": manifest,
        }
    return records


def build_checkpoint_decode_prompt(
    *, record: Mapping[str, Any], run_id: str, label: str
) -> dict[str, Any]:
    prefix_root = "MiniMaxH3_AudioRefine_Phase2_Decode"
    return {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "3": {
            "class_type": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
            "inputs": {
                "checkpoint_path": str(record["checkpoint_path"]),
                "expected_manifest_json": str(record["manifest_json"]),
                "expected_file_sha256": str(record["file_sha256"]),
                "hash_chunk_megabytes": 8,
            },
        },
        "4": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["4", 0],
                "filename_prefix": f"{prefix_root}/{run_id}_{label}_frame",
            },
        },
        "6": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["4", 1],
                "filename_prefix": f"{prefix_root}/{run_id}_{label}_audio",
            },
        },
        "7": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["3", 7]},
        },
    }


def _run_private_prompt(
    *,
    args: argparse.Namespace,
    run_root: Path,
    prompt: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    prompt_path = run_root / f"prompt_{label}.json"
    prompt_path.write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    original_command = smoke.shared._server_command
    smoke.shared._server_command = smoke._server_command
    server = smoke.shared.IsolatedServer(args, run_root, label)
    monitor = smoke.GpuPeakMonitor()
    phase: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    pid: int | None = None
    monitor.start()
    try:
        pid = server.start()
        phase = asyncio.run(
            _submit_prompt_capture(
                server=f"http://{args.host}:{args.port}",
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        server.stop()
        smoke.shared._server_command = original_command
        gpu = monitor.stop()
    result = {"pid": pid, "phase": phase, "error": error, "gpu": gpu}
    (run_root / f"phase_{label}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _decode_output_paths(
    run_root: Path, *, run_id: str, label: str
) -> tuple[list[Path], Path]:
    output_root = run_root / "output" / "MiniMaxH3_AudioRefine_Phase2_Decode"
    frames = sorted(output_root.glob(f"{run_id}_{label}_frame*.png"))
    audio = smoke.shared._latest_file(output_root, f"{run_id}_{label}_audio*.flac")
    return frames, audio


def _encode_png_audio_to_mp4(
    *,
    frames: list[Path],
    audio: Path,
    output: Path,
    ffmpeg: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: float = FPS,
) -> dict[str, Any]:
    from PIL import Image

    if len(frames) != FRAMES:
        raise RuntimeError(f"expected {FRAMES} decoded PNG frames, found {len(frames)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-frames:v",
        str(FRAMES),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for path in frames:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (width, height):
                    raise RuntimeError(
                        f"decoded frame {path.name} is {rgb.size}, expected {(width, height)}"
                    )
                process.stdin.write(rgb.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
    except Exception:
        with suppress(Exception):
            if process.stdin is not None:
                process.stdin.close()
        with suppress(Exception):
            process.kill()
        process.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"single-thread ffmpeg encode failed: {stderr[-4000:]}")
    return {
        "command": command,
        "returncode": returncode,
        "frame_count": len(frames),
        "audio_path": str(audio),
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
    }


async def _submit_prompt_capture(
    *, server: str, prompt: Mapping[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    """Submit one prompt and retain ComfyUI v3 executed-node outputs.

    Current ComfyUI builds may return ``history: null`` even when output nodes
    successfully published their UI report through the live ``executed`` event.
    Capturing both routes prevents a completed render from being discarded solely
    because history persistence changed.
    """

    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("aiohttp is required in the ComfyUI Python environment") from error

    client_id = uuid.uuid4().hex
    requested_prompt_id = str(uuid.uuid4())
    terminal = None
    events: list[dict[str, Any]] = []
    executed_outputs: dict[str, Any] = {}
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(
        total=None, connect=20, sock_connect=20, sock_read=None
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await smoke.shared._json_request(session, "GET", f"{server}/system_stats")
        ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(
            f"{ws_url}/ws?clientId={client_id}", heartbeat=30
        ) as websocket:
            submitted = await smoke.shared._json_request(
                session,
                "POST",
                f"{server}/prompt",
                json={
                    "prompt": dict(prompt),
                    "client_id": client_id,
                    "prompt_id": requested_prompt_id,
                },
            )
            prompt_id = str(submitted.get("prompt_id") or "")
            if prompt_id != requested_prompt_id:
                raise RuntimeError(
                    f"ComfyUI returned unexpected prompt_id {prompt_id!r}"
                )
            deadline = time.monotonic() + timeout_seconds
            while terminal is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for prompt {prompt_id}")
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(), timeout=min(1.0, remaining)
                    )
                except asyncio.TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    raise RuntimeError(
                        "ComfyUI WebSocket closed before a terminal event"
                    )
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                with suppress(json.JSONDecodeError):
                    packet = json.loads(message.data)
                    event_type = packet.get("type")
                    data = (
                        packet.get("data")
                        if isinstance(packet.get("data"), dict)
                        else {}
                    )
                    if data.get("prompt_id") not in {None, prompt_id}:
                        continue
                    node = str(data.get("node") or "")
                    if event_type in {
                        "executing",
                        "executed",
                        "progress",
                        "progress_state",
                        *smoke.shared.TERMINAL_EVENTS,
                    }:
                        events.append(
                            {
                                "elapsed_seconds": round(
                                    time.monotonic() - started, 4
                                ),
                                "type": event_type,
                                "node": node or None,
                            }
                        )
                    if event_type == "executed" and node:
                        executed_outputs[node] = data.get("output")
                    if (
                        event_type in smoke.shared.TERMINAL_EVENTS
                        and data.get("prompt_id") == prompt_id
                    ):
                        terminal = {"type": event_type, "data": data}
        history = await smoke.shared._json_request(
            session, "GET", f"{server}/history/{prompt_id}"
        )
    return {
        "prompt_id": prompt_id,
        "terminal": terminal,
        "history": history.get(prompt_id),
        "executed_outputs": executed_outputs,
        "events": events,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }


def _pcm_values_contract(values: array) -> dict[str, Any]:
    if len(values) == 0 or len(values) % 2:
        return {
            "passed": False,
            "finite": False,
            "stereo_interleaved": False,
            "reason": "empty or non-stereo-interleaved PCM",
        }
    finite = all(math.isfinite(float(value)) for value in values)
    if not finite:
        return {
            "passed": False,
            "finite": False,
            "stereo_interleaved": True,
            "reason": "decoded PCM contains NaN or Infinity",
        }
    left_energy = 0.0
    right_energy = 0.0
    peak = 0.0
    clipped = 0
    for index in range(0, len(values), 2):
        left = float(values[index])
        right = float(values[index + 1])
        left_energy += left * left
        right_energy += right * right
        local_peak = max(abs(left), abs(right))
        peak = max(peak, local_peak)
        clipped += int(abs(left) >= 0.999) + int(abs(right) >= 0.999)
    frames = len(values) // 2
    left_rms = math.sqrt(left_energy / frames)
    right_rms = math.sqrt(right_energy / frames)
    maximum_rms = max(left_rms, right_rms)
    minimum_rms = min(left_rms, right_rms)
    channel_ratio = minimum_rms / maximum_rms if maximum_rms > 1.0e-8 else 0.0
    clipping_ratio = clipped / len(values)
    non_silent = maximum_rms >= 1.0e-5
    channel_collapse = non_silent and channel_ratio < 0.01
    clipping_suspected = clipping_ratio > 0.001
    passed = finite and non_silent and not channel_collapse and not clipping_suspected
    return {
        "passed": passed,
        "finite": finite,
        "stereo_interleaved": True,
        "sample_frames": frames,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "minimum_to_maximum_channel_rms_ratio": channel_ratio,
        "channel_collapse_suspected": channel_collapse,
        "non_silent": non_silent,
        "peak_absolute": peak,
        "clipping_sample_ratio": clipping_ratio,
        "clipping_suspected": clipping_suspected,
    }


def decoded_pcm_contract(path: Path, *, ffmpeg: str) -> dict[str, Any]:
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "32000",
        "-ac",
        "2",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        return {
            "passed": False,
            "returncode": int(completed.returncode),
            "stderr": completed.stderr.decode("utf-8", errors="replace")[-4000:],
        }
    values = array("f")
    try:
        values.frombytes(completed.stdout)
    except (EOFError, ValueError) as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}
    report = _pcm_values_contract(values)
    report["returncode"] = int(completed.returncode)
    report["decoded_bytes"] = len(completed.stdout)
    return report


def _output_paths(run_root: Path, *, run_id: str, arm: str) -> dict[str, Path]:
    output_root = run_root / "output" / "MiniMaxH3_AudioRefine_Phase2"
    if arm == "base_ordinary8":
        return {
            "ordinary8": smoke.shared._latest_file(
                output_root, f"{run_id}_{arm}*.mp4"
            )
        }
    return {
        label: smoke.shared._latest_file(
            output_root, f"{run_id}_{arm}_{label}*.mp4"
        )
        for label in ("original", "candidate", "selected")
    }


def _write_report(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _run_checkpointed_refine_arm(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    first_image_name = None
    if args.reference_image is not None:
        input_root = run_root / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        first_image_name = f"{run_id}_reference{args.reference_image.suffix.lower()}"
        shutil.copy2(args.reference_image, input_root / first_image_name)
    prompt = build_prompt(
        arm=args.arm,
        run_id=run_id,
        seed=args.seed,
        audio_denoise=args.audio_denoise,
        checkpointed=True,
        prompt_text=args.prompt_text,
        task_type=args.task_type,
        first_image_name=first_image_name,
    )
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    post_gate = preflight(args)
    identity_unchanged = (
        preflight_report.get("bounded_asset_identity")
        == post_gate.get("bounded_asset_identity")
    )
    if not post_gate.get("ready_for_real_run") or not identity_unchanged:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_RESOURCE_OR_ASSET_CHANGED_BEFORE_START",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "post_gate": post_gate,
                "asset_identity_unchanged": identity_unchanged,
                "process_ids": [],
                "sampling_prompts_submitted": 0,
                "decode_only_prompts_submitted": 0,
            },
        )

    started = time.monotonic()
    runtime_error: dict[str, str] | None = None
    generation = _run_private_prompt(
        args=args,
        run_root=run_root,
        prompt=prompt,
        label="audio_refine_phase2_generation",
    )
    generation_phase = generation.get("phase") or {}
    generation_success = bool(
        generation_phase.get("terminal", {}).get("type") == "execution_success"
    )
    if generation.get("error"):
        runtime_error = dict(generation["error"])

    records: dict[str, dict[str, Any]] = {}
    if generation_success and runtime_error is None:
        try:
            records = _checkpoint_records(generation_phase)
        except Exception as exc:
            runtime_error = {"type": type(exc).__name__, "message": str(exc)}

    decode_results: dict[str, Any] = {}
    decode_reports: dict[str, Any] = {}
    decoded_assets: dict[str, Any] = {}
    encode_reports: dict[str, Any] = {}
    review_root = run_root / "review_media"
    if records and runtime_error is None:
        for label in ("original", "candidate"):
            decode_prompt = build_checkpoint_decode_prompt(
                record=records[label], run_id=run_id, label=label
            )
            result = _run_private_prompt(
                args=args,
                run_root=run_root,
                prompt=decode_prompt,
                label=f"audio_refine_phase2_decode_{label}",
            )
            decode_results[label] = result
            phase = result.get("phase") or {}
            success = bool(
                phase.get("terminal", {}).get("type") == "execution_success"
            )
            if result.get("error") or not success:
                error = result.get("error") or {
                    "type": "RuntimeError",
                    "message": f"{label} checkpoint decode did not reach execution_success",
                }
                runtime_error = dict(error)
                break
            try:
                decode_reports[label] = _history_json(phase, "7")
                frames, audio = _decode_output_paths(
                    run_root, run_id=run_id, label=label
                )
                output = review_root / f"{run_id}_{args.arm}_{label}.mp4"
                encode_reports[label] = _encode_png_audio_to_mp4(
                    frames=frames,
                    audio=audio,
                    output=output,
                    ffmpeg=str(preflight_report["ffmpeg"]),
                )
                decoded_assets[label] = {
                    "frames": [str(path) for path in frames],
                    "audio": str(audio),
                    "video": str(output),
                }
            except Exception as exc:
                runtime_error = {"type": type(exc).__name__, "message": str(exc)}
                break

    if runtime_error is None and "original" in decoded_assets:
        selected = review_root / f"{run_id}_{args.arm}_selected.mp4"
        shutil.copy2(decoded_assets["original"]["video"], selected)
        decoded_assets["selected"] = {
            "source": "byte_exact_copy_of_original_after_quality_gate_abstain",
            "video": str(selected),
        }
        encode_reports["selected"] = {
            "copied_from": decoded_assets["original"]["video"],
            "output_path": str(selected),
            "output_bytes": selected.stat().st_size,
        }

    media: dict[str, Any] = {}
    pcm: dict[str, Any] = {}
    media_checks: dict[str, bool] = {}
    if runtime_error is None:
        try:
            for label in ("original", "candidate", "selected"):
                path = Path(decoded_assets[label]["video"])
                media[label] = smoke.shared.media_report(
                    path,
                    ffmpeg=str(preflight_report["ffmpeg"]),
                    ffprobe=str(preflight_report["ffprobe"]),
                )
                arm_checks = smoke._media_checks(
                    media[label], width=WIDTH, height=HEIGHT, frames=FRAMES
                )
                media_checks.update(
                    {
                        f"{label}_{name}": bool(value)
                        for name, value in arm_checks.items()
                    }
                )
                pcm[label] = decoded_pcm_contract(
                    path, ffmpeg=str(preflight_report["ffmpeg"])
                )
                media_checks[f"{label}_pcm_contract"] = bool(pcm[label]["passed"])
        except Exception as exc:
            runtime_error = {"type": type(exc).__name__, "message": str(exc)}

    events = generation_phase.get("events", [])
    executed_nodes = {
        str(event.get("node"))
        for event in events
        if isinstance(event, Mapping)
        and event.get("type") == "executing"
        and event.get("node") is not None
    }
    reports = {
        "model_route": _history_json(generation_phase, "17"),
        "phase2_plan": _history_json(generation_phase, "18"),
        "phase2_setup": _history_json(generation_phase, "19"),
        "quality_gate": _history_json(generation_phase, "25"),
    }
    checkpoint_files_exist = bool(records) and all(
        Path(record.get("absolute_path", "")).is_file() for record in records.values()
    )
    generation_gpu = generation.get("gpu") or {}
    decode_gpu_reports = [
        result.get("gpu") or {} for result in decode_results.values()
    ]
    checks = {
        "one_sampling_prompt_submitted": bool(generation_phase.get("prompt_id")),
        "generation_execution_success": generation_success,
        "asset_identity_unchanged": identity_unchanged,
        "minimum_generation_free_vram_at_least_512_mib": bool(
            generation_gpu.get("minimum_free_mib") is not None
            and int(generation_gpu["minimum_free_mib"]) >= 512
        ),
        "three_checkpoints_saved_verified": len(records) == 3,
        "checkpoint_files_exist": checkpoint_files_exist,
        "selected_checkpoint_matches_original": bool(
            records
            and records["selected"].get("content_sha256")
            == records["original"].get("content_sha256")
        ),
        "two_decode_only_prompts_succeeded": len(decode_results) == 2
        and all(
            (result.get("phase") or {}).get("terminal", {}).get("type")
            == "execution_success"
            for result in decode_results.values()
        ),
        "decode_resume_verified": len(decode_reports) == 2
        and all(
            report
            and report.get("resume_verified") is True
            and report.get("external_manifest_verified") is True
            for report in decode_reports.values()
        ),
        "decode_processes_keep_512_mib_free": len(decode_gpu_reports) == 2
        and all(
            report.get("minimum_free_mib") is not None
            and int(report["minimum_free_mib"]) >= 512
            for report in decode_gpu_reports
        ),
        "first_sampler_completed": generation_success and "12" in executed_nodes,
        "refine_sampler_completed": generation_success and "20" in executed_nodes,
        "model_route_reported_allow": bool(
            reports["model_route"]
            and reports["model_route"].get("decision") == "ALLOW"
        ),
        "phase2_plan_reported_allow": bool(
            reports["phase2_plan"]
            and reports["phase2_plan"].get("decision") == "ALLOW"
        ),
        "phase2_setup_reported_allow": bool(
            reports["phase2_setup"]
            and reports["phase2_setup"].get("decision") == "ALLOW"
        ),
        "candidate_mechanically_eligible": bool(
            reports["quality_gate"]
            and reports["quality_gate"].get("candidate_mechanically_eligible")
        ),
        "quality_gate_defaulted_to_original": bool(
            reports["quality_gate"]
            and reports["quality_gate"].get("decision")
            == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
            and reports["quality_gate"].get("candidate_selected") is False
        ),
        **media_checks,
    }
    original = media.get("original", {})
    selected = media.get("selected", {})
    checks["selected_video_relocked_to_original"] = bool(
        original
        and selected
        and original.get("decoded_video", {}).get("sha256")
        == selected.get("decoded_video", {}).get("sha256")
    )
    checks["selected_audio_defaulted_to_original"] = bool(
        original
        and selected
        and original.get("decoded_audio", {}).get("sha256")
        == selected.get("decoded_audio", {}).get("sha256")
    )
    passed = bool(checks) and all(checks.values()) and runtime_error is None
    process_ids = [generation.get("pid")] + [
        result.get("pid") for result in decode_results.values()
    ]
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "arm": args.arm,
            "run_id": run_id,
            "run_root": str(run_root),
            "preflight": dict(preflight_report),
            "post_gate": post_gate,
            "process_ids": [pid for pid in process_ids if pid is not None],
            "sampling_prompts_submitted": int(bool(generation_phase.get("prompt_id"))),
            "decode_only_prompts_submitted": sum(
                int(bool((result.get("phase") or {}).get("prompt_id")))
                for result in decode_results.values()
            ),
            "checkpoint_architecture": {
                "generation_process_saves_native_av_latents": True,
                "h3_process_stopped_before_decode": True,
                "decode_processes": "one original and one candidate, strictly serial",
                "selected_media": "byte-exact copy of original after default ABSTAIN",
                "retry_or_stress": False,
            },
            "generation_phase": generation,
            "checkpoints": records,
            "decode_phases": decode_results,
            "decode_reports": decode_reports,
            "decoded_assets": decoded_assets,
            "encode_reports": encode_reports,
            "runtime_error": runtime_error,
            "generation_contract": {
                "scenario": args.scenario,
                "prompt": args.prompt_text,
                "task_type": args.task_type,
                "reference_image": (
                    str(args.reference_image) if args.reference_image else None
                ),
                "seed": args.seed,
                "width": WIDTH,
                "height": HEIGHT,
                "frames": FRAMES,
                "fps": FPS,
                "declared_total_nfe": 8,
                "executed_schedule_nfe_after_success": (
                    8
                    if checks["first_sampler_completed"]
                    and checks["refine_sampler_completed"]
                    else 0
                ),
                "first_pass_nfe": 4,
                "refine_nfe": 4,
                "audio_denoise": args.audio_denoise,
                "isolated_comfy_reserve_vram_gib": float(args.reserve_vram_gib),
                "training_distribution_equivalence_claim": False,
            },
            "runtime_reports": reports,
            "media": media,
            "decoded_pcm_contracts": pcm,
            "gpu": {
                "generation": generation_gpu,
                "decode": {
                    label: result.get("gpu")
                    for label, result in decode_results.items()
                },
                "final": smoke.shared.gpu_memory_mib(),
            },
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "checks": checks,
            "boundary": (
                "One sampling prompt is followed by two decode-only prompts in separate, "
                "strictly serial private processes. Native AV checkpoints make decoding "
                "recoverable without repeating sampling. PASS remains mechanical and does "
                "not establish transcript, speaker, performance, lip-sync, mixed-audio, "
                "quality, repeat stability, or universal 16GiB safety."
            ),
        },
    )


def run_real_arm(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    if args.arm == "base_refine4":
        return _run_checkpointed_refine_arm(args, preflight_report)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    prompt = build_prompt(
        arm=args.arm,
        run_id=run_id,
        seed=args.seed,
        audio_denoise=args.audio_denoise,
    )
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    post_gate = preflight(args)
    identity_unchanged = (
        preflight_report.get("bounded_asset_identity")
        == post_gate.get("bounded_asset_identity")
    )
    if not post_gate.get("ready_for_real_run") or not identity_unchanged:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_RESOURCE_OR_ASSET_CHANGED_BEFORE_START",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "post_gate": post_gate,
                "asset_identity_unchanged": identity_unchanged,
                "process_ids": [],
                "prompts_submitted": 0,
            },
        )

    original_command = smoke.shared._server_command
    smoke.shared._server_command = smoke._server_command
    server = smoke.shared.IsolatedServer(args, run_root, "audio_refine_phase2")
    monitor = smoke.GpuPeakMonitor()
    process_ids: list[int] = []
    phase: dict[str, Any] | None = None
    runtime_error: dict[str, str] | None = None
    started = time.monotonic()
    monitor.start()
    try:
        process_ids.append(server.start())
        phase = asyncio.run(
            _submit_prompt_capture(
                server=f"http://{args.host}:{args.port}",
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except Exception as error:
        runtime_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.stop()
        smoke.shared._server_command = original_command
        gpu_monitor = monitor.stop()

    execution_success = bool(
        phase and phase.get("terminal", {}).get("type") == "execution_success"
    )
    media: dict[str, Any] = {}
    pcm: dict[str, Any] = {}
    media_checks: dict[str, bool] = {}
    if execution_success:
        try:
            for label, path in _output_paths(
                run_root, run_id=run_id, arm=args.arm
            ).items():
                media[label] = smoke.shared.media_report(
                    path,
                    ffmpeg=str(preflight_report["ffmpeg"]),
                    ffprobe=str(preflight_report["ffprobe"]),
                )
                checks = smoke._media_checks(
                    media[label], width=WIDTH, height=HEIGHT, frames=FRAMES
                )
                media_checks.update(
                    {f"{label}_{name}": bool(value) for name, value in checks.items()}
                )
                pcm[label] = decoded_pcm_contract(
                    path, ffmpeg=str(preflight_report["ffmpeg"])
                )
                media_checks[f"{label}_pcm_contract"] = bool(pcm[label]["passed"])
        except Exception as error:
            runtime_error = {"type": type(error).__name__, "message": str(error)}

    events = (phase or {}).get("events", [])
    executed_nodes = {
        str(event.get("node"))
        for event in events
        if isinstance(event, Mapping)
        and event.get("type") == "executing"
        and event.get("node") is not None
    }
    reports: dict[str, Any] = {}
    checks = {
        "one_isolated_process": len(process_ids) == 1,
        "one_prompt_submitted": bool(phase and phase.get("prompt_id")),
        "execution_success": execution_success,
        "asset_identity_unchanged": identity_unchanged,
        "minimum_runtime_free_vram_at_least_512_mib": bool(
            gpu_monitor.get("minimum_free_mib") is not None
            and int(gpu_monitor["minimum_free_mib"]) >= 512
        ),
        **media_checks,
    }
    if args.arm == "base_ordinary8":
        checks["ordinary8_sampler_completed"] = execution_success and "12" in executed_nodes
        executed_nfe = 8 if checks["ordinary8_sampler_completed"] else 0
    else:
        reports = {
            "model_route": _history_json(phase or {}, "17"),
            "phase2_plan": _history_json(phase or {}, "18"),
            "phase2_setup": _history_json(phase or {}, "19"),
            "quality_gate": _history_json(phase or {}, "25"),
        }
        checks.update(
            {
                "first_sampler_completed": execution_success and "12" in executed_nodes,
                "refine_sampler_completed": execution_success and "20" in executed_nodes,
                "model_route_reported_allow": bool(
                    reports["model_route"]
                    and reports["model_route"].get("decision") == "ALLOW"
                ),
                "phase2_plan_reported_allow": bool(
                    reports["phase2_plan"]
                    and reports["phase2_plan"].get("decision") == "ALLOW"
                ),
                "phase2_setup_reported_allow": bool(
                    reports["phase2_setup"]
                    and reports["phase2_setup"].get("decision") == "ALLOW"
                ),
                "candidate_mechanically_eligible": bool(
                    reports["quality_gate"]
                    and reports["quality_gate"].get("candidate_mechanically_eligible")
                ),
                "quality_gate_defaulted_to_original": bool(
                    reports["quality_gate"]
                    and reports["quality_gate"].get("decision")
                    == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
                    and reports["quality_gate"].get("candidate_selected") is False
                ),
            }
        )
        original = media.get("original", {})
        selected = media.get("selected", {})
        checks["selected_video_relocked_to_original"] = bool(
            original
            and selected
            and original.get("decoded_video", {}).get("sha256")
            == selected.get("decoded_video", {}).get("sha256")
        )
        checks["selected_audio_defaulted_to_original"] = bool(
            original
            and selected
            and original.get("decoded_audio", {}).get("sha256")
            == selected.get("decoded_audio", {}).get("sha256")
        )
        executed_nfe = (
            8
            if checks["first_sampler_completed"] and checks["refine_sampler_completed"]
            else 0
        )

    passed = bool(checks) and all(checks.values()) and runtime_error is None
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "arm": args.arm,
            "run_id": run_id,
            "run_root": str(run_root),
            "preflight": dict(preflight_report),
            "post_gate": post_gate,
            "process_ids": process_ids,
            "prompts_submitted": int(bool(phase and phase.get("prompt_id"))),
            "phase": phase,
            "runtime_error": runtime_error,
            "generation_contract": {
                "prompt": PROMPT,
                "seed": args.seed,
                "width": WIDTH,
                "height": HEIGHT,
                "frames": FRAMES,
                "fps": FPS,
                "declared_total_nfe": 8,
                "executed_schedule_nfe_after_success": executed_nfe,
                "first_pass_nfe": 8 if args.arm == "base_ordinary8" else 4,
                "refine_nfe": 0 if args.arm == "base_ordinary8" else 4,
                "audio_denoise": (
                    None if args.arm == "base_ordinary8" else args.audio_denoise
                ),
                "isolated_comfy_reserve_vram_gib": float(args.reserve_vram_gib),
                "training_distribution_equivalence_claim": False,
            },
            "runtime_reports": reports,
            "media": media,
            "decoded_pcm_contracts": pcm,
            "gpu": {"monitor": gpu_monitor, "final": smoke.shared.gpu_memory_mib()},
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "checks": checks,
            "boundary": (
                "One missing arm, one private process and one prompt only. Asset identity "
                "uses bounded first/middle/last samples rather than a full-file hash. PASS "
                "is mechanical and does not establish transcript, speaker, performance, "
                "lip-sync, mixed-audio, quality, training-distribution equivalence, repeat "
                "stability, or universal 16GiB safety."
            ),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument(
        "--scenario",
        choices=("baseline_dialogue", "mixed_audio", "i2va_speech"),
        default="baseline_dialogue",
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument("--plugin-root", type=Path, default=project_root)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=project_root / "artifacts" / "audio-refine-phase2-20260826",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument(
        "--reserve-vram-gib",
        type=float,
        default=4.0,
        help=(
            "VRAM headroom reserved by the isolated ComfyUI process. Phase 2 uses "
            "the project's previously validated conservative 4GiB H3 policy after "
            "the 1.0GiB and 1.5GiB routes fell below the fixed 512MiB whole-device "
            "safety gate."
        ),
    )
    parser.add_argument("--seed", type=int, default=2608260404)
    parser.add_argument("--audio-denoise", type=float, choices=(0.35, 0.50), default=0.50)
    parser.add_argument("--fingerprint-sample-mib", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.width, args.height, args.frames = WIDTH, HEIGHT, FRAMES
    if args.scenario == "mixed_audio":
        args.prompt_text = MIXED_AUDIO_PROMPT
        args.task_type = "T2VA"
        args.reference_image = None
    elif args.scenario == "i2va_speech":
        args.prompt_text = I2VA_SPEECH_PROMPT
        args.task_type = "I2VA"
        if args.reference_image is None:
            args.reference_image = args.comfy_root / "input" / "10A.jpg"
    else:
        args.prompt_text = PROMPT
        args.task_type = "T2VA"
        args.reference_image = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    preflight_path = args.artifact_root / f"latest_preflight_{args.arm}.json"
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "arm": args.arm,
                    "ready_for_real_run": report["ready_for_real_run"],
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "arm": args.arm,
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = run_real_arm(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": args.arm,
                "passed": result["passed"],
                "run_root": result["run_root"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
