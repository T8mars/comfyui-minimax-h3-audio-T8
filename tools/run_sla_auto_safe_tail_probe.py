#!/usr/bin/env python3
"""Run one guarded 124-frame SLA auto-safe tail-stability validation.

This is a single-render diagnostic, not a stress test.  It reuses the exact prompt graph
embedded in the previously failed 736x416x124 SLA render, changes only the output prefix,
and requires the public ``apply_lightx2v_sla`` mode to mechanically audit as four fully
dense attention forwards.  Visual acceptance remains a separate human decision.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping
import uuid

from PIL import Image


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402


SCHEMA = "t8.minimax_h3.sla_auto_safe_tail_probe.v1"
WIDTH = 736
HEIGHT = 416
FRAME_COUNT = 124
STEPS = 4
SHIFT_VIDEO = 6.0
SHIFT_AUDIO = 3.0
SEED = 2608224201
EXPECTED_AUDIT_STATUS = "auto_safe_short_sequence_dense_fallback_verified"
EXPECTED_LORA_SHA256 = "5CAE6DF40A06EA825F85FC8876C9EA1C9692C833A9AF07BB8B3BAC9CE2A71BAC"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_prompt_path(project_root: Path) -> Path:
    return (
        project_root
        / "artifacts"
        / "sla-dense-control-20260826"
        / "output"
        / "MiniMaxH3_SLA_Diagnostic"
        / "basic_sla_default_736x416_124f_00001.png"
    )


def _required_paths(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    return {
        "comfy_main": args.comfy_root / "main.py",
        "python": args.python,
        "project": args.project_root,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "source_prompt_png": _source_prompt_path(args.project_root),
        "first_frame": args.comfy_root / "input" / "codex_prompt_relay_fl2va_first.png",
        "last_frame": args.comfy_root / "input" / "codex_prompt_relay_fl2va_last.png",
        "base": models / "diffusion_models" / "minimax_h3_fl2va_int8_convrot.safetensors",
        "clip": models / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "sla_lora": models
        / "loras"
        / "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors",
    }


def _load_exact_prompt(source_png: Path, run_id: str) -> dict[str, Any]:
    with Image.open(source_png) as image:
        raw = image.info.get("prompt")
    if not raw:
        raise ValueError(f"{source_png} has no embedded ComfyUI prompt")
    prompt = json.loads(raw)
    if not isinstance(prompt, dict):
        raise TypeError("embedded ComfyUI prompt must be an object")

    conditioning = prompt["7"]["inputs"]
    sampler = prompt["8"]["inputs"]
    sla = prompt["9"]["inputs"]
    noise = prompt["10"]["inputs"]
    exact = {
        "width": int(conditioning["width"]) == WIDTH,
        "height": int(conditioning["height"]) == HEIGHT,
        "frame_count": int(conditioning["length"]) == FRAME_COUNT,
        "task_type": conditioning["task_type"] == "FL2VA",
        "steps": int(sampler["steps"]) == STEPS,
        "shift_video": float(sampler["shift_video"]) == SHIFT_VIDEO,
        "shift_audio": float(sampler["shift_audio"]) == SHIFT_AUDIO,
        "seed": int(noise["noise_seed"]) == SEED,
        "public_auto_safe_mode": sla["mode"] == "apply_lightx2v_sla",
    }
    if not all(exact.values()):
        raise ValueError(f"source prompt contract changed: {exact}")

    # These two nodes were intentionally disconnected visual examples in the original UI
    # workflow.  ComfyUI still validates every API-prompt entry, so remove only those dead
    # branches from the isolated diagnostic graph; the output ancestor graph is unchanged.
    prompt.pop("19", None)
    prompt.pop("21", None)
    prompt["15"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3_SLA_AutoSafe/{run_id}_736x416_124f_dense_fallback"
    )
    return prompt


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def _extract_audit(history: Mapping[str, Any] | None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw in _walk_strings(history or {}):
        if "auto_safe_" not in raw and "lightx2v_" not in raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "model_forward_count" in payload:
            candidates.append(payload)
    if not candidates:
        raise ValueError("history did not contain the SLA runtime audit JSON")
    return candidates[-1]


async def _submit_prompt_capture(
    *, server: str, prompt: Mapping[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    """Submit once while retaining the v3 ``executed`` payload for the Audit node."""

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
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await shared._json_request(session, "GET", f"{server}/system_stats")
        ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(f"{ws_url}/ws?clientId={client_id}", heartbeat=30) as ws:
            submitted = await shared._json_request(
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
                raise RuntimeError(f"ComfyUI returned unexpected prompt_id {prompt_id!r}")
            deadline = time.monotonic() + timeout_seconds
            while terminal is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for prompt {prompt_id}")
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    raise RuntimeError("ComfyUI WebSocket closed before a terminal event")
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                with suppress(json.JSONDecodeError):
                    packet = json.loads(message.data)
                    event_type = packet.get("type")
                    data = packet.get("data") if isinstance(packet.get("data"), dict) else {}
                    if data.get("prompt_id") not in {None, prompt_id}:
                        continue
                    node = str(data.get("node") or "")
                    if event_type in {
                        "executing",
                        "executed",
                        "progress",
                        "progress_state",
                        *shared.TERMINAL_EVENTS,
                    }:
                        events.append(
                            {
                                "elapsed_seconds": round(time.monotonic() - started, 4),
                                "type": event_type,
                                "node": node or None,
                            }
                        )
                    if event_type == "executed" and node:
                        executed_outputs[node] = data.get("output")
                    if event_type in shared.TERMINAL_EVENTS and data.get("prompt_id") == prompt_id:
                        terminal = {"type": event_type, "data": data}
        history = await shared._json_request(session, "GET", f"{server}/history/{prompt_id}")
    return {
        "prompt_id": prompt_id,
        "terminal": terminal,
        "history": history.get(prompt_id),
        "executed_outputs": executed_outputs,
        "events": events,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }


def _media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_736x416": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == WIDTH
        and int(video[0].get("height") or 0) == HEIGHT,
        "decoded_video_exact_124_frames": int(
            report.get("decoded_video", {}).get("bytes") or 0
        )
        == FRAME_COUNT * WIDTH * HEIGHT * 3,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == 32_000
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(report.get("decoded_audio", {}).get("bytes") or 0)
        > 0,
    }


def _audit_checks(audit: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "status_is_short_sequence_dense_fallback": audit.get("status")
        == EXPECTED_AUDIT_STATUS,
        "four_model_forwards": int(audit.get("model_forward_count") or 0) == STEPS,
        "execution_plan_all_dense": audit.get("attention_execution_plan")
        == ["dense"] * STEPS,
        "all_50_blocks_dense_per_forward": audit.get("dense_control_calls_per_forward")
        == [50] * STEPS,
        "no_sparse_kernel_calls": audit.get("sparse_kernel_calls_per_forward")
        == [0] * STEPS,
        "no_kernel_failures": int(audit.get("kernel_failure_count") or 0) == 0,
    }


def _contact_sheet(video: Path, output: Path, ffmpeg: str) -> None:
    # Six exact timeline samples: frame 0 and then one frame per second through frame 120.
    select = "+".join(f"eq(n\\,{frame})" for frame in (0, 24, 48, 72, 96, 120))
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(video),
        "-vf",
        f"select='{select}',scale={WIDTH}:{HEIGHT},tile=3x2",
        "-frames:v",
        "1",
        str(output),
    ]
    shared._run_checked(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\python\python.exe"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=14_500)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sla-auto-safe-tail-validation-20260826"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()

    paths = _required_paths(args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    gpu = shared.gpu_memory_mib()
    checks = {
        "required_paths_present": not missing,
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "normal_user_service_8188_stopped": not shared.port_is_listening(args.host, 8188),
        "isolated_port_free": not shared.port_is_listening(args.host, args.port),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_at_least_14500_mib": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= int(args.min_free_vram_mib)
        ),
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "checks": checks,
        "missing": missing,
        "gpu": gpu,
        "ready": all(checks.values()),
    }
    print(_json(preflight), flush=True)
    if not preflight["ready"]:
        return 2

    # Exact small inputs are hashed in-run.  The 1.96 GB LoRA was already fully hashed in the
    # preceding repair audit; retain that authenticated digest without adding another disk pass.
    assets = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": shared._sha256_file(path)
            if name in {"source_prompt_png", "first_frame", "last_frame"}
            else None,
        }
        for name, path in paths.items()
    }
    assets["sla_lora"]["previously_verified_sha256"] = EXPECTED_LORA_SHA256

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / run_id
    prompt = _load_exact_prompt(paths["source_prompt_png"], run_id)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "run_id": run_id,
        "contract": {
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": FRAME_COUNT,
            "steps": STEPS,
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
            "seed": SEED,
            "mode": "apply_lightx2v_sla",
            "single_render_only": True,
        },
        "preflight": preflight,
        "assets": assets,
    }

    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    phase: dict[str, Any] | None = None
    try:
        with shared.IsolatedServer(args, run_root, "sla_auto_safe_tail"):
            monitor.start()
            phase = asyncio.run(
                _submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    finally:
        report["gpu_monitor"] = monitor.stop()

    report["phase"] = phase
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(_json(report), flush=True)
        return 1

    audit_capture_error = None
    try:
        audit = _extract_audit(
            {
                "history": phase.get("history"),
                "executed_outputs": phase.get("executed_outputs"),
            }
        )
    except (TypeError, ValueError) as error:
        audit = None
        audit_capture_error = f"{type(error).__name__}: {error}"
    output_dir = run_root / "output" / "MiniMaxH3_SLA_AutoSafe"
    video = shared._latest_file(output_dir, f"{run_id}_736x416_124f_dense_fallback*-audio.mp4")
    media = shared.media_report(video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    contact = run_root / "contact_0s_to_5s.png"
    _contact_sheet(video, contact, str(ffmpeg))

    report.update(
        {
            "audit": audit,
            "audit_capture_error": audit_capture_error,
            "audit_checks": _audit_checks(audit) if audit is not None else {},
            "media": media,
            "media_checks": _media_checks(media),
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
        }
    )
    report["resource_checks"] = {
        "minimum_free_vram_at_least_512_mib": int(
            report["gpu_monitor"].get("minimum_free_mib") or 0
        )
        >= 512
    }
    mechanical_pass = bool(report["audit_checks"]) and all(
        report["audit_checks"].values()
    ) and all(
        report["media_checks"].values()
    ) and all(report["resource_checks"].values())
    report["status"] = (
        "MECHANICAL_PASS_VISUAL_REVIEW_PENDING" if mechanical_pass else "FAIL_MECHANICAL"
    )
    report["visual_quality_claim"] = (
        "Pending full-duration human review; first-second success alone is insufficient."
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(_json(report), flush=True)
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
