#!/usr/bin/env python3
"""Run one guarded real MiniMax-H3 PDD FL2VA or Ref2VA validation.

This is deliberately a one-render tool. It starts an isolated ComfyUI process,
uses the matching full non-pruned base and PDD adapter, records the setup report,
strictly decodes the resulting audio/video, then stops the server. It never runs
both variants concurrently and it refuses to start below the configured VRAM gate.
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
from typing import Any, Mapping
import uuid

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402


SCHEMA = "t8.minimax_h3.pdd_real_validation.v1"
DEFAULT_WIDTH = 736
DEFAULT_HEIGHT = 416
DEFAULT_FRAME_COUNT = 124
FPS = 24
SEED = 2608270801
EXPECTED_LORA = {
    "FL2VA": (
        "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors",
        "95b79e73dbad645f4f4ccd7fb8c5d864e7b978022a4c372f8cfaba82d3ff40bf",
    ),
    "Ref2VA": (
        "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors",
        "f4522e368ad7da1af19a283a728fbeb1f2b18866569ef9169b73786c3d69e4d2",
    ),
}


async def _submit_prompt_capture(
    *, server: str, prompt: Mapping[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    """Submit one prompt and retain ComfyUI v3 executed-node outputs."""

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
        await shared._json_request(session, "GET", f"{server}/system_stats")
        ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(
            f"{ws_url}/ws?clientId={client_id}", heartbeat=30
        ) as websocket:
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
                        *shared.TERMINAL_EVENTS,
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
                        event_type in shared.TERMINAL_EVENTS
                        and data.get("prompt_id") == prompt_id
                    ):
                        terminal = {"type": event_type, "data": data}
        history = await shared._json_request(
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
BASE_NAME = {
    "FL2VA": "minimax_h3_fl2va_int8_convrot.safetensors",
    "Ref2VA": "minimax_h3_ref2va_int8_convrot.safetensors",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    lora_name, _ = EXPECTED_LORA[args.variant]
    return {
        "comfy_main": args.comfy_root / "main.py",
        "python": args.python,
        "project": args.project_root,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "base": models / "diffusion_models" / BASE_NAME[args.variant],
        "clip": models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "pdd_lora": models / "loras" / lora_name,
        "reference": args.comfy_root / "input" / "codex_prompt_relay_fl2va_first.png",
    }


def _prompt(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    lora_name, _ = EXPECTED_LORA[args.variant]
    prompt: dict[str, Any] = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": BASE_NAME[args.variant], "weight_dtype": "default"},
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": "codex_prompt_relay_fl2va_first.png"},
        },
        "8": {
            "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
            "inputs": {
                "model": ["4", 0],
                "av_latent": ["7", 1],
                "pdd_lora_name": lora_name,
                "base_variant": args.variant,
                "strength": 1.0,
            },
        },
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["10", 0],
                "sampler": ["8", 1],
                "sigmas": ["8", 2],
                "latent_image": ["7", 1],
            },
        },
        "12": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["11", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "audio": ["12", 1],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": (
                    f"MiniMaxH3_PDD_Validation/{run_id}_{args.variant.lower()}_"
                    f"{args.width}x{args.height}_{args.frame_count}f"
                ),
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
        "14": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["8", 3]},
            "_meta": {"title": "Capture PDD setup report"},
        },
    }
    common = {
        "clip": ["3", 0],
        "video_vae": ["1", 0],
        "audio_vae": ["2", 0],
        "width": args.width,
        "height": args.height,
        "length": args.frame_count,
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
    }
    if args.variant == "FL2VA":
        prompt["6"] = {
            "class_type": "LoadImage",
            "inputs": {"image": "codex_prompt_relay_fl2va_first.png"},
        }
        conditioning = {
            **common,
            "prompt": (
                "One continuous locked-medium cinematic night shot on a rain-wet neon "
                "street. Keep the same adult woman in a long red coat at the same portrait "
                "scale while she turns naturally, blinks and takes one small step. She clearly "
                "says in Mandarin: \"你在干嘛呢，我在这里呀，看看效果如何。\" Stable identity "
                "and anatomy, synchronized speech, fabric motion and city ambience, no cuts, "
                "camera pull-away or sudden scale change."
            ),
            "task_type": "FL2VA",
            "first_frame": ["5", 0],
            "last_frame": ["6", 0],
        }
    else:
        conditioning = {
            **common,
            "prompt": (
                "Use <Picture 1> as the visual identity and appearance reference. In one "
                "continuous natural cinematic portrait, the same adult woman turns gently "
                "toward the camera and blinks once. Preserve facial structure, hairstyle, skin "
                "tone and realistic proportions. She clearly says in Mandarin: \"你在干嘛呢，"
                "我在这里呀，看看效果如何。\" Synchronized speech and quiet room ambience, "
                "no music, subtitles, cuts or additional people."
            ),
            "task_type": "Ref2VA",
            "ref_images.ref_image_0": ["5", 0],
        }
    prompt["7"] = {
        "class_type": "MiniMaxH3AudioConditioningT8",
        "inputs": conditioning,
    }
    return prompt


def _phase_text(phase: Mapping[str, Any] | None, node_id: str) -> str:
    """Return a PreviewAny string from either v3 live output or legacy history.

    Current ComfyUI can omit UI-only PreviewAny values from ``/history`` while
    still sending them in the v3 WebSocket ``executed`` event.  Retain both
    routes so a successful render is never discarded merely because history
    persistence changed.
    """

    phase = phase or {}
    executed_outputs = phase.get("executed_outputs") or {}
    history = phase.get("history") or {}
    history_outputs = history.get("outputs") or {}
    candidates = [
        executed_outputs.get(node_id, {}),
        history_outputs.get(node_id, {}),
    ]
    for output in candidates:
        values = output.get("text") if isinstance(output, Mapping) else None
        if isinstance(values, list) and values:
            return str(values[-1])
    raise RuntimeError(
        f"execution did not retain PreviewAny text for node {node_id} in either "
        "the v3 executed event or history"
    )


def _contact_sheet(
    video: Path,
    output: Path,
    ffmpeg: str,
    *,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    frames = [round(index * (frame_count - 1) / 5) for index in range(6)]
    select = "+".join(f"eq(n\\,{frame})" for frame in frames)
    shared._run_checked(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-threads",
            "1",
            "-i",
            str(video),
            "-vf",
            f"select='{select}',scale={width}:{height},tile=3x2",
            "-frames:v",
            "1",
            str(output),
        ]
    )


def _audio_numeric(video: Path, ffmpeg: str) -> dict[str, Any]:
    raw = shared._run_checked(
        [
            ffmpeg,
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-",
        ],
        binary=True,
    ).stdout
    samples = np.frombuffer(raw, dtype="<f4")
    return {
        "sample_values": int(samples.size),
        "all_finite": bool(samples.size and np.isfinite(samples).all()),
        "peak_abs": float(np.max(np.abs(samples))) if samples.size else None,
        "clipped_fraction_ge_0p999": (
            float(np.mean(np.abs(samples) >= 0.999)) if samples.size else None
        ),
    }


def _media_checks(
    media: Mapping[str, Any],
    audio: Mapping[str, Any],
    *,
    width: int,
    height: int,
    frame_count: int,
) -> dict[str, bool]:
    streams = media.get("probe", {}).get("streams", [])
    videos = [value for value in streams if value.get("codec_type") == "video"]
    audios = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(media.get("strict_decode_passed")),
        "h264_exact_dimensions": len(videos) == 1
        and videos[0].get("codec_name") == "h264"
        and int(videos[0].get("width") or 0) == width
        and int(videos[0].get("height") or 0) == height,
        "exact_decoded_frames": int(
            media.get("decoded_video", {}).get("bytes") or 0
        )
        == frame_count * width * height * 3,
        "aac_32khz_stereo": len(audios) == 1
        and audios[0].get("codec_name") == "aac"
        and int(audios[0].get("sample_rate") or 0) == 32000
        and int(audios[0].get("channels") or 0) == 2,
        "audio_nonempty_finite": bool(audio.get("sample_values"))
        and bool(audio.get("all_finite")),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("FL2VA", "Ref2VA"), required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=14500)
    parser.add_argument(
        "--recover-run-root",
        type=Path,
        help=(
            "Post-process an already completed run without starting ComfyUI again. "
            "The directory must contain phase.json and the isolated output tree."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/pdd-real-validation-20260827"),
    )
    return parser


def _validate_contract(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.frame_count <= 0:
        raise ValueError("width, height and frame-count must be positive")
    if args.width % 32 or args.height % 32:
        raise ValueError("MiniMax H3 validation width and height must be divisible by 32")


def _finalize_completed_run(
    *,
    args: argparse.Namespace,
    run_root: Path,
    phase: Mapping[str, Any],
    report: dict[str, Any],
    ffmpeg: str,
    ffprobe: str,
) -> int:
    """Strictly inspect one completed render without running the model again."""

    setup = json.loads(_phase_text(phase, "14"))
    run_id = str(report["run_id"])
    output_dir = run_root / "output" / "MiniMaxH3_PDD_Validation"
    video = shared._latest_file(
        output_dir,
        f"{run_id}_{args.variant.lower()}_{args.width}x{args.height}_{args.frame_count}f*audio.mp4",
    )
    media = shared.media_report(video, ffmpeg=ffmpeg, ffprobe=ffprobe)
    audio = _audio_numeric(video, ffmpeg)
    contact = run_root / "contact_0s_to_5s.png"
    _contact_sheet(
        video,
        contact,
        ffmpeg,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
    )
    setup_checks = {
        "variant": setup.get("adapter", {}).get("base_variant") == args.variant,
        "mapped_258": int(setup.get("lora", {}).get("mapped_adapters") or 0) == 258,
        "hooks_258": int(setup.get("lora", {}).get("bypass_hooks") or 0) == 258,
        "block_indices_0_to_7": setup.get("sampling", {}).get("block_indices")
        == list(range(8)),
        "nfe_8": int(setup.get("sampling", {}).get("nfe") or 0) == 8,
        "offload_lifecycle_enabled": setup.get("lora", {}).get("eject_policy")
        == "move_adapter_weights_to_model_offload_device",
    }
    media_checks = _media_checks(
        media,
        audio,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
    )
    minimum_free = report.get("gpu_monitor", {}).get("minimum_free_mib")
    resource_checks = {
        "telemetry_available": minimum_free is not None,
        "minimum_free_vram_at_least_512_mib": (
            minimum_free is not None and int(minimum_free) >= 512
        ),
    }
    setup_and_media_pass = all(setup_checks.values()) and all(media_checks.values())
    mechanical_pass = setup_and_media_pass and all(resource_checks.values())
    if mechanical_pass:
        status = "MECHANICAL_PASS_HUMAN_REVIEW_PENDING"
    elif setup_and_media_pass and not resource_checks["telemetry_available"]:
        status = (
            "MEDIA_SETUP_PASS_RESOURCE_TELEMETRY_UNAVAILABLE_"
            "HUMAN_REVIEW_PENDING"
        )
    elif setup_and_media_pass:
        status = "MEDIA_SETUP_PASS_RESOURCE_GATE_FAIL_HUMAN_REVIEW_PENDING"
    else:
        status = "FAIL_MECHANICAL"
    report.update(
        {
            "setup_report": setup,
            "setup_checks": setup_checks,
            "media": media,
            "audio_numeric": audio,
            "media_checks": media_checks,
            "resource_checks": resource_checks,
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
            "status": status,
            "quality_claim": (
                "No quality claim until the complete clip is reviewed by a human."
            ),
        }
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if mechanical_pass else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_contract(args)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()
    if args.recover_run_root is not None:
        args.recover_run_root = args.recover_run_root.resolve()
    paths = _paths(args)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if args.recover_run_root is not None:
        if not ffmpeg or not ffprobe:
            raise RuntimeError("ffmpeg and ffprobe are required for PDD recovery")
        _, expected_hash = EXPECTED_LORA[args.variant]
        actual_hash = shared._sha256_file(paths["pdd_lora"]).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"installed {args.variant} PDD validation file hash changed: {actual_hash}"
            )
        run_root = args.recover_run_root
        phase_path = run_root / "phase.json"
        if not phase_path.is_file():
            raise FileNotFoundError(f"recovery run has no phase.json: {run_root}")
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        if not phase or phase.get("terminal", {}).get("type") != "execution_success":
            raise RuntimeError("recovery is only allowed for an execution_success run")
        report_path = run_root / "validation_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            run_id = run_root.name.rsplit("-", 1)[0]
            report = {
                "schema": SCHEMA,
                "created_at": _utc_now(),
                "variant": args.variant,
                "run_id": run_id,
                "contract": {
                    "width": args.width,
                    "height": args.height,
                    "frame_count": args.frame_count,
                    "nfe": 8,
                    "video_shift": 12.0,
                    "audio_shift": 3.0,
                    "sampler": "euler",
                    "scheduler": "simple",
                    "cfg": 1.0,
                    "strength": 1.0,
                    "single_render_only": True,
                },
                "preflight": {
                    "status": "RECOVERED_FROM_COMPLETED_RENDER",
                    "resource_telemetry_limit": (
                        "The original wrapper completed before persisting its GPU monitor."
                    ),
                },
                "gpu_monitor": {
                    "status": "unavailable_after_completed_render",
                    "minimum_free_mib": None,
                },
                "pdd_lora": {
                    "path": str(paths["pdd_lora"]),
                    "bytes": paths["pdd_lora"].stat().st_size,
                    "sha256": actual_hash,
                },
            }
        if report.get("variant") != args.variant:
            raise RuntimeError(
                f"recovery variant mismatch: report={report.get('variant')!r}, "
                f"requested={args.variant!r}"
            )
        report["phase"] = phase
        report["postprocess_recovered_without_model_rerun"] = True
        return _finalize_completed_run(
            args=args,
            run_root=run_root,
            phase=phase,
            report=report,
            ffmpeg=str(ffmpeg),
            ffprobe=str(ffprobe),
        )

    gpu = shared.gpu_memory_mib()
    checks = {
        "required_paths_present": all(path.exists() for path in paths.values()),
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "user_port_8188_free": not shared.port_is_listening(args.host, 8188),
        "isolated_port_free": not shared.port_is_listening(args.host, args.port),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib
        ),
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "variant": args.variant,
        "checks": checks,
        "gpu": gpu,
        "ready": all(checks.values()),
    }
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not preflight["ready"]:
        return 2

    _, expected_hash = EXPECTED_LORA[args.variant]
    actual_hash = shared._sha256_file(paths["pdd_lora"]).lower()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"installed {args.variant} PDD validation file hash changed: {actual_hash}"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / f"{run_id}-{args.variant.lower()}"
    run_root.mkdir(parents=True, exist_ok=False)
    prompt = _prompt(args, run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "variant": args.variant,
        "run_id": run_id,
        "contract": {
            "width": args.width,
            "height": args.height,
            "frame_count": args.frame_count,
            "nfe": 8,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "sampler": "euler",
            "scheduler": "simple",
            "cfg": 1.0,
            "strength": 1.0,
            "single_render_only": True,
        },
        "preflight": preflight,
        "pdd_lora": {
            "path": str(paths["pdd_lora"]),
            "bytes": paths["pdd_lora"].stat().st_size,
            "sha256": actual_hash,
        },
    }
    phase = None
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    try:
        with shared.IsolatedServer(args, run_root, f"pdd_{args.variant.lower()}"):
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
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Persist resource telemetry before any optional UI-output parsing.  This
    # makes a completed render recoverable if a future ComfyUI build changes
    # how PreviewAny is represented in history.
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1

    return _finalize_completed_run(
        args=args,
        run_root=run_root,
        phase=phase,
        report=report,
        ffmpeg=str(ffmpeg),
        ffprobe=str(ffprobe),
    )


if __name__ == "__main__":
    raise SystemExit(main())
