#!/usr/bin/env python3
"""Run a same-context/same-seed real H3 soft-context versus masked Plan B pair."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
import urllib.request

from safetensors import safe_open


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_sla_profile_router_probe as prior  # noqa: E402


SCHEMA = "t8.minimax_h3.native_masked_context.real_ab.v1"
WIDTH = 736
HEIGHT = 416
RENDER_FRAMES = 124
CONTEXT_FRAMES = 22
NEW_DURATION_SECONDS = 4.25
STEPS = 4
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
SAMPLER_NAME = "euler"
SCHEDULER_NAME = "native_flow"
SEGMENT_ZERO_SEED = 2609024100
CONTINUATION_SEED = 2609024101
BASE_MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
LEGACY_GENERIC_EMA_LORA = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
NEW_EMA_B_LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
TURBO_LORA = NEW_EMA_B_LORA
CORRECTED_FL2V_ALPHA8_LORA = (
    "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors"
)
ALLOWED_TURBO_LORAS = (
    NEW_EMA_B_LORA,
    CORRECTED_FL2V_ALPHA8_LORA,
    LEGACY_GENERIC_EMA_LORA,
)
CLIP_MODEL = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
RAIN_AMBIENCE_PROMPT = (
    "One continuous cinematic tracking shot on a rain-wet neon train platform at blue hour. "
    "A young East Asian woman with a short black bob wears a vivid red raincoat and walks "
    "steadily from left to right beside a yellow safety line while holding a transparent "
    "umbrella. The camera tracks laterally at chest height with constant direction and speed. "
    "Keep the same woman, face, red coat, umbrella, platform columns, lighting, movement "
    "direction, and camera velocity across the continuation. Natural footsteps and soft rain "
    "ambience, no dialogue, no cut, no transition, no teleport, no duplicate person, no dissolve."
)
INSTRUMENTAL_MUSIC_PROMPT = (
    "One continuous cinematic tracking shot on a rain-wet neon train platform at blue hour. "
    "A young East Asian woman with a short black bob wears a vivid red raincoat and walks "
    "steadily from left to right beside a yellow safety line while holding a transparent "
    "umbrella. The camera tracks laterally at chest height with constant direction and speed. "
    "Keep the same woman, face, red coat, umbrella, platform columns, lighting, movement "
    "direction, and camera velocity across the continuation. The soundtrack is clean continuous "
    "instrumental synthwave background music already in progress, with a steady 96 BPM pulse, "
    "warm sustained pads, a clear melodic line, and controlled dynamics. Music only: no rain "
    "sound, no footsteps, no environmental ambience, no dialogue, no vocals, no hiss, no static, "
    "no crackle, no clipping, and no distortion. No cut, no transition, no teleport, no duplicate "
    "person, no dissolve, and do not restart or abruptly change the music at the continuation."
)
CLASSICAL_MANDARIN_SPEECH_PROMPT = (
    "One continuous locked-off cinematic medium close-up of one adult East Asian woman facing "
    "the camera in a quiet, softly lit interior. She remains the only visible and audible person. "
    "She looks directly toward the camera and says exactly once, clearly and naturally in "
    "Mandarin Chinese: <d>[Chinese] 你在哪里</d>. Her mouth movement matches the one short sentence. "
    "No additional words, repeated speech, mumbling, whispering, singing, subtitles, or on-screen "
    "text. Clean classical chamber music plays continuously underneath from the beginning to the "
    "end: a warm solo cello and soft acoustic piano performing a slow adagio, with the voice clear "
    "in the foreground and the music at a lower level. No electronic instruments, rain, footsteps, "
    "room noise, crowd noise, hiss, static, crackle, clipping, or distortion. No cut, camera motion, "
    "duplicate person, reflection, or scene change."
)
CLASSICAL_CONTINUATION_MUSIC_PROMPT = (
    "Continue the exact same locked-off cinematic medium close-up of the same adult East Asian "
    "woman in the same softly lit interior. She remains silent throughout this continuation, "
    "breathes naturally, and makes no speech-like mouth movements. The clean classical chamber "
    "music from the previous segment is already in progress and continues without restarting: "
    "the same warm solo cello and soft acoustic piano maintain the same slow adagio, phrase, "
    "timbre, room, and controlled level. Music only in this continuation: no dialogue, vocals, "
    "words, whispering, humming, singing, electronic instruments, rain, footsteps, room noise, "
    "crowd noise, hiss, static, crackle, clipping, or distortion. No cut, camera motion, duplicate "
    "person, reflection, or scene change."
)
DEFAULT_AUDIO_PROFILE = "rain_ambience"
AUDIO_PROFILE_PROMPTS = {
    DEFAULT_AUDIO_PROFILE: RAIN_AMBIENCE_PROMPT,
    "instrumental_music": INSTRUMENTAL_MUSIC_PROMPT,
    "classical_mandarin_speech": CLASSICAL_MANDARIN_SPEECH_PROMPT,
}
# Preserve the original public constant for existing imports and recorded contracts.
PROMPT = RAIN_AMBIENCE_PROMPT
ROUTES = ("segment0", "soft_context", "hard_mask_plan_b")
RUN_SCOPES = {
    "full_pair": ROUTES,
    "segment0_only": ("segment0",),
}


def _prompt_for_profile(audio_profile: str, route: str) -> str:
    try:
        prompt = AUDIO_PROFILE_PROMPTS[audio_profile]
    except KeyError as error:
        raise ValueError(f"unknown audio profile {audio_profile!r}") from error
    if audio_profile == "classical_mandarin_speech" and route != "segment0":
        return CLASSICAL_CONTINUATION_MUSIC_PROMPT
    return prompt


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
        return json.load(response)


def _asset_paths(comfy_root: Path, *, turbo_lora: str = TURBO_LORA) -> dict[str, Path]:
    models = comfy_root / "models"
    return {
        "base_model": models / "diffusion_models" / BASE_MODEL,
        "turbo_lora": models / "loras" / turbo_lora,
        "clip": models / "text_encoders" / CLIP_MODEL,
        "video_vae": models / "vae" / VIDEO_VAE,
        "audio_vae": models / "vae" / AUDIO_VAE,
    }


def build_prompt(
    route: str,
    *,
    chain_id: str,
    run_id: str,
    audio_profile: str = DEFAULT_AUDIO_PROFILE,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    turbo_lora: str = TURBO_LORA,
) -> dict[str, dict[str, Any]]:
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    content_prompt = _prompt_for_profile(audio_profile, route)
    if turbo_lora not in ALLOWED_TURBO_LORAS:
        raise ValueError(f"unsupported diagnostic Turbo LoRA {turbo_lora!r}")
    segment_index = 0 if route == "segment0" else 1
    is_final = segment_index == 1
    seed = SEGMENT_ZERO_SEED if segment_index == 0 else CONTINUATION_SEED
    latent_source = ["8", 2]
    prompt: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_MODEL,
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": BASE_MODEL, "weight_dtype": "default"},
        },
        "5": {
            "class_type": "LoraLoaderBypassModelOnly",
            "inputs": {
                "lora_name": turbo_lora,
                "strength_model": 1.0,
                "model": ["4", 0],
            },
        },
        "6": {
            "class_type": "MiniMaxH3LongVideoPlannerT8",
            "inputs": {
                "chain_id": chain_id,
                "segment_index": segment_index,
                "new_duration_seconds": NEW_DURATION_SECONDS,
                "context_frames": CONTEXT_FRAMES,
                "minimum_render_frames": RENDER_FRAMES,
                "timeline_start_seconds": -1.0,
                "is_final_segment": is_final,
            },
        },
        "7": {
            "class_type": "MiniMaxH3LongVideoContextLoadT8",
            "inputs": {"chain_id": ["6", 0], "segment_index": ["6", 1]},
        },
        "8": {
            "class_type": "MiniMaxH3LongVideoConditioningT8",
            "inputs": {
                "model": ["5", 0],
                "clip": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
                "context": ["7", 0],
                "segment_index": ["6", 1],
                "context_frames": ["6", 3],
                "context_audio": "video_only",
                "prompt": content_prompt,
                "width": width,
                "height": height,
                "length": ["6", 2],
                "task_type": "auto",
                "audio_mode": "native",
                "audio_denoise_strength": 0.35,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "11": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["8", 1]},
        },
    }
    if route == "hard_mask_plan_b":
        prompt["18"] = {
            "class_type": "MiniMaxH3NativeMaskedVideoContextT8Advanced",
            "inputs": {
                "av_latent": ["8", 2],
                "context": ["7", 0],
                "planner_report_json": ["6", 9],
                "conditioning_report_json": ["8", 6],
            },
        }
        latent_source = ["18", 0]
    prompt.update(
        {
            "9": {
                "class_type": "MiniMaxH3DualClockSamplerT8",
                "inputs": {
                    "model": ["8", 0],
                    "av_latent": latent_source,
                    "steps": steps,
                    "shift_video": SHIFT_VIDEO,
                    "shift_audio": SHIFT_AUDIO,
                    "sampler_name": SAMPLER_NAME,
                    "scheduler": SCHEDULER_NAME,
                },
            },
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["11", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["9", 2],
                    "latent_image": latent_source,
                },
            },
            "14": {
                "class_type": "MiniMaxH3AVDecodeT8",
                "inputs": {
                    "av_latent": ["12", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
            },
            "15": {
                "class_type": "MiniMaxH3OutputTrimT8",
                "inputs": {
                    "frames": ["14", 0],
                    "start_seconds": ["6", 4],
                    "duration_seconds": ["6", 5],
                    "fps": 24.0,
                    "audio": ["14", 1],
                },
            },
            "16": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["19", 0],
                    "fps": 24.0,
                    "audio": ["15", 1],
                    "bit_depth": 8,
                },
            },
            "19": {
                "class_type": "MiniMaxH3LongVideoColorMatchT8Advanced",
                "inputs": {
                    "frames": ["15", 0],
                    "context": ["7", 0],
                    "chain_id": ["6", 0],
                    "segment_index": ["6", 1],
                    "enabled": True,
                    "reference_frames": 5,
                    "transition_frames": 24,
                    "strength": 1.0,
                    "minimum_jump": 0.0005,
                    "maximum_offset": 0.02,
                    "scene_cut_threshold": 0.18,
                },
            },
            "17": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["16", 0],
                    "filename_prefix": (
                        f"MiniMaxH3/NativeMaskedPlanB_AB/{run_id}_{route}_segment{segment_index}"
                    ),
                    "format": "mp4",
                    "codec": "h264",
                },
            },
        }
    )
    if segment_index == 0:
        prompt["13"] = {
            "class_type": "MiniMaxH3LongVideoContextSaveT8",
            "inputs": {
                "av_latent": ["12", 0],
                "chain_id": ["6", 0],
                "segment_index": ["6", 1],
                "save_context": ["6", 8],
                "model_id": f"{BASE_MODEL}+{turbo_lora}",
                "sampling_summary": (
                    f"{steps}-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
                ),
            },
        }
    return prompt


def _context_path(comfy_root: Path, chain_id: str) -> Path:
    return (
        comfy_root
        / "output"
        / "minimax_h3_t8_long_video"
        / chain_id
        / "segment_00000.context.safetensors"
    )


def _color_state_path(comfy_root: Path, chain_id: str, segment_index: int) -> Path:
    return (
        comfy_root
        / "output"
        / "minimax_h3_t8_long_video"
        / chain_id
        / f"segment_{int(segment_index):05d}.color.safetensors"
    )


def _read_color_report(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    report_text = metadata.get("report_json")
    if not report_text:
        raise ValueError(f"Color Match state has no report_json metadata: {path}")
    report = json.loads(report_text)
    if not isinstance(report, dict):
        raise ValueError(f"Color Match report is not an object: {path}")
    return report


def _find_output(comfy_root: Path, run_id: str, route: str) -> Path:
    folder = comfy_root / "output" / "MiniMaxH3" / "NativeMaskedPlanB_AB"
    matches = sorted(
        folder.glob(f"{run_id}_{route}_segment*.mp4"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not matches:
        raise FileNotFoundError(f"SaveVideo output not found for {route} under {folder}")
    return matches[-1]


def _phase_success(phase: Mapping[str, Any] | None) -> bool:
    return bool(phase) and phase.get("terminal", {}).get("type") == "execution_success"


def _result_success(
    phase: Mapping[str, Any] | None,
    media: Mapping[str, Any] | None,
) -> bool:
    return _phase_success(phase) and bool(media and media.get("strict_decode_passed"))


def _run_phase(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    chain_id: str,
    route: str,
    audio_profile: str,
) -> dict[str, Any]:
    prompt = build_prompt(
        route,
        chain_id=chain_id,
        run_id=run_id,
        audio_profile=audio_profile,
        width=args.width,
        height=args.height,
        steps=args.steps,
        turbo_lora=args.turbo_lora,
    )
    phase_root = run_root / route
    phase_root.mkdir(parents=True, exist_ok=False)
    (phase_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    monitor = prior.clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    phase = None
    monitor.start()
    try:
        phase = asyncio.run(
            prior._submit_prompt_capture(
                server=args.server,
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        )
    finally:
        gpu_monitor = monitor.stop()
    (phase_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result: dict[str, Any] = {
        "route": route,
        "execution_success": _phase_success(phase),
        "success": False,
        "elapsed_seconds": phase.get("elapsed_seconds") if phase else None,
        "gpu_monitor": gpu_monitor,
        "prompt": str((phase_root / "prompt.json").resolve()),
        "phase": str((phase_root / "phase.json").resolve()),
    }
    if not result["execution_success"]:
        result["terminal"] = phase.get("terminal") if phase else None
        result["failure_reason"] = "COMFY_EXECUTION_DID_NOT_SUCCEED"
        return result
    source = _find_output(args.comfy_root, run_id, route)
    target = phase_root / f"{route}.mp4"
    shutil.copy2(source, target)
    media = prior.shared.media_report(
        target,
        ffmpeg=str(args.ffmpeg),
        ffprobe=str(args.ffprobe),
    )
    result.update(
        {
            "source_output": str(source.resolve()),
            "artifact_video": str(target.resolve()),
            "video_bytes": target.stat().st_size,
            "video_sha256": prior.shared._sha256_file(target),
            "media": media,
            "success": _result_success(phase, media),
        }
    )
    segment_index = 0 if route == "segment0" else 1
    color_state = _color_state_path(args.comfy_root, chain_id, segment_index)
    if color_state.is_file():
        copied_color_state = phase_root / color_state.name
        shutil.copy2(color_state, copied_color_state)
        color_report = _read_color_report(copied_color_state)
        result["color_match"] = {
            "state_path": str(copied_color_state.resolve()),
            "state_sha256": prior.shared._sha256_file(copied_color_state),
            "report": color_report,
        }
    else:
        result["color_match"] = None
        result["success"] = False
        result["failure_reason"] = "COLOR_MATCH_STATE_MISSING"
    if not result["success"]:
        result.setdefault("failure_reason", "STRICT_MEDIA_DECODE_FAILED")
    return result


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    assets = _asset_paths(args.comfy_root, turbo_lora=args.turbo_lora)
    try:
        stats = _json_get(f"{args.server}/system_stats")
        queue = _json_get(f"{args.server}/queue")
        node = _json_get(
            f"{args.server}/object_info/MiniMaxH3NativeMaskedVideoContextT8Advanced"
        )
        color_node = _json_get(
            f"{args.server}/object_info/MiniMaxH3LongVideoColorMatchT8Advanced"
        )
        server_error = None
    except Exception as error:  # noqa: BLE001
        stats = None
        queue = None
        node = None
        color_node = None
        server_error = f"{type(error).__name__}: {error}"
    gpu = prior.shared.gpu_memory_mib()
    checks = {
        "server_reachable": server_error is None,
        "plan_b_node_loaded": bool(
            isinstance(node, Mapping)
            and "MiniMaxH3NativeMaskedVideoContextT8Advanced" in node
        ),
        "color_match_node_loaded": bool(
            isinstance(color_node, Mapping)
            and "MiniMaxH3LongVideoColorMatchT8Advanced" in color_node
        ),
        "server_queue_idle": bool(
            isinstance(queue, Mapping)
            and not queue.get("queue_running")
            and not queue.get("queue_pending")
        ),
        "all_assets_present": all(path.is_file() for path in assets.values()),
        "ffmpeg_present": bool(args.ffmpeg),
        "ffprobe_present": bool(args.ffprobe),
        "gpu_query_available": bool(gpu.get("available")),
        "initial_free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib
        ),
    }
    return {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "server": args.server,
        "checks": checks,
        "ready": all(checks.values()),
        "server_error": server_error,
        "system_stats": stats,
        "queue": queue,
        "gpu": gpu,
        "assets": {
            name: {"path": str(path), "bytes": path.stat().st_size if path.is_file() else None}
            for name, path in assets.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument("--server", default="http://127.0.0.1:8189")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=13_500)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--steps", type=int, choices=(4, 8), default=STEPS)
    parser.add_argument(
        "--turbo-lora",
        choices=ALLOWED_TURBO_LORAS,
        default=TURBO_LORA,
        help=(
            "Use the user-selected step600 EMA_B by default; the corrected official "
            "FL2V Alpha8 and historical generic EMA remain explicit diagnostic controls."
        ),
    )
    parser.add_argument(
        "--run-scope",
        choices=tuple(RUN_SCOPES),
        default="full_pair",
        help="Run the full continuation pair or only a segment-zero audio diagnostic.",
    )
    parser.add_argument(
        "--audio-profile",
        choices=tuple(AUDIO_PROFILE_PROMPTS),
        default=DEFAULT_AUDIO_PROFILE,
        help=(
            "Native H3 soundtrack prompt profile. instrumental_music requests clean continuous "
            "music without speech; classical_mandarin_speech requests chamber music plus exactly "
            "one Mandarin utterance: 你在哪里."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/native-masked-context-real-ab-20260902"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()
    args.ffmpeg = shutil.which("ffmpeg")
    args.ffprobe = shutil.which("ffprobe")
    preflight = _preflight(args)
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not preflight["ready"]:
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    routes = RUN_SCOPES[args.run_scope]
    chain_prefix = (
        "native_masked_planb_ab"
        if args.run_scope == "full_pair"
        else "native_audio_segment0_diagnostic"
    )
    chain_id = f"{chain_prefix}_{run_id.replace('-', '')}"
    run_root = args.artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    context_path = _context_path(args.comfy_root, chain_id)
    segment_zero_color_path = _color_state_path(args.comfy_root, chain_id, 0)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "RUNNING",
        "run_id": run_id,
        "chain_id": chain_id,
        "run_root": str(run_root.resolve()),
        "contract": {
            "routes": list(routes),
            "run_scope": args.run_scope,
            "only_ab_difference": (
                "hard_mask_plan_b inserts MiniMaxH3NativeMaskedVideoContextT8Advanced; "
                "soft_context samples the same Long Video Conditioning latent directly"
                if args.run_scope == "full_pair"
                else None
            ),
            "same_segment0_context": True,
            "same_continuation_prompt_seed_model_lora_nfe_shifts": (
                args.run_scope == "full_pair"
            ),
            "context_audio": "video_only",
            "width": args.width,
            "height": args.height,
            "render_frames": RENDER_FRAMES,
            "context_frames": CONTEXT_FRAMES,
            "new_visible_frames": RENDER_FRAMES - CONTEXT_FRAMES,
            "steps": args.steps,
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
            "sampler_name": SAMPLER_NAME,
            "scheduler": SCHEDULER_NAME,
            "segment_zero_seed": SEGMENT_ZERO_SEED,
            "continuation_seed": CONTINUATION_SEED,
            "audio_profile": args.audio_profile,
            "prompt": AUDIO_PROFILE_PROMPTS[args.audio_profile],
            "segment_zero_prompt": _prompt_for_profile(args.audio_profile, "segment0"),
            "continuation_prompt": _prompt_for_profile(args.audio_profile, "soft_context"),
            "global_dialogue_count_requested": (
                1 if args.audio_profile == "classical_mandarin_speech" else 0
            ),
            "base_model": BASE_MODEL,
            "turbo_lora": args.turbo_lora,
            "clip": CLIP_MODEL,
            "video_vae": VIDEO_VAE,
            "audio_vae": AUDIO_VAE,
            "server_reused_without_restart": True,
            "route_order": list(routes),
            "color_match": {
                "enabled": True,
                "default_enabled": True,
                "placement": "post_decode_post_output_trim_before_CreateVideo",
                "method": "bounded_uniform_reinhard_lab_spatial_rgb_with_fade",
                "reference_frames": 5,
                "transition_frames": 24,
                "maximum_offset": 0.02,
                "lab_scale_bounds": [0.85, 1.18],
                "spatial_grid": [5, 8],
                "audio_touched": False,
                "latent_touched": False,
            },
        },
        "preflight": preflight,
        "phases": {},
        "context": {"path": str(context_path.resolve())},
        "color_reference": {"path": str(segment_zero_color_path.resolve())},
    }
    (run_root / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for route in routes:
        result = _run_phase(
            args=args,
            run_root=run_root,
            run_id=run_id,
            chain_id=chain_id,
            route=route,
            audio_profile=args.audio_profile,
        )
        report["phases"][route] = result
        if route == "segment0" and result["success"] and context_path.is_file():
            report["context"].update(
                {
                    "bytes": context_path.stat().st_size,
                    "sha256_after_segment0": prior.shared._sha256_file(context_path),
                }
            )
            if segment_zero_color_path.is_file():
                report["color_reference"].update(
                    {
                        "bytes": segment_zero_color_path.stat().st_size,
                        "sha256_after_segment0": prior.shared._sha256_file(
                            segment_zero_color_path
                        ),
                    }
                )
        (run_root / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not result["success"]:
            report["status"] = f"FAIL_{route.upper()}"
            break

    if all(report["phases"].get(route, {}).get("success") for route in routes):
        final_context_hash = (
            prior.shared._sha256_file(context_path) if context_path.is_file() else None
        )
        report["context"]["sha256_after_pair"] = final_context_hash
        report["context"]["unchanged_during_pair"] = final_context_hash == report[
            "context"
        ].get("sha256_after_segment0")
        final_color_hash = (
            prior.shared._sha256_file(segment_zero_color_path)
            if segment_zero_color_path.is_file()
            else None
        )
        report["color_reference"]["sha256_after_pair"] = final_color_hash
        report["color_reference"]["unchanged_during_pair"] = final_color_hash == report[
            "color_reference"
        ].get("sha256_after_segment0")
        if not report["context"]["unchanged_during_pair"]:
            report["status"] = "FAIL_SHARED_CONTEXT_CHANGED"
        elif not report["color_reference"]["unchanged_during_pair"]:
            report["status"] = "FAIL_SHARED_COLOR_REFERENCE_CHANGED"
        elif args.run_scope == "segment0_only":
            report["status"] = "REAL_SEGMENT_ZERO_COMPLETE_ANALYSIS_PENDING"
        else:
            report["status"] = "REAL_PAIR_COMPLETE_ANALYSIS_PENDING"
    report["finished_at"] = _utc_now()
    (run_root / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] in {
        "REAL_PAIR_COMPLETE_ANALYSIS_PENDING",
        "REAL_SEGMENT_ZERO_COMPLETE_ANALYSIS_PENDING",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
