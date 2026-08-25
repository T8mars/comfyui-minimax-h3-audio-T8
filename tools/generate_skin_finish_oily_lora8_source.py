#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import av

import run_skin_finish_live_sam31_validation as base


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SEED_IMAGE_VIDEO = (
    ROOT
    / "artifacts"
    / "skin-finish-speaking-material-audit-20260825"
    / "source_speaking_960x544_124f.mp4"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-oily-lora8-source-20260825"
MODEL_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
WIDTH = 960
HEIGHT = 544
FRAME_COUNT = 124
FPS = 24.0
STEPS = 8
SEED = 2608258101
DIALOGUE = "你在干嘛呢，我在这里呀，看看效果如何。"


def _extract_seed_frame(source: Path, target: Path) -> None:
    with av.open(str(source)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        frame = next(container.decode(stream))
        image = frame.to_image().convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        raise RuntimeError(
            f"seed source must decode to {WIDTH}x{HEIGHT}, got {image.size}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")


def build_prompt(seed_image_name: str) -> dict[str, dict[str, Any]]:
    prompt = (
        "A cinematic photorealistic close-up of the same adult woman looking into the "
        "camera and speaking naturally. Soft frontal studio key light makes forehead and "
        "cheek highlights clearly visible; preserve realistic facial texture, eyes, lips, "
        "identity and subtle mouth motion. The camera remains steady and the face stays "
        "large in frame. <d>"
        f"{DIALOGUE}"
        "</d> Quiet indoor room tone, no music, no extra speech."
    )
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": MODEL_NAME, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderBypassModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_NAME,
                "strength_model": 1.0,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_NAME,
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE_NAME},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE_NAME},
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": seed_image_name},
        },
        "7": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "clip": ["3", 0],
                "video_vae": ["4", 0],
                "audio_vae": ["5", 0],
                "first_frame": ["6", 0],
                "prompt": prompt,
                "width": WIDTH,
                "height": HEIGHT,
                "length": FRAME_COUNT,
                "task_type": "I2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
        },
        "8": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["2", 0],
                "av_latent": ["7", 1],
                "steps": STEPS,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
        },
        "9": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": SEED},
        },
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
                "video_vae": ["4", 0],
                "audio_vae": ["5", 0],
            },
        },
        "13": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["12", 0],
                "audio": ["12", 1],
                "fps": FPS,
                "bit_depth": 8,
                "color_space": "sRGB",
            },
        },
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": "MiniMaxH3_SkinFinish/oily_lora8_speaking",
                "format": "mp4",
                "codec": {
                    "codec": "h264",
                    "encoding": {"encoding": "re-encode", "crf": 18.0},
                },
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-image-video", type=Path, default=DEFAULT_SEED_IMAGE_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=base.DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=1800.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    seed_video = args.seed_image_video.resolve()
    output_root = args.output.resolve()
    model_paths = {
        "model": COMFY_ROOT / "models" / "diffusion_models" / MODEL_NAME,
        "lora": COMFY_ROOT / "models" / "loras" / LORA_NAME,
        "clip": COMFY_ROOT / "models" / "text_encoders" / CLIP_NAME,
        "video_vae": COMFY_ROOT / "models" / "vae" / VIDEO_VAE_NAME,
        "audio_vae": COMFY_ROOT / "models" / "vae" / AUDIO_VAE_NAME,
    }
    gpu = base._gpu_sample()
    preflight = {
        "schema": "h3_t8_skin_finish_oily_lora8_source/preflight-v1",
        "created_at": base._utc_now(),
        "seed_video": str(seed_video),
        "seed_video_exists": seed_video.is_file(),
        "python": str(args.python.resolve()),
        "python_exists": args.python.is_file(),
        "ffmpeg_exists": shutil.which("ffmpeg") is not None,
        "models": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in model_paths.items()
        },
        "target_port_free": not base._port_is_listening(args.host, args.port),
        "user_port_8188_observed_only": base._port_is_listening(args.host, 8188),
        "gpu": gpu,
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "confirmed": bool(args.confirm_run),
        "generation_contract": {
            "task": "I2VA",
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "steps": STEPS,
            "shift_video": 12.0,
            "shift_audio": 3.0,
            "seed": SEED,
            "model": MODEL_NAME,
            "lora": LORA_NAME,
            "lora_strength": 1.0,
            "dialogue": DIALOGUE,
        },
    }
    preflight["ready"] = bool(
        preflight["seed_video_exists"]
        and preflight["python_exists"]
        and preflight["ffmpeg_exists"]
        and all(item["exists"] for item in preflight["models"].values())
        and preflight["target_port_free"]
        and gpu.get("available")
        and int(gpu.get("free_mib", 0)) >= args.minimum_free_vram_mib
        and args.confirm_run
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base._json_write(output_root / "preflight.json", preflight)
    if not preflight["ready"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = output_root / run_id
    for name in ("input", "output", "temp", "user", "logs", "prompt"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    seed_image = run_root / "input" / "skin_finish_oily_lora8_seed.png"
    _extract_seed_frame(seed_video, seed_image)
    prompt = build_prompt(seed_image.name)
    base._json_write(run_root / "prompt" / "prompt.json", prompt)

    monitor = base.GpuMonitor()
    server_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    history: dict[str, Any] | None = None
    prompt_id = None
    server_pid = None
    try:
        monitor.start()
        with base.IsolatedComfy(
            python=args.python.resolve(),
            host=args.host,
            port=args.port,
            run_root=run_root,
            start_timeout=args.server_start_timeout,
        ) as isolated:
            server_pid = int(isolated.process.pid) if isolated.process else None
            object_info = base._request_json("GET", f"{server_url}/object_info")
            missing = sorted(
                {
                    node["class_type"]
                    for node in prompt.values()
                    if node["class_type"] not in object_info
                }
            )
            if missing:
                raise RuntimeError(f"isolated ComfyUI is missing nodes: {missing}")
            response = base._request_json(
                "POST",
                f"{server_url}/prompt",
                {"prompt": prompt, "client_id": f"skin-finish-oily-{run_id}"},
            )
            prompt_id = str(response["prompt_id"])
            history = base._wait_for_history(
                server_url,
                prompt_id,
                args.prompt_timeout,
            )
            base._json_write(run_root / "history.json", history)
            errors = base._history_errors(history)
            if errors:
                raise RuntimeError(f"ComfyUI prompt failed: {errors[-1]}")
    finally:
        monitor.stop()

    outputs = sorted((run_root / "output").rglob("*.mp4"))
    if len(outputs) != 1:
        raise RuntimeError(f"expected one generated MP4, found {len(outputs)}")
    output = outputs[0]
    probe = base._probe(output)
    video = next(item for item in probe["streams"] if item["codec_type"] == "video")
    audio = next(item for item in probe["streams"] if item["codec_type"] == "audio")
    checks = {
        "exact_geometry": (int(video["width"]), int(video["height"]))
        == (WIDTH, HEIGHT),
        "exact_frame_count": int(video["nb_frames"]) == FRAME_COUNT,
        "exact_frame_rate": str(video["r_frame_rate"]) == "24/1",
        "audio_32k_stereo": int(audio["sample_rate"]) == 32000
        and int(audio["channels"]) == 2,
        "video_strict_decode": base._strict_decode(output, "video")["passed"],
        "audio_strict_decode": base._strict_decode(output, "audio")["passed"],
        "server_stopped": not base._port_is_listening(args.host, args.port),
        "user_8188_untouched": preflight["user_port_8188_observed_only"]
        == base._port_is_listening(args.host, 8188),
    }
    report = {
        "schema": "h3_t8_skin_finish_oily_lora8_source/v1",
        "created_at": base._utc_now(),
        "run_id": run_id,
        "server_pid": server_pid,
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "generation_contract": preflight["generation_contract"],
        "seed": {
            "video": str(seed_video),
            "video_sha256": base._sha256(seed_video),
            "image": str(seed_image),
            "image_sha256": base._sha256(seed_image),
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": base._sha256(output),
            "probe": probe,
        },
        "checks": checks,
        "gpu": monitor.report(),
        "passed": all(checks.values()),
        "claim_boundary": (
            "This is one user-requested full-FL2VA INT8 plus v1.0 LoRA, eight-step "
            "I2VA source-generation run for a Skin Finish visibility retest. It does "
            "not prove that the LoRA is universally oily, that Skin Finish improves "
            "the result, or that eight-step generation is universally memory safe."
        ),
    }
    base._json_write(run_root / "validation_report.json", report)
    base._json_write(
        output_root / "latest.json",
        {"run_id": run_id, "report": str(run_root / "validation_report.json")},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
