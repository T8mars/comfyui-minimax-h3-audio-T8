from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "i2va_lock_source",
        "task_type": "I2VA",
        "audio_mode": "lock_source",
        "audio_denoise_strength": 0.0,
        "first_frame": True,
        "last_frame": False,
        "output_audio": "locked_source",
        "prompt": (
            "Continue naturally from the first frame with coherent subject identity and "
            "camera motion. Preserve the supplied source soundtrack exactly; add no speech."
        ),
    },
    {
        "name": "fl2va_remix_source",
        "task_type": "FL2VA",
        "audio_mode": "remix_source",
        "audio_denoise_strength": 0.35,
        "first_frame": True,
        "last_frame": True,
        "output_audio": "generated_remix",
        "prompt": (
            "Move smoothly and naturally from the first frame to the last frame. Reinterpret "
            "the supplied source soundtrack while preserving its timing and rhythm; add no speech."
        ),
    },
    {
        "name": "l2va_native",
        "task_type": "L2VA",
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "first_frame": False,
        "last_frame": True,
        "output_audio": "generated_native",
        "prompt": (
            "Natural cinematic motion resolves exactly into the last frame. Generate synchronized "
            "environment ambience and motion sound; no speech."
        ),
    },
)


def _loaders(
    model_name: str,
    clip_name: str,
    video_vae_name: str,
    audio_vae_name: str,
) -> dict[str, dict[str, Any]]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Stock MiniMax H3 FL2VA model"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
            "_meta": {"title": "MiniMax H3 Qwen encoder"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
            "_meta": {"title": "MiniMax H3 video VAE"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
            "_meta": {"title": "MiniMax H3 audio VAE"},
        },
    }


def _validate_dimensions(width: int, height: int, length: int, steps: int) -> None:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("length must follow the H3 17n+5 grid")
    if steps != 20:
        raise ValueError("P2 multimodal validation is frozen to Stock20")


def build_multimodal_speed_prompts(
    *,
    source_video: str,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    steps: int = 20,
    seed: int = 2608192001,
    scales: str = "0.5,1.0",
    transition_sigma: str = "0.85",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    model_name: str = "minimax_h3_fl2va_int8_convrot.safetensors",
    clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors",
    filename_prefix: str = "MiniMaxH3/SPEED_P2_multimodal_v1",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _validate_dimensions(width, height, length, steps)
    if not source_video.strip():
        raise ValueError("source_video must not be empty")
    duration = length / 24.0
    prompts: dict[str, dict[str, Any]] = {}
    for case_index, case in enumerate(CASES):
        source_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "video_vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": case["prompt"],
            "length": length,
            "task_type": case["task_type"],
            "audio_mode": case["audio_mode"],
            "audio_denoise_strength": case["audio_denoise_strength"],
            "add_source_as_reference": False,
            "prompt_primary_audio_ordinal": 0,
            "strict_prompt_tags": True,
            "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s",
            "checkpoint_fingerprint": "unrecorded",
            "vae_fingerprint": "unrecorded",
        }
        if case["audio_mode"] != "native":
            source_inputs["drive_audio"] = ["10", 1]
        if case["first_frame"]:
            source_inputs["first_frame"] = ["11", 0]
        if case["last_frame"]:
            source_inputs["last_frame"] = ["12", 0]
        trim_audio_link = ["7", 1] if case["output_audio"] == "locked_source" else ["13", 1]
        case_seed = seed + case_index
        prompt: dict[str, Any] = {
            **_loaders(model_name, clip_name, video_vae_name, audio_vae_name),
            "5": {
                "class_type": "MiniMaxH3SPEEDPlanT8Advanced",
                "inputs": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "scales": scales,
                    "transition_mode": "manual_sigmas",
                    "manual_transition_sigmas": transition_sigma,
                    "delta": 0.01,
                    "shift_video": shift_video,
                    "transform": "dct",
                    "profile_policy": "require_validated_profile",
                    "fallback_policy": "error",
                },
                "_meta": {"title": "Two-stage official-math SPEED plan"},
            },
            "6": {
                "class_type": "MiniMaxH3SPEEDSourceT8Advanced",
                "inputs": source_inputs,
                "_meta": {
                    "title": f"Raw {case['task_type']} + {case['audio_mode']} stage source"
                },
            },
            "7": {
                "class_type": "MiniMaxH3SPEEDSamplerT8Advanced",
                "inputs": {
                    "model": ["1", 0],
                    "speed_plan": ["5", 0],
                    "speed_source": ["6", 0],
                    "shift_audio": shift_audio,
                    "seed": case_seed,
                    "execution_scope": "multimodal_research_exp",
                    "dct_chunk_size": 64,
                },
                "_meta": {"title": "P2 multimodal SPEED execution"},
            },
            "8": {
                "class_type": "LoadVideo",
                "inputs": {"file": source_video},
                "_meta": {"title": "Shared exact-24fps source video"},
            },
            "9": {
                "class_type": "Video Slice",
                "inputs": {
                    "video": ["8", 0],
                    "start_time": 0.0,
                    "duration": duration,
                    "strict_duration": True,
                },
                "_meta": {"title": "Bound source decoding to the 124-frame target window"},
            },
            "10": {
                "class_type": "GetVideoComponents",
                "inputs": {"video": ["9", 0]},
                "_meta": {"title": "Shared source frames and 32kHz soundtrack"},
            },
            "11": {
                "class_type": "ImageFromBatch",
                "inputs": {"image": ["10", 0], "batch_index": 0, "length": 1},
                "_meta": {"title": "Exact first frame"},
            },
            "12": {
                "class_type": "ImageFromBatch",
                "inputs": {"image": ["10", 0], "batch_index": -1, "length": 1},
                "_meta": {"title": "Exact last frame"},
            },
            "13": {
                "class_type": "MiniMaxH3AVDecodeT8",
                "inputs": {
                    "av_latent": ["7", 0],
                    "video_vae": ["3", 0],
                    "audio_vae": ["4", 0],
                },
                "_meta": {"title": "Decode generated H3 video and audio"},
            },
            "14": {
                "class_type": "MiniMaxH3OutputTrimT8",
                "inputs": {
                    "frames": ["13", 0],
                    "start_seconds": 0.0,
                    "duration_seconds": duration,
                    "fps": 24.0,
                    "audio": trim_audio_link,
                },
                "_meta": {"title": f"Exact A/V trim; audio={case['output_audio']}"},
            },
            "15": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["14", 0],
                    "fps": 24.0,
                    "audio": ["14", 1],
                    "bit_depth": 8,
                },
                "_meta": {"title": "Create exact 24fps review video"},
            },
            "16": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["15", 0],
                    "filename_prefix": f"{filename_prefix}/{case['name']}",
                    "format": "mp4",
                    "codec": "h264",
                },
                "_meta": {"title": "Save P2 SPEED result"},
            },
            "17": {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["7", 4],
                    "filename_prefix": f"{filename_prefix}/{case['name']}_speed_report",
                    "format": "json",
                },
                "_meta": {"title": "Save SPEED execution report"},
            },
            "18": {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["6", 1],
                    "filename_prefix": f"{filename_prefix}/{case['name']}_source_report",
                    "format": "json",
                },
                "_meta": {"title": "Save resolved task/source report"},
            },
        }
        prompts[str(case["name"])] = prompt
    manifest = {
        "schema": "minimax_h3_speed_p2_multimodal_validation_v1",
        "controlled": {
            "source_video": source_video,
            "width": width,
            "height": height,
            "length": length,
            "fps": 24.0,
            "steps": steps,
            "base_seed": seed,
            "shift_video": shift_video,
            "shift_audio": shift_audio,
            "scales": scales,
            "transition_sigma": transition_sigma,
            "model_name": model_name,
            "clip_name": clip_name,
            "video_vae_name": video_vae_name,
            "audio_vae_name": audio_vae_name,
        },
        "coverage": [
            {
                key: case[key]
                for key in (
                    "name",
                    "task_type",
                    "audio_mode",
                    "first_frame",
                    "last_frame",
                    "output_audio",
                )
            }
            for case in CASES
        ],
        "claims": {
            "mechanically_validated": False,
            "quality_validated": False,
            "audio_noninferiority_validated": False,
            "memory_safe_16gb": False,
        },
    }
    return prompts, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build three minimal H3 SPEED P2 multimodal API validation workflows."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source-video", default="face_refine_validation_dance_362_736x416.mp4"
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2608192001)
    parser.add_argument("--scales", default="0.5,1.0")
    parser.add_argument("--transition-sigma", default="0.85")
    parser.add_argument("--shift-video", type=float, default=12.0)
    parser.add_argument("--shift-audio", type=float, default=3.0)
    parser.add_argument(
        "--model-name", default="minimax_h3_fl2va_int8_convrot.safetensors"
    )
    parser.add_argument(
        "--clip-name", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    parser.add_argument(
        "--video-vae-name", default="minimax_h3_video_vae_fp16.safetensors"
    )
    parser.add_argument(
        "--audio-vae-name", default="minimax_h3_audio_vae_fp32.safetensors"
    )
    parser.add_argument(
        "--filename-prefix", default="MiniMaxH3/SPEED_P2_multimodal_v1"
    )
    args = parser.parse_args()
    prompts, manifest = build_multimodal_speed_prompts(
        source_video=args.source_video,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        scales=args.scales,
        transition_sigma=args.transition_sigma,
        shift_video=args.shift_video,
        shift_audio=args.shift_audio,
        model_name=args.model_name,
        clip_name=args.clip_name,
        video_vae_name=args.video_vae_name,
        audio_vae_name=args.audio_vae_name,
        filename_prefix=args.filename_prefix,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in prompts.items():
        (args.output_dir / f"{name}_api.json").write_text(
            json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
