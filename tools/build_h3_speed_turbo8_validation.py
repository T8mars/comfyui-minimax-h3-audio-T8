from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_turbo8_speed_prompt(
    *,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    seed: int = 2608194001,
    model_name: str = "minimax_h3_fl2va_int8_convrot.safetensors",
    lora_name: str = "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
    clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors",
) -> dict[str, dict[str, Any]]:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("length must follow the H3 17n+5 grid")
    duration = length / 24.0
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Stock MiniMax H3 FL2VA base"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": 1.0},
            "_meta": {"title": "Explicit user-selected Turbo LoRA"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
            "_meta": {"title": "MiniMax H3 Qwen encoder"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
            "_meta": {"title": "MiniMax H3 video VAE"},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
            "_meta": {"title": "MiniMax H3 audio VAE"},
        },
        "6": {
            "class_type": "MiniMaxH3SPEEDPlanT8Advanced",
            "inputs": {
                "width": width,
                "height": height,
                "steps": 8,
                "scales": "0.5,1.0",
                "transition_mode": "manual_sigmas",
                "manual_transition_sigmas": "0.85",
                "delta": 0.01,
                "shift_video": 12.0,
                "transform": "dct",
                "profile_policy": "require_validated_profile",
                "fallback_policy": "error",
            },
            "_meta": {"title": "Two-stage SPEED plan; total NFE stays eight"},
        },
        "7": {
            "class_type": "MiniMaxH3SPEEDSourceT8Advanced",
            "inputs": {
                "clip": ["3", 0],
                "video_vae": ["4", 0],
                "audio_vae": ["5", 0],
                "prompt": (
                    "A woman in flowing red Hanfu spins rapidly through the night sky. "
                    "Coherent anatomy, crisp fabric detail, cinematic moonlight, synchronized wind; no speech."
                ),
                "length": length,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "checkpoint_fingerprint": "unrecorded",
                "vae_fingerprint": "unrecorded",
            },
            "_meta": {"title": "Media-free T2VA Turbo8 source"},
        },
        "8": {
            "class_type": "MiniMaxH3SPEEDSamplerT8Advanced",
            "inputs": {
                "model": ["2", 0],
                "speed_plan": ["6", 0],
                "speed_source": ["7", 0],
                "shift_audio": 3.0,
                "seed": seed,
                "execution_scope": "turbo8_t2va_research_exp",
                "dct_chunk_size": 64,
            },
            "_meta": {"title": "Turbo8-specific fail-closed SPEED scope"},
        },
        "9": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["8", 0],
                "video_vae": ["4", 0],
                "audio_vae": ["5", 0],
            },
            "_meta": {"title": "Decode generated H3 video and audio"},
        },
        "10": {
            "class_type": "MiniMaxH3OutputTrimT8",
            "inputs": {
                "frames": ["9", 0],
                "audio": ["9", 1],
                "start_seconds": 0.0,
                "duration_seconds": duration,
                "fps": 24.0,
            },
            "_meta": {"title": "Exact 124-frame A/V trim"},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["10", 0],
                "audio": ["10", 1],
                "fps": 24.0,
                "bit_depth": 8,
            },
            "_meta": {"title": "Create review video"},
        },
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": "MiniMaxH3/SPEED_Turbo8_v1/t2va_turbo8",
                "format": "mp4",
                "codec": "h264",
            },
            "_meta": {"title": "Save Turbo8 SPEED result"},
        },
        "13": {
            "class_type": "SaveText",
            "inputs": {
                "text": ["8", 4],
                "filename_prefix": "MiniMaxH3/SPEED_Turbo8_v1/t2va_turbo8_speed_report",
                "format": "json",
            },
            "_meta": {"title": "Save Turbo8 execution report"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one H3 SPEED Turbo8 API workflow.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prompt = build_turbo8_speed_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
