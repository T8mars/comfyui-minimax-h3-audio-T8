from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "ref_image_native",
        "task_type": "Ref2VA",
        "prompt": (
            "Use <Picture 1> as the visual identity, material and style reference. "
            "Create natural cinematic motion with synchronized ambience and no speech."
        ),
        "first_frame": False,
        "ref_image": True,
        "ref_video_audio": False,
        "ref_audio": False,
    },
    {
        "name": "ref_video_audio_native",
        "task_type": "Ref2VA",
        "prompt": (
            "Use <Video 1> as the motion and visual reference and <Audio 1> as its associated "
            "soundtrack reference. Create a new coherent shot; add no speech."
        ),
        "first_frame": False,
        "ref_image": False,
        "ref_video_audio": True,
        "ref_audio": False,
    },
    {
        "name": "hybrid_first_image_audio",
        "task_type": "Hybrid",
        "prompt": (
            "Begin exactly from the supplied first frame. Use <Picture 1> as a separate identity "
            "and material reference and <Audio 1> as an ambience reference; add no speech."
        ),
        "first_frame": True,
        "ref_image": True,
        "ref_video_audio": False,
        "ref_audio": True,
    },
)


def _validate_dimensions(width: int, height: int, length: int, steps: int) -> None:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("length must follow the H3 17n+5 grid")
    if steps != 20:
        raise ValueError("P3 reference validation is frozen to Stock20")


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
            "_meta": {"title": "Stock MiniMax H3 Ref2VA model"},
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


def build_reference_speed_prompts(
    *,
    source_video: str,
    reference_image: str,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    steps: int = 20,
    seed: int = 2608193001,
    scales: str = "0.5,1.0",
    transition_sigma: str = "0.85",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    model_name: str = "minimax_h3_ref2va_int8_convrot.safetensors",
    clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors",
    filename_prefix: str = "MiniMaxH3/SPEED_P3_reference_v1",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _validate_dimensions(width, height, length, steps)
    if not source_video.strip() or not reference_image.strip():
        raise ValueError("source_video and reference_image must not be empty")
    target_duration = length / 24.0
    prompts: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(CASES):
        source_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "video_vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": case["prompt"],
            "length": length,
            "task_type": case["task_type"],
            "audio_mode": "native",
            "audio_denoise_strength": 1.0,
            "add_source_as_reference": False,
            "prompt_primary_audio_ordinal": 0,
            "strict_prompt_tags": True,
            "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s",
            "checkpoint_fingerprint": "unrecorded",
            "vae_fingerprint": "unrecorded",
        }
        if case["first_frame"]:
            source_inputs["first_frame"] = ["12", 0]
        if case["ref_image"]:
            source_inputs["ref_images.ref_image_0"] = ["11", 0]
        if case["ref_video_audio"]:
            source_inputs["ref_videos.ref_video_0"] = ["10", 0]
            source_inputs["ref_video_audios.ref_video_audio_0"] = ["10", 1]
        if case["ref_audio"]:
            source_inputs["ref_audios.ref_audio_0"] = ["10", 1]
        name = str(case["name"])
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
                "_meta": {"title": f"Raw {case['task_type']} reference source"},
            },
            "7": {
                "class_type": "MiniMaxH3SPEEDSamplerT8Advanced",
                "inputs": {
                    "model": ["1", 0],
                    "speed_plan": ["5", 0],
                    "speed_source": ["6", 0],
                    "shift_audio": shift_audio,
                    "seed": seed + index,
                    "execution_scope": "multimodal_research_exp",
                    "dct_chunk_size": 64,
                },
                "_meta": {"title": "P3 reference SPEED execution"},
            },
            "8": {
                "class_type": "LoadVideo",
                "inputs": {"file": source_video},
                "_meta": {"title": "Reference/source media"},
            },
            "9": {
                "class_type": "Video Slice",
                "inputs": {
                    "video": ["8", 0],
                    "start_time": 0.0,
                    "duration": 2.0,
                    "strict_duration": True,
                },
                "_meta": {"title": "Official-minimum two-second video reference"},
            },
            "10": {
                "class_type": "GetVideoComponents",
                "inputs": {"video": ["9", 0]},
                "_meta": {"title": "48-frame reference and its soundtrack"},
            },
            "11": {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image},
                "_meta": {"title": "Independent visual reference"},
            },
            "12": {
                "class_type": "ImageFromBatch",
                "inputs": {"image": ["10", 0], "batch_index": 0, "length": 1},
                "_meta": {"title": "Hybrid first-frame anchor"},
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
                    "audio": ["13", 1],
                    "start_seconds": 0.0,
                    "duration_seconds": target_duration,
                    "fps": 24.0,
                },
                "_meta": {"title": "Exact 124-frame A/V trim"},
            },
            "15": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["14", 0],
                    "audio": ["14", 1],
                    "fps": 24.0,
                    "bit_depth": 8,
                },
                "_meta": {"title": "Create exact 24fps review video"},
            },
            "16": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["15", 0],
                    "filename_prefix": f"{filename_prefix}/{name}",
                    "format": "mp4",
                    "codec": "h264",
                },
                "_meta": {"title": "Save P3 SPEED result"},
            },
            "17": {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["7", 4],
                    "filename_prefix": f"{filename_prefix}/{name}_speed_report",
                    "format": "json",
                },
                "_meta": {"title": "Save SPEED execution report"},
            },
            "18": {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["6", 1],
                    "filename_prefix": f"{filename_prefix}/{name}_source_report",
                    "format": "json",
                },
                "_meta": {"title": "Save resolved reference report"},
            },
        }
        prompts[name] = prompt
    manifest = {
        "schema": "minimax_h3_speed_p3_reference_validation_v1",
        "controlled": {
            "source_video": source_video,
            "reference_image": reference_image,
            "reference_video_seconds": 2.0,
            "width": width,
            "height": height,
            "length": length,
            "fps": 24.0,
            "steps": steps,
            "base_seed": seed,
            "scales": scales,
            "transition_sigma": transition_sigma,
            "model_name": model_name,
        },
        "coverage": [
            {
                key: case[key]
                for key in (
                    "name",
                    "task_type",
                    "first_frame",
                    "ref_image",
                    "ref_video_audio",
                    "ref_audio",
                )
            }
            for case in CASES
        ],
        "claims": {
            "mechanically_validated": False,
            "reference_quality_validated": False,
            "audio_noninferiority_validated": False,
            "memory_safe_16gb": False,
        },
    }
    return prompts, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build three minimal H3 SPEED P3 reference API workflows."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source-video", default="face_refine_validation_dance_362_736x416.mp4"
    )
    parser.add_argument("--reference-image", default="t8_dynamic_guidance_1v1_736x416.png")
    args = parser.parse_args()
    prompts, manifest = build_reference_speed_prompts(
        source_video=args.source_video,
        reference_image=args.reference_image,
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
