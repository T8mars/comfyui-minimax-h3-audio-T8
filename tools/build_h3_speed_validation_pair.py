from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "A cinematic night city in rain, camera gliding forward through reflections "
    "and mist. Natural synchronized rain, distant traffic and wind, no speech."
)


def _loader_nodes(
    *,
    model_name: str,
    clip_name: str,
    video_vae_name: str,
    audio_vae_name: str,
) -> dict[str, dict[str, Any]]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load stock non-pruned H3 model"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "minimax",
                "device": "default",
            },
            "_meta": {"title": "Load H3 Qwen text encoder"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
            "_meta": {"title": "Load H3 video VAE"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
            "_meta": {"title": "Load H3 audio VAE"},
        },
    }


def _save_nodes(*, latent_link: list[Any], filename_prefix: str) -> dict[str, dict[str, Any]]:
    return {
        "10": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": latent_link,
                "video_vae": ["3", 0],
                "audio_vae": ["4", 0],
            },
            "_meta": {"title": "Decode synchronized H3 audio and video"},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["10", 0],
                "fps": 24.0,
                "audio": ["10", 1],
                "bit_depth": 8,
            },
            "_meta": {"title": "Create synchronized 24fps video"},
        },
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": filename_prefix,
                "format": "mp4",
                "codec": "h264",
            },
            "_meta": {"title": "Save controlled SPEED validation output"},
        },
    }


def build_t2va_pair(
    *,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int,
    prompt: str,
    scales: str,
    transition_sigma: str,
    shift_video: float,
    shift_audio: float,
    model_name: str,
    clip_name: str,
    video_vae_name: str,
    audio_vae_name: str,
    filename_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("length must follow the H3 17n+5 frame grid")
    if steps != 20:
        raise ValueError("the strict SPEED P1 validation pair requires exactly 20 steps")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    common = _loader_nodes(
        model_name=model_name,
        clip_name=clip_name,
        video_vae_name=video_vae_name,
        audio_vae_name=audio_vae_name,
    )
    baseline = {
        **common,
        "5": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "clip": ["2", 0],
                "video_vae": ["3", 0],
                "audio_vae": ["4", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
            "_meta": {"title": "Full-resolution T2VA conditioning"},
        },
        "6": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["1", 0],
                "av_latent": ["5", 1],
                "steps": steps,
                "shift_video": shift_video,
                "shift_audio": shift_audio,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
            "_meta": {"title": "Full-resolution stock H3 Euler"},
        },
        "7": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["6", 0], "conditioning": ["5", 0]},
            "_meta": {"title": "CFG-false BasicGuider"},
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
            "_meta": {"title": "Controlled seed"},
        },
        "9": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["7", 0],
                "sampler": ["6", 1],
                "sigmas": ["6", 2],
                "latent_image": ["5", 1],
            },
            "_meta": {"title": "Full-resolution Stock20 baseline"},
        },
        **_save_nodes(
            latent_link=["9", 0],
            filename_prefix=f"{filename_prefix}/baseline_stock20",
        ),
    }

    speed = {
        **common,
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
            "_meta": {"title": "Official-math SPEED stage plan"},
        },
        "6": {
            "class_type": "MiniMaxH3SPEEDSourceT8Advanced",
            "inputs": {
                "clip": ["2", 0],
                "video_vae": ["3", 0],
                "audio_vae": ["4", 0],
                "prompt": prompt,
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
            "_meta": {"title": "Raw T2VA source rebuilt at each SPEED canvas"},
        },
        "7": {
            "class_type": "MiniMaxH3SPEEDSamplerT8Advanced",
            "inputs": {
                "model": ["1", 0],
                "speed_plan": ["5", 0],
                "speed_source": ["6", 0],
                "shift_audio": shift_audio,
                "seed": seed,
                "execution_scope": "strict_t2va_stock20",
                "dct_chunk_size": 64,
            },
            "_meta": {"title": "Strict T2VA Stock20 SPEED whole-chain sampler"},
        },
        **_save_nodes(
            latent_link=["7", 0],
            filename_prefix=f"{filename_prefix}/speed_stock20",
        ),
    }

    manifest = {
        "schema": "minimax_h3_speed_p1_t2va_pair_v1",
        "controlled": {
            "model_name": model_name,
            "clip_name": clip_name,
            "video_vae_name": video_vae_name,
            "audio_vae_name": audio_vae_name,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "seed": seed,
            "shift_video": shift_video,
            "shift_audio": shift_audio,
            "sampler": "Euler",
            "scheduler": "native H3 flow",
        },
        "treatment": {
            "scales": scales,
            "transition_sigma": transition_sigma,
            "transform": "orthonormal DCT",
            "total_nfe_unchanged": True,
        },
        "claims": {
            "quality_validated": False,
            "speedup_validated": False,
            "audio_noninferiority_validated": False,
            "vram_safe_16gb": False,
        },
    }
    return baseline, speed, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a controlled full-resolution Stock20 versus H3 SPEED T2VA API pair."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=1056)
    parser.add_argument("--height", type=int, default=608)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2608184001)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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
        "--filename-prefix", default="MiniMaxH3/SPEED_P1_T2VA_v1"
    )
    args = parser.parse_args()

    baseline, speed, manifest = build_t2va_pair(
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        prompt=args.prompt,
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
    for name, payload in (
        ("baseline_api.json", baseline),
        ("speed_api.json", speed),
        ("pair_manifest.json", manifest),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
