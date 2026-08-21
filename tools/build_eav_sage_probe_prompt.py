#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPT = (
    "Night, one continuous high-speed cinematic shot. An adult woman in flowing red "
    "Hanfu launches across a moonlit rooftop, spins rapidly through the air, and swings "
    "a sword that throws bright sparks. The camera whip-pans with her and then pulls far "
    "back until she becomes a small full-body figure against the city skyline. Preserve "
    "a coherent face, natural anatomy, crisp fabric folds, stable limbs and smooth "
    "large-amplitude motion. No dialogue. Synchronized rushing wind, cloth movement, "
    "footsteps, sword whooshes, sparks and distant night ambience."
)


def build_prompt() -> dict[str, Any]:
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
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "device": "default",
                "type": "minimax",
            },
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "5": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "clip": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
                "prompt": PROMPT,
                "width": 1152,
                "height": 640,
                "length": 124,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
        },
        "6": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["4", 0],
                "av_latent": ["5", 1],
                "steps": 20,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
        },
        "7": {
            "class_type": "MiniMaxH3EnhanceAVideoSageComposerT8Advanced",
            "inputs": {
                "model": ["6", 0],
                "sigmas": ["6", 2],
                "task_scope": "visual",
                "mode": "apply_exp",
                "tau": 4.0,
                "start_video_progress": 0.0,
                "end_video_progress": 1.0,
                "max_workspace_mib": 32,
                "g_hard_limit": 1.5,
                "sampling_profile": "stock20",
            },
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2608217001}},
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["7", 0], "conditioning": ["5", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["6", 1],
                "sigmas": ["6", 2],
                "latent_image": ["5", 1],
            },
        },
        "11": {
            "class_type": "MiniMaxH3EnhanceAVideoAuditT8Advanced",
            "inputs": {"av_latent": ["10", 0], "runtime": ["7", 1]},
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
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": (
                    "MiniMaxH3_EAV/eav_strict_sage_t2va_stock20_tau4_seed2608217001"
                ),
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one EAV + Strict Sage 0.7MP probe.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_prompt(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
