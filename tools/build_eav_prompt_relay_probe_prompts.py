#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


MODES = ("disabled", "apply_exp")
SEED = 2608217401
WIDTH = 736
HEIGHT = 416
GLOBAL_PROMPT = (
    "A single continuous cinematic night shot on a rain-wet neon street. The same adult "
    "woman in a long red coat remains clearly recognizable with natural anatomy, realistic "
    "skin, coherent clothing and stable lighting. At the beginning she says exactly once in "
    "Mandarin: ‘你在干嘛呢，我在这里呀，看看效果如何。’ After finishing that sentence she "
    "remains silent. Preserve intelligible natural speech, synchronized footsteps, fabric "
    "movement and city ambience. No cuts and no additional speech."
)
LOCAL_PROMPTS = (
    "The woman stands at the center of the street, looks toward the camera, and clearly "
    "raises her right hand above shoulder level.\n"
    "She lowers her hand, turns clockwise, and walks quickly toward the bright blue doorway "
    "behind her.\n"
    "The camera cranes upward and rapidly pulls far back while she continues walking, "
    "becoming a small full-body figure in the wide city view."
)


def _base_prompt() -> dict[str, Any]:
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
            "class_type": "MiniMaxH3PromptRelayPlanT8Advanced",
            "inputs": {
                "global_prompt": GLOBAL_PROMPT,
                "local_prompts": LOCAL_PROMPTS,
                "length": 124,
                "timing_mode": "frames",
                "time_ranges": "0-40\n41-81\n82-123",
                "math_profile": "paper_v1",
                "epsilon": 0.1,
                "allow_gaps": False,
                "allow_overlaps": False,
            },
        },
        "6": {
            "class_type": "MiniMaxH3PromptRelayConditioningT8Advanced",
            "inputs": {
                "model": ["4", 0],
                "clip": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
                "prompt_relay_plan": ["5", 0],
                "width": WIDTH,
                "height": HEIGHT,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "execution_mode": "apply_exp",
                "query_chunk_rows": 256,
            },
        },
        "7": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["6", 0],
                "av_latent": ["6", 2],
                "steps": 20,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "9": {
            "class_type": "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
            "inputs": {
                "model": ["7", 0],
                "sigmas": ["7", 2],
                "mode": "disabled",
                "tau": 4.0,
                "start_video_progress": 0.0,
                "end_video_progress": 1.0,
                "max_workspace_mib": 32,
                "g_hard_limit": 1.5,
                "sampling_profile": "stock20",
            },
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["9", 0], "conditioning": ["6", 1]},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["10", 0],
                "sampler": ["7", 1],
                "sigmas": ["7", 2],
                "latent_image": ["6", 2],
            },
        },
        "12": {
            "class_type": "MiniMaxH3EnhanceAVideoAuditT8Advanced",
            "inputs": {"av_latent": ["11", 0], "runtime": ["9", 1]},
        },
        "13": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["12", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        "14": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["13", 0],
                "audio": ["13", 1],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "",
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


def build_prompt(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported EAV Prompt Relay probe mode: {mode}")
    prompt = deepcopy(_base_prompt())
    prompt["9"]["inputs"]["mode"] = mode
    label = "relay_only" if mode == "disabled" else "relay_plus_eav"
    prompt["14"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3_EAV_PromptRelay/eav_prompt_relay_{label}_stock20_seed{SEED}"
    )
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a low-load 0.3MP Relay-only versus Relay+EAV API pair."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for mode in MODES:
        label = "relay_only" if mode == "disabled" else "relay_plus_eav"
        path = args.output_dir / f"{label}.json"
        path.write_text(
            json.dumps(build_prompt(mode), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        files.append(str(path.resolve()))
    print(json.dumps({"prompts": files}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
