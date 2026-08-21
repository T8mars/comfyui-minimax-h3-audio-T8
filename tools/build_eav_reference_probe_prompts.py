#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ("Ref2VA", "Hybrid")
MODES = ("disabled", "apply_exp")
REFERENCE_IMAGE = "ComfyUI_00002_pjoxu_1780997336.png"


def build_prompt(task: str, mode: str) -> dict[str, Any]:
    if task not in TASKS or mode not in MODES:
        raise ValueError(f"unsupported EAV reference probe: {task}/{mode}")
    seed = 2608217302 if task == "Ref2VA" else 2608217303
    if task == "Ref2VA":
        prompt_text = (
            "Use <Picture 1> as the authoritative identity, face, pink Hanfu and nine-tail "
            "appearance reference. In one continuous moonlit cinematic shot, the same adult fox "
            "woman raises the pink flame, spins quickly clockwise with sleeves and nine tails "
            "sweeping in wide arcs, then leaps backward through the courtyard while the camera "
            "cranes rapidly upward until she becomes a small full-body figure. Preserve a stable "
            "recognizable face, coherent costume, natural anatomy and fluid large-amplitude motion. "
            "No dialogue. Synchronized flame, cloth, wind, landing and night ambience."
        )
    else:
        prompt_text = (
            "Use <Picture 1> as the exact first frame and <Picture 2> as the authoritative identity, "
            "face, pink Hanfu and nine-tail appearance reference. In one continuous moonlit "
            "cinematic shot, the same adult fox woman lowers the pink flame, spins quickly clockwise "
            "with sleeves and nine tails sweeping in wide arcs, then leaps backward through the "
            "courtyard while the camera cranes rapidly upward until she becomes a small full-body "
            "figure. Preserve a stable recognizable face, coherent costume, natural anatomy and "
            "fluid large-amplitude motion. No dialogue. Synchronized flame, cloth, wind, landing "
            "and night ambience."
        )
    conditioning_inputs: dict[str, Any] = {
        "clip": ["3", 0],
        "video_vae": ["1", 0],
        "audio_vae": ["2", 0],
        "prompt": prompt_text,
        "width": 1152,
        "height": 640,
        "length": 124,
        "task_type": task,
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "ref_images.ref_image_0": ["14", 0],
    }
    if task == "Hybrid":
        conditioning_inputs["first_frame"] = ["15", 0]
    prefix = f"MiniMaxH3_EAV_Reference/eav_{task.lower()}_stock20_{mode}_seed{seed}"
    graph: dict[str, Any] = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
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
                "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "5": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": conditioning_inputs},
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
            "class_type": "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced",
            "inputs": {
                "model": ["6", 0],
                "sigmas": ["6", 2],
                "mode": mode,
                "tau": 4.0,
                "start_video_progress": 0.0,
                "end_video_progress": 1.0,
                "max_workspace_mib": 32,
                "g_hard_limit": 1.5,
            },
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
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
            "inputs": {"av_latent": ["11", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]},
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "audio": ["12", 1],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
        "14": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}},
    }
    if task == "Hybrid":
        graph["15"] = {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}}
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Build four real EAV reference-composer probe prompts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for task in TASKS:
        for mode in MODES:
            name = f"{task.lower()}_{mode}.json"
            path = args.output_dir / name
            path.write_text(
                json.dumps(build_prompt(task, mode), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            files.append(str(path.resolve()))
    print(json.dumps({"prompts": files}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
