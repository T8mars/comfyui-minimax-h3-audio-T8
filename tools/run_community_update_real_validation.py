#!/usr/bin/env python3
"""Run one serial, low-load real H3 validation for the 2026-08-29 nodes.

Each invocation performs exactly one model render.  The isolated ComfyUI process
is stopped before the script exits so callers can run the modes serially without
turning this into a pressure test.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import psutil


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402


SCHEMA = "t8.minimax_h3.community_update_real_validation.v1"
FPS = 24
SEED = 2608291001
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
FL_BASE = "minimax_h3_fl2va_int8_convrot.safetensors"
REF_BASE = "minimax_h3_ref2va_int8_convrot.safetensors"
FAST_LORA = r"dense-datafree\adapter_model.safetensors"
FAST_VSA_LORA = r"FastH3-VSA\vsa-datafree\adapter_model.safetensors"
PDD_FL = "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors"
PDD_REF = "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors"
TURBO_ALPHA8 = "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
REFERENCE_IMAGE = "codex_prompt_relay_fl2va_first.png"
STATIC_BACKGROUND_MASK = "codex_h3_static_background_subject_mask_576x320.png"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _video_mask_name(args: argparse.Namespace) -> str:
    """Return a ComfyUI input-relative mask name without changing old defaults."""

    value = Path(getattr(args, "video_mask", STATIC_BACKGROUND_MASK))
    if value.is_absolute():
        input_root = (args.comfy_root / "input").resolve()
        try:
            value = value.resolve().relative_to(input_root)
        except ValueError as error:
            raise ValueError(
                "--video-mask must be inside the selected ComfyUI input directory"
            ) from error
    return value.as_posix()


def _video_mask_path(args: argparse.Namespace) -> Path:
    value = Path(getattr(args, "video_mask", STATIC_BACKGROUND_MASK))
    if value.is_absolute():
        return value.resolve()
    return (args.comfy_root / "input" / value).resolve()


def _conditioning(
    *,
    task: str,
    clip_node: str,
    width: int,
    height: int,
    frames: int,
    prompt: str,
    reference_node: str | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "clip": [clip_node, 0],
        "video_vae": ["1", 0],
        "audio_vae": ["2", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
        "task_type": task,
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
    }
    if task == "FL2VA":
        inputs["first_frame"] = [reference_node or "5", 0]
        inputs["last_frame"] = [reference_node or "5", 0]
    elif task == "Ref2VA":
        inputs["ref_images.ref_image_0"] = [reference_node or "5", 0]
    return inputs


def _loaders(base: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"},
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": base, "weight_dtype": "default"},
        },
        "5": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}},
    }


def _save_nodes(
    *,
    sampled: list[Any],
    run_id: str,
    mode: str,
    decode_id: str,
    save_id: str,
) -> dict[str, Any]:
    return {
        decode_id: {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": sampled,
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        save_id: {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [decode_id, 0],
                "audio": [decode_id, 1],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": f"MiniMaxH3_Community_Real/{run_id}_{mode}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def _fast_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    use_vsa = args.mode == "fast_h3_vsa"
    selected_lora = FAST_VSA_LORA if use_vsa else FAST_LORA
    attention_profile = "external_vsa_if_available" if use_vsa else "dense_comfyui"
    output_mode = "fast_h3_vsa" if use_vsa else "fast_h3"
    prompt = _loaders(FL_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "lora_name": selected_lora,
                    "strength_model": 1.0,
                },
            },
            "7": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="T2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    prompt=(
                        "A photorealistic adult woman with a natural symmetrical face, shoulder-length brown "
                        "hair and a red coat stands on a softly lit night street. A steady medium close-up: "
                        "she remains facing the "
                        "camera with only a very small natural head movement and one gentle blink. Her lips "
                        "stay closed and relaxed; no speech and no mouth movement. Stable facial geometry, "
                        "stable eyes and anatomy, soft night city ambience, one continuous shot, no subtitles."
                    ),
                ),
            },
            "8": {
                "class_type": "MiniMaxH3FastH34StepSetupT8Advanced",
                "inputs": {
                    "model": ["6", 0],
                    "av_latent": ["7", 1],
                    "task_family": "t2va_only",
                    "attention_profile": attention_profile,
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
            "12": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
            "13": {"class_type": "PreviewAny", "inputs": {"source": ["8", 3]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["11", 0], run_id=run_id, mode=output_mode, decode_id="14", save_id="15"
        )
    )
    return prompt, {"lora": "12", "fast": "13"}


def _timed_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    prompt = _loaders(REF_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "VHS_LoadVideoPath",
                "inputs": {
                    "video": str(args.timed_source_video),
                    "force_rate": 24,
                    "custom_width": 320,
                    "custom_height": 192,
                    "frame_load_cap": 24,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
            },
            "7": {
                "class_type": "MiniMaxH3TimedImageReferenceT8Advanced",
                "inputs": {
                    "clip": ["3", 0],
                    "image": ["5", 0],
                    "prompt_tag": "lighting",
                    "time_seconds": 0.25,
                    "image_size": "256",
                },
            },
            "8": {
                "class_type": "MiniMaxH3TimedVideoReferenceT8Advanced",
                "inputs": {
                    "clip": ["7", 0],
                    "video_frames": ["6", 0],
                    "prompt_tag": "motion",
                    "target_start_seconds": 0.0,
                    "source_fps": 24.0,
                    "analysis_fps": 2.0,
                    "video_size": "256",
                },
            },
            "9": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="Ref2VA",
                    clip_node="8",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="5",
                    prompt=(
                        "Use <Picture 1> for the adult woman's identity. At #lighting, keep "
                        "the soft night illumination. Following #motion, use only a subtle, slow "
                        "head turn while keeping her identity and facial geometry unchanged. A steady "
                        "medium close-up with relaxed closed lips, no speech and no mouth movement, "
                        "stable eyes and anatomy, one continuous shot, no subtitles or cuts."
                    ),
                ),
            },
            "10": {
                "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "av_latent": ["9", 1],
                    "pdd_lora_name": PDD_REF,
                    "base_variant": "Ref2VA",
                    "strength": 1.0,
                },
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 1}},
            "12": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["10", 0], "conditioning": ["9", 0]},
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["12", 0],
                    "sampler": ["10", 1],
                    "sigmas": ["10", 2],
                    "latent_image": ["9", 1],
                },
            },
            "14": {"class_type": "PreviewAny", "inputs": {"source": ["9", 3]}},
            "15": {"class_type": "PreviewAny", "inputs": {"source": ["9", 5]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["13", 0], run_id=run_id, mode="timed_reference", decode_id="16", save_id="17"
        )
    )
    return prompt, {"conditioned_prompt": "14", "conditioning": "15"}


def _chunked_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    global_noise_v2 = args.mode == "chunked_two_pass_global_noise"
    spatial_strategy = (
        args.spatial_strategy if global_noise_v2 else "full_frame_safe"
    )
    prompt = _loaders(FL_BASE)
    # ComfyUI's native H3 FL2VA node intentionally keeps upstream parity:
    # ``first_frame`` is resized directly while ``last_frame`` is center-cropped.
    # Feeding the same off-aspect source to both sockets therefore creates two
    # different keyframe geometries and the model interpolates the mismatch over
    # time.  Normalize once at the final target canvas, then reuse that exact IMAGE
    # for both passes and both endpoint sockets.  This changes only this validation
    # graph; the stable Conditioning node and old workflows keep their contract.
    prompt["6"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["5", 0],
            "upscale_method": "lanczos",
            "width": args.target_width,
            "height": args.target_height,
            "crop": "center",
        },
    }
    prompt.update(
        {
            "7": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=(
                        "One continuous medium close-up of exactly the same adult woman from the first "
                        "and last frames. She remains nearly still, makes one very small natural head "
                        "movement and blinks once. Her lips stay closed and relaxed; no speech and no "
                        "mouth movement. Preserve facial geometry, eyes, skin and anatomy across every "
                        "frame, quiet room ambience, no cuts or subtitles."
                    ),
                ),
            },
            "8": {
                "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "av_latent": ["7", 1],
                    "pdd_lora_name": PDD_FL,
                    "base_variant": "FL2VA",
                    "strength": 1.0,
                },
            },
            "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["8", 2], "step": 4}},
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 2}},
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["10", 0],
                    "sampler": ["8", 1],
                    "sigmas": ["9", 0],
                    "latent_image": ["7", 1],
                },
            },
            "13": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.target_width,
                    height=args.target_height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=(
                        "One continuous medium close-up of exactly the same adult woman from the first "
                        "and last frames. She remains nearly still, makes one very small natural head "
                        "movement and blinks once. Her lips stay closed and relaxed; no speech and no "
                        "mouth movement. Preserve facial geometry, eyes, skin and anatomy across every "
                        "frame, quiet room ambience, no cuts or subtitles."
                    ),
                ),
            },
            "14": {
                "class_type": (
                    "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
                    if global_noise_v2
                    else "MiniMaxH3ChunkedTwoPassPlanT8Advanced"
                ),
                "inputs": {
                    "model_name": UPSCALER,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "temporal_chunk_frames": args.temporal_chunk_frames,
                    "temporal_overlap_frames": args.temporal_overlap_frames,
                    "anchor_strength": 0.999,
                    "tile_width": (
                        args.tile_width
                        if spatial_strategy == "independent_tiles_exp"
                        else args.target_width
                    ),
                    "tile_height": (
                        args.tile_height
                        if spatial_strategy == "independent_tiles_exp"
                        else args.target_height
                    ),
                    "spatial_overlap": (
                        args.spatial_overlap
                        if spatial_strategy == "independent_tiles_exp"
                        else 0
                    ),
                    "spatial_fade": (
                        args.spatial_fade
                        if spatial_strategy == "independent_tiles_exp"
                        else 0
                    ),
                    "minimum_tile_size": 128,
                    "overlap_blend": "smoothstep",
                    "precision": "fp16",
                    "release_policy": "offload_after",
                    "spatial_strategy": spatial_strategy,
                },
            },
            "15": {
                "class_type": "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
                "inputs": {
                    "model": ["8", 0],
                    "conditioning": ["13", 0],
                    "latent": ["12", 1],
                    "noise": ["11", 0],
                    "sampler": ["8", 1],
                    "sigmas": ["9", 1],
                    "plan": ["14", 0],
                    "cfg": 1.0,
                },
            },
            "16": {"class_type": "PreviewAny", "inputs": {"source": ["14", 1]}},
            "17": {"class_type": "PreviewAny", "inputs": {"source": ["15", 1]}},
        }
    )
    if global_noise_v2:
        prompt["14"]["inputs"]["temporal_strategy"] = args.temporal_strategy
    prompt.update(
        _save_nodes(
            sampled=["15", 0], run_id=run_id, mode=args.mode, decode_id="18", save_id="19"
        )
    )
    if args.save_draft:
        prompt.update(
            _save_nodes(
                sampled=["12", 1],
                run_id=run_id,
                mode=f"{args.mode}_draft_low4",
                decode_id="20",
                save_id="21",
            )
        )
    return prompt, {"plan": "16", "execution": "17"}


def _chunked_low_sigma_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Complete pass 1, then apply the upstream-style 3-step/0.30 refine.

    This is deliberately separate from the diagnosed PDD 4+4 prompt.  PDD output
    heads are bound to the official eight-step grid and cannot be repurposed as a
    low-noise refiner.
    """

    generation_prompt = (
        "One continuous medium close-up of exactly the same adult woman from the "
        "first and last frames. She remains nearly still, makes one very small natural "
        "head movement and blinks once. Her lips stay closed and relaxed; no speech and "
        "no mouth movement. Preserve facial geometry, eyes, skin and anatomy across "
        "every frame. Keep the street, signs, parked cars, pedestrians, lighting and all "
        "background objects fixed in place with no drifting, morphing, floating objects, "
        "cuts or subtitles."
    )
    prompt = _loaders(FL_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["5", 0],
                    "upscale_method": "lanczos",
                    "width": args.target_width,
                    "height": args.target_height,
                    "crop": "center",
                },
            },
            "7": {
                "class_type": "LoraLoaderBypassModelOnly",
                "inputs": {
                    "model": ["4", 0],
                    "lora_name": TURBO_ALPHA8,
                    "strength_model": 1.0,
                },
            },
            "8": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=generation_prompt,
                ),
            },
            "9": {
                "class_type": "MiniMaxH3DualClockSamplerT8",
                "inputs": {
                    "model": ["7", 0],
                    "av_latent": ["8", 1],
                    "steps": 8,
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "sampler_name": "dual_clock_euler",
                    "scheduler": "native_flow",
                },
            },
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 2}},
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["10", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["9", 2],
                    "latent_image": ["8", 1],
                },
            },
            "13": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.target_width,
                    height=args.target_height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=generation_prompt,
                ),
            },
            "14": {
                "class_type": "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced",
                "inputs": {
                    "model_name": UPSCALER,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "temporal_chunk_frames": args.temporal_chunk_frames,
                    "temporal_overlap_frames": args.temporal_overlap_frames,
                    "anchor_strength": 0.999,
                    "tile_width": args.target_width,
                    "tile_height": args.target_height,
                    "spatial_overlap": 0,
                    "spatial_fade": 0,
                    "minimum_tile_size": 128,
                    "overlap_blend": "smoothstep",
                    "precision": "fp16",
                    "release_policy": "offload_after",
                    "spatial_strategy": "full_frame_safe",
                    "temporal_strategy": "full_clip_safe",
                    "second_pass_audio_policy": "joint_av_preserve_input",
                },
            },
            "15": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["9", 0],
                    "scheduler": "simple",
                    "steps": 3,
                    "denoise": 0.30,
                },
            },
            "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 3}},
            "17": {
                "class_type": "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
                "inputs": {
                    "model": ["9", 0],
                    "conditioning": ["13", 0],
                    "latent": ["12", 1],
                    "noise": ["16", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["15", 0],
                    "plan": ["14", 0],
                    "cfg": 1.0,
                },
            },
            "18": {"class_type": "PreviewAny", "inputs": {"source": ["14", 1]}},
            "19": {"class_type": "PreviewAny", "inputs": {"source": ["17", 1]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["17", 0], run_id=run_id, mode=args.mode, decode_id="20", save_id="21"
        )
    )
    if args.save_draft:
        prompt.update(
            _save_nodes(
                sampled=["12", 1],
                run_id=run_id,
                mode=f"{args.mode}_draft_first_pass",
                decode_id="22",
                save_id="23",
            )
        )
    return prompt, {"plan": "18", "execution": "19"}


def _chunked_masked_low_sigma_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the v4 route with one static subject-edit mask inherited by both passes."""

    prompt, reports = _chunked_low_sigma_prompt(args, run_id)
    prompt["14"] = {
        "class_type": "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced",
        "inputs": {
            **prompt["14"]["inputs"],
            "video_mask_policy": "inherit_required",
        },
    }
    prompt["9"]["inputs"]["av_latent"] = ["29", 0]
    prompt["12"]["inputs"]["latent_image"] = ["29", 0]
    prompt.update(
        {
            "24": {
                "class_type": "RepeatImageBatch",
                "inputs": {"image": ["30", 0], "amount": args.frame_count},
            },
            "25": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["24", 0], "vae": ["1", 0]},
            },
            "26": {
                "class_type": "LoadImageMask",
                "inputs": {"image": _video_mask_name(args), "channel": "red"},
            },
            "27": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["25", 0], "mask": ["26", 0]},
            },
            "28": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["8", 1]},
            },
            "29": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["27", 0],
                    "audio_latent": ["28", 1],
                },
            },
            "30": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["5", 0],
                    "upscale_method": "lanczos",
                    "width": args.width,
                    "height": args.height,
                    "crop": "center",
                },
            },
        }
    )
    return prompt, reports


def _chunked_upscale_only_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Decode the learned-upscaled masked first pass without any second sampling."""

    prompt, _reports = _chunked_masked_low_sigma_prompt(args, run_id)
    for node_id in range(13, 24):
        prompt.pop(str(node_id), None)
    prompt.update(
        {
            "31": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["12", 1]},
            },
            "32": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["31", 0],
                    "audio_latent": ["31", 1],
                },
            },
            "33": {
                "class_type": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "inputs": {
                    "av_latent": ["32", 0],
                    "model_name": UPSCALER,
                    "size_mode": "target_dimensions",
                    "scale_by": 2.0,
                    "target_megapixels": 1.0,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "aspect_policy": "honor_dimensions_exp",
                    "max_anisotropy": 2.0,
                    "precision": "fp16",
                    "release_policy": "offload_after",
                },
            },
            "34": {"class_type": "PreviewAny", "inputs": {"source": ["33", 3]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["33", 0],
            run_id=run_id,
            mode=args.mode,
            decode_id="35",
            save_id="36",
        )
    )
    if args.save_draft:
        prompt.update(
            _save_nodes(
                sampled=["12", 1],
                run_id=run_id,
                mode=f"{args.mode}_draft_first_pass",
                decode_id="37",
                save_id="38",
            )
        )
    return prompt, {"upscale": "34"}


def _chunked_upstream_exact_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the audited upstream H3LoopingSampler as a 1x1/full-clip control.

    The source node remains external and is never vendored by this project.  With
    124 pixel frames the H3 latent has 37 video tokens, so the upstream temporal
    tile clamps to the full clip and performs one trajectory without a merge.
    """

    generation_prompt = (
        "One continuous medium close-up of exactly the same adult woman from the "
        "first and last frames. She remains nearly still, makes one very small natural "
        "head movement and blinks once. Her lips stay closed and relaxed; no speech and "
        "no mouth movement. Preserve facial geometry, eyes, skin and anatomy across "
        "every frame. Keep the street, signs, parked cars, pedestrians, lighting and all "
        "background objects fixed in place with no drifting, morphing, floating objects, "
        "cuts or subtitles."
    )
    prompt = _loaders(FL_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["5", 0],
                    "upscale_method": "lanczos",
                    "width": args.target_width,
                    "height": args.target_height,
                    "crop": "center",
                },
            },
            "7": {
                "class_type": "LoraLoaderBypassModelOnly",
                "inputs": {
                    "model": ["4", 0],
                    "lora_name": TURBO_ALPHA8,
                    "strength_model": 1.0,
                },
            },
            "8": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=generation_prompt,
                ),
            },
            "9": {
                "class_type": "MiniMaxH3DualClockSamplerT8",
                "inputs": {
                    "model": ["7", 0],
                    "av_latent": ["8", 1],
                    "steps": 8,
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "sampler_name": "dual_clock_euler",
                    "scheduler": "native_flow",
                },
            },
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 2}},
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["10", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["9", 2],
                    "latent_image": ["8", 1],
                },
            },
            "13": {
                "class_type": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "inputs": {
                    "av_latent": ["12", 1],
                    "model_name": UPSCALER,
                    "size_mode": "target_dimensions",
                    "scale_by": 2.0,
                    "target_megapixels": 1.0,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "aspect_policy": "honor_dimensions_exp",
                    "max_anisotropy": 2.0,
                    "precision": "fp16",
                    "release_policy": "offload_after",
                },
            },
            "14": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.target_width,
                    height=args.target_height,
                    frames=args.frame_count,
                    reference_node="6",
                    prompt=generation_prompt,
                ),
            },
            "15": {
                "class_type": "MiniMaxH3DualClockSamplerT8",
                "inputs": {
                    "model": ["7", 0],
                    "av_latent": ["13", 0],
                    "steps": 3,
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "sampler_name": "dual_clock_euler",
                    "scheduler": "native_flow",
                },
            },
            "16": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["15", 0],
                    "scheduler": "simple",
                    "steps": 3,
                    "denoise": 0.30,
                },
            },
            "17": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["15", 0], "conditioning": ["14", 0]},
            },
            "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 3}},
            "24": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
            },
            "19": {
                "class_type": "H3LoopingSampler",
                "inputs": {
                    "noise": ["18", 0],
                    "guider": ["17", 0],
                    "sampler": ["24", 0],
                    "sigmas": ["16", 0],
                    "latent_image": ["13", 0],
                    "temporal_tile_size": 101,
                    "temporal_overlap": 49,
                    "temporal_overlap_strength": 0.99,
                    "horizontal_tiles": 1,
                    "vertical_tiles": 1,
                    "spatial_overlap": 0,
                    "adain_factor": 0.0,
                },
            },
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["19", 1], run_id=run_id, mode=args.mode, decode_id="20", save_id="21"
        )
    )
    if args.save_draft:
        prompt.update(
            _save_nodes(
                sampled=["12", 1],
                run_id=run_id,
                mode=f"{args.mode}_draft_first_pass",
                decode_id="22",
                save_id="23",
            )
        )
    return prompt, {}


def _chunked_full_frame_euler_control_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Match O1 exactly but use ComfyUI's direct full-frame sampler.

    This validation-only control keeps the complete first pass, learned high-resolution
    latent, conditioning, plain Euler sampler, sigma schedule and seed identical to O1.
    Replacing only the upstream wrapper isolates wrapper behavior without adding a mask,
    a spatial tile, a temporal window or a T8 sampling algorithm.
    """

    prompt, reports = _chunked_upstream_exact_prompt(args, run_id)
    prompt["19"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["18", 0],
            "guider": ["17", 0],
            "sampler": ["24", 0],
            "sigmas": ["16", 0],
            "latent_image": ["13", 0],
        },
    }
    return prompt, reports


def _chunked_upstream_example_prompt(
    args: argparse.Namespace, run_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the author's checked-in 3x3 example from the same masked first pass.

    The upstream node forwards one full noise mask to every spatial tile instead of
    slicing it.  A semantic subject mask would therefore be resized independently in
    each tile and cease to describe the same image coordinates.  Keep the first-pass
    subject mask identical to the T8 control, then replace only the second-pass video
    mask with an explicit all-one mask.  This preserves the exact source trajectory
    while testing the author's 3x3/24-latent-overlap behavior without pretending that
    its tiled mask handling is equivalent to T8's coordinate-aware mask slicing.
    """

    prompt, reports = _chunked_upstream_exact_prompt(args, run_id)
    prompt["9"]["inputs"]["av_latent"] = ["29", 0]
    prompt["12"]["inputs"]["latent_image"] = ["29", 0]
    prompt["15"]["inputs"]["av_latent"] = ["34", 0]
    prompt["19"]["inputs"].update(
        {
            "sampler": ["35", 0],
            "latent_image": ["34", 0],
            "temporal_tile_size": 101,
            "temporal_overlap": 49,
            "temporal_overlap_strength": 0.99,
            "horizontal_tiles": 3,
            "vertical_tiles": 3,
            "spatial_overlap": 24,
            "adain_factor": 0.0,
        }
    )
    prompt.update(
        {
            "24": {
                "class_type": "RepeatImageBatch",
                "inputs": {"image": ["30", 0], "amount": args.frame_count},
            },
            "25": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["24", 0], "vae": ["1", 0]},
            },
            "26": {
                "class_type": "LoadImageMask",
                "inputs": {"image": _video_mask_name(args), "channel": "red"},
            },
            "27": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["25", 0], "mask": ["26", 0]},
            },
            "28": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["8", 1]},
            },
            "29": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["27", 0],
                    "audio_latent": ["28", 1],
                },
            },
            "30": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["5", 0],
                    "upscale_method": "lanczos",
                    "width": args.width,
                    "height": args.height,
                    "crop": "center",
                },
            },
            "31": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["13", 0]},
            },
            "32": {
                "class_type": "SolidMask",
                "inputs": {
                    "value": 1.0,
                    "width": args.target_width,
                    "height": args.target_height,
                },
            },
            "33": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["31", 0], "mask": ["32", 0]},
            },
            "34": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["33", 0],
                    "audio_latent": ["31", 1],
                },
            },
            "35": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
            },
        }
    )
    return prompt, reports


def _phase_text(phase: Mapping[str, Any], node_id: str) -> str:
    return pdd._phase_text(phase, node_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "fast_h3",
            "fast_h3_vsa",
            "timed_reference",
            "chunked_two_pass",
            "chunked_two_pass_global_noise",
            "chunked_two_pass_low_sigma",
            "chunked_two_pass_masked_low_sigma",
            "chunked_two_pass_upscale_only_control",
            "chunked_two_pass_full_frame_euler_control",
            "chunked_two_pass_upstream_exact",
            "chunked_two_pass_upstream_example",
        ),
        required=True,
    )
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--frame-count", type=int, default=124)
    parser.add_argument("--target-width", type=int, default=1152)
    parser.add_argument("--target-height", type=int, default=640)
    parser.add_argument("--temporal-chunk-frames", type=int, default=68)
    parser.add_argument("--temporal-overlap-frames", type=int, default=17)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--tile-height", type=int, default=512)
    parser.add_argument("--spatial-overlap", type=int, default=128)
    parser.add_argument("--spatial-fade", type=int, default=32)
    parser.add_argument(
        "--spatial-strategy",
        choices=("full_frame_safe", "independent_tiles_exp"),
        default="full_frame_safe",
    )
    parser.add_argument(
        "--temporal-strategy",
        choices=("full_clip_safe", "guarded_overlap_exp"),
        default="full_clip_safe",
    )
    parser.add_argument(
        "--save-draft",
        action="store_true",
        help="also decode and save the shared LOW-4 first-pass latent",
    )
    parser.add_argument(
        "--video-mask",
        type=Path,
        default=Path(STATIC_BACKGROUND_MASK),
        help=(
            "ComfyUI input-relative path (or an absolute path inside ComfyUI/input) "
            "for masked low-Sigma and upscale-only controls"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=11000)
    parser.add_argument("--min-free-ram-gib", type=float, default=50.0)
    parser.add_argument("--reserve-vram-gib", type=float, default=2.0)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comfy-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/community-update-real-validation-20260829"),
    )
    parser.add_argument(
        "--timed-source-video",
        type=Path,
        default=Path(
            "artifacts/pdd-ref2va-0p7mp-validation-20260827/"
            "20260827-023133-ref2va/output/MiniMaxH3_PDD_Validation/"
            "20260827-023133_ref2va_1152x640_124f_00001-audio.mp4"
        ),
    )
    return parser


def _required(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    paths = {
        "python": args.python,
        "comfy_main": args.comfy_root / "main.py",
        "project": args.project_root,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "clip": models / "text_encoders" / CLIP,
        "video_vae": models / "vae" / VIDEO_VAE,
        "audio_vae": models / "vae" / AUDIO_VAE,
        "reference": args.comfy_root / "input" / REFERENCE_IMAGE,
    }
    if args.mode in {"fast_h3", "fast_h3_vsa"}:
        paths.update(
            {
                "base": models / "diffusion_models" / FL_BASE,
                "fast_lora": models
                / "loras"
                / (FAST_VSA_LORA if args.mode == "fast_h3_vsa" else FAST_LORA),
            }
        )
    elif args.mode == "timed_reference":
        paths.update(
            {
                "base": models / "diffusion_models" / REF_BASE,
                "pdd": models / "loras" / PDD_REF,
                "source_video": args.timed_source_video,
            }
        )
    elif args.mode in {
        "chunked_two_pass_low_sigma",
        "chunked_two_pass_masked_low_sigma",
        "chunked_two_pass_upscale_only_control",
        "chunked_two_pass_full_frame_euler_control",
        "chunked_two_pass_upstream_exact",
        "chunked_two_pass_upstream_example",
    }:
        paths.update(
            {
                "base": models / "diffusion_models" / FL_BASE,
                "turbo": models / "loras" / TURBO_ALPHA8,
                "upscaler": models / "latent_upscale_models" / UPSCALER,
            }
        )
        if args.mode in {
            "chunked_two_pass_upstream_exact",
            "chunked_two_pass_upstream_example",
        }:
            paths["upstream_external_node"] = (
                args.comfy_root
                / "custom_nodes"
                / "H3TiledLoopSpaceTime_Audit"
                / "H3Loopsampler.py"
            )
        if args.mode in {
            "chunked_two_pass_masked_low_sigma",
            "chunked_two_pass_upscale_only_control",
            "chunked_two_pass_upstream_example",
        }:
            paths["video_mask"] = _video_mask_path(args)
    else:
        paths.update(
            {
                "base": models / "diffusion_models" / FL_BASE,
                "pdd": models / "loras" / PDD_FL,
                "upscaler": models / "latent_upscale_models" / UPSCALER,
            }
        )
    return paths


def _wait_for_mux_finalize(
    output_dir: Path,
    pattern: str,
    ffmpeg: str,
    *,
    timeout_seconds: float = 90.0,
) -> Path:
    """Keep the isolated server alive until VHS finishes the final AV mux.

    VideoHelperSuite can emit ComfyUI's ``execution_success`` before its final
    ``*-audio.mp4`` child process has flushed every packet.  Stopping the
    isolated server at that point leaves a valid video-only MP4 beside a
    truncated AV MP4.  A strict full decode is the completion signal here; file
    existence or a stable size alone is insufficient.
    """

    deadline = time.monotonic() + timeout_seconds
    last_error = "output did not appear"
    while time.monotonic() < deadline:
        candidates = sorted(output_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
        if candidates:
            candidate = candidates[-1]
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(candidate),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if completed.returncode == 0:
                return candidate
            last_error = (completed.stderr or completed.stdout or "decode failed").strip()
        time.sleep(1.0)
    raise RuntimeError(
        "final AV mux did not become strictly decodable before isolated-server shutdown: "
        + last_error[-1000:]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root if args.artifact_root.is_absolute() else args.project_root / args.artifact_root
    ).resolve()
    args.timed_source_video = (
        args.timed_source_video
        if args.timed_source_video.is_absolute()
        else args.project_root / args.timed_source_video
    ).resolve()
    args.lowvram = True
    args.extra_whitelist_custom_nodes = (
        ["H3TiledLoopSpaceTime_Audit"]
        if args.mode in {
            "chunked_two_pass_upstream_exact",
            "chunked_two_pass_upstream_example",
        }
        else []
    )
    paths = _required(args)
    gpu = shared.gpu_memory_mib()
    free_ram_gib = psutil.virtual_memory().available / 1024**3
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    checks = {
        "required_paths_present": all(path.exists() for path in paths.values()),
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "user_port_8188_free": not shared.port_is_listening(args.host, 8188),
        "isolated_port_free": not shared.port_is_listening(args.host, args.port),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(gpu.get("available") and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib),
        "free_ram_gate": free_ram_gib >= args.min_free_ram_gib,
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "mode": args.mode,
        "checks": checks,
        "missing_paths": [str(path) for path in paths.values() if not path.exists()],
        "gpu": gpu,
        "free_ram_gib": round(free_ram_gib, 3),
        "ready": all(checks.values()),
        "policy": "one real render, serial, isolated ComfyUI, lowvram, no pressure loop",
    }
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not preflight["ready"]:
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / f"{run_id}-{args.mode}"
    run_root.mkdir(parents=True, exist_ok=False)
    builders = {
        "fast_h3": _fast_prompt,
        "fast_h3_vsa": _fast_prompt,
        "timed_reference": _timed_prompt,
        "chunked_two_pass": _chunked_prompt,
        "chunked_two_pass_global_noise": _chunked_prompt,
        "chunked_two_pass_low_sigma": _chunked_low_sigma_prompt,
        "chunked_two_pass_masked_low_sigma": _chunked_masked_low_sigma_prompt,
        "chunked_two_pass_upscale_only_control": _chunked_upscale_only_prompt,
        "chunked_two_pass_full_frame_euler_control": (
            _chunked_full_frame_euler_control_prompt
        ),
        "chunked_two_pass_upstream_exact": _chunked_upstream_exact_prompt,
        "chunked_two_pass_upstream_example": _chunked_upstream_example_prompt,
    }
    prompt, report_nodes = builders[args.mode](args, run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "mode": args.mode,
        "run_id": run_id,
        "preflight": preflight,
        "contract": {
            "real_model_inference": True,
            "serial_single_render": True,
            "lowvram": True,
            "width": (
                args.target_width if args.mode.startswith("chunked_two_pass") else args.width
            ),
            "height": (
                args.target_height if args.mode.startswith("chunked_two_pass") else args.height
            ),
            "frame_count": args.frame_count,
        },
    }
    phase = None
    video: Path | None = None
    draft_video: Path | None = None
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    try:
        with shared.IsolatedServer(args, run_root, f"community_{args.mode}"):
            monitor.start()
            phase = asyncio.run(
                pdd._submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            if phase and phase.get("terminal", {}).get("type") == "execution_success":
                output_dir = run_root / "output" / "MiniMaxH3_Community_Real"
                video = _wait_for_mux_finalize(
                    output_dir,
                    f"{run_id}_{args.mode}_[0-9]*-audio.mp4",
                    str(ffmpeg),
                )
                if args.save_draft and args.mode.startswith("chunked_two_pass"):
                    draft_video = _wait_for_mux_finalize(
                        output_dir,
                        f"{run_id}_{args.mode}_draft_*audio.mp4",
                        str(ffmpeg),
                    )
    finally:
        report["gpu_monitor"] = monitor.stop()
    report["phase"] = phase
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1

    node_reports: dict[str, Any] = {}
    for name, node_id in report_nodes.items():
        text = _phase_text(phase, node_id)
        try:
            node_reports[name] = json.loads(text)
        except json.JSONDecodeError:
            node_reports[name] = text
    output_dir = run_root / "output" / "MiniMaxH3_Community_Real"
    if video is None:
        video = shared._latest_file(
            output_dir, f"{run_id}_{args.mode}_[0-9]*-audio.mp4"
        )
    if args.save_draft and args.mode.startswith("chunked_two_pass") and draft_video is None:
        draft_video = shared._latest_file(
            output_dir, f"{run_id}_{args.mode}_draft_*audio.mp4"
        )
    media = shared.media_report(video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    audio = pdd._audio_numeric(video, str(ffmpeg))
    expected_width = (
        args.target_width if args.mode.startswith("chunked_two_pass") else args.width
    )
    expected_height = (
        args.target_height if args.mode.startswith("chunked_two_pass") else args.height
    )
    expected_frames = args.frame_count
    media_checks = pdd._media_checks(
        media,
        audio,
        width=expected_width,
        height=expected_height,
        frame_count=expected_frames,
    )
    draft_report = None
    if args.save_draft and args.mode.startswith("chunked_two_pass"):
        draft_media = shared.media_report(
            draft_video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe)
        )
        draft_audio = pdd._audio_numeric(draft_video, str(ffmpeg))
        draft_checks = pdd._media_checks(
            draft_media,
            draft_audio,
            width=args.width,
            height=args.height,
            frame_count=expected_frames,
        )
        draft_stage = (
            "complete_first_pass"
            if args.mode in {
                "chunked_two_pass_low_sigma",
                "chunked_two_pass_masked_low_sigma",
                "chunked_two_pass_upscale_only_control",
                "chunked_two_pass_full_frame_euler_control",
                "chunked_two_pass_upstream_exact",
            }
            else "low4_partial_pass"
        )
        draft_contact = run_root / f"draft_{draft_stage}_contact_0s_to_end.png"
        pdd._contact_sheet(
            draft_video,
            draft_contact,
            str(ffmpeg),
            width=args.width,
            height=args.height,
            frame_count=expected_frames,
        )
        draft_report = {
            "stage": draft_stage,
            "media": draft_media,
            "audio_numeric": draft_audio,
            "media_checks": draft_checks,
            "output_video": str(draft_video.resolve()),
            "contact_sheet": str(draft_contact.resolve()),
        }
    if args.mode in {"fast_h3", "fast_h3_vsa"}:
        lora = node_reports["lora"]
        fast = node_reports["fast"]
        expected_attention = (
            "comfy_kitchen_vsa_h3_90pct_tile64"
            if args.mode == "fast_h3_vsa"
            else "dense_comfyui"
        )
        feature_checks = {
            "lora_applied": lora.get("status") == "applied" and int(lora.get("applied_patch_count") or 0) > 0,
            "direct_h3_aliases_used": int(lora.get("added_h3_direct_alias_count") or 0) > 0,
            "no_hash_gate": "sha256" not in json.dumps(lora).lower()
            and lora.get("file", {}).get("identity_policy")
            == "display_only_not_a_load_gate_no_hash_scan",
            "fast_h3_four_nfe": fast.get("trained_contract", {}).get("steps_nfe") == 4,
            "attention_profile_effective": fast.get("attention_profile_effective")
            == expected_attention,
        }
        if args.mode == "fast_h3_vsa":
            feature_checks.update(
                {
                    "vsa_gate_count": int(
                        lora.get("vsa_compression_gates", {}).get(
                            "attached_gate_count", 0
                        )
                    )
                    == 50,
                    "vsa_runtime_configured": fast.get("vsa_runtime", {}).get(
                        "status"
                    )
                    == "configured",
                }
            )
    elif args.mode == "timed_reference":
        activity = [
            event
            for event in phase.get("events", [])
            if event.get("type") in {"executing", "executed"} and event.get("node") is not None
        ]
        visited = {str(event.get("node")) for event in activity}
        first_position: dict[str, int] = {}
        for index, event in enumerate(activity):
            first_position.setdefault(str(event.get("node")), index)
        feature_checks = {
            "timed_image_executed": "7" in visited,
            "timed_video_executed": "8" in visited,
            "conditioning_received_timed_clip": "9" in visited,
            "timed_chain_order": all(node in first_position for node in ("7", "8", "9"))
            and first_position["7"] < first_position["8"] < first_position["9"],
            "conditioned_prompt_retained": all(
                token in str(node_reports["conditioned_prompt"])
                for token in ("#lighting", "#motion", "stable eyes")
            ),
        }
    else:
        execution = node_reports.get("execution", {})
        if args.mode == "chunked_two_pass_upscale_only_control":
            upscale = node_reports.get("upscale", {})
            feature_checks = {
                "complete_first_pass_not_split": prompt["12"]["inputs"].get(
                    "sigmas"
                )
                == ["9", 2]
                and not any(
                    node.get("class_type") == "SplitSigmas"
                    for node in prompt.values()
                ),
                "one_sampler_only": sum(
                    node.get("class_type") == "SamplerCustomAdvanced"
                    for node in prompt.values()
                )
                == 1,
                "no_second_pass_plan_or_executor": not any(
                    node.get("class_type")
                    in {
                        "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced",
                        "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced",
                        "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
                        "H3LoopingSampler",
                    }
                    for node in prompt.values()
                ),
                "learned_upscale_completed": upscale.get("status") == "ok",
                "learned_upscale_audio_preserved": upscale.get("audio_preserved")
                is True,
                "learned_upscale_target_geometry": upscale.get("geometry", {}).get(
                    "output_width"
                )
                == args.target_width
                and upscale.get("geometry", {}).get("output_height")
                == args.target_height,
                "precise_mask_loaded": prompt["26"]["inputs"].get("image")
                == _video_mask_name(args),
            }
        else:
            feature_checks = {
                "audio_exact_tensor_passthrough": bool(
                    execution.get("audio_preserved_by_identity")
                ),
                "no_project_pixel_ceiling": execution.get("pixel_limit_policy")
                == "no_project_pixel_area_limit",
            }
        if args.mode == "chunked_two_pass_upstream_exact":
            feature_checks = {
                "complete_first_pass_not_split": prompt["12"]["inputs"].get(
                    "sigmas"
                )
                == ["9", 2]
                and not any(
                    node.get("class_type") == "SplitSigmas"
                    for node in prompt.values()
                ),
                "upstream_external_node_used": prompt["19"]["class_type"]
                == "H3LoopingSampler",
                "upstream_full_context_control": prompt["19"]["inputs"].get(
                    "horizontal_tiles"
                )
                == 1
                and prompt["19"]["inputs"].get("vertical_tiles") == 1
                and prompt["19"]["inputs"].get("adain_factor") == 0.0
                and prompt["19"]["inputs"].get("temporal_tile_size")
                >= 37,
                "upstream_style_refine_schedule": prompt["16"]["inputs"]
                == {
                    "model": ["15", 0],
                    "scheduler": "simple",
                    "steps": 3,
                    "denoise": 0.30,
                },
                "upstream_checked_in_plain_euler_sampler": prompt["24"]
                == {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                }
                and prompt["19"]["inputs"].get("sampler") == ["24", 0],
                "high_geometry_sampler_bound": prompt["15"]["inputs"].get(
                    "av_latent"
                )
                == ["13", 0],
                "draft_saved_when_requested": (
                    not args.save_draft
                    or (
                        draft_report is not None
                        and draft_report.get("stage") == "complete_first_pass"
                        and all(draft_report["media_checks"].values())
                    )
                ),
            }
        elif args.mode == "chunked_two_pass_full_frame_euler_control":
            feature_checks = {
                "complete_first_pass_not_split": prompt["12"]["inputs"].get(
                    "sigmas"
                )
                == ["9", 2]
                and not any(
                    node.get("class_type") == "SplitSigmas"
                    for node in prompt.values()
                ),
                "direct_full_frame_sampler_used": prompt["19"]["class_type"]
                == "SamplerCustomAdvanced",
                "plain_euler_sampler_matches_o1": prompt["24"]
                == {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                }
                and prompt["19"]["inputs"].get("sampler") == ["24", 0],
                "same_high_resolution_latent_schedule_and_seed": prompt["19"][
                    "inputs"
                ]
                == {
                    "noise": ["18", 0],
                    "guider": ["17", 0],
                    "sampler": ["24", 0],
                    "sigmas": ["16", 0],
                    "latent_image": ["13", 0],
                }
                and prompt["18"]["inputs"].get("noise_seed") == SEED + 3,
                "same_upstream_style_refine_schedule": prompt["16"]["inputs"]
                == {
                    "model": ["15", 0],
                    "scheduler": "simple",
                    "steps": 3,
                    "denoise": 0.30,
                },
                "denoised_output_selected": prompt["20"]["inputs"].get(
                    "av_latent"
                )
                == ["19", 1],
                "no_spatial_or_temporal_wrapper": not any(
                    node.get("class_type") == "H3LoopingSampler"
                    for node in prompt.values()
                ),
                "draft_saved_when_requested": (
                    not args.save_draft
                    or (
                        draft_report is not None
                        and draft_report.get("stage") == "complete_first_pass"
                        and all(draft_report["media_checks"].values())
                    )
                ),
            }
        elif args.mode == "chunked_two_pass_upstream_example":
            upstream = prompt["19"]["inputs"]
            feature_checks = {
                "complete_first_pass_not_split": prompt["12"]["inputs"].get(
                    "sigmas"
                )
                == ["9", 2]
                and not any(
                    node.get("class_type") == "SplitSigmas"
                    for node in prompt.values()
                ),
                "same_precise_masked_first_pass": prompt["26"]["inputs"].get(
                    "image"
                )
                == _video_mask_name(args)
                and prompt["9"]["inputs"].get("av_latent") == ["29", 0]
                and prompt["12"]["inputs"].get("latent_image") == ["29", 0],
                "upstream_external_node_used": prompt["19"]["class_type"]
                == "H3LoopingSampler",
                "upstream_checked_in_example_parameters": all(
                    (
                        upstream.get("temporal_tile_size") == 101,
                        upstream.get("temporal_overlap") == 49,
                        upstream.get("temporal_overlap_strength") == 0.99,
                        upstream.get("horizontal_tiles") == 3,
                        upstream.get("vertical_tiles") == 3,
                        upstream.get("spatial_overlap") == 24,
                        upstream.get("adain_factor") == 0.0,
                    )
                ),
                "upstream_checked_in_plain_euler_sampler": prompt["35"]
                == {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                }
                and upstream.get("sampler") == ["35", 0],
                "second_pass_mask_is_explicit_full_edit": prompt["32"]
                == {
                    "class_type": "SolidMask",
                    "inputs": {
                        "value": 1.0,
                        "width": args.target_width,
                        "height": args.target_height,
                    },
                }
                and prompt["19"]["inputs"].get("latent_image") == ["34", 0],
                "author_denoised_output_selected": prompt["20"]["inputs"].get(
                    "av_latent"
                )
                == ["19", 1],
                "upstream_style_refine_schedule": prompt["16"]["inputs"]
                == {
                    "model": ["15", 0],
                    "scheduler": "simple",
                    "steps": 3,
                    "denoise": 0.30,
                },
                "draft_saved_when_requested": (
                    not args.save_draft
                    or (
                        draft_report is not None
                        and draft_report.get("stage") == "complete_first_pass"
                        and all(draft_report["media_checks"].values())
                    )
                ),
            }
        elif args.mode == "chunked_two_pass_global_noise":
            full_frame = execution.get("spatial_strategy") == "full_frame_safe"
            keyframe_canvas = prompt.get("6", {}).get("inputs", {})
            feature_checks.update(
                {
                    "keyframe_canvas_prealigned_once": keyframe_canvas
                    == {
                        "image": ["5", 0],
                        "upscale_method": "lanczos",
                        "width": args.target_width,
                        "height": args.target_height,
                        "crop": "center",
                    }
                    and prompt["7"]["inputs"].get("first_frame") == ["6", 0]
                    and prompt["7"]["inputs"].get("last_frame") == ["6", 0]
                    and prompt["13"]["inputs"].get("first_frame") == ["6", 0]
                    and prompt["13"]["inputs"].get("last_frame") == ["6", 0],
                    "global_noise_generated_once": execution.get("global_noise", {}).get(
                        "generate_noise_calls"
                    )
                    == 1,
                    "global_noise_policy": execution.get("noise_policy")
                    == "one_full_target_video_noise_then_exact_coordinate_slices",
                    "requested_spatial_strategy": execution.get("spatial_strategy")
                    == args.spatial_strategy,
                    "requested_temporal_strategy": execution.get(
                        "temporal_strategy"
                    )
                    == args.temporal_strategy,
                    "temporal_policy_matches_strategy": execution.get(
                        "temporal_merge_policy"
                    )
                    == (
                        "one_full_clip_no_temporal_stitching"
                        if args.temporal_strategy == "full_clip_safe"
                        else "previous_overlap_guarded_progressive_takeover"
                    ),
                    "spatial_grid_matches_strategy": all(
                        (
                            len(item.get("rows", [])) == 1
                            and len(item.get("cols", [])) == 1
                        )
                        if full_frame
                        else (
                            len(item.get("rows", []))
                            * len(item.get("cols", []))
                            >= 2
                        )
                        for item in execution.get("tiles", [])
                    ),
                    "zero_audio_noise": execution.get("global_noise", {}).get(
                        "audio_noise"
                    )
                    == "zero_per_piece",
                    "draft_saved_when_requested": (
                        not args.save_draft
                        or (
                            draft_report is not None
                            and all(draft_report["media_checks"].values())
                        )
                    ),
                }
            )
        elif args.mode in {
            "chunked_two_pass_low_sigma",
            "chunked_two_pass_masked_low_sigma",
        }:
            keyframe_canvas = prompt.get("6", {}).get("inputs", {})
            full_frame = execution.get("spatial_strategy") == "full_frame_safe"
            feature_checks.update(
                {
                    "keyframe_canvas_prealigned_once": keyframe_canvas
                    == {
                        "image": ["5", 0],
                        "upscale_method": "lanczos",
                        "width": args.target_width,
                        "height": args.target_height,
                        "crop": "center",
                    }
                    and prompt["8"]["inputs"].get("first_frame") == ["6", 0]
                    and prompt["8"]["inputs"].get("last_frame") == ["6", 0]
                    and prompt["13"]["inputs"].get("first_frame") == ["6", 0]
                    and prompt["13"]["inputs"].get("last_frame") == ["6", 0],
                    "complete_first_pass_not_split": prompt["12"]["inputs"].get(
                        "sigmas"
                    )
                    == ["9", 2]
                    and not any(
                        node.get("class_type") == "SplitSigmas"
                        for node in prompt.values()
                    ),
                    "upstream_style_refine_schedule": prompt["15"]["class_type"]
                    == "BasicScheduler"
                    and prompt["15"]["inputs"]
                    == {
                        "model": ["9", 0],
                        "scheduler": "simple",
                        "steps": 3,
                        "denoise": 0.30,
                    }
                    and execution.get("refine_nfe") == 3,
                    "first_pass_contract_reported": execution.get(
                        "first_pass_contract"
                    )
                    == "complete_trajectory_to_zero_before_upscale",
                    "joint_av_context_then_original_audio": execution.get(
                        "second_pass_audio_policy"
                    )
                    == "joint_av_preserve_input"
                    and execution.get("global_noise", {}).get("audio_noise")
                    == "generated_once_full_timeline",
                    "global_noise_generated_once": execution.get(
                        "global_noise", {}
                    ).get("generate_noise_calls")
                    == 1,
                    "one_full_context_trajectory": full_frame
                    and execution.get("temporal_strategy") == "full_clip_safe"
                    and int(execution.get("segment_count") or 0) == 1
                    and all(
                        len(item.get("rows", [])) == 1
                        and len(item.get("cols", [])) == 1
                        for item in execution.get("tiles", [])
                    ),
                    "draft_saved_when_requested": (
                        not args.save_draft
                        or (
                            draft_report is not None
                            and draft_report.get("stage") == "complete_first_pass"
                            and all(draft_report["media_checks"].values())
                        )
                    ),
                }
            )
            if args.mode == "chunked_two_pass_masked_low_sigma":
                inherited = execution.get("inherited_video_mask") or {}
                feature_checks.update(
                    {
                        "first_pass_uses_masked_video_latent": prompt["9"][
                            "inputs"
                        ].get("av_latent")
                        == ["29", 0]
                        and prompt["12"]["inputs"].get("latent_image")
                        == ["29", 0],
                        "mask_is_inherited_by_second_pass": inherited.get("status")
                        == "normalized"
                        and inherited.get("policy") == "inherit_required"
                        and all(
                            item.get("inherited_video_mask_applied") is True
                            for item in execution.get("tiles", [])
                        ),
                        "mask_resize_is_spatial_only": inherited.get(
                            "target_spatial_policy"
                        )
                        == "nearest_exact_spatial_only"
                        and inherited.get("temporal_policy")
                        in {
                            "exact_latent_tokens",
                            "single_frame_expanded",
                            "verified_static_then_expanded",
                        },
                    }
                )
        elif args.mode == "chunked_two_pass":
            feature_checks.update(
                {
                    "multiple_temporal_segments": int(execution.get("segment_count") or 0)
                    >= 2,
                    "full_frame_spatial_context": execution.get("spatial_strategy")
                    == "full_frame_safe"
                    and all(
                        len(item.get("rows", [])) == 1
                        and len(item.get("cols", [])) == 1
                        for item in execution.get("tiles", [])
                    ),
                }
            )
    passed = all(media_checks.values()) and all(feature_checks.values())
    contact = run_root / "contact_0s_to_end.png"
    pdd._contact_sheet(
        video,
        contact,
        str(ffmpeg),
        width=expected_width,
        height=expected_height,
        frame_count=expected_frames,
    )
    report.update(
        {
            "node_reports": node_reports,
            "feature_checks": feature_checks,
            "media": media,
            "audio_numeric": audio,
            "media_checks": media_checks,
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
            (
                "draft_first_pass"
                if args.mode in {
                    "chunked_two_pass_low_sigma",
                    "chunked_two_pass_masked_low_sigma",
                    "chunked_two_pass_upscale_only_control",
                    "chunked_two_pass_full_frame_euler_control",
                    "chunked_two_pass_upstream_exact",
                }
                else "draft_low4"
            ): draft_report,
            "status": (
                "MECHANICAL_RENDER_PASS_REVIEW_REQUIRED"
                if passed
                else "FAIL_REAL_VALIDATION"
            ),
            "quality_claim": (
                "Mechanical execution and strict media validation only. Face quality is not passed "
                "until the full video and face crops receive explicit human review."
            ),
            "perceptual_review": {
                "status": "PENDING_HUMAN_REVIEW" if passed else "NOT_ELIGIBLE",
                "required_checks": [
                    "face_geometry_stable_all_frames",
                    "eyes_and_mouth_not_deformed",
                    "identity_consistent",
                    "no_temporal_face_flicker",
                ],
            },
        }
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
