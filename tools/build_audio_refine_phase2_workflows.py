#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from api_to_frontend_workflow import _get_json, convert
from build_skin_finish_workflow import _note


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "examples" / "workflows" / "18-audio-refine"
FILENAMES = {
    "same_turbo_stack": (
        "2026-08-26_H3_Audio_Refine_Phase2_Same_Turbo4_Advanced_EXP.json"
    ),
    "base_without_turbo": (
        "2026-08-26_H3_Audio_Refine_Phase2_Base_Refine4_Advanced_EXP.json"
    ),
    "base_ordinary8": (
        "2026-08-26_H3_Audio_Refine_Phase2_Base_Ordinary8_Control_Advanced_EXP.json"
    ),
}
PROMPT = (
    "一位女性面对镜头自然地说：‘你在干嘛呢，我在这里呀，看看效果如何。’ "
    "安静的室内环境，声音清晰，无背景音乐。"
)
BASE_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
CLIP_NAME = "qwen3vl_8b_fp8_scaled.safetensors"
PROJECTION_NAME = "mmh3-8b-ClipProj-v3.1.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"


def _comfy_root() -> Path:
    for candidate in PROJECT_ROOT.parents:
        if candidate.name.lower() == "comfyui":
            return candidate
    raise ValueError("ComfyUI root was not found above the project")


def _common(seed: int) -> dict:
    return {
        "1": {"inputs": {"vae_name": VIDEO_VAE_NAME}, "class_type": "VAELoader"},
        "2": {"inputs": {"vae_name": AUDIO_VAE_NAME}, "class_type": "VAELoader"},
        "3": {
            "inputs": {"clip_name": CLIP_NAME, "type": "boogu", "device": "default"},
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {"clip": ["3", 0], "projection": PROJECTION_NAME},
            "class_type": "ClipProjApply",
        },
        "5": {
            "inputs": {
                "clip": ["4", 0],
                "encoder_family": "8B",
                "encoder_architecture": "qwen3_vl",
                "encoder_quantization": "fp8",
                "load_mode": "stock_pageable",
                "projection_path": PROJECTION_NAME,
                "has_reference_images": False,
                "has_reference_videos": False,
                "enforcement": "block_hard_conflicts",
            },
            "class_type": "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
        },
        "6": {
            "inputs": {"unet_name": BASE_NAME, "weight_dtype": "default"},
            "class_type": "UNETLoader",
        },
        "7": {
            "inputs": {
                "model": ["6", 0],
                "lora_name": LORA_NAME,
                "strength_model": 1.0,
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "8": {
            "inputs": {
                "prompt": PROMPT,
                "width": 1056,
                "height": 608,
                "length": 124,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "clip": ["5", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
        },
    }


def _ordinary8_prompt(seed: int) -> dict:
    prompt = _common(seed)
    prompt.update(
        {
            "9": {
                "inputs": {
                    "model": ["6", 0],
                    "av_latent": ["8", 1],
                    "steps": 8,
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "sampler_name": "dual_clock_euler",
                    "scheduler": "native_flow",
                },
                "class_type": "MiniMaxH3DualClockSamplerT8",
                "_meta": {"title": "Base without Turbo / 普通8 NFE成本对照"},
            },
            "10": {"inputs": {"noise_seed": seed}, "class_type": "RandomNoise"},
            "11": {
                "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
                "class_type": "BasicGuider",
            },
            "12": {
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["11", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["9", 2],
                    "latent_image": ["8", 1],
                },
                "class_type": "SamplerCustomAdvanced",
            },
            "13": {
                "inputs": {
                    "av_latent": ["12", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "14": {
                "inputs": {"images": ["13", 0], "fps": 24.0, "audio": ["13", 1]},
                "class_type": "CreateVideo",
            },
            "15": {
                "inputs": {
                    "video": ["14", 0],
                    "filename_prefix": "MiniMaxH3/AudioRefine/phase2_base_ordinary8",
                    "format": "mp4",
                    "codec": "h264",
                },
                "class_type": "SaveVideo",
            },
        }
    )
    return prompt


def _refine_prompt(seed: int, strategy: str) -> dict:
    prompt = _common(seed)
    refine_model = ["9", 0] if strategy == "same_turbo_stack" else ["6", 0]
    suffix = "same_turbo" if strategy == "same_turbo_stack" else "base"
    prompt.update(
        {
            "9": {
                "inputs": {
                    "model": ["7", 0],
                    "av_latent": ["8", 1],
                    "steps": 4,
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "sampler_name": "dual_clock_euler",
                    "scheduler": "native_flow",
                },
                "class_type": "MiniMaxH3DualClockSamplerT8",
            },
            "10": {"inputs": {"noise_seed": seed}, "class_type": "RandomNoise"},
            "11": {
                "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
                "class_type": "BasicGuider",
            },
            "12": {
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["11", 0],
                    "sampler": ["9", 1],
                    "sigmas": ["9", 2],
                    "latent_image": ["8", 1],
                },
                "class_type": "SamplerCustomAdvanced",
            },
            "13": {
                "inputs": {
                    "av_latent": ["12", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "14": {
                "inputs": {"images": ["13", 0], "fps": 24.0, "audio": ["13", 1]},
                "class_type": "CreateVideo",
            },
            "15": {
                "inputs": {
                    "video": ["14", 0],
                    "filename_prefix": f"MiniMaxH3/AudioRefine/phase2_{suffix}_original",
                    "format": "mp4",
                    "codec": "h264",
                },
                "class_type": "SaveVideo",
            },
            "16": {
                "inputs": {
                    "model": ["9", 0],
                    "positive": ["8", 0],
                    "av_latent": ["12", 0],
                    "conditioned_prompt": ["8", 3],
                    "media_map_json": ["8", 4],
                    "conditioning_report": ["8", 5],
                    "minimum_free_vram_mib": 512,
                    "minimum_commit_headroom_gib": 16.0,
                    "hash_chunk_megabytes": 8,
                },
                "class_type": "MiniMaxH3AudioRefineAuditT8Advanced",
            },
            "17": {
                "inputs": {
                    "audit": ["16", 0],
                    "first_pass_model": ["9", 0],
                    "refine_model": refine_model,
                    "route_strategy": strategy,
                    "declared_first_pass_nfe": 4,
                },
                "class_type": "MiniMaxH3AudioRefineModelRouteT8Advanced",
            },
            "18": {
                "inputs": {
                    "route": ["17", 1],
                    "refine_steps": 4,
                    "audio_denoise": 0.50,
                    "refine_seed": seed,
                },
                "class_type": "MiniMaxH3AudioRefinePhase2PlanT8Advanced",
            },
            "19": {
                "inputs": {
                    "plan": ["18", 0],
                    "refine_model": ["17", 0],
                    "positive": ["8", 0],
                    "av_latent": ["12", 0],
                },
                "class_type": "MiniMaxH3AudioRefineDualModelSetupT8Advanced",
            },
            "20": {
                "inputs": {
                    "noise": ["19", 1],
                    "guider": ["19", 2],
                    "sampler": ["19", 3],
                    "sigmas": ["19", 4],
                    "latent_image": ["19", 5],
                },
                "class_type": "SamplerCustomAdvanced",
            },
            "21": {
                "inputs": {
                    "second_pass_input": ["19", 5],
                    "second_pass_output": ["20", 0],
                    "expected_audio_strength": 1.0,
                    "fail_on_locked_mismatch": False,
                    "locked_atol": 0.0,
                },
                "class_type": "MiniMaxH3TwoPassAudioAuditT8Advanced",
            },
            "22": {
                "inputs": {
                    "av_latent": ["21", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "23": {
                "inputs": {"images": ["22", 0], "fps": 24.0, "audio": ["22", 1]},
                "class_type": "CreateVideo",
            },
            "24": {
                "inputs": {
                    "video": ["23", 0],
                    "filename_prefix": f"MiniMaxH3/AudioRefine/phase2_{suffix}_candidate",
                    "format": "mp4",
                    "codec": "h264",
                },
                "class_type": "SaveVideo",
            },
            "25": {
                "inputs": {
                    "original_av_latent": ["12", 0],
                    "candidate_av_latent": ["21", 0],
                    "original_audio": ["13", 1],
                    "candidate_audio": ["22", 1],
                    "accept_candidate": False,
                    "video_frame_count": 124,
                    "fps": 24.0,
                    "maximum_duration_delta_ms": 50.0,
                    "spectral_drift_threshold": 0.30,
                    "level_delta_threshold_db": 4.0,
                    "persistent_window_count": 3,
                },
                "class_type": "MiniMaxH3AudioRefineQualityGateT8Advanced",
            },
            "26": {
                "inputs": {
                    "av_latent": ["25", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "27": {
                "inputs": {"images": ["26", 0], "fps": 24.0, "audio": ["26", 1]},
                "class_type": "CreateVideo",
            },
            "28": {
                "inputs": {
                    "video": ["27", 0],
                    "filename_prefix": f"MiniMaxH3/AudioRefine/phase2_{suffix}_selected",
                    "format": "mp4",
                    "codec": "h264",
                },
                "class_type": "SaveVideo",
            },
        }
    )
    return prompt


def build_prompts(seed: int = 2608260404) -> dict[str, dict]:
    return {
        "same_turbo_stack": _refine_prompt(seed, "same_turbo_stack"),
        "base_without_turbo": _refine_prompt(seed, "base_without_turbo"),
        "base_ordinary8": _ordinary8_prompt(seed),
    }


def _append_notes(workflow: dict, strategy: str) -> None:
    route_text = {
        "same_turbo_stack": (
            "本文件是四臂中的 Turbo4 + 同 Turbo 栈 Refine4。Route 必须证明两端共享基座、patch UUID、LoRA metadata 和权重补丁结构。"
        ),
        "base_without_turbo": (
            "本文件是四臂中的 Turbo4 + 无 Turbo 基座 Refine4。首遍是 Turbo4；精修端必须是同一基座对象且权重补丁为零。"
        ),
        "base_ordinary8": (
            "本文件是四臂中的无 Turbo 基座普通 8 NFE 成本控制，只跑一次采样，不执行 Audio Refine。"
        ),
    }[strategy]
    notes = [
        (
            "FOUR-ARM SET / 四臂集合",
            "## 三个文件合起来才是四臂公平基线\n\nTurbo4 原始与同栈 Refine4 已有历史质量对；本批三个独立低负载文件分别复现同栈、补 base Refine4、补普通8 NFE。不要在一个队列并发运行。",
        ),
        (
            "THIS ROUTE / 本文件用途",
            f"## 路线说明\n\n{route_text}",
        ),
        (
            "NFE BOUNDARY / 训练分布边界",
            "## 总 NFE 相同不等于训练分布相同\n\n普通8 NFE是计算成本控制；Turbo4+Refine4是两个 partial/full 轨迹的组合。报告必须写真实NFE与模型/LoRA栈，不能把两者称为算法等价。",
        ),
        (
            "PARAMETERS / 参数",
            "## 固定可审计参数\n\n分辨率1056×608、124帧、24fps、相同提示词与seed；Refine固定4 NFE，只允许 audio_denoise 0.35或0.50。1.0属于完全音频重生成，不在这里。",
        ),
        (
            "HUMAN GATE / 人工质量门",
            "## 默认保留原始结果\n\n精修文件的 accept_candidate 默认 false。先听台词、音色、演绎、远近感、音乐/音效和口型；确认后才能打开。接受时只使用候选音频latent，视频严格回填首遍原始latent。",
        ),
        (
            "RESOURCES / 资源与运行",
            "## 串行低负载\n\n每次只运行一个工作流；未知补丁栈或资源遥测不足会旁路。无缓存Refine每步仍是完整联合AV Transformer前向，不承诺所有16GB显卡安全。",
        ),
    ]
    start_id = int(workflow["last_node_id"]) + 1
    for index, (title, text) in enumerate(notes):
        node = _note(
            start_id + index,
            title,
            text,
            (index * 840, 2800),
            (800, 340),
        )
        node["order"] = len(workflow["nodes"])
        workflow["nodes"].append(node)
    workflow["last_node_id"] = start_id + len(notes) - 1


def build_all(server: str, seed: int = 2608260404) -> dict[str, dict]:
    object_info = _get_json(f"{server.rstrip('/')}/object_info")
    workflows = {}
    for strategy, prompt in build_prompts(seed).items():
        missing = sorted({node["class_type"] for node in prompt.values()} - set(object_info))
        if missing:
            raise ValueError(f"server is missing nodes for {strategy}: {missing}")
        title = f"2026-08-26 MiniMax H3 Audio Refine Phase 2 {strategy} Advanced EXP"
        workflow = convert(prompt, object_info, title)
        _append_notes(workflow, strategy)
        workflow["extra"]["workflow_title"] = title
        workflows[strategy] = workflow
    return workflows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--seed", type=int, default=2608260404)
    parser.add_argument("--no-user-copy", action="store_true")
    args = parser.parse_args()
    workflows = build_all(args.server, args.seed)
    user_root = (
        _comfy_root()
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / "18-audio-refine"
    )
    for strategy, workflow in workflows.items():
        payload = json.dumps(workflow, ensure_ascii=False, indent=2)
        output = WORKFLOW_ROOT / FILENAMES[strategy]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(output)
        if not args.no_user_copy:
            user_output = user_root / output.name
            user_output.parent.mkdir(parents=True, exist_ok=True)
            user_output.write_text(payload, encoding="utf-8")
            print(user_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

