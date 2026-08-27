#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from api_to_frontend_workflow import _get_json, convert
from build_skin_finish_workflow import _note


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "workflows"
    / "18-audio-refine"
    / "2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP.json"
)


def _comfy_root() -> Path:
    for candidate in PROJECT_ROOT.parents:
        if candidate.name.lower() == "comfyui":
            return candidate
    raise ValueError("ComfyUI root was not found above the project")


USER_OUTPUT = (
    _comfy_root()
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "18-audio-refine"
    / OUTPUT.name
)


def build_prompt(seed: int = 2608260404) -> dict:
    prompt = (
        "一位女性面对镜头自然地说：‘你在干嘛呢，我在这里呀，看看效果如何。’ "
        "安静的室内环境，声音清晰，无背景音乐。"
    )
    return {
        "1": {"inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}, "class_type": "VAELoader"},
        "2": {"inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}, "class_type": "VAELoader"},
        "3": {
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {"unet_name": "minimax_h3_fl2va_int8_convrot.safetensors", "weight_dtype": "default"},
            "class_type": "UNETLoader",
        },
        "5": {
            "inputs": {
                "model": ["4", 0],
                "lora_name": "minimax_h3_turbo_4步加速ema_comfyui.safetensors",
                "strength_model": 1.0,
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "6": {
            "inputs": {
                "prompt": prompt,
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
                "clip": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
            "_meta": {"title": "1. Native H3 source candidate / 首遍原始候选"},
        },
        "7": {
            "inputs": {
                "model": ["5", 0],
                "av_latent": ["6", 1],
                "steps": 4,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
            "class_type": "MiniMaxH3DualClockSamplerT8",
        },
        "8": {"inputs": {"noise_seed": seed}, "class_type": "RandomNoise"},
        "9": {"inputs": {"model": ["7", 0], "conditioning": ["6", 0]}, "class_type": "BasicGuider"},
        "10": {
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["7", 1],
                "sigmas": ["7", 2],
                "latent_image": ["6", 1],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "11": {
            "inputs": {"av_latent": ["10", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]},
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "12": {
            "inputs": {"images": ["11", 0], "fps": 24.0, "audio": ["11", 1], "bit_depth": 8, "color_space": "sRGB"},
            "class_type": "CreateVideo",
        },
        "13": {
            "inputs": {"video": ["12", 0], "filename_prefix": "MiniMaxH3/AudioRefine/original", "format": "mp4", "codec": "h264"},
            "class_type": "SaveVideo",
            "_meta": {"title": "Save ORIGINAL first-pass candidate / 保存原始候选"},
        },
        "14": {
            "inputs": {
                "model": ["7", 0],
                "positive": ["6", 0],
                "av_latent": ["10", 0],
                "conditioned_prompt": ["6", 3],
                "media_map_json": ["6", 4],
                "conditioning_report": ["6", 5],
                "minimum_free_vram_mib": 512,
                "minimum_commit_headroom_gib": 16.0,
                "hash_chunk_megabytes": 8,
            },
            "class_type": "MiniMaxH3AudioRefineAuditT8Advanced",
        },
        "15": {
            "inputs": {
                "audit": ["14", 0],
                "refine_steps": 4,
                "audio_denoise": 0.5,
                "refine_seed": seed,
                "model_strategy": "connected_model_explicit",
            },
            "class_type": "MiniMaxH3AudioRefinePlanT8Advanced",
        },
        "16": {
            "inputs": {"plan": ["15", 0], "model": ["7", 0], "positive": ["6", 0], "av_latent": ["10", 0]},
            "class_type": "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
        },
        "17": {
            "inputs": {
                "noise": ["16", 1],
                "guider": ["16", 2],
                "sampler": ["16", 3],
                "sigmas": ["16", 4],
                "latent_image": ["16", 5],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "18": {
            "inputs": {
                "second_pass_input": ["16", 5],
                "second_pass_output": ["17", 0],
                "expected_audio_strength": 1.0,
                "fail_on_locked_mismatch": False,
                "locked_atol": 0.0,
            },
            "class_type": "MiniMaxH3TwoPassAudioAuditT8Advanced",
        },
        "19": {
            "inputs": {"av_latent": ["18", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]},
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "20": {
            "inputs": {"images": ["19", 0], "fps": 24.0, "audio": ["19", 1], "bit_depth": 8, "color_space": "sRGB"},
            "class_type": "CreateVideo",
        },
        "21": {
            "inputs": {"video": ["20", 0], "filename_prefix": "MiniMaxH3/AudioRefine/refined_candidate", "format": "mp4", "codec": "h264"},
            "class_type": "SaveVideo",
            "_meta": {"title": "Save RAW REFINE candidate for listening / 保存精修试听候选"},
        },
        "22": {
            "inputs": {
                "audio": ["19", 1], "video_frame_count": 124, "fps": 24.0,
                "opening_window_ms": 40.0, "comparison_window_ms": 250.0,
                "pop_jump_threshold": 0.15, "dc_jump_threshold": 0.02,
                "wrap_correlation_threshold": 0.985, "clipping_ratio_threshold": 0.001,
                "max_av_delta_ms": 50.0,
            },
            "class_type": "MiniMaxH3AudioIntegrityAuditT8Advanced",
        },
        "23": {
            "inputs": {
                "reference_audio": ["11", 1], "candidate_audio": ["19", 1],
                "analysis_window_ms": 500.0, "hop_ms": 100.0,
                "active_rms_floor_dbfs": -50.0, "spectral_drift_threshold": 0.30,
                "level_delta_threshold_db": 4.0, "persistent_window_count": 3,
                "max_duration_delta_ms": 50.0,
            },
            "class_type": "MiniMaxH3AudioPerceptualDriftAuditT8Advanced",
        },
        "24": {
            "inputs": {
                "original_av_latent": ["10", 0], "candidate_av_latent": ["18", 0],
                "original_audio": ["11", 1], "candidate_audio": ["19", 1],
                "accept_candidate": False, "video_frame_count": 124, "fps": 24.0,
                "maximum_duration_delta_ms": 50.0, "spectral_drift_threshold": 0.30,
                "level_delta_threshold_db": 4.0, "persistent_window_count": 3,
            },
            "class_type": "MiniMaxH3AudioRefineQualityGateT8Advanced",
            "_meta": {"title": "HUMAN GATE: false=original, true=reviewed audio / 人工质量门"},
        },
        "25": {
            "inputs": {"av_latent": ["24", 0], "video_vae": ["1", 0], "audio_vae": ["2", 0]},
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "26": {
            "inputs": {"images": ["25", 0], "fps": 24.0, "audio": ["25", 1], "bit_depth": 8, "color_space": "sRGB"},
            "class_type": "CreateVideo",
        },
        "27": {
            "inputs": {"video": ["26", 0], "filename_prefix": "MiniMaxH3/AudioRefine/selected_default_original", "format": "mp4", "codec": "h264"},
            "class_type": "SaveVideo",
            "_meta": {"title": "Save SELECTED result; default is ORIGINAL / 保存最终选择"},
        },
    }


def _append_notes(workflow: dict) -> None:
    notes = [
        (
            "START HERE / 使用顺序",
            "## 先保存并试听两条候选\n\n本工作流固定为 **Turbo4 首遍 + Refine4 尾段，共8次H3联合AV前向**。先比较 `original` 与 `refined_candidate`。质量门 `accept_candidate=false` 时，`selected` 必定回退原始结果；试听确认后才改成 `true`。",
        ),
        (
            "WHAT IT DOES / 它不是普通降噪",
            "## 生成式音频 latent 重采样\n\nRefine使用同一H3 Transformer、同一提示词和视频上下文重新生成音频latent。它可能改善低步数的闷、金属感或细节，也可能改字、增删音节、改变音色、演绎、音乐、音效、声场和口型同步；不能称为无损修复。",
        ),
        (
            "RECOMMENDED DEFAULTS / 建议参数",
            "## 首轮只改少量参数\n\n建议 `refine_steps=4`、`audio_denoise=0.50`、CFG固定1、shift固定视频12/音频3、`dual_clock_euler + native_flow`。`0.35`可作为更保守的后续试听点；`1.0`接近完全重生成，风险显著更高。不要同时叠加未知缓存、STG、EAV、SLA或第三方采样补丁。",
        ),
        (
            "VIDEO LOCK / 为什么必须经过质量门",
            "## 零视频mask不等于最终逐位不变\n\n真实机械验证发现：ComfyUI采样返回的候选视频latent仍可能变化。质量门在接受候选时强制组合 **原始视频latent + 候选音频latent**；硬失败或未接受则完整回退原始结果。不要绕过质量门直接交付候选latent。",
        ),
        (
            "AUDIO LOCK + RESOURCES / 禁用边界",
            "## 锁定音频不得精修\n\n`final_audio`、`lock_source`、受保护外部音轨和未验证的`remix_source`会ABSTAIN。无缓存Refine每一步仍接近完整H3前向，不是只算音频；资源遥测未知、整卡空闲低于512MiB或系统commit余量低于16GiB时节点旁路，不保证所有16GB显卡都安全。",
        ),
        (
            "HUMAN REVIEW / 最终验收",
            "## 代理指标不能替代人耳\n\n重点听中文是否改字/漏字/重复，人物音色与远近是否跳变，音乐/环境/瞬态是否连续，声道是否坍缩，并观看说话起止与口型。Integrity与Drift只给风险提示；它们不能证明候选更好。确认后再打开质量门。",
        ),
    ]
    start_id = int(workflow["last_node_id"]) + 1
    for index, (title, text) in enumerate(notes):
        node = _note(
            start_id + index,
            title,
            text,
            (index * 860, 2900),
            (820, 360),
        )
        node["order"] = len(workflow["nodes"])
        workflow["nodes"].append(node)
    workflow["last_node_id"] = start_id + len(notes) - 1
    workflow["extra"]["workflow_title"] = (
        "2026-08-26 MiniMax H3 Audio Refine Turbo4 + Refine4 Advanced EXP"
    )


def build(server: str) -> dict:
    prompt = build_prompt()
    object_info = _get_json(f"{server.rstrip('/')}/object_info")
    missing = sorted({node["class_type"] for node in prompt.values()} - set(object_info))
    if missing:
        raise ValueError(f"server is missing nodes: {missing}")
    workflow = convert(
        prompt,
        object_info,
        "2026-08-26 MiniMax H3 Audio Refine Turbo4 + Refine4 Advanced EXP",
    )
    _append_notes(workflow)
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--no-user-copy", action="store_true")
    args = parser.parse_args()
    workflow = build(args.server)
    payload = json.dumps(workflow, ensure_ascii=False, indent=2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload, encoding="utf-8")
    if not args.no_user_copy:
        USER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        USER_OUTPUT.write_text(payload, encoding="utf-8")
    print(OUTPUT)
    if not args.no_user_copy:
        print(USER_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
