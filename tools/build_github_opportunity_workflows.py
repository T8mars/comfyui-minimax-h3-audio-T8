#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import uuid

try:
    from .api_to_frontend_workflow import _get_json, convert
except ImportError:
    from api_to_frontend_workflow import _get_json, convert


ROOT = Path(__file__).resolve().parents[1]
SERVER = "http://127.0.0.1:8188"


def _node(class_type: str, title: str, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _add_note(workflow: dict, title: str, text: str, pos: tuple[int, int], color: str) -> None:
    node_id = max((int(node["id"]) for node in workflow["nodes"]), default=0) + 1
    workflow["nodes"].append(
        {
            "id": node_id,
            "type": "MarkdownNote",
            "title": title,
            "pos": list(pos),
            "size": [560, 270],
            "flags": {},
            "order": len(workflow["nodes"]),
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"Node name for S&R": "MarkdownNote"},
            "widgets_values": [text],
            "color": color,
            "bgcolor": "#111827",
        }
    )
    workflow["last_node_id"] = node_id


def _write(path: Path, workflow: dict) -> None:
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, path.relative_to(ROOT).as_posix()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_prompt_rewriter(object_info: dict) -> dict:
    prompt = {
        "rewriter": _node(
            "MiniMaxH3PromptRewriter8BT8Advanced",
            "MiniMax H3 8B 提示词重写（生成后卸载）",
            prompt="夜晚的海边，一名女子迎着海风慢慢转身，远处浪声清晰。",
            task="T2VA — 文生音视频",
            resolution="16:9",
            duration=10,
            base_model_path="Qwen3-VL-8B-Instruct",
            adapter_path="MiniMax-H3-Prompt-Rewriter-LoRA-8B",
            load_policy="auto_cpu_offload",
            dtype="bfloat16",
            decoding="greedy",
            seed=42,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            min_image_pixels=65536,
            max_image_pixels=1048576,
            unload_after_generate=True,
            free_comfy_models_before_load=True,
            allow_hub_download=False,
        ),
        "prompt_preview": _node("PreviewAny", "复制此完整提示词到 H3 Conditioning", source=["rewriter", 0]),
        "visual_preview": _node("PreviewAny", "画面描述", source=["rewriter", 1]),
        "sound_preview": _node("PreviewAny", "环境声描述", source=["rewriter", 2]),
        "music_preview": _node("PreviewAny", "非叙事音乐描述", source=["rewriter", 3]),
        "report_preview": _node("PreviewAny", "加载、耗时与卸载报告", source=["rewriter", 4]),
    }
    workflow = convert(prompt, object_info, "MiniMax H3 Prompt Rewriter 8B Advanced")
    _add_note(
        workflow,
        "NOTE 1 · 模型位置与规模",
        "## 模型位置\n- 基座：`ComfyUI/models/text_encoders/Qwen3-VL-8B-Instruct`\n- LoRA：`ComfyUI/models/loras/MiniMax-H3-Prompt-Rewriter-LoRA-8B`\n本机已按固定 revision 下载。它是独立的 8B Qwen3-VL 模型，不能把现有 32B H3 CLIP 当成该 LoRA 的基座。",
        (-80, -420),
        "#244a73",
    )
    _add_note(
        workflow,
        "NOTE 2 · 16GB 使用边界",
        "## 16GB 可运行，但并不快\n默认 `auto_cpu_offload + bfloat16`。本机真实加载成功且没有 OOM；256 token 上限的一次短测试约 482 秒。`max_new_tokens=1024` 能减少截断风险，但耗时可能更长。不要把它描述成轻量或实时节点。",
        (520, -420),
        "#733f24",
    )
    _add_note(
        workflow,
        "NOTE 3 · 输出与卸载",
        "## 推荐用法\n复制第一个输出到现有 H3 Conditioning 的 prompt。默认生成后在 `finally` 释放模型、处理器和 Accelerate hooks，并清理缓存；异常/OOM 路径也执行。当前上游只覆盖 T2VA、I2VA、L2VA、FL2VA，不支持 Ref2VA 提示词重写。",
        (1120, -420),
        "#24573a",
    )
    return workflow


def build_lanpaint(object_info: dict) -> dict:
    prompt = {
        "video": _node("LoadVideo", "替换为需要局部修复的源视频", file="replace_with_source_video.mp4"),
        "components": _node("GetVideoComponents", "读取原画面和原音频", video=["video", 0]),
        "window": _node(
            "MiniMaxH3SourceMediaWindowT8",
            "截取一个 124 帧 / 24fps 修复窗口",
            frames=["components", 0],
            source_fps=24.0,
            width=736,
            height=416,
            length=124,
            start_seconds=0.0,
            short_video_policy="strict",
            short_audio_policy="pad_silence",
            source_audio=["components", 1],
        ),
        "mask": _node("LoadImage", "白色=重绘；黑色=保持源画面", image="replace_with_binary_repair_mask.png"),
        "video_vae": _node("VAELoader", "MiniMax H3 视频 VAE", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "audio_vae": _node("VAELoader", "MiniMax H3 音频 VAE", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "model": _node(
            "UNETLoader",
            "Stock20 FL2VA 模型（不要接 Turbo LoRA）",
            unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            weight_dtype="default",
        ),
        "clip": _node(
            "CLIPLoader",
            "MiniMax H3 Qwen3-VL CLIP",
            clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            type="minimax",
            device="default",
        ),
        "conditioning": _node(
            "MiniMaxH3AudioConditioningT8",
            "描述修复区域应出现的画面和声音",
            clip=["clip", 0],
            video_vae=["video_vae", 0],
            audio_vae=["audio_vae", 0],
            prompt="保持人物身份、服装、镜头运动和原场景，只重绘白色蒙版区域；音频仅在声明区间内重绘。",
            width=736,
            height=416,
            length=124,
            task_type="T2VA — 文生音视频",
            audio_mode="native",
            audio_denoise_strength=1.0,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            allow_above_reference_area=False,
        ),
        "prepare": _node(
            "MiniMaxH3LanPaintAVPrepareT8Advanced",
            "生成 H3 嵌套 AV 蒙版 latent",
            frames=["window", 0],
            source_audio=["window", 1],
            video_vae=["video_vae", 0],
            audio_vae=["audio_vae", 0],
            audio_intervals='[{"start":1.0,"end":2.0}]',
            frame_policy="strict",
            require_lanpaint_sampler=True,
            video_mask=["mask", 1],
        ),
        "schedule": _node(
            "MiniMaxH3DualClockSamplerT8",
            "Stock20 双时钟（视频12 / 音频3）",
            model=["model", 0],
            av_latent=["prepare", 0],
            steps=20,
            shift_video=12.0,
            shift_audio=3.0,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
        ),
        "guider": _node("BasicGuider", "基础引导器", model=["schedule", 0], conditioning=["conditioning", 0]),
        "noise": _node("RandomNoise", "固定种子", noise_seed=2608221001),
        "lanpaint": _node(
            "LanPaint_SamplerCustomAdvanced",
            "LanPaint 外部采样器",
            noise=["noise", 0],
            guider=["guider", 0],
            sampler=["schedule", 1],
            sigmas=["schedule", 2],
            latent_image=["prepare", 0],
            LanPaint_NumSteps=5,
            LanPaint_Lambda=5.0,
            LanPaint_StepSize=0.2,
            LanPaint_PromptMode="Image First",
            LanPaint_Info="MiniMax H3 AV local repair",
        ),
        "decode": _node(
            "MiniMaxH3AVDecodeT8",
            "解码修复候选",
            av_latent=["lanpaint", 1],
            video_vae=["video_vae", 0],
            audio_vae=["audio_vae", 0],
        ),
        "composite": _node(
            "MiniMaxH3LanPaintAVCompositeT8Advanced",
            "只回贴声明的画面和音频区域",
            source_frames=["prepare", 1],
            repaired_frames=["decode", 0],
            source_audio=["prepare", 2],
            repaired_audio=["decode", 1],
            audio_intervals=["prepare", 3],
            video_blend_pixels=11,
            audio_crossfade_seconds=0.02,
            video_mask=["mask", 1],
        ),
        "create": _node("CreateVideo", "合并修复后音视频", images=["composite", 0], fps=24.0, audio=["composite", 1], bit_depth=8),
        "save": _node("SaveVideo", "保存局部修复结果", video=["create", 0], filename_prefix="MiniMaxH3/lanpaint_av_local_repair", format="mp4", codec="h264"),
        "prepare_report": _node("PreviewAny", "检查蒙版/音频区间/latent报告", source=["prepare", 3]),
        "composite_report": _node("PreviewAny", "检查回贴报告", source=["composite", 2]),
    }
    workflow = convert(prompt, object_info, "MiniMax H3 LanPaint AV Local Repair Advanced")
    _add_note(
        workflow,
        "NOTE 1 · 外部依赖",
        "## 需要 LanPaint 采样器\n单独安装 `scraed/LanPaint`，本机验证 revision：`32cf848e93971da380d868936e007f5611218bee`。本套件只构造 H3 的嵌套 AV 蒙版和安全回贴，不复制外部实现。缺少 `LanPaint_SamplerCustomAdvanced` 时 Prepare 会明确报错。",
        (-80, -500),
        "#244a73",
    )
    _add_note(
        workflow,
        "NOTE 2 · 蒙版与音频区间",
        "## 两类区域必须分别声明\n图片 MASK：白色重绘，黑色保持源画面。`audio_intervals` 使用秒，例如 `[{\"start\":1.0,\"end\":2.0}]`。Prepare 的 report 直接连到 Composite，保证两边使用同一份区间，不会误覆盖其余背景声。",
        (520, -500),
        "#24573a",
    )
    _add_note(
        workflow,
        "NOTE 3 · 推荐参数与边界",
        "## 推荐起点\nStock20、20步、视频 shift=12、音频 shift=3；LanPaint 先用 5 / 5.0 / 0.2 / Image First。输入严格按 17n+5 帧并对齐32倍数。该工作流是局部编辑路径，不是低显存保证；先用短窗口验证，不要直接跑长片。",
        (1120, -500),
        "#733f24",
    )
    return workflow


def build_blockswap(object_info: dict) -> dict:
    prompt = {
        "attention": _node("MiniMaxH3AttentionConfig", "RTX 50 系优先 PyTorch SDPA", backend="sdpa_flash", force_backend=False),
        "model": _node(
            "MiniMaxH3Loader",
            "外部流式运行时加载 pruned INT8",
            model_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            attn_backend=["attention", 0],
        ),
        "text_encoder": _node(
            "MiniMaxH3EncoderLoader",
            "外部运行时加载 H3 文本编码器",
            model_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            use_final_norm=False,
            group_size=4,
            pin_memory=True,
            disk_workers=2,
        ),
        "vae": _node(
            "MiniMaxH3VAELoader",
            "外部运行时加载视频/音频 VAE",
            vae_name="minimax_h3_video_vae_fp16.safetensors",
            audio_vae_name="minimax_h3_audio_vae_fp32.safetensors",
        ),
        "prompt": _node(
            "MiniMaxH3SimplePrompt",
            "Stock20 文生音视频提示词",
            text="夜晚的海边，一名女子迎着海风转身，衣摆自然飘动，浪声和脚步声清晰。",
            mode="T2VA",
            total_duration=5.0,
            ratio="16:9",
            negative_text="",
        ),
        "conditioning": _node(
            "MiniMaxH3Conditioning",
            "736×416 低负载起点",
            text_encoder=["text_encoder", 0],
            width=736,
            height=416,
            av_encoder=["vae", 0],
            prompt=["prompt", 0],
            ref_max=1280,
        ),
        "blockswap": _node(
            "MiniMaxH3ExternalBlockSwapBridgeT8Advanced",
            "外部 BlockSwap · 上游自动显存策略",
            profile="upstream_default_auto",
            block_to_swap=47,
            hot_blocks=0,
            prefetch=True,
            prefetch_count=2,
            pin_memory=True,
            disk_workers=2,
            auto_vram=True,
            vram_reserve_mb=0.0,
            offload_dit=False,
            dtype="bfloat16",
            require_external_runtime=True,
        ),
        "sample": _node(
            "MiniMaxH3KSampler",
            "Stock20 + BlockSwap（无 Turbo LoRA）",
            model=["model", 0],
            positive=["conditioning", 0],
            seed=2608221002,
            steps=20,
            cfg=1.0,
            sampler_name="euler",
            scheduler_name="normal",
            shift_video=12.0,
            shift_audio=3.0,
            denoise=1.0,
            use_adaln_cache=False,
            adaln_prebake_batch=3,
            negative=["conditioning", 1],
            latent=["conditioning", 2],
            block_swap_args=["blockswap", 0],
        ),
        "decode": _node("MiniMaxH3Decode", "外部运行时解码", latent=["sample", 0], av_encoder=["vae", 0]),
        "create": _node("CreateVideo", "合并外部运行时输出", images=["decode", 0], fps=24.0, audio=["decode", 1], bit_depth=8),
        "save": _node("SaveVideo", "保存 BlockSwap 结果", video=["create", 0], filename_prefix="MiniMaxH3/external_blockswap_stock20", format="mp4", codec="h264"),
        "bridge_report": _node("PreviewAny", "BlockSwap 参数报告", source=["blockswap", 1]),
        "sampler_report": _node("PreviewAny", "外部采样统计", source=["sample", 1]),
    }
    workflow = convert(prompt, object_info, "MiniMax H3 External BlockSwap Stock20 Advanced")
    _add_note(
        workflow,
        "NOTE 1 · 独立运行时，不接官方 MODEL",
        "## 类型边界\n本桥输出 `MINIMAX_H3_SWAP`，只连接外部 `MiniMaxH3KSampler`；不能连接 ComfyUI 官方 `MODEL`/SamplerCustom。这样不会改变任何现有官方工作流，也不会让旧工作流参数移位。",
        (-80, -500),
        "#244a73",
    )
    _add_note(
        workflow,
        "NOTE 2 · 外部项目与许可边界",
        "## 需要单独安装\n外部运行时：`xiaolibai-sys/ComfyUI-MiniMaxH3`，本机验证 revision：`099aa38c122cea030ce45a51eb1d83208b16a363`。上游当前未声明源码许可证，因此本仓库不复制或再分发其源码，只提供鸭子类型兼容桥。",
        (520, -500),
        "#733f24",
    )
    _add_note(
        workflow,
        "NOTE 3 · 推荐起点与未承诺项",
        "## 推荐起点\n`upstream_default_auto`：47个交换块、0热块、prefetch=2、auto_vram=true。此例使用 Stock20/20步，不接原始 2688 AdaLN Turbo LoRA；RTX 50系选 `sdpa_flash`。16GB 尚未做压力认证，不等于保证不会 OOM，先从 736×416 / 5秒开始。",
        (1120, -500),
        "#24573a",
    )
    return workflow


def main() -> int:
    object_info = _get_json(f"{SERVER}/object_info")
    required = {
        "MiniMaxH3PromptRewriter8BT8Advanced",
        "MiniMaxH3LanPaintAVPrepareT8Advanced",
        "MiniMaxH3LanPaintAVCompositeT8Advanced",
        "LanPaint_SamplerCustomAdvanced",
        "MiniMaxH3ExternalBlockSwapBridgeT8Advanced",
        "MiniMaxH3KSampler",
    }
    missing = sorted(required - set(object_info))
    if missing:
        raise RuntimeError("ComfyUI is missing required nodes: " + ", ".join(missing))

    targets = [
        (
            ROOT / "examples/workflows/14-prompt-relay/2026-08-22_H3_Prompt_Rewriter_8B_Advanced_EXP.json",
            build_prompt_rewriter(object_info),
        ),
        (
            ROOT / "examples/workflows/03-image-video-edit/2026-08-22_H3_LanPaint_AV_Local_Repair_Advanced_EXP.json",
            build_lanpaint(object_info),
        ),
        (
            ROOT / "examples/workflows/12-system-memory/2026-08-22_H3_External_BlockSwap_Stock20_Advanced_EXP.json",
            build_blockswap(object_info),
        ),
    ]
    for path, workflow in targets:
        _write(path, workflow)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
