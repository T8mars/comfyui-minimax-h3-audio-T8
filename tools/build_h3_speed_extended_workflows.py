from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

try:
    from .api_to_frontend_workflow import _get_json, convert
    from .build_h3_speed_multimodal_validation import build_multimodal_speed_prompts
    from .build_h3_speed_reference_validation import build_reference_speed_prompts
    from .build_h3_speed_turbo8_validation import build_turbo8_speed_prompt
except ImportError:
    from api_to_frontend_workflow import _get_json, convert  # type: ignore[no-redef]
    from build_h3_speed_multimodal_validation import (  # type: ignore[no-redef]
        build_multimodal_speed_prompts,
    )
    from build_h3_speed_reference_validation import (  # type: ignore[no-redef]
        build_reference_speed_prompts,
    )
    from build_h3_speed_turbo8_validation import (  # type: ignore[no-redef]
        build_turbo8_speed_prompt,
    )


DATE = "2026-08-19"


def _notes(name: str) -> tuple[str, str, str]:
    common = (
        "## SPEED公共边界\n\n"
        "这是Advanced/EXP整链采样，不会替换原采样节点。默认两级`0.5→1.0`、Stock20时为14+6 NFE；"
        "Turbo8时为6+2 NFE，总NFE不额外增加。每级都会重建H3 AV latent、PackedLayout和条件。"
        "当前仅机械跑通，未证明质量或加速收益。"
    )
    cases = {
        "i2va_lock_source": (
            "## I2VA + lock_source\n\n替换输入视频；首帧作为锚点，原音频锁定并在最终保存时旁路生成音频。"
            "实测首帧decoded correlation 0.9985，锁定音轨经AAC后waveform correlation 0.9831。",
            "## 参数与检查\n\n1024×576、124帧、20步、shift 12/3。检查首帧、124帧、32k双声道和完整原声；"
            "本机最低显存余量约455MiB，低于512MiB门槛。",
        ),
        "fl2va_remix_source": (
            "## FL2VA + remix_source\n\n输入视频首/第124帧分别作为首尾锚点；源音频以0.35强度进入联合AV重混，"
            "最终保存生成音频，不是原声旁路。",
            "## 参数与检查\n\n检查首尾是否命中、转场是否自然、音轨是否新增非要求语音。remix与源波形不应要求高相关；"
            "本机最低显存余量约445MiB。",
        ),
        "l2va_native": (
            "## L2VA + native audio\n\n输入视频第124帧作为尾帧锚点；音频由H3原生生成，不读取源音轨作为drive/ref。",
            "## 参数与检查\n\n检查尾帧、完整运动和生成声场。该代表链峰值约16108MiB，只余约272MiB，"
            "虽跑通但不能称16GB安全。",
        ),
        "ref_video_audio_native": (
            "## Ref2VA：视频与同编号音轨\n\n示例截取2秒/48帧参考视频，并把它的音频连到"
            "`ref_video_audios.ref_video_audio_0`；编号必须与`ref_video_0`一致。",
            "## 成本与检查\n\n固定参考token让本机耗时约417秒，明显抵消SPEED收益。检查动作/画面/声音参考遵循；"
            "机械跑通不代表比全分辨率更快或更好。",
        ),
        "hybrid_first_image_audio": (
            "## Hybrid：首帧 + 独立图 + 独立音频\n\n首帧、Picture 1和Audio 1同时进入条件。"
            "每级报告应为pictures=2, videos=0, audios=1，防止refs覆盖keyframe。",
            "## 参数与检查\n\n替换输入视频和参考图；检查首帧、图像身份/材质、音色/环境声是否串扰。"
            "当前只证明payload与输出机械正确，参考质量仍需人工审片/试听。",
        ),
        "turbo8_t2va": (
            "## Turbo8专用档\n\nMODEL必须由用户明确选择兼容Turbo LoRA后传入。运行时只能证明存在weight patches，"
            "不能从patch tensor反推LoRA文件身份，因此报告不会伪称自动验证了LoRA。",
            "## 成本与风险\n\n8步分成6+2 NFE，本机约150秒；峰值16258MiB，仅余约122MiB并发生换页。"
            "严格检查画面、声音与同步；不要标16GB安全，不要叠BlockCache/STG/ActivationChunk。",
        ),
    }
    case_note, check_note = cases[name]
    return common, case_note, check_note


def add_notes(workflow: dict, name: str) -> dict:
    positions = ((0, -720), (920, -720), (1840, -720))
    titles = ("SPEED机制与发布边界", "本工作流用途", "参数、显存与审片清单")
    colors = (("#2d3f66", "#111827"), ("#3f5d36", "#152115"), ("#6a4425", "#24170f"))
    last_id = int(workflow["last_node_id"])
    for index, (text, position, title, color) in enumerate(
        zip(_notes(name), positions, titles, colors), start=1
    ):
        workflow["nodes"].append(
            {
                "id": last_id + index,
                "type": "MarkdownNote",
                "title": title,
                "pos": list(position),
                "size": [820, 430],
                "flags": {},
                "order": len(workflow["nodes"]),
                "mode": 0,
                "color": color[0],
                "bgcolor": color[1],
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [text],
            }
        )
    workflow["last_node_id"] = last_id + 3
    return workflow


def build_workflows(object_info: dict) -> dict[str, dict]:
    multimodal, _ = build_multimodal_speed_prompts(
        source_video="replace_with_exact_24fps_source.mp4"
    )
    references, _ = build_reference_speed_prompts(
        source_video="replace_with_reference_video.mp4",
        reference_image="replace_with_reference_image.png",
    )
    prompts = {
        "i2va_lock_source": multimodal["i2va_lock_source"],
        "fl2va_remix_source": multimodal["fl2va_remix_source"],
        "l2va_native": multimodal["l2va_native"],
        "ref_video_audio_native": references["ref_video_audio_native"],
        "hybrid_first_image_audio": references["hybrid_first_image_audio"],
        "turbo8_t2va": build_turbo8_speed_prompt(),
    }
    metadata = {
        "i2va_lock_source": ("H3_SPEED_I2VA_Lock_Stock20_Advanced_EXP", "I2VA锁定原声"),
        "fl2va_remix_source": ("H3_SPEED_FL2VA_Remix_Stock20_Advanced_EXP", "FL2VA音频重混"),
        "l2va_native": ("H3_SPEED_L2VA_Native_Stock20_Advanced_EXP", "L2VA原生音频"),
        "ref_video_audio_native": (
            "H3_SPEED_RefVideoAudio_Stock20_Advanced_EXP",
            "Ref2VA视频和同编号音轨",
        ),
        "hybrid_first_image_audio": (
            "H3_SPEED_Hybrid_FirstImageAudio_Stock20_Advanced_EXP",
            "Hybrid首帧图像音频参考",
        ),
        "turbo8_t2va": ("H3_SPEED_T2VA_Turbo8_Advanced_EXP", "T2VA Turbo8"),
    }
    workflows = {}
    for name, prompt in prompts.items():
        stem, title = metadata[name]
        workflow = convert(prompt, object_info, title)
        workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"minimax-h3-t8/{DATE}/{stem}"))
        workflows[f"{DATE}_{stem}.json"] = add_notes(workflow, name)
    return workflows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dated importable H3 SPEED examples.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    args = parser.parse_args()
    object_info = _get_json(f"{args.server.rstrip('/')}/object_info")
    workflows = build_workflows(object_info)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, workflow in workflows.items():
        path = args.output_dir / filename
        path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
