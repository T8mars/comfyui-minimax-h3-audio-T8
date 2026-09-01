from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V2_Ref2VA_8Step_Advanced_EXP.json"
)


def _node(
    node_id,
    node_type,
    pos,
    size,
    order,
    *,
    title="",
    inputs=None,
    outputs=None,
    widgets=None,
    color=None,
    bgcolor=None,
):
    result = {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets or [],
    }
    if title:
        result["title"] = title
    if color:
        result["color"] = color
    if bgcolor:
        result["bgcolor"] = bgcolor
    return result


def build() -> dict:
    nodes = [
        _node(
            1,
            "UNETLoader",
            (0, 0),
            (360, 82),
            0,
            title="1. Load local MiniMax H3 Ref2VA model",
            outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}],
            widgets=["minimax_h3_ref2va_int8_convrot.safetensors", "default"],
        ),
        _node(
            2,
            "LoraLoaderBypassModelOnly",
            (400, 0),
            (430, 105),
            1,
            title="2. Local Turbo4 LoRA · keep 8 sampling steps",
            inputs=[{"name": "model", "type": "MODEL", "link": 1}],
            outputs=[{"name": "MODEL", "type": "MODEL", "links": [2]}],
            widgets=["minimax_h3_turbo_4步加速ema_comfyui.safetensors", 1.0],
        ),
        _node(
            3,
            "CLIPLoader",
            (0, 150),
            (360, 106),
            2,
            title="3. Local H3 Qwen3-VL CLIP",
            outputs=[{"name": "CLIP", "type": "CLIP", "links": [3]}],
            widgets=["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
        ),
        _node(
            4,
            "VAELoader",
            (0, 310),
            (360, 58),
            3,
            title="4. H3 video VAE",
            outputs=[{"name": "VAE", "type": "VAE", "links": [4]}],
            widgets=["minimax_h3_video_vae_fp16.safetensors"],
        ),
        _node(
            5,
            "VAELoader",
            (0, 420),
            (360, 58),
            4,
            title="5. H3 audio VAE",
            outputs=[{"name": "VAE", "type": "VAE", "links": [5]}],
            widgets=["minimax_h3_audio_vae_fp32.safetensors"],
        ),
        _node(
            6,
            "LoadImage",
            (0, 560),
            (360, 310),
            5,
            title="6. Replace with performer reference image",
            outputs=[
                {"name": "IMAGE", "type": "IMAGE", "links": [6]},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            widgets=["replace_with_performer_reference.png", "image"],
        ),
        _node(
            7,
            "LoadAudio",
            (0, 930),
            (360, 120),
            6,
            title="7. Complete original song · final mux only",
            outputs=[{"name": "AUDIO", "type": "AUDIO", "links": [7, 8]}],
            widgets=["replace_with_complete_song.wav"],
        ),
        _node(
            8,
            "LoadAudio",
            (400, 930),
            (430, 120),
            7,
            title="8. Required isolated vocal / clear speech · Vocal Lock",
            outputs=[{"name": "AUDIO", "type": "AUDIO", "links": [9, 10]}],
            widgets=["replace_with_timeline_aligned_isolated_vocal.wav"],
        ),
        _node(
            9,
            "MiniMaxH3MVVocalLockScenePlannerV2T8Advanced",
            (930, 0),
            (600, 470),
            8,
            title="9. V2 local isolated-vocal scene plan · no API",
            inputs=[
                {"name": "full_song", "type": "AUDIO", "link": 7},
                {"name": "vocal_lock_audio", "type": "AUDIO", "link": 9},
            ],
            outputs=[
                {"name": "scene_plan", "type": "H3_T8_MV_SCENE_PLAN", "links": [11]},
                {"name": "scene_count", "type": "INT", "links": []},
                {"name": "duration_seconds", "type": "FLOAT", "links": []},
                {"name": "timeline_json", "type": "STRING", "links": []},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            widgets=[5.0, 7.0, 10.0, 50, 0.12, ""],
            color="#325f6d",
            bgcolor="#19333a",
        ),
        _node(
            10,
            "MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced",
            (930, 540),
            (720, 720),
            9,
            title="10. Official six-section Ref2VA Vocal Lock prompt",
            inputs=[{"name": "scene_plan", "type": "H3_T8_MV_SCENE_PLAN", "link": 11}],
            outputs=[
                {
                    "name": "mv_vocal_lock_prompt_plan",
                    "type": "H3_T8_MV_VOCAL_LOCK_PROMPT_PLAN",
                    "links": [12],
                },
                {"name": "segment_prompts_json", "type": "STRING", "links": []},
                {
                    "name": "prompt_relay_events",
                    "type": "H3_T8_PROMPT_RELAY_EVENTS",
                    "links": [],
                },
                {"name": "prompt_preview", "type": "STRING", "links": []},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            widgets=[
                "A coherent cinematic performance focused on the same lead performer, with natural expression and restrained motion.",
                "the same lead performer shown in the reference picture",
                "cinematic realism, natural skin, realistic light and texture",
                "locked-off static camera with stable framing, no camera movement, and continuous mouth visibility",
                "singing",
                "English",
                "",
                "keeps the mouth naturally closed and breathes with the rhythm",
            ],
            color="#5d4d79",
            bgcolor="#30283f",
        ),
        _node(
            11,
            "MiniMaxH3LocalMVVocalLockRendererV2T8Advanced",
            (1740, 0),
            (780, 960),
            10,
            title="11. Local serial H3 Vocal Lock render · full song final mux",
            inputs=[
                {"name": "model", "type": "MODEL", "link": 2},
                {"name": "clip", "type": "CLIP", "link": 3},
                {"name": "video_vae", "type": "VAE", "link": 4},
                {"name": "audio_vae", "type": "VAE", "link": 5},
                {"name": "reference_image", "type": "IMAGE", "link": 6},
                {"name": "full_song", "type": "AUDIO", "link": 8},
                {"name": "vocal_lock_audio", "type": "AUDIO", "link": 10},
                {
                    "name": "mv_vocal_lock_prompt_plan",
                    "type": "H3_T8_MV_VOCAL_LOCK_PROMPT_PLAN",
                    "link": 12,
                },
            ],
            outputs=[
                {"name": "video", "type": "VIDEO", "links": []},
                {"name": "video_path", "type": "STRING", "links": []},
                {"name": "manifest_path", "type": "STRING", "links": []},
                {"name": "completed_scenes", "type": "INT", "links": []},
                {"name": "status", "type": "STRING", "links": []},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            widgets=[
                "h3_local_mv_vocal_lock_v2",
                736,
                416,
                123456789,
                8,
                6.0,
                3.0,
                "dual_clock_euler",
                "native_flow",
                True,
                "H3_Local_MV_VocalLock_V2",
                8,
                18,
                "minimax_h3_ref2va_int8_convrot+turbo4-vocal-lock-v2",
            ],
            color="#6d3f71",
            bgcolor="#381f3b",
        ),
        _node(
            12,
            "MarkdownNote",
            (2600, 0),
            (650, 330),
            11,
            title="NOTE 1 · 两条音频职责不同",
            widgets=[
                "## 必须连接同时间线的两条音频\n`vocal_lock_audio` 必须是隔离人声或清晰对白，逐场景进入 H3 `lock_source`。`full_song` 不进入 H3，也不进入分段候选，只在全部画面合成后一次性混入最终成片。两条音频必须从同一时间零点开始并解析为相同的24fps时长。"
            ],
            color="#604b1d",
            bgcolor="#332810",
        ),
        _node(
            13,
            "MarkdownNote",
            (2600, 390),
            (650, 330),
            12,
            title="NOTE 2 · 官方格式与口型可见性",
            widgets=[
                "## V2不是旧版自定义六行提示词\nPrompt Compiler 固定输出官方 `subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music`，并绑定 `<Subject 1>`、`<Picture 1>`、`<Audio 1>: fully_copy`。人声场景强制中近景、正面或3/4脸、完整无遮挡嘴部。"
            ],
            color="#214d35",
            bgcolor="#12281d",
        ),
        _node(
            14,
            "MarkdownNote",
            (2600, 780),
            (650, 360),
            13,
            title="NOTE 3 · 验收边界",
            widgets=[
                "## 串行运行，并实际听人声看嘴型\n默认736×416、8步和锁定机位，优先保证单条低负载验证并降低人物轮廓拖影。参考图尽量使用与目标正面/3/4脸方向一致的清晰人物图；侧脸强行转正会要求模型补出不可见信息，可能产生柔化、光晕或身份漂移。H3 Vocal Lock是音频条件生成，不是确定性音素求解器；成片仍要做音画偏移检查，并按正常速度完整观看嘴唇开合、辅音闭合、身份、手和背景。更换素材或参数时使用新的 `chain_id`。不会提交HTTP `/prompt`，也不调用外部API。"
            ],
            color="#6b3030",
            bgcolor="#371919",
        ),
    ]
    return {
        "id": "3881c367-6f1a-44bd-b6e6-d07fd1c56ec3",
        "revision": 0,
        "last_node_id": 14,
        "last_link_id": 12,
        "nodes": nodes,
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 2, 0, 11, 0, "MODEL"],
            [3, 3, 0, 11, 1, "CLIP"],
            [4, 4, 0, 11, 2, "VAE"],
            [5, 5, 0, 11, 3, "VAE"],
            [6, 6, 0, 11, 4, "IMAGE"],
            [7, 7, 0, 9, 0, "AUDIO"],
            [8, 7, 0, 11, 5, "AUDIO"],
            [9, 8, 0, 9, 1, "AUDIO"],
            [10, 8, 0, 11, 6, "AUDIO"],
            [11, 9, 0, 10, 0, "H3_T8_MV_SCENE_PLAN"],
            [12, 10, 0, 11, 7, "H3_T8_MV_VOCAL_LOCK_PROMPT_PLAN"],
        ],
        "groups": [
            {
                "id": 1,
                "title": "Local H3 models + reference + two local audio tracks",
                "bounding": [-40, -70, 900, 1210],
                "color": "#3f789e",
                "font_size": 26,
                "flags": {},
            },
            {
                "id": 2,
                "title": "V2 isolated-vocal planning + official Ref2VA compiler",
                "bounding": [890, -70, 800, 1390],
                "color": "#397382",
                "font_size": 26,
                "flags": {},
            },
            {
                "id": 3,
                "title": "Strictly serial local H3 Vocal Lock generation",
                "bounding": [1700, -70, 860, 1090],
                "color": "#704c91",
                "font_size": 26,
                "flags": {},
            },
            {
                "id": 4,
                "title": "Read before running",
                "bounding": [2560, -70, 730, 1280],
                "color": "#8b6b32",
                "font_size": 26,
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.52, "offset": [70, 130]},
            "ue_links": [],
            "t8_mv_state": "all_local_mv_vocal_lock_v2",
        },
        "version": 0.4,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
