from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json"
)


def _node(node_id, node_type, pos, size, order, *, title="", inputs=None, outputs=None,
          widgets=None, color=None, bgcolor=None):
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
            1, "UNETLoader", (0, 0), (360, 82), 0,
            title="1. Load local MiniMax H3 Ref2VA model",
            outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}],
            widgets=["minimax_h3_ref2va_int8_convrot.safetensors", "default"],
        ),
        _node(
            2, "LoraLoaderBypassModelOnly", (400, 0), (430, 105), 1,
            title="2. Turbo4 LoRA · local model only",
            inputs=[{"name": "model", "type": "MODEL", "link": 1}],
            outputs=[{"name": "MODEL", "type": "MODEL", "links": [2]}],
            widgets=["minimax_h3_turbo_4步加速ema_comfyui.safetensors", 1.0],
        ),
        _node(
            3, "CLIPLoader", (0, 150), (360, 106), 2,
            title="3. Local H3 Qwen3-VL CLIP",
            outputs=[{"name": "CLIP", "type": "CLIP", "links": [3]}],
            widgets=["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
        ),
        _node(
            4, "VAELoader", (0, 310), (360, 58), 3,
            title="4. H3 video VAE",
            outputs=[{"name": "VAE", "type": "VAE", "links": [4]}],
            widgets=["minimax_h3_video_vae_fp16.safetensors"],
        ),
        _node(
            5, "VAELoader", (0, 420), (360, 58), 4,
            title="5. H3 audio VAE",
            outputs=[{"name": "VAE", "type": "VAE", "links": [5]}],
            widgets=["minimax_h3_audio_vae_fp32.safetensors"],
        ),
        _node(
            6, "LoadImage", (0, 560), (360, 310), 5,
            title="6. Replace with performer reference image",
            outputs=[
                {"name": "IMAGE", "type": "IMAGE", "links": [6]},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            widgets=["replace_with_performer_reference.png", "image"],
        ),
        _node(
            7, "LoadAudio", (0, 930), (360, 120), 6,
            title="7. Replace with the complete original song",
            outputs=[{"name": "AUDIO", "type": "AUDIO", "links": [7, 8]}],
            widgets=["replace_with_complete_song.wav"],
        ),
        _node(
            8, "MiniMaxH3MVVocalScenePlannerT8Advanced", (900, 0), (560, 430), 7,
            title="8. Local song analysis · no API",
            inputs=[
                {"name": "full_song", "type": "AUDIO", "link": 7},
                {"name": "vocal_stem", "type": "AUDIO", "link": None, "shape": 7},
            ],
            outputs=[
                {"name": "scene_plan", "type": "H3_T8_MV_SCENE_PLAN", "links": [9]},
                {"name": "scene_count", "type": "INT", "links": []},
                {"name": "duration_seconds", "type": "FLOAT", "links": []},
                {"name": "timeline_json", "type": "STRING", "links": []},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            widgets=[5.0, 7.0, 10.0, 100, "assume_vocal", ""],
            color="#325f6d", bgcolor="#19333a",
        ),
        _node(
            9, "MiniMaxH3MVRef2VAPromptCompilerT8Advanced", (900, 500), (660, 640), 8,
            title="9. Deterministic Ref2VA scene prompts · no LLM",
            inputs=[{"name": "scene_plan", "type": "H3_T8_MV_SCENE_PLAN", "link": 9}],
            outputs=[
                {"name": "mv_prompt_plan", "type": "H3_T8_MV_PROMPT_PLAN", "links": [10]},
                {"name": "segment_prompts_json", "type": "STRING", "links": []},
                {"name": "prompt_relay_events", "type": "H3_T8_PROMPT_RELAY_EVENTS", "links": []},
                {"name": "prompt_preview", "type": "STRING", "links": []},
                {"name": "report_json", "type": "STRING", "links": []},
            ],
            widgets=[
                "A singer performs through a coherent cinematic music video with natural expression and intentional scene changes.",
                "the same lead performer shown in the reference picture",
                "cinematic music video, natural skin, realistic light and texture",
                "stable medium close-up with subtle handheld movement\nsmooth lateral tracking medium shot\nrestrained slow push-in with a stable background",
                "keeps the mouth naturally closed and moves with the rhythm",
                "",
            ],
            color="#5d4d79", bgcolor="#30283f",
        ),
        _node(
            10, "MiniMaxH3LocalMVInNodeRendererT8Advanced", (1650, 0), (720, 900), 9,
            title="10. All-local serial H3 MV render + original song",
            inputs=[
                {"name": "model", "type": "MODEL", "link": 2},
                {"name": "clip", "type": "CLIP", "link": 3},
                {"name": "video_vae", "type": "VAE", "link": 4},
                {"name": "audio_vae", "type": "VAE", "link": 5},
                {"name": "reference_image", "type": "IMAGE", "link": 6},
                {"name": "full_song", "type": "AUDIO", "link": 8},
                {"name": "mv_prompt_plan", "type": "H3_T8_MV_PROMPT_PLAN", "link": 10},
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
                "h3_local_mv_demo", 1056, 608, 123456789, 4, 12.0, 3.0,
                "dual_clock_euler", "native_flow", True, "H3_Local_MV", 8, 18,
                "minimax_h3_ref2va_int8_convrot+turbo4",
            ],
            color="#6d3f71", bgcolor="#381f3b",
        ),
        _node(
            11, "MarkdownNote", (2450, 0), (600, 300), 10,
            title="NOTE 1 · 完全本地，不使用外部API",
            widgets=[
                "## 所有 H3 视频都由本工作流生成\n分镜只做本地 CPU 音频分析；提示词是确定性模板；Renderer 直接调用已连接的本地 H3 MODEL。不会提交 HTTP `/prompt`，也不会调用在线 H3、LLM、TTS、音乐或视频 API。"
            ], color="#214d35", bgcolor="#12281d",
        ),
        _node(
            12, "MarkdownNote", (2450, 350), (600, 330), 11,
            title="NOTE 2 · 原曲与断点续跑",
            widgets=[
                "## 完整原曲只混入一次\n每段生成时歌曲作为 `<Audio 1>` 驱动表演，但分段生成音频不会进入最终成片。视频合成后一次性混入完整原曲。中断后保持全部参数不变再次运行；更换模型、素材、提示词或尺寸时使用新的 `chain_id`。"
            ], color="#604b1d", bgcolor="#332810",
        ),
        _node(
            13, "MarkdownNote", (2450, 730), (600, 350), 12,
            title="NOTE 3 · 质量与显存",
            widgets=[
                "## 串行运行，不要并发\n默认 Ref2VA + Turbo4，1056×608。显存不足先降到 832×480 或 736×416。`assume_vocal` 适合普通歌曲；有本地人声干声时再连接 Planner 的 `vocal_stem`。这不是音素级口型模型，最终要人工检查口型、脸、手、背景和切镜。"
            ], color="#6b3030", bgcolor="#371919",
        ),
    ]
    return {
        "id": "cf51de44-4850-4fa5-b876-9d0e9c689bd8",
        "revision": 0,
        "last_node_id": 13,
        "last_link_id": 10,
        "nodes": nodes,
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 2, 0, 10, 0, "MODEL"],
            [3, 3, 0, 10, 1, "CLIP"],
            [4, 4, 0, 10, 2, "VAE"],
            [5, 5, 0, 10, 3, "VAE"],
            [6, 6, 0, 10, 4, "IMAGE"],
            [7, 7, 0, 8, 0, "AUDIO"],
            [8, 7, 0, 10, 5, "AUDIO"],
            [9, 8, 0, 9, 0, "H3_T8_MV_SCENE_PLAN"],
            [10, 9, 0, 10, 6, "H3_T8_MV_PROMPT_PLAN"],
        ],
        "groups": [
            {"id": 1, "title": "Local H3 models + local media", "bounding": [-40, -70, 900, 1190], "color": "#3f789e", "font_size": 26, "flags": {}},
            {"id": 2, "title": "Local audio planning + deterministic prompts", "bounding": [860, -70, 740, 1260], "color": "#397382", "font_size": 26, "flags": {}},
            {"id": 3, "title": "Strictly serial local H3 generation", "bounding": [1610, -70, 800, 1040], "color": "#704c91", "font_size": 26, "flags": {}},
            {"id": 4, "title": "Read before running", "bounding": [2410, -70, 680, 1220], "color": "#8b6b32", "font_size": 26, "flags": {}},
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.55, "offset": [70, 130]},
            "ue_links": [],
            "t8_mv_state": "all_local_mv_v1",
        },
        "version": 0.4,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
