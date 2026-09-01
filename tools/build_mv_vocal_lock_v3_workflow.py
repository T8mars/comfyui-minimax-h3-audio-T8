from __future__ import annotations

import json
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V2_Ref2VA_8Step_Advanced_EXP.json"
)
OUTPUT = (
    ROOT
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V3_Official_Ref2V_Turbo4_Advanced_EXP.json"
)
USER_OUTPUT = (
    ROOT.parents[1]
    / "user/default/workflows/MiniMax H3 T8/24-mv-lipsync"
    / OUTPUT.name
)

OFFICIAL_REF2V_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def build() -> dict:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow["id"] = str(uuid.UUID("796b561b-8b75-4dc4-9ca0-da4d531865ae"))
    nodes = {int(node["id"]): node for node in workflow["nodes"]}

    lora_loader = nodes[2]
    lora_loader["title"] = "2. Official Ref2V Turbo v0.1 · 4 NFE · strength 1.0"
    lora_loader["widgets_values"] = [OFFICIAL_REF2V_LORA, 1.0]

    clip_loader = nodes[3]
    clip_loader["outputs"][0]["links"] = [13]
    clip_loader["title"] = "3. Local H3 Qwen3-VL CLIP · cached by exact reference prefix"

    director = nodes[10]
    director["type"] = "MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced"
    director["title"] = "10. V3 explicit shot plan · one person / one face"
    director["properties"]["Node name for S&R"] = director["type"]
    director["widgets_values"] = [
        "A coherent cinematic performance focused on the same lead performer, with natural expression and restrained motion.",
        "the same lead performer shown in the reference picture",
        "cinematic realism, natural skin, realistic light and texture",
        "",
        "singing",
        "English",
        "",
        "keeps the mouth naturally closed and breathes with the rhythm",
    ]

    renderer = nodes[11]
    renderer["type"] = "MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced"
    renderer["title"] = "11. V3 local serial render · independent visual resume contract"
    renderer["properties"]["Node name for S&R"] = renderer["type"]
    renderer["widgets_values"][0] = "h3_local_mv_vocal_lock_v3_visual"
    renderer["widgets_values"][1] = 1024
    renderer["widgets_values"][2] = 768
    renderer["widgets_values"][4] = 4
    renderer["widgets_values"][5] = 12.0
    renderer["widgets_values"][6] = 3.0
    renderer["widgets_values"][7] = "euler"
    renderer["widgets_values"][8] = "simple"
    renderer["widgets_values"][10] = "H3_Local_MV_VocalLock_V3"
    renderer["widgets_values"][13] = (
        "minimax_h3_ref2va_int8_convrot+official-ref2v-turbo4-v0.1-vocal-lock-v3"
    )

    nodes[12]["widgets_values"] = [
        "## 官方 Ref2V Turbo4 合同\n必须使用 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`、strength 1.0、Euler/simple、4 NFE、shift 12/3。`vocal_lock_audio`逐场景进入H3 `lock_source`；`full_song`只在所有画面通过并合成后混入一次。"
    ]
    nodes[13]["title"] = "NOTE 2 · V3单人物空间合同"
    nodes[13]["widgets_values"] = [
        "## 每一镜都不是只换camera pattern\nV3为每个scene保存camera / lighting / performance / emotion。留空使用确定性安全棚拍弧线；也可填与scene_count等长的JSON对象列表。整幅画面强制恰好一名人物和一张人脸，并禁止镜像、反射、投影、屏幕、海报、肖像、重影、双重曝光、背景人物和可见道具。"
    ]
    nodes[14]["title"] = "NOTE 3 · 失败镜头不能靠accepted计数通过"
    nodes[14]["widgets_values"] = [
        "## accepted只表示文件和合同已落盘，不表示画质通过\n首次32秒阶段门的第4镜出现巨幅重复脸，已判失败并停止第5镜。V3只能降低已知风险，不能自动证明画面正确。每镜必须检查全程单人物、身份、背景、轮廓和口型；失败后换新chain，不能续用旧accepted片段。32秒/5镜通过后才允许约90秒终验。"
    ]

    cache_node = {
        "id": 15,
        "type": "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
        "pos": [430, 150],
        "size": [430, 190],
        "flags": {},
        "order": 10,
        "mode": 0,
        "inputs": [{"name": "clip", "type": "CLIP", "link": 13}],
        "outputs": [
            {"name": "clip", "type": "CLIP", "links": [3]},
            {"name": "cache_handle", "type": "H3_T8_QWEN_PREFIX_CACHE", "links": []},
            {"name": "report_json", "type": "STRING", "links": []},
        ],
        "properties": {
            "Node name for S&R": "MiniMaxH3QwenReferencePrefixCacheT8Advanced"
        },
        "widgets_values": ["memory_lru_exp", 1, 1024.0, 0],
        "title": "3b. Same-reference Qwen prefix cache · bounded CPU LRU",
        "color": "#315b55",
        "bgcolor": "#19302d",
    }
    workflow["nodes"].append(cache_node)
    workflow["links"] = [
        [13, 3, 0, 15, 0, "CLIP"],
        *[
            [link[0], 15, link[2], link[3], link[4], link[5]]
            if int(link[0]) == 3
            else link
            for link in workflow["links"]
        ],
    ]
    workflow["last_node_id"] = 15
    workflow["last_link_id"] = 13
    workflow["extra"]["t8_mv_state"] = "all_local_mv_vocal_lock_v3_visual_director"
    return workflow


def main() -> None:
    text = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    USER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    USER_OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)
    print(USER_OUTPUT)


if __name__ == "__main__":
    main()
