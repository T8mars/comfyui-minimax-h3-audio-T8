from __future__ import annotations

import json
from pathlib import Path

try:
    from .frontend_workflow_compat import normalize_native_widget_inputs
except ImportError:  # Direct execution puts tools/ on sys.path.
    from frontend_workflow_compat import normalize_native_widget_inputs


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples"
    / "workflows"
    / "03-image-video-edit"
    / "2026-08-10_H3_Ref2VA_Visual_Reference_Strength_EXP.json"
)
TARGET = (
    ROOT
    / "examples"
    / "workflows"
    / "12-system-memory"
    / "2026-08-28_H3_Qwen_Prefix_and_Block_Cache_Ref2VA_Stock20_Advanced_EXP.json"
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == int(node_id))


def _disconnect(workflow: dict, link_id: int) -> None:
    link = next(link for link in workflow["links"] if int(link[0]) == int(link_id))
    _, source_id, output_slot, target_id, input_slot, _link_type = link
    source = _node(workflow, source_id)
    target = _node(workflow, target_id)
    source["outputs"][output_slot]["links"] = [
        value
        for value in (source["outputs"][output_slot].get("links") or [])
        if int(value) != int(link_id)
    ] or None
    target["inputs"][input_slot]["link"] = None
    workflow["links"] = [
        value for value in workflow["links"] if int(value[0]) != int(link_id)
    ]


def _connect(
    workflow: dict,
    source: dict,
    output_slot: int,
    target: dict,
    input_slot: int,
    link_type: str,
) -> int:
    link_id = int(workflow["last_link_id"]) + 1
    workflow["last_link_id"] = link_id
    workflow["links"].append(
        [link_id, source["id"], output_slot, target["id"], input_slot, link_type]
    )
    links = source["outputs"][output_slot].get("links") or []
    source["outputs"][output_slot]["links"] = [*links, link_id]
    target["inputs"][input_slot]["link"] = link_id
    return link_id


def _qwen_cache_node() -> dict:
    return {
        "id": 14,
        "type": "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
        "title": "Qwen reference-prefix LRU · one bounded CPU entry",
        "pos": [430, -300],
        "size": [520, 300],
        "flags": {},
        "order": 5,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "mode", "type": "COMBO", "widget": {"name": "mode"}, "link": None},
            {"name": "max_entries", "type": "INT", "widget": {"name": "max_entries"}, "link": None},
            {"name": "maximum_cache_mib", "type": "FLOAT", "widget": {"name": "maximum_cache_mib"}, "link": None},
            {"name": "cache_epoch", "type": "INT", "widget": {"name": "cache_epoch"}, "link": None},
        ],
        "outputs": [
            {"name": "clip", "type": "CLIP", "links": []},
            {"name": "cache_handle", "type": "H3_T8_QWEN_PREFIX_CACHE", "links": []},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "properties": {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
        },
        "widgets_values": ["memory_lru_exp", 1, 256.0, 0],
    }


def _qwen_stats_node() -> dict:
    return {
        "id": 15,
        "type": "MiniMaxH3QwenPrefixCacheStatsT8Advanced",
        "title": "Qwen cache stats · forced after Conditioning",
        "pos": [1090, -300],
        "size": [470, 150],
        "flags": {},
        "order": 15,
        "mode": 0,
        "inputs": [
            {"name": "cache_handle", "type": "H3_T8_QWEN_PREFIX_CACHE", "link": None},
            {"name": "after_report", "type": "STRING", "link": None, "shape": 7},
        ],
        "outputs": [{"name": "report_json", "type": "STRING", "links": None}],
        "properties": {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": "MiniMaxH3QwenPrefixCacheStatsT8Advanced",
        },
        "widgets_values": [],
    }


def _block_cache_node() -> dict:
    return {
        "id": 16,
        "type": "MiniMaxH3BlockCacheT8",
        "title": "External H3 BlockCache · conservative CPU profile",
        "pos": [430, 840],
        "size": [520, 310],
        "flags": {},
        "order": 6,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "residual_diff_threshold", "type": "FLOAT", "widget": {"name": "residual_diff_threshold"}, "link": None},
            {"name": "start_percent", "type": "FLOAT", "widget": {"name": "start_percent"}, "link": None},
            {"name": "end_percent", "type": "FLOAT", "widget": {"name": "end_percent"}, "link": None},
            {"name": "max_consecutive_hits", "type": "INT", "widget": {"name": "max_consecutive_hits"}, "link": None},
            {"name": "cache_device", "type": "COMBO", "widget": {"name": "cache_device"}, "link": None},
            {"name": "metric_stride", "type": "INT", "widget": {"name": "metric_stride"}, "link": None},
            {"name": "verbose", "type": "BOOLEAN", "widget": {"name": "verbose"}, "link": None},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "properties": {"Node name for S&R": "MiniMaxH3BlockCacheT8"},
        "widgets_values": [0.08, 0.08, 0.95, 2, "cpu", 8, False],
    }


def _note(node_id: int, title: str, pos: list[int], text: str) -> dict:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "title": title,
        "pos": pos,
        "size": [620, 260],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "MarkdownNote"},
        "widgets_values": text,
    }


def build() -> dict:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow["id"] = "20d93ef5-c5c0-4ec0-b21c-7edec9f20a44"
    workflow["revision"] = 0
    workflow["extra"]["workflow_title"] = (
        "MiniMax H3 Qwen Prefix + External BlockCache Ref2VA Stock20 Advanced EXP"
    )

    unet = _node(workflow, 1)
    clip = _node(workflow, 2)
    conditioning = _node(workflow, 6)
    strength = _node(workflow, 7)
    sampler = _node(workflow, 8)
    guider = _node(workflow, 10)
    save = _node(workflow, 13)

    # Remove the unrelated visual-strength experiment and its two links.
    for link in list(workflow["links"]):
        if int(link[1]) == strength["id"] or int(link[3]) == strength["id"]:
            _disconnect(workflow, int(link[0]))
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != strength["id"]]

    # Replace direct CLIP and MODEL routes with two orthogonal bounded caches.
    _disconnect(workflow, next(link[0] for link in workflow["links"] if link[1] == 2 and link[3] == 6))
    _disconnect(workflow, next(link[0] for link in workflow["links"] if link[1] == 1 and link[3] == 8))

    qwen_cache = _qwen_cache_node()
    stats = _qwen_stats_node()
    block_cache = _block_cache_node()
    workflow["nodes"].extend([qwen_cache, stats, block_cache])

    _connect(workflow, clip, 0, qwen_cache, 0, "CLIP")
    _connect(workflow, qwen_cache, 0, conditioning, 0, "CLIP")
    _connect(workflow, qwen_cache, 1, stats, 0, "H3_T8_QWEN_PREFIX_CACHE")
    _connect(workflow, conditioning, 5, stats, 1, "STRING")
    _connect(workflow, unet, 0, block_cache, 0, "MODEL")
    _connect(workflow, block_cache, 0, sampler, 0, "MODEL")
    _connect(workflow, conditioning, 0, guider, 1, "CONDITIONING")

    conditioning["title"] = "Ref2VA Stock20 · cached Qwen reference prefix"
    conditioning["widgets_values"][0] = (
        "Use <Picture 1> as the visual reference. A natural cinematic portrait in soft window "
        "light, realistic skin and fabric texture, stable identity and composition. Natural room "
        "ambience, no music."
    )
    save["title"] = "Save synchronized cache-composition candidate"
    save["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3_T8/QwenPrefix_BlockCache_Ref2VA_Stock20_EXP"
    )

    workflow["nodes"].extend(
        [
            _note(
                17,
                "① 两种缓存的职责",
                [0, 1220],
                "## CLIP缓存与MODEL缓存互不替代\n\n- Qwen Prefix Cache只复用相同参考媒体的视觉前缀；新的文字后缀仍重算。\n- 外部BlockCache只缓存H3 DiT中间块；block 0会重算，最多连续命中2次。\n- 首次运行是预热；换参考图、模型、LoRA或合同后应视为新任务。",
            ),
            _note(
                18,
                "② 安装与推荐参数",
                [650, 1220],
                "## 需要单独安装外部BlockCache\n\n此工作流依赖`comfyui-minimax-h3-blockcache-T8`，本项目不会复制它的源码。推荐从`0.08 / CPU / max 2 hits`开始；Qwen缓存保持1条、256MiB。不要同时串联EasyCache、LazyCache、Spectrum或其他DiT block replacement。",
            ),
            _note(
                19,
                "③ 结果边界",
                [1300, 1220],
                "## 性能优先EXP，不是无损模式\n\n受测Qwen缓存和BlockCache都可能因浮点路径或跳过block改变画面/音频；`0.08`只是历史Stock20质量优先候选，不保证所有素材。它们也不等于省显存或16GB安全。重要成片请与Cache OFF同seed试听和审片。",
            ),
        ]
    )
    workflow["last_node_id"] = 19
    workflow["nodes"] = sorted(workflow["nodes"], key=lambda item: int(item["id"]))
    normalize_native_widget_inputs(workflow)
    return workflow


def main() -> None:
    workflow = build()
    TARGET.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
