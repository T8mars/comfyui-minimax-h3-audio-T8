from __future__ import annotations

import json
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "workflows" / "23-flashvsr"


PROFILES = {
    "Quality_Locked": {
        "profile": "quality_locked",
        "title": "推荐：固定LCSA质量预算 / Recommended",
        "note": "固定2.0/3.0/11；优先保持细节和稳定性。",
    },
    "Balanced_Dynamic_EXP": {
        "profile": "balanced_dynamic_exp",
        "title": "动态预算候选 / Opt-in EXP",
        "note": "只在低运动内部块降低预算；首尾和高运动块保持2.0/3.0/11，必须人工复核。",
    },
    "Memory_Safe": {
        "profile": "memory_safe",
        "title": "低显存分块 / Memory Safe",
        "note": "保持2.0/3.0/11，通过128像素分块与阶段卸载降低峰值；速度通常更慢。",
    },
}


def _node(node_id, node_type, title, pos, size, order, inputs, outputs, widgets):
    cnr = "comfy-core" if node_type in {
        "MarkdownNote", "LoadVideo", "GetVideoComponents", "PreviewImage", "CreateVideo", "SaveVideo"
    } else "minimax-h3-audio-T8"
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"cnr_id": cnr, "Node name for S&R": node_type},
        "widgets_values": widgets,
    }


def _workflow(config):
    profile = config["profile"]
    note = (
        "## MiniMax H3 FlashVSR v1.1 后处理超分\n\n"
        f"- {config['note']}\n"
        "- 模型放在 `ComfyUI/models/FlashVSR-v1.1/`，需包含 DiT、LQ_proj_in、TCDecoder、Wan VAE 和 posi_prompt。\n"
        "- 默认 Tiny + BF16 + 2倍；官方主要为4倍训练，2倍适合保守试用。不要强行拉伸画幅。\n"
        "- 节点不设像素上限；资源不足时选 Memory Safe，不会因为哈希或文件大小拒绝模型。\n"
        "- AUDIO原对象直通，不重采样、不降噪、不改音量。先预览候选，再决定是否采用。"
    )
    nodes = [
        _node(1, "MarkdownNote", "使用说明 / Read first", [0, 0], [920, 300], 0, [], [], [note]),
        _node(2, "LoadVideo", "选择H3成片", [0, 350], [380, 124], 1, [], [{"name": "VIDEO", "type": "VIDEO", "links": [1]}], ["replace_with_h3_video.mp4"]),
        _node(3, "GetVideoComponents", "解出帧与原音频", [430, 350], [380, 126], 2, [{"name": "video", "type": "VIDEO", "link": 1}], [{"name": "images", "type": "IMAGE", "links": [2, 5]}, {"name": "audio", "type": "AUDIO", "links": [6]}, {"name": "fps", "type": "FLOAT", "links": None}, {"name": "bit_depth", "type": "COMBO", "links": None}, {"name": "color_space", "type": "COMBO", "links": None}], []),
        _node(4, "MiniMaxH3FlashVSRModelT8Advanced", "官方FlashVSR-v1.1 Tiny", [430, 520], [430, 190], 3, [], [{"name": "flashvsr_model", "type": "H3_T8_FLASHVSR_MODEL", "links": [3]}, {"name": "report_json", "type": "STRING", "links": None}], ["FlashVSR-v1.1", "tiny", "bf16"]),
        _node(5, "MiniMaxH3FlashVSRExecutionPlanT8Advanced", config["title"], [910, 300], [520, 390], 4, [{"name": "frames", "type": "IMAGE", "link": 2}], [{"name": "plan", "type": "H3_T8_FLASHVSR_PLAN", "links": [4]}, {"name": "report_json", "type": "STRING", "links": None}], [profile, "auto", "auto", 2.0, 3.0, 11, 128 if profile == "memory_safe" else 256, 16 if profile == "memory_safe" else 24]),
        _node(6, "MiniMaxH3FlashVSRRestoreT8Advanced", "2倍超分；原音频直通", [1480, 300], [560, 360], 5, [{"name": "flashvsr_model", "type": "H3_T8_FLASHVSR_MODEL", "link": 3}, {"name": "plan", "type": "H3_T8_FLASHVSR_PLAN", "link": 4}, {"name": "frames", "type": "IMAGE", "link": 5}, {"name": "audio", "type": "AUDIO", "link": 6}], [{"name": "restored_frames", "type": "IMAGE", "links": [7, 9]}, {"name": "source_frames", "type": "IMAGE", "links": [8]}, {"name": "audio", "type": "AUDIO", "links": [10]}, {"name": "report_json", "type": "STRING", "links": None}], [2, 26083001, True, "offload_after"]),
        _node(7, "PreviewImage", "原片 / Source", [2100, 100], [380, 150], 6, [{"name": "images", "type": "IMAGE", "link": 8}], [{"name": "images", "type": "IMAGE", "links": None}], []),
        _node(8, "PreviewImage", "FlashVSR候选", [2100, 300], [380, 150], 7, [{"name": "images", "type": "IMAGE", "link": 7}], [{"name": "images", "type": "IMAGE", "links": None}], []),
        _node(9, "CreateVideo", "24fps合成并接回原音频", [2100, 510], [380, 264], 8, [{"name": "images", "type": "IMAGE", "link": 9}, {"name": "audio", "type": "AUDIO", "link": 10, "shape": 7}], [{"name": "VIDEO", "type": "VIDEO", "links": [11]}], [24.0, 8, "sRGB"]),
        _node(10, "SaveVideo", "保存FlashVSR候选", [2530, 510], [380, 238], 9, [{"name": "video", "type": "VIDEO", "link": 11}, {"name": "format", "type": "COMFY_DYNAMICCOMBO_V3", "link": None}, {"name": "codec", "type": "COMFY_DYNAMICCOMBO_V3", "link": None, "shape": 7}], [{"name": "video", "type": "VIDEO", "links": None}], [f"MiniMaxH3/FlashVSR/{profile}"]),
    ]
    links = [
        [1, 2, 0, 3, 0, "VIDEO"],
        [2, 3, 0, 5, 0, "IMAGE"],
        [3, 4, 0, 6, 0, "H3_T8_FLASHVSR_MODEL"],
        [4, 5, 0, 6, 1, "H3_T8_FLASHVSR_PLAN"],
        [5, 3, 0, 6, 2, "IMAGE"],
        [6, 3, 1, 6, 3, "AUDIO"],
        [7, 6, 0, 8, 0, "IMAGE"],
        [8, 6, 1, 7, 0, "IMAGE"],
        [9, 6, 0, 9, 0, "IMAGE"],
        [10, 6, 2, 9, 1, "AUDIO"],
        [11, 9, 0, 10, 0, "VIDEO"],
    ]
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8-flashvsr-{profile}")),
        "revision": 0,
        "last_node_id": 10,
        "last_link_id": 11,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.78, "offset": [100, 70]}, "frontendVersion": "1.24.3"},
        "version": 0.4,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, config in PROFILES.items():
        path = OUTPUT / f"2026-08-30_H3_FlashVSR_{suffix}_Advanced_EXP.json"
        path.write_text(
            json.dumps(_workflow(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
