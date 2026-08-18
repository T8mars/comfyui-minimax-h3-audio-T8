from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
BASE = WORKFLOWS / "2026-08-18_H3_SPEED_T2VA_Stock20_Advanced_EXP.json"


def node(workflow: dict, node_id: int) -> dict:
    return next(item for item in workflow["nodes"] if item["id"] == node_id)


def load_image_node(node_id: int, x: int, y: int, filename: str, link_id: int) -> dict:
    return {
        "id": node_id,
        "type": "LoadImage",
        "pos": [x, y],
        "size": [340, 320],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
            {"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None},
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [link_id]},
            {"name": "MASK", "type": "MASK", "links": None},
        ],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "LoadImage"},
        "widgets_values": [filename, "image"],
    }


def build_fl2va(base: dict) -> dict:
    workflow = copy.deepcopy(base)
    workflow["id"] = "d6e59e2c-0a2e-4c30-b10b-5a0f95f85002"
    workflow["last_node_id"] = 12
    workflow["last_link_id"] = 13
    source = node(workflow, 6)
    source["title"] = "2. FL2VA raw frames; VAE-encoded again at every SPEED canvas"
    source["widgets_values"][0] = (
        "The scene begins exactly from the first frame and ends exactly at the last frame. "
        "Natural cinematic motion between them, synchronized environment sound, no speech."
    )
    source["widgets_values"][2] = "FL2VA"
    source["inputs"][17]["link"] = 12
    source["inputs"][18]["link"] = 13
    sampler = node(workflow, 7)
    sampler["title"] = "3. Multimodal research scope: FL2VA stage rebuild"
    sampler["widgets_values"][2] = "multimodal_research_exp"
    combine = node(workflow, 9)
    combine["widgets_values"]["filename_prefix"] = "MiniMaxH3/speed_fl2va_stock20_exp"
    note = node(workflow, 10)
    note["title"] = "FL2VA mechanics passed one GPU chain; quality/speed remain EXP"
    note["widgets_values"] = [
        "## FL2VA机械链已完成一条真实GPU验证，仍是EXP\n\n替换两张同画幅首尾帧。每个SPEED阶段会按该阶段画布重新缩放并用video VAE重新编码，避免最终尺寸keyframe latent与粗阶段PackedLayout行数不一致。\n\n本机1024×576、124帧、Stock20代表链已确认首尾锚点、124帧、32kHz双声道和A/V时长机械正确；最低显存余量约445MiB，未达到512MiB安全门槛。尚未证明相对全分辨率的画质非劣、稳定加速或通用16GB安全。`multimodal_research_exp`仍是显式风险开关；不要叠其他sampler/model wrapper。"
    ]
    workflow["nodes"].extend(
        [
            load_image_node(11, 40, 540, "replace_with_first_frame.png", 12),
            load_image_node(12, 40, 900, "replace_with_last_frame.png", 13),
        ]
    )
    workflow["links"].extend(
        [[12, 11, 0, 6, 17, "IMAGE"], [13, 12, 0, 6, 18, "IMAGE"]]
    )
    return workflow


def build_ref2va(base: dict) -> dict:
    workflow = copy.deepcopy(base)
    workflow["id"] = "d6e59e2c-0a2e-4c30-b10b-5a0f95f85003"
    workflow["last_node_id"] = 11
    workflow["last_link_id"] = 12
    node(workflow, 1)["widgets_values"][0] = "minimax_h3_ref2va_int8_convrot.safetensors"
    source = node(workflow, 6)
    source["title"] = "2. Ref2VA raw reference; rebuilt at every SPEED canvas"
    source["widgets_values"][0] = (
        "Use <Picture 1> as the visual reference. Preserve its identity, materials and style "
        "while creating natural cinematic motion. Synchronized ambience, no speech."
    )
    source["widgets_values"][2] = "Ref2VA"
    source["inputs"].append(
        {
            "label": "ref_image_0",
            "name": "ref_images.ref_image_0",
            "type": "IMAGE",
            "link": 12,
        }
    )
    sampler = node(workflow, 7)
    sampler["title"] = "3. Multimodal research scope: Ref2VA stage rebuild"
    sampler["widgets_values"][2] = "multimodal_research_exp"
    combine = node(workflow, 9)
    combine["widgets_values"]["filename_prefix"] = "MiniMaxH3/speed_ref2va_stock20_exp"
    note = node(workflow, 10)
    note["title"] = "Ref2VA mechanics passed one GPU chain; adherence/speed remain EXP"
    note["widgets_values"] = [
        "## Ref2VA机械链已完成一条真实GPU验证，仍是EXP\n\n替换参考图。`ref_image_size=match`会按每级画布重新调整和编码参考图；`max`可能让固定reference token在粗阶段占主导并抵消SPEED收益。\n\n本机1024×576、124帧、Stock20单参考图代表链已确认124帧、32kHz双声道和A/V时长机械正确；最低显存余量约337MiB，未达到512MiB安全门槛。尚未证明身份/构图/参考遵循、音频非劣、稳定加速或通用16GB安全。`multimodal_research_exp`仍是显式风险开关。"
    ]
    workflow["nodes"].append(
        load_image_node(11, 40, 560, "replace_with_reference_image.png", 12)
    )
    workflow["links"].append([12, 11, 0, 6, 19, "IMAGE"])
    return workflow


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    outputs = {
        WORKFLOWS / "2026-08-09_H3_SPEED_FL2VA_Stock20_Advanced_EXP.json": build_fl2va(base),
        WORKFLOWS / "2026-08-09_H3_SPEED_Ref2VA_Stock20_Advanced_EXP.json": build_ref2va(base),
    }
    for path, workflow in outputs.items():
        path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
