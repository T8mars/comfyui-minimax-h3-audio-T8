"""Create three explicit postprocess modules from actual Core schemas.

No sampling, silent workflow migration, new publisher, or inferred execution
order. ImageCropV2's socketless BoundingBox uses the native saved widget shape.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import urllib.request

try:
    from .api_to_frontend_workflow import convert
    from .check_windows_paths import validate_paths
except ImportError:
    from api_to_frontend_workflow import convert
    from check_windows_paths import validate_paths


POST = "MiniMaxH3PostprocessSaveEXPT8"
SAFE = "MiniMaxH3SafeAVSaveT8Advanced"
FILENAMES = ("Full_Master_Crop_EXP.json", "Cold_Master_Crop_EXP.json", "Cold_Master_Contain_EXP.json")


def make_workflow(info, *, cold=True, contain=False, master_file="SELECT_SAVED_MASTER.mp4",
                  width=1920, height=1080, y=4, confirm=False, expected_sha=""):
    """Only explicit full/cold pixel branches. Old schemas remain untouched."""
    crop_spec = info["ImageCropV2"]["input"]["required"]["crop_region"]
    if (crop_spec[0] != "BOUNDING_BOX" or crop_spec[1].get("socketless") is not True
            or crop_spec[1].get("component") != "ImageCrop"):
        raise ValueError("Actual Core crop component differs; inspect native serialization first")
    source = {"class_type": "LoadVideo", "inputs": {"file": master_file}} if cold else {
        "class_type": SAFE, "inputs": {"filename_prefix": "MiniMaxH3/Radar/master", "crf": 18}}
    region = {"x": 0, "y": y, "width": width, "height": height}
    pixels = {"class_type": "ResizeAndPadImage", "inputs": {
        "image": ["2", 0], "target_width": width, "target_height": height,
        "padding_color": "black", "interpolation": "lanczos"}} if contain else {
        "class_type": "ImageCropV2", "inputs": {"image": ["2", 0]}}
    graph = {"1": source, "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
             "3": pixels, "4": {"class_type": POST, "inputs": {
                 "master_video": ["1", 0], "processed_frames": ["3", 0],
                 "filename_prefix": "MiniMaxH3/Radar/contain" if contain else "MiniMaxH3/Radar/crop",
                 "confirm_postprocess": confirm, "crf": 18, "expected_master_sha256": expected_sha}}}
    title = "Cold Master Contain" if contain else ("Cold Master Crop" if cold else "Full Master Crop")
    workflow = convert(graph, info, title + " EXP")
    by_id = {node["id"]: node for node in workflow["nodes"]}
    if not cold:
        # Deliberate attachment module, not a fabricated standalone model graph.
        # Connect complete final RGB and AUDIO after every MASK/Face paste.
        if [item["name"] for item in by_id[1]["inputs"]] != ["images", "audio"]:
            raise ValueError("Safe master schema differs; do not infer attachment sockets")
    if not contain:
        # Native legacy-canvas fallback saves the region plus four controls.
        # It is still one socketless BoundingBox widget, not four API inputs.
        node = by_id[3]
        node["inputs"] = [{"name": "image", "type": "IMAGE", "link": node["inputs"][0]["link"]},
                          {"name": "crop_region", "type": "BOUNDING_BOX",
                           "widget": {"name": "crop_region"}, "link": None}]
        node["widgets_values"] = [deepcopy(region), region["x"], region["y"], width, height]
        node["widgets_values_named"] = {"crop_region": deepcopy(region), **region}
        graph["3"]["inputs"]["crop_region"] = region
    locations = {1: [0, 0], 2: [430, 0], 3: [860, 0], 4: [1290, 0]}
    for identifier, node in by_id.items():
        node["pos"] = locations[identifier]
        node["size"] = [380, 490 if identifier in (1, 3) else 380]
    text = ("COLD：只读取你选择的完整已保存 master；无模型／采样／旧 Stage 依赖。\n"
            "LoadVideo 需选择自己的文件，默认占位不可直接运行。\n" if cold else
            "FULL 附加模块：把完整最终 RGB 与 AUDIO 接入左侧安全保存节点。\n"
            "必须接在全部 MASK／Face 回贴之后；不接 partial LOW 或中间 x0。\n")
    text += ("CONTAIN：保留全画面，等比例缩放后黑色留边；明确改变几何，不是无损像素。\n" if contain else
             "CROP：默认 1920×1088 →1920×1080，x0/y4；会裁掉上下内容。\n"
             "按 master 的实际尺寸调整控件，不修改 latent 或 MASK 坐标。\n")
    text += ("已有 master 是真实接线依赖，不靠节点摆放保证先保存。\n"
             "完整 SDR、零起点 24fps，帧数不变；复制 master 音频包并核 PCM/PTS。\n"
             "确认开关默认 false；确认后只写新文件，不重跑生成或覆盖 master。\n"
             "失败：postprocess_failed，成片出口阻断；master_video 独立出口保留原片。\n"
             "预览和回执不代表人工画质／听感通过。")
    workflow["nodes"].append({"id": 5, "type": "Note", "title": title + " · 接线与边界",
        "pos": [0, 550], "size": [900, 290], "flags": {}, "order": 4, "mode": 0,
        "inputs": [], "outputs": [], "properties": {"text": text}, "widgets_values": [text]})
    workflow["last_node_id"] = 5
    workflow["extra"]["ds"] = {"scale": .65, "offset": [90, 180]}
    workflow["extra"]["radar_r6_delivery"] = {"schema": "t8.r6.delivery.template/v1",
        "mode": "cold" if cold else "full_attachment", "pixel_policy": "contain" if contain else "crop",
        "automatic_accept": False, "new_sampling_NFE": 0}
    return workflow, graph


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas-master", help="Explicit input-directory video for one CPU canvas variant")
    parser.add_argument("--canvas-sha", default="")
    args = parser.parse_args()
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    root = Path(__file__).resolve().parents[1]
    planned = [(FILENAMES[0], {"cold": False}), (FILENAMES[1], {}),
               (FILENAMES[2], {"contain": True})]
    if args.canvas_master:
        if len(args.canvas_sha) != 64 or any(c not in "0123456789abcdef" for c in args.canvas_sha):
            raise ValueError("Canvas variant needs an explicit exact master SHA256")
        planned.append(("R6_Cold_Crop_Test.json", {"master_file": args.canvas_master,
            "width": 512, "height": 280, "y": 4, "confirm": True, "expected_sha": args.canvas_sha}))
    for name, _ in planned:
        destination = args.output / name
        relative = destination.resolve().relative_to(root).as_posix() if destination.resolve().is_relative_to(root) else name
        errors = validate_paths([relative])
        if errors or destination.exists():
            raise ValueError(errors or f"Refusing to overwrite {destination}")
    for name, settings in planned:
        value, _ = make_workflow(info, **settings)
        write_new(args.output / name, value)
        print(args.output / name)


if __name__ == "__main__":
    main()
