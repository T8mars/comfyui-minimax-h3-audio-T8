from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SOURCE = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-09-05_H3_Face_Refine_Window_Studio_Serial_Advanced_EXP.json"
)
FRONTEND_OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-09-05_H3_Face_Refine_Window_Studio_Compose_Advanced_EXP.json"
)
API_OUTPUT = (
    ROOT
    / "tests"
    / "fixtures"
    / "api"
    / "face_refine_window_studio_compose_advanced_api.json"
)


def _rebuild_links(nodes: dict[int, dict], links: list[list]) -> None:
    for node in nodes.values():
        for output in node.get("outputs", []):
            output["links"] = None
        for input_socket in node.get("inputs", []):
            input_socket["link"] = None
    for link in links:
        link_id, source_id, source_slot, target_id, target_slot, _type = link
        output = nodes[int(source_id)]["outputs"][int(source_slot)]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(int(link_id))
        nodes[int(target_id)]["inputs"][int(target_slot)]["link"] = int(link_id)


def build_frontend() -> dict:
    source = json.loads(FRONTEND_SOURCE.read_text(encoding="utf-8"))
    keep = {1, 2, 21, 22, 25, 26, 30}
    nodes = {
        int(node["id"]): copy.deepcopy(node)
        for node in source["nodes"]
        if int(node["id"]) in keep
    }
    nodes[1]["title"] = "Load the exact source bound to this Studio manifest"
    nodes[21]["title"] = "Mux the complete untouched original soundtrack"
    nodes[22]["title"] = "Save recovered final video; never overwrite source"
    nodes[22]["widgets_values"][0] = "MiniMaxH3/face_refine_window_studio_composed"
    nodes[25]["title"] = "Face Refine Studio crash-safe composition"
    nodes[25]["widgets_values"] = [
        "## 只合成已决定的窗口，不加载H3\n\n"
        "当所有窗口已经接受/拒绝，或最后一次提交后在保存前崩溃时，使用本工作流。"
        "必须选择与Studio完全相同的源视频、repair_ranges、计划参数和studio_id；任一来源或plan hash不同都会拒绝。\n\n"
        "Compose只读取lossless accepted overlay。pending/rejected区域逐像素来自原片，最终CreateVideo仍接完整原音轨。"
    ]
    nodes[26]["title"] = "Recreate the exact source-bound plan used by Studio"
    nodes[30]["title"] = "Compose durable accepted overlays without H3 generation"
    nodes[30]["inputs"] = [socket for socket in nodes[30]["inputs"] if socket["name"] != "commit_barrier"]
    nodes[30]["pos"] = [1320, 260]
    nodes[21]["pos"] = [1760, 260]
    nodes[22]["pos"] = [2200, 260]
    nodes[25]["pos"] = [0, -420]
    nodes[25]["size"] = [2600, 300]

    links = [
        [1, 1, 0, 2, 0, "VIDEO"],
        [2, 2, 0, 26, 0, "IMAGE"],
        [3, 2, 0, 30, 0, "IMAGE"],
        [4, 26, 0, 30, 1, "H3_T8_FACE_REFINE_WINDOW_PLAN"],
        [5, 30, 0, 21, 0, "IMAGE"],
        [6, 2, 1, 21, 1, "AUDIO"],
        [7, 21, 0, 22, 0, "VIDEO"],
    ]
    _rebuild_links(nodes, links)
    order = [1, 2, 26, 30, 21, 22, 25]
    workflow = copy.deepcopy(source)
    workflow["id"] = "8c609682-fceb-46e4-a114-4f667b7789e2"
    workflow["nodes"] = [nodes[node_id] for node_id in order]
    for index, node in enumerate(workflow["nodes"]):
        node["order"] = index
    workflow["links"] = links
    workflow["last_node_id"] = max(nodes)
    workflow["last_link_id"] = max(link[0] for link in links)
    workflow["extra"]["workflow_title"] = "H3 Face Refine Window Studio Compose"
    return workflow


def build_api() -> dict:
    return {
        "1": {
            "inputs": {"file": "replace_with_exact_24fps_source.mp4"},
            "class_type": "LoadVideo",
            "_meta": {"title": "Load the exact source bound to this Studio manifest"},
        },
        "2": {
            "inputs": {"video": ["1", 0]},
            "class_type": "GetVideoComponents",
            "_meta": {"title": "Complete source frames and original soundtrack"},
        },
        "3": {
            "inputs": {
                "fps": 24.0,
                "repair_ranges": "0-23",
                "range_mode": "frames_inclusive",
                "context_before_frames": 24,
                "context_after_frames": 42,
                "min_render_frames": 90,
                "max_render_frames": 362,
                "scene_cut_threshold": 0.28,
                "overlap_policy": "reject",
                "short_shot_policy": "edge_hold_exp",
                "enabled": True,
                "base_frames": ["2", 0],
            },
            "class_type": "MiniMaxH3FaceRefineWindowPlanT8Advanced",
            "_meta": {"title": "Recreate the exact source-bound plan used by Studio"},
        },
        "4": {
            "inputs": {
                "studio_id": "face_refine_project_01",
                "base_frames": ["2", 0],
                "window_plan": ["3", 0],
            },
            "class_type": "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced",
            "_meta": {"title": "Compose durable accepted overlays without H3 generation"},
        },
        "5": {
            "inputs": {
                "images": ["4", 0],
                "fps": 24.0,
                "audio": ["2", 1],
                "bit_depth": 8,
            },
            "class_type": "CreateVideo",
            "_meta": {"title": "Mux the complete untouched original soundtrack"},
        },
        "6": {
            "inputs": {
                "video": ["5", 0],
                "filename_prefix": "MiniMaxH3/face_refine_window_studio_composed",
                "format": "mp4",
                "codec": "h264",
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Save recovered final video; never overwrite source"},
        },
    }


def main() -> None:
    FRONTEND_OUTPUT.write_text(
        json.dumps(build_frontend(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    API_OUTPUT.write_text(
        json.dumps(build_api(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(FRONTEND_OUTPUT)
    print(API_OUTPUT)


if __name__ == "__main__":
    main()
