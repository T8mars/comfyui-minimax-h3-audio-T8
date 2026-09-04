from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json"
)
OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-09-05_H3_Face_Refine_Window_Manual_Review_Advanced_EXP.json"
)


def _socket(name: str, type_name: str, link: int) -> dict:
    return {"name": name, "type": type_name, "link": link}


def _output(name: str, type_name: str, links: list[int] | None = None) -> dict:
    return {"name": name, "type": type_name, "links": links}


def _properties(node_type: str) -> dict:
    return {
        "cnr_id": "minimax-h3-audio-T8",
        "Node name for S&R": node_type,
    }


def _window_plan_node() -> dict:
    node_type = "MiniMaxH3FaceRefineWindowPlanT8Advanced"
    return {
        "id": 26,
        "type": node_type,
        "title": "Plan one shot-local repair window; 0-based inclusive source frames",
        "pos": [880, 0],
        "size": [390, 590],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [_socket("base_frames", "IMAGE", 2)],
        "outputs": [
            _output("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN", [44]),
            _output("repair_mask_preview", "MASK"),
            _output("window_count", "INT"),
            _output("report_json", "STRING"),
        ],
        "properties": _properties(node_type),
        "widgets_values": [
            24.0,
            "0-23",
            "frames_inclusive",
            24,
            42,
            90,
            362,
            0.28,
            "reject",
            "edge_hold_exp",
            True,
        ],
    }


def _window_extract_node() -> dict:
    node_type = "MiniMaxH3FaceRefineWindowExtractT8Advanced"
    return {
        "id": 27,
        "type": node_type,
        "title": "Extract one legal 17n+5 window; audio padding is silence",
        "pos": [1320, 0],
        "size": [390, 380],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            _socket("base_frames", "IMAGE", 45),
            _socket("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN", 44),
            _socket("source_audio", "AUDIO", 12),
        ],
        "outputs": [
            _output("render_frames", "IMAGE", [16, 32]),
            _output("render_audio", "AUDIO", [46]),
            _output("window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING", [50]),
            _output("source_start_seconds", "FLOAT"),
            _output("render_duration_seconds", "FLOAT"),
            _output("accept_relative_ranges_json", "STRING"),
            _output("report_json", "STRING"),
        ],
        "properties": _properties(node_type),
        "widgets_values": [0, "edge_hold_exp"],
    }


def _manual_review_node() -> dict:
    node_type = "MiniMaxH3FaceRefineManualReviewT8Advanced"
    return {
        "id": 28,
        "type": node_type,
        "title": "Preview first; only explicit confirmed source ranges can be accepted",
        "pos": [5320, 0],
        "size": [430, 430],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            _socket("base_frames", "IMAGE", 47),
            _socket("candidate_window_frames", "IMAGE", 48),
            _socket("changed_mask", "MASK", 49),
            _socket("window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING", 50),
        ],
        "outputs": [
            _output("review_frames", "IMAGE"),
            _output("result_frames", "IMAGE", [41]),
            _output("accepted_change_mask", "MASK"),
            _output("rejected_change_mask", "MASK"),
            _output("accepted_frame_count", "INT"),
            _output("rejected_frame_count", "INT"),
            _output("report_json", "STRING"),
        ],
        "properties": _properties(node_type),
        "widgets_values": ["preview_only", "0-23", False, 2],
    }


def build() -> dict:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow = copy.deepcopy(workflow)
    nodes = {int(node["id"]): node for node in workflow["nodes"]}

    for node in nodes.values():
        x, y = node["pos"]
        if x >= 880 and node["type"] != "MarkdownNote":
            node["pos"] = [x + 880, y]
    nodes[21]["pos"] = [5800, 0]
    nodes[22]["pos"] = [6240, 0]
    nodes[25]["size"] = [7000, 760]
    nodes[25]["widgets_values"] = [
        "## 用途：只重绘明确选中的坏脸窗口，不再把整段正常脸一起生成\n\n"
        "1. `Window Plan`里的范围是**原视频0起算、首尾都包含**的帧号，例如`0-23`。默认只规划一个窗口；多个窗口改`window_index`后逐个串行跑，禁止并发抢显存。\n"
        "2. 89帧来源可显式选择`edge_hold_exp`形成90帧H3窗口；补出的图像只作上下文，音频对应位置补0，padding永远不会被回贴。跨镜头范围直接拒绝。\n"
        "3. Parity分支继续使用MANUAL512、crop 2.5、relative_to_clip 0.8/0.35、21/51平滑、24/24 face-only stitch和锁定窗口原声。H3解出的音频丢弃。\n"
        "4. `Manual Review`默认`preview_only`，完整原片逐位不变。看完`review_frames`后才能改成`accept_selected`并打开`confirm_accept`；只能接受计划中的原始坏帧或其子范围，context与padding不能混入。\n"
        "5. 最终`CreateVideo`始终连接完整来源音轨，不使用窗口音频。输出是待审候选，不保证身份、画质或通用16GB安全；保存到新文件，不覆盖原片。"
    ]

    nodes[26] = _window_plan_node()
    nodes[27] = _window_extract_node()
    nodes[28] = _manual_review_node()

    links = [list(link) for link in workflow["links"]]
    by_id = {int(link[0]): link for link in links}
    by_id[2][3:5] = [26, 0]
    by_id[12][3:5] = [27, 2]
    by_id[32][1:3] = [27, 0]
    by_id[41][1:3] = [28, 1]
    links.extend(
        [
            [44, 26, 0, 27, 1, "H3_T8_FACE_REFINE_WINDOW_PLAN"],
            [45, 2, 0, 27, 0, "IMAGE"],
            [46, 27, 1, 11, 6, "AUDIO"],
            [47, 2, 0, 28, 0, "IMAGE"],
            [48, 23, 0, 28, 1, "IMAGE"],
            [49, 20, 1, 28, 2, "MASK"],
            [50, 27, 2, 28, 3, "H3_T8_FACE_REFINE_WINDOW_MAPPING"],
            [51, 27, 0, 4, 0, "IMAGE"],
        ]
    )

    for node in nodes.values():
        for output in node.get("outputs", []):
            output["links"] = None
        for input_socket in node.get("inputs", []):
            input_socket["link"] = None
    for link in links:
        link_id, source_id, source_slot, target_id, target_slot, _ = link
        source = nodes[int(source_id)]["outputs"][int(source_slot)]
        if source["links"] is None:
            source["links"] = []
        source["links"].append(int(link_id))
        nodes[int(target_id)]["inputs"][int(target_slot)]["link"] = int(link_id)

    old_order = [int(node["id"]) for node in workflow["nodes"]]
    insert_at = old_order.index(4)
    ordered_ids = old_order[:insert_at] + [26, 27] + old_order[insert_at:]
    ordered_ids.insert(ordered_ids.index(21), 28)
    workflow["nodes"] = [nodes[node_id] for node_id in ordered_ids]
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    workflow["links"] = sorted(links, key=lambda item: int(item[0]))
    workflow["last_node_id"] = 28
    workflow["last_link_id"] = 51
    return workflow


def main() -> None:
    OUTPUT.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
