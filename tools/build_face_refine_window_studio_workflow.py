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
    / "2026-09-05_H3_Face_Refine_Window_Manual_Review_Advanced_EXP.json"
)
FRONTEND_OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-09-05_H3_Face_Refine_Window_Studio_Serial_Advanced_EXP.json"
)
API_SOURCE = ROOT / "tests" / "fixtures" / "api" / "face_refine_window_advanced_api.json"
API_OUTPUT = (
    ROOT / "tests" / "fixtures" / "api" / "face_refine_window_studio_advanced_api.json"
)


def _socket(name: str, type_name: str, link: int, *, widget: bool = False) -> dict:
    result = {"name": name, "type": type_name, "link": link}
    if widget:
        result["widget"] = {"name": name}
    return result


def _output(name: str, type_name: str, links: list[int] | None = None) -> dict:
    return {"name": name, "type": type_name, "links": links}


def _properties(node_type: str, *, core: bool = False) -> dict:
    return {
        "cnr_id": "comfy-core" if core else "minimax-h3-audio-T8",
        "Node name for S&R": node_type,
    }


def _start_node() -> dict:
    node_type = "MiniMaxH3FaceRefineWindowStudioStartT8Advanced"
    return {
        "id": 28,
        "type": node_type,
        "title": "Preview first; background mode queues only after an explicit decision",
        "pos": [1320, -610],
        "size": [430, 360],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [_socket("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN", 52)],
        "outputs": [
            _output("window_index", "INT", [53]),
            _output("chain_id", "STRING"),
            _output("auto_continue", "BOOLEAN", [60]),
            _output("job_id", "STRING", [59]),
            _output("manifest_path", "STRING"),
            _output("complete", "BOOLEAN"),
            _output("background_state_json", "STRING"),
            _output("report_json", "STRING"),
        ],
        "properties": _properties(node_type),
        "widgets_values": [
            "face_refine_project_01",
            "review_only",
            1,
            2.0,
            "clear_execution_cache",
        ],
    }


def _commit_node() -> dict:
    node_type = "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced"
    return {
        "id": 29,
        "type": node_type,
        "title": "Explicit human accept/reject becomes an immutable atomic decision",
        "pos": [5320, 0],
        "size": [470, 520],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            _socket("base_frames", "IMAGE", 54),
            _socket("candidate_window_frames", "IMAGE", 55),
            _socket("changed_mask", "MASK", 56),
            _socket("window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING", 57),
            _socket("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN", 58),
            _socket("job_id", "STRING", 59),
            _socket("auto_continue", "BOOLEAN", 60),
        ],
        "outputs": [
            _output("review_frames", "IMAGE", [65]),
            _output("current_result_frames", "IMAGE"),
            _output("accepted_change_mask", "MASK"),
            _output("rejected_change_mask", "MASK"),
            _output("committed", "BOOLEAN"),
            _output("manifest_path", "STRING"),
            _output("resolved_window_count", "INT"),
            _output("complete", "BOOLEAN"),
            _output("background_state_json", "STRING"),
            _output("report_json", "STRING", [61]),
        ],
        "properties": _properties(node_type),
        "widgets_values": [
            "face_refine_project_01",
            "preview_only",
            "",
            False,
            2,
        ],
    }


def _compose_node() -> dict:
    node_type = "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced"
    return {
        "id": 30,
        "type": node_type,
        "title": "Compose accepted overlays; rejected/pending windows remain exact source",
        "pos": [5820, 0],
        "size": [430, 270],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            _socket("base_frames", "IMAGE", 62),
            _socket("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN", 63),
            _socket("commit_barrier", "STRING", 61),
        ],
        "outputs": [
            _output("result_frames", "IMAGE", [64]),
            _output("combined_accepted_mask", "MASK"),
            _output("complete", "BOOLEAN"),
            _output("report_json", "STRING"),
        ],
        "properties": _properties(node_type),
        "widgets_values": ["face_refine_project_01"],
    }


def _preview_node() -> dict:
    return {
        "id": 31,
        "type": "PreviewImage",
        "title": "Review: source window on the left, candidate on the right",
        "pos": [5820, 340],
        "size": [430, 300],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [_socket("images", "IMAGE", 65)],
        "outputs": [],
        "properties": _properties("PreviewImage", core=True),
        "widgets_values": [],
    }


def _rebuild_links(nodes: dict[int, dict], links: list[list]) -> None:
    for node in nodes.values():
        for output in node.get("outputs", []):
            output["links"] = None
        for input_socket in node.get("inputs", []):
            input_socket["link"] = None
    for link in links:
        link_id, source_id, source_slot, target_id, target_slot, _type = link
        source = nodes[int(source_id)]["outputs"][int(source_slot)]
        if source["links"] is None:
            source["links"] = []
        source["links"].append(int(link_id))
        nodes[int(target_id)]["inputs"][int(target_slot)]["link"] = int(link_id)


def build_frontend() -> dict:
    workflow = copy.deepcopy(json.loads(FRONTEND_SOURCE.read_text(encoding="utf-8")))
    nodes = {int(node["id"]): node for node in workflow["nodes"] if int(node["id"]) != 28}
    links = [
        list(link)
        for link in workflow["links"]
        if int(link[1]) != 28 and int(link[3]) != 28
    ]
    links = [link for link in links if int(link[0]) != 41]

    # A connected widget becomes a named input socket. Keep source_audio after it and update the
    # existing audio link target slot accordingly.
    extract = nodes[27]
    extract["inputs"] = [
        extract["inputs"][0],
        extract["inputs"][1],
        _socket("window_index", "INT", 53, widget=True),
        extract["inputs"][2],
    ]
    extract["widgets_values"] = ["edge_hold_exp"]
    for link in links:
        if int(link[0]) == 12:
            link[4] = 3

    nodes[28] = _start_node()
    nodes[29] = _commit_node()
    nodes[30] = _compose_node()
    nodes[31] = _preview_node()
    nodes[21]["pos"] = [6300, 0]
    nodes[22]["pos"] = [6740, 0]
    nodes[25]["size"] = [7440, 820]
    nodes[25]["widgets_values"] = [
        "## 多窗口串行 Face Refine Studio（Advanced / EXP）\n\n"
        "1. `Window Plan`一次写入多个0起算、首尾都包含的坏脸范围；Studio manifest绑定原片和plan hash。\n"
        "2. 第一次保持`review_only + preview_only`，看Preview里左原片/右候选，不会排队、接受或覆盖任何内容。\n"
        "3. 决定后把Commit改为`accept_selected`并打开确认，或明确选`reject`。需要自动进入下一个窗口时，再把Start改为`explicit_accept_and_continue`；每次只排一个任务，绝不并发跑H3。\n"
        "4. 已接受/拒绝的窗口不可回退、不可重做；崩溃后重新运行会从第一个未决窗口恢复。接受内容以lossless crop overlay保存，source文件从不改写。\n"
        "5. Compose只应用已接受overlay，pending/rejected保持原片；CreateVideo始终接完整原音频。功能仍是EXP，人工检查与512MiB显存余量门不能跳过。"
    ]
    links.extend(
        [
            [52, 26, 0, 28, 0, "H3_T8_FACE_REFINE_WINDOW_PLAN"],
            [53, 28, 0, 27, 2, "INT"],
            [54, 2, 0, 29, 0, "IMAGE"],
            [55, 23, 0, 29, 1, "IMAGE"],
            [56, 20, 1, 29, 2, "MASK"],
            [57, 27, 2, 29, 3, "H3_T8_FACE_REFINE_WINDOW_MAPPING"],
            [58, 26, 0, 29, 4, "H3_T8_FACE_REFINE_WINDOW_PLAN"],
            [59, 28, 3, 29, 5, "STRING"],
            [60, 28, 2, 29, 6, "BOOLEAN"],
            [61, 29, 9, 30, 2, "STRING"],
            [62, 2, 0, 30, 0, "IMAGE"],
            [63, 26, 0, 30, 1, "H3_T8_FACE_REFINE_WINDOW_PLAN"],
            [64, 30, 0, 21, 0, "IMAGE"],
            [65, 29, 0, 31, 0, "IMAGE"],
        ]
    )
    _rebuild_links(nodes, links)
    old_order = [int(node["id"]) for node in workflow["nodes"] if int(node["id"]) != 28]
    old_order.insert(old_order.index(27), 28)
    old_order.insert(old_order.index(21), 29)
    old_order.insert(old_order.index(21), 30)
    old_order.insert(old_order.index(21), 31)
    workflow["nodes"] = [nodes[node_id] for node_id in old_order]
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    workflow["links"] = sorted(links, key=lambda item: int(item[0]))
    workflow["last_node_id"] = 31
    workflow["last_link_id"] = 65
    return workflow


def build_api() -> dict:
    graph = copy.deepcopy(json.loads(API_SOURCE.read_text(encoding="utf-8")))
    graph["5"]["inputs"]["window_index"] = ["28", 0]
    graph.pop("24")
    graph["25"]["inputs"]["images"] = ["30", 0]
    graph["28"] = {
        "inputs": {
            "studio_id": "face_refine_project_01",
            "execution_mode": "review_only",
            "max_retries": 1,
            "retry_delay_seconds": 2.0,
            "release_policy": "clear_execution_cache",
            "window_plan": ["4", 0],
        },
        "class_type": "MiniMaxH3FaceRefineWindowStudioStartT8Advanced",
        "_meta": {"title": "Resume the first unresolved window without automatic acceptance"},
    }
    graph["29"] = {
        "inputs": {
            "studio_id": "face_refine_project_01",
            "decision": "preview_only",
            "accepted_subranges": "",
            "confirm_accept": False,
            "edge_fade_frames": 2,
            "base_frames": ["2", 0],
            "candidate_window_frames": ["23", 0],
            "changed_mask": ["22", 1],
            "window_mapping": ["5", 2],
            "window_plan": ["4", 0],
            "job_id": ["28", 3],
            "auto_continue": ["28", 2],
        },
        "class_type": "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced",
        "_meta": {"title": "Preview first; an explicit durable decision is required"},
    }
    graph["30"] = {
        "inputs": {
            "studio_id": "face_refine_project_01",
            "base_frames": ["2", 0],
            "window_plan": ["4", 0],
            "commit_barrier": ["29", 9],
        },
        "class_type": "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced",
        "_meta": {"title": "Compose only immutable accepted overlays"},
    }
    graph["31"] = {
        "inputs": {"images": ["29", 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "Source on the left, candidate on the right"},
    }
    return graph


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
