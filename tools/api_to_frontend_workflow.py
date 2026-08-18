#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import urllib.request
import uuid

try:
    from .repair_frontend_workflow_order import repair_workflow
except ImportError:  # Direct script execution puts tools/ on sys.path.
    from repair_frontend_workflow_order import repair_workflow


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_link(value, prompt: dict) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and str(value[0]) in prompt
        and isinstance(value[1], int)
    )


def _input_type(info: dict, key: str) -> str:
    for section in ("required", "optional", "hidden"):
        value = info.get("input", {}).get(section, {}).get(key)
        if isinstance(value, list) and value:
            return str(value[0])
    if "." in key:
        group = key.split(".", 1)[0]
        value = info.get("input", {}).get("required", {}).get(group) or info.get("input", {}).get("optional", {}).get(group)
        if isinstance(value, list) and value and value[0] == "COMFY_AUTOGROW_V3":
            template = value[1].get("template", {}).get("input", {}).get("required", {})
            if template:
                first = next(iter(template.values()))
                if isinstance(first, list) and first:
                    return str(first[0])
    return "*"


def _levels(prompt: dict) -> dict[str, int]:
    memo = {}

    def visit(node_id: str, stack: set[str]) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in stack:
            return 0
        parents = [str(value[0]) for value in prompt[node_id].get("inputs", {}).values() if _is_link(value, prompt)]
        memo[node_id] = 0 if not parents else max(visit(parent, stack | {node_id}) for parent in parents) + 1
        return memo[node_id]

    for node_id in prompt:
        visit(node_id, set())
    return memo


def convert(prompt: dict, object_info: dict, title: str) -> dict:
    levels = _levels(prompt)
    rows = defaultdict(int)
    id_map = {node_id: index + 1 for index, node_id in enumerate(prompt)}
    frontend_nodes = []
    node_lookup = {}
    for order, (node_id, source) in enumerate(prompt.items()):
        class_type = source["class_type"]
        info = object_info[class_type]
        level = levels[node_id]
        row = rows[level]
        rows[level] += 1
        inputs = []
        widgets = []
        for key, value in source.get("inputs", {}).items():
            input_type = _input_type(info, key)
            if _is_link(value, prompt):
                inputs.append({"name": key, "type": input_type, "link": None})
            else:
                inputs.append({"name": key, "type": input_type, "widget": {"name": key}, "link": None})
                widgets.append(value)
                if key in {"seed", "noise_seed"}:
                    widgets.append("fixed")
        outputs = [
            {"name": name, "type": output_type, "links": None}
            for name, output_type in zip(info.get("output_name", []), info.get("output", []))
        ]
        node = {
            "id": id_map[node_id],
            "type": class_type,
            "title": source.get("_meta", {}).get("title", info.get("display_name", class_type)),
            "pos": [level * 440, row * 300],
            "size": [390, max(110, 80 + 26 * len(inputs) + 18 * len(widgets))],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {
                "cnr_id": "minimax-h3-audio-T8" if class_type.startswith("MiniMaxH3") else "comfy-core",
                "Node name for S&R": class_type,
            },
            "widgets_values": widgets,
        }
        frontend_nodes.append(node)
        node_lookup[node_id] = node

    links = []
    link_id = 0
    for target_id, source in prompt.items():
        target = node_lookup[target_id]
        for target_slot, (key, value) in enumerate(source.get("inputs", {}).items()):
            if not _is_link(value, prompt):
                continue
            source_id, source_slot = str(value[0]), int(value[1])
            origin = node_lookup[source_id]
            if source_slot >= len(origin["outputs"]):
                raise ValueError(f"{source_id} output slot {source_slot} does not exist")
            link_id += 1
            link_type = target["inputs"][target_slot]["type"]
            target["inputs"][target_slot]["link"] = link_id
            if origin["outputs"][source_slot]["links"] is None:
                origin["outputs"][source_slot]["links"] = []
            origin["outputs"][source_slot]["links"].append(link_id)
            links.append([link_id, origin["id"], source_slot, target["id"], target_slot, link_type])

    workflow = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": len(frontend_nodes),
        "last_link_id": link_id,
        "nodes": frontend_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [120, 120]}, "workflow_title": title},
        "version": 0.4,
    }
    result = repair_workflow(workflow, object_info, force=True)
    if result["skipped"]:
        raise ValueError(
            "Could not serialize frontend input order: " + "; ".join(result["skipped"])
        )
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a ComfyUI API prompt into an importable frontend workflow.")
    parser.add_argument("api_prompt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--title", default="MiniMax H3 T8 example")
    args = parser.parse_args()
    prompt = json.loads(args.api_prompt.read_text(encoding="utf-8"))
    info = _get_json(f"{args.server.rstrip('/')}/object_info")
    missing = sorted({node["class_type"] for node in prompt.values()} - set(info))
    if missing:
        raise ValueError(f"server is missing nodes: {missing}")
    workflow = convert(prompt, info, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
