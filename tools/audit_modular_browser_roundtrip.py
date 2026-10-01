"""Audit a native ComfyUI save of a modular workflow without executing it.

The current frontend adds widget sockets and may reorder target slot numbers.
Compare links by their named endpoint instead of comparing raw slot numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _nodes(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes = graph["nodes"]
    result = {node["id"]: node for node in nodes}
    if len(result) != len(nodes) or not all(type(key) is int for key in result):
        raise ValueError("duplicate or invalid node ID")
    return result


def _edges(graph: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> set[tuple[Any, ...]]:
    edges: set[tuple[Any, ...]] = set()
    for link in graph["links"]:
        if not isinstance(link, list) or len(link) != 6:
            raise ValueError("invalid link record")
        link_id, source_id, source_slot, target_id, target_slot, wire_type = link
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        if target.get("link") != link_id or link_id not in (source.get("links") or []):
            raise ValueError(f"link {link_id} lost endpoint ownership")
        if wire_type != "*" and (
            source["type"] not in (wire_type, "*")
            or target["type"] not in (wire_type, "*")
        ):
            raise ValueError(f"link {link_id} type mismatch")
        edges.add((link_id, source_id, source["name"], target_id, target["name"], wire_type))
    if len(edges) != len(graph["links"]):
        raise ValueError("duplicate semantic edge")
    return edges


def audit_roundtrip(
    original: dict[str, Any],
    saved: dict[str, Any],
    *,
    edits: dict[tuple[int, int], Any] | None = None,
    appended_widgets: dict[int, list[Any]] | None = None,
) -> dict[str, Any]:
    edits = edits or {}
    appended_widgets = appended_widgets or {}
    before, after = _nodes(original), _nodes(saved)
    if before.keys() != after.keys():
        raise ValueError("browser changed the node set")
    if _edges(original, before) != _edges(saved, after):
        raise ValueError("browser changed a named execution edge")

    checked_widgets = 0
    for node_id, old in before.items():
        new = after[node_id]
        if (old["type"], old.get("mode", 0)) != (new["type"], new.get("mode", 0)):
            raise ValueError(f"browser changed node type/mode: {node_id}")
        # Core may remove a title equal to its display name on native save.
        if new.get("title") is not None and old.get("title") not in (None, new["title"]):
            raise ValueError(f"browser changed a custom title: {node_id}")
        old_inputs = {pin["name"]: pin for pin in old.get("inputs", [])}
        new_inputs = {pin["name"]: pin for pin in new.get("inputs", [])}
        if len(new_inputs) != len(new.get("inputs", [])):
            raise ValueError(f"duplicate input name: {node_id}")
        for name, pin in old_inputs.items():
            other = new_inputs.get(name)
            if other is None or (pin["type"], pin.get("link")) != (
                other["type"], other.get("link")
            ):
                raise ValueError(f"browser changed input {node_id}.{name}")
        if any(pin.get("link") is not None and name not in old_inputs
               for name, pin in new_inputs.items()):
            raise ValueError(f"browser added a connected input: {node_id}")
        if [(pin["name"], pin["type"], pin.get("links") or [])
            for pin in old.get("outputs", [])] != [
                (pin["name"], pin["type"], pin.get("links") or [])
                for pin in new.get("outputs", [])
            ]:
            raise ValueError(f"browser changed outputs: {node_id}")

        expected = list(old.get("widgets_values") or [])
        for (edited_node, index), value in edits.items():
            if edited_node == node_id:
                if index >= len(expected):
                    raise ValueError(f"invalid permitted edit: {node_id}:{index}")
                expected[index] = value
        expected.extend(appended_widgets.get(node_id, []))
        actual = list(new.get("widgets_values") or [])
        if not _same_value(expected, actual):
            raise ValueError(f"browser changed widgets unexpectedly: {node_id}")
        named = new.get("widgets_values_named") or {}
        if expected and (len(named) != len(actual)
                         or not _same_value(list(named.values()), actual)):
            raise ValueError(f"browser lost named widgets: {node_id}")
        checked_widgets += len(actual)

    if any(node_id not in before for node_id, _ in edits) or any(
        node_id not in before for node_id in appended_widgets
    ):
        raise ValueError("permitted change references an absent node")
    return {
        "nodes": len(before),
        "named_edges": len(original["links"]),
        "widgets": checked_widgets,
        "edits": [f"{node_id}:{index}" for node_id, index in sorted(edits)],
        "browser_normalized_optional_widgets": sorted(appended_widgets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-full", type=Path, required=True)
    parser.add_argument("--saved-full", type=Path, required=True)
    parser.add_argument("--original-resume", type=Path, required=True)
    parser.add_argument("--saved-resume", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    files = {
        label: path for label, path in (
            ("original_full", args.original_full), ("saved_full", args.saved_full),
            ("original_resume", args.original_resume), ("saved_resume", args.saved_resume),
        )
    }
    graphs = {label: json.loads(path.read_bytes()) for label, path in files.items()}
    original_full = _nodes(graphs["original_full"])
    saved_full = _nodes(graphs["saved_full"])
    if original_full[27]["type"] != "MiniMaxH3PromptRelayPlanT8Advanced":
        raise ValueError("expected LOW Relay Plan 27 is absent")
    if saved_full[27]["widgets_values"][1] != (
        original_full[27]["widgets_values"][1] + " [QA browser edit]"
    ):
        raise ValueError("native UI edit was not preserved")
    if original_full[12]["type"] != "SaveVideo" or saved_full[12]["widgets_values_named"].get("codec") != "auto":
        raise ValueError("unexpected SaveVideo dynamic widget normalization")
    results = {
        "full": audit_roundtrip(
            graphs["original_full"], graphs["saved_full"],
            edits={(27, 1): saved_full[27]["widgets_values"][1]},
            appended_widgets={12: ["auto"]},
        ),
        "resume": audit_roundtrip(graphs["original_resume"], graphs["saved_resume"],
                                  appended_widgets={6: ["auto"]}),
    }
    record = {
        "status": "native_browser_save_roundtrip_pass_not_generation_acceptance",
        "workflows": results,
        "file_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                        for label, path in files.items()},
        "queued_inference": False,
        "limits": ["No GPU/pretrained model, media decode/save, or human quality check.",
                   "Only these two V2 candidates were browser-tested."],
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
