from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_native_widget_inputs(workflow: dict[str, Any]) -> dict[str, Any]:
    """Serialize widgets like native ComfyUI while preserving linked widget sockets.

    Native frontend workflows keep ordinary widget values in ``widgets_values``.
    Only widgets converted to an input socket belong in ``inputs``.  Some strict
    third-party importers interpret an unlinked widget entry as a missing required
    socket, so remove those entries and repair every affected target slot.
    """

    nodes = {
        node.get("id"): node
        for node in workflow.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), int)
    }
    changed_nodes: list[str] = []
    target_slots: dict[tuple[int, int], int] = {}

    for node_id, node in nodes.items():
        saved_inputs = node.get("inputs")
        if not isinstance(saved_inputs, list):
            continue
        normalized: list[dict[str, Any]] = []
        removed = False
        for item in saved_inputs:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            if "widget" in item and item.get("link") is None:
                removed = True
                continue
            new_slot = len(normalized)
            normalized.append(item)
            link_id = item.get("link")
            if isinstance(link_id, int):
                target_slots[(node_id, link_id)] = new_slot
        if removed:
            node["inputs"] = normalized
            changed_nodes.append(f"{node_id}:{node.get('type', '')}")

    repaired_links: list[int] = []
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 5:
            continue
        link_id, target_id = link[0], link[3]
        if not isinstance(link_id, int) or not isinstance(target_id, int):
            continue
        new_slot = target_slots.get((target_id, link_id))
        if new_slot is not None and link[4] != new_slot:
            link[4] = new_slot
            repaired_links.append(link_id)

    return {
        "normalized": changed_nodes,
        "repaired_links": repaired_links,
    }


def _workflow_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.rglob("*.json")))
        else:
            result.append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize ComfyUI frontend widget inputs for strict importers."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    changed = 0
    for path in _workflow_paths(args.paths):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            continue
        report = normalize_native_widget_inputs(workflow)
        if not report["normalized"]:
            continue
        changed += 1
        print(
            f"{path}: normalized {len(report['normalized'])} node(s), "
            f"repaired {len(report['repaired_links'])} link slot(s)"
        )
        if args.write:
            path.write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(f"changed_workflows={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
