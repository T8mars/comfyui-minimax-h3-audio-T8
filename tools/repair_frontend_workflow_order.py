#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any
import urllib.request

try:
    from .frontend_workflow_compat import normalize_native_widget_inputs
except ImportError:  # Direct script execution puts tools/ on sys.path.
    from frontend_workflow_compat import normalize_native_widget_inputs


PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}
SEED_WIDGETS = {"seed", "noise_seed"}
SEED_CONTROLS = {"fixed", "increment", "decrement", "randomize"}


def _is_widget_spec(spec: Any) -> bool:
    if not isinstance(spec, list) or not spec:
        return False
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return not options.get("forceInput") and (
        isinstance(spec[0], list) or spec[0] in PRIMITIVE_TYPES
    )


def _frontend_type(spec: Any) -> str:
    if not isinstance(spec, list) or not spec:
        return "*"
    return "COMBO" if isinstance(spec[0], list) else str(spec[0])


def _default_value(spec: Any) -> Any:
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in options:
        return deepcopy(options["default"])
    if isinstance(kind, list):
        return deepcopy(kind[0]) if kind else ""
    return {
        "STRING": "",
        "INT": 0,
        "FLOAT": 0.0,
        "BOOLEAN": False,
        "COMBO": "",
    }.get(kind)


def _autogrow_matches(
    group_name: str,
    spec: Any,
    saved_names: list[str],
) -> list[tuple[str, Any, bool]]:
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    template = options.get("template", {})
    prefix = str(template.get("prefix", ""))
    template_inputs = template.get("input", {}).get("required", {})
    template_spec = next(iter(template_inputs.values()), ["*"])
    dotted = f"{group_name}."

    def belongs_to_group(name: str) -> bool:
        if name.startswith(dotted):
            local_name = name[len(dotted) :]
        elif "." not in name:
            local_name = name
        else:
            return False
        if not prefix or not local_name.startswith(prefix):
            return False
        return local_name[len(prefix) :].isdigit()

    matches = [
        name
        for name in saved_names
        if belongs_to_group(name)
    ]

    def sort_key(name: str) -> tuple[str, int, str]:
        tail = name.rsplit("_", 1)[-1]
        return (name.rsplit("_", 1)[0], int(tail) if tail.isdigit() else 1 << 30, name)

    return [(name, template_spec, True) for name in sorted(matches, key=sort_key)]


def _format_dynamic_matches(
    spec: Any,
    saved_names: list[str],
) -> list[tuple[str, Any, bool]]:
    """Recover VHS-style widget inputs selected by a parent format combo."""

    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    formats = options.get("formats")
    if not isinstance(formats, dict):
        return []
    saved = set(saved_names)
    candidates: list[tuple[int, list[list[Any]]]] = []
    for dynamic in formats.values():
        if not isinstance(dynamic, list):
            continue
        names = {
            str(item[0])
            for item in dynamic
            if isinstance(item, list) and len(item) >= 2
        }
        if saved and saved.issubset(names):
            candidates.append((len(names - saved), dynamic))
    if not candidates:
        return []
    dynamic = min(candidates, key=lambda item: item[0])[1]
    entries: list[tuple[str, Any, bool]] = []
    for item in dynamic:
        if not isinstance(item, list) or len(item) < 2:
            continue
        name = str(item[0])
        if name not in saved:
            continue
        dynamic_spec = [item[1]]
        if len(item) > 2 and isinstance(item[2], dict):
            dynamic_spec.append(item[2])
        entries.append((name, dynamic_spec, False))
    return entries


def canonical_entries(
    info: dict[str, Any],
    saved_names: list[str],
) -> tuple[list[tuple[str, Any, bool]], list[str]]:
    entries: list[tuple[str, Any, bool]] = []
    reserved_names = {
        name
        for section in ("required", "optional")
        for name in info.get("input", {}).get(section, {})
    }
    consumed: set[str] = set()
    for section in ("required", "optional"):
        for name, spec in info.get("input", {}).get(section, {}).items():
            if isinstance(spec, list) and spec and spec[0] == "COMFY_AUTOGROW_V3":
                dynamic_names = [item for item in saved_names if item not in reserved_names]
                expanded = _autogrow_matches(name, spec, dynamic_names)
                entries.extend(expanded)
                consumed.update(item[0] for item in expanded)
                continue
            entries.append((name, spec, section == "optional"))
            consumed.add(name)
            dynamic_names = [item for item in saved_names if item not in reserved_names]
            expanded = _format_dynamic_matches(spec, dynamic_names)
            entries.extend(expanded)
            consumed.update(item[0] for item in expanded)
    unknown = [name for name in saved_names if name not in consumed]
    return entries, unknown


def _decode_saved_widgets(
    node: dict[str, Any],
    entries: list[tuple[str, Any, bool]],
) -> dict[str, list[Any]]:
    values = node.get("widgets_values")
    if not isinstance(values, list):
        raise ValueError("widgets_values is not a positional list")
    widget_items = [item for item in node.get("inputs", []) if "widget" in item]

    def decode(items: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], int] | None:
        candidate: dict[str, list[Any]] = {}
        cursor = 0
        for item in items:
            if cursor >= len(values):
                return None
            name = str(item["name"])
            group = [deepcopy(values[cursor])]
            cursor += 1
            if (
                name in SEED_WIDGETS
                and cursor < len(values)
                and values[cursor] in SEED_CONTROLS
            ):
                group.append(deepcopy(values[cursor]))
                cursor += 1
            candidate[name] = group
        return candidate, cursor

    decoded = decode(widget_items)
    if decoded is None:
        decoded = decode([item for item in widget_items if item.get("link") is None])
    if decoded is None:
        raise ValueError("serialized widget values end before all unlinked widgets")
    groups, cursor = decoded
    canonical_widgets = [
        name for name, spec, _optional in entries if _is_widget_spec(spec)
    ]
    missing = [name for name in canonical_widgets if name not in groups]
    for name in missing:
        if cursor >= len(values):
            break
        group = [deepcopy(values[cursor])]
        cursor += 1
        if (
            name in SEED_WIDGETS
            and cursor < len(values)
            and values[cursor] in SEED_CONTROLS
        ):
            group.append(deepcopy(values[cursor]))
            cursor += 1
        groups[name] = group
    if cursor != len(values):
        raise ValueError(f"{len(values) - cursor} widget value(s) remain ambiguous")
    return groups


def _expected_widget_count(entries: list[tuple[str, Any, bool]]) -> int:
    return sum(
        2 if name in SEED_WIDGETS else 1
        for name, spec, _optional in entries
        if _is_widget_spec(spec)
    )


def node_needs_repair(node: dict[str, Any], info: dict[str, Any]) -> bool:
    saved_inputs = node.get("inputs", [])
    saved_names = [str(item.get("name", "")) for item in saved_inputs]
    entries, unknown = canonical_entries(info, saved_names)
    if unknown:
        return False
    if any(
        "widget" in item and item.get("link") is None
        for item in saved_inputs
    ):
        return True
    expected_names = [
        name for name, _spec, _optional in entries if name in saved_names
    ]
    if saved_names != expected_names:
        return True
    canonical_slot = {name: slot for slot, name in enumerate(expected_names)}
    if any(
        isinstance(item.get("link"), int)
        and canonical_slot.get(str(item.get("name", ""))) != saved_slot
        for saved_slot, item in enumerate(saved_inputs)
    ):
        return True
    saved_by_name = {str(item.get("name", "")): item for item in saved_inputs}
    if any(
        _is_widget_spec(spec)
        and name in saved_by_name
        and "widget" not in saved_by_name[name]
        for name, spec, _optional in entries
    ):
        return True
    return False


def repair_node(
    node: dict[str, Any],
    info: dict[str, Any],
    *,
    force: bool = False,
) -> bool:
    if not force and not node_needs_repair(node, info):
        return False
    saved_inputs = node.get("inputs", [])
    saved_names = [str(item.get("name", "")) for item in saved_inputs]
    entries, unknown = canonical_entries(info, saved_names)
    if unknown:
        raise ValueError("unknown saved input(s): " + ", ".join(unknown))
    saved_by_name = {str(item.get("name", "")): item for item in saved_inputs}
    widget_groups = _decode_saved_widgets(node, entries)
    new_inputs: list[dict[str, Any]] = []
    new_widgets: list[Any] = []
    for name, spec, optional in entries:
        item = deepcopy(saved_by_name.get(name, {}))
        item["name"] = name
        item["type"] = _frontend_type(spec)
        item.setdefault("link", None)
        if _is_widget_spec(spec):
            item["widget"] = {"name": name}
            group = widget_groups.get(name)
            if group is None:
                group = [_default_value(spec)]
                if name in SEED_WIDGETS:
                    group.append("fixed")
            new_widgets.extend(group)
        else:
            item.pop("widget", None)
            if optional:
                item.setdefault("shape", 7)
        new_inputs.append(item)
    node["inputs"] = new_inputs
    node["widgets_values"] = new_widgets
    return True


def repair_workflow(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    nodes = {node["id"]: node for node in workflow.get("nodes", [])}
    repaired: list[str] = []
    repaired_ids: set[int] = set()
    skipped: list[str] = []
    for node in nodes.values():
        node_type = str(node.get("type", ""))
        info = object_info.get(node_type)
        if not isinstance(info, dict):
            continue
        try:
            if repair_node(node, info, force=force):
                repaired.append(f"{node['id']}:{node_type}")
                repaired_ids.add(node["id"])
        except ValueError as error:
            skipped.append(f"{node['id']}:{node_type}: {error}")
    link_slots: dict[tuple[int, int], tuple[int, str]] = {}
    for node_id, node in nodes.items():
        if node_id not in repaired_ids:
            continue
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if isinstance(link_id, int):
                link_slots[(node_id, link_id)] = (slot, str(item.get("type", "*")))
    for link in workflow.get("links", []):
        key = (link[3], link[0])
        if key in link_slots:
            link[4], link[5] = link_slots[key]
    normalized = normalize_native_widget_inputs(workflow)
    for item in normalized["normalized"]:
        if item not in repaired:
            repaired.append(item)
    return {"repaired": repaired, "skipped": skipped}


def _load_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _iter_workflows(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair positional ComfyUI widget values and target slots from object_info."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    object_info = _load_url(f"{args.server.rstrip('/')}/object_info")
    changed = 0
    skipped: list[str] = []
    for path in _iter_workflows(args.paths):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            continue
        original = deepcopy(workflow)
        result = repair_workflow(workflow, object_info)
        if result["skipped"]:
            skipped.extend(f"{path}: {item}" for item in result["skipped"])
        if workflow == original:
            continue
        changed += 1
        print(f"{path}: repaired {len(result['repaired'])} node(s)")
        if not args.write:
            continue
        if args.backup_dir is not None:
            destination = args.backup_dir / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"changed_files": changed, "skipped": skipped}, ensure_ascii=False))
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
