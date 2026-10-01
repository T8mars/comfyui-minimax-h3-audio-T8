"""Audit all bundled definitions, boundaries and proxy bindings on native save.

Subgraph boundary nodes are represented only inside this read-only auditor.
They are never registered, written into workflows or sent to Core execution.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tools import audit_modular_legacy_browser_save as legacy
from tools.audit_modular_browser_roundtrip import _edges, _nodes, _same_value, audit_roundtrip
from tools.audit_modular_legacy_core_scope import frontend_type_inventory
from tools.build_quickstart_subgraphs import _widget_index
from tools.run_modular_m0_legacy_browser import _audit_reopened as flat_reopened


def _definitions(graph: dict) -> dict:
    frontend_type_inventory(graph)  # Includes cycles and unused definitions.
    return {item["id"]: item for item in (graph.get("definitions") or {}).get("subgraphs", [])}


def _boundary_pins(definition: dict, direction: str) -> list[dict]:
    pins = definition.get(direction, [])
    if not isinstance(pins, list):
        raise ValueError("Subgraph boundary pins must be a list")
    result = []
    for pin in pins:
        if (not isinstance(pin, dict) or not isinstance(pin.get("id"), str)
                or not isinstance(pin.get("name"), str) or not isinstance(pin.get("type"), str)
                or not isinstance(pin.get("linkIds"), list)
                or any(type(value) is not int for value in pin["linkIds"])
                or len(set(pin["linkIds"])) != len(pin["linkIds"])
                or direction == "outputs" and len(pin["linkIds"]) > 1):
            raise ValueError("Invalid subgraph boundary pin or multiple output sources")
        # app.graph.serialize() exposes undefined optional presentation fields
        # as null through Playwright, while native JSON save omits them.
        nullable_ui = {"localized_name", "label", "dir", "shape", "color_off", "color_on"}
        result.append({key: value for key, value in pin.items()
                       if key != "pos" and not (key in nullable_ui and value is None)})
    if (len({pin["id"] for pin in result}) != len(result)
            or len({pin["name"] for pin in result}) != len(result)):
        raise ValueError("Duplicate subgraph boundary pin")
    return result


def _level(definition: dict) -> dict:
    nodes = deepcopy(definition["nodes"])
    links = []
    fields = ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")
    for link in definition["links"]:
        if (not isinstance(link, dict) or set(link) != set(fields)
                or any(type(link.get(key)) is not int for key in fields[:-1])
                or not isinstance(link.get("type"), str)):
            raise ValueError("Unexpected embedded link serialization")
        links.append([link[key] for key in fields])
    input_id, output_id = definition["inputNode"]["id"], definition["outputNode"]["id"]
    if type(input_id) is not int or type(output_id) is not int or input_id == output_id:
        raise ValueError("Invalid subgraph boundary node identities")
    nodes.extend([
        {"id": input_id, "type": "__audit_subgraph_input__", "inputs": [],
         "outputs": [{"name": pin["name"], "type": pin["type"], "links": pin["linkIds"]}
                     for pin in _boundary_pins(definition, "inputs")]},
        {"id": output_id, "type": "__audit_subgraph_output__", "outputs": [],
         "inputs": [{"name": pin["name"], "type": pin["type"],
                     "link": pin["linkIds"][0] if pin["linkIds"] else None}
                    for pin in _boundary_pins(definition, "outputs")]},
    ])
    return {"nodes": nodes, "links": links}


def _proxy_bindings(before: dict, after: dict) -> None:
    old = {node["id"]: node for node in before["nodes"]}
    new = {node["id"]: node for node in after["nodes"]}
    if old.keys() != new.keys():
        raise ValueError("Embedded graph node set changed")
    for node_id, node in old.items():
        if ((node.get("properties") or {}).get("proxyWidgets")
                != (new[node_id].get("properties") or {}).get("proxyWidgets")):
            raise ValueError(f"Subgraph promoted-widget binding changed: {node_id}")


def normalize_proxy_promotions(original: dict, saved: dict) -> tuple[dict, dict]:
    """Derive promoted defaults from the frozen inner graph, never from save."""
    normalized = deepcopy(original)
    definitions = _definitions(original)
    actual_nodes = _nodes(saved)
    records = {}
    for node in normalized["nodes"]:
        proxies = (node.get("properties") or {}).get("proxyWidgets")
        if proxies is None:
            continue
        actual = actual_nodes[node["id"]]
        if ((actual.get("properties") or {}).get("proxyWidgets") == proxies):
            continue
        if (node["type"] not in definitions or not isinstance(proxies, list)
                or not proxies or node.get("widgets_values") != []
                or node.get("widgets_values_named")
                or (actual.get("properties") or {}).get("proxyWidgetErrorQuarantine")
                or "proxyWidgets" in (actual.get("properties") or {})):
            raise ValueError("Unsupported proxy-widget migration")
        definition = definitions[node["type"]]
        inner = _nodes(definition)
        pins = _boundary_pins(definition, "inputs")
        links = {link["id"]: link for link in definition["links"]}
        _edges(_level(definition), _nodes(_level(definition)))
        exposed = [pin for pin in node["inputs"] if "widget" in pin]
        if len(exposed) != len(proxies):
            raise ValueError("Proxy count does not match promoted inputs")
        expected = {}
        bindings = []
        actual_pins = {pin["name"]: pin for pin in actual["inputs"]}
        for pin, proxy in zip(exposed, proxies, strict=True):
            if (not isinstance(proxy, list) or len(proxy) != 2
                    or any(not isinstance(value, str) for value in proxy)
                    or pin["name"] in expected or pin.get("link") is not None
                    or pin["widget"] != {"name": pin["name"]}
                    or actual_pins.get(pin["name"], {}).get("widget") != pin["widget"]):
                raise ValueError("Invalid promoted widget contract")
            matches = [(slot, boundary) for slot, boundary in enumerate(pins)
                       if boundary["name"] == pin["name"]]
            if len(matches) != 1:
                raise ValueError("Promoted widget has no unique boundary")
            slot, boundary = matches[0]
            if boundary["type"] != pin["type"] or len(boundary["linkIds"]) != 1:
                raise ValueError("Promoted widget boundary changed")
            link = links[boundary["linkIds"][0]]
            target = inner[link["target_id"]]
            target_pin = target["inputs"][link["target_slot"]]
            if (link["origin_id"] != definition["inputNode"]["id"]
                    or link["origin_slot"] != slot
                    or proxy != [str(target["id"]), target_pin["name"]]
                    or target_pin.get("widget") != {"name": proxy[1]}):
                raise ValueError("Proxy no longer matches its boundary target")
            index = _widget_index(target, proxy[1])
            expected[pin["name"]] = deepcopy(target["widgets_values"][index])
            bindings.append({"name": pin["name"], "target": proxy, "widget_index": index})
        named = actual.get("widgets_values_named") or {}
        if (list(named) != list(expected)
                or not _same_value(list(named.values()), list(expected.values()))
                or not _same_value(actual.get("widgets_values"), list(expected.values()))):
            raise ValueError("Promoted values differ from frozen inner defaults")
        node["widgets_values"] = list(expected.values())
        node["widgets_values_named"] = expected
        del node["properties"]["proxyWidgets"]
        records[str(node["id"])] = bindings
    return normalized, records


def normalize_node_collisions(old: dict, new: dict, outer_ids: set[int]) -> tuple[dict, dict]:
    """Reproduce native sequential ID allocation; then audit all remapped data."""
    _nodes(old)
    _nodes(new)
    if [node["id"] for node in old["nodes"]] == [node["id"] for node in new["nodes"]]:
        return new, {}
    if len(old["nodes"]) != len(new["nodes"]) or not outer_ids:
        raise ValueError("Embedded node count or allocation scope changed")
    used = set(outer_ids)
    last_id = max(used)
    inverse = {}
    for a, b in zip(old["nodes"], new["nodes"], strict=True):
        expected = a["id"] if a["id"] not in used else last_id + 1
        if b["id"] != expected:
            raise ValueError("Embedded IDs are not the native collision allocation")
        used.add(expected)
        last_id = max(last_id, expected)
        inverse[expected] = a["id"]
    result = deepcopy(new)
    for node in result["nodes"]:
        node["id"] = inverse[node["id"]]
        if (node.get("properties") or {}).get("proxyWidgets"):
            raise ValueError("Nested legacy proxy migration is not qualified")
    for link in result["links"]:
        for field in ("origin_id", "target_id"):
            link[field] = inverse.get(link[field], link[field])
    return result, {str(old_id): new_id for new_id, old_id in inverse.items() if new_id != old_id}


def normalize_boundary_link_ids(old: dict, new: dict, outer_ids: set[int]) -> tuple[dict, dict]:
    """Allow only input-boundary collision repair, with identical named edges."""
    a, b = _level(old), _level(new)
    old_edges, new_edges = _edges(a, _nodes(a)), _edges(b, _nodes(b))
    old_ids = {edge[0] for edge in old_edges}
    new_ids = {edge[0] for edge in new_edges}
    if len(old_ids) != len(old_edges) or len(new_ids) != len(new_edges):
        raise ValueError("Duplicate embedded link ID")
    if old_ids == new_ids:
        return new, {}
    before = {edge[1:]: edge[0] for edge in old_edges}
    after = {edge[1:]: edge[0] for edge in new_edges}
    if (len(before) != len(old_edges) or len(after) != len(new_edges)
            or before.keys() != after.keys()):
        raise ValueError("Renumbered boundary changed its named execution edge")
    inverse = {}
    for edge, old_id in before.items():
        new_id = after[edge]
        if old_id == new_id:
            continue
        if (old_id not in outer_ids or edge[0] != old["inputNode"]["id"]
                or new_id <= max(old_ids) or new_id in outer_ids):
            raise ValueError("Link renumbering is not boundary collision repair")
        inverse[new_id] = old_id
    result = deepcopy(new)
    for link in result["links"]:
        link["id"] = inverse.get(link["id"], link["id"])
    for direction in ("inputs", "outputs"):
        for pin in result[direction]:
            pin["linkIds"] = [inverse.get(value, value) for value in pin["linkIds"]]
    for node in result["nodes"]:
        for pin in node.get("inputs", []):
            if pin.get("link") in inverse:
                pin["link"] = inverse[pin["link"]]
        for pin in node.get("outputs", []):
            if pin.get("links") is not None:
                pin["links"] = [inverse.get(value, value) for value in pin["links"]]
    return result, {str(old_id): new_id for new_id, old_id in sorted(inverse.items())}


def _audit_level(a: dict, b: dict, current_nodes: dict | None) -> dict:
    if current_nodes is None:
        return flat_reopened(a, b)
    a = deepcopy(a)
    normalized, changes = legacy.normalize_known_ui_changes(
        a, b, current_nodes,
        labels_sha256=legacy._sha(legacy.ROOT / "web/task_type_labels.js"))
    # Legacy Quick definitions retain unconnected widget sockets, unlike most
    # flat saves. These two current Core widgets have changed UI socket types.
    # Connected sockets, values, and other types are not normalized.
    old_nodes = _nodes(a)
    socket_changes = []
    for node in normalized["nodes"]:
        for name, old_type, new_type in (
            ("bit_depth", "INT", "COMBO") if node["type"] == "CreateVideo" else
            ("format", "COMBO", "COMFY_DYNAMICCOMBO_V3") if node["type"] == "SaveVideo" else
            (None, None, None),
        ):
            if name is None:
                continue
            old_pin = next((pin for pin in old_nodes[node["id"]].get("inputs", [])
                            if pin["name"] == name), None)
            new_pin = next((pin for pin in node.get("inputs", []) if pin["name"] == name), None)
            if old_pin is None or new_pin is None or old_pin["type"] == new_pin["type"]:
                continue
            inputs = current_nodes[node["type"]]["info"]["input"]
            spec = {**inputs.get("required", {}), **inputs.get("optional", {})}[name]
            actual_type = "COMBO" if isinstance(spec[0], list) else spec[0]
            if (old_pin["type"] != old_type or new_pin["type"] != new_type
                    or actual_type != new_type or old_pin.get("link") is not None
                    or new_pin.get("link") is not None
                    or old_pin.get("widget") != {"name": name}
                    or new_pin.get("widget") != {"name": name}):
                raise ValueError("Unexpected Core widget socket migration")
            new_pin["type"] = old_type
            socket_changes.append({"node_id": node["id"], "field": name,
                                   "old_type": old_type, "native_type": new_type})
    for node in a["nodes"]:
        if node["type"] == "VHS_VideoCombine" and isinstance(node.get("widgets_values"), dict):
            node["widgets_values"] = list(node["widgets_values"].values())
        if node["type"] == "MarkdownNote" and isinstance(node.get("widgets_values"), str):
            node["widgets_values"] = [node["widgets_values"]]
    appends = legacy.schema_default_appends(a, normalized, current_nodes)
    audit = audit_roundtrip(a, normalized, appended_widgets=appends)
    audit["ui_normalizations"] = changes
    audit["disconnected_core_widget_socket_changes"] = socket_changes
    audit["appended_optional_defaults"] = {str(key): value for key, value in appends.items()}
    return audit


def audit_definitions(original: dict, saved: dict, current_nodes: dict | None = None,
                      *, migrated_definitions: set[str] | None = None) -> dict:
    before, after = _definitions(original), _definitions(saved)
    if not before or before.keys() != after.keys():
        raise ValueError("Bundled definition set changed or is empty")
    _proxy_bindings(original, saved)
    result = {}
    for key, old in before.items():
        new = after[key]
        renumbered = {}
        node_ids = {}
        dropped_category = None
        if key in (migrated_definitions or set()):
            # Only the seven frozen Quick graphs use this legacy conversion.
            # Category is catalog metadata discarded by native subgraph save;
            # record the exact loss, not an execution-equivalence claim for it.
            if (len(original["nodes"]) != 1 or original["nodes"][0]["type"] != key
                    or old.get("category") != "MiniMax H3 T8/Quick Start"
                    or "category" in new):
                raise ValueError("Unknown legacy Quick metadata migration")
            dropped_category = old["category"]
            new = deepcopy(new)
            new["category"] = dropped_category
            new, node_ids = normalize_node_collisions(
                old, new, {node["id"] for node in original["nodes"]})
        if current_nodes is not None:
            new, renumbered = normalize_boundary_link_ids(
                old, new, {link[0] for link in original["links"]})
        for name in ("id", "name", "category", "description", "config", "widgets"):
            if old.get(name) != new.get(name):
                raise ValueError(f"Subgraph contract changed: {key}.{name}")
        for direction in ("inputs", "outputs"):
            if _boundary_pins(old, direction) != _boundary_pins(new, direction):
                raise ValueError(f"Subgraph boundary changed: {key}.{direction}")
        for field in ("inputNode", "outputNode"):
            if old[field]["id"] != new[field]["id"]:
                raise ValueError(f"Subgraph boundary identity changed: {key}.{field}")
        a, b = _level(old), _level(new)
        _proxy_bindings(a, b)
        audit = _audit_level(a, b, current_nodes)
        if renumbered:
            audit["boundary_link_collision_repairs"] = renumbered
        if node_ids:
            audit["native_node_id_collision_repairs"] = node_ids
        if dropped_category:
            audit["native_dropped_catalog_category"] = dropped_category
        result[key] = audit
    return result


def audit_pair(root, profile, baseline, current, original_relative, saved_name):
    root, profile = root.resolve(), profile.resolve()
    source = (root / original_relative).resolve()
    expected = baseline["files"].get(original_relative)
    if (expected is None or not source.is_relative_to(root) or not source.is_file()
            or legacy._sha(source) != expected):
        raise ValueError("Original graph is not the unchanged M0 frozen file")
    if Path(saved_name).name != saved_name or not saved_name.endswith(".json"):
        raise ValueError("Saved graph must be a bare JSON filename")
    target = (profile / "user/default/workflows" / saved_name).resolve()
    if not target.is_relative_to(profile) or not target.is_file():
        raise ValueError("Native browser save is absent from the isolated profile")
    original, saved = json.loads(source.read_bytes()), json.loads(target.read_bytes())
    normalized, promotions = normalize_proxy_promotions(original, saved)
    live = {node["id"]: node for node in current["nodes"]}
    return {"original": original_relative, "original_sha256": expected,
            "saved": target.relative_to(profile).as_posix(), "saved_sha256": legacy._sha(target),
            "semantic_audit": _audit_level(normalized, saved, live),
            "promoted_proxy_widgets": promotions,
            "proxy_layout_source_sha256": legacy._sha(root / "tools/build_quickstart_subgraphs.py"),
            "embedded_semantic_audit": audit_definitions(normalized, saved, live,
                migrated_definitions={node["type"] for node in normalized["nodes"]
                                      if str(node["id"]) in promotions})}


def audit_reopened(saved, reopened):
    result = flat_reopened(saved, reopened)
    result["embedded_semantic_audit"] = audit_definitions(saved, reopened)
    return result
