"""Read-only execution contract checks, distinct from native browser evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def link(value, ids):
    return (isinstance(value, list) and len(value) == 2 and str(value[0]) in ids
            and type(value[1]) is int)


def widget(spec):
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return not options.get("forceInput") and (isinstance(spec[0], list)
        or spec[0] in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})


def audit_dynamic_socket(name, info):
    if "." not in name:
        return
    group, local = name.split(".", 1)
    specs = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
    spec = specs.get(group)
    if not spec or spec[0] != "COMFY_AUTOGROW_V3":
        return
    template = spec[1].get("template", {})
    names = template.get("names")
    if names is None and isinstance(template.get("max"), int) and "prefix" in template:
        names = [f"{template['prefix']}{i}" for i in range(template["max"])]
    if names is not None and local not in names:
        raise ValueError(f"Dynamic socket is not declared by the actual native schema: {name}")


def audit_candidate(prompt, workflow, object_info):
    """Check literal values in schema order and physical edges without the builder."""
    ids = {key: i + 1 for i, key in enumerate(prompt)}
    real_list = [n for n in workflow["nodes"] if n["type"] != "MarkdownNote"]
    real = {n["id"]: n for n in real_list}
    if len(real_list) != len(real) or set(real) != set(ids.values()):
        raise ValueError("Execution node identities changed")
    edges = {edge[0]: edge for edge in workflow["links"]}
    if len(edges) != len(workflow["links"]):
        raise ValueError("Duplicate edge identities")
    consumed, values_count = set(), 0
    for key, source in prompt.items():
        node = real[ids[key]]
        if node["type"] != source["class_type"] or node.get("mode", 0) != 0:
            raise ValueError("Node type or execution mode changed")
        info = object_info[source["class_type"]]
        groups, defaults = {}, {}
        pos = 0
        values = node["widgets_values"]
        for section in ("required", "optional"):
            for name, spec in info.get("input", {}).get(section, {}).items():
                if not widget(spec):
                    continue
                if pos >= len(values):
                    raise ValueError(f"Missing serialized widget value: {key}.{name} at index {pos}")
                groups[name] = values[pos]
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                kind = spec[0]
                fallback = ((kind[0] if kind else "") if isinstance(kind, list) else
                    {"INT": 0, "FLOAT": 0., "BOOLEAN": False, "STRING": "", "COMBO": ""}.get(kind))
                defaults[name] = options.get("default", fallback)
                pos += 1
                # Honor the declared UI control, including explicit False.
                # Name-based fallback only applies to older schemas without metadata.
                if options.get("control_after_generate", name in {"seed", "noise_seed"}):
                    if pos >= len(values) or values[pos] != "fixed":
                        raise ValueError("Seed control must stay fixed")
                    pos += 1
        if pos != len(values):
            raise ValueError(f"Unaccounted serialized widgets: {key} {node['type']} expected {pos}, actual {len(values)}")
        pins = {pin["name"]: (i, pin) for i, pin in enumerate(node["inputs"])}
        if len(pins) != len(node["inputs"]):
            raise ValueError("Ambiguous input sockets")
        for name, value in source["inputs"].items():
            audit_dynamic_socket(name, info)
            if link(value, ids):
                if name not in pins:
                    raise ValueError("Missing linked socket")
                slot, pin = pins[name]
                edge = edges.get(pin.get("link"))
                expected = [ids[str(value[0])], value[1], ids[key], slot]
                if edge is None or edge[1:5] != expected:
                    raise ValueError("Execution edge changed")
                outputs = real[expected[0]]["outputs"]
                if expected[1] >= len(outputs) or pin["link"] not in (outputs[expected[1]].get("links") or []):
                    raise ValueError("Output lost edge ownership")
                consumed.add(pin["link"])
            else:
                if name not in groups or not equal(groups[name], value):
                    raise ValueError(f"Widget value changed: {key}.{name}")
                if name in pins and pins[name][1].get("link") is not None:
                    raise ValueError("Literal widget became an unexpected link")
                values_count += 1
        for name in set(groups) - set(source["inputs"]):
            if not equal(groups[name], defaults[name]):
                raise ValueError("Unspecified widget no longer uses its schema default")
    if consumed != set(edges):
        raise ValueError("Unaccounted execution links")
    return {"nodes": len(real), "edges": len(consumed), "explicit_widget_values": values_count,
            "scope": "candidate_serialization_only_not_browser_roundtrip"}


def audit_api(prompt, api, object_info):
    """Compare an actual native API export; only exact schema defaults may be added."""
    ids = {key: str(i + 1) for i, key in enumerate(prompt)}
    if set(api) != set(ids.values()):
        raise ValueError("Exported execution node set changed")
    defaults = []
    for key, source in prompt.items():
        actual = api[ids[key]]
        if actual["class_type"] != source["class_type"]:
            raise ValueError("Exported node type changed")
        expected = deepcopy(source["inputs"])
        for name, value in expected.items():
            if link(value, ids):
                expected[name] = [ids[str(value[0])], value[1]]
        info = object_info[source["class_type"]]["input"]
        specs = {**info.get("required", {}), **info.get("optional", {})}
        for name in set(actual["inputs"]) - set(expected):
            spec = specs.get(name, [])
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if "default" not in options or not equal(actual["inputs"][name], options["default"]):
                raise ValueError("Unexpected exported input/default")
            expected[name] = options["default"]
            defaults.append(f"{key}.{name}")
        if not equal(actual["inputs"], expected):
            raise ValueError("Exported execution inputs changed")
    return {"scope": "supplied_API_matches_recipe_not_proof_of_browser_provenance", "schema_defaults_added": defaults}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prompt", "workflow", "object-info", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    research = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909"
    if not args.output.resolve().is_relative_to(research) or args.output.exists():
        raise ValueError("Use a new evidence file under acceleration research")
    paths = [args.prompt, args.workflow, args.object_info]
    data = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    result = audit_candidate(*data)
    result["files"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("scope", "nodes", "edges", "explicit_widget_values")}))


if __name__ == "__main__":
    main()
