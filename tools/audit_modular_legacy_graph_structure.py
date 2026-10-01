"""Read-only structural audit of the frozen M0 JSON corpus.

This checks file identity and saved graph references, not Core import,
execution, external node availability, numerical parity, or media quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.audit_modular_sampling_compat import SCHEMA, write_new

ROOT = Path(__file__).resolve().parents[1]


def _issue(issues, path, kind, **details):
    issues.append({"path": path, "kind": kind, **details})


def _frontend(relative, value, issues, types):
    nodes, links = value["nodes"], value["links"]
    by_id = {}
    for node in nodes:
        if (not isinstance(node, dict) or type(node.get("id")) is not int
                or not isinstance(node.get("type"), str)):
            _issue(issues, relative, "malformed_frontend_node")
            continue
        node_id = node["id"]
        if node_id in by_id:
            _issue(issues, relative, "duplicate_frontend_node_id", node_id=node_id)
        by_id[node_id] = node
        types[node["type"]] = types.get(node["type"], 0) + 1
    link_ids = set()
    link_table = {}
    for link in links:
        if (not isinstance(link, list) or len(link) < 6
                or any(type(link[index]) is not int for index in range(5))):
            _issue(issues, relative, "malformed_frontend_link")
            continue
        link_id, source, source_slot, target, target_slot = link[:5]
        if link_id in link_ids:
            _issue(issues, relative, "duplicate_frontend_link_id", link_id=link_id)
        link_ids.add(link_id)
        link_table[link_id] = link
        if source not in by_id or target not in by_id:
            _issue(issues, relative, "dangling_frontend_link", link_id=link_id)
            continue
        for node_id, slot, pin_key in ((source, source_slot, "outputs"),
                                       (target, target_slot, "inputs")):
            pins = by_id[node_id].get(pin_key)
            if (not isinstance(pins, list) or slot < 0 or slot >= len(pins)):
                _issue(issues, relative, "invalid_frontend_slot", link_id=link_id,
                       node_id=node_id, pin_key=pin_key, slot=slot)
    for node_id, node in by_id.items():
        pins = node.get("inputs", [])
        if not isinstance(pins, list):
            _issue(issues, relative, "malformed_frontend_inputs", node_id=node_id)
            continue
        for slot, pin in enumerate(pins):
            if not isinstance(pin, dict):
                _issue(issues, relative, "malformed_frontend_input", node_id=node_id, slot=slot)
                continue
            link_id = pin.get("link")
            if link_id is None:
                continue
            if type(link_id) is not int:
                _issue(issues, relative, "malformed_input_link_id", node_id=node_id, slot=slot)
                continue
            link = link_table.get(link_id)
            if link is None or link[3] != node_id or link[4] != slot:
                _issue(issues, relative, "input_pin_link_mismatch", node_id=node_id,
                       slot=slot, link_id=link_id)
    return len(by_id), len(links)


def audit(root: Path, baseline: dict) -> dict:
    if baseline.get("schema") != SCHEMA or not isinstance(baseline.get("files"), dict):
        raise ValueError("Expected the frozen M0 compatibility baseline")
    root = Path(root).resolve()
    issues, types = [], {}
    kinds = {"frontend": 0, "api": 0, "other": 0}
    node_count = link_count = 0
    for relative, expected_sha in sorted(baseline["files"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            _issue(issues, relative, "missing_or_escaping_frozen_file")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            _issue(issues, relative, "frozen_file_sha_changed")
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _issue(issues, relative, "invalid_json")
            continue
        if (isinstance(value, dict) and isinstance(value.get("nodes"), list)
                and isinstance(value.get("links"), list)):
            kinds["frontend"] += 1
            nodes, links = _frontend(relative, value, issues, types)
            node_count += nodes
            link_count += links
        elif (isinstance(value, dict) and value
              and all(isinstance(node, dict) and isinstance(node.get("class_type"), str)
                      and isinstance(node.get("inputs"), dict) for node in value.values())):
            kinds["api"] += 1
            for node in value.values():
                kind = node["class_type"]
                types[kind] = types.get(kind, 0) + 1
        else:
            kinds["other"] += 1
    return {"schema": "t8.modular-sampling.legacy-json-structure.v1",
            "status": "pass" if not issues else "fail",
            "frozen_json_count": len(baseline["files"]), "kinds": kinds,
            "frontend_nodes": node_count, "frontend_links": link_count,
            "distinct_node_types": len(types), "issues": issues,
            "qualification": "Frozen SHA and saved graph structure only. No Core import, model inference, "
                             "external-node availability, numerical parity, browser or human AV review."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    target = options.output.resolve()
    if not target.is_relative_to(ROOT / "artifacts") or target.exists():
        parser.error("Use a new private artifacts output")
    baseline = json.loads(options.baseline.read_text(encoding="utf-8"))
    report = audit(ROOT, baseline)
    write_new(target, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
