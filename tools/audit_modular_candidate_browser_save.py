"""Check SHA-pinned private modular candidates after native ComfyUI Save As.

This checks saved graph semantics, not execution or model quality. A manifest
must explicitly pin each candidate and every intentional UI text edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.audit_modular_legacy_browser_save import (
    fetch_core_schemas,
    normalize_known_ui_changes,
    schema_default_appends,
)
from tools.audit_modular_sampling_compat import SCHEMA, write_new

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "artifacts" / "development"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_case(case: dict, *, profile: Path, current_nodes: dict) -> dict:
    relative = case["original"]
    original = (ROOT / relative).resolve()
    if (not original.is_relative_to(PRIVATE_ROOT.resolve()) or not original.is_file()
            or _sha(original) != case["original_sha256"]):
        raise ValueError("Candidate path/SHA is not the pinned private source")
    saved_name = case["saved"]
    if Path(saved_name).name != saved_name or not saved_name.endswith(".json"):
        raise ValueError("Native save must be a bare JSON filename")
    saved = (profile / "user/default/workflows" / saved_name).resolve()
    if not saved.is_relative_to(profile) or not saved.is_file():
        raise ValueError("Native save is absent from the isolated profile")
    before = json.loads(original.read_bytes())
    after = json.loads(saved.read_bytes())
    if not (isinstance(before, dict) and isinstance(before.get("nodes"), list)
            and isinstance(after, dict) and isinstance(after.get("nodes"), list)):
        raise ValueError("Expected two frontend workflow graphs")
    labels_path = ROOT / "web/task_type_labels.js"
    normalized, ui_changes = normalize_known_ui_changes(
        before, after, current_nodes, labels_sha256=_sha(labels_path))
    appends = schema_default_appends(before, normalized, current_nodes)
    original_nodes = {node["id"]: node for node in before["nodes"]}
    edits = {}
    for edit in case.get("edits", []):
        node_id, index = edit["node_id"], edit["widget_index"]
        if (type(node_id) is not int or type(index) is not int or index < 0
                or (node_id, index) in edits or node_id not in original_nodes
                or original_nodes[node_id]["widgets_values"][index] != edit["before"]):
            raise ValueError("Edit is not pinned to a unique original widget")
        edits[node_id, index] = edit["after"]
    semantic = audit_roundtrip(before, normalized, edits=edits, appended_widgets=appends)
    return {"original": relative, "original_sha256": case["original_sha256"],
            "saved": saved.relative_to(profile).as_posix(), "saved_sha256": _sha(saved),
            "semantic_audit": semantic, "ui_normalizations": ui_changes,
            "appended_optional_defaults": {str(node_id): values for node_id, values in appends.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--core-base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()) or output.exists():
        parser.error("Use a new private development report")
    manifest = json.loads(args.manifest.read_text(encoding="utf8"))
    current = json.loads(args.current.read_text(encoding="utf8"))
    if manifest.get("schema") != "t8.modular-sampling.browser-candidates.v1" or current.get("schema") != SCHEMA:
        parser.error("Candidate manifest or current project snapshot has wrong schema")
    cases_spec = manifest.get("cases")
    if (not isinstance(cases_spec, list) or not cases_spec
            or any(not isinstance(case, dict) or not isinstance(case.get("saved"), str)
                   for case in cases_spec)
            or len({case["saved"] for case in cases_spec}) != len(cases_spec)):
        parser.error("Candidate manifest needs nonempty cases with unique saved names")
    core = fetch_core_schemas(args.core_base_url)
    nodes = {node["id"]: node for node in current["nodes"]}
    if nodes.keys() & core.keys():
        parser.error("Installed UI schema collides with a project node ID")
    nodes.update(core)
    profile = args.profile.resolve()
    if not profile.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("Profile must be a private development directory")
    cases = []
    for case in cases_spec:
        try:
            cases.append({"status": "pass", **audit_case(case, profile=profile, current_nodes=nodes)})
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            cases.append({"status": "fail", "original": case.get("original"),
                          "saved": case.get("saved"), "reason": str(error)})
    status = "browser_save_semantics_pass_not_execution_qualification" if all(
        case["status"] == "pass" for case in cases) else "fail"
    report = {"schema": "t8.modular-sampling.candidate-browser-audit.v1", "status": status,
              "manifest_sha256": _sha(args.manifest), "current_sha256": _sha(args.current),
              "installed_ui_schema_sha256": {
                  name: hashlib.sha256(json.dumps(node["info"], sort_keys=True).encode()).hexdigest()
                  for name, node in core.items()},
              "cases": cases,
              "limits": "Native UI actions are external to this file audit; no sampling, media, GPU or human qualification."}
    write_new(output, report)
    print(json.dumps({"status": status, "cases": [
        {"original": case.get("original"), "status": case["status"], "reason": case.get("reason")}
        for case in cases]}, ensure_ascii=False))
    return 0 if status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
