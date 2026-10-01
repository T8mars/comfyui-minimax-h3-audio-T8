"""Classify installed-asset menu drift without weakening the exact M0 gate.

The strict compatibility report remains authoritative. This secondary report
only explains whether its old-schema differences are additive installed asset
filenames; it never rewrites the frozen baseline or grants runtime parity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools.audit_modular_sampling_compat import compare, write_new


MENU_CATEGORIES = {"lora_name": "loras", "turbo_lora_name": "loras",
                   "sla_lora_name": "loras", "pdd_lora_name": "loras",
                   "control_net_name": "controlnet",
                   "video_vae_name": "vae", "native_video_vae": "vae"}
# model_name is too broad to normalize globally. Only this old loader obtains
# its menu from the registered taehv category (which also includes models/vae).
NODE_MENU_CATEGORIES = {
    ("MiniMaxH3SemanticBridgeConfigT8", "model_name"): "semantic_bridge",
    ("MiniMaxH3SolEngineTAEHVLoaderT8Advanced", "model_name"): "taehv",
    ("MiniMaxH3HybridPairInspectorT8Advanced", "quality_base"): "diffusion_models",
    ("MiniMaxH3HybridPairInspectorT8Advanced", "reference_overlay"): "diffusion_models",
    ("MiniMaxH3HybridModelLoaderT8Advanced", "quality_base"): "diffusion_models",
    ("MiniMaxH3SPEEDModelVAEFingerprintT8Advanced", "checkpoint_name"): "diffusion_models",
    ("MiniMaxH3RavenGuardedLoaderT8Advanced", "unet_name"): "diffusion_models",
}
MENU_FIELDS = frozenset(MENU_CATEGORIES) | {field for _, field in NODE_MENU_CATEGORIES}
WIDGET_TYPES = frozenset({"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"})


def _is_ordered_subsequence(old: list[str], new: list[str]) -> bool:
    remaining = iter(new)
    return all(any(candidate == value for candidate in remaining) for value in old)


def installed_roots(lora_root: Path, *, controlnet_root=None, vae_root=None,
                    taehv_root=None, model_patches_root=None,
                    diffusion_models_root=None, unet_root=None):
    """Explicit model-root inventory; never normalize generic model_name fields."""
    vae = (vae_root or lora_root.parent / "vae").resolve()
    return {"loras": (lora_root.resolve(),),
        "controlnet": ((controlnet_root or lora_root.parent / "controlnet").resolve(),
                       (model_patches_root or lora_root.parent / "model_patches").resolve()),
        "vae": (vae,), "taehv": ((taehv_root or lora_root.parent / "taehv").resolve(), vae),
        "semantic_bridge": ((lora_root.parent / "semantic_bridge").resolve(),),
        "diffusion_models": ((diffusion_models_root or lora_root.parent / "diffusion_models").resolve(),
                             (unet_root or lora_root.parent / "unet").resolve())}


def installed_asset(roots, name):
    for root in roots:
        path = (root / name).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise ValueError("Added asset path is absent or escapes installed category roots")


def diagnose(baseline: dict, current: dict, *, lora_root: Path,
             controlnet_root: Path | None = None, vae_root: Path | None = None,
             taehv_root: Path | None = None,
             model_patches_root: Path | None = None,
             diffusion_models_root: Path | None = None,
             unet_root: Path | None = None) -> dict:
    exact = compare(baseline, current)
    current_nodes = {entry["id"]: entry for entry in current["nodes"]}
    roots = installed_roots(lora_root, controlnet_root=controlnet_root,
        model_patches_root=model_patches_root, vae_root=vae_root, taehv_root=taehv_root,
        diffusion_models_root=diffusion_models_root, unet_root=unet_root)
    findings = []
    issues = []
    for violation in exact["violations"]:
        if violation["kind"] != "old_schema_changed":
            issues.append(violation)
    for old_entry in baseline["nodes"]:
        new_entry = current_nodes.get(old_entry["id"])
        if new_entry is None or old_entry["info"] == new_entry["info"]:
            continue
        normalized = deepcopy(new_entry["info"])
        old_fields = old_entry["info"].get("input", {}).get("required", {})
        new_fields = normalized.get("input", {}).get("required", {})
        for field in MENU_FIELDS & old_fields.keys() & new_fields.keys():
            category = (NODE_MENU_CATEGORIES.get((old_entry["id"], field))
                        or MENU_CATEGORIES.get(field))
            if category is None:
                continue
            old_spec, new_spec = old_fields[field], new_fields[field]
            try:
                old_options = old_spec[1]["options"]
                new_options = new_spec[1]["options"]
            except (IndexError, KeyError, TypeError):
                continue
            if old_options == new_options:
                continue
            if (not isinstance(old_options, list) or
                    not isinstance(new_options, list) or
                    not all(isinstance(value, str) for value in old_options + new_options) or
                    len(set(old_options)) != len(old_options) or
                    len(set(new_options)) != len(new_options) or
                    not _is_ordered_subsequence(old_options, new_options)):
                issues.append({"kind": "menu_removed_reordered_or_invalid",
                               "id": old_entry["id"], "field": field})
                continue
            added = [value for value in new_options if value not in old_options]
            for name in added:
                try:
                    installed_asset(roots[category], name)
                except ValueError:
                    issues.append({"kind": "added_asset_file_not_verified",
                                   "id": old_entry["id"], "field": field,
                                   "category": category, "name": name})
            new_spec[1]["options"] = old_options
            findings.append({"id": old_entry["id"], "field": field,
                             "category": category,
                             "old_count": len(old_options), "new_count": len(new_options),
                             "added": added})
        if normalized != old_entry["info"]:
            issues.append({"kind": "other_old_schema_difference", "id": old_entry["id"]})
    return {
        "schema": "t8.modular-sampling.menu-drift-diagnostic.v1",
        "exact_compatibility_status": exact["status"],
        "structural_status": "pass" if not issues else "fail",
        "old_node_count": exact["old_node_count"],
        "current_node_count": exact["current_node_count"],
        "old_json_count": exact["old_json_count"],
        "menu_additions": findings,
        "issues": issues,
        "qualification": "Secondary diagnosis only; exact M0 compatibility remains unchanged. "
                         "No sampling, workflow migration, GPU or quality qualification.",
    }


def _frontend_widget_index(info: dict, field: str) -> tuple[int, int]:
    required = info["input"]["required"]
    order = info["input_order"]["required"]
    widgets = [name for name in order if required[name][0] in WIDGET_TYPES]
    if field not in widgets:
        raise ValueError(f"Menu field is not a required widget: {field}")
    return widgets.index(field), len(widgets)


def audit_saved_menu_selections(root: Path, baseline: dict, current: dict,
                                menu_findings: list[dict]) -> dict:
    """Check saved choices, not just schema, across every SHA-bound old JSON."""
    root = root.resolve()
    base_nodes = {node["id"]: node for node in baseline["nodes"]}
    new_nodes = {node["id"]: node for node in current["nodes"]}
    affected = {}
    for finding in menu_findings:
        affected.setdefault(finding["id"], []).append(finding["field"])
    issues = []
    selections = []
    counts = {"frozen_files": 0, "frontend_instances": 0, "api_instances": 0,
              "checked_values": 0, "linked_values_not_checked": 0,
              "preexisting_unavailable_values": 0, "current_menu_regressions": 0}

    def check_choice(path: str, node_id: str, node_type: str, field: str, value) -> None:
        old = base_nodes[node_type]["info"]["input"]["required"][field][1]["options"]
        new = new_nodes[node_type]["info"]["input"]["required"][field][1]["options"]
        if not isinstance(value, str) or value not in old:
            counts["preexisting_unavailable_values"] += 1
            issues.append({"kind": "saved_choice_absent_from_frozen_menu", "path": path,
                           "node_id": node_id, "type": node_type, "field": field,
                           "saved_value": value})
            return
        if value not in new:
            counts["current_menu_regressions"] += 1
            issues.append({"kind": "old_saved_choice_lost_in_current_menu", "path": path,
                           "node_id": node_id, "type": node_type, "field": field,
                           "saved_value": value})
            return
        counts["checked_values"] += 1
        selections.append({"path": path, "node_id": node_id, "type": node_type,
                           "field": field, "saved_value": value})

    for relative, expected_sha in sorted(baseline["files"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append({"kind": "frozen_file_missing_or_escaping", "path": relative})
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            issues.append({"kind": "frozen_file_sha_changed", "path": relative})
            continue
        counts["frozen_files"] += 1
        try:
            graph = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            issues.append({"kind": "frozen_file_invalid_json", "path": relative})
            continue
        if not isinstance(graph, dict):
            continue
        if isinstance(graph.get("nodes"), list):
            for node in graph["nodes"]:
                if not isinstance(node, dict) or node.get("type") not in affected:
                    continue
                node_type = node["type"]
                counts["frontend_instances"] += 1
                values = node.get("widgets_values")
                for field in affected[node_type]:
                    try:
                        index, expected_count = _frontend_widget_index(
                            base_nodes[node_type]["info"], field)
                        if not isinstance(values, list) or len(values) != expected_count:
                            raise ValueError("Saved widget count does not match frozen schema")
                        value = values[index]
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        issues.append({"kind": "frontend_menu_value_unverifiable", "path": relative,
                                       "node_id": str(node.get("id")), "type": node_type,
                                       "field": field, "reason": str(error)})
                        continue
                    check_choice(relative, str(node.get("id")), node_type, field, value)
        elif graph and all(isinstance(node, dict) and "class_type" in node
                           and "inputs" in node for node in graph.values()):
            for node_id, node in graph.items():
                node_type = node["class_type"]
                if node_type not in affected:
                    continue
                counts["api_instances"] += 1
                for field in affected[node_type]:
                    value = node["inputs"].get(field)
                    if isinstance(value, list) and len(value) == 2:
                        counts["linked_values_not_checked"] += 1
                        issues.append({"kind": "linked_api_menu_value_unverifiable", "path": relative,
                                       "node_id": str(node_id), "type": node_type, "field": field})
                    else:
                        check_choice(relative, str(node_id), node_type, field, value)
    regression_issues = [issue for issue in issues
                         if issue["kind"] != "saved_choice_absent_from_frozen_menu"]
    return {"status": "pass" if not issues else "fail",
            "regression_status": "pass" if not regression_issues else "fail",
            "counts": counts,
            "selections": selections, "issues": issues,
            "qualification": "Saved choices only for explicitly recognized changed asset menus in SHA-bound old JSON. "
                             "Pre-existing absent assets are not new menu regressions. This does not validate "
                             "graph import, execution, models or media."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    parser.add_argument("--controlnet-root", type=Path)
    parser.add_argument("--model-patches-root", type=Path)
    parser.add_argument("--vae-root", type=Path)
    parser.add_argument("--taehv-root", type=Path)
    parser.add_argument("--diffusion-models-root", type=Path)
    parser.add_argument("--unet-root", type=Path)
    parser.add_argument("--graph-root", type=Path,
                        help="Optional frozen old-JSON root for saved menu-choice audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    report = diagnose(baseline, current, lora_root=args.lora_root,
                      controlnet_root=args.controlnet_root,
                      model_patches_root=args.model_patches_root,
                      vae_root=args.vae_root, taehv_root=args.taehv_root,
                      diffusion_models_root=args.diffusion_models_root, unet_root=args.unet_root)
    if args.graph_root is not None:
        report["saved_graph_impact"] = audit_saved_menu_selections(
            args.graph_root, baseline, current, report["menu_additions"])
    write_new(args.output, report)
    displayed = ({"exact_compatibility_status": report["exact_compatibility_status"],
                  "structural_status": report["structural_status"],
                  "saved_graph_impact": {key: report["saved_graph_impact"][key]
                                         for key in ("status", "regression_status", "counts", "issues")}}
                 if args.graph_root is not None else report)
    print(json.dumps(displayed, ensure_ascii=False, indent=2))
    return 0 if (report["structural_status"] == "pass" and
                 report.get("saved_graph_impact", {}).get("status", "pass") == "pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
