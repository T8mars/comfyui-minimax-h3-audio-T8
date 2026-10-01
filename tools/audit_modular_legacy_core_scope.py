"""Read-only M0 Core-scoped audit of every frozen old JSON graph.

This runs current CPU Core input validation for API graphs whose node types are
registered in the selected Core/T8/explicit-plugin scope. Frontend graphs get
node availability inventory only; browser import/serialization and plugin
behavior are separate gates. A successful Core return is accepted only when
*every* output validates.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

from tools.audit_modular_sampling_compat import SCHEMA, write_new

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/development/modular-sampling-m0-20260922/baseline.json"
UI_ONLY = frozenset({"MarkdownNote", "Reroute", "PrimitiveNode", "Note"})


def frozen_graphs(root: Path, baseline: dict):
    if baseline.get("schema") != SCHEMA or not isinstance(baseline.get("files"), dict):
        raise ValueError("Expected the original frozen M0 baseline")
    root = root.resolve()
    for relative, expected_sha in sorted(baseline["files"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            yield relative, "missing_or_escaping_frozen_file", None
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            yield relative, "frozen_file_sha_changed", None
            continue
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            yield relative, "invalid_json", None
            continue
        if (isinstance(value, dict) and isinstance(value.get("nodes"), list)
                and isinstance(value.get("links"), list)):
            yield relative, "frontend", value
        elif (isinstance(value, dict) and value
              and all(isinstance(node, dict) and isinstance(node.get("class_type"), str)
                      and isinstance(node.get("inputs"), dict) for node in value.values())):
            yield relative, "api", value
        else:
            yield relative, "other", value


def assess_core_result(result: tuple, outputs: set[str]) -> dict:
    good = set(result[2])
    errors = result[3] if isinstance(result[3], dict) else {}
    ok = bool(result[0]) and bool(outputs) and good == outputs and not errors
    kinds = sorted({error.get("type", "unknown") for detail in errors.values()
                    for error in detail.get("errors", []) if isinstance(error, dict)})
    if not result[0] and isinstance(result[1], dict):
        kinds.append(result[1].get("type", "unknown"))
    reasons = [{"node_id": str(node_id), "class_type": detail.get("class_type"),
                "type": error.get("type"), "details": str(error.get("details", ""))[:240]}
               for node_id, detail in errors.items() for error in detail.get("errors", [])
               if isinstance(error, dict)]
    return {"status": "core_all_outputs_valid" if ok else "core_invalid_or_ignored_output",
            "expected_outputs": sorted(outputs), "validated_outputs": sorted(good),
            "error_kinds": sorted(set(kinds)), "error_reasons": reasons[:12]}


def frontend_type_inventory(graph: dict) -> tuple[set[str], list[str], int]:
    """Inspect bundled subgraphs, not mistake their UUIDs for missing plugins.

    Every definition is inspected, including unused ones. This is type
    inventory only: it does not flatten graph links or validate execution.
    """
    definitions = graph.get("definitions", {})
    # Legacy SelfLift exports explicitly store null when there are no bundles.
    if definitions is None:
        definitions = {}
    if not isinstance(definitions, dict):
        raise ValueError("Frontend definitions must be an object")
    subgraphs = definitions.get("subgraphs", [])
    if not isinstance(subgraphs, list):
        raise ValueError("Frontend subgraphs must be a list")
    by_id = {}
    for definition in subgraphs:
        if (not isinstance(definition, dict)
                or not isinstance(definition.get("id"), str) or not definition["id"]
                or definition["id"] in by_id
                or not isinstance(definition.get("nodes"), list)
                or not isinstance(definition.get("links"), list)
                or definition.get("definitions")):
            raise ValueError("Invalid, duplicate or locally nested subgraph definition")
        by_id[definition["id"]] = definition
    leaves, visited, active = set(), set(), set()

    def visit(nodes):
        for node in nodes:
            if (not isinstance(node, dict) or not isinstance(node.get("type"), str)
                    or not node["type"]):
                raise ValueError("Frontend node has no valid type")
            kind = node["type"]
            if kind not in by_id:
                leaves.add(kind)
                continue
            visit_definition(kind)

    def visit_definition(kind):
        if kind in active:
            raise ValueError("Cyclic embedded subgraph reference")
        if kind not in visited:
            active.add(kind)
            visit(by_id[kind]["nodes"])
            active.remove(kind)
            visited.add(kind)

    visit(graph["nodes"])
    for kind in by_id:
        visit_definition(kind)
    return leaves, sorted(by_id), sum(len(item["nodes"]) for item in subgraphs)


async def audit(root: Path, baseline: dict, registry: dict, validator) -> dict:
    frozen_project_ids = {node["id"] for node in baseline.get("nodes", [])
                          if isinstance(node, dict) and isinstance(node.get("id"), str)}
    files = []
    counts = {"frontend": 0, "api": 0, "other": 0, "frozen_error": 0,
              "frontend_all_types_registered": 0, "frontend_missing_project_type": 0,
              "frontend_needs_external_or_ui_types": 0, "api_all_outputs_valid": 0,
              "api_missing_types": 0, "api_invalid_or_no_outputs": 0,
              "frontend_embedded_types_available": 0, "frontend_invalid_definitions": 0}
    missing_types = {}
    for relative, kind, graph in frozen_graphs(root, baseline):
        entry = {"path": relative, "kind": kind}
        if kind in ("frontend", "api"):
            counts[kind] += 1
            if kind == "frontend":
                try:
                    types, embedded, embedded_count = frontend_type_inventory(graph)
                    if set(embedded) & (set(registry) | UI_ONLY):
                        raise ValueError("Embedded definition shadows a registered/UI node type")
                except ValueError as error:
                    counts["frontend_invalid_definitions"] += 1
                    entry.update(status="invalid_frontend_definitions", reason=str(error))
                    files.append(entry)
                    continue
                if embedded:
                    entry.update(embedded_subgraph_ids=embedded,
                                 embedded_node_count=embedded_count)
            else:
                types = {node["class_type"] for node in graph.values()}
            missing = sorted(item for item in types if isinstance(item, str)
                             and item not in registry and item not in UI_ONLY)
            ui_only = sorted(item for item in types if item in UI_ONLY)
            entry["missing_registered_types"] = missing
            if ui_only:
                entry["ui_only_types"] = ui_only
            for item in missing:
                missing_types[item] = missing_types.get(item, 0) + 1
            if kind == "frontend":
                if any(item in frozen_project_ids for item in missing):
                    counts["frontend_missing_project_type"] += 1
                    entry["status"] = "missing_project_node"
                elif missing:
                    counts["frontend_needs_external_or_ui_types"] += 1
                    entry["status"] = "registry_scope_incomplete"
                elif embedded:
                    counts["frontend_embedded_types_available"] += 1
                    entry["status"] = "embedded_types_available_not_browser_validated"
                elif ui_only:
                    counts["frontend_needs_external_or_ui_types"] += 1
                    entry["status"] = "registry_scope_incomplete"
                else:
                    counts["frontend_all_types_registered"] += 1
                    entry["status"] = "types_registered_not_browser_validated"
                entry["node_count"] = len(graph["nodes"])
            elif missing:
                counts["api_missing_types"] += 1
                entry["status"] = "registry_scope_incomplete"
            else:
                outputs = {str(node_id) for node_id, node in graph.items()
                           if getattr(registry[node["class_type"]], "OUTPUT_NODE", False) is True}
                if not outputs:
                    entry.update(status="no_registered_output", expected_outputs=[])
                    counts["api_invalid_or_no_outputs"] += 1
                else:
                    try:
                        result = await validator(relative, deepcopy(graph), None)
                        entry.update(assess_core_result(result, outputs))
                    except Exception as error:  # Validation must not stop auditing the remaining frozen files.
                        entry.update(status="core_validation_exception",
                                     exception_type=type(error).__name__)
                    if entry["status"] == "core_all_outputs_valid":
                        counts["api_all_outputs_valid"] += 1
                    else:
                        counts["api_invalid_or_no_outputs"] += 1
        elif kind == "other":
            counts["other"] += 1
            entry["status"] = "not_a_graph"
        else:
            counts["frozen_error"] += 1
            entry["status"] = kind
        files.append(entry)
    expected_total = len(baseline["files"])
    if sum(counts[key] for key in ("frontend", "api", "other", "frozen_error")) != expected_total:
        raise RuntimeError("Frozen corpus count changed during Core-scope audit")
    return {"schema": "t8.modular-sampling.legacy-core-scope.v1",
            "status": ("frozen_identity_failed" if counts["frozen_error"]
                       else "frontend_structure_failed" if counts["frontend_invalid_definitions"]
                       else "scoped_audit_completed_not_workflow_qualification"),
            "frozen_json_count": expected_total, "counts": counts,
            "missing_type_file_counts": dict(sorted(missing_types.items())), "files": files,
            "qualification": "Current CPU Core plus local T8 and any explicitly loaded plugin registrations. "
                             "API Core validation requires every output; frontend graphs receive node-type "
                             "inventory only, recursively including all bundled definitions. Subgraph "
                             "links/ports are not validated here. Plugin behavior, browser import/save, queue execution, numerical "
                             "parity, GPU and human AV quality are NOT certified. Empty isolated input/model "
                             "directories and missing types can reflect the audit environment, not graph damage."}


async def current_registry(core: Path, root: Path,
                           extra_custom_nodes: tuple[Path, ...] = (),
                           isolated_profile: Path | None = None) -> tuple[dict, int, list[dict]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    core = core.resolve()
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from comfy.cli_args import args
    args.cpu = True
    if isolated_profile is not None:
        profile = isolated_profile.resolve()
        if profile.exists() or not profile.is_relative_to(root.resolve() / "artifacts"):
            raise ValueError("Isolated profile must be a new private artifacts directory")
        profile.mkdir(parents=True)
        import folder_paths
        for name, setter in (("user", folder_paths.set_user_directory),
                             ("input", folder_paths.set_input_directory),
                             ("output", folder_paths.set_output_directory),
                             ("temp", folder_paths.set_temp_directory)):
            directory = profile / name
            directory.mkdir()
            setter(str(directory))
        args.front_end_root = str(core / "web")
        args.enable_assets = False
        args.enable_manager = False
        from app.assets.manager import default_asset_manager
        import server
        server.PromptServer(asyncio.get_running_loop(), default_asset_manager())
    import nodes
    await nodes.init_builtin_extra_nodes()
    builtin_count = len(nodes.NODE_CLASS_MAPPINGS)
    extra_results = []
    custom_root = (core / "custom_nodes").resolve()
    for extra in extra_custom_nodes:
        path = extra.resolve()
        if (not path.is_relative_to(custom_root) or path == root.resolve()
                or not path.is_dir()):
            raise ValueError("Extra registry source must be another installed custom-node directory")
        before = set(nodes.NODE_CLASS_MAPPINGS)
        loaded = await nodes.load_custom_node(str(path), before)
        extra_results.append({"directory": path.name, "loaded": bool(loaded),
                              "added_node_types": len(set(nodes.NODE_CLASS_MAPPINGS) - before)})
    spec = importlib.util.spec_from_file_location(
        "_t8_modular_legacy_core_scope", root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = await package.comfy_entrypoint().get_node_list()
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Legacy Core audit unexpectedly initialized CUDA")
    return nodes.NODE_CLASS_MAPPINGS, builtin_count, extra_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-custom-node", action="append", default=[], type=Path,
                        help="Explicit installed custom-node directory to include in this scoped audit")
    parser.add_argument("--isolated-prompt-server", action="store_true",
                        help="Initialize a real CPU PromptServer without listening, using a private artifact profile")
    options = parser.parse_args()
    output = options.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new private artifacts output")
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))

    async def run():
        registry, builtin_count, extra_results = await current_registry(
            options.core, ROOT, tuple(options.extra_custom_node),
            output.with_suffix(".profile") if options.isolated_prompt_server else None)
        import execution
        # Core logs per-node failures as errors even though this audit collects
        # them as data; keep the structured result, not hundreds of duplicate logs.
        previous = logging.root.manager.disable
        logging.disable(logging.ERROR)
        try:
            report = await audit(ROOT, baseline, registry, execution.validate_prompt)
        finally:
            logging.disable(previous)
        report["core_builtin_registered_count"] = builtin_count
        report["scoped_registered_count"] = len(registry)
        report["extra_custom_node_loads"] = extra_results
        report["isolated_prompt_server"] = options.isolated_prompt_server
        report["cuda_initialized"] = False
        return report

    report = asyncio.run(run())
    write_new(output, report)
    print(json.dumps({key: report[key] for key in ("status", "frozen_json_count", "counts",
                                               "core_builtin_registered_count", "scoped_registered_count",
                                               "cuda_initialized")}, ensure_ascii=False))
    return 0 if report["status"] == "scoped_audit_completed_not_workflow_qualification" else 1


if __name__ == "__main__":
    raise SystemExit(main())
