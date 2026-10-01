"""Resolve the complete S01-S29 catalogue against current source and live IDs.

Static resolution is not numerical coverage. This tool creates a new evidence
snapshot only; it does not rewrite the M0 baseline or execute any sampler.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_catalogue():
    # Import only the inert data file, not the project entrypoint or Core.
    path = ROOT / "h3_t8/modular_sampling/catalogue.py"
    name = "_t8_modular_route_catalogue"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols(tree):
    found = {}

    class Visitor(ast.NodeVisitor):
        scope = ()

        def definition(self, node):
            old = self.scope
            self.scope = (*old, node.name)
            key = ".".join(self.scope)
            if key in found:
                raise ValueError("Ambiguous duplicate source definition: " + key)
            found[key] = node
            self.generic_visit(node)
            self.scope = old

        visit_ClassDef = definition
        visit_FunctionDef = definition
        visit_AsyncFunctionDef = definition

    Visitor().visit(tree)
    return found


def resolve_sources(root, routes):
    root = Path(root).resolve()
    cache, rows, issues = {}, [], []
    for route in routes:
        bound = []
        for source in route.sources:
            path = (root / source.path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source binding escapes the project")
            if source.path not in cache:
                try:
                    text = path.read_text(encoding="utf-8-sig")
                    cache[source.path] = digest(path), symbols(ast.parse(text, filename=source.path))
                except (OSError, SyntaxError, ValueError) as error:
                    cache[source.path] = None, {"__error__": str(error)}
            file_sha, definitions = cache[source.path]
            node = definitions.get(source.symbol)
            if node is None:
                issues.append({"route": route.id, "source": asdict(source), "kind": "unresolved_source_symbol",
                               "detail": definitions.get("__error__", "Missing qualified definition")})
                continue
            calls = sorted({ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)})
            bound.append({**asdict(source), "line": node.lineno, "end_line": node.end_lineno,
                          "source_sha256": file_sha,
                          "symbol_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                          "calls": calls})
        rows.append({**asdict(route), "sources": bound})
    return rows, issues


def attach_progress(root, rows, manifest, dimensions):
    """Bind evidence bytes, not a claim that arbitrary attachments prove success."""
    known = {row["id"]: row for row in rows}
    for row in rows:
        row["rollout"] = {key: {"status": "pending", "scope": "Not qualified for this rollout", "evidence": []}
                          for key in dimensions}
    if manifest is None:
        return
    if set(manifest) != {"schema", "entries"} or manifest["schema"] != "t8.modular-sampling.progress.v1":
        raise ValueError("Unknown progress manifest")
    seen = set()
    root = Path(root).resolve()
    for item in manifest["entries"]:
        route, dimension = item["route"], item["dimension"]
        if route not in known or dimension not in dimensions or (route, dimension) in seen:
            raise ValueError("Unknown or duplicate route/dimension in progress")
        seen.add((route, dimension))
        # Completion requires separate behavioral/qualification review. A
        # source/file inventory cannot promote itself to a qualified algorithm.
        if item["status"] != "partial" or not item["scope"] or not item["files"]:
            raise ValueError("This inventory may record partial evidence, not self-certify completion")
        evidence = []
        for relative in item["files"]:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Progress evidence must be an existing in-project file")
            evidence.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path)})
        known[route]["rollout"][dimension] = {"status": "partial", "scope": item["scope"], "evidence": evidence}


def audit(root=ROOT, *, catalogue=None, live_ids=None, progress=None):
    catalogue = load_catalogue() if catalogue is None else catalogue
    routes = catalogue.validate_catalogue()
    rows, issues = resolve_sources(root, routes)
    if live_ids is not None:
        for row in rows:
            for node in row["public_nodes"]:
                if node not in live_ids:
                    issues.append({"route": row["id"], "kind": "missing_live_public_node", "node": node})
    attach_progress(root, rows, progress, catalogue.DIMENSIONS)
    return {"schema": "t8.modular-sampling.route-inventory.v1", "status": "pass" if not issues else "fail",
            "routes": rows, "issues": issues, "route_count": len(rows),
            "source_binding_count": sum(len(row["sources"]) for row in rows),
            "live_node_ids_checked": live_ids is not None,
            "qualification": "Symbol/registration resolution and evidence byte bindings only. Not proof of branch execution, "
                             "numerical parity, effect coverage, browser roundtrip, GPU, human quality or future-source completeness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-core", action="store_true")
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifact output; old evidence is never overwritten")
    sys.path.insert(0, str(ROOT / "tools"))
    from audit_modular_sampling_compat import capture, write_new
    live_ids = {item["id"] for item in capture()["nodes"]} if args.live_core else None
    progress = json.loads(args.progress.read_text(encoding="utf-8")) if args.progress else None
    report = audit(live_ids=live_ids, progress=progress)
    write_new(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "routes"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
