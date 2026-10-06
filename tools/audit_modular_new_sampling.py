"""Fail closed when new sampler/delegated multi-stage callsites lack split-route admission.

This is an AST/frontend-topology development tripwire, not proof that the
original S01-S29 routes are complete. The baseline freezes existing callsites;
new ones require explicit route, independently wired stage/effect nodes, a
resume graph and actual later runtime evidence. Director graph builders are
fingerprinted as complete AST functions because they construct sampler nodes
dynamically rather than calling sampler.sample() directly.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v11.json"
SCHEMA = "t8.modular-sampling.new-sampling-admission.v1"
SAMPLING_CALLEES = {"sample", "sample_custom", "sample_stage", "sample_low",
                    "sample_high", "run_worker", "_sample_one_segment",
                    "_sample_prepared_segment", "sample_head", "sample_tail", "_native_stage"}
MULTISTAGE = re.compile(r"two.?pass|dual.?model|second.?pass|pass.?2|progressive|restart|multi.?stage|high.?stage", re.I)
DELEGATE = re.compile(r"^(?:sample|run|execute|generate|refine|continue|process)(?:_[a-z0-9_]+)?$")
DIRECTOR_SAMPLER = re.compile(r"sampler|hyperflow.?split|two.?pass|dual.?model", re.I)


def _source_paths(root):
    root = Path(root)
    paths = [*root.glob("*.py"), *(root / "h3_t8").rglob("*.py")]
    for path in sorted(set(paths)):
        if path.stem != "__init__":
            yield path


def _is_director(path, root):
    relative = path.relative_to(root).as_posix().lower()
    name = path.stem.lower()
    return (name == "director" or name.startswith("director_")
            or name.startswith("nodes_director")
            or "/director/" in "/" + relative)


def _director_paths(root):
    yield from (path for path in _source_paths(root) if _is_director(path, Path(root)))


def _functions(tree):
    class Visitor(ast.NodeVisitor):
        scope = ()

        def __init__(self):
            self.values = []

        def _visit(self, node):
            previous = self.scope
            self.scope = (*previous, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.values.append((".".join(self.scope), node))
            self.generic_visit(node)
            self.scope = previous

        visit_ClassDef = _visit
        visit_FunctionDef = _visit
        visit_AsyncFunctionDef = _visit

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.values


def _own_calls(function):
    class Calls(ast.NodeVisitor):
        def __init__(self):
            self.values = []

        def visit_Call(self, node):
            self.values.append(node)
            self.generic_visit(node)

        # A nested function is a separate candidate with its own symbol.
        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    visitor = Calls()
    for statement in function.body:
        visitor.visit(statement)
    return visitor.values


def _director_graph_builder(function):
    """Find Director functions that create or delegate a sampling graph."""
    for node in ast.walk(function):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "class_type"
                        and isinstance(value, ast.Constant) and isinstance(value.value, str)
                        and DIRECTOR_SAMPLER.search(value.value)):
                    return True
        if isinstance(node, ast.Call):
            simple = ast.unparse(node.func).rsplit(".", 1)[-1]
            if (simple == "add" and node.args
                    and any(isinstance(item, ast.Constant) and isinstance(item.value, str)
                            and DIRECTOR_SAMPLER.search(item.value)
                            for item in ast.walk(node.args[0]))):
                return True
            if re.match(r"^apply_(?:.*(?:two_?pass|hyperflow|sampl)|.*_graph)$", simple, re.I):
                return True
    return False


def discover(root=ROOT):
    root = Path(root).resolve()
    result = []
    for path in _source_paths(root):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        for symbol, function in _functions(tree):
            multi = bool(MULTISTAGE.search(relative + " " + symbol))
            for call in _own_calls(function):
                callee = ast.unparse(call.func)
                simple = callee.rsplit(".", 1)[-1]
                if simple not in SAMPLING_CALLEES and not (multi and DELEGATE.match(simple)):
                    continue
                signature = hashlib.sha256(ast.dump(call, include_attributes=False).encode()).hexdigest()
                result.append({"path": relative, "symbol": symbol, "callee": callee,
                               "call_sha256": signature})
    for path in _director_paths(root):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        for symbol, function in _functions(tree):
            if _director_graph_builder(function):
                signature = hashlib.sha256(
                    ast.dump(function, include_attributes=False).encode()).hexdigest()
                result.append({"path": relative, "symbol": symbol,
                               "callee": "<director_graph_builder>", "call_sha256": signature})
    return sorted(result, key=lambda row: (row["path"], row["symbol"], row["callee"], row["call_sha256"]))


def _key(site):
    return (site["path"], site["symbol"], site["callee"], site["call_sha256"])


def reviewed_layout_sites(root, baseline):
    """Preserve v10; qualify absent dormant aliases by exact replacement AST.

    This is a finite reviewed deployment distinction, not discovery/capture
    of a smaller passing baseline. Present aliases remain fully checked.
    Missing actual runtime sources or altered full module AST fail closed.
    """
    root = Path(root).resolve()
    path = root / "tests/fixtures/modular_reviewed_runtime_layout_v1.json"
    if not path.is_file():
        return list(baseline["sites"]), []
    review = json.loads(path.read_text(encoding="utf8"))
    if (set(review) != {"schema", "reason", "retired_sites", "replacements"}
            or review["schema"] != "t8.modular-sampling.reviewed-runtime-layout.v1"
            or len(review["retired_sites"]) != 21 or len(review["replacements"]) != 9):
        raise ValueError("Invalid reviewed runtime layout")
    retired = {_key(site) for site in review["retired_sites"]}
    if len(retired) != 21:
        raise ValueError("Duplicate reviewed compatibility site")
    absent = set()
    for name, replacement in review["replacements"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Layout root alias must be an exact file basename")
        alias = root / name
        if alias.exists():
            if not alias.is_file() or alias.is_symlink():
                raise ValueError("Invalid existing root compatibility alias")
            continue
        if (set(replacement) != {"path", "ast_sha256"}
                or replacement["path"] != "h3_t8/" + name
                or not re.fullmatch(r"[0-9a-f]{64}", replacement["ast_sha256"])):
            raise ValueError("Invalid reviewed replacement module")
        target = root / replacement["path"]
        if not target.is_file() or target.is_symlink():
            raise ValueError("Reviewed actual runtime replacement is missing")
        tree = ast.parse(target.read_text(encoding="utf-8-sig"))
        actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        if actual != replacement["ast_sha256"]:
            raise ValueError("Reviewed actual runtime replacement AST changed: " + name)
        absent.add(name)
    omitted = [site for site in review["retired_sites"] if site["path"] in absent]
    if omitted and not retired <= {_key(site) for site in baseline["sites"]}:
        raise ValueError("Reviewed alias sites must all remain in the declared baseline")
    keys = {_key(site) for site in omitted}
    return [site for site in baseline["sites"] if _key(site) not in keys], omitted


def _workflow_graph(root, relative):
    path = (Path(root).resolve() / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Admission workflow must be an existing in-project JSON")
    graph = json.loads(path.read_text(encoding="utf8"))
    nodes, links = graph.get("nodes"), graph.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Admission requires an importable frontend workflow")
    by_id = {}
    for node in nodes:
        if (not isinstance(node, dict) or type(node.get("id")) is not int
                or not isinstance(node.get("type"), str) or node["id"] in by_id):
            raise ValueError("Admission workflow has malformed or duplicate nodes")
        by_id[node["id"]] = node["type"]
    edges = {node_id: set() for node_id in by_id}
    for link in links:
        if (not isinstance(link, list) or len(link) < 4
                or type(link[1]) is not int or type(link[3]) is not int
                or link[1] not in by_id or link[3] not in by_id):
            raise ValueError("Admission workflow has a malformed or dangling link")
        edges[link[1]].add(link[3])
    return by_id, edges


def _reaches(edges, source, target):
    pending, seen = [source], set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(edges[current] - seen)
    return False


def _effect_nodes(entry):
    """Require independent EAV/Relay controls on both stages and cold HIGH."""
    configured = entry.get("external_effect_nodes")
    if not isinstance(configured, dict) or set(configured) != {"workflow", "resume_workflow"}:
        return None
    full, resume = configured["workflow"], configured["resume_workflow"]
    names = {"eav", "prompt_relay"}
    if (not isinstance(full, dict) or set(full) != names
            or not isinstance(resume, dict) or set(resume) != names):
        return None
    if not all(isinstance(full[name], list) and len(full[name]) == 2 for name in names):
        return None
    full_ids = [node_id for name in ("eav", "prompt_relay") for node_id in full[name]]
    resume_ids = [resume[name] for name in ("eav", "prompt_relay")]
    if (len(full_ids) != 4 or any(type(node_id) is not int for node_id in full_ids)
            or len(set(full_ids)) != 4 or any(type(node_id) is not int for node_id in resume_ids)
            or len(set(resume_ids)) != 2):
        return None
    return configured


def _admission_issues(site, entry, root, routes, live_ids):
    issues = []
    route = routes.get(entry.get("route"))
    if route is None:
        return [{"kind": "unknown_route", "site": site}]
    stages = entry.get("stage_nodes")
    if (not isinstance(stages, list) or len(stages) != 2
            or any(not isinstance(stage, str) or not stage for stage in stages)
            or not set(stages) <= set(route.public_nodes)):
        issues.append({"kind": "missing_public_stages", "site": site})
        return issues
    if live_ids is None:
        issues.append({"kind": "live_stage_registration_not_checked", "site": site})
    elif not set(stages) <= set(live_ids):
        issues.append({"kind": "stage_not_registered", "site": site})
    if not isinstance(entry.get("variant"), str) or not entry["variant"].strip():
        issues.append({"kind": "missing_variant", "site": site})
    if entry.get("external_effects") != {"eav": "supported", "prompt_relay": "supported"}:
        issues.append({"kind": "external_effects_not_qualified", "site": site})
    effect_nodes = _effect_nodes(entry)
    if effect_nodes is None:
        issues.append({"kind": "missing_independent_effect_nodes", "site": site})
    for key in ("workflow", "resume_workflow"):
        try:
            by_id, edges = _workflow_graph(root, entry[key])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            issues.append({"kind": "missing_or_invalid_" + key, "site": site})
            continue
        if key == "workflow":
            ids = entry.get("workflow_stage_ids")
            valid = (isinstance(ids, list) and len(ids) == 2
                     and all(type(value) is int for value in ids) and ids[0] != ids[1]
                     and all(by_id.get(node_id) == kind for node_id, kind in zip(ids, stages))
                     and _reaches(edges, ids[0], ids[1]))
        else:
            high_id, source_id = entry.get("resume_stage_id"), entry.get("resume_source_id")
            source_type = entry.get("resume_source_type")
            valid = (type(high_id) is int and type(source_id) is int and high_id != source_id
                     and isinstance(source_type, str) and source_type.strip()
                     and by_id.get(high_id) == stages[1] and by_id.get(source_id) == source_type
                     and live_ids is not None and source_type in live_ids
                     and _reaches(edges, source_id, high_id)
                     and (stages[0] == stages[1] or stages[0] not in by_id.values())
                     and (stages[0] != stages[1]
                          or sum(kind == stages[0] for kind in by_id.values()) == 1))
        if not valid:
            issues.append({"kind": "invalid_stage_topology_" + key, "site": site})
        if effect_nodes is not None:
            configured = effect_nodes[key]
            if key == "workflow":
                targets = (ids if isinstance(ids, list) and len(ids) == 2
                           and all(type(node_id) is int for node_id in ids)
                           else (None, None))
                pairs = ((name, configured[name][index], target)
                         for name in ("eav", "prompt_relay")
                         for index, target in enumerate(targets))
                excluded = {node_id for node_id in targets if node_id is not None}
            else:
                pairs = ((name, configured[name], high_id)
                         for name in ("eav", "prompt_relay"))
                excluded = {high_id, source_id}
            for name, effect_id, target in pairs:
                node_type = by_id.get(effect_id)
                if (effect_id in excluded or node_type not in route.public_nodes
                        or live_ids is None or node_type not in live_ids
                        or not _reaches(edges, effect_id, target)):
                    issues.append({"kind": "invalid_" + name + "_topology_" + key,
                                   "site": site, "effect_node_id": effect_id,
                                   "stage_node_id": target})
    return issues


def audit(root, baseline, admissions, routes, live_ids=None):
    if baseline.get("schema") != SCHEMA or set(baseline) != {"schema", "scope", "sites"}:
        raise ValueError("Invalid new-sampling baseline schema")
    if admissions.get("schema") != SCHEMA or set(admissions) != {"schema", "entries"}:
        raise ValueError("Invalid new-sampling admissions schema")
    current = discover(root)
    baseline_sites, layout_omissions = reviewed_layout_sites(root, baseline)
    old = Counter(_key(site) for site in baseline_sites)
    now = Counter(_key(site) for site in current)
    new = now - old
    removed = old - now
    issues = [{"kind": "existing_sampling_site_changed_or_removed", "site": list(key), "count": count}
              for key, count in removed.items()]
    listed = {}
    for entry in admissions["entries"]:
        key = tuple(entry.get("site", ()))
        if len(key) != 4 or key in listed:
            raise ValueError("Duplicate or malformed admission site")
        listed[key] = entry
    for key, count in new.items():
        site = {"path": key[0], "symbol": key[1], "callee": key[2], "call_sha256": key[3]}
        entry = listed.get(key)
        if entry is None or count != 1:
            issues.append({"kind": "new_sampling_site_requires_admission", "site": site, "count": count})
        else:
            issues.extend(_admission_issues(site, entry, root, routes, live_ids))
    for key in listed:
        if key not in new:
            issues.append({"kind": "stale_or_unneeded_admission", "site": list(key)})
    return {"schema": SCHEMA, "status": "pass" if not issues else "fail",
            "declared_baseline_sites": len(baseline["sites"]),
            "reviewed_absent_root_alias_sites": len(layout_omissions),
            "baseline_sites": sum(old.values()), "current_sites": sum(now.values()),
            "new_sites": sum(new.values()), "removed_sites": sum(removed.values()),
            "issues": issues,
            "qualification": "AST and frontend-topology tripwire only; historical routes, delegated calls "
                             "not matching this scanner, Director dynamic branches not exercised by AST, runtime effect calls, Core import/execution "
                             "and future source completeness still require review."}


def guard(root, baseline):
    """Dependency-free CI check; local admission precedes baseline refresh."""
    return audit(root, baseline, {"schema": SCHEMA, "entries": []}, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "compare", "guard"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--admissions", type=Path)
    parser.add_argument("--live-core", action="store_true",
                        help="Required to qualify any new admitted public stage IDs")
    options = parser.parse_args()
    if options.mode != "guard" and options.output is None:
        parser.error("capture and compare require --output")
    if options.mode == "guard" and (options.admissions is not None or options.live_core):
        parser.error("guard is source-only; use compare for local admission qualification")
    target = options.output.resolve() if options.output is not None else None
    if target is not None:
        if target.exists() or not target.is_relative_to(ROOT / "artifacts"):
            parser.error("Use a new private artifacts output")
        target.parent.mkdir(parents=True, exist_ok=True)
    if options.mode == "capture":
        report = {"schema": SCHEMA, "scope": "production Python sampler calls and Director dynamic graph builders",
                  "sites": discover(ROOT)}
    elif options.mode == "guard":
        baseline = json.loads(options.baseline.read_text(encoding="utf8"))
        report = guard(ROOT, baseline)
    else:
        sys.path.insert(0, str(ROOT))
        from h3_t8.modular_sampling.catalogue import ROUTES
        baseline = json.loads(options.baseline.read_text(encoding="utf8"))
        admissions = ({"schema": SCHEMA, "entries": []} if options.admissions is None else
                      json.loads(options.admissions.read_text(encoding="utf8")))
        live_ids = None
        if options.live_core:
            from tools.audit_modular_sampling_compat import capture
            live_ids = {node["id"] for node in capture()["nodes"]}
        report = audit(ROOT, baseline, admissions, {route.id: route for route in ROUTES}, live_ids)
    if target is not None:
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"sites", "issues"}}, ensure_ascii=False))
    return 0 if report.get("status", "pass") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
