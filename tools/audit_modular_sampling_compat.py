"""CPU-only, append-only compatibility baseline for the modular sampling rollout.

Capture before changing registration; compare after each milestone. No server,
model loading, sampling, migration, or edits to existing workflows are performed.
Dynamic model-file menus are captured exactly: an environment change is a review
item, not silently normalized into a passing compatibility result.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t8.modular-sampling.compatibility.v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_files(root: Path) -> dict:
    result = {}
    for directory in ("examples", "subgraphs", "tests/fixtures"):
        for path in sorted((root / directory).rglob("*.json")):
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def snapshot_runtime_sources(
    root: Path = ROOT, package_name: str = "_t8_modular_compat_capture",
) -> dict[str, str]:
    """Hash registration-time package imports, including non-node helpers.

    This is a rolling source identity gate, not a reconstruction of the older
    live environment from which the v1 node/schema baseline was captured.
    Runtime-only lazy imports remain outside this capture.
    """
    sources = {}
    root = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != package_name and not name.startswith(package_name + "."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        source_hash = digest(path)
        if relative in sources and sources[relative] != source_hash:
            raise ValueError(f"Conflicting loaded source identity: {relative}")
        sources[relative] = source_hash
    return dict(sorted(sources.items()))


def capture(root: Path = ROOT) -> dict:
    # Set CPU before any Core/model imports. Do not mutate the user's server.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Installed custom_nodes layout is optional for isolated release clones.
    # Otherwise use the caller's explicit Core PYTHONPATH, never a guessed drive.
    core_root = root.parents[1] if len(root.parents) > 1 else None
    if core_root is not None and (core_root / "comfy").is_dir():
        sys.path.insert(0, str(core_root))
    from comfy.cli_args import args
    args.cpu = True
    package_name = "_t8_modular_compat_capture"
    spec = importlib.util.spec_from_file_location(
        package_name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    classes = asyncio.run(module.comfy_entrypoint().get_node_list())
    entries = []
    for cls in classes:
        info = cls.GET_NODE_INFO_V1()
        # Round trip rejects non-JSON objects instead of inventing identities.
        info = json.loads(json.dumps(info, ensure_ascii=False, allow_nan=False))
        source = Path(sys.modules[cls.__module__].__file__).resolve()
        entries.append({"id": cls.define_schema().node_id, "info": info,
                        "source": source.relative_to(root).as_posix(),
                        "source_sha256": digest(source)})
    ids = [item["id"] for item in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Live registry contains duplicate node IDs")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Compatibility capture unexpectedly initialized CUDA")
    return {"schema": SCHEMA, "nodes": entries, "files": snapshot_files(root),
            "runtime_sources": snapshot_runtime_sources(root, package_name),
            "cuda_initialized": False,
            "entrypoint_sha256": digest(root / "__init__.py"),
            "qualification": "Live CPU schema, JSON and registration-time package-source identity only; "
                             "not numerical or GPU qualification."}


def compare(baseline: dict, current: dict) -> dict:
    if baseline.get("schema") != SCHEMA or current.get("schema") != SCHEMA:
        raise ValueError("Unsupported compatibility snapshot schema")
    previous_ids = [item["id"] for item in baseline["nodes"]]
    current_ids = [item["id"] for item in current["nodes"]]
    violations = []
    if current_ids[:len(previous_ids)] != previous_ids:
        violations.append({"kind": "registration_prefix_changed"})
    if len(current_ids) != len(set(current_ids)):
        violations.append({"kind": "duplicate_node_id"})
    current_nodes = {item["id"]: item for item in current["nodes"]}
    source_changes = []
    for node in baseline["nodes"]:
        now = current_nodes.get(node["id"])
        if now is None:
            violations.append({"kind": "old_node_missing", "id": node["id"]})
            continue
        if node["info"] != now["info"]:
            violations.append({"kind": "old_schema_changed", "id": node["id"]})
        if node["source"] != now["source"] or node["source_sha256"] != now["source_sha256"]:
            source_changes.append(node["id"])
            violations.append({"kind": "old_source_changed_requires_numerical_review", "id": node["id"]})
    for name, expected in baseline["files"].items():
        if current["files"].get(name) != expected:
            violations.append({"kind": "old_json_changed_or_missing", "path": name})
    baseline_runtime = baseline.get("runtime_sources")
    runtime_changes = []
    added_runtime = []
    if baseline_runtime is not None:
        if not isinstance(baseline_runtime, dict):
            raise ValueError("Invalid runtime source baseline")
        current_runtime = current.get("runtime_sources")
        if not isinstance(current_runtime, dict):
            violations.append({"kind": "runtime_source_capture_missing"})
        else:
            for name, expected in baseline_runtime.items():
                if current_runtime.get(name) != expected:
                    runtime_changes.append(name)
                    violations.append({"kind": "runtime_source_changed_requires_review",
                                       "path": name})
            added_runtime = sorted(set(current_runtime) - set(baseline_runtime))
    return {"schema": SCHEMA, "status": "pass" if not violations else "fail",
            "old_node_count": len(previous_ids), "current_node_count": len(current_ids),
            "old_json_count": len(baseline["files"]),
            "added_nodes": [value for value in current_ids if value not in previous_ids],
            "added_json": sorted(set(current["files"]) - set(baseline["files"])),
            "runtime_source_coverage": "compared" if baseline_runtime is not None else "unbaselined",
            "added_runtime_sources": added_runtime,
            "violations": violations, "source_changes_requiring_numerical_review": source_changes,
            "runtime_source_changes_requiring_review": runtime_changes,
            "qualification": "Schema/file/source identity only; changed old source fails pending numerical review. "
                             "A v1 baseline without runtime_sources does not cover indirect helpers; "
                             "execution-time lazy imports are outside this capture. "
                             "Unchanged source is not proof of runtime or GPU equivalence."}


def coverage_status(root: Path = ROOT) -> dict:
    inventory = root / "artifacts/research/two-pass-modular-audit-20260922/INVENTORY.json"
    routes = json.loads(inventory.read_text(encoding="utf-8"))["coverage_routes"]
    if [item["id"] for item in routes] != [f"S{i:02d}" for i in range(1, 30)]:
        raise ValueError("Expected the complete ordered 29-entry research coverage")
    dimensions = ("public_stages", "editable_workflow", "external_eav", "external_relay",
                  "stage_resume", "legacy_regression", "gpu_mechanical", "human_review")
    rows = []
    for route in routes:
        source = root / route["path"]
        rows.append({**route, "source_sha256": digest(source),
                     "rollout": {key: {"status": "pending", "evidence": []} for key in dimensions}})
    return {"schema": "t8.modular-sampling.coverage.v1", "research_sha256": digest(inventory),
            "note": "Pending means not qualified for this rollout; existing features are retained in current/surface.",
            "routes": rows}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Baselines and evidence are append-only; never overwrite a previous run.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "compare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Output must be a new evidence file")
    if options.mode == "compare" and options.baseline is None:
        parser.error("compare requires --baseline")
    current = capture()
    if options.mode == "capture":
        write_new(options.output, current)
        write_new(options.output.with_name(options.output.stem + "-coverage.json"), coverage_status())
        print(json.dumps({"status": "captured", "nodes": len(current["nodes"]),
                          "json_files": len(current["files"]), "output": str(options.output)}))
        return 0
    baseline = json.loads(options.baseline.read_text(encoding="utf-8"))
    report = compare(baseline, current)
    write_new(options.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
