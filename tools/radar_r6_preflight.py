"""Read-only RADAR r6 P0 inventory; writes only a new owned evidence directory.

No model load, dependency installation, GPU initialization, remote execution,
service restart or Git mutation. Static Core capabilities are not qualification.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (".git/index", "__init__.py", "h3_t8/sampling.py", "meta.json", "features.json")
CORE_FILES = ("comfy/sd.py", "comfy/model_base.py", "comfy/ldm/minimax/model.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(piece)
    return value.hexdigest()


def git_read(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], encoding="utf8").strip()


def workflow_snapshot(root):
    entries = {p.relative_to(root).as_posix(): digest(p)
               for p in sorted((root / "examples/workflows").rglob("*.json"))}
    # Match the Windows fix's raw-byte multiset representation.
    multiset = "\n".join(sorted(v.upper() for v in entries.values())).encode()
    return {"count": len(entries), "raw_sha256_multiset": hashlib.sha256(multiset).hexdigest(),
            "files": entries}


def static_core_capabilities(core):
    text = (core / "comfy/sd.py").read_text(encoding="utf8")
    tree = ast.parse(text)
    dynamic_vae_layers = any(isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "MiniMaxH3VideoVAE"
        and any(arg.arg == "num_layers" and isinstance(arg.value, ast.Name)
                and arg.value.id == "minimax_layers" for arg in node.keywords)
        for node in ast.walk(tree))
    return {"scope": "source_inspection_not_weight_or_GPU_qualification",
            "h3_vae_weight_selected_layer_count": dynamic_vae_layers}


def snapshot(project, core):
    packages = {}
    for name in ("torch", "transformers", "diffusers", "triton-windows"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    metadata = {name: json.loads((project / name).read_text(encoding="utf8"))
                for name in ("meta.json", "features.json")}
    return {"schema": "t8.radar-r6.preflight.v1",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "project": str(project), "core": str(core),
            "local_head": git_read(project, "rev-parse", "HEAD"),
            "core_head": git_read(core, "rev-parse", "HEAD"),
            "dirty_status": git_read(project, "status", "--porcelain=v1"),
            "protected_sha256": {name: digest(project / name) for name in PROTECTED},
            "core_sha256": {name: digest(core / name) for name in CORE_FILES},
            "workflow_snapshot": workflow_snapshot(project),
            "metadata_versions": {name: value.get("version") for name, value in metadata.items()},
            "metadata_node_counts": {name: len(value.get("nodes", [])) for name, value in metadata.items()},
            "platform": platform.platform(), "packages": packages,
            "core_capabilities": static_core_capabilities(core),
            "actions": {"GPU_initialized": False, "model_loaded": False,
                        "dependencies_installed": False, "service_restarted": False,
                        "git_mutated": False},
            "qualification": "inventory_only_not_completion_or_quality"}


def save_snapshot(core, output):
    core, output = Path(core).resolve(strict=True), Path(output).resolve()
    owned = (ROOT / "artifacts/development/radar-r6-20261005").resolve()
    if output == owned or not output.is_relative_to(owned) or output.exists():
        raise ValueError("Use a new evidence subdirectory in the owned RADAR r6 root")
    report = snapshot(ROOT, core)
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    # Re-read protected data after evidence output; do not certify changed state.
    stable = ({name: digest(ROOT / name) for name in PROTECTED} == report["protected_sha256"]
              and {name: digest(core / name) for name in CORE_FILES} == report["core_sha256"]
              and workflow_snapshot(ROOT) == report["workflow_snapshot"])
    (output / "terminal.json").write_text(json.dumps({"protected_state_stable": stable,
        "workflow_count": report["workflow_snapshot"]["count"], "GPU_initialized": False}, indent=2), encoding="utf8")
    if not stable:
        raise RuntimeError("Protected sources or workflows changed during preflight")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = save_snapshot(args.core, args.output)
    print(json.dumps({"output": str(args.output), "head": result["local_head"],
        "workflows": result["workflow_snapshot"]["count"],
        "static_capabilities": result["core_capabilities"]}, ensure_ascii=False))
