"""Compare complete finalized Core schemas in independent CPU processes.

No prompt execution, server restart, weight load or count-only compatibility claim.
The reference must be an explicit checkout, not an invented historical fixture.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(project, core, output):
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path[:0] = [str(core), str(project)]
    sys.argv = ["radar-r6-schema", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("radar_r6_schema_pkg", project / "__init__.py",
                                                submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    schemas = [asdict(cls.GET_SCHEMA().get_v1_info(cls)) for cls in classes]
    ids = [value["name"] for value in schemas]
    metadata = json.loads((project / "features.json").read_text(encoding="utf8"))
    if len(ids) != len(set(ids)) or metadata["nodes"] != ids:
        raise RuntimeError("Actual registry order/uniqueness differs from declared features")
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("Schema comparison must be CPU-only")
    result = {"project": str(project), "count": len(ids), "ids": ids, "schemas": schemas,
              "cuda_initialized": False, "queued": False,
              "head": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()}
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--append-id", action="append", default=[],
                        help="Explicit newly added ID; old complete schemas must remain the exact ordered prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    owned = (ROOT / "artifacts/development/radar-r6-20261005").resolve()
    if output == owned or not output.is_relative_to(owned) or output.exists():
        raise ValueError("Use a new output inside this task's evidence root")
    if args.capture:
        capture(args.project.resolve(), args.core.resolve(), output)
        return
    if args.reference is None:
        raise ValueError("An explicit published reference checkout is required")
    output.mkdir(parents=True, exist_ok=False)
    values = []
    for label, project in (("reference", args.reference), ("candidate", args.project)):
        path = output / (label + ".json")
        command = [sys.executable, str(Path(__file__).resolve()), "--capture", "--project", str(project.resolve()),
                   "--core", str(args.core.resolve()), "--output", str(path)]
        with (output / (label + ".log")).open("x", encoding="utf8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            raise RuntimeError(f"{label} schema capture failed; retained {label}.log")
        values.append(json.loads(path.read_text(encoding="utf8")))
    reference, candidate = values
    differences = [candidate["ids"][index] for index, (old, new) in
                   enumerate(zip(reference["schemas"], candidate["schemas"])) if old != new]
    old_count = len(reference["ids"])
    old_exact = (reference["ids"] == candidate["ids"][:old_count]
                 and reference["schemas"] == candidate["schemas"][:old_count])
    additions = candidate["ids"][old_count:]
    exact = old_exact and additions == args.append_id
    receipt = {"status": "pass" if exact else "failed", "reference_head": reference["head"],
               "candidate_head": candidate["head"], "reference_count": reference["count"],
               "candidate_count": candidate["count"],
               "complete_finalized_schemas_exact": reference["schemas"] == candidate["schemas"],
               "compatibility_gate_passed": exact,
               "old_complete_ordered_schema_prefix_exact": old_exact,
               "declared_append_ids": args.append_id, "actual_append_ids": additions,
               "different_schema_ids": differences, "cuda_initialized": False, "queued": False}
    (output / "terminal.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps(receipt))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
