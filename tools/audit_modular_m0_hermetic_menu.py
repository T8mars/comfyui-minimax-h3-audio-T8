"""Replay frozen M0 asset menus without editing old nodes or live menus.

The original live strict comparison remains authoritative and may stay red when
the host installs new LoRAs. This separate CPU audit verifies that *only* those
installed additions explain the drift, then captures current node schemas
under the frozen inventory in a fresh process. It never rewrites the baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1]
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from tools.audit_modular_sampling_compat import (  # noqa: E402
    SCHEMA, capture, compare, write_new,
)
from tools.diagnose_modular_menu_drift import (  # noqa: E402
    audit_saved_menu_selections, diagnose, installed_roots, installed_asset,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_additions(baseline: dict, live: dict, lora_root: Path,
                       graph_root: Path, controlnet_root: Path | None = None) -> tuple[dict[str, set[str]], dict, dict]:
    if baseline.get("schema") != SCHEMA or live.get("schema") != SCHEMA:
        raise ValueError("M0 snapshot schema differs")
    if live.get("cuda_initialized") is not False:
        raise ValueError("M0 live snapshot was not captured in CPU-only mode")
    diagnosis = diagnose(baseline, live, lora_root=lora_root, controlnet_root=controlnet_root)
    saved = audit_saved_menu_selections(graph_root, baseline, live,
                                        diagnosis["menu_additions"])
    allowed_ids = {entry["id"] for entry in diagnosis["menu_additions"]}
    exact = compare(baseline, live)
    if (diagnosis["structural_status"] != "pass"
            or saved["regression_status"] != "pass"
            or not diagnosis["menu_additions"]
            or any(item["kind"] != "old_schema_changed" or item["id"] not in allowed_ids
                   for item in exact["violations"])):
        raise ValueError("Live M0 drift is not purely additive installed asset menus")
    additions = {"loras": set(), "controlnet": set()}
    for entry in diagnosis["menu_additions"]:
        additions.setdefault(entry["category"], set()).update(entry["added"])
    if not any(additions.values()):
        raise ValueError("No installed asset additions to replay")
    return additions, diagnosis, saved


def replay(baseline: dict, live: dict, *, lora_root: Path, graph_root: Path,
           controlnet_root: Path | None = None) -> dict:
    additions, diagnosis, saved = verified_additions(
        baseline, live, lora_root, graph_root, controlnet_root)
    roots = installed_roots(lora_root, controlnet_root=controlnet_root)
    assets = {}
    for category, names in additions.items():
        for name in sorted(names):
            path = installed_asset(roots[category], name)
            key = name if category == "loras" else f"{category}:{name}"
            assets[key] = {"category": category, "name": name,
                           "path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}

    # Patch only categories that the old nodes enumerate, and only inside
    # this short-lived CPU capture process. Live Core/frontend behavior and
    # future LoRA discovery remain untouched.
    import folder_paths

    original = folder_paths.get_filename_list
    observations = {"calls": {category: 0 for category in additions},
                    "removed": {category: set() for category in additions}}

    def frozen_menu(category: str):
        values = original(category)
        if category not in additions or not additions[category]:
            return values
        observations["calls"][category] += 1
        observations["removed"][category].update(set(values) & additions[category])
        return [name for name in values if name not in additions[category]]

    folder_paths.get_filename_list = frozen_menu
    try:
        hermetic = capture()
    finally:
        folder_paths.get_filename_list = original
    for recorded in assets.values():
        path = installed_asset(roots[recorded["category"]], recorded["name"])
        if (str(path) != recorded["path"] or path.stat().st_size != recorded["bytes"]
                or _sha(path) != recorded["sha256"]):
            raise ValueError("Installed asset changed during frozen-inventory replay")
    if any(observations["calls"][category] <= 0 or observations["removed"][category] != names
           for category, names in additions.items() if names):
        raise ValueError("Frozen inventory was not exercised for every verified addition")
    if hermetic.get("cuda_initialized") is not False:
        raise ValueError("Hermetic M0 capture initialized CUDA")
    strict = compare(baseline, hermetic)
    return {
        "schema": "t8.modular-m0-hermetic-menu-audit.v1",
        "status": "pass_same_inventory_only" if strict["status"] == "pass" else "fail",
        "live_exact_status": diagnosis["exact_compatibility_status"],
        "live_structural_status": diagnosis["structural_status"],
        "live_saved_menu_regression_status": saved["regression_status"],
        "live_saved_menu_counts": saved["counts"],
        "masked_installed_additions": assets,
        "masked_field_count": len(diagnosis["menu_additions"]),
        "masked_old_node_count": len({row["id"] for row in diagnosis["menu_additions"]}),
        "frozen_inventory_lora_calls": observations["calls"]["loras"],
        "frozen_inventory_category_calls": observations["calls"],
        "hermetic_strict": {key: strict[key] for key in (
            "status", "old_node_count", "current_node_count", "old_json_count",
            "violations", "source_changes_requiring_numerical_review",
            "runtime_source_changes_requiring_review")},
        "hermetic_cuda_initialized": hermetic["cuda_initialized"],
        "qualification": "Exact legacy schema/file/source identity only under the old installed-asset "
                         "inventory, plus live additive-menu/saved-choice diagnostics. The "
                         "live strict baseline remains red; no GPU, numerical parity, "
                         "full old-workflow execution or human review is inferred.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or not output.is_relative_to(
            (PROJECT / "artifacts/development").resolve())):
        parser.error("Use a fresh private evidence path")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    baseline = json.loads(args.baseline.read_bytes())
    live = json.loads(args.live.read_bytes())
    report = replay(baseline, live, lora_root=args.lora_root,
                    graph_root=args.graph_root)
    write_new(output, report)
    print(json.dumps({"status": report["status"], "live_exact": report["live_exact_status"],
                      "hermetic": report["hermetic_strict"],
                      "masked": report["masked_installed_additions"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass_same_inventory_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
