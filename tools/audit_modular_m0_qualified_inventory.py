"""Replay M0's frozen model menus while keeping source drifts visible.

This is a qualified diagnostic, never a replacement for the strict M0 gate.
Only verified installed filenames are hidden inside one fresh CPU capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.audit_modular_m0_additive_source import audit as source_audit
from tools.audit_modular_sampling_compat import capture, compare, write_new
from tools.diagnose_modular_menu_drift import audit_saved_menu_selections, diagnose


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/development/modular-sampling-m0-20260922/baseline.json"
LIVE = ROOT / "artifacts/development/modular-sampling-m0-resume-20260927/live-v2.json"


def _asset(path_roots: tuple[Path, ...], name: str) -> dict:
    matches = []
    for root in path_roots:
        base = root.resolve()
        candidate = (base / name).resolve()
        if candidate.is_relative_to(base) and candidate.is_file():
            stat = candidate.stat()
            matches.append({"root": str(base), "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns})
    if not matches:
        raise ValueError(f"Added menu filename is not installed under approved roots: {name}")
    return {"name": name, "locations": matches}


def replay(baseline: dict, live: dict, *, preimage_root: Path | None, lora_root: Path,
           controlnet_root: Path, model_patches_root: Path, vae_root: Path,
           taehv_root: Path, graph_root: Path) -> dict:
    if live.get("cuda_initialized") is not False:
        raise ValueError("Live M0 snapshot was not captured in CPU-only mode")
    live_exact = compare(baseline, live)
    source_ids = set(live_exact["source_changes_requiring_numerical_review"])
    if source_ids:
        if preimage_root is None:
            raise ValueError("Changed old-node sources require an exact frozen preimage")
        source = source_audit(baseline, live, preimage_root)
        if (source.get("status") != "pass_textual_additions_only_not_runtime_parity"
                or set(source.get("affected_old_nodes", [])) != source_ids):
            raise ValueError("Old-node source additions were not qualified")
    diagnosis = diagnose(baseline, live, lora_root=lora_root,
                         controlnet_root=controlnet_root,
                         model_patches_root=model_patches_root,
                         vae_root=vae_root, taehv_root=taehv_root)
    expected_source_issues = {
        ("old_source_changed_requires_numerical_review", node_id)
        for node_id in source_ids}
    if (not diagnosis["menu_additions"] or
            {(item["kind"], item.get("id")) for item in diagnosis["issues"]}
            != expected_source_issues):
        raise ValueError("Live drift is not exactly installed menus plus source-bound additions")
    saved = audit_saved_menu_selections(graph_root, baseline, live,
                                        diagnosis["menu_additions"])
    if saved["regression_status"] != "pass":
        raise ValueError("Frozen old graph lost a saved menu choice")
    menu_ids = {item["id"] for item in diagnosis["menu_additions"]}
    allowed = {("old_schema_changed", item) for item in menu_ids} | {
        ("old_source_changed_requires_numerical_review", item) for item in source_ids}
    if {(item["kind"], item.get("id")) for item in live_exact["violations"]} != allowed:
        raise ValueError("Live strict drift includes an unqualified violation")

    roots = {"loras": (lora_root,),
             "controlnet": (controlnet_root, model_patches_root),
             "vae": (vae_root,), "taehv": (taehv_root, vae_root)}
    names = {category: set() for category in roots}
    for item in diagnosis["menu_additions"]:
        names[item["category"]].update(item["added"])
    assets = {category: [_asset(roots[category], name) for name in sorted(added)]
              for category, added in names.items() if added}
    # TAEHV searches both taehv and vae. Fun Control searches both controlnet
    # and model_patches. Mask their one verified logical addition in either API.
    mask = {"loras": names["loras"], "controlnet": names["controlnet"],
            "model_patches": names["controlnet"], "vae": names["vae"],
            "taehv": names["taehv"]}
    import folder_paths

    original = folder_paths.get_filename_list
    calls = {category: 0 for category in mask}
    removed = {category: set() for category in mask}

    def frozen_menu(category: str):
        values = original(category)
        if category not in mask:
            return values
        calls[category] += 1
        removed[category].update(set(values) & mask[category])
        return [name for name in values if name not in mask[category]]

    folder_paths.get_filename_list = frozen_menu
    try:
        hermetic = capture()
    finally:
        folder_paths.get_filename_list = original
    if hermetic.get("cuda_initialized") is not False:
        raise ValueError("Same-inventory M0 capture initialized CUDA")
    for category, records in assets.items():
        for record in records:
            if _asset(roots[category], record["name"]) != record:
                raise ValueError("Installed menu asset changed during CPU replay")
    for category, expected in names.items():
        if not expected:
            continue
        observed = (removed["controlnet"] | removed["model_patches"]
                    if category == "controlnet" else removed[category])
        if observed != expected:
            raise ValueError(f"Frozen asset menu was not exercised: {category}")

    hermetic_strict = compare(baseline, hermetic)
    remaining = {(item["kind"], item.get("id"))
                 for item in hermetic_strict["violations"]}
    expected_remaining = {("old_source_changed_requires_numerical_review", item)
                          for item in source_ids}
    if remaining != expected_remaining:
        raise ValueError("Same-inventory capture retained unexpected old schema/file drift")
    return {
        "schema": "t8.modular-m0-qualified-inventory.v1",
        "status": ("pass_old_schema_same_inventory_source_review_still_required" if source_ids
                   else "pass_old_schema_and_direct_source_same_inventory"),
        "live_exact_status": live_exact["status"],
        "same_inventory_exact_status": hermetic_strict["status"],
        "same_inventory_remaining_violations": hermetic_strict["violations"],
        "frozen_old_nodes": hermetic_strict["old_node_count"],
        "frozen_old_json": hermetic_strict["old_json_count"],
        "menu_fields": len(diagnosis["menu_additions"]),
        "saved_menu_counts": saved["counts"],
        "source_bound_old_node_ids": sorted(source_ids),
        "direct_old_node_sources_exact": not source_ids,
        "installed_asset_files": assets,
        "capture_calls": calls,
        "removed_filenames": {key: sorted(value) for key, value in removed.items()},
        "cuda_initialized": hermetic["cuda_initialized"],
        "qualification": "Fresh CPU capture under the frozen filename inventory: old schemas and "
                         "old JSON return exactly. Any remaining direct source differences are "
                         "listed, never waived. Live strict M0 is unchanged. A baseline without "
                         "runtime_sources does not cover indirect helpers. File presence/metadata, "
                         "not weight content, was checked; no full execution, GPU or quality proof.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preimage-root", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--live", type=Path, default=LIVE)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "artifacts/development").resolve()):
        parser.error("Use a new private artifacts output")
    models = args.models_root.resolve()
    report = replay(json.loads(args.baseline.read_bytes()), json.loads(args.live.read_bytes()),
                    preimage_root=args.preimage_root.resolve() if args.preimage_root else None,
                    lora_root=models / "loras", controlnet_root=models / "controlnet",
                    model_patches_root=models / "model_patches",
                    vae_root=models / "vae", taehv_root=models / "taehv",
                    graph_root=ROOT)
    report["frozen_baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    report["live_snapshot_sha256"] = hashlib.sha256(args.live.read_bytes()).hexdigest()
    report["auditor_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(output, report)
    print(json.dumps({"status": report["status"], "menu_fields": report["menu_fields"],
                      "remaining": len(report["same_inventory_remaining_violations"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
