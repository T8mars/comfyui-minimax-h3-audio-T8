"""Decide M0 source/schema compatibility when installed asset menus grow.

The frozen exact comparator is never changed or reported as passing when live
menus differ. This additional decision permits only verified installed-file
additions after a strict same-inventory replay; it is not workflow execution,
numerical parity, or an M0 completion certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.audit_modular_m0_hermetic_menu import replay
from tools.audit_modular_sampling_compat import capture, compare, write_new


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "t8.modular-m0-asset-policy.v1"


def _snapshot_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decide(baseline: dict, live: dict, *, lora_root: Path, graph_root: Path,
           controlnet_root: Path | None = None) -> dict:
    exact = compare(baseline, live)
    report = {"schema": SCHEMA, "exact_status": exact["status"],
              "exact_violations": exact["violations"],
              "baseline_sha256": _snapshot_sha256(baseline),
              "live_sha256": _snapshot_sha256(live),
              "old_node_count": exact["old_node_count"],
              "current_node_count": exact["current_node_count"],
              "old_json_count": exact["old_json_count"],
              "qualification": "M0 source/schema/frozen-JSON compatibility decision only. "
                               "Not full workflow execution, numerical/GPU parity, human review, "
                               "or overall M0 and two-pass rollout completion."}
    if live.get("cuda_initialized") is not False:
        return {**report, "status": "fail", "reason": "Live M0 capture was not CPU-only"}
    if exact["status"] == "pass":
        return {**report, "status": "pass_exact", "reason": "Frozen exact comparator passed"}
    try:
        hermetic = replay(baseline, live, lora_root=lora_root, graph_root=graph_root,
                          controlnet_root=controlnet_root)
    except (ValueError, OSError) as error:
        return {**report, "status": "fail", "reason": f"{type(error).__name__}: {error}"}
    if (hermetic["status"] != "pass_same_inventory_only"
            or hermetic["live_exact_status"] != "fail"
            or hermetic["live_structural_status"] != "pass"
            or hermetic["live_saved_menu_regression_status"] != "pass"
            or hermetic["hermetic_strict"]["status"] != "pass"):
        return {**report, "status": "fail", "reason": "Same-inventory replay did not pass",
                "hermetic": hermetic}
    return {**report, "status": "pass_verified_asset_additions",
            "reason": "Only installed asset menu additions differ from the frozen inventory",
            "hermetic": hermetic,
            "warning": "Exact live M0 remains fail; pre-existing unavailable saved choices "
                       "and execution/quality qualifications are separate gates."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    parser.add_argument("--controlnet-root", type=Path,
                        help="Installed ControlNet directory; defaults to the sibling of the LoRA directory")
    parser.add_argument("--graph-root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    live_path = output.with_name(output.stem + "-live.json")
    if (output.exists() or live_path.exists() or not output.is_relative_to(
            (PROJECT / "artifacts/development").resolve())):
        parser.error("Use a fresh private evidence path")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    baseline = json.loads(args.baseline.read_bytes())
    live = capture()
    report = decide(baseline, live, lora_root=args.lora_root, graph_root=args.graph_root,
                    controlnet_root=args.controlnet_root)
    write_new(live_path, live)
    write_new(output, {**report, "live_snapshot_path": str(live_path)})
    print(json.dumps({"status": report["status"], "exact_status": report["exact_status"],
                      "old_node_count": report["old_node_count"],
                      "current_node_count": report["current_node_count"],
                      "old_json_count": report["old_json_count"],
                      "reason": report["reason"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
