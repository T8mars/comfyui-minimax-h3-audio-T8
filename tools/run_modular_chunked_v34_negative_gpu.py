"""Reject three corrupted v3/v4 HIGH-only receipts before any sampler executes.

Default is read-only. --confirm-run starts one owned isolated Core and appends
failure evidence beside a previously verified real-asset control/apply run.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_chunked_v34_gpu as base  # noqa: E402


SCHEMA = "t8.modular-sampling.chunked-v34-negative-receipts.v1"
CASES = {
    "bad_file_sha": "file SHA-256 mismatch",
    "bad_external_manifest": "Native latent resume manifest mismatch",
    "path_traversal": "traversal path component",
}


def negative_graphs(variant: str, control: dict, receipt: dict) -> dict[str, dict]:
    spec = base.CONFIG[variant]
    graphs = {name: deepcopy(control) for name in CASES}
    load = spec["load"]
    graphs["bad_file_sha"][load]["inputs"]["expected_file_sha256"] = "0" * 64
    manifest = json.loads(receipt["manifest_json"])
    manifest["checkpoint_id"] = "intentionally_wrong_checkpoint_id"
    graphs["bad_external_manifest"][load]["inputs"]["expected_manifest_json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    graphs["path_traversal"][load]["inputs"]["checkpoint_path"] = (
        "../not_an_owned_checkpoint.safetensors")
    return graphs


def audit_rejection(variant: str, case: str, graph: dict, phase: dict) -> dict[str, bool]:
    spec = base.CONFIG[variant]
    terminal = phase.get("terminal") or {}
    data = terminal.get("data") or {}
    events = phase.get("events") or []
    return {
        "expected_execution_error": terminal.get("type") == "execution_error",
        "load_node_rejected": str(data.get("node_id")) == spec["load"],
        "value_error": data.get("exception_type") == "ValueError",
        "precise_reason": CASES[case] in str(data.get("exception_message", "")),
        "no_high_sampler_started": not any(
            str(event.get("node")) == spec["high"] and
            event.get("type") in {"executing", "progress"}
            for event in events),
        "no_first_pass_in_graph": spec["low"] not in graph,
        "no_first_pass_progress": not any(
            str(event.get("node")) == spec["low"] and
            event.get("type") == "progress" for event in events),
    }


def load_owned_control(variant: str, run_root: Path) -> tuple[dict, dict, dict, dict]:
    run_root = run_root.resolve()
    if run_root.parent != (base.ARTIFACTS / variant).resolve():
        raise ValueError("Negative probe requires an owned v3/v4 run")
    original = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((run_root / "audit.json").read_text(encoding="utf-8"))
    applied = json.loads((run_root / "apply-report.json").read_text(encoding="utf-8"))
    if (original.get("variant") != variant or
            audit.get("status") !=
            "small_canvas_real_assets_mechanical_pass_not_quality_acceptance" or
            applied.get("status") !=
            "small_canvas_high_only_eav_apply_observed_not_quality_acceptance" or
            not all(audit.get("checks", {}).values()) or
            not all(applied.get("checks", {}).values())):
        raise ValueError("Negative probe requires verified owned control and apply runs")
    receipt = original["receipt"]
    frozen = Path(receipt["absolute_path"]).resolve()
    control_media = Path(audit["media"]["path"]).resolve()
    applied_media = Path(applied["media"]["path"]).resolve()
    protected = {"frozen": (frozen, receipt["file_sha256"]),
                 "control_media": (control_media, audit["media"]["sha256"]),
                 "applied_media": (applied_media, applied["media"]["sha256"])}
    if any(not path.is_relative_to(run_root) or
           base._sha(path).upper() != expected.upper()
           for path, expected in protected.values()):
        raise ValueError("Owned frozen or media evidence changed")
    control = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    width, height, _ = original["test_canvas"]
    base.build_apply_graph(variant, control, receipt, width=width, height=height)
    return original, receipt, control, protected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v3", "v4"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8236)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.run_root = args.run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    _original, receipt, control, protected = load_owned_control(
        args.variant, args.run_root)
    existing = [name for case in CASES for name in
                (f"negative-{case}-prompt.json", f"negative-{case}-phase.json")]
    existing.append("negative-report.json")
    gpu = base.shared.gpu_memory_mib()
    checks = {
        "owned_port_free": not base.shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                              gpu["free_mib"] >= args.min_free_vram_mib),
        "no_prior_negative_evidence": not any(
            (args.run_root / name).exists() for name in existing),
        "source_candidates_unchanged": all(
            base._sha(base._path(args.variant, kind)) ==
            base.CONFIG[args.variant]["hashes"][index]
            for index, kind in enumerate(("freeze", "resume"))),
    }
    readiness = {"schema": SCHEMA + ".preflight", "variant": args.variant,
                 "run_root": str(args.run_root), "checks": checks,
                 "gpu": gpu, "ready": all(checks.values())}
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    graphs = negative_graphs(args.variant, control, receipt)
    args.host = "127.0.0.1"
    args.input_directory = args.run_root / "input"
    args.extra_model_paths_config = args.run_root / "paths.json"
    report = {"schema": SCHEMA, "variant": args.variant,
              "run_root": str(args.run_root), "status": "started",
              "preflight": readiness, "cases": {}}
    try:
        with base.shared.IsolatedServer(
                args, args.run_root, f"chunked-{args.variant}-negative") as server:
            report["owned_core_pid"] = server.process.pid
            for case, graph in graphs.items():
                base._write_new(args.run_root / f"negative-{case}-prompt.json", graph)
                phase = asyncio.run(base.pdd._submit_prompt_capture(
                    server=f"http://127.0.0.1:{args.port}", prompt=graph,
                    timeout_seconds=args.timeout_seconds))
                base._write_new(args.run_root / f"negative-{case}-phase.json", phase)
                case_checks = audit_rejection(args.variant, case, graph, phase)
                report["cases"][case] = {
                    "checks": case_checks,
                    "terminal_type": (phase.get("terminal") or {}).get("type"),
                    "error": ((phase.get("terminal") or {}).get("data") or {}).get(
                        "exception_message"),
                }
                if not all(case_checks.values()):
                    raise RuntimeError(f"{case} did not reject before HIGH sampling")
        report["integrity_checks"] = {
            "protected_files_unchanged": all(
                base._sha(path).upper() == expected.upper()
                for path, expected in protected.values()),
            "source_candidates_unchanged": all(
                base._sha(base._path(args.variant, kind)) ==
                base.CONFIG[args.variant]["hashes"][index]
                for index, kind in enumerate(("freeze", "resume"))),
            "owned_core_stopped": not base.shared.port_is_listening("127.0.0.1", args.port),
        }
        report["status"] = ("all_three_bad_receipts_rejected_before_high"
                            if all(report["integrity_checks"].values()) else "fail")
        return 0 if report["status"] != "fail" else 1
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        base._write_new(args.run_root / "negative-report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
