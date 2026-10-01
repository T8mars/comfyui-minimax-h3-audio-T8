"""Private S08 dense FastH3 V2 Relay + EAV control, applied, and cold-HIGH probe.

The saved full/resume candidates are SHA-pinned and only copied into an owned
isolated Core. This is a small-canvas mechanical comparison, not image-quality
acceptance or proof that native sparse VSA supports Prompt Relay.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_modular_s08_fastv2_gpu as full  # noqa: E402
import run_modular_s08_fastv2_resume_gpu as resume  # noqa: E402
import run_modular_s08_vsa_eav_gpu as effect  # noqa: E402
from run_modular_motion_storage_gpu import _stream_hash  # noqa: E402

SCHEMA = "t8.modular-sampling.s08-dense-relay-eav-real-gpu.v1"
ROOT = full.PROJECT / "artifacts/development/modular-sampling-m1-v2-dense-eav-relay-20260925"
FULL_SHA = "7FB7A830F3AB3BC54C8AF6D94A044C926EDBA59D0EB70BFAE7545F804C6067E2"
RESUME_SHA = "91D403ECCA1BBB5804492987F49DE42D906C1744E2F3F7FD5DBB4E37F99EDE64"
PASS_STATUS = "dense_relay_eav_small_mechanical_pass_not_quality_acceptance"


def _configure_eav(graph: dict, node_ids: tuple[str, ...], mode: str) -> None:
    if mode not in ("report_only", "apply_exp"):
        raise ValueError("S08 EAV probe mode must be report_only or apply_exp")
    for node_id in node_ids:
        node = graph[node_id]
        if node["class_type"] != "MiniMaxH3StageEAVConfigEXPT8":
            raise ValueError("S08 external EAV config stage changed")
        node["inputs"].update(mode=mode, tau=4.0, start_video_progress=0.0,
                              end_video_progress=1.0, g_hard_limit=3.0)


def build_full(mode: str) -> tuple[dict, str]:
    graph, sha = full.build_probe_graph()
    if sha != FULL_SHA:
        raise ValueError("Pinned S08 full candidate changed")
    if any(graph[node]["class_type"] != "MiniMaxH3PromptRelayPlanT8Advanced"
           for node in ("40", "47")):
        raise ValueError("S08 full candidate lost independent Relay plans")
    if any(graph[node]["inputs"].get("profile") != "dense_compat_exp"
           for node in ("10", "26")):
        raise ValueError("S08 probe would not use the qualified dense profile")
    _configure_eav(graph, ("41", "42"), mode)
    return graph, sha


def build_cold(receipt: dict) -> tuple[dict, str]:
    graph, sha = resume.build_resume_graph(receipt)
    if sha != RESUME_SHA:
        raise ValueError("Pinned S08 HIGH-only candidate changed")
    if graph["47"]["class_type"] != "MiniMaxH3PromptRelayPlanT8Advanced" or \
            graph["26"]["inputs"].get("profile") != "dense_compat_exp":
        raise ValueError("S08 cold HIGH lost its dense Relay boundary")
    _configure_eav(graph, ("42",), "apply_exp")
    return graph, sha


def _effect_checks(audit: dict, mode: str) -> dict[str, bool]:
    dispatch = audit.get("v2_dispatch") or {}
    feta = audit.get("feta") or {}
    checks = {
        "observed_mode": audit.get("status") == "observed_" + mode,
        "four_real_forwards": audit.get("completed_forwards") == audit.get("planned_forwards") == 4,
        "separate_relay_executed": audit.get("relay_required") is True
                                   and audit.get("relay_attention_calls") == 200,
        "dense_not_silent_sparse_fallback": dispatch.get("profile") == "dense_compat_exp"
                                           and dispatch.get("actual_vsa_dispatched") is False
                                           and audit.get("sparse_producer_calls") == 0,
        "stage_clock_matches": audit.get("clock_match") is True,
        "eav_measured_on_all_forwards": feta.get("active_forward_count") == 4,
    }
    if mode == "apply_exp":
        checks["gain_above_identity"] = float(feta.get("g_max") or 0) > 1.0001
    return checks


def _media(run_root: Path, stem: str) -> dict:
    output = run_root / "output/MiniMaxH3/S08_FastV2_RealSplit"
    files = list(output.glob(f"{stem}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one S08 joint AV result")
    path = files[0]
    return {"path": str(path), "file_sha256": full.shared._sha256_file(path),
            "rgb24_sha256": _stream_hash(path, "video"),
            "pcm_s16le_sha256": _stream_hash(path, "audio")}


def _same_except_eav_mode(control: dict, applied: dict) -> bool:
    import copy

    observed = copy.deepcopy(control)
    for node in ("41", "42"):
        observed[node]["inputs"]["mode"] = "apply_exp"
    return observed == applied


def _preflight(args: argparse.Namespace, source_sha: str, *, low: tuple | None) -> dict:
    if low is None:
        result = full.preflight(args, source_sha)
    else:
        _receipt, manifest, state = low
        result = resume.preflight(args, manifest, state, source_sha)
    ram = native_gpu._free_physical_mib()
    result["schema"] = SCHEMA + ".preflight"
    result["physical_free_mib"] = ram
    result["checks"]["system_free_ram_gate"] = ram is not None and ram >= args.min_free_ram_mib
    result["ready"] = all(result["checks"].values())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("control", "applied", "cold"), required=True)
    parser.add_argument("--control-run-root", type=Path)
    parser.add_argument("--full-run-root", type=Path)
    parser.add_argument("--comfy-root", type=Path, default=full.PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8875)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=90000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    low = None
    source_report = None
    if args.kind == "cold":
        if args.full_run_root is None:
            parser.error("--cold requires --full-run-root")
        args.full_run_root = args.full_run_root.resolve()
        low = resume.verified_low(args.full_run_root, expected_status=PASS_STATUS)
        source_report = json.loads((args.full_run_root / "report.json").read_bytes())
        if source_report.get("kind") != "applied":
            raise ValueError("S08 cold HIGH requires an applied full source")
        graph, source_sha = build_cold(low[0])
    else:
        graph, source_sha = build_full("report_only" if args.kind == "control" else "apply_exp")
        if args.kind == "applied":
            if args.control_run_root is None:
                parser.error("--applied requires --control-run-root")
            args.control_run_root = args.control_run_root.resolve()
            source_report = json.loads((args.control_run_root / "report.json").read_bytes())
            if source_report.get("status") != PASS_STATUS or source_report.get("kind") != "control":
                raise ValueError("S08 applied requires a passing control")
            control_prompt = json.loads((args.control_run_root / "prompt.json").read_bytes())
            if not _same_except_eav_mode(control_prompt, graph):
                raise ValueError("S08 control and applied differ beyond external EAV modes")
    readiness = _preflight(args, source_sha, low=low)
    if args.kind == "cold":
        readiness["checks"]["full_candidate_sha_bound"] = (
            full.shared._sha256_file(full.CANDIDATE) == FULL_SHA)
    readiness["ready"] = all(readiness["checks"].values())
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / f"{run_id}-{args.kind}"
    run_root.mkdir(parents=True, exist_ok=False)
    if low is not None:
        resume.copy_low(*low, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(full.probe_resource_config(args.comfy_root, full.PROJECT), indent=2),
                     encoding="utf8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "kind": args.kind, "status": "started",
              "run_root": str(run_root), "candidate_sha256": source_sha,
              "preflight": readiness, "test_canvas": [128, 64, 22],
              "source_run_root": str(args.full_run_root if args.kind == "cold"
                                     else args.control_run_root) if source_report else None}
    try:
        with full.shared.IsolatedServer(args, run_root, f"s08-dense-relay-eav-{args.kind}") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(full.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2),
                                             encoding="utf8")
        checks = (resume.stage_checks(phase) if args.kind == "cold" else full._stage_checks(phase))
        report["checks"] = checks
        if checks["terminal_success"]:
            mode = "report_only" if args.kind == "control" else "apply_exp"
            for stage, save_id, audit_id in (("high", "51", "46"),) if args.kind == "cold" else (
                    ("low", "50", "45"), ("high", "51", "46")):
                report[stage + "_save"] = full._stage_save_receipt(phase, save_id)
                checks[stage + "_artifact_sha"] = full._check_stage_file(
                    run_root, report[stage + "_save"])
                report[stage + "_eav"] = effect._audit(phase, audit_id)
                checks.update({stage + "_" + key: value for key, value in
                               _effect_checks(report[stage + "_eav"], mode).items()})
            stem = "high_only_selected" if args.kind == "cold" else "full_selected"
            checks.update(full._media_checks(run_root, stem))
            report["media"] = _media(run_root, stem)
            if args.kind == "applied":
                checks["control_and_applied_picture_differ"] = (
                    report["media"]["rgb24_sha256"] != source_report["media"]["rgb24_sha256"])
            if args.kind == "cold":
                checks["high_request_exact"] = (report["high_save"]["report"]["request_sha256"] ==
                                                source_report["high_save"]["report"]["request_sha256"])
                checks["high_state_exact"] = (report["high_save"]["report"]["state_sha256"] ==
                                              source_report["high_save"]["report"]["state_sha256"])
                checks["rgb24_exact"] = (report["media"]["rgb24_sha256"] ==
                                         source_report["media"]["rgb24_sha256"])
                checks["pcm_exact"] = (report["media"]["pcm_s16le_sha256"] ==
                                       source_report["media"]["pcm_s16le_sha256"])
        checks["candidate_unchanged"] = full.shared._sha256_file(
            resume.CANDIDATE if args.kind == "cold" else full.CANDIDATE) == source_sha
        if low is not None:
            checks["source_full_candidate_unchanged"] = full.shared._sha256_file(full.CANDIDATE) == FULL_SHA
            checks["frozen_low_manifest_unchanged"] = full.shared._sha256_file(low[1]) == low[0]["sha256"].upper()
            checks["frozen_low_state_unchanged"] = (full.shared._sha256_file(low[2]) ==
                                                    low[0]["report"]["state_sha256"].upper())
        checks["server_stopped"] = not full.shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = PASS_STATUS if all(checks.values()) else "fail"
        if report["status"] == "fail":
            raise RuntimeError("S08 dense Relay+EAV mechanical gate failed")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
