"""Cold-Core HIGH-only S08 run from the authored-canvas LOW freeze."""

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

import run_modular_s08_fastv2_resume_gpu as resume  # noqa: E402
import run_modular_s08_saved_canvas_gpu as full  # noqa: E402

SCHEMA = "t8.modular-sampling.s08-authored-canvas-cold-high-real-gpu.v1"
FULL_STATUS = "authored_canvas_real_v2_mechanical_pass_not_quality_acceptance"
PASS_STATUS = "authored_canvas_cold_high_mechanical_pass_not_quality_acceptance"


def build_graph(receipt: dict) -> tuple[dict, str]:
    graph, source_sha = resume.build_resume_graph(receipt)
    if source_sha != full.dense.RESUME_SHA:
        raise ValueError("Pinned S08 authored HIGH-only candidate changed")
    original = json.loads(resume.CANDIDATE.read_bytes())
    graph["26"]["inputs"]["min_tokens"] = original["26"]["inputs"]["min_tokens"]
    graph["47"]["inputs"]["length"] = original["47"]["inputs"]["length"]
    graph["16"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/S08_FastV2_AuthoredCanvas/high_only_selected")
    expected = json.loads(json.dumps(original))
    expected["60"]["inputs"].update(artifact_path=receipt["path"],
                                     artifact_sha256=receipt["sha256"])
    expected["16"]["inputs"]["filename_prefix"] = graph["16"]["inputs"]["filename_prefix"]
    if graph != expected:
        raise ValueError("S08 HIGH-only graph changed beyond receipt and private output prefix")
    if graph["47"]["inputs"]["length"] != 73 or graph["26"]["inputs"]["min_tokens"] != 12288:
        raise ValueError("S08 HIGH-only authored Relay clock or profile gate changed")
    return graph, source_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=full.base.PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=90000)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    receipt, manifest, state = resume.verified_low(
        args.full_run_root, expected_status=FULL_STATUS)
    full_report = json.loads((args.full_run_root / "report.json").read_bytes())
    if full_report["candidate_sha256"] != full.dense.FULL_SHA:
        raise ValueError("S08 full candidate SHA is not the authored source")
    graph, source_sha = build_graph(receipt)
    readiness = resume.preflight(args, manifest, state, source_sha)
    ram = full.native_gpu._free_physical_mib()
    readiness["schema"] = SCHEMA + ".preflight"
    readiness["physical_free_mib"] = ram
    readiness["checks"]["system_free_ram_gate"] = ram is not None and ram >= args.min_free_ram_mib
    readiness["checks"]["full_candidate_unchanged"] = (
        full.base.shared._sha256_file(full.base.CANDIDATE) == full.dense.FULL_SHA)
    readiness["ready"] = all(readiness["checks"].values())
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = full.ROOT / f"{run_id}-cold-high"
    run_root.mkdir(parents=True, exist_ok=False)
    resume.copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2),
                                          encoding="utf8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(full.base.probe_resource_config(args.comfy_root, full.base.PROJECT),
                                indent=2), encoding="utf8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "candidate_sha256": source_sha,
              "low_artifact_path": receipt["path"], "low_artifact_sha256": receipt["sha256"],
              "preflight": readiness}
    try:
        with full.base.shared.IsolatedServer(args, run_root, "s08-authored-cold-high") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(full.base.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2),
                                             encoding="utf8")
        checks = resume.stage_checks(phase)
        report["checks"] = checks
        if checks["terminal_success"]:
            report["high_save"] = full.base._stage_save_receipt(phase, "51")
            checks["high_stage_file_sha"] = full.base._check_stage_file(run_root, report["high_save"])
            effects = full._effects(phase, ("high",))
            report["effects"] = effects["audits"]
            checks.update({key: value for key, value in effects["checks"].items()
                           if key.startswith("high_")})
            report["media"] = full._media(run_root, "high_only_selected")
            checks.update(report["media"]["checks"])
            checks["high_request_exact"] = (report["high_save"]["report"]["request_sha256"] ==
                                            full_report["high_save"]["report"]["request_sha256"])
            checks["high_state_exact"] = (report["high_save"]["report"]["state_sha256"] ==
                                          full_report["high_save"]["report"]["state_sha256"])
            checks["rgb24_exact"] = (report["media"]["rgb24_sha256"] ==
                                     full_report["media"]["rgb24_sha256"])
            checks["pcm_exact"] = (report["media"]["pcm_s16le_sha256"] ==
                                   full_report["media"]["pcm_s16le_sha256"])
        checks["source_candidate_unchanged"] = full.base.shared._sha256_file(resume.CANDIDATE) == source_sha
        checks["full_candidate_unchanged"] = full.base.shared._sha256_file(
            full.base.CANDIDATE) == full.dense.FULL_SHA
        checks["frozen_low_unchanged"] = full.base.shared._sha256_file(manifest) == receipt["sha256"].upper()
        checks["server_stopped"] = not full.base.shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = PASS_STATUS if all(checks.values()) else "fail"
        if report["status"] == "fail":
            raise RuntimeError("S08 authored-canvas HIGH-only mechanical gate failed")
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
