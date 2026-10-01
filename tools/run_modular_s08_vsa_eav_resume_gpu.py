"""Cold-Core trained-VSA HIGH-only probe from an exact external-EAV LOW freeze.

The saved HIGH-only candidate is copied, not edited. Prompt Relay is removed
explicitly from this private execution copy because the native VSA producer has
no per-query temporal-bias interface. No dense fallback is accepted as proof.
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

import run_modular_s08_fastv2_resume_gpu as resume  # noqa: E402
import run_modular_s08_vsa_eav_gpu as full  # noqa: E402

SCHEMA = "t8.modular-sampling.s08-vsa-eav-cold-high-gpu.v1"
FULL_STATUS = "small_canvas_real_vsa_eav_mechanical_pass_not_quality_acceptance"


def build_probe_graph(receipt: dict) -> tuple[dict, str]:
    graph, source_sha = resume.build_resume_graph(receipt)
    full.replace_relay_with_native_conditioning(graph, "24", "22", "47")
    if any("PromptRelay" in node["class_type"] for node in graph.values()):
        raise ValueError("Sparse HIGH-only probe still contains unsupported Relay")
    graph["26"]["inputs"].update(profile="trained_vsa_exp", min_tokens=0)
    graph["42"]["inputs"].update(mode="apply_exp", tau=4.0,
        start_video_progress=0.0, end_video_progress=1.0, g_hard_limit=3.0)
    return graph, source_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=resume.PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8864)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    receipt, manifest, state = resume.verified_low(args.full_run_root, expected_status=FULL_STATUS)
    source_report = json.loads((args.full_run_root / "report.json").read_text(encoding="utf8"))
    graph, source_sha = build_probe_graph(receipt)
    readiness = resume.preflight(args, manifest, state, source_sha)
    full_source_sha = full.base.shared._sha256_file(full.base.CANDIDATE)
    assets = full.base.preflight(args, full_source_sha)
    readiness["schema"] = SCHEMA + ".preflight"
    readiness["source_full_candidate_sha256"] = full_source_sha
    readiness["installed_assets"] = assets["assets"]
    readiness["checks"]["full_candidate_unchanged"] = (
        full_source_sha == source_report["candidate_sha256"])
    readiness["checks"]["installed_assets_present"] = assets["checks"]["all_assets_installed"]
    readiness["ready"] = all(readiness["checks"].values())
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (resume.PROJECT / "artifacts/development/modular-sampling-m1-v2-vsa-eav-20260924"
                / f"{run_id}-high-only")
    run_root.mkdir(parents=True, exist_ok=False)
    resume.copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(resume.probe_resource_config(args.comfy_root, resume.PROJECT), indent=2),
                     encoding="utf8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "resume_candidate_sha256": source_sha,
              "full_candidate_sha256": full_source_sha,
              "low_artifact_path": receipt["path"], "low_artifact_sha256": receipt["sha256"],
              "preflight": readiness}
    try:
        with resume.shared.IsolatedServer(args, run_root, "s08-vsa-eav-high-only") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(resume.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf8")
        checks = resume.stage_checks(phase)
        report["checks"] = checks
        if checks["terminal_success"]:
            report["high_save"] = full.base._stage_save_receipt(phase, "51")
            checks["high_artifact_file_sha"] = full.base._check_stage_file(run_root, report["high_save"])
            checks["high_request_matches_uninterrupted"] = (
                report["high_save"]["report"]["request_sha256"] ==
                source_report["high_save"]["report"]["request_sha256"])
            checks["high_state_matches_uninterrupted"] = (
                report["high_save"]["report"]["state_sha256"] ==
                source_report["high_save"]["report"]["state_sha256"])
            report["high_eav"] = full._audit(phase, "46")
            checks.update({"high_" + key: value for key, value in full._effect_checks(report["high_eav"]).items()})
            checks.update(full.base._media_checks(run_root, "high_only_selected"))
        checks["source_resume_candidate_unchanged"] = resume.shared._sha256_file(resume.CANDIDATE) == source_sha
        checks["source_full_candidate_unchanged"] = (
            resume.shared._sha256_file(full.base.CANDIDATE) == full_source_sha)
        checks["original_low_sha_unchanged"] = resume.shared._sha256_file(manifest) == receipt["sha256"].upper()
        checks["copied_low_sha_unchanged"] = resume.shared._sha256_file(
            run_root / "output" / "MiniMaxH3" / "stage_artifacts" / receipt["path"]) == receipt["sha256"].upper()
        checks["server_stopped"] = not resume.shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_high_only_real_vsa_eav_mechanical_pass_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S08 trained VSA/EAV HIGH-only mechanical qualification failed")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
