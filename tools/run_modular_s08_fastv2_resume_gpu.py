"""Cold-Core HIGH-only run from one exact real FastH3 V2 LOW freeze.

The saved HIGH-only candidate is copied and filled with the prior LOW path/SHA.
The two immutable LOW files are copied into a new owned output store. There is
no LOW sampler/model/conditioning in the graph, and no old workflow is edited.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s08_fastv2_gpu as full  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s08-fastv2-real-high-only-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m1-v2-independent-loader-20260924/"
             "candidate-resume-v2/FastH3_V2_Frozen_LOW_Resume_HIGH_EAV_Relay_EXP.api.json")


def verified_low(full_root: Path, *, expected_status: str =
                 "real_v2_split_small_media_mechanical_pass_not_quality_acceptance") -> tuple[dict, Path, Path]:
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != expected_status:
        raise ValueError("S08 full graph did not pass the real-asset mechanical gate")
    receipt = report["low_save"]
    if receipt["report"]["portable_identity"] is not True or \
            receipt["report"]["stage_context"]["stage"] != "low_0_4":
        raise ValueError("S08 LOW artifact is not a portable completed LOW stage")
    store = (full_root / "output" / "MiniMaxH3" / "stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or not manifest.is_file() or \
            shared._sha256_file(manifest) != receipt["sha256"].upper():
        raise ValueError("S08 LOW manifest changed or escaped the owned store")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or shared._sha256_file(state) != receipt["report"]["state_sha256"].upper():
        raise ValueError("S08 LOW tensor file changed")
    return receipt, manifest, state


def build_resume_graph(receipt: dict) -> tuple[dict, str]:
    source_sha = shared._sha256_file(CANDIDATE)
    graph = deepcopy(json.loads(CANDIDATE.read_text(encoding="utf-8")))
    if graph.get("60", {}).get("class_type") != "MiniMaxH3StageLoadEXPT8" or \
            graph.get("29", {}).get("class_type") != "MiniMaxH3StageSamplerEXPT8" or \
            graph.get("22", {}).get("class_type") != "MiniMaxH3StageUNETLoaderAfterEXPT8":
        raise ValueError("Saved S08 HIGH-only graph lost strict stage boundaries")
    if any(node in graph for node in ("1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50")):
        raise ValueError("Saved S08 HIGH-only graph unexpectedly contains a LOW branch")
    if graph["22"]["inputs"]["unet_name"] != full.MODEL or \
            graph["23"]["inputs"]["model_name"] != full.UPSCALE:
        raise ValueError("Saved S08 HIGH model/lift asset changed")
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    graph["26"]["inputs"]["min_tokens"] = 0
    graph["47"]["inputs"]["length"] = 22
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S08_FastV2_RealSplit/high_only_selected"
    return graph, source_sha


def preflight(args, manifest: Path, state: Path, source_sha: str) -> dict:
    gpu = shared.gpu_memory_mib()
    checks = {
        "candidate_sha_bound": bool(source_sha),
        "verified_low_files_present": manifest.is_file() and state.is_file(),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "source_sha256": source_sha,
            "low_manifest_sha256": shared._sha256_file(manifest), "gpu": gpu,
            "checks": checks, "ready": all(checks.values())}


def copy_low(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output" / "MiniMaxH3" / "stage_artifacts").resolve()
    target_manifest = (store / receipt["path"]).resolve()
    if not target_manifest.is_relative_to(store) or target_manifest.name != "manifest.json" or \
            target_manifest.exists() or (target_manifest.parent / "state.safetensors").exists():
        raise ValueError("Refusing an ambiguous S08 LOW copy destination")
    target_manifest.parent.mkdir(parents=True, exist_ok=False)
    target_state = target_manifest.parent / "state.safetensors"
    shutil.copy2(state, target_state)
    if shared._sha256_file(target_state) != receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied S08 LOW tensor SHA mismatch")
    shutil.copy2(manifest, target_manifest)
    if shared._sha256_file(target_manifest) != receipt["sha256"].upper():
        raise ValueError("Copied S08 LOW manifest SHA mismatch")


def stage_checks(phase: dict) -> dict[str, bool]:
    progress = [str(event.get("node")) for event in phase.get("events", []) if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in phase.get("events", []) if event.get("type") == "executing"]
    required = ("60", "22", "23", "29", "51")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_high_four_steps": progress == ["29"] * 4,
        "load_independent_model_trained_lift_high_save_in_order":
            [node for node in executing if node in required] == list(required),
        "no_low_execution": not any(node in executing for node in ("1", "9", "10", "11", "12", "13", "50")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
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
    receipt, manifest, state = verified_low(args.full_run_root)
    graph, source_sha = build_resume_graph(receipt)
    readiness = preflight(args, manifest, state, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m1-v2-real-gpu-20260924" / f"{run_id}-high-only"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "candidate_sha256": source_sha,
              "low_artifact_path": receipt["path"], "low_artifact_sha256": receipt["sha256"],
              "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, "s08-fastv2-high-only") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["high_save"] = full._stage_save_receipt(phase, "51")
            report["checks"]["high_artifact_file_sha"] = full._check_stage_file(run_root, report["high_save"])
            full_report = json.loads((args.full_run_root / "report.json").read_text(encoding="utf-8"))
            report["checks"]["high_request_matches_uninterrupted"] = (
                report["high_save"]["report"]["request_sha256"] ==
                full_report["high_save"]["report"]["request_sha256"])
            report["checks"]["high_state_matches_uninterrupted"] = (
                report["high_save"]["report"]["state_sha256"] ==
                full_report["high_save"]["report"]["state_sha256"])
            report["checks"].update(full._media_checks(run_root, "high_only_selected"))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(CANDIDATE) == source_sha
        report["checks"]["frozen_low_sha_unchanged"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_high_only_real_v2_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S08 cold HIGH-only graph failed one or more mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
