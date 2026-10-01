"""Cold RF RESTART-only from one exact real-weight two-stage BASE receipt.

The default is a read-only preflight. ``--confirm-run`` copies the sealed
BASE artifact into a fresh owned Core store. Request, output tensor and
decoded AV must match the uninterrupted graph; this is not quality review.
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

import run_modular_s29_rf_gpu as full  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s29-rf-standalone-real-cold-restart-gpu.v1"
SOURCE = full.CANDIDATES / "RF_standalone_resume_effects_EXP.api.json"
FULL_PASS = "real_rf_standalone_small_media_mechanical_pass_not_quality_acceptance"


def verified_base(full_root: Path, entry: str = "standalone") -> tuple[dict, dict, Path, Path]:
    if entry not in full.ENTRIES:
        raise ValueError("Expected one supported RF two-stage entry")
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    expected_status = (FULL_PASS if entry == "standalone" else
                       "real_rf_detail_mixer_small_media_mechanical_pass_not_quality_acceptance")
    if report.get("schema") != full.SCHEMA or report.get("status") != expected_status or \
            report.get("entry", "standalone") != entry or \
            report.get("run_root") != str(full_root) or \
            report.get("source_candidate_sha256") != shared._sha256_file(full.candidate(entry)) or \
            report.get("test_dimensions") != [128, 64, 22] or \
            not all(report.get("checks", {}).values()):
        raise ValueError("Source must be the exact successful owned RF small full graph")
    phase = json.loads((full_root / "phase.json").read_text(encoding="utf-8"))
    if not all(full.stage_checks(phase, entry).values()) or \
            not all(full.media_checks(full_root, 128, 64, 22,
                                      prefix=full.output_prefix(entry)).values()):
        raise ValueError("Source RF execution or media changed after the full-run report")
    receipt = report.get("base_save")
    if not isinstance(receipt, dict) or not common._artifact_sha_matches(full_root, receipt):
        raise ValueError("Source RF BASE receipt is missing or changed")
    if not isinstance(report.get("restart_save"), dict) or \
            not common._artifact_sha_matches(full_root, report["restart_save"]):
        raise ValueError("Source RF RESTART receipt is missing or changed")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source RF BASE path escaped its owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != "rf_base" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source RF BASE is not an exact portable stage")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source RF BASE tensor bytes changed")
    return report, receipt, manifest, state


def build_resume_graph(receipt: dict, entry: str = "standalone") -> tuple[dict, str]:
    source = full.candidate(entry, "resume_effects")
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"22": "UNETLoader", "26": "MiniMaxH3RFRestartStageSetupEXPT8",
                "29": "MiniMaxH3StageSamplerEXPT8", "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "42": "MiniMaxH3StageEAVConfigEXPT8", "46": "MiniMaxH3StageEAVAuditEXPT8",
                "51": "MiniMaxH3StageSaveEXPT8", "60": "MiniMaxH3StageLoadEXPT8",
                "90": "MiniMaxH3RFHandoffEXPT8"}
    forbidden = {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "88",
                 "89", "110", "111"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in required.items()) or \
            forbidden & graph.keys() or \
            graph["22"]["inputs"].get("unet_name") != full.MODEL or \
            "completed_stage" in graph["22"]["inputs"] or \
            graph["60"]["inputs"].get("expected_stage") != "rf_base" or \
            graph["90"]["inputs"].get("completed_av") != ["60", 0] or \
            graph["26"]["inputs"].get("restart_steps") != 3 or \
            graph["26"]["inputs"].get("restart_video_sigma") != .15 or \
            graph["42"]["inputs"].get("mode") != "report_only":
        raise ValueError("Saved RF resume graph lost its RESTART-only boundary")
    if entry == "detail_mixer":
        if graph.get("120", {}).get("class_type") != "MiniMaxH3ModelTimeBiasSamplerT8Advanced" or \
                graph.get("121", {}).get("class_type") != "MiniMaxH3SpatioTemporalGuidanceT8Advanced" or \
                graph["26"]["inputs"].get("model") != ["121", 0]:
            raise ValueError("Saved RF Detail Mixer lost independent RESTART Bias/STG")
    elif any(key in graph for key in ("120", "121")):
        raise ValueError("RF standalone resume unexpectedly gained Detail Mixer effects")
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    graph["82"]["inputs"]["value"] = 22
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0.,
                                  end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = \
        f"MiniMaxH3/S29_RF_RealSplit/{full.output_prefix(entry, cold=True)}"
    return graph, source_sha


def copy_base(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.name != "manifest.json" or \
            target.exists() or (target.parent / "state.safetensors").exists():
        raise ValueError("Refusing ambiguous frozen RF BASE destination")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state, target.parent / "state.safetensors")
    if shared._sha256_file(target.parent / "state.safetensors") != \
            receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied RF BASE tensor SHA changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied RF BASE manifest SHA changed")


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("60", "90", "22", "29", "51", "46")
    present = all(node in executing for node in required)
    ordered = present and executing.index("60") < executing.index("90") < \
        executing.index("29") < executing.index("51") and executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_restart_three_progress": progress == ["29"] * 3,
        "frozen_base_handoff_restart_order": ordered,
        "no_base_execution": not any(key in executing for key in
                                  ("1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "88")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--entry", choices=full.ENTRIES, default="standalone")
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    source_report, receipt, manifest, state = verified_base(args.full_run_root, args.entry)
    graph, source_sha = build_resume_graph(receipt, args.entry)
    readiness = full.preflight(args, source_sha)
    readiness["schema"] = SCHEMA + ".preflight"
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-rf-real-gpu-20260925" / \
        f"{run_id}-{args.entry}-cold-restart"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_base(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "entry": args.entry,
              "full_run_root": str(args.full_run_root), "port": args.port,
              "candidate_sha256": source_sha, "base_artifact_path": receipt["path"],
              "base_artifact_sha256": receipt["sha256"], "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, f"s29-rf-{args.entry}-cold-restart") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["restart_save"] = common._save_receipt(phase, "51")
            report["checks"]["restart_artifact_sha"] = common._artifact_sha_matches(run_root, report["restart_save"])
            report["checks"]["copied_base_sha"] = common._artifact_sha_matches(run_root, receipt)
            report["checks"]["source_base_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
            report["checks"].update(full.media_checks(
                run_root, 128, 64, 22, prefix=full.output_prefix(args.entry, cold=True)))
            report["checks"]["restart_request_matches_uninterrupted"] = (
                report["restart_save"]["report"]["request_sha256"] ==
                source_report["restart_save"]["report"]["request_sha256"])
            report["checks"]["restart_state_matches_uninterrupted"] = (
                report["restart_save"]["report"]["state_sha256"] ==
                source_report["restart_save"]["report"]["state_sha256"])
            complete_media = full.media_path(args.full_run_root, full.output_prefix(args.entry))
            resumed_media = full.media_path(run_root, full.output_prefix(args.entry, cold=True))
            report["decoded_sha256"] = {
                stream: {"full": full.native_gpu.decoded_stream_hash(complete_media, stream),
                         "cold": full.native_gpu.decoded_stream_hash(resumed_media, stream)}
                for stream in ("video", "audio")}
            for stream in ("video", "audio"):
                report["checks"][f"decoded_{stream}_matches_uninterrupted"] = (
                    report["decoded_sha256"][stream]["full"] == report["decoded_sha256"][stream]["cold"])
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(full.candidate(args.entry, "resume_effects")) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_rf_restart_request_state_decoded_av_parity_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("RF cold RESTART-only graph failed a mechanical parity check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
