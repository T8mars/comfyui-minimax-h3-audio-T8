"""Cold Native Dual HIGH-only from an exact successful owned LOW manifest.

This loads a frozen LOW and never includes its model, conditioning, sampler or
external effects. A separate Core renders the HIGH media and checks request,
stage tensor and decoded AV parity with the uninterrupted source.
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

import run_modular_s02_native_gpu as full  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s02-native-dual-real-cold-high-gpu.v1"
FULL_PASS = "real_native_dual_small_media_mechanical_pass_not_quality_acceptance"


def verified_low(full_root: Path, coarse: int, refine: int) -> tuple[dict, dict, Path, Path]:
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    source_pass = report.get("status") == FULL_PASS
    if not source_pass:
        # A prior controller had an overstrict *audit* order while Core had
        # already saved both stages/media successfully. Keep that raw failure;
        # accept only a separately appended corrected offline audit, then
        # independently re-evaluate its stage dependency evidence here.
        audit = json.loads((full_root / "stage-audit-v2.json").read_text(encoding="utf-8"))
        phase = json.loads((full_root / "phase.json").read_text(encoding="utf-8"))
        source_pass = (audit.get("schema") == full.SCHEMA + ".stage-audit.v2" and
                       audit.get("status") == "pass" and audit.get("run_root") == str(full_root) and
                       audit.get("coarse") == coarse and audit.get("refine") == refine and
                       audit.get("original_report_status") == report.get("status") and
                       all(full.stage_checks(phase, coarse, refine).values()) and
                       all(full.media_checks(full_root, 128, 64, 22).values()))
    if not source_pass or report.get("run_root") != str(full_root) or \
            report.get("coarse") != coarse or report.get("refine") != refine or \
            report.get("source_candidate_sha256") != \
            shared._sha256_file(full.candidate(coarse, refine)) or \
            report.get("test_dimensions") != [128, 64, 22]:
        raise ValueError("Source must be the exact successful owned Native Dual small full graph")
    receipt = report.get("low_save")
    if not isinstance(receipt, dict) or not common._artifact_sha_matches(full_root, receipt):
        raise ValueError("Source LOW receipt is missing or changed")
    if not isinstance(report.get("high_save"), dict) or \
            not common._artifact_sha_matches(full_root, report["high_save"]):
        raise ValueError("Source HIGH receipt is missing or changed")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source LOW path escaped the owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != f"dual_low_{coarse}" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source LOW is not an exact portable Native Dual stage")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source LOW tensor bytes changed")
    return report, receipt, manifest, state


def build_resume_graph(coarse: int, refine: int, receipt: dict) -> tuple[dict, str]:
    source = full.candidate(coarse, refine, "resume_effects")
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"22": "UNETLoader", "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "25": "MiniMaxH3NativeDualHandoffEXPT8",
                "26": "MiniMaxH3NativeDualStageSetupEXPT8",
                "29": "MiniMaxH3StageSamplerEXPT8",
                "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "51": "MiniMaxH3StageSaveEXPT8",
                "60": "MiniMaxH3StageLoadEXPT8",
                "71": "MiniMaxH3LoRACompatibilityLoaderT8Advanced"}
    if any(graph.get(key, {}).get("class_type") != name for key, name in required.items()) or \
            any(key in graph for key in ("1", "9", "10", "11", "12", "13", "40", "41",
                                         "43", "45", "50", "70")):
        raise ValueError("Saved Native Dual resume graph lost its HIGH-only boundary")
    if graph["22"]["inputs"]["unet_name"] != full.MODEL or \
            graph["71"]["inputs"]["lora_name"] != full.LORA or \
            graph["71"]["inputs"]["strength_model"] != 1. or \
            graph["23"]["inputs"]["model_name"] != full.UPSCALE or \
            graph["23"]["inputs"]["av_latent"] != ["60", 1] or \
            graph["60"]["inputs"]["expected_stage"] != f"dual_low_{coarse}" or \
            graph["26"]["inputs"]["stage"] != f"dual_high_{refine}" or \
            graph["25"]["inputs"]["first_pass_steps"] != str(coarse) or \
            graph["25"]["inputs"]["second_audio_source"] != "auto":
        raise ValueError("Saved Native Dual HIGH model/handoff contract changed")
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    graph["47"]["inputs"]["length"] = 22
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0.,
                                  end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S02_NativeDual_RealSplit/high_only_selected"
    return graph, source_sha


def copy_low(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.name != "manifest.json" or \
            target.exists() or (target.parent / "state.safetensors").exists():
        raise ValueError("Refusing ambiguous frozen Native Dual LOW destination")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state, target.parent / "state.safetensors")
    if shared._sha256_file(target.parent / "state.safetensors") != \
            receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied Native Dual LOW tensor SHA changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied Native Dual LOW manifest SHA changed")


def stage_checks(phase: dict, refine: int) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("60", "23", "22", "29", "51")
    present = all(node in executing for node in required)
    ordered = present and executing.index("60") < executing.index("23") < \
        executing.index("29") < executing.index("51") and executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_native_high_progress": progress == ["29"] * refine,
        "frozen_low_lift_independent_high_order": ordered,
        "no_low_execution": not any(node in executing for node in
                                ("1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "70")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--coarse", type=int, choices=(4, 20), required=True)
    parser.add_argument("--refine", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8874)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=70000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    source_report, receipt, manifest, state = verified_low(args.full_run_root, args.coarse, args.refine)
    graph, source_sha = build_resume_graph(args.coarse, args.refine, receipt)
    readiness = full.preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-native-dual-real-gpu-20260925" / \
        f"{run_id}-LOW{args.coarse}-HIGH{args.refine}-high-only"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "port": args.port,
              "coarse": args.coarse, "refine": args.refine, "candidate_sha256": source_sha,
              "source_full_original_report_status": source_report["status"],
              "low_artifact_path": receipt["path"], "low_artifact_sha256": receipt["sha256"],
              "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, "s02-native-dual-high-only") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase, args.refine)
        if report["checks"]["terminal_success"]:
            report["high_save"] = common._save_receipt(phase, "51")
            report["checks"]["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
            report["checks"]["copied_low_sha"] = common._artifact_sha_matches(run_root, receipt)
            report["checks"]["source_low_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
            report["checks"].update(full.media_checks(run_root, 128, 64, 22, prefix="high_only_selected"))
            report["checks"]["high_request_matches_uninterrupted"] = (
                report["high_save"]["report"]["request_sha256"] ==
                source_report["high_save"]["report"]["request_sha256"])
            report["checks"]["high_state_matches_uninterrupted"] = (
                report["high_save"]["report"]["state_sha256"] ==
                source_report["high_save"]["report"]["state_sha256"])
            full_media = full.media_path(args.full_run_root, "full_selected")
            cold_media = full.media_path(run_root, "high_only_selected")
            report["decoded_sha256"] = {
                name: {"full": full.decoded_stream_hash(full_media, name),
                       "cold": full.decoded_stream_hash(cold_media, name)}
                for name in ("video", "audio")}
            report["checks"]["decoded_video_matches_uninterrupted"] = (
                report["decoded_sha256"]["video"]["full"] == report["decoded_sha256"]["video"]["cold"])
            report["checks"]["decoded_audio_matches_uninterrupted"] = (
                report["decoded_sha256"]["audio"]["full"] == report["decoded_sha256"]["audio"]["cold"])
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(full.candidate(args.coarse, args.refine, "resume_effects")) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_high_only_request_state_decoded_av_parity_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("Native Dual cold HIGH-only graph failed a mechanical parity check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
