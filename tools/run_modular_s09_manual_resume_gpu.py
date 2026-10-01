"""Cold S09 SECOND-only from one verified real-weight FIRST receipt.

Default verifies the completed source and current resources without running.
``--confirm-run`` copies the sealed FIRST state to a fresh owned Core and
checks SECOND request/tensor/decoded AV parity with the uninterrupted run.
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

import run_modular_s09_manual_gpu as full  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s09-manual-real-cold-second-gpu.v1"
FULL_PASS = "real_manual_two_stage_small_media_mechanical_pass_not_quality_acceptance"


def verified_first(full_root: Path, noise: str) -> tuple[dict, dict, Path, Path]:
    if noise not in full.NOISE:
        raise ValueError("Expected one S09 noise variant")
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != full.SCHEMA or report.get("status") != FULL_PASS or \
            report.get("noise") != noise or report.get("run_root") != str(full_root) or \
            report.get("source_candidate_sha256") != shared._sha256_file(full.candidate(noise)) or \
            report.get("test_dimensions") != [128, 64, 22] or \
            not all(report.get("checks", {}).values()):
        raise ValueError("Source must be the exact successful owned S09 small full graph")
    phase = json.loads((full_root / "phase.json").read_text(encoding="utf-8"))
    if not all(full.stage_checks(phase).values()) or \
            not all(full.media_checks(full_root, noise, 128, 64, 22).values()):
        raise ValueError("Source S09 execution or media changed after its report")
    receipt = report.get("first_save")
    if not isinstance(receipt, dict) or not common._artifact_sha_matches(full_root, receipt):
        raise ValueError("Source S09 FIRST receipt is missing or changed")
    if not isinstance(report.get("second_save"), dict) or \
            not common._artifact_sha_matches(full_root, report["second_save"]):
        raise ValueError("Source S09 SECOND receipt is missing or changed")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json":
        raise ValueError("Source S09 FIRST path escaped its owned stage store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "t8.modular-sampling.frozen-stage.v1" or \
            data.get("portable_identity") is not True or \
            data.get("stage_context", {}).get("stage") != "manual_first" or \
            data.get("state_file") != "state.safetensors":
        raise ValueError("Source S09 FIRST is not an exact portable stage")
    state = manifest.parent / "state.safetensors"
    if not state.is_file() or state.stat().st_size != data["state_bytes"] or \
            shared._sha256_file(state) != data["state_sha256"].upper():
        raise ValueError("Source S09 FIRST tensor bytes changed")
    return report, receipt, manifest, state


def build_resume_graph(noise: str, receipt: dict) -> tuple[dict, str]:
    source = full.candidate(noise, "resume_effects")
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"22": "UNETLoader", "24": "MiniMaxH3PromptRelayConditioningT8Advanced",
                "26": "MiniMaxH3ManualPassStageSetupEXPT8", "29": "MiniMaxH3StageSamplerEXPT8",
                "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "42": "MiniMaxH3StageEAVConfigEXPT8", "46": "MiniMaxH3StageEAVAuditEXPT8",
                "51": "MiniMaxH3StageSaveEXPT8", "60": "MiniMaxH3StageLoadEXPT8",
                "84": "MiniMaxH3StageNoiseEXPT8"}
    forbidden = {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "83"}
    expected_noise = "disabled" if noise == "NativeNoise" else "variance_preserving_blend"
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in required.items()) or \
            forbidden & graph.keys() or \
            graph["22"]["inputs"].get("unet_name") != full.MODEL or \
            "completed_stage" in graph["22"]["inputs"] or \
            graph["60"]["inputs"].get("expected_stage") != "manual_first" or \
            graph["26"]["inputs"].get("stage") != "manual_second" or \
            graph["26"]["inputs"].get("manual_sigmas") != full.MANUAL_SIGMAS or \
            graph["26"]["inputs"].get("av_latent") != ["60", 0] or \
            graph["29"]["inputs"].get("latent_image") != ["60", 0] or \
            graph["84"]["inputs"].get("mode") != expected_noise or \
            graph["42"]["inputs"].get("mode") != "report_only" or \
            graph["51"]["inputs"].get("stage_result") != ["29", 2]:
        raise ValueError("Saved S09 SECOND-only graph lost its cold boundary")
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    for key, value in (("80", 128), ("81", 64), ("82", 22)):
        graph[key]["inputs"]["value"] = value
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0.,
                                  end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = f"{full.OUTPUT}/{noise}_cold_selected"
    return graph, source_sha


def copy_first(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.name != "manifest.json" or \
            target.exists() or (target.parent / "state.safetensors").exists():
        raise ValueError("Refusing ambiguous frozen S09 FIRST destination")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state, target.parent / "state.safetensors")
    if shared._sha256_file(target.parent / "state.safetensors") != \
            receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied S09 FIRST tensor SHA changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied S09 FIRST manifest SHA changed")


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("60", "22", "24", "26", "84", "29", "51", "46")
    present = all(node in executing for node in required)
    ordered = present and executing.index("60") < executing.index("26") < \
        executing.index("29") < executing.index("51") < executing.index("46") and \
        executing.index("22") < executing.index("24") < executing.index("26") and \
        executing.index("84") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_manual_second_three_progress": progress == ["29"] * 3,
        "frozen_first_independent_second_order": ordered,
        "no_first_execution": not any(key in executing for key in
                                  ("1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "83")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=full.NOISE, required=True)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8871)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=70000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    source_report, receipt, manifest, state = verified_first(args.full_run_root, args.noise)
    graph, source_sha = build_resume_graph(args.noise, receipt)
    readiness = full.preflight(args, source_sha)
    readiness["schema"] = SCHEMA + ".preflight"
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-manual-real-gpu-20260925" / \
        f"{run_id}-{args.noise}-cold-second"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_first(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "full_run_root": str(args.full_run_root), "noise": args.noise, "port": args.port,
              "candidate_sha256": source_sha, "first_artifact_path": receipt["path"],
              "first_artifact_sha256": receipt["sha256"], "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, f"s09-{args.noise}-cold-second") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["second_save"] = common._save_receipt(phase, "51")
            report["checks"]["second_artifact_sha"] = common._artifact_sha_matches(run_root, report["second_save"])
            report["checks"]["copied_first_sha"] = common._artifact_sha_matches(run_root, receipt)
            report["checks"]["source_first_sha"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
            report["checks"].update(full.media_checks(run_root, args.noise, 128, 64, 22, cold=True))
            report["checks"]["second_request_matches_uninterrupted"] = (
                report["second_save"]["report"]["request_sha256"] ==
                source_report["second_save"]["report"]["request_sha256"])
            report["checks"]["second_state_matches_uninterrupted"] = (
                report["second_save"]["report"]["state_sha256"] ==
                source_report["second_save"]["report"]["state_sha256"])
            complete_media = full.media_path(args.full_run_root, args.noise)
            resumed_media = full.media_path(run_root, args.noise, cold=True)
            report["decoded_sha256"] = {
                stream: {"full": full.native_gpu.decoded_stream_hash(complete_media, stream),
                         "cold": full.native_gpu.decoded_stream_hash(resumed_media, stream)}
                for stream in ("video", "audio")}
            for stream in ("video", "audio"):
                report["checks"][f"decoded_{stream}_matches_uninterrupted"] = (
                    report["decoded_sha256"][stream]["full"] == report["decoded_sha256"][stream]["cold"])
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(full.candidate(args.noise, "resume_effects")) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_manual_second_request_state_decoded_av_parity_mechanical_pass"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S09 cold SECOND-only graph failed mechanical parity")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
