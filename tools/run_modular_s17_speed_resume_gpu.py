"""S17 SPEED cold HIGH-only real-weight probe from an exact frozen LOW artifact.

The source candidate and full run remain immutable. Preflight is read-only;
``--confirm-run`` copies only the two explicit LOW artifact files into an
isolated output store and starts a separately owned Core.
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

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_modular_s17_speed_gpu as full_gpu  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s17-speed-cold-high-real-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m3-speed-s17-20260923/"
             "candidate-v4-2-resume1-effects/SPEED_2Stage_EAV_Relay_Resume1_DRAFT_api.json")
DEFAULT_FULL = (full_gpu.RUNS / "20260925T045905Z-two-stage-full")


def source_full(full_root: Path) -> tuple[dict, dict]:
    full_root = full_root.resolve(strict=True)
    if not full_root.is_relative_to(full_gpu.RUNS.resolve()):
        raise ValueError("S17 cold resume needs the owned S17 full run")
    original = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    if original.get("schema") != full_gpu.SCHEMA:
        raise ValueError("Not an S17 full-run report")
    if (original.get("status") ==
            "real_speed_two_stage_small_media_mechanical_pass_not_quality_acceptance"
            and all(original.get("checks", {}).values())):
        evidence = original
        evidence_path = full_root / "report.json"
    else:
        evidence_path = full_root / "audit-receipts-v2.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (evidence.get("schema") != full_gpu.SCHEMA + ".receipt-audit.v2"
                or evidence.get("original_report_sha256") !=
                shared._sha256_file(full_root / "report.json").lower()
                or not all(evidence.get("checks", {}).values())
                or evidence.get("status") !=
                "real_speed_two_stage_small_media_mechanical_pass_not_quality_acceptance"):
            raise ValueError("S17 full run lacks a bound passing receipt audit")
    low = full_gpu.speed_artifact(full_root, 0)
    high = full_gpu.speed_artifact(full_root, 1)
    if (low["sha256"] != evidence["low_save"]["sha256"]
            or high["sha256"] != evidence["high_save"]["sha256"]):
        raise ValueError("S17 full stage changed after receipt audit")
    return original, {"low": low, "high": high, "evidence_path": evidence_path}


def build_resume_graph(low: dict, *, width: int = 256, height: int = 128,
                       frames: int = 22) -> tuple[dict, str]:
    if width < 128 or height < 128 or width % 32 or height % 32 or frames < 2:
        raise ValueError("S17 small canvas needs width/height >=128, 32-aligned and >=2 frames")
    source_sha = shared._sha256_file(CANDIDATE)
    graph = deepcopy(json.loads(CANDIDATE.read_text(encoding="utf-8")))
    expected = {"5": "MiniMaxH3SPEEDStageLoadEXPT8",
                "22": "UNETLoader", "23": "MiniMaxH3SPEEDSourceT8Advanced",
                "24": "MiniMaxH3SPEEDStageSetupEXPT8",
                "26": "MiniMaxH3SPEEDStageSampleEXPT8",
                "27": "MiniMaxH3SPEEDDCTTransitionEXPT8",
                "29": "MiniMaxH3SPEEDRelayApplyEXPT8",
                "30": "MiniMaxH3StageEAVConfigEXPT8",
                "31": "MiniMaxH3StageEAVApplyEXPT8",
                "32": "MiniMaxH3StageEAVAuditEXPT8", "82": "SaveVideo"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in expected.items()):
        raise ValueError("Saved S17 HIGH-only candidate changed its route")
    if (graph["24"]["inputs"].get("previous_spec") != ["5", 1]
            or graph["27"]["inputs"].get("completed_stage") != ["5", 0]
            or graph["26"]["inputs"].get("model") != ["31", 0]
            or graph["29"]["inputs"].get("model") != ["24", 0]
            or graph["31"]["inputs"].get("model") != ["29", 0]
            or graph["30"]["inputs"].get("mode") != "report_only"
            or graph["5"]["inputs"].get("next_stage_index") != 1
            or graph["22"]["inputs"].get("unet_name") != full_gpu.MODEL
            or any(key in graph for key in ("10", "11", "12", "14", "21"))):
        raise ValueError("Saved HIGH-only graph unexpectedly includes or rewires LOW")
    plan = graph["4"]["inputs"]
    if any(plan.get(key) != value for key, value in {
        "steps": 20, "scales": "0.5,1.0", "transition_mode": "manual_sigmas",
        "manual_transition_sigmas": "0.85", "transform": "dct",
        "fallback_policy": "error"}.items()):
        raise ValueError("Saved HIGH-only SPEED schedule changed")
    plan.update(width=width, height=height)
    graph["23"]["inputs"]["length"] = frames
    graph["28"]["inputs"]["length"] = frames
    graph["30"]["inputs"].update(tau=.2, start_video_progress=0.,
                                 end_video_progress=1., g_hard_limit=3.)
    graph["5"]["inputs"].update(artifact_path=low["path"],
                                artifact_sha256=low["sha256"])
    graph["82"]["inputs"]["filename_prefix"] = f"{full_gpu.OUTPUT}/cold_selected"
    graph["85"] = {"class_type": "MiniMaxH3SPEEDStageSaveEXPT8",
                   "inputs": {"stage_result": ["26", 1]}}
    graph["86"] = {"class_type": "PreviewAny", "inputs": {"source": ["85", 1]}}
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str, low: dict) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / full_gpu.MODEL,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    gpu, free_ram = shared.gpu_memory_mib(), native_gpu._free_physical_mib()
    checks = {
        "candidate_sha_bound": bool(source_sha),
        "low_receipt_bound": low["receipt"]["callbacks"] == list(range(14)),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {key: str(value) for key, value in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def stage_checks(phase: dict, high: dict, full_high: dict) -> dict[str, bool]:
    executing = [str(event.get("node")) for event in phase.get("events") or []
                 if event.get("type") == "executing"]
    required = ("5", "24", "27", "26", "32", "85", "82")
    present = all(node in executing for node in required)
    ordered = present and all(executing.index(a) < executing.index(b) for a, b in (
        ("5", "24"), ("24", "27"), ("27", "26"), ("26", "32"),
        ("26", "85"), ("32", "82")))
    receipt = high["receipt"]
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "load_dct_high_save_media_order": ordered,
        "no_low_stage_executed": not any(node in executing for node in ("12", "14", "21")),
        "exact_high6_with_external_effects": (
            receipt["callbacks"] == list(range(6))
            and receipt["sampler_known"] is True
            and receipt["effects"]["completed_forwards"] == 6
            and receipt["effects"]["planned_forwards"] == 6
            and receipt["effects"]["relay_attention_calls"] == 300
            and receipt["effects"]["status"] == "observed_report_only"),
        "same_high_request_and_native_av_output": (
            receipt["request_sha256"] == full_high["receipt"]["request_sha256"]
            and receipt["output"] == full_high["receipt"]["output"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=95000)
    parser.add_argument("--timeout-seconds", type=float, default=1200.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    original, source = source_full(args.full_run)
    width, height, frames = original["test_dimensions"]
    graph, source_sha = build_resume_graph(source["low"], width=width, height=height, frames=frames)
    readiness = preflight(args, source_sha, source["low"])
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = full_gpu.RUNS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-cold-high")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host, args.extra_model_paths_config = "127.0.0.1", paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "source_candidate_sha256": source_sha, "port": args.port,
              "full_run": str(args.full_run.resolve()), "full_evidence_sha256": shared._sha256_file(
                  source["evidence_path"]).lower(),
              "preflight": readiness, "test_dimensions": [width, height, frames]}
    try:
        low_root = args.full_run / "output/MiniMaxH3/modular_speed_stages"
        source_manifest = low_root / source["low"]["path"]
        target = run_root / "output/MiniMaxH3/modular_speed_stages" / source_manifest.parent.name
        target.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "speed-stage.safetensors"):
            shutil.copy2(source_manifest.parent / name, target / name)
        copied = full_gpu.speed_artifact(run_root, 0)
        if (copied["sha256"] != source["low"]["sha256"]
                or copied["report"]["state_sha256"] != source["low"]["report"]["state_sha256"]):
            raise ValueError("Copied LOW artifact differs from original")
        with shared.IsolatedServer(args, run_root, "s17-speed-cold-high") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (phase.get("terminal") or {}).get("type") == "execution_success":
            report["high_save"] = full_gpu.speed_artifact(run_root, 1)
            report["checks"] = stage_checks(phase, report["high_save"], source["high"])
            report["checks"].update(full_gpu.media_checks(run_root, width, height, frames, cold=True))
            full_media = next((args.full_run / "output" / full_gpu.OUTPUT).glob("full_selected*.mp4"))
            cold_media = next((run_root / "output" / full_gpu.OUTPUT).glob("cold_selected*.mp4"))
            report["decoded_sha256"] = {
                key: {"full": native_gpu.decoded_stream_hash(full_media, key),
                      "cold": native_gpu.decoded_stream_hash(cold_media, key)}
                for key in ("video", "audio")}
            report["checks"]["decoded_video_audio_exact"] = all(
                item["full"] == item["cold"] for item in report["decoded_sha256"].values())
        else:
            report["checks"] = {"terminal_success": False}
        report["checks"]["candidate_unchanged"] = shared._sha256_file(CANDIDATE) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_speed_cold_high_exact_parity_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S17 SPEED cold HIGH failed a mechanical check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
