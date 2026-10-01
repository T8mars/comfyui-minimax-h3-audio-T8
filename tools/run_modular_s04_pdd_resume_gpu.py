"""Cold-Core HIGH-only PDD run from a verified, real saved LOW stage.

The source graph and prior LOW artifact are copied into an owned isolated Core.
This checks mechanical execution and full AV decode, not human picture quality.
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

import run_modular_s04_pdd_gpu as full  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s04-pdd-real-high-only-gpu.v1"


def verified_low(full_root: Path, base: str) -> tuple[dict, dict, Path, Path]:
    report = json.loads((full_root / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((full_root / "stage-audit-v2.json").read_text(encoding="utf-8"))
    if report.get("base") != base or audit.get("base") != base or audit.get("status") != "pass" or \
            not all(audit.get("checks", {}).values()) or report.get("run_root") != str(full_root):
        raise ValueError("Source full PDD run lacks a matching passing mechanical audit")
    receipt = report["low_save"]
    if receipt["report"]["portable_identity"] is not True or \
            receipt["report"]["stage_context"]["stage"] != "pdd_low_0_4":
        raise ValueError("Source LOW is not a portable PDD stage")
    store = (full_root / "output/MiniMaxH3/stage_artifacts").resolve()
    manifest = (store / receipt["path"]).resolve()
    state = manifest.parent / "state.safetensors"
    if not manifest.is_relative_to(store) or manifest.name != "manifest.json" or \
            not manifest.is_file() or not state.is_file() or \
            shared._sha256_file(manifest) != receipt["sha256"].upper() or \
            shared._sha256_file(state) != receipt["report"]["state_sha256"].upper():
        raise ValueError("Source LOW path or SHA changed")
    return report, receipt, manifest, state


def build_resume_graph(base: str, receipt: dict, dimensions: list[int]) -> tuple[dict, str]:
    source = full.CANDIDATES / f"PDD_{base}_4plus4_resume_effects_EXP.api.json"
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"22": "UNETLoader", "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "29": "MiniMaxH3StageSamplerEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "60": "MiniMaxH3StageLoadEXPT8"}
    if any(graph.get(key, {}).get("class_type") != value for key, value in required.items()) or \
            any(key in graph for key in ("1", "9", "10", "11", "12", "13", "40", "41", "50")):
        raise ValueError("Saved resume graph lost its independent HIGH-only boundary")
    model, adapter = full.ASSETS[base]
    if graph["22"]["inputs"]["unet_name"] != model or \
            graph["92"]["inputs"]["pdd_lora_name"] != adapter or \
            graph["92"]["inputs"]["base_variant"] != base or \
            graph["92"]["inputs"]["strength"] != 1. or \
            graph["23"]["inputs"]["model_name"] != \
            "minimax_h3_latent_upscaler_3d_fp16.safetensors":
        raise ValueError("Saved HIGH model, PDD adapter or learned lift changed")
    if graph["23"]["inputs"]["av_latent"] != ["60", 1] or \
            graph["60"]["inputs"]["expected_stage"] != "pdd_low_0_4":
        raise ValueError("HIGH graph does not consume the frozen LOW latent")
    width, height, frames = dimensions
    graph["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    graph["47"]["inputs"]["length"] = frames
    graph["42"]["inputs"].update(tau=.2, start_video_progress=0., end_video_progress=1.,
                                 g_hard_limit=3.)
    for key in ("97", "98"):
        if key in graph:
            graph[key]["inputs"].update(width=width, height=height)
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S04_PDD_RealSplit/high_only_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, manifest: Path, state: Path, source_sha: str) -> dict:
    model, adapter = full.ASSETS[args.base]
    assets = [
        args.comfy_root / "models/diffusion_models" / model,
        args.comfy_root / "models/loras" / adapter,
        args.comfy_root / "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        args.comfy_root / "models/vae/minimax_h3_video_vae_fp16.safetensors",
        args.comfy_root / "models/vae/minimax_h3_audio_vae_fp32.safetensors",
        args.comfy_root / "models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors",
        args.comfy_root / "input/10A.jpg",
    ]
    gpu = shared.gpu_memory_mib()
    checks = {
        "saved_resume_candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets),
        "source_low_files_present": manifest.is_file() and state.is_file(),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "gpu": gpu, "checks": checks, "ready": all(checks.values())}


def copy_low(receipt: dict, manifest: Path, state: Path, run_root: Path) -> None:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    target = (store / receipt["path"]).resolve()
    if not target.is_relative_to(store) or target.name != "manifest.json" or \
            target.exists() or (target.parent / "state.safetensors").exists():
        raise ValueError("Refusing an ambiguous PDD LOW copy destination")
    target.parent.mkdir(parents=True, exist_ok=False)
    target_state = target.parent / "state.safetensors"
    shutil.copy2(state, target_state)
    if shared._sha256_file(target_state) != receipt["report"]["state_sha256"].upper():
        raise ValueError("Copied PDD LOW state SHA changed")
    shutil.copy2(manifest, target)
    if shared._sha256_file(target) != receipt["sha256"].upper():
        raise ValueError("Copied PDD LOW manifest SHA changed")


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("60", "22", "23", "29", "51")
    order = all(node in executing for node in required) and \
        executing.index("60") < executing.index("23") < executing.index("29") < executing.index("51") and \
        executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "only_high_four_steps": progress == ["29"] * 4,
        "frozen_low_and_independent_model_before_high": order,
        "no_low_execution": not any(node in executing for node in
                                    ("1", "9", "10", "11", "12", "13", "40", "41", "50")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=tuple(full.ASSETS), required=True)
    parser.add_argument("--full-run-root", required=True, type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.full_run_root = args.full_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    source_report, receipt, manifest, state = verified_low(args.full_run_root, args.base)
    dimensions = source_report["test_dimensions"]
    graph, source_sha = build_resume_graph(args.base, receipt, dimensions)
    readiness = preflight(args, manifest, state, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-pdd-real-gpu-20260925" / \
        f"{run_id}-{args.base}-high-only"
    run_root.mkdir(parents=True, exist_ok=False)
    copy_low(receipt, manifest, state, run_root)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "base": args.base, "port": args.port,
              "run_root": str(run_root), "full_run_root": str(args.full_run_root),
              "candidate_sha256": source_sha, "low_artifact_path": receipt["path"],
              "low_artifact_sha256": receipt["sha256"], "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, f"s04-pdd-{args.base}-high-only") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["high_save"] = full._save_receipt(phase, "51")
            report["checks"]["high_artifact_sha"] = full._artifact_sha_matches(run_root, report["high_save"])
            report["checks"]["high_request_matches_uninterrupted"] = (
                report["high_save"]["report"]["request_sha256"] ==
                source_report["high_save"]["report"]["request_sha256"])
            report["checks"]["high_state_matches_uninterrupted"] = (
                report["high_save"]["report"]["state_sha256"] ==
                source_report["high_save"]["report"]["state_sha256"])
            report["checks"].update(full._media_checks(
                run_root, args.base, *dimensions, prefix="high_only_selected"))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(
            full.CANDIDATES / f"PDD_{args.base}_4plus4_resume_effects_EXP.api.json") == source_sha
        report["checks"]["source_low_sha_unchanged"] = shared._sha256_file(manifest) == receipt["sha256"].upper()
        report["checks"]["copied_low_sha_unchanged"] = full._artifact_sha_matches(run_root, receipt)
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("cold_high_only_real_pdd_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("PDD cold HIGH-only graph failed one or more mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
