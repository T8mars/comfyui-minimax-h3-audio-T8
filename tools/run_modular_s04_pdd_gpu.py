"""Run one saved separated PDD 4+4 graph with installed assets.

Default invocation only checks the sealed source, assets, port and VRAM.
``--confirm-run`` starts an owned isolated Core on a reduced copied graph.
The old one-piece workflows and user Core are untouched. This is mechanical
execution and media validation, not human picture/audio quality acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s04-pdd-real-split-gpu.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m2-pdd-dynamic-20260923/candidates-v3"
ASSETS = {
    "FL2VA": ("minimax_h3_fl2va_int8_convrot.safetensors",
              "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors"),
    "Ref2VA": ("minimax_h3_ref2va_int8_convrot.safetensors",
               "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors"),
}


def candidate(base: str) -> Path:
    if base not in ASSETS:
        raise ValueError("Expected one registered PDD base variant")
    return CANDIDATES / f"PDD_{base}_4plus4_save_effects_EXP.api.json"


def build_probe_graph(base: str, *, width: int, height: int, frames: int) -> tuple[dict, str]:
    if base not in ASSETS or min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a registered PDD base and positive 32-aligned test canvas")
    source = candidate(base)
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    expected = {"1": "UNETLoader", "10": "MiniMaxH3PDDStageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "26": "MiniMaxH3PDDStageSetupEXPT8", "29": "MiniMaxH3StageSamplerEXPT8",
                "50": "MiniMaxH3StageSaveEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "90": "MiniMaxH3PDD8StepSetupT8Advanced",
                "92": "MiniMaxH3PDD8StepSetupT8Advanced"}
    if any(graph.get(key, {}).get("class_type") != node for key, node in expected.items()):
        raise ValueError("Saved PDD candidate lost a required independent stage")
    model, adapter = ASSETS[base]
    if any(graph[key]["inputs"]["unet_name"] != model for key in ("1", "22")):
        raise ValueError("Saved LOW/HIGH PDD MODEL changed")
    if any(graph[key]["inputs"]["pdd_lora_name"] != adapter or
           graph[key]["inputs"]["base_variant"] != base or
           graph[key]["inputs"]["strength"] != 1. for key in ("90", "92")):
        raise ValueError("Saved LOW/HIGH PDD adapter contract changed")
    expected_frames = 124 if base == "FL2VA" else 22
    if any(graph[key]["inputs"]["length"] != expected_frames for key in ("40", "47")):
        raise ValueError("Saved LOW/HIGH Relay Plans have different timelines")
    if graph["22"]["inputs"]["completed_stage"] != ["13", 2]:
        raise ValueError("Complete PDD graph lost its deferred HIGH MODEL load")
    if graph["13"]["inputs"]["stage_context"] != ["10", 3] or \
            graph["29"]["inputs"]["stage_context"] != ["26", 3]:
        raise ValueError("Saved PDD stage results lost their actual stage contexts")
    graph["9"]["inputs"].update(width=width, height=height)
    for key in ("40", "47"):
        graph[key]["inputs"]["length"] = frames
    for key in ("41", "42"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                     end_video_progress=1., g_hard_limit=3.)
    for key in ("97", "98"):
        if key in graph:
            graph[key]["inputs"].update(width=width, height=height)
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S04_PDD_RealSplit/full_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    model, adapter = ASSETS[args.base]
    assets = {
        "base": models / "diffusion_models" / model,
        "pdd_adapter": models / "loras" / adapter,
        "clip": models / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "trained_3d_upscaler": models / "latent_upscale_models" /
                              "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "reference_image": args.comfy_root / "input" / "10A.jpg",
    }
    gpu = shared.gpu_memory_mib()
    checks = {
        "saved_candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "base": args.base, "candidate_sha256": source_sha,
            "assets": {key: str(value) for key, value in assets.items()}, "gpu": gpu,
            "checks": checks, "ready": all(checks.values())}


def _save_receipt(phase: dict, node_id: str) -> dict:
    text = (phase.get("executed_outputs") or {}).get(node_id, {}).get("text")
    if not isinstance(text, list) or len(text) < 3 or \
            not text[0].startswith("artifact_path: ") or \
            not text[1].startswith("artifact_sha256: "):
        raise ValueError("PDD StageSave did not expose path/SHA/report")
    return {"path": text[0].removeprefix("artifact_path: "),
            "sha256": text[1].removeprefix("artifact_sha256: "),
            "report": json.loads(text[2])}


def _artifact_sha_matches(run_root: Path, receipt: dict) -> bool:
    store = (run_root / "output/MiniMaxH3/stage_artifacts").resolve()
    path = (store / receipt["path"]).resolve()
    return path.is_relative_to(store) and path.is_file() and \
        shared._sha256_file(path).upper() == receipt["sha256"].upper()


def _media_checks(run_root: Path, base: str, width: int, height: int, frames: int,
                  *, prefix: str = "full_selected") -> dict[str, bool]:
    output = run_root / "output/MiniMaxH3/S04_PDD_RealSplit"
    files = list(output.glob(f"{prefix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one private PDD MP4")
    video = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    picture = [item for item in streams if item.get("codec_type") == "video"]
    sound = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=90, check=False)
    factor = 2. if base == "FL2VA" else 1.5
    target_width = round(width * factor)
    target_height = round(height * factor)
    return {
        "h264_expected_frames_geometry": len(picture) == 1 and picture[0].get("codec_name") == "h264"
                                         and int(picture[0].get("nb_frames", 0)) == frames
                                         and (picture[0].get("width"), picture[0].get("height")) ==
                                         (target_width, target_height),
        "one_aac_audio": len(sound) == 1 and sound[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": video.is_relative_to(run_root) and video.stat().st_size > 0,
    }


def stage_execution_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("13", "50", "23", "22", "29", "51")
    ordered = all(node in executing for node in required)
    # Core may schedule trained lift and the independent HIGH loader in either
    # order. Both must follow LOW and precede HIGH; neither is allowed to pull
    # LOW back into a cold HIGH-only graph.
    dependencies = ordered and all(executing.index("13") < executing.index(node)
                                   for node in ("50", "23", "22", "29")) and \
        all(executing.index(node) < executing.index("29") for node in ("23", "22")) and \
        executing.index("29") < executing.index("51")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_low4_high4_progress": progress == ["13"] * 4 + ["29"] * 4,
        "independent_high_load_and_trained_lift_dependency_order": dependencies,
        "both_stage_saves_executed": ordered,
    }


def audit_existing(run_root: Path, base: str) -> dict:
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    phase = json.loads((run_root / "phase.json").read_text(encoding="utf-8"))
    if report.get("base") != base or not report.get("preflight", {}).get("ready"):
        raise ValueError("Only the matching owned PDD run can be audited")
    width, height, frames = report["test_dimensions"]
    checks = stage_execution_checks(phase)
    checks["low_artifact_sha"] = _artifact_sha_matches(run_root, report["low_save"])
    checks["high_artifact_sha"] = _artifact_sha_matches(run_root, report["high_save"])
    checks.update(_media_checks(run_root, base, width, height, frames))
    checks["source_candidate_unchanged"] = shared._sha256_file(candidate(base)) == \
        report["source_candidate_sha256"]
    checks["server_stopped"] = not shared.port_is_listening("127.0.0.1", report.get("port", 8865))
    result = {"schema": SCHEMA + ".stage-audit.v2", "base": base, "run_root": str(run_root),
              "original_report_status": report["status"], "checks": checks,
              "status": "pass" if all(checks.values()) else "fail",
              "note": ("Original process exit1/report fail retained. v1 audit required an invalid total "
                       "order between trained lift and HIGH model load; Core execution and saved media "
                       "were successful." if report["status"] == "fail" else
                       "Independent offline recheck of the successful full PDD run; not human quality acceptance.")}
    target = run_root / "stage-audit-v2.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing stage audit")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=tuple(ASSETS), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8865)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Offline audit of one completed owned run; starts no Core/GPU")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.audit_run_root is not None:
        result = audit_existing(args.audit_run_root.resolve(), args.base)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 1
    graph, source_sha = build_probe_graph(args.base, width=args.width,
                                          height=args.height, frames=args.frames)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run:
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-pdd-real-gpu-20260925" / f"{run_id}-{args.base}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "paths.json").write_text(
        json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA, "base": args.base, "port": args.port,
              "source_candidate_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames],
              "run_root": str(run_root), "status": "started"}
    try:
        with shared.IsolatedServer(args, run_root, f"s04-pdd-{args.base}-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_execution_checks(phase)
        if report["checks"]["terminal_success"]:
            report["low_save"] = _save_receipt(phase, "50")
            report["high_save"] = _save_receipt(phase, "51")
            report["checks"]["low_artifact_sha"] = _artifact_sha_matches(run_root, report["low_save"])
            report["checks"]["high_artifact_sha"] = _artifact_sha_matches(run_root, report["high_save"])
            report["checks"].update(_media_checks(run_root, args.base, args.width, args.height, args.frames))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(candidate(args.base)) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_pdd_split_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("PDD real separated graph failed one or more mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
