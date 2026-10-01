"""Run the saved separated FastH3 V2 LOW/HIGH graph with installed assets.

Default invocation only checks sources/assets/resources. ``--confirm-run``
starts one owned isolated Core and uses a reduced test canvas on a copied API
graph. Original candidate, accepted one-piece workflows and user Core stay
untouched. This is a mechanical run, not human quality acceptance.
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


SCHEMA = "t8.modular-sampling.s08-fastv2-real-split-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m1-v2-independent-loader-20260924/"
             "candidate-full-v2/FastH3_V2_Separate_LOW_HIGH_Save_Stages_EAV_Relay_EXP.api.json")
MODEL = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
UPSCALE = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def build_probe_graph(*, width: int = 128, height: int = 64, frames: int = 22) -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("FastH3 V2 test canvas must be positive and 32-aligned")
    source_sha = shared._sha256_file(CANDIDATE)
    original = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    graph = deepcopy(original)
    required = {"1": "UNETLoader", "10": "MiniMaxH3FastH3V2StageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "26": "MiniMaxH3FastH3V2StageSetupEXPT8",
                "29": "MiniMaxH3StageSamplerEXPT8", "50": "MiniMaxH3StageSaveEXPT8",
                "51": "MiniMaxH3StageSaveEXPT8"}
    if any(graph.get(key, {}).get("class_type") != value for key, value in required.items()):
        raise ValueError("Saved FastH3 V2 candidate lost a required separated stage")
    if graph["1"]["inputs"]["unet_name"] != MODEL or graph["22"]["inputs"]["unet_name"] != MODEL:
        raise ValueError("Saved FastH3 V2 LOW/HIGH model asset changed")
    if graph["23"]["inputs"]["model_name"] != UPSCALE:
        raise ValueError("Saved FastH3 V2 learned 3D handoff changed")
    if graph["13"]["inputs"]["stage_context"] != ["10", 3] or \
            graph["29"]["inputs"]["stage_context"] != ["26", 3]:
        raise ValueError("Saved FastH3 V2 stage context wiring changed")
    graph["9"]["inputs"].update(width=width, height=height)
    graph["10"]["inputs"]["min_tokens"] = 0
    graph["26"]["inputs"]["min_tokens"] = 0
    graph["40"]["inputs"]["length"] = frames
    graph["47"]["inputs"]["length"] = frames
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S08_FastV2_RealSplit/full_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "v2_model": models / "diffusion_models" / MODEL,
        "clip": models / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "trained_3d_upscaler": models / "latent_upscale_models" / UPSCALE,
    }
    gpu = shared.gpu_memory_mib()
    checks = {
        "candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {key: str(value) for key, value in assets.items()},
            "gpu": gpu, "checks": checks, "ready": all(checks.values())}


def _stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ["13", "22", "23", "29"]
    present = all(node in executing for node in (*required, "50", "51"))
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "low_four_then_high_four": progress == ["13"] * 4 + ["29"] * 4,
        "verified_low_before_independent_high_loader_and_trained_lift":
            present and [node for node in executing if node in required] == required,
        "stage_saves_after_respective_sampler": present and
            executing.index("13") < executing.index("50") and
            executing.index("29") < executing.index("51"),
    }


def _stage_save_receipt(phase: dict, node_id: str) -> dict:
    text = (phase.get("executed_outputs") or {}).get(node_id, {}).get("text")
    if not isinstance(text, list) or len(text) < 3:
        raise ValueError("FastH3 V2 StageSave did not expose path/SHA/report UI")
    if not text[0].startswith("artifact_path: ") or not text[1].startswith("artifact_sha256: "):
        raise ValueError("FastH3 V2 StageSave UI receipt changed")
    return {"path": text[0].removeprefix("artifact_path: "),
            "sha256": text[1].removeprefix("artifact_sha256: "),
            "report": json.loads(text[2])}


def _check_stage_file(run_root: Path, receipt: dict) -> bool:
    store = (run_root / "output" / "MiniMaxH3" / "stage_artifacts").resolve()
    path = (store / receipt["path"]).resolve()
    return path.is_relative_to(store) and path.is_file() and \
        shared._sha256_file(path).upper() == receipt["sha256"].upper()


def _media_checks(run_root: Path, stem: str = "full_selected") -> dict[str, bool]:
    output = run_root / "output" / "MiniMaxH3" / "S08_FastV2_RealSplit"
    files = list(output.glob(f"{stem}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one isolated FastH3 V2 MP4")
    video = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=60, check=False)
    return {
        "h264_22_frames_256x128": len(video_streams) == 1 and
                                   video_streams[0].get("codec_name") == "h264" and
                                   int(video_streams[0].get("nb_frames", 0)) == 22 and
                                   (video_streams[0].get("width"), video_streams[0].get("height")) == (256, 128),
        "one_aac_audio": len(audio_streams) == 1 and audio_streams[0].get("codec_name") == "aac",
        "full_decode": decode.returncode == 0,
        "nonempty_private_file": video.stat().st_size > 0 and video.is_relative_to(run_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8864)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_probe_graph()
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run:
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m1-v2-real-gpu-20260924" / f"{run_id}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "candidate_sha256": source_sha, "preflight": readiness,
              "test_canvas": [128, 64, 22], "status": "started", "run_root": str(run_root)}
    try:
        with shared.IsolatedServer(args, run_root, "s08-fastv2-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = _stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["low_save"] = _stage_save_receipt(phase, "50")
            report["high_save"] = _stage_save_receipt(phase, "51")
            report["checks"]["low_artifact_file_sha"] = _check_stage_file(run_root, report["low_save"])
            report["checks"]["high_artifact_file_sha"] = _check_stage_file(run_root, report["high_save"])
            report["checks"].update(_media_checks(run_root))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(CANDIDATE) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_v2_split_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("FastH3 V2 real split graph failed one or more mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
