"""Real-weight S09 FIRST20 -> independent manual SECOND3 GPU probe.

Default is a read-only asset/resource preflight. ``--confirm-run`` starts an
owned isolated Core with a small copy of one sealed NativeNoise/FreeNoise
candidate. Existing one-piece workflows and the user Core are untouched.
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

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s09-manual-real-split-gpu.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m2-manual-pass-20260925/candidates-v3"
MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
OUTPUT = "MiniMaxH3/S09_ManualPass_RealSplit"
NOISE = ("NativeNoise", "FreeNoise")
MANUAL_SIGMAS = "0.5,0.412,0.35,0"


def candidate(noise: str, kind: str = "save_effects") -> Path:
    if noise not in NOISE or kind not in ("save_effects", "resume_effects"):
        raise ValueError("Expected one S09 noise variant and saved/resume effects")
    return CANDIDATES / f"Manual_Pass_{kind}_{noise}_EXP.api.json"


def build_probe_graph(noise: str, *, width: int = 128, height: int = 64,
                      frames: int = 22) -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a positive, 32-aligned manual-pass test canvas")
    source = candidate(noise)
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    expected = {"1": "UNETLoader", "10": "MiniMaxH3ManualPassStageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "26": "MiniMaxH3ManualPassStageSetupEXPT8",
                "29": "MiniMaxH3StageSamplerEXPT8", "40": "MiniMaxH3PromptRelayPlanT8Advanced",
                "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "41": "MiniMaxH3StageEAVConfigEXPT8", "42": "MiniMaxH3StageEAVConfigEXPT8",
                "45": "MiniMaxH3StageEAVAuditEXPT8", "46": "MiniMaxH3StageEAVAuditEXPT8",
                "50": "MiniMaxH3StageSaveEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "83": "MiniMaxH3StageNoiseEXPT8", "84": "MiniMaxH3StageNoiseEXPT8"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in expected.items()):
        raise ValueError("Saved S09 candidate lost an independent stage or external effect")
    if any(graph[key]["inputs"].get("unet_name") != MODEL for key in ("1", "22")) or \
            graph["22"]["inputs"].get("completed_stage") != ["13", 2] or \
            graph["10"]["inputs"].get("stage") != "manual_first" or \
            graph["26"]["inputs"].get("stage") != "manual_second" or \
            any(graph[key]["inputs"].get("first_steps") != 20 or
                graph[key]["inputs"].get("manual_sigmas") != MANUAL_SIGMAS
                for key in ("10", "26")) or \
            graph["26"]["inputs"].get("av_latent") != ["45", 0] or \
            graph["29"]["inputs"].get("latent_image") != ["45", 0] or \
            graph["14"]["inputs"].get("av_latent") != ["46", 0] or \
            graph["50"]["inputs"].get("stage_result") != ["13", 2] or \
            graph["51"]["inputs"].get("stage_result") != ["29", 2]:
        raise ValueError("Saved S09 model/schedule/output handoff changed")
    expected_noise = "disabled" if noise == "NativeNoise" else "variance_preserving_blend"
    if any(graph[key]["inputs"].get("mode") != expected_noise for key in ("83", "84")) or \
            any(graph[key]["inputs"].get("mode") != "report_only" for key in ("41", "42")):
        raise ValueError("Saved S09 external noise/EAV mode changed")
    for key, value in (("80", width), ("81", height), ("82", frames)):
        graph[key]["inputs"]["value"] = value
    for key in ("41", "42"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                     end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = f"{OUTPUT}/{noise}_full_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / MODEL,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    gpu = shared.gpu_memory_mib()
    free_ram = native_gpu._free_physical_mib()
    checks = {
        "saved_candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "noise": args.noise,
            "candidate_sha256": source_sha, "assets": {name: str(path) for name, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def media_path(run_root: Path, noise: str, *, cold: bool = False) -> Path:
    if noise not in NOISE:
        raise ValueError("Expected one S09 noise variant")
    output = run_root / "output" / OUTPUT
    suffix = "cold_selected" if cold else "full_selected"
    files = list(output.glob(f"{noise}_{suffix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one private S09 MP4")
    return files[0]


def media_checks(run_root: Path, noise: str, width: int, height: int, frames: int,
                 *, cold: bool = False) -> dict[str, bool]:
    video = media_path(run_root, noise, cold=cold)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    picture = [item for item in streams if item.get("codec_type") == "video"]
    sound = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=90, check=False)
    return {
        "h264_expected_frames_geometry": len(picture) == 1 and picture[0].get("codec_name") == "h264"
                                         and int(picture[0].get("nb_frames", 0)) == frames
                                         and (picture[0].get("width"), picture[0].get("height")) ==
                                         (width, height),
        "one_aac_audio": len(sound) == 1 and sound[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": video.is_relative_to(run_root) and video.stat().st_size > 0,
    }


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("83", "13", "50", "45", "22", "26", "84", "29", "51", "46")
    present = all(node in executing for node in required)
    ordered = present and all(executing.index("13") < executing.index(node)
                              for node in ("50", "45", "22", "26", "29")) and \
        executing.index("45") < executing.index("26") < executing.index("29") < \
        executing.index("51") < executing.index("46") and \
        executing.index("22") < executing.index("26") and \
        executing.index("83") < executing.index("13") and \
        executing.index("84") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_first20_second3_progress": progress == ["13"] * 20 + ["29"] * 3,
        "independent_second_model_noise_and_handoff_order": ordered,
        "both_stage_saves_and_eav_audits_executed": present,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=NOISE, required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_probe_graph(args.noise, width=args.width,
                                          height=args.height, frames=args.frames)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-manual-real-gpu-20260925" / \
        f"{run_id}-{args.noise}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "noise": args.noise, "port": args.port, "source_candidate_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames]}
    try:
        with shared.IsolatedServer(args, run_root, f"s09-{args.noise}-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["first_save"] = common._save_receipt(phase, "50")
            report["second_save"] = common._save_receipt(phase, "51")
            report["checks"]["first_artifact_sha"] = common._artifact_sha_matches(run_root, report["first_save"])
            report["checks"]["second_artifact_sha"] = common._artifact_sha_matches(run_root, report["second_save"])
            report["checks"].update(media_checks(run_root, args.noise, args.width, args.height, args.frames))
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(candidate(args.noise)) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_manual_two_stage_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S09 manual-pass graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
