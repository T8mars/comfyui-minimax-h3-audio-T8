"""Run one saved Native Dual LOW4/20 -> independent HIGH3/4/5 graph.

The default is a read-only candidate/asset/resource preflight. An explicit
``--confirm-run`` starts one owned isolated Core with a reduced *copy* of the
saved candidate. Existing one-piece workflows and the user's Core are inert.
This is mechanical evidence, never human picture/audio acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s02-native-dual-real-split-gpu.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m2-native-dual-20260922/candidates-v2"
MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
UPSCALE = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def candidate(coarse: int, refine: int, kind: str = "save_effects") -> Path:
    if coarse not in (4, 20) or refine not in (3, 4, 5) or kind not in ("save_effects", "resume_effects"):
        raise ValueError("Native Dual candidate requires LOW4/20, HIGH3/4/5 and saved/resume effects")
    return CANDIDATES / f"Native_Dual_LOW{coarse}_HIGH{refine}_{kind}_EXP.api.json"


def build_probe_graph(coarse: int, refine: int, *, width: int, height: int,
                      frames: int) -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a positive 32-aligned Native Dual test canvas")
    source = candidate(coarse, refine)
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    expected = {"1": "UNETLoader", "10": "MiniMaxH3NativeDualStageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "25": "MiniMaxH3NativeDualHandoffEXPT8",
                "26": "MiniMaxH3NativeDualStageSetupEXPT8",
                "29": "MiniMaxH3StageSamplerEXPT8",
                "40": "MiniMaxH3PromptRelayPlanT8Advanced",
                "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "41": "MiniMaxH3StageEAVConfigEXPT8",
                "42": "MiniMaxH3StageEAVConfigEXPT8",
                "50": "MiniMaxH3StageSaveEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "71": "MiniMaxH3LoRACompatibilityLoaderT8Advanced"}
    if any(graph.get(key, {}).get("class_type") != name for key, name in expected.items()):
        raise ValueError("Saved Native Dual candidate lost a required split stage or external effect")
    if any(graph[key]["inputs"]["unet_name"] != MODEL for key in ("1", "22")) or \
            graph["22"]["inputs"].get("completed_stage") != ["13", 2] or \
            graph["71"]["inputs"]["lora_name"] != LORA or \
            graph["71"]["inputs"]["strength_model"] != 1. or \
            graph["23"]["inputs"]["model_name"] != UPSCALE or \
            graph["23"]["inputs"]["av_latent"] != ["45", 0]:
        raise ValueError("Saved Native Dual model or learned handoff changed")
    if coarse == 4 and (graph.get("70", {}).get("class_type") !=
                        "MiniMaxH3LoRACompatibilityLoaderT8Advanced" or
                        graph["70"]["inputs"].get("lora_name") != LORA or
                        graph["70"]["inputs"].get("strength_model") != 1.):
        raise ValueError("Saved LOW4 EMA B branch changed")
    if graph["10"]["inputs"]["stage"] != f"dual_low_{coarse}" or \
            graph["26"]["inputs"]["stage"] != f"dual_high_{refine}" or \
            graph["25"]["inputs"]["first_pass_steps"] != str(coarse) or \
            graph["25"]["inputs"]["second_audio_source"] != "auto" or \
            graph["25"]["inputs"]["second_audio_strength"] != 0. or \
            graph["40"]["inputs"]["length"] != graph["47"]["inputs"]["length"]:
        raise ValueError("Saved Native Dual stage/audio/Relay timeline contract changed")
    graph["9"]["inputs"].update(width=width, height=height)
    for key in ("40", "47"):
        graph[key]["inputs"]["length"] = frames
    for key in ("41", "42"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                    end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S02_NativeDual_RealSplit/full_selected"
    return graph, source_sha


def _free_physical_mib() -> int | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys // (1024 * 1024)


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / MODEL,
        "ema_b_lora": models / "loras" / LORA,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
        "trained_3d_upscaler": models / "latent_upscale_models" / UPSCALE,
    }
    gpu = shared.gpu_memory_mib()
    free_ram = _free_physical_mib()
    checks = {
        "saved_candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def media_path(run_root: Path, prefix: str) -> Path:
    output = run_root / "output/MiniMaxH3/S02_NativeDual_RealSplit"
    files = list(output.glob(f"{prefix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected one private Native Dual MP4")
    return files[0]


def media_checks(run_root: Path, width: int, height: int, frames: int,
                 *, prefix: str = "full_selected") -> dict[str, bool]:
    video = media_path(run_root, prefix)
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
                                         (width * 2, height * 2),
        "one_aac_audio": len(sound) == 1 and sound[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": video.is_relative_to(run_root) and video.stat().st_size > 0,
    }


def decoded_stream_hash(video: Path, stream: str) -> str:
    if stream == "video":
        options = ["-map", "0:v:0", "-c:v", "rawvideo", "-pix_fmt", "rgb24"]
    elif stream == "audio":
        options = ["-map", "0:a:0", "-c:a", "pcm_s16le", "-ar", "32000", "-ac", "2"]
    else:
        raise ValueError("Expected video or audio stream")
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), *options,
                             "-f", "hash", "-hash", "SHA256", "-"],
                            capture_output=True, text=True, timeout=90, check=True)
    if not result.stdout.startswith("SHA256="):
        raise ValueError("ffmpeg did not produce a decoded SHA256")
    return result.stdout.strip().removeprefix("SHA256=")


def stage_checks(phase: dict, coarse: int, refine: int) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("13", "50", "45", "23", "22", "29", "51")
    present = all(node in executing for node in required)
    # StageSave and EAV Audit independently consume LOW's completed result.
    # Core may evaluate either first. The learned lift depends on EAV Audit;
    # the deferred HIGH loader and lift both precede HIGH sampling.
    ordered = present and all(executing.index("13") < executing.index(node)
                              for node in ("50", "45", "22", "23", "29")) and \
        executing.index("45") < executing.index("23") < executing.index("29") < \
        executing.index("51") and executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_low_high_progress": progress == ["13"] * coarse + ["29"] * refine,
        "low_save_eav_lift_independent_high_order": ordered,
        "both_stage_saves_executed": present,
    }


def audit_existing(run_root: Path, coarse: int, refine: int) -> dict:
    """Append a corrected stage-order audit; never overwrite the raw report."""
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    phase = json.loads((run_root / "phase.json").read_text(encoding="utf-8"))
    if report.get("run_root") != str(run_root) or report.get("coarse") != coarse or \
            report.get("refine") != refine or not report.get("preflight", {}).get("ready"):
        raise ValueError("Only the matching owned Native Dual run can be audited")
    width, height, frames = report["test_dimensions"]
    checks = stage_checks(phase, coarse, refine)
    checks["low_artifact_sha"] = common._artifact_sha_matches(run_root, report["low_save"])
    checks["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
    checks.update(media_checks(run_root, width, height, frames))
    checks["source_candidate_unchanged"] = \
        shared._sha256_file(candidate(coarse, refine)) == report["source_candidate_sha256"]
    checks["server_stopped"] = not shared.port_is_listening("127.0.0.1", report["port"])
    result = {"schema": SCHEMA + ".stage-audit.v2", "run_root": str(run_root),
              "coarse": coarse, "refine": refine,
              "original_report_status": report["status"], "checks": checks,
              "status": "pass" if all(checks.values()) else "fail",
              "note": "Original exit1/report fail retained. StageSave and EAV Audit are independent LOW consumers; only their actual dependencies are ordered. Not human quality acceptance."}
    target = run_root / "stage-audit-v2.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite the Native Dual stage audit")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, choices=(4, 20), required=True)
    parser.add_argument("--refine", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8873)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Offline corrected audit of one completed owned full run; starts no Core")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.audit_run_root is not None:
        result = audit_existing(args.audit_run_root.resolve(), args.coarse, args.refine)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 1
    graph, source_sha = build_probe_graph(args.coarse, args.refine,
                                          width=args.width, height=args.height, frames=args.frames)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-native-dual-real-gpu-20260925" / \
        f"{run_id}-LOW{args.coarse}-HIGH{args.refine}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "port": args.port, "coarse": args.coarse, "refine": args.refine,
              "source_candidate_sha256": source_sha, "preflight": readiness,
              "test_dimensions": [args.width, args.height, args.frames]}
    try:
        with shared.IsolatedServer(args, run_root, "s02-native-dual-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase, args.coarse, args.refine)
        if report["checks"]["terminal_success"]:
            report["low_save"] = common._save_receipt(phase, "50")
            report["high_save"] = common._save_receipt(phase, "51")
            report["checks"]["low_artifact_sha"] = common._artifact_sha_matches(run_root, report["low_save"])
            report["checks"]["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
            report["checks"].update(media_checks(run_root, args.width, args.height, args.frames))
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(candidate(args.coarse, args.refine)) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_native_dual_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("Native Dual complete graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
