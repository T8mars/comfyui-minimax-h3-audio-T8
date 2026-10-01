"""Run a saved VDN DMD8/B50 -> independent native HIGH3/4/5 split graph.

Without --confirm-run this only checks source/assets/resources. The confirmed
run uses its own Core, output directory and shortened copied API graph. Old
one-piece workflows and the user's Core are not modified. Mechanical evidence
is not a human judgement of video or audio quality.
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

import run_modular_s04_pdd_gpu as common  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s05-vdn-dmd8-native4-real-split-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m2-vdn-relay-20260923/candidates-v2/"
             "VDNRelay_stage_dmd_8nfe_native4_save_relay_eav_EXP.api.json")
MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
HIGH_LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
VDN_REL = "diffusion_models/OpenVDN/vdn-minimax-h3/stage-dmd-step-250"
TRAINING = ("stage_dmd_8nfe", "stage_b_50nfe")
VDN_STAGE_DIR = {"stage_dmd_8nfe": "stage-dmd-step-250",
                 "stage_b_50nfe": "stage-b-step-2000"}


def candidate(refine: int, training: str = "stage_dmd_8nfe") -> Path:
    if refine not in (3, 4, 5):
        raise ValueError("Supported DMD8/native HIGH steps are 3, 4 or 5")
    if training not in TRAINING:
        raise ValueError("Expected DMD8 or B50 VDN training stage")
    return CANDIDATE.with_name(
        f"VDNRelay_{training}_native{refine}_save_relay_eav_EXP.api.json")


def schema(refine: int, training: str = "stage_dmd_8nfe") -> str:
    candidate(refine, training)
    if training == "stage_b_50nfe":
        return f"t8.modular-sampling.s03-vdn-b50-native{refine}-real-split-gpu.v1"
    return SCHEMA if refine == 4 else SCHEMA.replace("s05-vdn", "s03-vdn").replace(
        "native4", f"native{refine}")


def media_dir(refine: int, training: str = "stage_dmd_8nfe") -> str:
    candidate(refine, training)
    if training == "stage_b_50nfe":
        return "S03_VDN_B50_RealSplit"
    return "S05_VDN_RealSplit" if refine == 4 else "S03_VDN_RealSplit"


def build_probe_graph(*, width: int, height: int, frames: int,
                      refine: int = 4,
                      training: str = "stage_dmd_8nfe") -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a positive 32-aligned probe canvas")
    source = candidate(refine, training)
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    required = {"1": "UNETLoader", "10": "MiniMaxH3VDNStageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "UNETLoader",
                "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "26": "MiniMaxH3NativeStageBindEXPT8", "29": "MiniMaxH3StageSamplerEXPT8",
                "50": "MiniMaxH3StageSaveEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "70": "MiniMaxH3VDNModelComposerT8Advanced",
                "71": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                "110": "MiniMaxH3VDNRelayApplyEXPT8", "112": "MiniMaxH3VDNRelayAuditEXPT8"}
    if any(graph.get(key, {}).get("class_type") != name for key, name in required.items()):
        raise ValueError("Saved DMD8/native candidate lost a required separated stage")
    if any(graph[key]["inputs"]["unet_name"] != MODEL for key in ("1", "22")) or \
            graph["70"]["inputs"]["stage"] != training or \
            graph["70"]["inputs"]["vdn_root"] != "OpenVDN/vdn-minimax-h3" or \
            graph["71"]["inputs"]["lora_name"] != HIGH_LORA or \
            graph["71"]["inputs"]["strength_model"] != 1. or \
            graph["10"]["inputs"]["stage"] != "vdn_complete" or \
            graph["26"]["inputs"]["stage"] != "native_high" or \
            graph["23"]["inputs"]["model_name"] != \
            "minimax_h3_latent_upscaler_3d_fp16.safetensors":
        raise ValueError("DMD8/clean native HIGH model contract changed")
    if graph["23"]["inputs"]["av_latent"] != ["112", 0] or \
            graph["25"]["inputs"]["second_pass_audio_source"] != "first_pass" or \
            graph["92"]["inputs"]["steps"] != 8 or \
            graph["93"]["inputs"]["refine_steps"] != refine or \
            graph["110"]["inputs"]["mode"] != "apply_exp":
        raise ValueError("VDN handoff, locked audio, HIGH table or external Relay changed")
    prefix = f"VDNRelay/{training}/native{refine}/"
    if graph["50"]["inputs"].get("prefix") != prefix + "LOW" or \
            graph["51"]["inputs"].get("prefix") != prefix + "HIGH":
        raise ValueError("Saved VDN stage prefix changed")
    if graph["40"]["inputs"]["length"] != graph["140"]["inputs"]["length"]:
        raise ValueError("Saved LOW/HIGH Relay timeline differs")
    graph["9"]["inputs"].update(width=width, height=height)
    for key in ("40", "140"):
        graph[key]["inputs"]["length"] = frames
    for key in ("41", "42"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                    end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = \
        f"MiniMaxH3/{media_dir(refine, training)}/full_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str, *,
              include_low_assets: bool = True) -> dict:
    models = args.comfy_root / "models"
    training = getattr(args, "training", "stage_dmd_8nfe")
    if training not in TRAINING:
        raise ValueError("Expected DMD8 or B50 VDN training stage")
    vdn = models / "diffusion_models/OpenVDN/vdn-minimax-h3" / VDN_STAGE_DIR[training]
    assets = {
        "base": models / "diffusion_models" / MODEL,
        "native_high_lora": models / "loras" / HIGH_LORA,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
        "trained_3d_upscaler": models / "latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors",
    }
    if include_low_assets:
        assets.update({
            "vdn_linear_branch": vdn / "linear_branch/model.safetensors",
            "vdn_default_adapter": vdn / "adapters/default/adapter_model.safetensors",
            "vdn_spec": vdn / "model_spec.json",
            "vdn_metadata": vdn / "metadata.json",
        })
        if training == "stage_dmd_8nfe":
            assets["vdn_turbo_adapter"] = vdn / "adapters/turbo/adapter_model.safetensors"
    gpu = shared.gpu_memory_mib()
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    memory = MemoryStatus()
    memory.dwLength = ctypes.sizeof(memory)
    physical_free_mib = (memory.ullAvailPhys // (1024 * 1024)
                         if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) else None)
    checks = {
        "candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": physical_free_mib is not None and
                                physical_free_mib >= args.min_free_ram_mib,
    }
    return {"schema": schema(getattr(args, "refine", 4), training) + ".preflight",
            "candidate_sha256": source_sha,
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "physical_free_mib": physical_free_mib,
            "checks": checks, "ready": all(checks.values())}


def _media_checks(run_root: Path, width: int, height: int, frames: int,
                  *, prefix: str = "full_selected", refine: int = 4,
                  training: str = "stage_dmd_8nfe") -> dict[str, bool]:
    output = run_root / "output/MiniMaxH3" / media_dir(refine, training)
    files = list(output.glob(f"{prefix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one private VDN full MP4")
    video = files[0]
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


def stage_checks(phase: dict, refine: int = 4,
                 training: str = "stage_dmd_8nfe") -> dict[str, bool]:
    candidate(refine, training)
    events = phase.get("events") or []
    progress = [str(event.get("node")) for event in events if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in events if event.get("type") == "executing"]
    required = ("13", "50", "112", "23", "22", "29", "51")
    ordered = all(node in executing for node in required)
    dependency = ordered and executing.index("13") < executing.index("50") < \
        executing.index("112") < executing.index("23") < executing.index("29") < \
        executing.index("51") and executing.index("22") < executing.index("29")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        ("dmd8_then_native_high4" if training == "stage_dmd_8nfe" and refine == 4
         else "dmd8_then_native_high_steps" if training == "stage_dmd_8nfe"
         else "b50_then_native_high_steps"):
            progress == ["13"] * (8 if training == "stage_dmd_8nfe" else 50) + ["29"] * refine,
        "relay_audit_handoff_and_independent_model_order": dependency,
        "both_stage_saves_executed": ordered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8869)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--refine", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--training", choices=TRAINING, default="stage_dmd_8nfe")
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    # The first successful split run fell to roughly 4 GiB free from about
    # 99 GiB. Keep a larger entry cushion; this is not an OOM guarantee.
    parser.add_argument("--min-free-ram-mib", type=int, default=105000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_probe_graph(width=args.width, height=args.height,
                                          frames=args.frames, refine=args.refine,
                                          training=args.training)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-vdn-real-gpu-20260925" / \
        f"{run_id}-{'DMD8' if args.training == 'stage_dmd_8nfe' else 'B50'}-native{args.refine}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "paths.json").write_text(
        json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": schema(args.refine, args.training), "status": "started", "run_root": str(run_root),
              "port": args.port, "source_candidate_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames],
              "refine": args.refine, "training": args.training}
    try:
        with shared.IsolatedServer(args, run_root,
                                   f"s{'05' if args.training == 'stage_dmd_8nfe' and args.refine == 4 else '03'}-vdn-{'dmd8' if args.training == 'stage_dmd_8nfe' else 'b50'}-native{args.refine}-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase, args.refine, args.training)
        if report["checks"]["terminal_success"]:
            report["low_save"] = common._save_receipt(phase, "50")
            report["high_save"] = common._save_receipt(phase, "51")
            report["checks"]["low_artifact_sha"] = common._artifact_sha_matches(run_root, report["low_save"])
            report["checks"]["high_artifact_sha"] = common._artifact_sha_matches(run_root, report["high_save"])
            report["checks"].update(_media_checks(run_root, args.width, args.height,
                                                   args.frames, refine=args.refine,
                                                   training=args.training))
        report["checks"]["source_candidate_unchanged"] = \
            shared._sha256_file(candidate(args.refine, args.training)) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_vdn_split_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("VDN separated graph failed one or more mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
