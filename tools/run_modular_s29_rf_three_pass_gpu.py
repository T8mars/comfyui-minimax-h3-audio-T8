"""Real-weight Native LOW4 -> learned 3D -> RF BASE5 -> RESTART3 probe.

Default is a read-only sealed-candidate/asset/resource preflight. Only an
explicit ``--confirm-run`` starts an owned isolated Core and writes a small
private AV result. It cannot establish full-size or perceptual quality.
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
import run_modular_s29_rf_gpu as two_stage  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s29-rf-three-pass-real-split-gpu.v1"
CANDIDATES = two_stage.CANDIDATES
MODEL = two_stage.MODEL
LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
UPSCALE = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
OUTPUT = "MiniMaxH3/S29_RF_ThreePass"


def candidate(kind: str = "save_effects") -> Path:
    if kind not in ("save_effects", "resume_effects"):
        raise ValueError("Expected saved or RF-only three-pass candidate")
    return CANDIDATES / f"RF_two_pass_detail_mixer_{kind}_EXP.api.json"


def build_probe_graph(*, width: int = 128, height: int = 64,
                      frames: int = 22) -> tuple[dict, str]:
    if min(width, height, frames) < 1 or width % 32 or height % 32:
        raise ValueError("Use a positive, 32-aligned RF three-pass test canvas")
    source = candidate()
    source_sha = shared._sha256_file(source)
    graph = deepcopy(json.loads(source.read_text(encoding="utf-8")))
    expected = {"1": "UNETLoader", "10": "MiniMaxH3NativeDualStageSetupEXPT8",
                "13": "MiniMaxH3StageSamplerEXPT8", "22": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "23": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "25": "MiniMaxH3NativeDualHandoffEXPT8",
                "26": "MiniMaxH3RFBaseStageSetupEXPT8", "29": "MiniMaxH3StageSamplerEXPT8",
                "88": "MiniMaxH3NativeDualStageSetupEXPT8",
                "89": "MiniMaxH3AVTailDetailScheduleT8Advanced",
                "90": "MiniMaxH3RFHandoffEXPT8",
                "93": "MiniMaxH3StageUNETLoaderAfterEXPT8",
                "96": "MiniMaxH3RFRestartStageSetupEXPT8",
                "99": "MiniMaxH3StageSamplerEXPT8",
                "40": "MiniMaxH3PromptRelayPlanT8Advanced",
                "47": "MiniMaxH3PromptRelayPlanT8Advanced",
                "104": "MiniMaxH3PromptRelayPlanT8Advanced",
                "41": "MiniMaxH3StageEAVConfigEXPT8",
                "42": "MiniMaxH3StageEAVConfigEXPT8",
                "100": "MiniMaxH3StageEAVConfigEXPT8",
                "45": "MiniMaxH3StageEAVAuditEXPT8",
                "46": "MiniMaxH3StageEAVAuditEXPT8",
                "102": "MiniMaxH3StageEAVAuditEXPT8",
                "50": "MiniMaxH3StageSaveEXPT8", "51": "MiniMaxH3StageSaveEXPT8",
                "103": "MiniMaxH3StageSaveEXPT8"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in expected.items()):
        raise ValueError("Saved three-pass candidate lost a required explicit stage/effect")
    if any(graph[key]["inputs"].get("unet_name") != MODEL for key in ("1", "22", "93")) or \
            graph["22"]["inputs"].get("completed_stage") != ["13", 2] or \
            graph["93"]["inputs"].get("completed_stage") != ["29", 2] or \
            any(graph[key]["inputs"].get("lora_name") != LORA or
                graph[key]["inputs"].get("strength_model") != 1. for key in ("70", "71", "94")) or \
            graph["23"]["inputs"].get("model_name") != UPSCALE or \
            graph["23"]["inputs"].get("av_latent") != ["45", 0] or \
            graph["10"]["inputs"].get("stage") != "dual_low_4" or \
            graph["88"]["inputs"].get("stage") != "dual_high_3" or \
            graph["89"]["inputs"].get("extra_tail_steps") != 2 or \
            graph["26"]["inputs"].get("sigmas") != ["89", 0] or \
            graph["90"]["inputs"].get("completed_av") != ["46", 0] or \
            graph["96"]["inputs"].get("restart_steps") != 3 or \
            graph["96"]["inputs"].get("restart_video_sigma") != .15 or \
            graph["14"]["inputs"].get("av_latent") != ["102", 0] or \
            any(graph[key]["inputs"].get("stage_result") != [sample, 2]
                for key, sample in (("50", "13"), ("51", "29"), ("103", "99"))):
        raise ValueError("Saved three-pass model/LoRA/3D/RF stage contract changed")
    if any(graph[key]["inputs"].get("mode") != "report_only" for key in ("41", "42", "100")):
        raise ValueError("RF three-pass probe requires three external report-only EAV configs")
    if graph["95"]["inputs"].get("prompt_relay_plan") != ["104", 0] or \
            graph["26"]["inputs"].get("model") != ["111", 0] or \
            graph["96"]["inputs"].get("model") != ["121", 0]:
        raise ValueError("RF three-pass external Relay/Bias/STG branches changed")
    graph["9"]["inputs"].update(width=width, height=height)
    for key in ("40", "47", "104"):
        graph[key]["inputs"]["length"] = frames
    for key in ("41", "42", "100"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                     end_video_progress=1., g_hard_limit=3.)
    graph["16"]["inputs"]["filename_prefix"] = f"{OUTPUT}/full_selected"
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / MODEL,
        "ema_b_lora": models / "loras" / LORA,
        "trained_3d_upscaler": models / "latent_upscale_models" / UPSCALE,
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
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {name: str(path) for name, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def media_path(run_root: Path, prefix: str = "full_selected") -> Path:
    output = run_root / "output" / OUTPUT
    files = list(output.glob(f"{prefix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one private RF three-pass MP4")
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


def stage_checks(phase: dict) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("13", "50", "45", "23", "22", "29", "51", "46", "90", "93",
                "99", "103", "102", "89", "110", "111", "120", "121")
    present = all(node in executing for node in required)
    ordered = present and all(executing.index("13") < executing.index(node)
                              for node in ("50", "45", "23", "22", "29")) and \
        executing.index("45") < executing.index("23") < executing.index("29") and \
        executing.index("22") < executing.index("29") < executing.index("51") < \
        executing.index("46") < executing.index("90") < executing.index("99") < \
        executing.index("103") and executing.index("29") < executing.index("93") < executing.index("99")
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_low4_rfbase5_restart3_progress": progress == ["13"] * 4 + ["29"] * 5 + ["99"] * 3,
        "three_stage_deferred_models_and_effects_order": ordered,
        "all_three_stage_saves_and_audits_executed": present,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8867)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=105000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_probe_graph(width=args.width, height=args.height, frames=args.frames)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-m2-rf-real-gpu-20260925" / \
        f"{run_id}-three-pass-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "port": args.port, "source_candidate_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames]}
    try:
        with shared.IsolatedServer(args, run_root, "s29-rf-three-pass-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = stage_checks(phase)
        if report["checks"]["terminal_success"]:
            report["low_save"] = common._save_receipt(phase, "50")
            report["base_save"] = common._save_receipt(phase, "51")
            report["restart_save"] = common._save_receipt(phase, "103")
            for name in ("low", "base", "restart"):
                report["checks"][f"{name}_artifact_sha"] = \
                    common._artifact_sha_matches(run_root, report[f"{name}_save"])
            report["checks"].update(media_checks(run_root, args.width, args.height, args.frames))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(candidate()) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_rf_three_pass_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("RF three-pass graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
