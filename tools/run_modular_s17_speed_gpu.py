"""Real-weight S17 SPEED two-stage split probe (LOW14 -> HIGH6).

The saved draft and old whole-chain workflows are read-only. Without
``--confirm-run`` this performs only a resource/asset preflight; execution
uses a private small-canvas copy and an owned isolated Core.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from safetensors import safe_open

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as core_capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s17-speed-two-stage-real-gpu.v1"
CANDIDATE = (PROJECT / "artifacts/development/modular-sampling-m3-speed-s17-20260923/"
             "candidate-v4-2-effects/SPEED_2Stage_EAV_Relay_DRAFT_api.json")
MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
OUTPUT = "MiniMaxH3/S17_SPEED_RealSplit"
RUNS = PROJECT / "artifacts/development/modular-sampling-m3-speed-real-gpu-20260925"


def build_probe_graph(*, width: int = 256, height: int = 128,
                      frames: int = 22) -> tuple[dict, str]:
    if width < 128 or height < 128 or width % 32 or height % 32 or frames < 2:
        raise ValueError("S17 small canvas needs width/height >=128, 32-aligned and >=2 frames")
    source_sha = shared._sha256_file(CANDIDATE)
    graph = deepcopy(json.loads(CANDIDATE.read_text(encoding="utf-8")))
    required = {"4": "MiniMaxH3SPEEDPlanT8Advanced", "10": "UNETLoader",
                "11": "MiniMaxH3SPEEDSourceT8Advanced",
                "12": "MiniMaxH3SPEEDStageSetupEXPT8",
                "14": "MiniMaxH3SPEEDStageSampleEXPT8",
                "17": "MiniMaxH3SPEEDRelayApplyEXPT8",
                "18": "MiniMaxH3StageEAVConfigEXPT8",
                "19": "MiniMaxH3StageEAVApplyEXPT8",
                "20": "MiniMaxH3StageEAVAuditEXPT8",
                "21": "MiniMaxH3SPEEDStageSaveEXPT8", "22": "UNETLoader",
                "23": "MiniMaxH3SPEEDSourceT8Advanced",
                "24": "MiniMaxH3SPEEDStageSetupEXPT8",
                "26": "MiniMaxH3SPEEDStageSampleEXPT8",
                "27": "MiniMaxH3SPEEDDCTTransitionEXPT8",
                "29": "MiniMaxH3SPEEDRelayApplyEXPT8",
                "30": "MiniMaxH3StageEAVConfigEXPT8",
                "31": "MiniMaxH3StageEAVApplyEXPT8",
                "32": "MiniMaxH3StageEAVAuditEXPT8", "82": "SaveVideo"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in required.items()):
        raise ValueError("Saved SPEED candidate lost a separate stage or external effect")
    plan = graph["4"]["inputs"]
    if any(plan.get(key) != value for key, value in {
        "steps": 20, "scales": "0.5,1.0", "transition_mode": "manual_sigmas",
        "manual_transition_sigmas": "0.85", "transform": "dct",
        "fallback_policy": "error"}.items()):
        raise ValueError("Saved SPEED two-stage schedule changed")
    if (any(graph[key]["inputs"].get("unet_name") != MODEL for key in ("10", "22"))
            or graph["12"]["inputs"].get("model") != ["10", 0]
            or graph["24"]["inputs"].get("model") != ["22", 0]
            or graph["24"]["inputs"].get("previous_spec") != ["12", 5]
            or graph["21"]["inputs"].get("stage_result") != ["14", 1]
            or graph["27"]["inputs"].get("completed_stage") != ["21", 0]
            or graph["27"]["inputs"].get("next_stage") != ["24", 5]
            or graph["80"]["inputs"].get("av_latent") != ["32", 0]):
        raise ValueError("Saved SPEED model/DCT/frozen-stage handoff changed")
    for stage, relay, eav, apply, sampler, audit in (
        ("12", "17", "18", "19", "14", "20"),
        ("24", "29", "30", "31", "26", "32"),
    ):
        if (graph[relay]["inputs"].get("model") != [stage, 0]
                or graph[relay]["inputs"].get("execution_mode") != "apply_exp"
                or graph[eav]["inputs"].get("mode") != "report_only"
                or graph[apply]["inputs"].get("model") != [relay, 0]
                or graph[sampler]["inputs"].get("model") != [apply, 0]
                or graph[sampler]["inputs"].get("positive") != [relay, 1]
                or graph[audit]["inputs"].get("av_latent") != [sampler, 0]):
            raise ValueError("Saved SPEED Relay/EAV is not independent per stage")
    plan.update(width=width, height=height)
    for key in ("11", "23", "16", "28"):
        graph[key]["inputs"]["length"] = frames
    for key in ("18", "30"):
        graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                     end_video_progress=1., g_hard_limit=3.)
    graph["82"]["inputs"]["filename_prefix"] = f"{OUTPUT}/full_selected"
    # The original draft saves LOW for DCT. A second private Save plus an
    # output sink makes the final stage's exact state inspectable too.
    graph["85"] = {"class_type": "MiniMaxH3SPEEDStageSaveEXPT8",
                   "inputs": {"stage_result": ["26", 1]}}
    graph["86"] = {"class_type": "PreviewAny", "inputs": {"source": ["85", 1]}}
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / MODEL,
        "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    gpu, free_ram = shared.gpu_memory_mib(), native_gpu._free_physical_mib()
    checks = {
        "candidate_sha_bound": bool(source_sha),
        "all_assets_installed": all(path.is_file() for path in assets.values()),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "gpu_free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "system_free_ram_gate": free_ram is not None and free_ram >= args.min_free_ram_mib,
    }
    return {"schema": SCHEMA + ".preflight", "candidate_sha256": source_sha,
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "physical_free_mib": free_ram,
            "checks": checks, "ready": all(checks.values())}


def speed_artifact(run_root: Path, stage: int) -> dict:
    store = (run_root / "output/MiniMaxH3/modular_speed_stages").resolve()
    files = list(store.glob(f"stage-{stage}-*/manifest.json"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one committed SPEED stage {stage}")
    manifest = files[0].resolve()
    if not manifest.is_relative_to(store):
        raise ValueError("SPEED artifact escaped the owned store")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    state = manifest.parent / "speed-stage.safetensors"
    if (data.get("schema") != "t8.modular-sampling.frozen-speed-stage.v1"
            or data.get("stage_index") != stage or data.get("portable_identity") is not True
            or data.get("state_file") != state.name or not state.is_file()
            or state.stat().st_size != data.get("state_bytes")
            or shared._sha256_file(state).lower() != data.get("state_sha256")):
        raise ValueError("SPEED completed stage artifact failed size/SHA/identity validation")
    with safe_open(str(state), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if type(metadata) is not dict or set(metadata) != {"speed_json"}:
        raise ValueError("SPEED stage metadata is missing or unknown")
    payload = json.loads(metadata["speed_json"])
    if payload.get("schema") != data["schema"]:
        raise ValueError("SPEED stage payload schema differs from manifest")
    receipt = json.loads(payload["receipt_json"])
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_sha256", None)
    def canonical_sha(value):
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
    request = receipt.get("request") or {}
    if (digest != data.get("receipt_sha256") or digest != canonical_sha(unsigned)
            or receipt.get("request_sha256") != canonical_sha(request)
            or receipt.get("schema") != "t8.modular-sampling.speed-result.v1"
            or request.get("plan_sha256") != data.get("plan_sha256")
            or request.get("stage_index") != stage
            or receipt.get("verified_recipe_completion") is not True
            or receipt.get("portable_identity") is not True):
        raise ValueError("SPEED stage receipt integrity or identity failed")
    effects = receipt.get("execution", {}).get("effects") or {}
    summary = {
        "receipt_sha256": digest,
        "request_sha256": receipt["request_sha256"],
        "callbacks": receipt["execution"]["callbacks"],
        "sampler_known": receipt["execution"]["sampler_known"],
        "effects": {key: effects.get(key) for key in (
            "kind", "mode", "status", "aborted", "clock_match",
            "completed_forwards", "planned_forwards", "relay_required",
            "relay_attention_calls", "selector_calls", "blocks")},
        "output": receipt.get("output"),
        "noise_provider": request.get("noise_provider"),
    }
    return {"path": manifest.relative_to(store).as_posix(),
            "sha256": shared._sha256_file(manifest).lower(), "report": data,
            "receipt": summary}


def media_checks(run_root: Path, width: int, height: int, frames: int,
                 *, cold: bool = False) -> dict[str, bool]:
    prefix = "cold_selected" if cold else "full_selected"
    files = list((run_root / "output" / OUTPUT).glob(f"{prefix}*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one private SPEED media output")
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
                                         and (picture[0].get("width"), picture[0].get("height")) == (width, height),
        "one_aac_audio": len(sound) == 1 and sound[0].get("codec_name") == "aac",
        "full_av_decode": decode.returncode == 0,
        "private_file_nonempty": video.is_relative_to(run_root) and video.stat().st_size > 0,
    }


def stage_checks(phase: dict, low: dict | None = None,
                 high: dict | None = None) -> dict[str, bool]:
    events = phase.get("events") or []
    progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
    executing = [str(item.get("node")) for item in events if item.get("type") == "executing"]
    required = ("14", "21", "20", "24", "27", "26", "32", "85", "82")
    present = all(node in executing for node in required)
    ordered = present and (executing.index("14") < executing.index("21") <
                           executing.index("27") < executing.index("26") <
                           executing.index("85") and executing.index("26") <
                           executing.index("32") < executing.index("82"))
    receipts = (low or {}).get("receipt"), (high or {}).get("receipt")
    if all(receipts):
        low_receipt, high_receipt = receipts
        exact_progress = (low_receipt["sampler_known"] is True
                          and high_receipt["sampler_known"] is True
                          and low_receipt["callbacks"] == list(range(14))
                          and high_receipt["callbacks"] == list(range(6)))
        effects = all(
            item["effects"]["kind"] == "eav"
            and item["effects"]["mode"] == "report_only"
            and item["effects"]["status"] == "observed_report_only"
            and item["effects"]["aborted"] is False
            and item["effects"]["clock_match"] is True
            and item["effects"]["completed_forwards"] == steps
            and item["effects"]["planned_forwards"] == steps
            and item["effects"]["relay_required"] is True
            and item["effects"]["relay_attention_calls"] == steps * 50
            and item["effects"]["selector_calls"] == steps * 50
            for item, steps in ((low_receipt, 14), (high_receipt, 6)))
        high_noise = high_receipt["noise_provider"] or {}
        handoff_receipt = (high_noise.get("source_receipt_sha256") ==
                           low_receipt["receipt_sha256"]
                           and high_noise.get("source_stage_index") == 0)
    else:
        exact_progress = progress == ["14"] * 14 + ["26"] * 6
        effects = present
        handoff_receipt = True
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exact_planned_low14_high6_progress": exact_progress,
        "dct_handoff_then_independent_high": ordered and handoff_receipt,
        "both_stage_saves_and_external_audits_executed": present and effects,
    }


def audit_existing(run_root: Path) -> dict:
    """Append a corrected receipt-based audit; never edit the original report."""
    run_root = run_root.resolve(strict=True)
    if not run_root.is_relative_to(RUNS.resolve()):
        raise ValueError("S17 audit accepts only its private run directory")
    original = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    if (original.get("schema") != SCHEMA or original.get("status") != "fail"
            or original.get("run_root") != str(run_root)
            or original.get("error") !=
            "RuntimeError: S17 SPEED two-stage graph failed a mechanical stage check"):
        raise ValueError("Not the preserved S17 progress-event false-negative")
    original_checks = original.get("checks") or {}
    if (set(original_checks) != {
            "terminal_success", "exact_planned_low14_high6_progress",
            "dct_handoff_then_independent_high",
            "both_stage_saves_and_external_audits_executed",
            "h264_expected_frames_geometry", "one_aac_audio", "full_av_decode",
            "private_file_nonempty", "source_candidate_unchanged", "server_stopped"}
            or original_checks["exact_planned_low14_high6_progress"] is not False
            or not all(value is True for key, value in original_checks.items()
                       if key != "exact_planned_low14_high6_progress")):
        raise ValueError("Original failure had another cause")
    width, height, frames = original["test_dimensions"]
    graph, source_sha = build_probe_graph(width=width, height=height, frames=frames)
    if (source_sha != original["source_candidate_sha256"]
            or graph != json.loads((run_root / "prompt.json").read_text(encoding="utf-8"))):
        raise ValueError("Original candidate or submitted graph changed")
    phase = json.loads((run_root / "phase.json").read_text(encoding="utf-8"))
    low, high = speed_artifact(run_root, 0), speed_artifact(run_root, 1)
    if (low["sha256"] != original["low_save"]["sha256"]
            or high["sha256"] != original["high_save"]["sha256"]):
        raise ValueError("Stage artifacts changed after the original run")
    checks = stage_checks(phase, low, high)
    checks.update(media_checks(run_root, width, height, frames))
    checks["source_candidate_unchanged"] = shared._sha256_file(CANDIDATE) == source_sha
    checks["server_stopped"] = not shared.port_is_listening("127.0.0.1", original["port"])
    audit = {"schema": SCHEMA + ".receipt-audit.v2",
             "status": ("real_speed_two_stage_small_media_mechanical_pass_not_quality_acceptance"
                        if all(checks.values()) else "fail"),
             "original_report_sha256": shared._sha256_file(run_root / "report.json").lower(),
             "original_report_status": original["status"],
             "progress_event_count": sum(item.get("type") == "progress"
                                         for item in phase.get("events") or []),
             "checks": checks, "low_save": low, "high_save": high}
    target = run_root / "audit-receipts-v2.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8871)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=95000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-existing", type=Path)
    args = parser.parse_args(argv)
    if args.audit_existing is not None:
        audit = audit_existing(args.audit_existing)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return 0 if audit["status"] != "fail" else 3
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    graph, source_sha = build_probe_graph(width=args.width, height=args.height, frames=args.frames)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = RUNS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-two-stage-full")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host, args.extra_model_paths_config = "127.0.0.1", paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "port": args.port, "source_candidate_sha256": source_sha,
              "preflight": readiness, "test_dimensions": [args.width, args.height, args.frames]}
    try:
        with shared.IsolatedServer(args, run_root, "s17-speed-two-stage-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(core_capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        if (phase.get("terminal") or {}).get("type") == "execution_success":
            report["low_save"] = speed_artifact(run_root, 0)
            report["high_save"] = speed_artifact(run_root, 1)
        report["checks"] = stage_checks(phase, report.get("low_save"), report.get("high_save"))
        if report["checks"]["terminal_success"]:
            report["checks"].update(media_checks(run_root, args.width, args.height, args.frames))
        report["checks"]["source_candidate_unchanged"] = shared._sha256_file(CANDIDATE) == source_sha
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("real_speed_two_stage_small_media_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S17 SPEED two-stage graph failed a mechanical stage check")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
