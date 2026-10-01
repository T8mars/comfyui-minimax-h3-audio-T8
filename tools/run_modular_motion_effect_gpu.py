"""Real-weight small-canvas S25 Motion pass1 -> separate Relay/EAV pass2.

Default is read-only preflight. --confirm-run submits only a private reduced
copy of a SHA-pinned candidate to an owned isolated Core, never the user Core.
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
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402

SCHEMA = "t8.modular-sampling.s25-motion-effects-real-gpu.v1"
CANDIDATES = PROJECT / "artifacts/development/modular-sampling-m4-motion-effects-20260925/candidate-v2"
SOURCE_SHA = {
    "Fullclip": "b085ebfeb9e9da15363e7e0f47fef5e2d39caec10e300e8e6108cd42eb615009",
    "Windowed": "0204f104710191dca1a8a5e60e78292e29ed0778a60a3dd193797282ba419160",
}
MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PROMPT = "A woman in red crosses a moonlit courtyard. One continuous camera shot. " \
         "She turns and moves forward; coherent body motion, stable clothing, natural ambient sound."


def candidate(variant: str) -> Path:
    if variant not in SOURCE_SHA:
        raise ValueError("Unknown Motion Recovery variant")
    return CANDIDATES / f"Motion_Recovery_{variant}_Relay_EAV_Separate_Pass2_EXP.api.json"


def build_probe_graph(variant: str, *, eav_mode: str = "report_only") -> tuple[dict, str]:
    if eav_mode not in ("report_only", "apply_exp"):
        raise ValueError("S25 real-weight probe needs an active external EAV mode")
    source = candidate(variant)
    actual_sha = shared._sha256_file(source).lower()
    if actual_sha != SOURCE_SHA[variant]:
        raise ValueError("Sealed S25 Motion effects API candidate changed")
    graph = deepcopy(json.loads(source.read_bytes()))
    windowed = variant == "Windowed"
    kinds = {"1": "UNETLoader", "5": "MiniMaxH3AudioConditioningT8",
             "9": "SamplerCustomAdvanced", "12": "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
             "13" if windowed else "14": (
                 "MiniMaxH3MotionSegmentPlanT8Advanced" if windowed
                 else "MiniMaxH3MotionRecoveryComposerT8Advanced"),
             "19" if windowed else "18": "MiniMaxH3StageSamplerEXPT8",
             "30" if windowed else "28": "MiniMaxH3MotionStageBindEXPT8",
             "31" if windowed else "29": "MiniMaxH3MotionStageAuditEXPT8",
             "33" if windowed else "31": "MiniMaxH3PromptRelayPlanT8Advanced",
             "34" if windowed else "32": "MiniMaxH3PromptRelayConditioningT8Advanced",
             "35" if windowed else "33": "MiniMaxH3MotionRelayBindEXPT8",
             "36" if windowed else "34": "MiniMaxH3StageEAVConfigEXPT8",
             "37" if windowed else "35": "MiniMaxH3StageEAVApplyEXPT8",
             "38" if windowed else "36": "MiniMaxH3StageEAVAuditEXPT8"}
    if any(graph.get(key, {}).get("class_type") != kind for key, kind in kinds.items()):
        raise ValueError("S25 candidate lost a separate pass or external effect")
    if (graph["1"]["inputs"].get("unet_name") != MODEL
            or graph["12"]["inputs"].get("mode") != "auto_conservative_exp"
            or graph["6"]["inputs"].get("steps") != 20
            or graph["15" if not windowed else "16"]["inputs"].get("steps") != 20):
        raise ValueError("S25 original model/schedule boundary changed")
    if windowed and (graph["13"]["inputs"].get("handle_frames") != 12
                     or graph["13"]["inputs"].get("coverage") != "hot_ranges_only"):
        raise ValueError("Windowed S25 source window semantics changed")
    graph["5"]["inputs"].update(prompt=PROMPT, width=128, height=64, length=22)
    graph["12"]["inputs"].update(mode="manual_ranges", manual_ranges="5-14:3")
    relay_plan = "33" if windowed else "31"
    relay_conditioning = "34" if windowed else "32"
    eav_config = "36" if windowed else "34"
    eav_audit = "38" if windowed else "36"
    relay_bind = "35" if windowed else "33"
    motion_audit = "31" if windowed else "29"
    graph[relay_plan]["inputs"]["global_prompt"] = PROMPT
    graph[relay_conditioning]["inputs"].update(width=128, height=64)
    # This reduced clip has 17 latent video frames.  With tau=.2, measured
    # CFI around .04 makes apply_exp clamp to g=1, indistinguishable from
    # report_only.  Use a probe-only tau that exercises actual EAV math.
    probe_tau = 10. if eav_mode == "apply_exp" else .2
    graph[eav_config]["inputs"].update(
        mode=eav_mode, tau=probe_tau, start_video_progress=0.,
        end_video_progress=1., g_hard_limit=3.)
    graph["11"]["inputs"]["filename_prefix"] = f"MiniMaxH3/S25_Motion_{variant}/pass1"
    graph["24" if windowed else "22"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/S25_Motion_{variant}/recovered")
    graph["200"] = {"class_type": "PreviewAny", "inputs": {"source": [eav_audit, 1]}}
    graph["201"] = {"class_type": "PreviewAny", "inputs": {"source": [relay_bind, 2]}}
    graph["202"] = {"class_type": "PreviewAny", "inputs": {"source": [motion_audit, 1]}}
    return graph, actual_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    models = args.comfy_root / "models"
    assets = {"base": models / "diffusion_models" / MODEL,
              "clip": models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
              "video_vae": models / "vae/minimax_h3_video_vae_fp16.safetensors",
              "audio_vae": models / "vae/minimax_h3_audio_vae_fp32.safetensors"}
    gpu = shared.gpu_memory_mib()
    ram = native_gpu._free_physical_mib()
    checks = {"candidate_sha_bound": source_sha == SOURCE_SHA[args.variant],
              "all_assets_installed": all(path.is_file() for path in assets.values()),
              "port_free": not shared.port_is_listening("127.0.0.1", args.port),
              "gpu_free_vram_gate": bool(gpu.get("available") and
                                         gpu["free_mib"] >= args.min_free_vram_mib),
              "system_free_ram_gate": ram is not None and ram >= args.min_free_ram_mib}
    return {"schema": SCHEMA + ".preflight", "variant": args.variant,
            "candidate_sha256": source_sha, "assets": {k: str(v) for k, v in assets.items()},
            "gpu": gpu, "physical_free_mib": ram, "checks": checks,
            "ready": all(checks.values())}


def media_checks(run_root: Path, variant: str) -> dict:
    output = run_root / "output/MiniMaxH3" / f"S25_Motion_{variant}"
    result = {}
    for name in ("pass1", "recovered"):
        # VHS emits both a video-only sidecar and a joint AV file. Its UI
        # receipt points to the latter; never count the sidecar as a duplicate.
        files = list(output.glob(f"{name}_*-audio.mp4"))
        if len(files) != 1:
            result[f"{name}_one_mp4"] = False
            continue
        video = files[0]
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                str(video)], capture_output=True, text=True, timeout=30, check=True)
        streams = json.loads(probe.stdout)["streams"]
        picture = [item for item in streams if item.get("codec_type") == "video"]
        sound = [item for item in streams if item.get("codec_type") == "audio"]
        decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                                 "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                                capture_output=True, timeout=90, check=False)
        result[f"{name}_one_mp4"] = True
        result[f"{name}_22_frames_128x64_h264"] = (len(picture) == 1 and
            picture[0].get("codec_name") == "h264" and
            int(picture[0].get("nb_frames", 0)) == 22 and
            (picture[0].get("width"), picture[0].get("height")) == (128, 64))
        result[f"{name}_aac"] = len(sound) == 1 and sound[0].get("codec_name") == "aac"
        result[f"{name}_joint_decode"] = decode.returncode == 0
        result[f"{name}_sha256"] = shared._sha256_file(video)
    return result


def _preview_report(phase: dict, node_id: str) -> dict:
    items = (phase.get("executed_outputs") or {}).get(node_id, {}).get("text")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError(f"PreviewAny {node_id} did not expose one report")
    return json.loads(items[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(SOURCE_SHA), required=True)
    parser.add_argument("--eav-mode", choices=("report_only", "apply_exp"),
                        default="report_only")
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    graph, source_sha = build_probe_graph(args.variant, eav_mode=args.eav_mode)
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (PROJECT / "artifacts/development/modular-sampling-m4-motion-effects-20260925"
                / f"{run_id}-{args.variant}-{args.eav_mode}-real-gpu")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT),
                                indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "variant": args.variant,
              "run_root": str(run_root), "port": args.port,
              "source_candidate_sha256": source_sha, "preflight": readiness,
              "test_dimensions": [128, 64, 22],
              "manual_motion_range": "5-14:3", "eav_mode": args.eav_mode,
              "eav_probe_tau": graph["36" if args.variant == "Windowed" else "34"]["inputs"]["tau"]}
    try:
        with shared.IsolatedServer(args, run_root, f"s25-{args.variant}-full") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(
            json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        events = phase.get("events") or []
        progress = [str(item.get("node")) for item in events if item.get("type") == "progress"]
        second = "19" if args.variant == "Windowed" else "18"
        checks = {"terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
                  # The two independent VHS sinks can cause Core to evaluate
                  # the seeded pass1 branch twice in one prompt. Both must
                  # still be complete Stock20 runs; record the observed count.
                  "pass1_stock20_progress": progress.count("9") in (20, 40),
                  "independent_pass2_progress": progress.count(second) >= 2,
                  "source_candidate_unchanged": shared._sha256_file(candidate(args.variant)).lower()
                                                == source_sha,
                  "server_stopped": not shared.port_is_listening("127.0.0.1", args.port)}
        if checks["terminal_success"]:
            report["eav"] = _preview_report(phase, "200")
            report["relay"] = _preview_report(phase, "201")
            report["motion_source"] = _preview_report(phase, "202")
            checks["eav_real_forward_and_relay"] = (report["eav"]["status"] ==
                "observed_" + args.eav_mode and report["eav"]["relay_required"] and
                report["eav"]["completed_forwards"] >= 2 and
                report["eav"]["relay_attention_calls"] > 0)
            # An apply_exp receipt alone is insufficient: a small clip can
            # clamp every EAV gain to 1 and produce no numerical effect.
            checks["eav_numeric_gain_observed"] = (
                args.eav_mode == "report_only" or
                float(report["eav"]["feta"].get("g_max") or 0.) > 1.001)
            checks["relay_source_paired"] = (report["relay"]["status"] ==
                "paired_retained_model_and_conditioning")
            checks["motion_source_audited"] = (report["motion_source"]["status"] ==
                "source_bound_candidate_audited")
            media = media_checks(run_root, args.variant)
            report["media"] = media
            checks["both_media_decode"] = all(value for key, value in media.items()
                                               if not key.endswith("_sha256"))
        report["pass1_progress_events"] = progress.count("9")
        report["pass2_progress_events"] = progress.count(second)
        report["checks"] = checks
        report["status"] = ("pass_small_real_weight_motion_effects_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S25 real-weight Motion graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
