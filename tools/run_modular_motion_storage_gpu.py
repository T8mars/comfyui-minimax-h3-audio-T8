"""Real-weight small-canvas Motion first-pass freeze -> cold pass-2 parity.

Read-only preflight by default. --confirm-run uses two owned isolated Cores,
exact external native AV receipt and a fresh second process with no pass1
sampler. It never edits old graphs or the user Core.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from safetensors import safe_open

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_motion_effect_gpu as effect  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as capture  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402

SCHEMA = "t8.modular-sampling.s25-motion-cold-real-gpu.v1"
TARGET = PROJECT / "artifacts/development/modular-sampling-m4-motion-storage-20260925/candidate-v1"
SOURCE_SHA = {
    "Fullclip": {
        "Freeze_Pass1": "2e7c0a07cb85a964af382b25136e0f2745d0bd97ba60b6d436be9264b7798adf",
        "Resume_Only_Pass2": "ac562529b81acafcf14466fdcbb36153ed35bb7fb446a9ff6292a0d641874d73",
    },
    "Windowed": {
        "Freeze_Pass1": "b79821fe2e907582aeb4982dd4903d9d1e734d99667b17a026a8ded22650b67b",
        "Resume_Only_Pass2": "a207ae51a692cd1d259d9cbc9562edf6ff681de8b4e4abe5ef96243c199b1ca3",
    },
}
CONTINUOUS_RGB_PCM = {
    "Fullclip": {
        "report_only": ("153745ea485b4059a72b9cd3f3f92db10189a13df8f6a9dd4d2bae81a85a0c81",
                        "01652cc9ceddba0940e9524649b83d67b1efba06366f02f1cf6941b029fc9130"),
        "apply_exp": ("d5d06b6f453d212150ee8d18463ea216a35ac0b1472c2d00d9ad5b30abcef1e5",
                      "01652cc9ceddba0940e9524649b83d67b1efba06366f02f1cf6941b029fc9130"),
    },
    "Windowed": {
        "report_only": ("ad70e1718132dbf081973d14913a8620dd51a0be19efcbd7c12cb6139e014df5",
                        "01652cc9ceddba0940e9524649b83d67b1efba06366f02f1cf6941b029fc9130"),
        "apply_exp": ("6a53783a3ed02c0bc31d41bc75decc56d949f79c03a30a7c44d44076f7515e08",
                      "01652cc9ceddba0940e9524649b83d67b1efba06366f02f1cf6941b029fc9130"),
    },
}


def candidate(variant: str, phase: str) -> dict:
    path = TARGET / f"Motion_Recovery_{variant}_{phase}_Relay_EAV_EXP.api.json"
    expected = SOURCE_SHA[variant][phase]
    if shared._sha256_file(path).lower() != expected:
        raise ValueError("Pinned Motion freeze/cold source changed")
    return json.loads(path.read_bytes())


def probe_graphs(variant: str, *, eav_mode: str = "report_only") -> tuple[dict, dict]:
    if eav_mode not in ("report_only", "apply_exp"):
        raise ValueError("Motion cold probe needs an active external EAV mode")
    freeze = deepcopy(candidate(variant, "Freeze_Pass1"))
    cold = deepcopy(candidate(variant, "Resume_Only_Pass2"))
    freeze["5"]["inputs"].update(prompt=effect.PROMPT, width=128, height=64, length=22)
    save_id = "39" if variant == "Windowed" else "37"
    freeze[save_id]["inputs"]["confirm_save"] = True
    load = cold[save_id]
    if (freeze[save_id]["class_type"] != "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
            or load["class_type"] != "MiniMaxH3MotionFrozenFirstPassLoadEXPT8"
            or any(node["class_type"] == "SamplerCustomAdvanced" for node in cold.values())):
        raise ValueError("Motion cold boundary lost exact native Save/Load or retained pass1")
    load["inputs"].update(expected_frame_count=22, expected_width=128, expected_height=64)
    cold["12"]["inputs"].update(mode="manual_ranges", manual_ranges="5-14:3")
    relay_plan = "33" if variant == "Windowed" else "31"
    relay_cond = "34" if variant == "Windowed" else "32"
    eav = "36" if variant == "Windowed" else "34"
    delivered = "24" if variant == "Windowed" else "22"
    audit = "38" if variant == "Windowed" else "36"
    paired = "35" if variant == "Windowed" else "33"
    source = "31" if variant == "Windowed" else "29"
    cold[relay_plan]["inputs"]["global_prompt"] = effect.PROMPT
    cold[relay_cond]["inputs"].update(width=128, height=64)
    cold[eav]["inputs"].update(mode=eav_mode, tau=10. if eav_mode == "apply_exp" else .2,
                                start_video_progress=0., end_video_progress=1.,
                                g_hard_limit=3.)
    cold[delivered]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/S25_Motion_Cold_{variant}/recovered")
    for key, node in (("200", audit), ("201", paired), ("202", source)):
        cold[key] = {"class_type": "PreviewAny", "inputs": {
            "source": [node, 1 if key != "201" else 2]}}
    return freeze, cold


def _checkpoint(output: Path) -> dict:
    root = output / "MiniMaxH3/latent_checkpoints"
    matches = []
    for path in root.rglob("*.h3latent.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            payload = json.loads(handle.metadata()["t8_native_latent_checkpoint_json"])
        if payload.get("checkpoint_id") == "motion_recovery_firstpass":
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen Motion first pass, got {len(matches)}")
    path, payload = matches[0]
    return {"path": path.relative_to(root).as_posix(),
            "file_sha256": shared._sha256_file(path),
            "manifest_json": json.dumps(payload["manifest"], ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")),
            "content_sha256": payload["manifest"]["content_sha256"],
            "source_file": str(path)}


def _stream_hash(path: Path, kind: str) -> str:
    command = (["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map",
                "0:v:0", "-pix_fmt", "rgb24", "-c:v", "rawvideo"] if kind == "video" else
               ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map",
                "0:a:0", "-c:a", "pcm_s16le"])
    result = subprocess.run(command + ["-f", "hash", "-hash", "sha256", "-"],
                            capture_output=True, text=True, timeout=120, check=True)
    line = result.stdout.strip()
    if not line.startswith("SHA256=") or len(line) != 71:
        raise ValueError("Strict decoded stream hash was not emitted")
    return line.removeprefix("SHA256=").lower()


def _media(run_root: Path, variant: str) -> dict:
    directory = run_root / f"output/MiniMaxH3/S25_Motion_Cold_{variant}"
    files = list(directory.glob("recovered_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("Expected one cold Motion joint AV MP4")
    path = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    picture = [item for item in streams if item.get("codec_type") == "video"]
    sound = [item for item in streams if item.get("codec_type") == "audio"]
    if (len(picture) != 1 or len(sound) != 1
            or picture[0].get("codec_name") != "h264" or sound[0].get("codec_name") != "aac"
            or (picture[0].get("width"), picture[0].get("height")) != (128, 64)
            or int(picture[0].get("nb_frames", 0)) != 22):
        raise ValueError("Cold Motion media streams differ from the probe contract")
    return {"path": str(path), "container_sha256": shared._sha256_file(path),
            "decoded_rgb24_sha256": _stream_hash(path, "video"),
            "decoded_pcm_s16le_sha256": _stream_hash(path, "audio")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(SOURCE_SHA), required=True)
    parser.add_argument("--eav-mode", choices=("report_only", "apply_exp"),
                        default="report_only")
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8882)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=85000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.)
    parser.add_argument("--server-start-timeout", type=float, default=240.)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root, args.python = args.comfy_root.resolve(), args.python.resolve()
    args.host = "127.0.0.1"
    freeze, cold = probe_graphs(args.variant, eav_mode=args.eav_mode)
    readiness = effect.preflight(args, effect.SOURCE_SHA[args.variant])
    readiness["checks"]["cold_port_free"] = not shared.port_is_listening(args.host, args.port + 1)
    readiness["ready"] = all(readiness["checks"].values())
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (PROJECT / "artifacts/development/modular-sampling-m4-motion-storage-20260925"
            / f"{run_id}-{args.variant}-{args.eav_mode}-real-gpu")
    root.mkdir(parents=True, exist_ok=False)
    freeze_root, cold_root = root / "freeze", root / "cold"
    freeze_root.mkdir()
    cold_root.mkdir()
    (root / "freeze-prompt.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    paths = root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2),
                     encoding="utf-8")
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "variant": args.variant,
              "eav_mode": args.eav_mode,
              "source_api_sha256": SOURCE_SHA[args.variant], "preflight": readiness,
              "continuous_reference_rgb_pcm": CONTINUOUS_RGB_PCM[args.variant][args.eav_mode],
              "run_root": str(root)}
    try:
        with shared.IsolatedServer(args, freeze_root, f"s25-{args.variant}-freeze") as server:
            report["freeze_pid"] = server.process.pid
            phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=freeze,
                timeout_seconds=args.timeout_seconds))
        (root / "freeze-phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        freeze_progress = [str(item.get("node")) for item in phase.get("events") or []
                           if item.get("type") == "progress"]
        if ((phase.get("terminal") or {}).get("type") != "execution_success"
                or freeze_progress.count("9") != 20):
            raise RuntimeError("Motion first-pass-only freeze did not complete Stock20")
        receipt = _checkpoint(freeze_root / "output")
        report["freeze_receipt"] = receipt
        target_root = cold_root / "output/MiniMaxH3/latent_checkpoints"
        target = target_root / receipt["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt["source_file"], target)
        if shared._sha256_file(target) != receipt["file_sha256"]:
            raise ValueError("Motion frozen first-pass transport changed file SHA")
        report["transport_file_sha256"] = receipt["file_sha256"]
        load_id = "39" if args.variant == "Windowed" else "37"
        cold[load_id]["inputs"].update(checkpoint_path=receipt["path"],
                                       expected_manifest_json=receipt["manifest_json"],
                                       expected_file_sha256=receipt["file_sha256"])
        (root / "cold-prompt.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        args.port += 1
        with shared.IsolatedServer(args, cold_root, f"s25-{args.variant}-cold") as server:
            report["cold_pid"] = server.process.pid
            phase = asyncio.run(capture._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=cold,
                timeout_seconds=args.timeout_seconds))
        (root / "cold-phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        cold_progress = [str(item.get("node")) for item in phase.get("events") or []
                         if item.get("type") == "progress"]
        stage = "19" if args.variant == "Windowed" else "18"
        checks = {"freeze_success": True,
                  "cold_success": (phase.get("terminal") or {}).get("type") == "execution_success",
                  "cold_no_firstpass": cold_progress.count("9") == 0 and "9" not in cold,
                  "cold_only_pass2_10": cold_progress.count(stage) == 10,
                  "both_servers_stopped": not any(shared.port_is_listening("127.0.0.1", port)
                                                  for port in (args.port - 1, args.port)),
                  "candidate_sources_unchanged": all(
                      shared._sha256_file(TARGET / f"Motion_Recovery_{args.variant}_{phase}_Relay_EAV_EXP.api.json").lower()
                      == expected for phase, expected in SOURCE_SHA[args.variant].items())}
        if checks["cold_success"]:
            report["eav"] = effect._preview_report(phase, "200")
            report["relay"] = effect._preview_report(phase, "201")
            report["motion_source"] = effect._preview_report(phase, "202")
            checks["effects_observed"] = (
                report["eav"]["status"] == "observed_" + args.eav_mode and
                report["eav"]["relay_attention_calls"] > 0 and
                report["relay"]["status"] == "paired_retained_model_and_conditioning" and
                report["motion_source"]["status"] == "source_bound_candidate_audited")
            checks["eav_numeric_gain_observed"] = (
                args.eav_mode == "report_only" or
                float(report["eav"]["feta"].get("g_max") or 0.) > 1.001)
            media = _media(cold_root, args.variant)
            report["cold_media"] = media
            checks["decoded_stream_parity"] = (
                (media["decoded_rgb24_sha256"], media["decoded_pcm_s16le_sha256"])
                == CONTINUOUS_RGB_PCM[args.variant][args.eav_mode])
        report["freeze_firstpass_progress"] = freeze_progress.count("9")
        report["cold_firstpass_progress"] = cold_progress.count("9")
        report["cold_secondpass_progress"] = cold_progress.count(stage)
        report["checks"] = checks
        report["status"] = ("pass_small_real_weight_cold_motion_parity_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("Motion cold real-weight mechanical parity failed")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
