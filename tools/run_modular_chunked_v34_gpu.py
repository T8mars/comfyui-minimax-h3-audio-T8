"""Run SHA-pinned Chunked v3/v4 freeze and HIGH-only drafts on owned Cores.

The default is read-only preflight. --confirm-run executes reduced-canvas
copies with installed weights in separate isolated Core processes. Neither
the source candidates nor old workflows are edited or promoted.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.chunked-v34-real-freeze-resume.v1"
SOURCE = (PROJECT / "artifacts/development/"
          "modular-sampling-m4-chunked-v2v4-resume-20260924/candidate-v2")
ARTIFACTS = PROJECT / "artifacts/development/modular-sampling-m4-chunked-v34-real-gpu-20260924"
IMAGE = "codex_prompt_relay_fl2va_first.png"
MASK = "codex_h3_static_background_subject_mask_576x320.png"
CONFIG = {
    "v3": {
        "hashes": ("45c6e6c9cfddf0577e78865feb000a9192d5a14c4142e6e0b63c7dc7057e0380",
                   "7313d04a7eb035f66a8e55835b8167365e1555d31640cf08e7d0d456966b4241"),
        "save": "37", "load": "37", "high": "27", "relay_plan": "31",
        "relay_conditioning": "32", "eav_config": "34", "eav_audit": "36",
        "video": "21", "low": "12",
    },
    "v4": {
        "hashes": ("153f1e47c35cd927ee82ff3b0f0497367d03b7ebb0408e6b219c13884f508dbc",
                   "2ca8a3e5a38bbcfe9da2ed0f31628d2f80c082e944c79893027aed627ec171ea"),
        "save": "47", "load": "47", "high": "37", "relay_plan": "41",
        "relay_conditioning": "42", "eav_config": "44", "eav_audit": "46",
        "video": "21", "low": "12",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(variant: str, kind: str) -> Path:
    name = ("01_freeze_first_pass.api.json" if kind == "freeze"
            else "02_resume_high_only_DRAFT.api.json")
    return SOURCE / variant / name


def _candidate(variant: str, kind: str) -> dict:
    path = _path(variant, kind)
    expected = CONFIG[variant]["hashes"][0 if kind == "freeze" else 1]
    if _sha(path) != expected:
        raise ValueError(f"Pinned Chunked {variant} source changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(variant: str, kind: str, *, width: int, height: int) -> dict:
    if (variant not in CONFIG or kind not in {"freeze", "resume"} or
            min(width, height) < 64 or width % 32 or height % 32):
        raise ValueError("Use v3/v4 freeze/resume and a 32-aligned canvas >=64")
    spec = CONFIG[variant]
    graph = deepcopy(_candidate(variant, kind))
    required = {"5": "LoadImage", "6": "ImageScale",
                "8": "MiniMaxH3AudioConditioningT8"}
    if kind == "freeze":
        required.update({"12": "SamplerCustomAdvanced",
                         spec["save"]: "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"})
    else:
        required.update({"14": ("MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
                                if variant == "v3" else
                                "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"),
                         spec["high"]: "MiniMaxH3ChunkedPass2SegmentEXPT8",
                         spec["load"]: "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
                         spec["eav_audit"]: "MiniMaxH3ChunkedPass2EAVAuditEXPT8",
                         spec["video"]: "VHS_VideoCombine"})
    if any(graph.get(node, {}).get("class_type") != cls for node, cls in required.items()):
        raise ValueError("Pinned Chunked graph lost a required stage")
    if (graph["5"]["inputs"]["image"] != IMAGE or
            graph["8"]["inputs"]["length"] != 124):
        raise ValueError("Pinned Chunked source/media clock changed")
    if kind == "resume" and spec["low"] in graph:
        raise ValueError("HIGH-only graph contains a first-pass sampler")
    target_width, target_height = width * 2, height * 2
    graph["6"]["inputs"].update(width=target_width, height=target_height)
    graph["8"]["inputs"].update(width=width, height=height, length=22)
    if variant == "v4":
        if graph["26"]["inputs"]["image"] != MASK:
            raise ValueError("Pinned inherited-mask source changed")
        graph["30"]["inputs"].update(width=width, height=height)
        graph["24"]["inputs"]["amount"] = 22
    if kind == "freeze":
        graph[spec["save"]]["inputs"].update(
            confirm_save=True, filename_prefix=f"chunked_{variant}_real_first_pass")
        graph["200"] = {"class_type": "PreviewAny",
                        "inputs": {"source": [spec["save"], 4]}}
        graph["201"] = {"class_type": "PreviewAny",
                        "inputs": {"source": [spec["save"], 5]}}
    else:
        plan = graph["14"]["inputs"]
        plan.update(target_width=target_width, target_height=target_height,
                    tile_width=target_width, tile_height=target_height,
                    minimum_tile_size=64)
        graph[spec["relay_plan"]]["inputs"]["length"] = 22
        graph[spec["relay_conditioning"]]["inputs"].update(
            width=target_width, height=target_height)
        graph[spec["video"]]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/Chunked_{variant}_Real/resume")
        graph["202"] = {"class_type": "PreviewAny",
                        "inputs": {"source": [spec["load"], 7]}}
        graph["203"] = {"class_type": "PreviewAny",
                        "inputs": {"source": [spec["eav_audit"], 1]}}
    return graph


def preflight(args: argparse.Namespace) -> dict:
    variant = args.variant
    freeze, resume = _candidate(variant, "freeze"), _candidate(variant, "resume")
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / freeze["4"]["inputs"]["unet_name"],
        "clip": models / "text_encoders" / freeze["3"]["inputs"]["clip_name"],
        "video_vae": models / "vae" / freeze["1"]["inputs"]["vae_name"],
        "audio_vae": models / "vae" / freeze["2"]["inputs"]["vae_name"],
        "turbo_lora": models / "loras" / freeze["7"]["inputs"]["lora_name"],
        "trained_upscaler": (models / "latent_upscale_models" /
                             resume["14"]["inputs"]["model_name"]),
        "first_last_frame": args.comfy_root / "input" / IMAGE,
    }
    if variant == "v4":
        assets["inherited_video_mask"] = args.comfy_root / "input" / MASK
    gpu = shared.gpu_memory_mib()
    checks = {
        "candidate_sha_pinned": True,
        "assets_installed": all(path.is_file() for path in assets.values()),
        "owned_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                               gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "variant": variant,
            "candidate_sha256": {"freeze": CONFIG[variant]["hashes"][0],
                                 "resume": CONFIG[variant]["hashes"][1]},
            "assets": {key: str(path) for key, path in assets.items()},
            "gpu": gpu, "checks": checks, "ready": all(checks.values())}


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _receipt(run_root: Path, phase: dict) -> dict:
    manifest = pdd._phase_text(phase, "200")
    report = json.loads(pdd._phase_text(phase, "201"))
    if report.get("status") != "SAVED_VERIFIED":
        raise ValueError("First-pass AV checkpoint was not verified")
    store = (run_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    path = (store / report["checkpoint_path"]).resolve()
    if not path.is_relative_to(store) or not path.is_file():
        raise ValueError("First-pass AV checkpoint missing or outside owned profile")
    if _sha(path).upper() != report["file_sha256"].upper():
        raise ValueError("First-pass checkpoint file hash differs from receipt")
    decoded = json.loads(manifest)
    if not isinstance(decoded, dict):
        raise ValueError("First-pass checkpoint manifest is not an object")
    return {"path": report["checkpoint_path"], "manifest_json": manifest,
            "file_sha256": report["file_sha256"], "absolute_path": str(path),
            "manifest": decoded}


def _media(run_root: Path, variant: str, width: int, height: int,
           *, prefix: str = "resume") -> dict:
    if prefix not in {"resume", "eav_apply", "cancel_retry",
                      "cache_v1_base", "cache_v1_repeat",
                      "cache_v1_seed_edit", "cache_v1_seed_repeat"}:
        raise ValueError("Only owned Chunked probe outputs are auditable")
    folder = run_root / f"output/MiniMaxH3/Chunked_{variant}_Real"
    files = list(folder.glob(f"{prefix}_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one owned HIGH-only media file")
    path = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                            str(path)], capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=90, check=False)
    return {"path": str(path), "sha256": _sha(path), "checks": {
        "h264_22_frames_2x_canvas": len(video) == 1 and
            video[0].get("codec_name") == "h264" and
            int(video[0].get("nb_frames", 0)) == 22 and
            (video[0].get("width"), video[0].get("height")) == (width * 2, height * 2),
        "aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
        "full_decode": decode.returncode == 0,
        "private_nonempty_file": path.stat().st_size > 0 and
            path.resolve().is_relative_to(run_root.resolve()),
    }}


def _combined_effect_checks(eav: dict, *, mode: str = "report_only") -> dict[str, bool]:
    if mode not in {"report_only", "apply_exp"}:
        raise ValueError("Unsupported external EAV mode")
    forward_plan = eav.get("forward_plan") or {}
    attention = sum(int(item.get("attention_blocks", 0))
                    for item in forward_plan.get("forwards") or [])
    feta = eav.get("feta") or {}
    return {
        "composed_effect_mode": eav.get("status") == (
            "observed_report_only" if mode == "report_only" else "observed_apply_exp") and
            (eav.get("config") or {}).get("mode") == mode,
        "three_high_forwards": eav.get("completed_forwards") == 3 and
            feta.get("model_forward_count") == 3,
        "relay_required": eav.get("relay_required") is True,
        "relay_attention_covered": attention > 0 and
            eav.get("selector_calls") == attention and
            eav.get("relay_attention_calls") == attention,
        "eav_measured": feta.get("attention_measurement_count", 0) > 0,
        "apply_gain_when_enabled": mode != "apply_exp" or feta.get("g_max", 0) > 1.0,
        "absolute_clock_matched": eav.get("clock_match") is True,
        "no_quality_or_cache_claim": eav.get("quality_accepted") is False and
            eav.get("cache_reuse_authorized") is False,
    }


def _audit(run_root: Path, variant: str, freeze: dict, resume: dict,
           receipt: dict, width: int, height: int, port: int) -> dict:
    spec = CONFIG[variant]
    resumed = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    loaded = json.loads(pdd._phase_text(resume, "202"))
    eav = json.loads(pdd._phase_text(resume, "203"))
    media = _media(run_root, variant, width, height)
    def steps(phase, node):
        return sum(event.get("type") == "progress" and str(event.get("node")) == node
                   for event in phase.get("events", []))
    checks = {
        "freeze_success": (freeze.get("terminal") or {}).get("type") == "execution_success",
        "resume_success": (resume.get("terminal") or {}).get("type") == "execution_success",
        "first_pass_full_8_steps": steps(freeze, spec["low"]) == 8,
        "high_only_low_sigma_3_steps": steps(resume, spec["high"]) == 3 and
            steps(resume, spec["low"]) == 0,
        "resume_graph_has_no_first_pass": spec["low"] not in resumed,
        "exact_checkpoint_receipt": resumed[spec["load"]]["inputs"] == {
            "checkpoint_path": receipt["path"],
            "expected_manifest_json": receipt["manifest_json"],
            "expected_file_sha256": receipt["file_sha256"],
            "hash_chunk_megabytes": 8,
        },
        "verified_external_load": loaded.get("status") == "MATCH_EXTERNAL" and
            loaded.get("external_manifest_verified") is True,
        "checkpoint_unchanged": _sha(Path(receipt["absolute_path"])).upper() ==
            receipt["file_sha256"].upper(),
        **_combined_effect_checks(eav),
        "source_candidates_unchanged": all(
            _sha(_path(variant, kind)) == spec["hashes"][index]
            for index, kind in enumerate(("freeze", "resume"))),
        "owned_core_stopped": not shared.port_is_listening("127.0.0.1", port),
        **media["checks"],
    }
    return {"schema": SCHEMA + ".audit", "variant": variant, "checks": checks,
            "media": media, "effect_report": eav,
            "checkpoint_manifest": receipt["manifest"],
            "status": "small_canvas_real_assets_mechanical_pass_not_quality_acceptance"
                      if all(checks.values()) else "fail"}


def _decoded_video_md5(path: Path) -> str:
    command = ["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
               "-map", "0:v:0", "-c:v", "rawvideo", "-pix_fmt", "rgb24",
               "-f", "md5", "-"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=True)
    if not result.stdout.startswith("MD5="):
        raise ValueError("No decoded RGB video digest")
    return result.stdout.strip()


def build_apply_graph(variant: str, control: dict, receipt: dict,
                      *, width: int, height: int) -> dict:
    spec = CONFIG[variant]
    expected_receipt = {
        "checkpoint_path": receipt["path"],
        "expected_manifest_json": receipt["manifest_json"],
        "expected_file_sha256": receipt["file_sha256"],
        "hash_chunk_megabytes": 8,
    }
    expected = build_graph(variant, "resume", width=width, height=height)
    expected[spec["load"]]["inputs"].update(expected_receipt)
    if control != expected:
        raise ValueError("Owned HIGH-only control graph or checkpoint receipt changed")
    graph = deepcopy(control)
    graph[spec["eav_config"]]["inputs"]["mode"] = "apply_exp"
    graph[spec["video"]]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/Chunked_{variant}_Real/eav_apply")
    return graph


def run_apply_existing(args: argparse.Namespace) -> dict:
    variant = args.variant
    run_root = args.apply_run_root.resolve()
    if run_root.parent != (ARTIFACTS / variant).resolve():
        raise ValueError("EAV apply target must be an owned Chunked variant run")
    if any((run_root / name).exists() for name in
           ("apply-prompt.json", "apply-phase.json", "apply-report.json")):
        raise FileExistsError("Owned EAV apply evidence already exists")
    prior = json.loads((run_root / "audit.json").read_text(encoding="utf-8"))
    original = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    if (prior.get("status") !=
            "small_canvas_real_assets_mechanical_pass_not_quality_acceptance" or
            not all(prior.get("checks", {}).values()) or
            original.get("variant") != variant):
        raise ValueError("EAV apply needs a passing owned frozen control run")
    receipt = original["receipt"]
    frozen = Path(receipt["absolute_path"])
    control_media = Path(prior["media"]["path"])
    if (not frozen.resolve().is_relative_to(run_root) or
            not control_media.resolve().is_relative_to(run_root) or
            _sha(frozen).upper() != receipt["file_sha256"].upper() or
            _sha(control_media) != prior["media"]["sha256"]):
        raise ValueError("Owned frozen AV or control media changed")
    control = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    width, height, _ = original["test_canvas"]
    graph = build_apply_graph(variant, control, receipt, width=width, height=height)
    gpu = shared.gpu_memory_mib()
    checks = {
        "owned_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                              gpu["free_mib"] >= args.min_free_vram_mib),
        "source_candidates_unchanged": all(
            _sha(_path(variant, kind)) == CONFIG[variant]["hashes"][index]
            for index, kind in enumerate(("freeze", "resume"))),
    }
    if not args.confirm_run:
        return {"status": "preflight_only", "checks": checks, "gpu": gpu,
                "run_root": str(run_root)}
    if not all(checks.values()):
        raise RuntimeError("Owned EAV apply preflight did not pass")
    _write_new(run_root / "apply-prompt.json", graph)
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA + ".apply", "variant": variant,
              "run_root": str(run_root), "status": "started", "preflight": checks,
              "frozen_file_sha256": receipt["file_sha256"],
              "control_media_sha256": prior["media"]["sha256"]}
    try:
        with shared.IsolatedServer(args, run_root, f"chunked-{variant}-eav-apply") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "apply-phase.json", phase)
        if (phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Owned HIGH-only EAV apply graph failed")
        eav = json.loads(pdd._phase_text(phase, "203"))
        loaded = json.loads(pdd._phase_text(phase, "202"))
        media = _media(run_root, variant, width, height, prefix="eav_apply")
        control_video = _decoded_video_md5(control_media)
        applied_video = _decoded_video_md5(Path(media["path"]))
        spec = CONFIG[variant]
        checks = {
            **_combined_effect_checks(eav, mode="apply_exp"), **media["checks"],
            "verified_external_load": loaded.get("status") == "MATCH_EXTERNAL" and
                loaded.get("external_manifest_verified") is True,
            "no_first_pass_in_graph": spec["low"] not in graph,
            "only_high_three_steps": sum(
                event.get("type") == "progress" and
                str(event.get("node")) == spec["high"]
                for event in phase.get("events", [])) == 3 and not any(
                    event.get("type") == "progress" and
                    str(event.get("node")) == spec["low"]
                    for event in phase.get("events", [])),
            "decoded_video_changed_by_eav": control_video != applied_video,
            "frozen_file_unchanged": _sha(frozen).upper() == receipt["file_sha256"].upper(),
            "control_media_unchanged": _sha(control_media) == prior["media"]["sha256"],
            "source_candidates_unchanged": all(
                _sha(_path(variant, kind)) == spec["hashes"][index]
                for index, kind in enumerate(("freeze", "resume"))),
            "owned_core_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        }
        report.update(checks=checks, effect_report=eav, media=media,
                      decoded_video_md5={"control": control_video,
                                         "apply": applied_video})
        report["status"] = (
            "small_canvas_high_only_eav_apply_observed_not_quality_acceptance"
            if all(checks.values()) else "fail")
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_new(run_root / "apply-report.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(CONFIG), required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8236)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--apply-run-root", type=Path,
                        help="From a verified owned checkpoint run HIGH-only EAV apply_exp")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.apply_run_root is not None:
        result = run_apply_existing(args)
        print(json.dumps({"status": result["status"], "checks": result["checks"]},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if result["status"] in {
            "preflight_only", "small_canvas_high_only_eav_apply_observed_not_quality_acceptance"
        } else 1
    freeze = build_graph(args.variant, "freeze", width=args.width, height=args.height)
    resume = build_graph(args.variant, "resume", width=args.width, height=args.height)
    readiness = preflight(args)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = (ARTIFACTS / args.variant /
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "input").mkdir()
    shutil.copy2(args.comfy_root / "input" / IMAGE, run_root / "input" / IMAGE)
    if args.variant == "v4":
        shutil.copy2(args.comfy_root / "input" / MASK, run_root / "input" / MASK)
    _write_new(run_root / "paths.json", probe_resource_config(args.comfy_root, PROJECT))
    _write_new(run_root / "freeze-prompt.json", freeze)
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA, "variant": args.variant, "run_root": str(run_root),
              "test_canvas": [args.width, args.height, 22], "status": "started",
              "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, f"chunked-{args.variant}-freeze") as server:
            report["freeze_core_pid"] = server.process.pid
            frozen_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=freeze,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "freeze-phase.json", frozen_phase)
        if (frozen_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Real first pass did not complete")
        receipt = _receipt(run_root, frozen_phase)
        report["receipt"] = receipt
        resume[CONFIG[args.variant]["load"]]["inputs"].update(
            checkpoint_path=receipt["path"],
            expected_manifest_json=receipt["manifest_json"],
            expected_file_sha256=receipt["file_sha256"],
        )
        _write_new(run_root / "resume-prompt.json", resume)
        with shared.IsolatedServer(args, run_root, f"chunked-{args.variant}-resume") as server:
            report["resume_core_pid"] = server.process.pid
            resume_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=resume,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "resume-phase.json", resume_phase)
        if (resume_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Real HIGH-only low-sigma pass did not complete")
        audit = _audit(run_root, args.variant, frozen_phase, resume_phase, receipt,
                       args.width, args.height, args.port)
        _write_new(run_root / "audit.json", audit)
        report["status"] = audit["status"]
        report["audit_status"] = audit["status"]
        return 0 if audit["status"] != "fail" else 1
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_new(run_root / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
