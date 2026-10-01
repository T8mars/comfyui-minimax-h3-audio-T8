"""Qualify the pinned Chunked v2 freeze/HIGH-only pair with installed assets.

Default is read-only preflight. --confirm-run creates an owned isolated Core
profile, runs reduced-canvas copies in two separate Core processes, and keeps
all evidence private. This is mechanical qualification, not human acceptance.
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


SCHEMA = "t8.modular-sampling.chunked-v2-real-freeze-resume.v1"
CANDIDATES = (PROJECT / "artifacts/development/"
              "modular-sampling-m4-chunked-v2v4-resume-20260924/candidate-v2/v2")
FREEZE = CANDIDATES / "01_freeze_first_pass.api.json"
RESUME = CANDIDATES / "02_resume_high_only_DRAFT.api.json"
HASHES = {
    FREEZE: "f1a567dd77e65dc1f8dc969e4549e57b0ee676797cd2caa3a26e0897f530431a",
    RESUME: "2d935d4ec3159f7babb3fe1593c4839a42c45ddcdd6852ca6bef3847ff620aff",
}
ARTIFACTS = PROJECT / "artifacts/development/modular-sampling-m4-chunked-v2-real-gpu-20260924"
IMAGE = "10A.jpg"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> dict:
    if _sha(path) != HASHES[path]:
        raise ValueError(f"Chunked v2 candidate changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(kind: str, *, width: int, height: int) -> dict:
    if kind not in {"freeze", "resume"} or min(width, height) < 64 or width % 32 or height % 32:
        raise ValueError("Use freeze/resume and a 32-aligned test canvas >=64")
    graph = deepcopy(_candidate(FREEZE if kind == "freeze" else RESUME))
    required = {"5": "LoadImage", "6": "ImageScale",
                "7": "MiniMaxH3AudioConditioningT8"}
    required.update({"freeze": {"12": "SamplerCustomAdvanced",
                               "35": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"},
                     "resume": {"14": "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced",
                                "19": "VHS_VideoCombine",
                                "26": "MiniMaxH3ChunkedPass2SegmentEXPT8",
                                "31": "MiniMaxH3ChunkedPass2RelayBindEXPT8",
                                "33": "MiniMaxH3ChunkedPass2EAVApplyEXPT8",
                                "34": "MiniMaxH3ChunkedPass2EAVAuditEXPT8",
                                "35": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"}}[kind])
    if any(graph.get(node, {}).get("class_type") != name for node, name in required.items()):
        raise ValueError("Pinned Chunked v2 graph lost a required stage")
    if graph["7"]["inputs"]["length"] != 22 or graph["5"]["inputs"]["image"] != IMAGE:
        raise ValueError("Pinned Chunked v2 first-pass recipe changed")
    if kind == "resume" and "12" in graph:
        raise ValueError("HIGH-only graph unexpectedly contains first-pass sampler")
    high_width, high_height = width * 2, height * 2
    graph["6"]["inputs"].update(width=high_width, height=high_height)
    graph["7"]["inputs"].update(width=width, height=height)
    if kind == "freeze":
        graph["35"]["inputs"].update(confirm_save=True,
                                      filename_prefix="chunked_v2_real_first_pass")
        graph["200"] = {"class_type": "PreviewAny", "inputs": {"source": ["35", 4]}}
        graph["201"] = {"class_type": "PreviewAny", "inputs": {"source": ["35", 5]}}
    else:
        graph["14"]["inputs"].update(
            target_width=high_width, target_height=high_height,
            tile_width=high_width, tile_height=high_height, minimum_tile_size=64,
        )
        graph["30"]["inputs"].update(width=high_width, height=high_height)
        graph["19"]["inputs"]["filename_prefix"] = "MiniMaxH3/Chunked_v2_Real/resume"
        graph["202"] = {"class_type": "PreviewAny", "inputs": {"source": ["35", 7]}}
        graph["203"] = {"class_type": "PreviewAny", "inputs": {"source": ["34", 1]}}
    return graph


def preflight(args: argparse.Namespace) -> dict:
    freeze, resume = _candidate(FREEZE), _candidate(RESUME)
    models = args.comfy_root / "models"
    assets = {
        "base": models / "diffusion_models" / freeze["4"]["inputs"]["unet_name"],
        "clip": models / "text_encoders" / freeze["3"]["inputs"]["clip_name"],
        "video_vae": models / "vae" / freeze["1"]["inputs"]["vae_name"],
        "audio_vae": models / "vae" / freeze["2"]["inputs"]["vae_name"],
        "pdd_adapter": models / "loras" / freeze["8"]["inputs"]["pdd_lora_name"],
        "trained_upscaler": (models / "latent_upscale_models" /
                             resume["14"]["inputs"]["model_name"]),
        "first_last_frame": args.comfy_root / "input" / IMAGE,
    }
    gpu = shared.gpu_memory_mib()
    checks = {
        "candidate_sha_pinned": True,
        "assets_installed": all(path.is_file() for path in assets.values()),
        "owned_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                               gpu["free_mib"] >= args.min_free_vram_mib),
    }
    return {"schema": SCHEMA + ".preflight", "assets": {k: str(v) for k, v in assets.items()},
            "candidate_sha256": {"freeze": HASHES[FREEZE], "resume": HASHES[RESUME]},
            "gpu": gpu, "checks": checks, "ready": all(checks.values())}


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _text(phase: dict, node_id: str) -> str:
    return pdd._phase_text(phase, node_id)


def _receipt(run_root: Path, phase: dict) -> dict:
    manifest = _text(phase, "200")
    report = json.loads(_text(phase, "201"))
    if report.get("status") != "SAVED_VERIFIED":
        raise ValueError("Real first-pass checkpoint was not verified")
    store = (run_root / "output/MiniMaxH3/latent_checkpoints").resolve()
    path = (store / report["checkpoint_path"]).resolve()
    if not path.is_relative_to(store) or not path.is_file():
        raise ValueError("Real first-pass checkpoint escaped or was not saved")
    if _sha(path).upper() != report["file_sha256"].upper():
        raise ValueError("Real checkpoint file hash differs from receipt")
    if not isinstance(json.loads(manifest), dict):
        raise ValueError("Real checkpoint manifest is not JSON object")
    return {"path": report["checkpoint_path"], "file_sha256": report["file_sha256"],
            "manifest_json": manifest, "absolute_path": str(path)}


def _media(run_root: Path, *, width: int, height: int,
           prefix: str = "resume") -> dict:
    if prefix not in {"resume", "eav_apply"}:
        raise ValueError("Only owned Chunked v2 media prefixes are auditable")
    folder = run_root / "output/MiniMaxH3/Chunked_v2_Real"
    files = list(folder.glob(f"{prefix}_*-audio.mp4"))
    if len(files) != 1:
        raise ValueError("Expected one owned v2 H.264/AAC output")
    path = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                            str(path)], capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                              "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                             capture_output=True, timeout=90, check=False)
    return {"path": str(path), "sha256": _sha(path), "checks": {
        "full_h264_video": len(video) == 1 and video[0].get("codec_name") == "h264" and
            int(video[0].get("nb_frames", 0)) == 22 and
            (video[0].get("width"), video[0].get("height")) == (width, height),
        "aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
        "full_decode": decoded.returncode == 0,
        "private_nonempty_file": path.stat().st_size > 0 and
            path.resolve().is_relative_to(run_root.resolve()),
    }}


def _combined_effect_checks(eav: dict, *, mode: str = "report_only") -> dict[str, bool]:
    """Relay+EAV has one composed owner; standalone Relay counts are inapplicable."""
    if mode not in {"report_only", "apply_exp"}:
        raise ValueError("Unknown composed EAV audit mode")
    forward_plan = eav.get("forward_plan") or {}
    forwards = forward_plan.get("forwards") or []
    attention = sum(int(item.get("attention_blocks", 0)) for item in forwards)
    feta = eav.get("feta") or {}
    return {
        "composed_eav_mode_observed": eav.get("status") ==
            ("observed_report_only" if mode == "report_only" else "observed_apply_exp") and
            (eav.get("config") or {}).get("mode") == mode,
        "combined_four_forwards": eav.get("completed_forwards") == 4 and
            feta.get("model_forward_count") == 4,
        "composed_relay_required": eav.get("relay_required") is True,
        "relay_routed_attention_covered": attention > 0 and
            eav.get("selector_calls") == attention and
            eav.get("relay_attention_calls") == attention,
        "actual_eav_attention_measurement": feta.get("attention_measurement_count", 0) > 0,
        "apply_gain_when_enabled": mode != "apply_exp" or feta.get("g_max", 0) > 1.0,
        "absolute_clock_matched": eav.get("clock_match") is True,
        "no_false_quality_or_cache_claim": eav.get("quality_accepted") is False and
            eav.get("cache_reuse_authorized") is False,
    }


def _audit(run_root: Path, freeze: dict, resume: dict, receipt: dict,
           *, width: int, height: int, port: int) -> dict:
    resume_graph = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    loaded = json.loads(_text(resume, "202"))
    eav = json.loads(_text(resume, "203"))
    media = _media(run_root, width=width * 2, height=height * 2)
    def steps(phase, node):
        return sum(event.get("type") == "progress" and str(event.get("node")) == node
                   for event in phase.get("events", []))
    checks = {
        "freeze_success": (freeze.get("terminal") or {}).get("type") == "execution_success",
        "resume_success": (resume.get("terminal") or {}).get("type") == "execution_success",
        "first_pass_4_steps": steps(freeze, "12") == 4,
        "resume_only_high_4_steps": steps(resume, "26") == 4 and steps(resume, "12") == 0,
        "resume_graph_has_no_first_pass": "12" not in resume_graph,
        "exact_checkpoint_receipt": resume_graph["35"]["inputs"] == {
            "checkpoint_path": receipt["path"],
            "expected_manifest_json": receipt["manifest_json"],
            "expected_file_sha256": receipt["file_sha256"],
            "hash_chunk_megabytes": 8,
        },
        "verified_external_load": loaded.get("status") == "MATCH_EXTERNAL" and
            loaded.get("external_manifest_verified") is True,
        "checkpoint_still_unchanged": _sha(Path(receipt["absolute_path"])).upper() ==
            receipt["file_sha256"].upper(),
        **_combined_effect_checks(eav),
        "source_candidates_unchanged": all(_sha(path) == value
                                            for path, value in HASHES.items()),
        "owned_core_stopped": not shared.port_is_listening("127.0.0.1", port),
        **media["checks"],
    }
    return {"schema": SCHEMA + ".audit", "checks": checks, "media": media,
            "effect_reports": {"composed_eav_relay": eav},
            "standalone_relay_audit_scope": "not_applicable_after_EAV_composition",
            "status": "small_canvas_mechanical_pass_not_quality_acceptance"
                      if all(checks.values()) else "fail"}


def _decoded_video_md5(path: Path) -> str:
    command = ["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
               "-map", "0:v:0", "-c:v", "rawvideo", "-pix_fmt", "rgb24",
               "-f", "md5", "-"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=True)
    if not result.stdout.startswith("MD5="):
        raise ValueError("No decoded RGB video digest")
    return result.stdout.strip()


def run_apply_existing(args: argparse.Namespace) -> dict:
    run_root = args.apply_run_root.resolve()
    if run_root.parent != ARTIFACTS.resolve():
        raise ValueError("EAV apply target must be an owned Chunked v2 run")
    prior = json.loads((run_root / "audit-v2.json").read_text(encoding="utf-8"))
    original = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    if (prior.get("status") != "small_canvas_mechanical_pass_not_quality_acceptance"
            or not all(prior.get("checks", {}).values())):
        raise ValueError("EAV apply needs a passing owned frozen first-pass audit")
    if any((run_root / name).exists() for name in
           ("apply-prompt.json", "apply-phase.json", "apply-report.json")):
        raise FileExistsError("Owned EAV apply evidence already exists")
    receipt = original["receipt"]
    frozen = Path(receipt["absolute_path"])
    control_media = Path(prior["media"]["path"])
    if (not frozen.resolve().is_relative_to(run_root) or
            not control_media.resolve().is_relative_to(run_root) or
            _sha(frozen).upper() != receipt["file_sha256"].upper() or
            _sha(control_media) != prior["media"]["sha256"]):
        raise ValueError("Owned frozen input or control media changed")
    graph = json.loads((run_root / "resume-prompt.json").read_text(encoding="utf-8"))
    if ("12" in graph or graph["32"]["inputs"]["mode"] != "report_only" or
            graph["35"]["inputs"]["expected_file_sha256"] != receipt["file_sha256"]):
        raise ValueError("Owned HIGH-only control graph changed")
    graph.pop("204", None)  # Standalone Relay audit is inapplicable after EAV composition.
    graph.pop("205", None)
    graph["32"]["inputs"]["mode"] = "apply_exp"
    graph["19"]["inputs"]["filename_prefix"] = "MiniMaxH3/Chunked_v2_Real/eav_apply"
    gpu = shared.gpu_memory_mib()
    checks = {"owned_port_free": not shared.port_is_listening("127.0.0.1", args.port),
              "vram_headroom": bool(gpu.get("available") and
                                    gpu["free_mib"] >= args.min_free_vram_mib),
              "source_candidates_unchanged": all(_sha(p) == h for p, h in HASHES.items())}
    if not args.confirm_run:
        return {"status": "preflight_only", "checks": checks, "gpu": gpu,
                "run_root": str(run_root)}
    if not all(checks.values()):
        raise RuntimeError("Owned EAV apply preflight did not pass")
    _write_new(run_root / "apply-prompt.json", graph)
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA + ".apply", "run_root": str(run_root),
              "status": "started", "preflight": checks,
              "frozen_file_sha256": receipt["file_sha256"],
              "control_media_sha256": prior["media"]["sha256"]}
    try:
        with shared.IsolatedServer(args, run_root, "chunked-v2-eav-apply") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "apply-phase.json", phase)
        if (phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Owned HIGH-only EAV apply graph failed")
        eav = json.loads(_text(phase, "203"))
        loaded = json.loads(_text(phase, "202"))
        media = _media(run_root, width=original["test_canvas"][0] * 2,
                       height=original["test_canvas"][1] * 2,
                       prefix="eav_apply")
        effect_checks = _combined_effect_checks(eav, mode="apply_exp")
        control_video = _decoded_video_md5(control_media)
        applied_video = _decoded_video_md5(Path(media["path"]))
        checks = {
            **effect_checks, **media["checks"],
            "verified_external_load": loaded.get("status") == "MATCH_EXTERNAL" and
                loaded.get("external_manifest_verified") is True,
            "no_first_pass_in_graph": "12" not in graph,
            "only_high_four_steps": sum(event.get("type") == "progress" and
                str(event.get("node")) == "26" for event in phase.get("events", [])) == 4 and
                not any(event.get("type") == "progress" and
                        str(event.get("node")) == "12" for event in phase.get("events", [])),
            "decoded_video_changed_by_eav": control_video != applied_video,
            "frozen_file_unchanged": _sha(frozen).upper() == receipt["file_sha256"].upper(),
            "control_media_unchanged": _sha(control_media) == prior["media"]["sha256"],
            "source_candidates_unchanged": all(_sha(p) == h for p, h in HASHES.items()),
            "owned_core_stopped": not shared.port_is_listening("127.0.0.1", args.port),
        }
        report.update(checks=checks, effect_report=eav, media=media,
                      decoded_video_md5={"control": control_video, "apply": applied_video})
        report["status"] = ("small_canvas_high_only_eav_apply_observed_not_quality_acceptance"
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
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8235)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Append a corrected combined-effect audit of an owned completed run")
    parser.add_argument("--apply-run-root", type=Path,
                        help="From a verified owned checkpoint run HIGH-only with EAV apply_exp")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    if args.audit_run_root is not None:
        run_root = args.audit_run_root.resolve()
        if run_root.parent != ARTIFACTS.resolve():
            raise ValueError("Audit target must be an owned Chunked v2 run")
        previous = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
        freeze_phase = json.loads((run_root / "freeze-phase.json").read_text(encoding="utf-8"))
        resume_phase = json.loads((run_root / "resume-phase.json").read_text(encoding="utf-8"))
        audit = _audit(run_root, freeze_phase, resume_phase, previous["receipt"],
                       width=previous["test_canvas"][0],
                       height=previous["test_canvas"][1], port=args.port)
        _write_new(run_root / "audit-v2.json", audit)
        print(json.dumps({"status": audit["status"], "checks": audit["checks"]},
                         ensure_ascii=False, indent=2))
        return 0 if audit["status"] != "fail" else 1
    if args.apply_run_root is not None:
        result = run_apply_existing(args)
        print(json.dumps({"status": result["status"], "checks": result["checks"]},
                         ensure_ascii=False, indent=2))
        return 0 if result["status"] in {
            "preflight_only", "small_canvas_high_only_eav_apply_observed_not_quality_acceptance"
        } else 1
    freeze = build_graph("freeze", width=args.width, height=args.height)
    resume = build_graph("resume", width=args.width, height=args.height)
    readiness = preflight(args)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_root = ARTIFACTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "input").mkdir()
    shutil.copy2(args.comfy_root / "input" / IMAGE, run_root / "input" / IMAGE)
    _write_new(run_root / "paths.json", probe_resource_config(args.comfy_root, PROJECT))
    _write_new(run_root / "freeze-prompt.json", freeze)
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA, "run_root": str(run_root), "status": "started",
              "test_canvas": [args.width, args.height, 22], "preflight": readiness}
    try:
        with shared.IsolatedServer(args, run_root, "chunked-v2-freeze") as server:
            report["freeze_core_pid"] = server.process.pid
            freeze_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=freeze,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "freeze-phase.json", freeze_phase)
        if (freeze_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Real v2 first pass did not complete")
        receipt = _receipt(run_root, freeze_phase)
        report["receipt"] = receipt
        resume["35"]["inputs"].update(
            checkpoint_path=receipt["path"],
            expected_manifest_json=receipt["manifest_json"],
            expected_file_sha256=receipt["file_sha256"],
        )
        _write_new(run_root / "resume-prompt.json", resume)
        with shared.IsolatedServer(args, run_root, "chunked-v2-resume") as server:
            report["resume_core_pid"] = server.process.pid
            resume_phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=resume,
                timeout_seconds=args.timeout_seconds))
        _write_new(run_root / "resume-phase.json", resume_phase)
        if (resume_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("Real v2 HIGH-only pass did not complete")
        audit = _audit(run_root, freeze_phase, resume_phase, receipt,
                       width=args.width, height=args.height, port=args.port)
        _write_new(run_root / "audit.json", audit)
        report["audit_status"] = audit["status"]
        report["status"] = audit["status"]
        return 0 if audit["status"] != "fail" else 1
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_new(run_root / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
