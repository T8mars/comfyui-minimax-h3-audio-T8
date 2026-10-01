"""Cold-run an S26 PDD audio-only graph from a verified real-asset freeze file.

Default preflight is read-only. ``--confirm-run`` copies the exact verified
checkpoint into a new owned output store and starts one isolated GPU Core.
The saved candidate API is used with only its required receipt/frame widgets
filled; no first-pass sampler or production/user workflow is changed.
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

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s26_pdd_freeze_gpu as freeze  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.modular-sampling.s26-pdd-real-freeze-cold-audio-gpu.v1"
ROUTES = {
    "pdd8": {"load": "33", "guard": "34", "sampler": "21",
             "route_report": ["18", 3], "decision": ["26", 3], "save": "29"},
    "pdd4plus4": {"load": "43", "guard": "44", "sampler": "31",
                   "route_report": ["28", 3], "decision": ["36", 3], "save": "39"},
}


def _source(route: str) -> Path:
    return freeze.CANDIDATES / freeze.ROUTES[route][0] / "resume_audio.api.json"


def _freeze_evidence(run_root: Path, route: str) -> tuple[dict, Path]:
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((run_root / "stage-audit.json").read_text(encoding="utf-8"))
    if report.get("route") != route or audit.get("route") != route or audit.get("status") != "stage_execution_pass":
        raise ValueError("PDD freeze evidence does not match the requested route or stage gate")
    if report.get("status") != "saved_pdd_freeze_real_assets_mechanical_pass_not_quality_acceptance":
        raise ValueError("PDD freeze was not completed with real assets")
    store = (run_root / "output" / "MiniMaxH3" / "latent_checkpoints").resolve()
    path = (store / report["checkpoint"]["relative_path"]).resolve()
    if not path.is_relative_to(store) or not path.is_file():
        raise ValueError("PDD frozen file is missing or outside its owned store")
    if freeze._sha(path) != report["checkpoint"]["file_sha256"]:
        raise ValueError("PDD frozen file changed after the accepted Save receipt")
    return report, path


def build_resume_graph(route: str, freeze_report: dict) -> tuple[dict, str]:
    source = _source(route)
    original = json.loads(source.read_text(encoding="utf-8"))
    graph = deepcopy(original)
    pins = ROUTES[route]
    if sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) != 1:
        raise ValueError("Saved PDD audio-only graph no longer has exactly one sampler")
    if graph[pins["load"]]["class_type"] != "MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8":
        raise ValueError("Saved PDD audio-only graph lost strict verified AV Load")
    manifest = freeze_report["checkpoint"]["embedded_manifest"]
    graph[pins["load"]]["inputs"].update(
        checkpoint_path=freeze_report["checkpoint"]["relative_path"],
        expected_manifest_json=json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        expected_file_sha256=freeze_report["checkpoint"]["file_sha256"])
    graph[pins["guard"]]["inputs"]["expected_video_frame_count"] = 22
    graph[pins["save"]]["inputs"]["filename_prefix"] = f"MiniMaxH3/S26_PDD_ColdAudio/{route}_selected"
    graph["200"] = {"class_type": "PreviewAny", "inputs": {"source": pins["route_report"]}}
    graph["201"] = {"class_type": "PreviewAny", "inputs": {"source": pins["decision"]}}
    graph["202"] = {"class_type": "PreviewAny", "inputs": {"source": [pins["load"], 1]}}
    return graph, freeze._sha(source)


def preflight(args: argparse.Namespace, graph: dict, freeze_file: Path, source_sha: str) -> dict:
    gpu = shared.gpu_memory_mib()
    checks = {
        "source_candidate_sha_bound": bool(source_sha),
        "verified_freeze_file_present": freeze_file.is_file(),
        "isolated_port_free": not shared.port_is_listening("127.0.0.1", args.port),
        "free_vram_gate": bool(gpu.get("available") and gpu["free_mib"] >= args.min_free_vram_mib),
        "one_audio_tail_sampler": sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) == 1,
    }
    return {"schema": SCHEMA + ".preflight", "route": args.route, "checks": checks,
            "source_sha256": source_sha, "freeze_file_sha256": freeze._sha(freeze_file),
            "gpu": gpu, "ready": all(checks.values())}


def _copy_freeze_to_owned_store(source: Path, run_root: Path, relative: str, expected_sha: str) -> Path:
    store = (run_root / "output" / "MiniMaxH3" / "latent_checkpoints").resolve()
    target = (store / relative).resolve()
    if not target.is_relative_to(store) or target.exists():
        raise ValueError("Refusing ambiguous or existing destination for PDD frozen file")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if freeze._sha(target) != expected_sha:
        raise ValueError("Copied PDD frozen file does not match the source receipt")
    return target


def _tail_checks(route: str, phase: dict) -> dict[str, bool]:
    pins = ROUTES[route]
    progress = [str(event.get("node")) for event in phase.get("events", [])
                if event.get("type") == "progress"]
    executing = [str(event.get("node")) for event in phase.get("events", [])
                 if event.get("type") == "executing"]
    return {
        "terminal_success": (phase.get("terminal") or {}).get("type") == "execution_success",
        "exactly_four_audio_tail_steps": progress == [pins["sampler"]] * 4,
        "strict_load_before_tail": (pins["load"] in executing and pins["sampler"] in executing
                                   and executing.index(pins["load"]) < executing.index(pins["sampler"])),
        "no_first_pass_sampler_nodes": not any(node in executing for node in
                                           (("11",) if route == "pdd8" else ("12", "19"))),
    }


def _delivery_checks(report: dict, phase: dict) -> dict[str, bool]:
    route = report["route"]
    tail_id = "22" if route == "pdd8" else "32"
    audit = json.loads(pdd._phase_text(phase, tail_id))
    route_report = json.loads(report["route_report"])
    return {
        "strict_external_manifest_loaded": report.get("load_status") == "MATCH_EXTERNAL",
        "compatibility_route_allowed": route_report.get("decision") == "ALLOW",
        "candidate_audio_changed_and_finite": audit.get("audio_exact_equal") is False and
                                              audit.get("audio_finite") is True and
                                              float(audit.get("audio_rmse", 0)) > 0,
        "candidate_video_numerically_preserved": audit.get("video_finite") is True and
                                                  float(audit.get("video_max_abs", 1)) < 1e-5,
        "default_quality_gate_retained_original": report.get("quality_decision") ==
                                                  "ABSTAIN_HUMAN_REVIEW_REQUIRED",
    }


def _media_checks(run_root: Path, route: str) -> dict[str, bool]:
    output = run_root / "output" / "MiniMaxH3" / "S26_PDD_ColdAudio"
    files = list(output.glob(f"{route}_selected*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected exactly one owned PDD cold-tail MP4")
    video = files[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(video)], capture_output=True, text=True,
                           timeout=30, check=True)
    metadata = json.loads(probe.stdout)
    video_streams = [stream for stream in metadata["streams"] if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=60, check=False)
    expected_canvas = 128 if route == "pdd8" else 192
    return {
        "one_h264_video_22_frames": len(video_streams) == 1 and
                                     video_streams[0].get("codec_name") == "h264" and
                                     int(video_streams[0].get("nb_frames", 0)) == 22,
        "expected_scaled_canvas": len(video_streams) == 1 and
                                  (video_streams[0].get("width"), video_streams[0].get("height")) ==
                                  (expected_canvas, expected_canvas),
        "one_aac_audio_stream": len(audio_streams) == 1 and audio_streams[0].get("codec_name") == "aac",
        "full_audio_video_decode": decode.returncode == 0,
        "nonempty_private_file": video.stat().st_size > 0 and video.is_relative_to(run_root),
    }


def audit_existing(run_root: Path, route: str) -> dict:
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    phase = json.loads((run_root / "phase.json").read_text(encoding="utf-8"))
    if report.get("route") != route or report.get("status") != \
            "cold_audio_only_four_step_mechanical_pass_not_quality_acceptance":
        raise ValueError("Only a completed matching owned PDD cold-tail run can be audited")
    checks = {**_tail_checks(route, phase), **_delivery_checks(report, phase),
              **_media_checks(run_root, route)}
    result = {"schema": SCHEMA + ".delivery-audit", "route": route,
              "run_root": str(run_root), "checks": checks,
              "status": "delivery_mechanical_pass" if all(checks.values()) else "delivery_mechanical_fail",
              "boundary": "Tiny-size real-asset mechanical AV, not original-size or human quality acceptance."}
    output = run_root / "delivery-audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing PDD delivery audit")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--freeze-run-root", type=Path)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8863)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--audit-run-root", type=Path,
                        help="Offline media/delivery audit of an owned run; writes a new JSON report")
    args = parser.parse_args(argv)
    if args.audit_run_root is not None:
        result = audit_existing(args.audit_run_root.resolve(), args.route)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "delivery_mechanical_pass" else 1
    if args.freeze_run_root is None:
        parser.error("--freeze-run-root is required unless --audit-run-root is supplied")
    args.freeze_run_root = args.freeze_run_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    frozen, frozen_file = _freeze_evidence(args.freeze_run_root, args.route)
    graph, source_sha = build_resume_graph(args.route, frozen)
    readiness = preflight(args, graph, frozen_file, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run:
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT / "artifacts/development/modular-sampling-s26-pdd-resume-20260924" / f"{run_id}-{args.route}"
    run_root.mkdir(parents=True, exist_ok=False)
    receipt = frozen["checkpoint"]
    copied = _copy_freeze_to_owned_store(frozen_file, run_root, receipt["relative_path"], receipt["file_sha256"])
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(probe_resource_config(args.comfy_root, PROJECT), indent=2), encoding="utf-8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "route": args.route, "source_sha256": source_sha,
              "freeze_run_root": str(args.freeze_run_root), "freeze_file_sha256": receipt["file_sha256"],
              "copied_freeze": str(copied), "preflight": readiness, "status": "started"}
    try:
        with shared.IsolatedServer(args, run_root, f"s26-resume-{args.route}") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        report["checks"] = _tail_checks(args.route, phase)
        report["checks"]["candidate_unchanged"] = freeze._sha(_source(args.route)) == source_sha
        report["checks"]["source_freeze_unchanged"] = freeze._sha(frozen_file) == receipt["file_sha256"]
        report["checks"]["copied_freeze_unchanged"] = freeze._sha(copied) == receipt["file_sha256"]
        report["checks"]["server_stopped"] = not shared.port_is_listening("127.0.0.1", args.port)
        if report["checks"]["terminal_success"]:
            report["route_report"] = pdd._phase_text(phase, "200")
            report["quality_decision"] = pdd._phase_text(phase, "201")
            report["load_status"] = pdd._phase_text(phase, "202")
        report["status"] = ("cold_audio_only_four_step_mechanical_pass_not_quality_acceptance"
                            if all(report["checks"].values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("PDD saved cold audio-tail graph failed mechanical checks")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
