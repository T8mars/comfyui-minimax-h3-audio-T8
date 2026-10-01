"""Run the SHA-pinned S08 split candidate at its authored 448x256/73f canvas.

Only the private SaveVideo prefix changes. Default invocation is read-only
preflight; --confirm-run starts one owned Core. No legacy workflow is edited.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_s02_native_gpu as native_gpu  # noqa: E402
import run_modular_s08_fastv2_gpu as base  # noqa: E402
import run_modular_s08_dense_eav_relay_gpu as dense  # noqa: E402
import run_modular_s08_vsa_eav_gpu as effect  # noqa: E402
from run_modular_motion_storage_gpu import _stream_hash  # noqa: E402

SCHEMA = "t8.modular-sampling.s08-authored-canvas-real-gpu.v1"
PREFIX = "MiniMaxH3/S08_FastV2_AuthoredCanvas/full_selected"
ROOT = base.PROJECT / "artifacts/development/modular-sampling-m1-v2-authored-canvas-20260925"


def build_graph() -> tuple[dict, str]:
    graph, source_sha = base.build_probe_graph(width=448, height=256, frames=73)
    if source_sha != dense.FULL_SHA:
        raise ValueError("Pinned S08 authored candidate changed")
    original = json.loads(base.CANDIDATE.read_bytes())
    for node in ("10", "26"):
        graph[node]["inputs"]["min_tokens"] = original[node]["inputs"]["min_tokens"]
    graph["16"]["inputs"]["filename_prefix"] = PREFIX
    control = json.loads(json.dumps(original))
    control["16"]["inputs"]["filename_prefix"] = PREFIX
    if graph != control:
        raise ValueError("S08 authored graph changed beyond private output prefix")
    if (graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"],
            graph["40"]["inputs"]["length"], graph["47"]["inputs"]["length"]) != (
                448, 256, 73, 73):
        raise ValueError("S08 authored canvas or Relay clock changed")
    return graph, source_sha


def preflight(args: argparse.Namespace, source_sha: str) -> dict:
    result = base.preflight(args, source_sha)
    ram = native_gpu._free_physical_mib()
    result["schema"] = SCHEMA + ".preflight"
    result["physical_free_mib"] = ram
    result["checks"]["system_free_ram_gate"] = ram is not None and ram >= args.min_free_ram_mib
    result["checks"]["candidate_sha_pinned"] = source_sha == dense.FULL_SHA
    result["ready"] = all(result["checks"].values())
    return result


def _media(run_root: Path, stem: str = "full_selected") -> dict:
    files = list((run_root / "output/MiniMaxH3/S08_FastV2_AuthoredCanvas").glob(
        stem + "*.mp4"))
    if len(files) != 1:
        raise ValueError("Expected one authored-canvas S08 joint AV result")
    video = files[0].resolve()
    if not video.is_relative_to(run_root.resolve()):
        raise ValueError("S08 video escaped owned output")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)["streams"]
    picture = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                             "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                            capture_output=True, timeout=180, check=False)
    return {"path": str(video), "file_sha256": base.shared._sha256_file(video),
            "rgb24_sha256": _stream_hash(video, "video"),
            "pcm_s16le_sha256": _stream_hash(video, "audio"),
            "checks": {
                "h264_73_frames_896x512": len(picture) == 1 and
                    picture[0].get("codec_name") == "h264" and
                    int(picture[0].get("nb_frames", 0)) == 73 and
                    (picture[0].get("width"), picture[0].get("height")) == (896, 512),
                "one_aac_audio": len(audio) == 1 and audio[0].get("codec_name") == "aac",
                "full_decode": decode.returncode == 0,
                "nonempty_private_file": video.stat().st_size > 0,
            }}


def _effects(phase: dict, stages: tuple[str, ...] = ("low", "high")) -> dict:
    if not stages or any(stage not in ("low", "high") for stage in stages):
        raise ValueError("Unknown S08 effect audit stage")
    audits = {stage: effect._audit(phase, "45" if stage == "low" else "46")
              for stage in stages}
    checks = {}
    for stage, audit in audits.items():
        checks[stage + "_four_forwards"] = audit.get("completed_forwards") == 4
        checks[stage + "_relay_200_calls"] = audit.get("relay_required") is True and \
            audit.get("relay_attention_calls") == 200
        checks[stage + "_dense_profile"] = (audit.get("v2_dispatch") or {}).get("profile") == \
            "dense_compat_exp" and (audit.get("v2_dispatch") or {}).get("actual_vsa_dispatched") is False
        checks[stage + "_eav_authored_mode"] = (audit.get("config") or {}).get("mode") == "report_only"
    return {"audits": audits, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=base.PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--min-free-ram-mib", type=int, default=90000)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    graph, source_sha = build_graph()
    readiness = preflight(args, source_sha)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return 0 if readiness["ready"] else 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / f"{run_id}-full"
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf8")
    paths = run_root / "paths.json"
    paths.write_text(json.dumps(base.probe_resource_config(args.comfy_root, base.PROJECT), indent=2),
                     encoding="utf8")
    args.host = "127.0.0.1"
    args.extra_model_paths_config = paths
    report = {"schema": SCHEMA, "status": "started", "run_root": str(run_root),
              "candidate_sha256": source_sha, "preflight": readiness,
              "authored_canvas": [448, 256, 73]}
    try:
        with base.shared.IsolatedServer(args, run_root, "s08-authored-canvas") as server:
            report["owned_core_pid"] = server.process.pid
            phase = asyncio.run(base.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=graph,
                timeout_seconds=args.timeout_seconds))
        (run_root / "phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2),
                                             encoding="utf8")
        checks = base._stage_checks(phase)
        report["checks"] = checks
        if checks["terminal_success"]:
            for stage, node in (("low", "50"), ("high", "51")):
                report[stage + "_save"] = base._stage_save_receipt(phase, node)
                checks[stage + "_stage_file_sha"] = base._check_stage_file(
                    run_root, report[stage + "_save"])
            effects = _effects(phase)
            report["effects"] = effects["audits"]
            checks.update(effects["checks"])
            report["media"] = _media(run_root)
            checks.update(report["media"]["checks"])
        checks["source_candidate_unchanged"] = base.shared._sha256_file(base.CANDIDATE) == source_sha
        checks["server_stopped"] = not base.shared.port_is_listening("127.0.0.1", args.port)
        report["status"] = ("authored_canvas_real_v2_mechanical_pass_not_quality_acceptance"
                            if all(checks.values()) else "fail")
        if report["status"] == "fail":
            raise RuntimeError("S08 authored-canvas mechanical gate failed")
        return 0
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
