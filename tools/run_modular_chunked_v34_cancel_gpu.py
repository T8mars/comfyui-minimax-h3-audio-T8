"""Interrupt owned Chunked v3/v4 HIGH, then retry HIGH on a fresh Core.

Default is read-only preflight. Existing frozen assets, media and candidate
graphs are never overwritten. --confirm-run creates separate private outputs.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_chunked_v34_gpu as base  # noqa: E402
import run_modular_chunked_v34_negative_gpu as negative  # noqa: E402


SCHEMA = "t8.modular-sampling.chunked-v34-cancel-retry.v1"


def output_graph(variant: str, control: dict, *, prefix: str) -> dict:
    if prefix not in {"cancel_attempt", "cancel_retry"}:
        raise ValueError("Only owned cancellation output names are allowed")
    graph = deepcopy(control)
    spec = base.CONFIG[variant]
    if spec["low"] in graph or graph[spec["video"]]["class_type"] != "VHS_VideoCombine":
        raise ValueError("Cancellation graph must contain only the HIGH sampler")
    graph[spec["video"]]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/Chunked_{variant}_Real/{prefix}")
    return graph


def decoded_md5(path: Path, stream: str) -> str:
    if stream == "video":
        return base._decoded_video_md5(path)
    if stream != "audio":
        raise ValueError("Expected video or audio stream")
    command = ["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
               "-map", "0:a:0", "-ac", "2", "-ar", "32000",
               "-c:a", "pcm_s16le", "-f", "md5", "-"]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=90, check=True)
    if not result.stdout.startswith("MD5="):
        raise ValueError("No decoded PCM audio digest")
    return result.stdout.strip()


def phase_checks(variant: str, interrupted: dict, retry: dict,
                 attempt_graph: dict, retry_graph: dict) -> dict[str, bool]:
    spec = base.CONFIG[variant]
    initial_events = interrupted.get("events") or []
    retry_events = retry.get("events") or []
    progress = interrupted.get("progress_at_interrupt") or {}
    high_attempt = [event for event in initial_events
                    if str(event.get("node")) == spec["high"] and
                    event.get("type") == "progress"]
    high_retry = [event for event in retry_events
                  if str(event.get("node")) == spec["high"] and
                  event.get("type") == "progress"]
    return {
        "interrupt_acknowledged_during_high":
            interrupted.get("interrupt_response") is not None and
            (interrupted.get("terminal") or {}).get("type") ==
            "execution_interrupted" and
            int(progress.get("value") or 0) >= 1 and
            int(progress.get("value") or 0) < 3 and
            bool(high_attempt),
        "no_first_pass_in_either_graph": spec["low"] not in attempt_graph and
            spec["low"] not in retry_graph,
        "no_first_pass_progress": not any(
            str(event.get("node")) == spec["low"] and
            event.get("type") in {"executing", "progress"}
            for event in [*initial_events, *retry_events]),
        "fresh_core_retry_high_three_steps":
            (retry.get("terminal") or {}).get("type") == "execution_success" and
            len(high_retry) == 3,
    }


def run(args: argparse.Namespace) -> dict:
    variant = args.variant
    run_root = args.run_root.resolve()
    original, receipt, control, protected = negative.load_owned_control(
        variant, run_root)
    width, height, _ = original["test_canvas"]
    attempt_graph = output_graph(variant, control, prefix="cancel_attempt")
    retry_graph = output_graph(variant, control, prefix="cancel_retry")
    folder = run_root / f"output/MiniMaxH3/Chunked_{variant}_Real"
    names = ("cancel-attempt-prompt.json", "cancel-attempt-phase.json",
             "cancel-retry-prompt.json", "cancel-retry-phase.json",
             "cancel-report.json")
    gpu = base.shared.gpu_memory_mib()
    checks = {
        "owned_port_free": not base.shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                              gpu["free_mib"] >= args.min_free_vram_mib),
        "no_prior_cancel_evidence": not any((run_root / name).exists()
                                            for name in names) and
            not list(folder.glob("cancel_attempt_*-audio.mp4")) and
            not list(folder.glob("cancel_retry_*-audio.mp4")),
        "source_candidates_unchanged": all(
            base._sha(base._path(variant, kind)) ==
            base.CONFIG[variant]["hashes"][index]
            for index, kind in enumerate(("freeze", "resume"))),
    }
    readiness = {"schema": SCHEMA + ".preflight", "variant": variant,
                 "run_root": str(run_root), "checks": checks, "gpu": gpu,
                 "ready": all(checks.values())}
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
    if not args.confirm_run or not readiness["ready"]:
        return {"status": "preflight_only" if readiness["ready"] else "not_ready",
                "preflight": readiness}
    args.host = "127.0.0.1"
    args.input_directory = run_root / "input"
    args.extra_model_paths_config = run_root / "paths.json"
    report = {"schema": SCHEMA, "variant": variant, "run_root": str(run_root),
              "status": "started", "preflight": readiness}
    try:
        base._write_new(run_root / names[0], attempt_graph)
        with base.shared.IsolatedServer(
                args, run_root, f"chunked-{variant}-cancel-attempt") as server:
            report["interrupt_core_pid"] = server.process.pid
            interrupted = asyncio.run(base.shared.submit_prompt(
                server=f"http://127.0.0.1:{args.port}", prompt=attempt_graph,
                timeout_seconds=args.timeout_seconds,
                interrupt_node=base.CONFIG[variant]["high"],
                interrupt_after_step=1))
        base._write_new(run_root / names[1], interrupted)
        if ((interrupted.get("terminal") or {}).get("type") !=
                "execution_interrupted" or
                list(folder.glob("cancel_attempt_*-audio.mp4"))):
            raise RuntimeError("HIGH was not interrupted before private media output")
        base._write_new(run_root / names[2], retry_graph)
        with base.shared.IsolatedServer(
                args, run_root, f"chunked-{variant}-cancel-retry") as server:
            report["retry_core_pid"] = server.process.pid
            retried = asyncio.run(base.pdd._submit_prompt_capture(
                server=f"http://127.0.0.1:{args.port}", prompt=retry_graph,
                timeout_seconds=args.timeout_seconds))
        base._write_new(run_root / names[3], retried)
        media = base._media(run_root, variant, width, height, prefix="cancel_retry")
        loaded = json.loads(base.pdd._phase_text(retried, "202"))
        eav = json.loads(base.pdd._phase_text(retried, "203"))
        control_media = protected["control_media"][0]
        retry_media = Path(media["path"])
        digests = {
            label: {kind: decoded_md5(path, kind) for kind in ("video", "audio")}
            for label, path in (("control", control_media), ("retry", retry_media))
        }
        audit = phase_checks(variant, interrupted, retried,
                             attempt_graph, retry_graph)
        audit.update({
            "same_explicit_receipt": all(
                graph[base.CONFIG[variant]["load"]]["inputs"] ==
                control[base.CONFIG[variant]["load"]]["inputs"]
                for graph in (attempt_graph, retry_graph)) and
                control[base.CONFIG[variant]["load"]]["inputs"][
                    "expected_file_sha256"] == receipt["file_sha256"],
            "verified_external_load": loaded.get("status") == "MATCH_EXTERNAL" and
                loaded.get("external_manifest_verified") is True,
            **base._combined_effect_checks(eav),
            "no_interrupted_media": not list(folder.glob("cancel_attempt_*-audio.mp4")),
            "decoded_video_identical":
                digests["control"]["video"] == digests["retry"]["video"],
            "decoded_audio_identical":
                digests["control"]["audio"] == digests["retry"]["audio"],
            "protected_files_unchanged": all(
                base._sha(path).upper() == expected.upper()
                for path, expected in protected.values()),
            "source_candidates_unchanged": checks["source_candidates_unchanged"] and all(
                base._sha(base._path(variant, kind)) ==
                base.CONFIG[variant]["hashes"][index]
                for index, kind in enumerate(("freeze", "resume"))),
            "owned_core_stopped": not base.shared.port_is_listening(
                "127.0.0.1", args.port),
            **media["checks"],
        })
        report.update(checks=audit, media=media, decoded_md5=digests)
        report["status"] = (
            "interrupted_high_then_fresh_core_high_only_retry_pass"
            if all(audit.values()) else "fail")
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        base._write_new(run_root / "cancel-report.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(base.CONFIG), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8236)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    report = run(args)
    print(json.dumps({"status": report["status"],
                      "checks": report.get("checks", report["preflight"]["checks"])},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] not in {"fail", "not_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
