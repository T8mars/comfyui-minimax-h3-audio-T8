"""Probe v3/v4 HIGH-only seed edit and same-Core cache invalidation.

Default is read-only. --confirm-run uses one owned LRU64 Core; it never edits
the source candidate, frozen first-pass asset or original control/apply media.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_modular_chunked_v34_gpu as base  # noqa: E402
import run_modular_chunked_v34_negative_gpu as negative  # noqa: E402
import run_modular_chunked_v34_cancel_gpu as cancel  # noqa: E402


SCHEMA = "t8.modular-sampling.chunked-v34-cache-seed-edit.v1"
CASES = ("base", "repeat", "seed_edit", "seed_repeat")
EXPECTED_HIGH_STEPS = {"base": 3, "repeat": 0, "seed_edit": 3,
                       "seed_repeat": 0}


def cache_graphs(variant: str, control: dict) -> dict[str, dict]:
    spec = base.CONFIG[variant]
    if spec["low"] in control or control.get("16", {}).get("class_type") != "RandomNoise":
        raise ValueError("Only the pinned HIGH-only graph with explicit noise is allowed")
    seed = control["16"]["inputs"]["noise_seed"]
    if not isinstance(seed, int) or not 0 <= seed < 2**64 - 1:
        raise ValueError("HIGH seed is not safely incrementable")
    graphs = {}
    for case in CASES:
        graph = deepcopy(control)
        graph[spec["video"]]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/Chunked_{variant}_Real/cache_v1_{case}")
        if case in {"seed_edit", "seed_repeat"}:
            graph["16"]["inputs"]["noise_seed"] = seed + 1
        graphs[case] = graph
    return graphs


def phase_checks(variant: str, phases: dict[str, dict]) -> dict[str, bool]:
    spec = base.CONFIG[variant]
    checks = {}
    for case, expected_steps in EXPECTED_HIGH_STEPS.items():
        phase = phases.get(case) or {}
        events = phase.get("events") or []
        high_steps = sum(event.get("type") == "progress" and
                         str(event.get("node")) == spec["high"] for event in events)
        checks[f"{case}_high_steps_{expected_steps}"] = (
            (phase.get("terminal") or {}).get("type") == "execution_success" and
            high_steps == expected_steps)
        checks[f"{case}_no_low_events"] = not any(
            str(event.get("node")) == spec["low"] and
            event.get("type") in {"executing", "progress"} for event in events)
    return checks


def run(args: argparse.Namespace) -> dict:
    variant = args.variant
    run_root = args.run_root.resolve()
    original, receipt, control, protected = negative.load_owned_control(
        variant, run_root)
    width, height, _ = original["test_canvas"]
    graphs = cache_graphs(variant, control)
    folder = run_root / f"output/MiniMaxH3/Chunked_{variant}_Real"
    evidence_names = [f"cache-{case}-{kind}.json" for case in CASES
                      for kind in ("prompt", "phase")]
    evidence_names.append("cache-report.json")
    gpu = base.shared.gpu_memory_mib()
    checks = {
        "owned_port_free": not base.shared.port_is_listening("127.0.0.1", args.port),
        "vram_headroom": bool(gpu.get("available") and
                              gpu["free_mib"] >= args.min_free_vram_mib),
        "no_prior_cache_evidence": not any((run_root / name).exists()
                                           for name in evidence_names) and
            not list(folder.glob("cache_v1_*-audio.mp4")) and
            not list((run_root / "logs").glob(f"chunked-{variant}-cache.*.log")),
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
              "status": "started", "cache_mode": "lru_64", "preflight": readiness}
    try:
        phases = {}
        original_command = base.shared._server_command

        def cached_command(server_args: argparse.Namespace, server_root: Path):
            command = original_command(server_args, server_root)
            if command.count("--cache-none") != 1:
                raise ValueError("Owned Core no-cache startup contract changed")
            position = command.index("--cache-none")
            command[position:position + 1] = ["--cache-lru", "64"]
            return command

        with patch.object(base.shared, "_server_command", cached_command):
            with base.shared.IsolatedServer(
                    args, run_root, f"chunked-{variant}-cache") as server:
                report["owned_core_pid"] = server.process.pid
                for case, graph in graphs.items():
                    base._write_new(run_root / f"cache-{case}-prompt.json", graph)
                    phase = asyncio.run(base.pdd._submit_prompt_capture(
                        server=f"http://127.0.0.1:{args.port}", prompt=graph,
                        timeout_seconds=args.timeout_seconds))
                    phases[case] = phase
                    base._write_new(run_root / f"cache-{case}-phase.json", phase)
                    if not all(phase_checks(variant, {case: phase})[key]
                               for key in (f"{case}_high_steps_{EXPECTED_HIGH_STEPS[case]}",
                                           f"{case}_no_low_events")):
                        raise RuntimeError(f"{case} Core cache stage behavior differs")
        media = {case: base._media(run_root, variant, width, height,
                                   prefix=f"cache_v1_{case}") for case in CASES}
        digests = {
            case: {stream: cancel.decoded_md5(Path(item["path"]), stream)
                   for stream in ("video", "audio")}
            for case, item in media.items()
        }
        control_digests = {
            stream: cancel.decoded_md5(protected["control_media"][0], stream)
            for stream in ("video", "audio")
        }
        loaded = json.loads(base.pdd._phase_text(phases["base"], "202"))
        effects = {case: json.loads(base.pdd._phase_text(phases[case], "203"))
                   for case in ("base", "seed_edit")}
        audit = phase_checks(variant, phases)
        audit.update({
            "only_noise_seed_and_private_output_edits": graphs == cache_graphs(
                variant, control) and all(
                    json.loads((run_root / f"cache-{case}-prompt.json").read_text(
                        encoding="utf-8")) == graph
                    for case, graph in graphs.items()),
            "no_first_pass_in_graphs": all(
                base.CONFIG[variant]["low"] not in graph for graph in graphs.values()),
            "external_load_verified_and_receipt_constant":
                loaded.get("status") == "MATCH_EXTERNAL" and
                loaded.get("external_manifest_verified") is True and
                all(graph[base.CONFIG[variant]["load"]]["inputs"] ==
                    control[base.CONFIG[variant]["load"]]["inputs"]
                    for graph in graphs.values()),
            "real_relay_eav_on_both_sampling_runs": all(
                all(base._combined_effect_checks(effect).values())
                for effect in effects.values()),
            "base_matches_original_decoded_av": digests["base"] == control_digests,
            "repeat_matches_base_decoded_av": digests["repeat"] == digests["base"],
            "seed_edit_changes_decoded_video":
                digests["seed_edit"]["video"] != digests["base"]["video"],
            "seed_repeat_matches_edited_decoded_av":
                digests["seed_repeat"] == digests["seed_edit"],
            "first_pass_audio_preserved": len({
                control_digests["audio"],
                *(item["audio"] for item in digests.values())}) == 1,
            "protected_files_unchanged": all(
                base._sha(path).upper() == expected.upper()
                for path, expected in protected.values()),
            "source_candidates_unchanged": checks["source_candidates_unchanged"] and all(
                base._sha(base._path(variant, kind)) ==
                base.CONFIG[variant]["hashes"][index]
                for index, kind in enumerate(("freeze", "resume"))),
            "owned_core_stopped": not base.shared.port_is_listening(
                "127.0.0.1", args.port),
            "all_four_media_decode": all(all(item["checks"].values())
                                         for item in media.values()),
        })
        report.update(checks=audit, media=media,
                      decoded_md5={"original": control_digests, **digests},
                      frozen_file_sha256=receipt["file_sha256"])
        report["status"] = (
            "same_core_seed_edit_invalidates_only_high_cache"
            if all(audit.values()) else "fail")
        return report
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        base._write_new(run_root / "cache-report.json", report)


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
