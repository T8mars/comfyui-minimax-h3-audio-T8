#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def _validator():
    spec = importlib.util.spec_from_file_location("h3_t8_vram_validator", ROOT / "validate_h3_vram.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _set_seed(prompt: dict, seed: int) -> None:
    changed = False
    for node in prompt.values():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs and isinstance(inputs[key], int):
                inputs[key] = seed
                changed = True
    if not changed:
        raise ValueError("workflow contains no literal seed or noise_seed")


async def _run(args, validator):
    base = validator.load_api_prompt(args.workflow)
    runs = []
    for index in range(args.runs):
        prompt = deepcopy(base)
        seed = args.seed_start + index
        _set_seed(prompt, seed)
        runtime = await validator.collect_run(
            prompt,
            server=args.server,
            device_index=args.device_index,
            poll_interval=args.poll_interval,
            baseline_seconds=args.baseline_seconds,
            timeout_seconds=args.timeout,
            preview_method="none",
        )
        summary = runtime["summary"]
        total = runtime["server_snapshot"]["devices"][args.device_index]["vram_total"]
        peak = int(summary["peak_vram_used_bytes"])
        runs.append(
            {
                "index": index,
                "seed": seed,
                "status": runtime["status"],
                "duration_seconds": runtime["duration_seconds"],
                "baseline_vram_used_bytes": summary["baseline_vram_used_bytes"],
                "peak_vram_used_bytes": peak,
                "minimum_free_headroom_bytes": total - peak,
                "prompt_id": runtime["prompt_id"],
                "cached_nodes": [
                    event["data"].get("nodes", [])
                    for event in runtime["events"]
                    if event["type"] == "execution_cached"
                ],
            }
        )
        await asyncio.sleep(args.between_runs_seconds)
    await asyncio.sleep(args.post_wait_seconds)
    post = await validator.inspect_server(args.server)
    return runs, post


def main() -> int:
    parser = argparse.ArgumentParser(description="Run same-process speech VRAM repeats with distinct seeds.")
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=2608103401)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--between-runs-seconds", type=float, default=1.0)
    parser.add_argument("--post-wait-seconds", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    validator = _validator()
    runs, post = asyncio.run(_run(args, validator))
    peaks = [run["peak_vram_used_bytes"] for run in runs]
    baselines = [run["baseline_vram_used_bytes"] for run in runs]
    headrooms = [run["minimum_free_headroom_bytes"] for run in runs]
    device = post["devices"][args.device_index]
    report = {
        "schema": "minimax_h3_t8_speech_vram_repeat_matrix_v1",
        "workflow": str(args.workflow.resolve()),
        "run_count": len(runs),
        "runs": runs,
        "all_success": all(run["status"] == "success" for run in runs),
        "minimum_headroom_mib": min(headrooms) / (1024**2),
        "peak_range_mib": (max(peaks) - min(peaks)) / (1024**2),
        "baseline_range_mib": (max(baselines) - min(baselines)) / (1024**2),
        "baseline_first_to_last_mib": (baselines[-1] - baselines[0]) / (1024**2),
        "peak_mean_mib": statistics.fmean(peaks) / (1024**2),
        "headroom_512mib_gate_pass": min(headrooms) >= 512 * 1024**2,
        "post_wait_seconds": args.post_wait_seconds,
        "post_wait_used_mib": (device["vram_total"] - device["vram_free"]) / (1024**2),
        "peak_staircase_status": (
            "no_material_peak_staircase_in_this_matrix"
            if max(peaks) - min(peaks) < 256 * 1024**2
            else "material_peak_range_requires_review"
        ),
        "baseline_staircase_status": (
            "material_residency_staircase_detected"
            if baselines[-1] - baselines[0] >= 256 * 1024**2
            else "no_material_baseline_staircase_in_this_matrix"
        ),
        "boundary": "One same-process matrix does not establish cross-GPU memory safety.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
