#!/usr/bin/env python3
"""Run the Roadmap 29 Face Refine Window VRAM matrix strictly serially.

The default invocation is preflight-only. A real run requires ``--confirm-run``. The tool owns
one isolated ComfyUI process at a time, refuses an active user 8188 service, submits exactly one
prompt at a time, samples whole-device/ComfyUI-process memory at 100 ms, reads PyTorch peak
allocated/reserved counters from the plugin's local diagnostic route, waits 15 seconds after every
prompt, and atomically checkpoints its JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import psutil
import pynvml

try:
    from tools.run_nfe_resume_real_probe import (
        IsolatedServer,
        port_is_listening,
        submit_prompt,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` invocation.
    from run_nfe_resume_real_probe import IsolatedServer, port_is_listening, submit_prompt


SCHEMA = "t8.minimax_h3.face_refine_window_memory_matrix.v1"
STAIRCASE_LIMIT_MIB = 256.0
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_OUTPUT = (
    ROOT.parents[1]
    / "output"
    / "MiniMaxH3"
    / "roadmap29_window_p0_stock20_seed42_accept0_23_20260905_00001_.mp4"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _request_json(method: str, url: str) -> dict:
    request = Request(url, method=method)
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _ffprobe_prompt(path: Path, ffprobe: str) -> dict:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format_tags=prompt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    raw = json.loads(completed.stdout)["format"]["tags"]["prompt"]
    prompt = json.loads(raw)
    for node in prompt.values():
        if isinstance(node, dict):
            node.pop("is_changed", None)
    return prompt


def build_case_prompt(
    source_prompt: dict,
    *,
    case: str,
    seed: int,
    output_label: str,
    window_index: int = 0,
) -> dict:
    prompt = copy.deepcopy(source_prompt)
    prompt["16"]["inputs"]["noise_seed"] = int(seed)
    prompt["27"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/roadmap29_memory_{output_label}"
    )
    if case == "90":
        return prompt
    if case == "124":
        prompt["1"]["inputs"]["file"] = "face_refine_validation_dance_124_736x416.mp4"
        prompt["3"]["inputs"]["image"] = "face_refine_identity_frame60.png"
        prompt["26"]["inputs"]["image"] = "face_refine_identity_frame60.png"
        plan = prompt["4"]["inputs"]
        plan.update(
            {
                "repair_ranges": "0-23",
                "context_before_frames": 0,
                "context_after_frames": 0,
                "min_render_frames": 124,
                "max_render_frames": 124,
                "scene_cut_threshold": 1.0,
                "short_shot_policy": "reject",
            }
        )
        prompt["5"]["inputs"].update({"window_index": 0, "pad_policy": "reject"})
        prompt["24"]["inputs"]["accepted_subranges"] = "0-23"
        return prompt
    if case == "consecutive":
        prompt["1"]["inputs"]["file"] = "face_refine_validation_dance_362_736x416.mp4"
        prompt["3"]["inputs"]["image"] = "face_refine_identity_frame60.png"
        prompt["26"]["inputs"]["image"] = "face_refine_identity_frame60.png"
        ranges = ("0-23", "124-147", "248-271")
        plan = prompt["4"]["inputs"]
        plan.update(
            {
                "repair_ranges": ",".join(ranges),
                "context_before_frames": 24,
                "context_after_frames": 42,
                "min_render_frames": 90,
                "max_render_frames": 90,
                "scene_cut_threshold": 1.0,
                "short_shot_policy": "reject",
            }
        )
        prompt["5"]["inputs"].update(
            {"window_index": int(window_index), "pad_policy": "reject"}
        )
        prompt["24"]["inputs"]["accepted_subranges"] = ranges[int(window_index)]
        return prompt
    raise ValueError(f"Unknown Face Refine memory case: {case}")


def matrix_groups() -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for case in ("90", "124"):
        groups.append((f"{case}_cold_1", [{"case": case, "phase": "cold", "ordinal": 1}]))
        groups.append((f"{case}_cold_2", [{"case": case, "phase": "cold", "ordinal": 2}]))
        runs = [{"case": case, "phase": "cold", "ordinal": 3}]
        runs.extend(
            {"case": case, "phase": "warm", "ordinal": index}
            for index in range(1, 4)
        )
        groups.append((f"{case}_cold_3_plus_warm", runs))
    groups.append(
        (
            "consecutive_three_windows",
            [
                {
                    "case": "consecutive",
                    "phase": "consecutive",
                    "ordinal": index + 1,
                    "window_index": index,
                }
                for index in range(3)
            ],
        )
    )
    return groups


class MemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.1):
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        pynvml.nvmlInit()
        self._device = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._process = psutil.Process(self.pid)

    def _run(self) -> None:
        started = time.monotonic()
        next_sample = started
        while not self._stop.is_set():
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._device)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._device)
                # ``memory_full_info`` is ~150 ms on this Windows host and would destroy the
                # required 100 ms GPU cadence. ``memory_info`` exposes the same private field on
                # Windows without that expensive USS scan.
                process_memory = self._process.memory_info()
                private = getattr(process_memory, "private", process_memory.rss)
                self.rows.append(
                    {
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "gpu_total_mib": memory.total / 1024**2,
                        "gpu_used_mib": memory.used / 1024**2,
                        "gpu_free_mib": memory.free / 1024**2,
                        "gpu_utilization_percent": int(util.gpu),
                        "process_rss_mib": process_memory.rss / 1024**2,
                        "process_private_mib": private / 1024**2,
                    }
                )
            except (psutil.Error, pynvml.NVMLError):
                pass
            next_sample += self.interval_seconds
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]) if self.rows else [])
            if self.rows:
                writer.writeheader()
                writer.writerows(self.rows)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            raise RuntimeError("Memory telemetry produced no samples")
        elapsed = max(
            float(self.rows[-1]["elapsed_seconds"])
            - float(self.rows[0]["elapsed_seconds"]),
            0.0,
        )
        observed_hz = (len(self.rows) - 1) / elapsed if elapsed and len(self.rows) > 1 else 0.0
        return {
            "sample_count": len(self.rows),
            "interval_seconds": self.interval_seconds,
            "observed_sample_hz": observed_hz,
            "max_gpu_used_mib": max(row["gpu_used_mib"] for row in self.rows),
            "min_gpu_free_mib": min(row["gpu_free_mib"] for row in self.rows),
            "max_process_rss_mib": max(row["process_rss_mib"] for row in self.rows),
            "max_process_private_mib": max(
                row["process_private_mib"] for row in self.rows
            ),
            "final_gpu_used_mib": self.rows[-1]["gpu_used_mib"],
            "final_gpu_free_mib": self.rows[-1]["gpu_free_mib"],
            "final_process_private_mib": self.rows[-1]["process_private_mib"],
        }


def _strict_media(path: Path, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    commands = {
        "video": [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        "audio": [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"],
        "joint": [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
    }
    exits = {}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, timeout=1800)
        exits[name] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"Strict {name} decode failed for {path}")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "strict_decode_exit": exits,
        "streams": json.loads(probe.stdout)["streams"],
    }


def _server_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        host=args.host,
        port=args.port,
        python=args.python,
        comfy_root=args.comfy_root,
        reserve_vram_gib=args.reserve_vram_gib,
        lowvram=False,
        server_start_timeout=args.server_start_timeout,
        extra_whitelist_custom_nodes=("ComfyUI-H3-FaceRefine",),
    )


def _output_for_label(run_root: Path, label: str) -> Path:
    candidates = sorted(
        (run_root / "output" / "MiniMaxH3").glob(f"roadmap29_memory_{label}*.mp4")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one output for {label}, found {len(candidates)}")
    return candidates[0]


def _run_one(
    args: argparse.Namespace,
    run_root: Path,
    source_prompt: dict,
    server: IsolatedServer,
    run: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    label = f"{sequence:02d}_{run['case']}_{run['phase']}_{run['ordinal']}"
    prompt = build_case_prompt(
        source_prompt,
        case=run["case"],
        seed=42,
        output_label=label,
        window_index=int(run.get("window_index", 0)),
    )
    server_url = f"http://{args.host}:{args.port}"
    before = _request_json("POST", f"{server_url}/minimax_h3_t8/runtime_memory/reset_peak")
    sampler = MemorySampler(server.process.pid, args.sample_interval_seconds)
    sampler.start()
    started = time.monotonic()
    try:
        execution = asyncio.run(
            submit_prompt(
                server=server_url,
                prompt=prompt,
                timeout_seconds=args.prompt_timeout_seconds,
            )
        )
        terminal = execution.get("terminal") or {}
        if terminal.get("type") != "execution_success":
            raise RuntimeError(f"Prompt {label} did not succeed: {terminal}")
        time.sleep(args.post_run_seconds)
        after = _request_json("GET", f"{server_url}/minimax_h3_t8/runtime_memory")
    finally:
        sampler.stop()
    telemetry_path = run_root / "telemetry" / f"{label}.csv"
    sampler.write_csv(telemetry_path)
    media = _strict_media(_output_for_label(run_root, label), args.ffmpeg, args.ffprobe)
    summary = sampler.summary()
    summary["passes_512_mib"] = summary["min_gpu_free_mib"] >= 512.0
    return {
        **run,
        "label": label,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prompt_id": execution["prompt_id"],
        "terminal": terminal,
        "torch_before": before,
        "torch_after": after,
        "telemetry_csv": str(telemetry_path.resolve()),
        "memory": summary,
        "media": media,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    required_inputs = [
        args.source_output,
        args.comfy_root / "input" / "face_refine_upstream_badface_fixture_90f_320x320.mp4",
        args.comfy_root / "input" / "face_refine_validation_dance_124_736x416.mp4",
        args.comfy_root / "input" / "face_refine_validation_dance_362_736x416.mp4",
        args.comfy_root / "input" / "face_refine_identity_frame60.png",
    ]
    missing = [str(path) for path in required_inputs if not path.is_file()]
    pynvml.nvmlInit()
    memory = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
    free_mib = memory.free / 1024**2
    checks = {
        "required_files": not missing,
        "port_free": not port_is_listening(args.host, args.port),
        "user_8188_stopped": not port_is_listening("127.0.0.1", 8188),
        "start_free_vram": free_mib >= args.min_start_free_mib,
        "ffmpeg": bool(args.ffmpeg),
        "ffprobe": bool(args.ffprobe),
    }
    return {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "ready": all(checks.values()),
        "checks": checks,
        "missing": missing,
        "free_vram_mib": free_mib,
        "minimum_start_free_vram_mib": args.min_start_free_mib,
        "reserve_vram_gib": args.reserve_vram_gib,
        "strictly_serial": True,
        "matrix_run_count": sum(len(runs) for _name, runs in matrix_groups()),
    }


def _wait_for_group_headroom(minimum_mib: float, timeout_seconds: float = 300.0) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    deadline = time.monotonic() + timeout_seconds
    while True:
        free_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**2
        if free_mib >= float(minimum_mib):
            return free_mib
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU free VRAM stayed below {minimum_mib:.0f} MiB before the next serial group "
                f"(last observation {free_mib:.1f} MiB)"
            )
        time.sleep(1.0)


def summarize_runs(
    runs: list[dict[str, Any]], *, expected_run_count: int
) -> dict[str, Any]:
    """Build the release gate from durable per-prompt observations."""
    minima = [float(item["memory"]["min_gpu_free_mib"]) for item in runs]
    sample_rates = [float(item["memory"]["observed_sample_hz"]) for item in runs]

    def _private_delta(first_label: str, last_label: str) -> float | None:
        by_label = {str(item["label"]): item for item in runs}
        if first_label not in by_label or last_label not in by_label:
            return None
        first = float(by_label[first_label]["memory"]["final_process_private_mib"])
        last = float(by_label[last_label]["memory"]["final_process_private_mib"])
        return last - first

    staircase_deltas = {
        "90_cold3_to_warm3": _private_delta("03_90_cold_3", "06_90_warm_3"),
        "124_cold3_to_warm3": _private_delta("09_124_cold_3", "12_124_warm_3"),
        "consecutive_first_to_third": _private_delta(
            "13_consecutive_consecutive_1", "15_consecutive_consecutive_3"
        ),
    }
    measured_deltas = [
        value for value in staircase_deltas.values() if value is not None
    ]
    is_full_matrix = expected_run_count == 15
    staircase_complete = (not is_full_matrix) or len(measured_deltas) == 3
    no_staircase_growth = staircase_complete and all(
        value <= STAIRCASE_LIMIT_MIB for value in measured_deltas
    )
    all_media_strict_decode = all(
        all(code == 0 for code in item["media"]["strict_decode_exit"].values())
        for item in runs
    )
    all_prompts_succeeded = all(
        item.get("terminal", {}).get("type") == "execution_success" for item in runs
    )
    run_count_complete = len(runs) == expected_run_count
    all_runs_pass_512_mib = bool(minima) and all(value >= 512.0 for value in minima)
    telemetry_10hz = bool(sample_rates) and all(9.0 <= value <= 11.0 for value in sample_rates)
    gate_pass = all(
        (
            run_count_complete,
            all_runs_pass_512_mib,
            all_media_strict_decode,
            all_prompts_succeeded,
            telemetry_10hz,
            no_staircase_growth,
        )
    )
    return {
        "run_count": len(runs),
        "expected_run_count": expected_run_count,
        "run_count_complete": run_count_complete,
        "minimum_gpu_free_mib": min(minima) if minima else None,
        "all_runs_pass_512_mib": all_runs_pass_512_mib,
        "all_media_strict_decode": all_media_strict_decode,
        "all_prompts_succeeded": all_prompts_succeeded,
        "telemetry_10hz": telemetry_10hz,
        "minimum_observed_sample_hz": min(sample_rates) if sample_rates else None,
        "maximum_observed_sample_hz": max(sample_rates) if sample_rates else None,
        "staircase_limit_mib": STAIRCASE_LIMIT_MIB,
        "persistent_private_deltas_mib": staircase_deltas,
        "staircase_observations_complete": staircase_complete,
        "no_staircase_growth": no_staircase_growth,
        "gate_pass": gate_pass,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _preflight(args)
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "mode": args.mode,
        "status": "PREFLIGHT_READY" if preflight["ready"] else "ABSTAIN_PREFLIGHT",
        "preflight": preflight,
        "runs": [],
        "strictly_serial": True,
        "concurrent_prompt_count": 1,
    }
    _atomic_json(args.run_root / "report.json", report)
    if not args.confirm_run or not preflight["ready"]:
        return report
    source_prompt = _ffprobe_prompt(args.source_output, args.ffprobe)
    if args.mode == "single":
        groups = [
            (
                f"single_{args.single_case}",
                [
                    {
                        "case": args.single_case,
                        "phase": "single",
                        "ordinal": 1,
                        "window_index": args.single_window_index,
                    }
                ],
            )
        ]
    else:
        groups = matrix_groups()
    sequence = 0
    try:
        for group_name, runs in groups:
            if port_is_listening(args.host, args.port):
                raise RuntimeError("Isolated port became busy before a serial server group")
            _wait_for_group_headroom(args.min_start_free_mib)
            group_root = args.run_root / "servers" / group_name
            with IsolatedServer(_server_args(args), group_root, group_name) as server:
                for run_spec in runs:
                    sequence += 1
                    result = _run_one(
                        args, group_root, source_prompt, server, run_spec, sequence
                    )
                    report["runs"].append(result)
                    _atomic_json(args.run_root / "report.json", report)
        report["summary"] = summarize_runs(
            report["runs"],
            expected_run_count=sum(len(runs) for _name, runs in groups),
        )
        report["status"] = "PASS" if report["summary"]["gate_pass"] else "FAIL_GATE"
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["completed_at"] = _utc_now()
        _atomic_json(args.run_root / "report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--mode", choices=["single", "matrix"], default="single")
    parser.add_argument("--single-case", choices=["90", "124", "consecutive"], default="90")
    parser.add_argument("--single-window-index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--reserve-vram-gib", type=float, default=2.0)
    parser.add_argument("--min-start-free-mib", type=float, default=14000.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--post-run-seconds", type=float, default=15.0)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--comfy-root", type=Path, default=ROOT.parents[1]
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT.parents[2] / "python" / "python.exe"
    )
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "artifacts" / "roadmap29-window-memory-matrix-20260905",
    )
    args = parser.parse_args(argv)
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.source_output = args.source_output.resolve()
    args.run_root = args.run_root.resolve()
    args.ffmpeg = shutil.which("ffmpeg") or str(
        (args.comfy_root.parent / "ffmpeg" / "bin" / "ffmpeg.exe").resolve()
    )
    args.ffprobe = shutil.which("ffprobe") or str(
        (args.comfy_root.parent / "ffmpeg" / "bin" / "ffprobe.exe").resolve()
    )
    if not Path(args.ffmpeg).is_file() or not Path(args.ffprobe).is_file():
        parser.error("ffmpeg and ffprobe are required")
    if args.single_case != "consecutive" and args.single_window_index != 0:
        parser.error("--single-window-index applies only to the consecutive case")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"PREFLIGHT_READY", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
