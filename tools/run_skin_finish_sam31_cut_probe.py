#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import uuid

from run_skin_finish_live_sam31_validation import (
    DEFAULT_PYTHON,
    ROOT,
    SAM_MODEL,
    GpuMonitor,
    IsolatedComfy,
    _gpu_sample,
    _history_errors,
    _json_write,
    _port_is_listening,
    _probe,
    _request_json,
    _sha256,
    _strict_decode,
    _utc_now,
    _wait_for_history,
)


DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-live-sam31-validation-20260825"
    / "skin_finish_sam31_obvious_cut_832x736x22.mp4"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-sam31-cut-probe-20260825"


def _build_prompt(
    source_name: str,
    scene_cut_threshold: float,
    *,
    sam_text: str = "front-facing person with a visible face",
    detection_threshold: float = 0.50,
    maximum_people: int = 3,
) -> dict:
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": SAM_MODEL.name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": str(sam_text),
                "clip": ["3", 1],
            },
        },
        "5": {
            "class_type": "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            "inputs": {
                "frames": ["2", 0],
                "model": ["3", 0],
                "conditioning": ["4", 0],
                "fps": 24.0,
                "maximum_people": int(maximum_people),
                "detection_threshold": float(detection_threshold),
                "detect_interval": 3,
                "scene_cut_threshold": float(scene_cut_threshold),
                "analysis_max_side": 640,
                "preview_stride": 1,
                "release_policy": "offload_sam31_after_track",
            },
        },
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}},
        "7": {"class_type": "PreviewAny", "inputs": {"source": ["5", 2]}},
        "8": {"class_type": "PreviewAny", "inputs": {"source": ["5", 3]}},
        "9": {"class_type": "PreviewAny", "inputs": {"source": ["5", 4]}},
    }


def _history_text(history: dict, node_id: str) -> str:
    outputs = history.get("outputs", {})
    node = outputs.get(node_id, {}) if isinstance(outputs, dict) else {}
    text = node.get("text", []) if isinstance(node, dict) else []
    if not text:
        raise RuntimeError(f"history did not retain PreviewAny text for node {node_id}")
    return str(text[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.28)
    parser.add_argument(
        "--sam-text",
        default="front-facing person with a visible face",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.50)
    parser.add_argument("--maximum-people", type=int, choices=(2, 3), default=3)
    parser.add_argument("--expected-minimum-shots", type=int, default=2)
    parser.add_argument("--expected-minimum-objects-per-shot", type=int, default=1)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=300.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    user_8188_before = _port_is_listening(args.host, 8188)
    gpu_before = _gpu_sample()
    preflight = {
        "schema": "h3_t8_skin_finish_sam31_cut_probe/preflight-v1",
        "created_at": _utc_now(),
        "source": str(source),
        "source_exists": source.is_file(),
        "python": str(args.python.resolve()),
        "python_exists": args.python.is_file(),
        "sam31": {"path": str(SAM_MODEL), "exists": SAM_MODEL.is_file()},
        "target_port_free": not _port_is_listening(args.host, args.port),
        "user_port_8188_observed_only": user_8188_before,
        "gpu": gpu_before,
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "confirmed": bool(args.confirm_run),
    }
    preflight["ready"] = bool(
        preflight["source_exists"]
        and preflight["python_exists"]
        and preflight["sam31"]["exists"]
        and preflight["target_port_free"]
        and gpu_before.get("available")
        and int(gpu_before.get("free_mib", 0)) >= args.minimum_free_vram_mib
        and args.expected_minimum_shots >= 1
        and 1 <= args.expected_minimum_objects_per_shot <= args.maximum_people
        and args.confirm_run
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _json_write(output_root / "preflight.json", preflight)
    if not preflight["ready"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("input", "output", "temp", "user", "logs", "prompt"):
        (run_root / name).mkdir()
    input_source = run_root / "input" / source.name
    shutil.copy2(source, input_source)
    prompt = _build_prompt(
        input_source.name,
        args.scene_cut_threshold,
        sam_text=args.sam_text,
        detection_threshold=args.detection_threshold,
        maximum_people=args.maximum_people,
    )
    _json_write(run_root / "prompt" / "prompt.json", prompt)

    monitor = GpuMonitor()
    server_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    history = None
    prompt_id = None
    server_pid = None
    try:
        monitor.start()
        with IsolatedComfy(
            python=args.python.resolve(),
            host=args.host,
            port=args.port,
            run_root=run_root,
            start_timeout=args.server_start_timeout,
        ) as isolated:
            server_pid = int(isolated.process.pid) if isolated.process else None
            response = _request_json(
                "POST",
                f"{server_url}/prompt",
                {"prompt": prompt, "client_id": f"skin-finish-cut-{run_id}"},
            )
            prompt_id = str(response["prompt_id"])
            history = _wait_for_history(server_url, prompt_id, args.prompt_timeout)
            _json_write(run_root / "history.json", history)
            errors = _history_errors(history)
            if errors:
                raise RuntimeError(f"ComfyUI prompt failed: {errors[-1]}")
    finally:
        monitor.stop()

    assert history is not None
    track_report = json.loads(_history_text(history, "7"))
    shot_count = int(_history_text(history, "8"))
    track_count = int(_history_text(history, "9"))
    previews = sorted((run_root / "temp").rglob("*.png"))
    report = {
        "schema": "h3_t8_skin_finish_sam31_cut_probe/v1",
        "created_at": _utc_now(),
        "run_id": run_id,
        "server_pid": server_pid,
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "probe": _probe(source),
            "strict_video": _strict_decode(source, "video"),
        },
        "scene_cut_threshold": float(args.scene_cut_threshold),
        "sam_text": str(args.sam_text),
        "detection_threshold": float(args.detection_threshold),
        "maximum_people": int(args.maximum_people),
        "expected_minimum_shots": int(args.expected_minimum_shots),
        "expected_minimum_objects_per_shot": int(
            args.expected_minimum_objects_per_shot
        ),
        "shot_count": shot_count,
        "shot_local_track_count": track_count,
        "track_report": track_report,
        "gpu": monitor.report(),
        "preview_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in previews
        ],
        "checks": {
            "source_strict_decode": True,
            "shot_count_matches_report": shot_count == int(track_report["shot_count"]),
            "minimum_shot_count_met": shot_count >= int(args.expected_minimum_shots),
            "minimum_objects_per_shot_met": all(
                int(value) >= int(args.expected_minimum_objects_per_shot)
                for value in track_report.get("objects_per_shot", [])
            )
            and len(track_report.get("objects_per_shot", [])) == shot_count,
            "preview_files_present": bool(previews),
            "sam_selectively_offloaded": bool(
                track_report.get("release", {}).get("performed")
            ),
            "server_stopped": not _port_is_listening(args.host, args.port),
            "user_8188_untouched": user_8188_before
            == _port_is_listening(args.host, 8188),
        },
        "boundary": (
            "One 22-frame, 0.612MP, deliberately obvious hard-cut SAM3.1 tracking probe. "
            "It validates shot-local reset mechanics only, not ParseNet, Skin Finish quality, "
            "identity truth, long-video continuity, pressure behavior or universal 16GiB safety."
        ),
    }
    report["checks"]["source_strict_decode"] = bool(
        report["source"]["strict_video"]["passed"]
    )
    report["passed"] = all(report["checks"].values())
    _json_write(run_root / "validation_report.json", report)
    _json_write(
        output_root / "latest.json",
        {"run_id": run_id, "report": str(run_root / "validation_report.json")},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
