"""Generate low-load SAM3 subject-mask candidates for H3 background-lock tests.

This tool runs one isolated ComfyUI prompt on a single image.  It does not load
MiniMax H3, sample video, or overwrite the active validation mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

import run_skin_finish_live_sam31_validation as shared


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_PYTHON = COMFY_ROOT.parent / "python" / "python.exe"
DEFAULT_SOURCE = COMFY_ROOT / "input" / "codex_prompt_relay_fl2va_first.png"
DEFAULT_OUTPUT = ROOT / "artifacts" / "h3-subject-mask-probe-20260830"
SAM_MODEL = COMFY_ROOT / "models" / "checkpoints" / "sam3.1_multiplex_fp16.safetensors"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_prompt(source_name: str, width: int, height: int) -> dict[str, Any]:
    positive = json.dumps(
        [
            {"x": width // 2, "y": round(height * 0.31)},
            {"x": width // 2, "y": round(height * 0.62)},
            {"x": width // 2, "y": round(height * 0.84)},
        ]
    )
    negative = json.dumps(
        [
            {"x": round(width * 0.10), "y": height // 2},
            {"x": round(width * 0.90), "y": height // 2},
            {"x": width // 2, "y": round(height * 0.04)},
        ]
    )
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
        },
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": SAM_MODEL.name},
        },
        "4": {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["3", 0],
                "image": ["2", 0],
                "positive_coords": positive,
                "negative_coords": negative,
                "threshold": 0.50,
                "refine_iterations": 2,
                "individual_masks": False,
            },
        },
        "5": {
            "class_type": "ThresholdMask",
            "inputs": {"mask": ["4", 0], "value": 0.50},
        },
        "6": {
            "class_type": "GrowMask",
            "inputs": {"mask": ["5", 0], "expand": 16, "tapered_corners": True},
        },
        "7": {
            "class_type": "GrowMask",
            "inputs": {"mask": ["5", 0], "expand": 32, "tapered_corners": True},
        },
        "8": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["4", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "mask_probe/raw"},
        },
        "10": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["5", 0]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": "mask_probe/binary"},
        },
        "12": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["6", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "mask_probe/grow16"},
        },
        "14": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["7", 0]},
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {"images": ["14", 0], "filename_prefix": "mask_probe/grow32"},
        },
        "16": {
            "class_type": "EmptyImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
                "color": 0xFF2020,
            },
        },
        "17": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["2", 0],
                "source": ["16", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
                "mask": ["6", 0],
            },
        },
        "18": {
            "class_type": "ImageBlend",
            "inputs": {
                "image1": ["2", 0],
                "image2": ["17", 0],
                "blend_factor": 0.35,
                "blend_mode": "normal",
            },
        },
        "19": {
            "class_type": "SaveImage",
            "inputs": {"images": ["18", 0], "filename_prefix": "mask_probe/overlay16"},
        },
        "20": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["2", 0],
                "source": ["16", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
                "mask": ["7", 0],
            },
        },
        "21": {
            "class_type": "ImageBlend",
            "inputs": {
                "image1": ["2", 0],
                "image2": ["20", 0],
                "blend_factor": 0.35,
                "blend_mode": "normal",
            },
        },
        "22": {
            "class_type": "SaveImage",
            "inputs": {"images": ["21", 0], "filename_prefix": "mask_probe/overlay32"},
        },
    }
    return prompt


def _mask_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = image.convert("L")
        histogram = gray.histogram()
        total = max(1, gray.width * gray.height)
        weighted = sum(index * count for index, count in enumerate(histogram))
        return {
            "width": gray.width,
            "height": gray.height,
            "mean": weighted / (255.0 * total),
            "nonzero_fraction": sum(histogram[1:]) / total,
            "over_half_fraction": sum(histogram[128:]) / total,
            "full_fraction": histogram[255] / total,
        }


def _largest_component_candidates(
    binary_path: Path,
    source_path: Path,
    output_dir: Path,
    *,
    width: int,
    height: int,
) -> dict[str, Path]:
    """Remove SAM islands before expanding the subject safety margin.

    A tiny disconnected SAM hit becomes a large editable background patch after
    GrowMask.  Keep only the largest foreground component, then grow that one
    subject with an elliptical kernel.  These are diagnostic artifacts; the
    original SAM outputs remain untouched.
    """

    with Image.open(binary_path) as image:
        binary = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("SAM mask has no foreground component")
    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(1 + np.argmax(foreground_areas))
    clean = (labels == largest_label).astype(np.uint8) * 255

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    clean_path = output_dir / "clean_binary.png"
    Image.fromarray(clean, mode="L").save(clean_path)
    paths["clean_binary"] = clean_path

    with Image.open(source_path) as source_image:
        source_rgb = source_image.convert("RGB")
        source_width, source_height = source_rgb.size
        source_scale = max(width / source_width, height / source_height)
        resized_width = max(width, round(source_width * source_scale))
        resized_height = max(height, round(source_height * source_scale))
        source_rgb = source_rgb.resize(
            (resized_width, resized_height), Image.Resampling.LANCZOS
        )
        left = max(0, (resized_width - width) // 2)
        top = max(0, (resized_height - height) // 2)
        source_rgb = source_rgb.crop((left, top, left + width, top + height))
        source_array = np.asarray(source_rgb, dtype=np.float32)

    for radius in (16, 32):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        grown = cv2.dilate(clean, kernel, iterations=1)
        mask_name = f"clean_grow{radius}"
        mask_path = output_dir / f"{mask_name}.png"
        Image.fromarray(grown, mode="L").save(mask_path)
        paths[mask_name] = mask_path

        alpha = (grown.astype(np.float32) / 255.0 * 0.35)[..., None]
        red = np.empty_like(source_array)
        red[:] = (255.0, 32.0, 32.0)
        overlay = np.clip(source_array * (1.0 - alpha) + red * alpha, 0, 255)
        overlay_name = f"clean_overlay{radius}"
        overlay_path = output_dir / f"{overlay_name}.png"
        Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(overlay_path)
        paths[overlay_name] = overlay_path

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=600.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    preflight = {
        "schema": "h3_t8_subject_mask_probe/preflight-v1",
        "created_at": _utc_now(),
        "source": str(source),
        "source_exists": source.is_file(),
        "python": str(args.python.resolve()),
        "python_exists": args.python.is_file(),
        "sam_model": str(SAM_MODEL),
        "sam_model_exists": SAM_MODEL.is_file(),
        "target_port_free": not shared._port_is_listening(args.host, args.port),
        "gpu": shared._gpu_sample(),
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "confirmed": bool(args.confirm_run),
    }
    preflight["ready"] = bool(
        preflight["source_exists"]
        and preflight["python_exists"]
        and preflight["sam_model_exists"]
        and preflight["target_port_free"]
        and preflight["gpu"].get("available")
        and int(preflight["gpu"].get("free_mib", 0)) >= args.minimum_free_vram_mib
        and args.confirm_run
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "preflight.json", preflight)
    if not preflight["ready"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = output_root / run_id
    for name in ("input", "output", "temp", "user", "logs", "prompt"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    input_source = run_root / "input" / source.name
    shutil.copy2(source, input_source)
    prompt = build_prompt(input_source.name, args.width, args.height)
    _write_json(run_root / "prompt" / "prompt.json", prompt)

    monitor = shared.GpuMonitor()
    server_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    history = None
    server_pid = None
    try:
        monitor.start()
        with shared.IsolatedComfy(
            python=args.python.resolve(),
            host=args.host,
            port=args.port,
            run_root=run_root,
            start_timeout=args.server_start_timeout,
        ) as isolated:
            server_pid = int(isolated.process.pid) if isolated.process else None
            object_info = shared._request_json(
                "GET", f"{server_url}/object_info/SAM3_Detect"
            )
            if "SAM3_Detect" not in object_info:
                raise RuntimeError("SAM3_Detect is not registered")
            response = shared._request_json(
                "POST",
                f"{server_url}/prompt",
                {"prompt": prompt, "client_id": f"h3-mask-{run_id}"},
            )
            prompt_id = str(response["prompt_id"])
            history = shared._wait_for_history(
                server_url, prompt_id, args.prompt_timeout
            )
            _write_json(run_root / "history.json", history)
            errors = shared._history_errors(history)
            if errors:
                raise RuntimeError(f"ComfyUI prompt failed: {errors[-1]}")
    finally:
        monitor.stop()

    outputs = sorted((run_root / "output" / "mask_probe").glob("*.png"))
    by_prefix = {path.stem.split("_")[0]: path for path in outputs}
    required = {"raw", "binary", "grow16", "grow32", "overlay16", "overlay32"}
    if not required.issubset(by_prefix):
        raise RuntimeError(f"missing mask probe outputs: {required - set(by_prefix)}")
    masks = {
        name: {
            "path": str(by_prefix[name].resolve()),
            "sha256": _sha256(by_prefix[name]),
            "stats": _mask_stats(by_prefix[name]),
        }
        for name in ("raw", "binary", "grow16", "grow32")
    }
    cleaned = _largest_component_candidates(
        by_prefix["binary"],
        source,
        run_root / "output" / "mask_probe_cleaned",
        width=args.width,
        height=args.height,
    )
    cleaned_masks = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "stats": _mask_stats(path),
        }
        for name, path in cleaned.items()
        if "overlay" not in name
    }
    report = {
        "schema": "h3_t8_subject_mask_probe/v1",
        "created_at": _utc_now(),
        "run_id": run_id,
        "server_pid": server_pid,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "source": {"path": str(source), "sha256": _sha256(source)},
        "canvas": {"width": args.width, "height": args.height},
        "masks": masks,
        "cleaned_largest_component_masks": cleaned_masks,
        "overlays": {
            name: {
                "path": str(by_prefix[name].resolve()),
                "sha256": _sha256(by_prefix[name]),
            }
            for name in ("overlay16", "overlay32")
        },
        "cleaned_overlays": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in cleaned.items()
            if "overlay" in name
        },
        "gpu_monitor": monitor.report(),
        "status": "MASK_CANDIDATES_READY_FOR_VISUAL_REVIEW",
        "boundary": (
            "Single-image SAM3 mask generation only; no MiniMax H3 model, video "
            "sampling, quality claim, or active-mask overwrite."
        ),
    }
    _write_json(run_root / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
