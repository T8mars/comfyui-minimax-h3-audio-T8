#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import urllib.error
import urllib.request
import uuid


ROUTES = ("tail3", "time_bias", "rf_restart", "stg", "temporal_detail")


def _json_request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_route_prompt(template: dict, route: str, output_prefix: str) -> dict:
    if route not in ROUTES:
        raise ValueError(f"unsupported route: {route}")
    prompt = copy.deepcopy(template)
    prompt["13"]["inputs"]["filename_prefix"] = output_prefix

    if route == "tail3":
        prompt["14"] = {
            "_meta": {"title": "Three gradual tail detail calls to exact zero"},
            "class_type": "MiniMaxH3AVTailDetailScheduleT8Advanced",
            "inputs": {
                "sigmas": ["8", 2],
                "extra_tail_steps": 3,
                "spacing": "video_sigma_linear",
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "profile": "turbo_standard8",
            },
        }
        prompt["11"]["inputs"]["sigmas"] = ["14", 0]
        prompt["11"]["_meta"]["title"] = "Sample 8+3 H3 joint AV forwards"
    elif route == "time_bias":
        prompt["8"] = {
            "_meta": {"title": "Smooth shared-AV model-time bias; unchanged integrator"},
            "class_type": "MiniMaxH3ModelTimeBiasSamplerT8Advanced",
            "inputs": {
                "model": ["5", 0],
                "av_latent": ["7", 1],
                "steps": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "bias": -0.05,
                "start_progress": 0.70,
                "end_progress": 1.0,
                "bias_domain": "video_sigma",
            },
        }
    elif route == "rf_restart":
        prompt["8"] = {
            "_meta": {"title": "True joint AV rectified-flow restart after base 8 steps"},
            "class_type": "MiniMaxH3RectifiedFlowRestartSamplerT8Advanced",
            "inputs": {
                "model": ["5", 0],
                "av_latent": ["7", 1],
                "steps": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "restart_video_sigma": 0.15,
                "restart_steps": 3,
                "restart_seed": 2608183001,
            },
        }
    elif route == "stg":
        prompt["14"] = {
            "_meta": {"title": "H3 shared-AV skip-block spatio-temporal guidance"},
            "class_type": "MiniMaxH3SpatioTemporalGuidanceT8Advanced",
            "inputs": {
                "model": ["5", 0],
                "scale": 0.60,
                "double_blocks": "25",
                "start_progress": 0.25,
                "end_progress": 0.85,
                "shift_video": 12.0,
                "rescale": 0.0,
            },
        }
        prompt["8"]["inputs"]["model"] = ["14", 0]
    else:
        prompt["14"] = {
            "_meta": {"title": "Motion-gated decoded-frame luma detail"},
            "class_type": "MiniMaxH3TemporalDetailEnhanceT8Advanced",
            "inputs": {
                "frames": ["12", 0],
                "upscale_factor": 1.0,
                "strength": 0.35,
                "blur_radius": 2,
                "blur_sigma": 1.2,
                "motion_threshold": 0.04,
                "temporal_guard": 0.85,
                "frame_chunk_size": 8,
                "maximum_output_megapixels": 2.1,
            },
        }
        prompt["13"]["inputs"]["images"] = ["14", 0]
    return prompt


def _history_outputs(history: dict, prompt_id: str, comfy_root: Path) -> list[dict]:
    record = history.get(prompt_id, history)
    outputs = []
    for node_id, payload in record.get("outputs", {}).items():
        for key in ("gifs", "images", "audio"):
            for item in payload.get(key, []):
                filename = item.get("filename")
                if not filename:
                    continue
                root = comfy_root / ("output" if item.get("type", "output") == "output" else "temp")
                path = root / item.get("subfolder", "") / filename
                outputs.append(
                    {
                        "node_id": str(node_id),
                        "kind": key,
                        "path": str(path.resolve()),
                        "exists": path.is_file(),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": _sha256(path) if path.is_file() else None,
                    }
                )
    return outputs


def run_prompt(
    server: str,
    prompt: dict,
    *,
    comfy_root: Path,
    timeout_seconds: float,
) -> dict:
    queued = _json_request(
        f"{server.rstrip('/')}/prompt",
        {"prompt": prompt, "client_id": str(uuid.uuid4())},
    )
    if queued.get("node_errors"):
        raise RuntimeError(f"ComfyUI node validation failed: {queued['node_errors']}")
    prompt_id = str(queued["prompt_id"])
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        history = _json_request(f"{server.rstrip('/')}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI execution failed: {status}")
            if record.get("outputs"):
                return {
                    "prompt_id": prompt_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "history": history,
                    "outputs": _history_outputs(history, prompt_id, comfy_root),
                }
        time.sleep(2.0)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds} seconds")


def build_blind_review(
    output_dir: Path,
    baseline: Path,
    route_records: list[dict],
    *,
    seed: int,
) -> None:
    sources = [("baseline8", baseline)]
    for record in route_records:
        videos = [
            Path(item["path"])
            for item in record.get("outputs", [])
            if item.get("kind") == "gifs" and Path(item["path"]).suffix.lower() == ".mp4"
        ]
        if not videos:
            raise RuntimeError(f"route {record['route']} produced no MP4")
        sources.append((record["route"], videos[0]))

    order = list(sources)
    random.Random(seed).shuffle(order)
    blind_root = output_dir / "blind"
    media_root = blind_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    public = []
    key = []
    for index, (route, source) in enumerate(order):
        code = chr(ord("A") + index)
        target = media_root / f"{code}.mp4"
        shutil.copy2(source, target)
        digest = _sha256(target)
        public.append({"code": code, "media": f"media/{target.name}"})
        key.append({"code": code, "route": route, "source": str(source.resolve()), "sha256": digest})

    _write_json(blind_root / "blind_key.json", {"rows": key})
    cards = "\n".join(
        f'<section><h2>{item["code"]}</h2><video controls preload="metadata" src="{item["media"]}"></video></section>'
        for item in public
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniMax H3 五条细节路线匿名对比</title><style>
body{{margin:0;background:#0f1115;color:#f3f4f6;font-family:system-ui,sans-serif}}header{{padding:18px 24px;background:#171a21;position:sticky;top:0;z-index:2}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;padding:18px}}section{{background:#181c24;border:1px solid #343b49;border-radius:12px;padding:12px}}video{{width:100%;max-height:480px;background:#000}}h2{{margin:0 0 8px}}@media(max-width:900px){{main{{grid-template-columns:1fr}}}}</style></head>
<body><header><h1>红色汉服夜空高速旋转：匿名六路对比</h1><p>包含原8步基线与五条新路线。请完整看画面并听声音，再记录排序；页面不暴露身份。</p></header><main>{cards}</main></body></html>"""
    (blind_root / "blind_review.html").write_text(html, encoding="utf-8")


def analyze_comparison(
    baseline: Path,
    route_records: list[dict],
    *,
    ffmpeg: str,
) -> list[dict]:
    try:
        from run_h3_motion_quality_matrix import (
            av_duration_contract,
            motion_metrics,
            strict_decode_metrics,
        )
        from run_hybrid_model_matrix import audio_metrics, video_metrics
        from validate_h3_vram import ValidationError
    except ImportError:
        from tools.run_h3_motion_quality_matrix import (  # type: ignore[no-redef]
            av_duration_contract,
            motion_metrics,
            strict_decode_metrics,
        )
        from tools.run_hybrid_model_matrix import (  # type: ignore[no-redef]
            audio_metrics,
            video_metrics,
        )
        from tools.validate_h3_vram import ValidationError  # type: ignore[no-redef]

    sources = [("baseline8", baseline)]
    for record in route_records:
        videos = [
            Path(item["path"])
            for item in record.get("outputs", [])
            if item.get("kind") == "gifs" and Path(item["path"]).suffix.lower() == ".mp4"
        ]
        if not videos:
            raise RuntimeError(f"route {record['route']} produced no MP4")
        sources.append((record["route"], videos[0]))

    comparison = []
    for route, path in sources:
        video = video_metrics(path)
        audio = audio_metrics(path, ffmpeg)
        try:
            strict_decode = strict_decode_metrics(path, ffmpeg)
        except ValidationError as exc:
            strict_decode = {
                "validated": False,
                "diagnostic": str(exc),
                "accepted_for_comparison": route == "baseline8",
                "acceptance_reason": (
                    "User explicitly accepted the upstream baseline's isolated bad frame."
                    if route == "baseline8"
                    else None
                ),
            }
            if route != "baseline8":
                raise
        comparison.append(
            {
                "route": route,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "strict_decode": strict_decode,
                "video": video,
                "motion": motion_metrics(path),
                "audio": audio,
                "av_duration": av_duration_contract(video, audio),
            }
        )
        outcome = "passed" if strict_decode.get("validated") else "accepted known baseline defect"
        print(f"[{route}] strict decode/metrics: {outcome}", flush=True)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the five H3 detail routes on one fixed API prompt.")
    parser.add_argument("template", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--blind-seed", type=int, default=2608183999)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for route in args.routes:
        prefix = f"MiniMaxH3/H3_Detail_Routes_v1/{route}_hanfu_spin_0p7mp"
        prompt = build_route_prompt(template, route, prefix)
        _write_json(args.output_dir / "prompts" / f"{route}.json", prompt)
        run_path = args.output_dir / "runs" / f"{route}.json"
        if run_path.is_file():
            prior = json.loads(run_path.read_text(encoding="utf-8"))
            if prior.get("route") == route and prior.get("outputs") and all(
                item.get("exists") and Path(item["path"]).is_file()
                for item in prior["outputs"]
            ):
                records.append(prior)
                print(f"[{route}] reuse completed run", flush=True)
                continue
        print(f"[{route}] queued", flush=True)
        started_at = datetime.now(timezone.utc).isoformat()
        result = run_prompt(
            args.server,
            prompt,
            comfy_root=args.comfy_root,
            timeout_seconds=args.timeout_seconds,
        )
        record = {
            "schema": "t8.minimax_h3.detail_route_run.v1",
            "route": route,
            "started_at": started_at,
            **result,
        }
        records.append(record)
        _write_json(run_path, record)
        print(f"[{route}] success in {result['elapsed_seconds']:.1f}s", flush=True)

    build_blind_review(
        args.output_dir,
        args.baseline_video,
        records,
        seed=args.blind_seed,
    )
    comparison = analyze_comparison(
        args.baseline_video,
        records,
        ffmpeg=args.ffmpeg,
    )
    _write_json(
        args.output_dir / "summary.json",
        {
            "schema": "t8.minimax_h3.detail_route_matrix.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_video": str(args.baseline_video.resolve()),
            "routes": records,
            "comparison": comparison,
            "blind_review": str((args.output_dir / "blind" / "blind_review.html").resolve()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
