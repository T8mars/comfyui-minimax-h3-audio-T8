#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

import av


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_PYTHON = COMFY_ROOT.parent / "python" / "python.exe"
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-live-sam31-validation-20260824"
    / "skin_finish_live_sam31_3person_hardcut_v2_832x736x124.mp4"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-live-sam31-real-run-20260824"
)
SAM_MODEL = COMFY_ROOT / "models" / "checkpoints" / "sam3.1_multiplex_fp16.safetensors"
YUNET_MODEL = COMFY_ROOT / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx"
PARSENET_MODEL = COMFY_ROOT / "models" / "facedetection" / "parsing_parsenet.pth"
SAMPLE_FRAMES = (0, 30, 61, 62, 92, 123)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _port_is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def _gpu_sample() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = completed.stdout.strip().splitlines()[0]
        index, name, total, used, free = [part.strip() for part in first.split(",", 4)]
        return {
            "available": True,
            "index": int(index),
            "name": name,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
        }
    except Exception as error:
        return {"available": False, "detail": str(error)}


class GpuMonitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                sample = _gpu_sample()
                sample["time"] = time.time()
                self.samples.append(sample)
                self._stop.wait(0.5)

        self._thread = threading.Thread(target=loop, name="skin-finish-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def report(self) -> dict[str, Any]:
        valid = [item for item in self.samples if item.get("available")]
        return {
            "sample_count": len(self.samples),
            "peak_used_mib": max((int(item["used_mib"]) for item in valid), default=None),
            "minimum_free_mib": min((int(item["free_mib"]) for item in valid), default=None),
            "first": self.samples[0] if self.samples else None,
            "last": self.samples[-1] if self.samples else None,
        }


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {error.code}: {detail}") from error
    result = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return result


class IsolatedComfy:
    def __init__(
        self,
        *,
        python: Path,
        host: str,
        port: int,
        run_root: Path,
        start_timeout: float,
    ) -> None:
        self.python = python
        self.host = host
        self.port = port
        self.run_root = run_root
        self.start_timeout = start_timeout
        self.process: subprocess.Popen[str] | None = None
        self.stdout = None
        self.stderr = None

    def start(self) -> int:
        if _port_is_listening(self.host, self.port):
            raise RuntimeError(f"refusing to use occupied port {self.port}")
        for name in ("input", "output", "temp", "user", "logs"):
            (self.run_root / name).mkdir(parents=True, exist_ok=True)
        self.stdout = (self.run_root / "logs" / "server.stdout.log").open(
            "w", encoding="utf-8"
        )
        self.stderr = (self.run_root / "logs" / "server.stderr.log").open(
            "w", encoding="utf-8"
        )
        command = [
            str(self.python),
            "main.py",
            "--listen",
            self.host,
            "--port",
            str(self.port),
            "--disable-auto-launch",
            "--preview-method",
            "none",
            "--cache-none",
            "--reserve-vram",
            "1.0",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "minimax-h3-audio-T8",
            "--input-directory",
            str((self.run_root / "input").resolve()),
            "--output-directory",
            str((self.run_root / "output").resolve()),
            "--temp-directory",
            str((self.run_root / "temp").resolve()),
            "--user-directory",
            str((self.run_root / "user").resolve()),
            "--database-url",
            "sqlite:///:memory:",
        ]
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            command,
            cwd=COMFY_ROOT,
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
            env=env,
            creationflags=flags,
        )
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"isolated ComfyUI exited with {self.process.returncode}; inspect "
                    f"{self.run_root / 'logs'}"
                )
            if _port_is_listening(self.host, self.port):
                return int(self.process.pid)
            time.sleep(0.5)
        raise TimeoutError("isolated ComfyUI did not begin listening in time")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        for handle in (self.stdout, self.stderr):
            if handle is not None:
                handle.close()
        deadline = time.monotonic() + 30
        while _port_is_listening(self.host, self.port) and time.monotonic() < deadline:
            time.sleep(0.25)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.stop()


def _sample_preview_nodes(
    prompt: dict[str, Any],
    *,
    first_id: int,
    image_source: list[Any],
) -> int:
    node_id = first_id
    for frame_index in SAMPLE_FRAMES:
        prompt[str(node_id)] = {
            "class_type": "ImageFromBatch",
            "inputs": {
                "image": image_source,
                "batch_index": frame_index,
                "length": 1,
            },
        }
        prompt[str(node_id + 1)] = {
            "class_type": "PreviewImage",
            "inputs": {"images": [str(node_id), 0]},
        }
        node_id += 2
    return node_id


def _build_prompt(source_name: str) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": SAM_MODEL.name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "front-facing person with a visible face",
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
                "maximum_people": 3,
                "detection_threshold": 0.50,
                "detect_interval": 3,
                "scene_cut_threshold": 0.28,
                "analysis_max_side": 640,
                "preview_stride": 8,
                "release_policy": "offload_sam31_after_track",
            },
        },
        "6": {
            "class_type": "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
            "inputs": {
                "frames": ["2", 0],
                "track_plan": ["5", 0],
                "parser_model": "facexlib_parsenet_v0.2.2_pinned",
                "detection_threshold": 0.45,
                "minimum_face_height_px": 32.0,
                "minimum_detail": 0.01,
                "minimum_person_overlap": 0.20,
                "minimum_track_quality": 0.10,
                "minimum_class_probability": 0.55,
                "feature_protection_px": 3,
                "include_neck": False,
                "minimum_skin_area_per_face": 0.00005,
                "maximum_skin_area_per_frame": 0.35,
                "maximum_alignment_rms": 0.08,
                "minimum_ready_frame_fraction": 0.50,
                "preview_count": 6,
            },
        },
        "7": {
            "class_type": "MiniMaxH3SkinFinishAdvancedT8",
            "inputs": {
                "frames": ["2", 0],
                "mask_source": "external_exact",
                "preset": "subtle",
                "amount": 0.35,
                "texture_keep": 0.90,
                "shine_control": 0.35,
                "tone_adjust": 0.0,
                "execution_mode": "candidate_only",
                "accept_candidate": False,
                "protect_features": True,
                "minimum_mask_area": 0.00005,
                "maximum_mask_area": 0.35,
                "mask_feather_px": 2,
                "temporal_mask_radius": 0,
                "proxy_long_side": 640,
                "chunk_frames": 4,
                "skin_mask": ["6", 0],
                "audio": ["2", 1],
            },
        },
        "8": {
            "class_type": "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "inputs": {
                "source_frames": ["7", 1],
                "candidate_frames": ["7", 0],
                "used_skin_mask": ["7", 4],
                "shadow_protection": 0.10,
                "highlight_protection": 0.94,
                "transition_width": 0.06,
                "minimum_texture_ratio": 0.78,
                "minimum_reference_texture": 0.003,
                "maximum_new_clipped_fraction": 0.0005,
                "clipping_epsilon": 1.0 / 255.0,
                "texture_radius": 1,
                "chunk_frames": 4,
                "accept_candidate": False,
                "audio": ["7", 3],
            },
        },
        "9": {
            "class_type": "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            "inputs": {
                "source_video": ["1", 0],
                "processed_frames": ["8", 0],
                "filename_prefix": "MiniMaxH3/SkinFinish/live_sam31_multiface",
                "crf": 18.0,
                "accept_candidate": True,
            },
        },
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}},
        "11": {"class_type": "PreviewImage", "inputs": {"images": ["6", 1]}},
        "12": {"class_type": "MaskToImage", "inputs": {"mask": ["8", 4]}},
        "13": {
            "class_type": "MiniMaxH3SkinFinishPreviewAuditT8Advanced",
            "inputs": {
                "source_frames": ["7", 1],
                "candidate_frames": ["7", 0],
                "used_mask": ["7", 4],
                "rejected_mask": ["7", 5],
                "skin_finish_state": ["7", 7],
                "gate_report_json": ["7", 8],
                "frame_index": 62,
                "comparison_position": 0.50,
                "accept_candidate": False,
                "audio_source": ["2", 1],
                "audio_passthrough": ["7", 3],
            },
        },
        "14": {"class_type": "PreviewImage", "inputs": {"images": ["13", 1]}},
        "15": {"class_type": "PreviewImage", "inputs": {"images": ["13", 4]}},
        "16": {"class_type": "PreviewImage", "inputs": {"images": ["13", 5]}},
        "17": {"class_type": "PreviewImage", "inputs": {"images": ["13", 6]}},
        "18": {"class_type": "PreviewAny", "inputs": {"source": ["5", 2]}},
        "19": {"class_type": "PreviewAny", "inputs": {"source": ["5", 3]}},
    }
    next_id = 20
    for source in (["7", 0], ["8", 0], ["12", 0], ["8", 6]):
        next_id = _sample_preview_nodes(prompt, first_id=next_id, image_source=source)
    return prompt


def _wait_for_history(server: str, prompt_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = _request_json("GET", f"{server}/history/{prompt_id}", timeout=15)
        if prompt_id in payload:
            return payload[prompt_id]
        time.sleep(2.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout_seconds}s")


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,sample_rate,channels,duration:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _strict_decode(path: Path, stream: str) -> dict[str, Any]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-threads",
        "1",
        "-i",
        str(path),
    ]
    if stream == "video":
        command += ["-map", "0:v:0"]
    elif stream == "audio":
        command += ["-map", "0:a:0"]
    command += ["-f", "null", "-"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "diagnostic": completed.stderr[-3000:],
    }


def _decoded_audio_sha(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return {
        "bytes": len(completed.stdout),
        "sha256": hashlib.sha256(completed.stdout).hexdigest().upper(),
    }


def _audio_packet_payload(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(streams) != 1:
            raise RuntimeError(f"expected one audio stream in {path}, found {len(streams)}")
        stream = streams[0]
        for packet in container.demux(stream):
            payload = bytes(packet)
            if not payload:
                continue
            digest.update(payload)
            count += 1
            total += len(payload)
    return {"packet_count": count, "payload_bytes": total, "sha256": digest.hexdigest().upper()}


def _history_errors(history: dict[str, Any]) -> list[dict[str, Any]]:
    status = history.get("status") if isinstance(history, dict) else None
    messages = status.get("messages", []) if isinstance(status, dict) else []
    return [item for item in messages if item and item[0] in {"execution_error", "execution_interrupted"}]


def _history_text(history: dict[str, Any], node_id: str) -> str:
    outputs = history.get("outputs", {})
    node = outputs.get(node_id, {}) if isinstance(outputs, dict) else {}
    values = node.get("text", []) if isinstance(node, dict) else []
    if not values:
        raise RuntimeError(f"history did not retain PreviewAny text for node {node_id}")
    return str(values[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    preflight = {
        "schema": "h3_t8_skin_finish_live_sam31_validation/preflight-v1",
        "created_at": _utc_now(),
        "source": str(source),
        "source_exists": source.is_file(),
        "python": str(args.python.resolve()),
        "python_exists": args.python.is_file(),
        "models": {
            "sam31": {"path": str(SAM_MODEL), "exists": SAM_MODEL.is_file()},
            "yunet": {"path": str(YUNET_MODEL), "exists": YUNET_MODEL.is_file()},
            "parsenet": {"path": str(PARSENET_MODEL), "exists": PARSENET_MODEL.is_file()},
        },
        "target_port_free": not _port_is_listening(args.host, args.port),
        "user_port_8188_observed_only": _port_is_listening(args.host, 8188),
        "gpu": _gpu_sample(),
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "confirmed": bool(args.confirm_run),
    }
    preflight["ready"] = bool(
        preflight["source_exists"]
        and preflight["python_exists"]
        and all(item["exists"] for item in preflight["models"].values())
        and preflight["target_port_free"]
        and preflight["gpu"].get("available")
        and int(preflight["gpu"].get("free_mib", 0)) >= args.minimum_free_vram_mib
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
    prompt = _build_prompt(input_source.name)
    _json_write(run_root / "prompt" / "prompt.json", prompt)

    monitor = GpuMonitor()
    server_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    history: dict[str, Any] | None = None
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
            object_info = _request_json(
                "GET",
                f"{server_url}/object_info/MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
            )
            if "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced" not in object_info:
                raise RuntimeError("Skin Finish multi-person semantic node is not registered")
            response = _request_json(
                "POST",
                f"{server_url}/prompt",
                {"prompt": prompt, "client_id": f"skin-finish-{run_id}"},
            )
            prompt_id = str(response["prompt_id"])
            history = _wait_for_history(server_url, prompt_id, args.prompt_timeout)
            _json_write(run_root / "history.json", history)
            errors = _history_errors(history)
            if errors:
                raise RuntimeError(f"ComfyUI prompt failed: {errors[-1]}")
    finally:
        monitor.stop()

    candidates = sorted((run_root / "output").rglob("*.mp4"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one finalized MP4, found {len(candidates)}: {candidates}")
    candidate = candidates[0]
    assert history is not None
    track_report = json.loads(_history_text(history, "18"))
    shot_count = int(_history_text(history, "19"))
    source_packet = _audio_packet_payload(source)
    candidate_packet = _audio_packet_payload(candidate)
    source_pcm = _decoded_audio_sha(source)
    candidate_pcm = _decoded_audio_sha(candidate)
    preview_files = sorted((run_root / "temp").rglob("*.png"))
    report = {
        "schema": "h3_t8_skin_finish_live_sam31_validation/v1",
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
            "strict_audio": _strict_decode(source, "audio"),
            "audio_packet_payload": source_packet,
            "decoded_audio": source_pcm,
        },
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": _sha256(candidate),
            "probe": _probe(candidate),
            "strict_video": _strict_decode(candidate, "video"),
            "strict_audio": _strict_decode(candidate, "audio"),
            "audio_packet_payload": candidate_packet,
            "decoded_audio": candidate_pcm,
        },
        "checks": {
            "source_candidate_audio_packet_exact": source_packet == candidate_packet,
            "source_candidate_decoded_audio_exact": source_pcm == candidate_pcm,
            "candidate_strict_decode": True,
            "preview_files_present": bool(preview_files),
            "shot_count_matches_report": shot_count == int(track_report["shot_count"]),
            "server_stopped": not _port_is_listening(args.host, args.port),
            "user_8188_untouched": preflight["user_port_8188_observed_only"] == _port_is_listening(args.host, 8188),
        },
        "gpu": monitor.report(),
        "track_report": track_report,
        "shot_count": shot_count,
        "preview_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in preview_files
        ],
        "history_path": str((run_root / "history.json").resolve()),
        "boundary": (
            "One real native SAM3.1/YuNet/ParseNet run on one 0.612MP, 124-frame, "
            "three-person two-segment source. Similar-looking segments may remain one detected "
            "shot; use the separate obvious-cut probe for the shot-reset gate. This is not a "
            "pressure matrix, a universal 16GiB "
            "claim, cross-shot identity proof or human aesthetic acceptance."
        ),
    }
    report["checks"]["candidate_strict_decode"] = bool(
        report["candidate"]["strict_video"]["passed"]
        and report["candidate"]["strict_audio"]["passed"]
    )
    report["passed"] = all(report["checks"].values())
    _json_write(run_root / "validation_report.json", report)
    _json_write(output_root / "latest.json", {"run_id": run_id, "report": str(run_root / "validation_report.json")})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
