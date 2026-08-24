#!/usr/bin/env python3
"""Run one clear-material Creator AV single-decode versus separate-decode probe.

The graph samples one bright, frame-filling 4B ClipProj T2VA latent at eight NFE, reuses that exact
latent as both 22-frame segments, concatenates it on the native H3 AV clocks, and decodes the
39-frame result once. It also decodes the source latent once. After the isolated ComfyUI process
exits, both review arms are encoded with the same FFmpeg command:

* candidate: the 39 frames/audio from native latent concat followed by one VAE decode;
* control: source frames 0..21 plus source frames 5..21, and source audio plus source audio after
  nine 40-Hz latent steps, representing separate VAE decodes followed by media composition.

Reusing the exact latent removes prompt, seed and independent-segment randomness from the method
comparison. The tool refuses an active 8188 service, a busy private port, missing assets and its
free-VRAM gate. It never touches user-owned services or the stable sampler implementation.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping
import uuid

import numpy as np
from PIL import Image
import soundfile as sf

import build_external_bridge_blind_review as blind_builder
import run_clear_clipproj_triplet_probe as triplet
import run_nfe_resume_real_probe as shared


SCHEMA = "t8.minimax_h3.clear_creator_av_probe.v1"
SOURCE_FRAMES = 22
DROP_VIDEO_FRAMES = 5
OUTPUT_FRAMES = SOURCE_FRAMES * 2 - DROP_VIDEO_FRAMES
AUDIO_RATE = 32_000
SOURCE_AUDIO_SAMPLES = 29_600
DROP_AUDIO_SAMPLES = 9 * AUDIO_RATE // 40
OUTPUT_AUDIO_SAMPLES = SOURCE_AUDIO_SAMPLES * 2 - DROP_AUDIO_SAMPLES
ARM = "clipproj_4b"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(*, run_id: str) -> dict[str, Any]:
    """Build one sampled latent, one native concat and the two required VAE decodes."""

    prompt = deepcopy(triplet.build_prompt(arm=ARM, run_id=run_id))
    prompt.pop("13")
    prompt.pop("14")
    prompt.update(
        {
            "23": {
                "inputs": {
                    "first_segment": ["12", 0],
                    "second_segment": ["12", 0],
                    "output_device": "cpu",
                    "require_identical_metadata": True,
                },
                "class_type": "MiniMaxH3NativeLatentTimelineConcatT8Advanced",
            },
            "24": {
                "inputs": {
                    "av_latent": ["23", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "25": {
                "inputs": {
                    "images": ["24", 0],
                    "filename_prefix": f"MiniMaxH3_ClearCreator/{run_id}_combined",
                },
                "class_type": "SaveImage",
            },
            "26": {
                "inputs": {
                    "av_latent": ["12", 0],
                    "video_vae": ["1", 0],
                    "audio_vae": ["2", 0],
                },
                "class_type": "MiniMaxH3AVDecodeT8",
            },
            "27": {
                "inputs": {
                    "images": ["26", 0],
                    "filename_prefix": f"MiniMaxH3_ClearCreator/{run_id}_source",
                },
                "class_type": "SaveImage",
            },
            "30": {
                "inputs": {
                    "audio": ["24", 1],
                    "filename_prefix": f"MiniMaxH3_ClearCreator/{run_id}_combined",
                },
                "class_type": "SaveAudio",
            },
            "31": {
                "inputs": {
                    "audio": ["26", 1],
                    "filename_prefix": f"MiniMaxH3_ClearCreator/{run_id}_source",
                },
                "class_type": "SaveAudio",
            },
        }
    )
    return prompt


def _sorted_pngs(root: Path, prefix: str) -> list[Path]:
    paths = sorted(path for path in root.glob(f"{prefix}*.png") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no PNG frames matched {prefix!r} under {root}")
    return paths


def _load_frames(paths: list[Path]) -> np.ndarray:
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    result = np.stack(frames)
    if result.shape[1:] != (triplet.HEIGHT, triplet.WIDTH, 3):
        raise ValueError(f"unexpected frame tensor shape: {result.shape}")
    return result


def _encode_review_arm(
    *, frames: np.ndarray, audio_path: Path, output_path: Path, ffmpeg: str
) -> None:
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{triplet.WIDTH}x{triplet.HEIGHT}",
        "-r",
        "24",
        "-i",
        "-",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-frames:v",
        str(len(frames)),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        input=frames.tobytes(),
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg review encode failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )


def _review_media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    """Validate the 39-frame review arm instead of the 22-frame source probe."""

    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_256x256": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == triplet.WIDTH
        and int(video[0].get("height") or 0) == triplet.HEIGHT,
        "decoded_video_exact_frames": int(
            report.get("decoded_video", {}).get("bytes") or 0
        )
        == OUTPUT_FRAMES * triplet.WIDTH * triplet.HEIGHT * 3,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == AUDIO_RATE
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(
            report.get("decoded_audio", {}).get("bytes") or 0
        )
        > 0,
    }


def _prepare_review_media(
    *, run_root: Path, run_id: str, ffmpeg: str, ffprobe: str
) -> dict[str, Any]:
    media_root = run_root / "output" / "MiniMaxH3_ClearCreator"
    combined_frames = _load_frames(_sorted_pngs(media_root, f"{run_id}_combined"))
    source_frames = _load_frames(_sorted_pngs(media_root, f"{run_id}_source"))
    if len(combined_frames) != OUTPUT_FRAMES or len(source_frames) != SOURCE_FRAMES:
        raise ValueError(
            f"unexpected frame counts: combined={len(combined_frames)}, "
            f"source={len(source_frames)}"
        )
    separate_frames = np.concatenate(
        [source_frames, source_frames[DROP_VIDEO_FRAMES:]], axis=0
    )

    combined_audio_path = shared._latest_file(media_root, f"{run_id}_combined*.flac")
    source_audio_path = shared._latest_file(media_root, f"{run_id}_source*.flac")
    combined_audio, combined_rate = sf.read(
        combined_audio_path, dtype="float32", always_2d=True
    )
    source_audio, source_rate = sf.read(
        source_audio_path, dtype="float32", always_2d=True
    )
    if (combined_rate, source_rate) != (AUDIO_RATE, AUDIO_RATE):
        raise ValueError("unexpected lossless audio sample rate")
    if len(combined_audio) != OUTPUT_AUDIO_SAMPLES or len(source_audio) != SOURCE_AUDIO_SAMPLES:
        raise ValueError(
            f"unexpected audio samples: combined={len(combined_audio)}, source={len(source_audio)}"
        )
    separate_audio = np.concatenate(
        [source_audio, source_audio[DROP_AUDIO_SAMPLES:]], axis=0
    )
    separate_audio_path = run_root / "separate_decode_composed.flac"
    sf.write(separate_audio_path, separate_audio, AUDIO_RATE, subtype="PCM_24")

    normalized_combined_audio = run_root / "single_decode_combined.flac"
    sf.write(normalized_combined_audio, combined_audio, AUDIO_RATE, subtype="PCM_24")
    candidate = run_root / "single_decode_candidate.mp4"
    control = run_root / "separate_decode_control.mp4"
    _encode_review_arm(
        frames=combined_frames,
        audio_path=normalized_combined_audio,
        output_path=candidate,
        ffmpeg=ffmpeg,
    )
    _encode_review_arm(
        frames=separate_frames,
        audio_path=separate_audio_path,
        output_path=control,
        ffmpeg=ffmpeg,
    )
    candidate_report = shared.media_report(candidate, ffmpeg=ffmpeg, ffprobe=ffprobe)
    control_report = shared.media_report(control, ffmpeg=ffmpeg, ffprobe=ffprobe)
    candidate_checks = _review_media_checks(candidate_report)
    control_checks = _review_media_checks(control_report)
    return {
        "candidate": candidate_report,
        "control": control_report,
        "candidate_checks": candidate_checks,
        "control_checks": control_checks,
        "source_frames": len(source_frames),
        "combined_frames": len(combined_frames),
        "separate_frames": len(separate_frames),
        "source_audio_samples": len(source_audio),
        "combined_audio_samples": len(combined_audio),
        "separate_audio_samples": len(separate_audio),
        "candidate_path": candidate,
        "control_path": control,
    }


def _build_blind_package(run_root: Path, media: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema": blind_builder.MANIFEST_SCHEMA,
        "review_id": f"creator-clear-av-{run_root.name}",
        "page_title": "MiniMax H3 Creator 清晰素材匿名音画评审",
        "page_intro": (
            "同一份清晰latent重复两次；只比较拼latent后一次VAE解码，与分别VAE解码后再拼媒体。"
            "先选择本组是否可判断，再同步静音观看接缝并完整试听两边。"
        ),
        "export_filename": "creator_clear_av_blind_review.json",
        "analysis_generalization": (
            "This review covers one exact repeated-latent 256x256x39 bright metronome material. "
            "It isolates one VAE decode after native latent concat versus separate VAE decode "
            "and media composition; it cannot establish universal seamlessness or quality."
        ),
        "pairs": [
            {
                "pair_id": "creator-clear-single-vs-separate-decode",
                "label": "清晰重复latent：一次解码 vs 分别解码",
                "task_type": "T2VA AV Review",
                "prompt": triplet.CLEAR_PROMPT,
                "control": str(Path(media["control_path"]).resolve()),
                "candidate": str(Path(media["candidate_path"]).resolve()),
                "control_method": "Separate VAE decode then media composition",
                "candidate_method": "Native latent concat then one VAE decode",
                "reference_images": [],
                "reference_metrics": [],
            }
        ],
    }
    manifest_path = run_root / "blind_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    key = blind_builder.build_package(
        manifest, run_root, run_root / "blind", blind_seed=2608242001
    )
    return {
        "manifest": str(manifest_path),
        "review_page": str((run_root / "blind" / "blind_review.html").resolve()),
        "key": key,
    }


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    paths = triplet._model_paths(args.comfy_root.resolve(), ARM)
    spec = triplet.ARM_ASSETS[ARM]
    hashes = {
        "clip": shared._sha256_file(paths["clip"]),
        "projection": shared._sha256_file(paths["projection"]),
    }
    hash_checks = {
        "clip": hashes["clip"] == str(spec["clip_sha256"]),
        "projection": hashes["projection"] == str(spec["projection_sha256"]),
    }
    if not all(hash_checks.values()):
        return {
            "schema": SCHEMA,
            "status": "ABSTAIN_ASSET_HASH_MISMATCH",
            "passed": False,
            "run_root": str(run_root),
            "hashes": hashes,
            "hash_checks": hash_checks,
        }
    post_hash = triplet.preflight(args)
    if not post_hash["ready_for_real_run"]:
        return {
            "schema": SCHEMA,
            "status": "ABSTAIN_RESOURCE_CHANGED_AFTER_ASSET_HASH",
            "passed": False,
            "run_root": str(run_root),
            "post_hash_preflight": post_hash,
        }

    prompt = build_prompt(run_id=run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_gpu = dict(post_hash["gpu"])
    monitor = triplet.GpuPeakMonitor()
    server = shared.IsolatedServer(args, run_root, "clear_creator_av")
    phase = None
    runtime_error = None
    process_ids: list[int] = []
    started = time.monotonic()
    monitor.start()
    try:
        process_ids.append(server.start())
        phase = asyncio.run(
            shared.submit_prompt(
                server=f"http://{args.host}:{args.port}",
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except Exception as error:
        runtime_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.stop()
        gpu_monitor = monitor.stop()
    final_gpu = shared._wait_gpu_return(int(baseline_gpu.get("used_mib", 0)) + 512)
    execution_success = bool(
        phase and phase.get("terminal", {}).get("type") == "execution_success"
    )
    media = None
    blind = None
    if execution_success:
        try:
            media = _prepare_review_media(
                run_root=run_root,
                run_id=run_id,
                ffmpeg=str(post_hash["ffmpeg"]),
                ffprobe=str(post_hash["ffprobe"]),
            )
            blind = _build_blind_package(run_root, media)
        except Exception as error:
            runtime_error = {"type": type(error).__name__, "message": str(error)}
    checks = {
        "asset_hashes": all(hash_checks.values()),
        "one_isolated_process": len(process_ids) == 1,
        "execution_success": execution_success,
        "source_frames_22": bool(media and media["source_frames"] == SOURCE_FRAMES),
        "both_review_arms_39_frames": bool(
            media
            and media["combined_frames"] == OUTPUT_FRAMES
            and media["separate_frames"] == OUTPUT_FRAMES
        ),
        "both_review_arms_52000_lossless_samples": bool(
            media
            and media["combined_audio_samples"] == OUTPUT_AUDIO_SAMPLES
            and media["separate_audio_samples"] == OUTPUT_AUDIO_SAMPLES
        ),
        "candidate_media_contract": bool(
            media and all(media["candidate_checks"].values())
        ),
        "control_media_contract": bool(media and all(media["control_checks"].values())),
        "blind_package_created": bool(blind),
    }
    passed = all(checks.values()) and runtime_error is None
    result = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
        "passed": passed,
        "run_root": str(run_root),
        "contract": {
            "prompt": triplet.CLEAR_PROMPT,
            "seed": triplet.SEED,
            "source_frames": SOURCE_FRAMES,
            "drop_video_frames": DROP_VIDEO_FRAMES,
            "output_frames": OUTPUT_FRAMES,
            "source_audio_samples": SOURCE_AUDIO_SAMPLES,
            "drop_audio_samples": DROP_AUDIO_SAMPLES,
            "output_audio_samples": OUTPUT_AUDIO_SAMPLES,
            "steps": triplet.STEPS,
            "single_sampled_latent_reused_twice": True,
        },
        "hashes": hashes,
        "preflight": dict(preflight_report),
        "post_hash_preflight": post_hash,
        "process_ids": process_ids,
        "prompt_to_terminal_seconds": round(time.monotonic() - started, 3),
        "phase": phase,
        "gpu_monitor": gpu_monitor,
        "baseline_gpu": baseline_gpu,
        "final_gpu": final_gpu,
        "media": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in (media or {}).items()
        },
        "blind": blind,
        "checks": checks,
        "runtime_error": runtime_error,
        "boundary": (
            "This isolates the fixed repeated-latent decode/composition difference. One clear "
            "pair and one reviewer still cannot establish universal Creator quality or seamlessness."
        ),
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "creator-clear-av-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=13_000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    args.arm = ARM
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = triplet.preflight(args)
    preflight_path = args.artifact_root / "latest_preflight.json"
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_real_run": report["ready_for_real_run"],
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "run_root": result["run_root"],
                "review_page": (result.get("blind") or {}).get("review_page"),
            },
            ensure_ascii=False,
        )
    )
    if str(result["status"]).startswith("ABSTAIN"):
        return 3
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
