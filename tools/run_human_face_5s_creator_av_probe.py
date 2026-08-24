#!/usr/bin/env python3
"""Run one guarded long human-face Creator AV decode/composition comparison.

The probe samples one SHA-locked 512x256x124 I2VA latent at eight NFE and reuses that exact
joint AV latent twice. Native H3 latent concatenation removes the repeated five-frame video
prefix and the corresponding cumulative 24fps/40Hz audio phase, producing 243 frames
(10.125 seconds). The candidate decodes that combined latent once. The control separately
decodes the source latent, then composes the same 243 frames and 324000 lossless audio samples.

This replaces the earlier 39-frame metronome review that the reviewer correctly marked unsure.
It refuses an active user service, a busy private port, asset drift, input-image drift and a
reviewed 13500 MiB free-VRAM floor. It never edits stable sampling code or existing workflows.
"""

from __future__ import annotations

import argparse
import asyncio
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
import run_human_face_5s_clipproj_probe as face


SCHEMA = "t8.minimax_h3.human_face_5s_creator_av_probe.v1"
ARM = "clipproj_4b"
SOURCE_FRAMES = face.FRAME_COUNT
DROP_VIDEO_FRAMES = 5
OUTPUT_FRAMES = SOURCE_FRAMES * 2 - DROP_VIDEO_FRAMES
AUDIO_RATE = 32_000
AUDIO_LATENT_RATE = 40
SOURCE_AUDIO_LATENT_STEPS = round(SOURCE_FRAMES / face.FPS * AUDIO_LATENT_RATE)
OUTPUT_AUDIO_LATENT_STEPS = round(OUTPUT_FRAMES / face.FPS * AUDIO_LATENT_RATE)
SAMPLES_PER_AUDIO_LATENT_STEP = AUDIO_RATE // AUDIO_LATENT_RATE
DROP_AUDIO_LATENT_STEPS = (
    SOURCE_AUDIO_LATENT_STEPS * 2 - OUTPUT_AUDIO_LATENT_STEPS
)
SOURCE_AUDIO_SAMPLES = SOURCE_AUDIO_LATENT_STEPS * SAMPLES_PER_AUDIO_LATENT_STEP
DROP_AUDIO_SAMPLES = DROP_AUDIO_LATENT_STEPS * SAMPLES_PER_AUDIO_LATENT_STEP
OUTPUT_AUDIO_SAMPLES = OUTPUT_AUDIO_LATENT_STEPS * SAMPLES_PER_AUDIO_LATENT_STEP
MIN_FREE_VRAM_MIB = 13_500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(*, run_id: str) -> dict[str, Any]:
    """Build one sample, one native AV concat and two lossless VAE decode branches."""

    prompt = face.build_prompt(arm=ARM, run_id=run_id)
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
                    "filename_prefix": f"MiniMaxH3_HumanFaceCreator/{run_id}_combined",
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
                    "filename_prefix": f"MiniMaxH3_HumanFaceCreator/{run_id}_source",
                },
                "class_type": "SaveImage",
            },
            "30": {
                "inputs": {
                    "audio": ["24", 1],
                    "filename_prefix": f"MiniMaxH3_HumanFaceCreator/{run_id}_combined",
                },
                "class_type": "SaveAudio",
            },
            "31": {
                "inputs": {
                    "audio": ["26", 1],
                    "filename_prefix": f"MiniMaxH3_HumanFaceCreator/{run_id}_source",
                },
                "class_type": "SaveAudio",
            },
        }
    )
    return prompt


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = face.preflight(args)
    report["schema"] = f"{SCHEMA}.preflight"
    report["contract"].update(
        {
            "source_frames": SOURCE_FRAMES,
            "drop_video_frames": DROP_VIDEO_FRAMES,
            "output_frames": OUTPUT_FRAMES,
            "output_duration_seconds": OUTPUT_FRAMES / face.FPS,
            "source_audio_latent_steps": SOURCE_AUDIO_LATENT_STEPS,
            "drop_audio_latent_steps": DROP_AUDIO_LATENT_STEPS,
            "output_audio_latent_steps": OUTPUT_AUDIO_LATENT_STEPS,
            "source_audio_samples": SOURCE_AUDIO_SAMPLES,
            "drop_audio_samples": DROP_AUDIO_SAMPLES,
            "output_audio_samples": OUTPUT_AUDIO_SAMPLES,
            "single_sampled_latent_reused_twice": True,
        }
    )
    return report


def _sorted_pngs(root: Path, prefix: str) -> list[Path]:
    paths = sorted(path for path in root.glob(f"{prefix}*.png") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no PNG frames matched {prefix!r} under {root}")
    return paths


def _load_frames(paths: list[Path]) -> np.ndarray:
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    result = np.stack(frames)
    expected = (face.HEIGHT, face.WIDTH, 3)
    if result.shape[1:] != expected:
        raise ValueError(f"unexpected frame tensor shape: {result.shape}; expected (*,{expected})")
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
        f"{face.WIDTH}x{face.HEIGHT}",
        "-r",
        str(face.FPS),
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
        timeout=300,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"FFmpeg review encode failed ({completed.returncode}): {stderr}")


def _review_media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_512x256": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == face.WIDTH
        and int(video[0].get("height") or 0) == face.HEIGHT,
        "decoded_video_exact_frames": int(report.get("decoded_video", {}).get("bytes") or 0)
        == OUTPUT_FRAMES * face.WIDTH * face.HEIGHT * 3,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == AUDIO_RATE
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(report.get("decoded_audio", {}).get("bytes") or 0) > 0,
    }


def _prepare_review_media(
    *, run_root: Path, run_id: str, ffmpeg: str, ffprobe: str
) -> dict[str, Any]:
    media_root = run_root / "output" / "MiniMaxH3_HumanFaceCreator"
    combined_frames = _load_frames(_sorted_pngs(media_root, f"{run_id}_combined"))
    source_frames = _load_frames(_sorted_pngs(media_root, f"{run_id}_source"))
    if len(combined_frames) != OUTPUT_FRAMES or len(source_frames) != SOURCE_FRAMES:
        raise ValueError(
            f"unexpected frame counts: combined={len(combined_frames)}, source={len(source_frames)}"
        )
    separate_frames = np.concatenate(
        [source_frames, source_frames[DROP_VIDEO_FRAMES:]], axis=0
    )

    combined_audio_path = face.base.shared._latest_file(
        media_root, f"{run_id}_combined*.flac"
    )
    source_audio_path = face.base.shared._latest_file(media_root, f"{run_id}_source*.flac")
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
    if len(separate_audio) != OUTPUT_AUDIO_SAMPLES:
        raise ValueError(f"separate audio accounting changed: {len(separate_audio)}")

    single_audio_path = run_root / "single_decode_combined.flac"
    separate_audio_path = run_root / "separate_decode_composed.flac"
    sf.write(single_audio_path, combined_audio, AUDIO_RATE, subtype="PCM_24")
    sf.write(separate_audio_path, separate_audio, AUDIO_RATE, subtype="PCM_24")
    candidate = run_root / "single_decode_candidate.mp4"
    control = run_root / "separate_decode_control.mp4"
    _encode_review_arm(
        frames=combined_frames,
        audio_path=single_audio_path,
        output_path=candidate,
        ffmpeg=ffmpeg,
    )
    _encode_review_arm(
        frames=separate_frames,
        audio_path=separate_audio_path,
        output_path=control,
        ffmpeg=ffmpeg,
    )
    candidate_report = face.base.shared.media_report(
        candidate, ffmpeg=ffmpeg, ffprobe=ffprobe
    )
    control_report = face.base.shared.media_report(control, ffmpeg=ffmpeg, ffprobe=ffprobe)
    return {
        "candidate": candidate_report,
        "control": control_report,
        "candidate_checks": _review_media_checks(candidate_report),
        "control_checks": _review_media_checks(control_report),
        "source_frames": len(source_frames),
        "combined_frames": len(combined_frames),
        "separate_frames": len(separate_frames),
        "source_audio_samples": len(source_audio),
        "combined_audio_samples": len(combined_audio),
        "separate_audio_samples": len(separate_audio),
        "candidate_path": candidate,
        "control_path": control,
    }


def _build_blind_package(
    run_root: Path, media: Mapping[str, Any], reference_image: Path
) -> dict[str, Any]:
    manifest = {
        "schema": blind_builder.MANIFEST_SCHEMA,
        "review_id": f"creator-human-face-long-{run_root.name}",
        "page_title": "MiniMax H3 Creator 10秒近景人脸音画盲评",
        "page_intro": (
            "按人工反馈重做：同一SHA锁定近景正脸latent重复两次，总长10.125秒。"
            "只比较拼latent后一次VAE解码，与分别VAE解码后再拼媒体。请先判断素材是否可评，"
            "再完整观看接缝前后的人脸、动作和口型，并分别试听中文对白及中点音色变化。"
        ),
        "export_filename": "creator_human_face_long_av_blind_review.json",
        "analysis_generalization": (
            "One fixed SHA-locked close human portrait and one reused 124-frame I2VA joint AV "
            "latent, producing a 243-frame/10.125-second pair. It isolates one VAE decode after "
            "native latent concat versus separate VAE decode plus media composition; it cannot "
            "establish universal seamlessness, identity quality or audio noninferiority."
        ),
        "pairs": [
            {
                "pair_id": "creator-human-face-single-vs-separate-decode",
                "label": "10.125秒近景人脸：一次解码 vs 分别解码",
                "task_type": "I2VA AV Review",
                "prompt": face.PROMPT,
                "control": str(Path(media["control_path"]).resolve()),
                "candidate": str(Path(media["candidate_path"]).resolve()),
                "control_method": "Separate VAE decode then media composition",
                "candidate_method": "Native latent concat then one VAE decode",
                "reference_images": [str(reference_image.resolve())],
                "reference_metrics": ["first_frame", "identity"],
            }
        ],
    }
    manifest_path = run_root / "blind_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    key = blind_builder.build_package(
        manifest, run_root, run_root / "blind", blind_seed=2608245003
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "review_page": str((run_root / "blind" / "blind_review.html").resolve()),
        "key": key,
    }


def _write_report(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    paths = face.base._model_paths(args.comfy_root.resolve(), ARM)
    spec = face.base.ARM_ASSETS[ARM]
    input_path = face._input_path(args)
    hashes = {
        "clip": face.base.shared._sha256_file(paths["clip"]),
        "projection": face.base.shared._sha256_file(paths["projection"]),
        "input_image": face.base.shared._sha256_file(input_path),
    }
    hash_checks = {
        "clip": hashes["clip"] == str(spec["clip_sha256"]),
        "projection": hashes["projection"] == str(spec["projection_sha256"]),
        "input_image": hashes["input_image"] == face.INPUT_IMAGE_SHA256,
    }
    if not all(hash_checks.values()):
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_ASSET_HASH_MISMATCH",
                "passed": False,
                "run_root": str(run_root),
                "hashes": hashes,
                "hash_checks": hash_checks,
            },
        )

    post_hash = preflight(args)
    if not post_hash["ready_for_real_run"]:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_RESOURCE_CHANGED_AFTER_ASSET_HASH",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "post_hash_preflight": post_hash,
                "hashes": hashes,
                "hash_checks": hash_checks,
            },
        )

    prompt = build_prompt(run_id=run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_gpu = dict(post_hash["gpu"])
    server = face.base.shared.IsolatedServer(args, run_root, "human_face_creator_long")
    monitor = face.base.GpuPeakMonitor()
    process_ids: list[int] = []
    phase = None
    runtime_error = None
    started = time.monotonic()
    monitor.start()
    try:
        process_ids.append(server.start())
        phase = asyncio.run(
            face.base.shared.submit_prompt(
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
    final_gpu = face.base.shared._wait_gpu_return(
        int(baseline_gpu.get("used_mib", 0)) + 512
    )
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
            blind = _build_blind_package(run_root, media, input_path)
        except Exception as error:
            runtime_error = {"type": type(error).__name__, "message": str(error)}
    checks = {
        "asset_hashes": all(hash_checks.values()),
        "one_isolated_process": len(process_ids) == 1,
        "execution_success": execution_success,
        "source_frames_124": bool(media and media["source_frames"] == SOURCE_FRAMES),
        "both_review_arms_243_frames": bool(
            media
            and media["combined_frames"] == OUTPUT_FRAMES
            and media["separate_frames"] == OUTPUT_FRAMES
        ),
        "both_review_arms_324000_lossless_samples": bool(
            media
            and media["combined_audio_samples"] == OUTPUT_AUDIO_SAMPLES
            and media["separate_audio_samples"] == OUTPUT_AUDIO_SAMPLES
        ),
        "candidate_media_contract": bool(media and all(media["candidate_checks"].values())),
        "control_media_contract": bool(media and all(media["control_checks"].values())),
        "blind_package_created": bool(blind),
    }
    passed = all(checks.values()) and runtime_error is None
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "run_root": str(run_root),
            "contract": preflight_report["contract"],
            "hashes": hashes,
            "hash_checks": hash_checks,
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
                "One fixed SHA-locked human-face latent reused twice isolates decode/composition "
                "behavior. Mechanical PASS does not establish visual or audio preference; the "
                "human blind review remains required."
            ),
        },
    )


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
        / "human-face-5s-creator-av-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=MIN_FREE_VRAM_MIB)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1_200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if args.min_free_vram_mib < MIN_FREE_VRAM_MIB:
        parser.error(
            f"--min-free-vram-mib cannot be lower than the reviewed Creator floor "
            f"({MIN_FREE_VRAM_MIB} MiB)"
        )
    args.arm = ARM
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
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
                    "ready_for_real_run": False,
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 2
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result.get("passed", False),
                "run_root": result.get("run_root"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
