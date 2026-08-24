#!/usr/bin/env python3
"""Run the guarded 0.6MP Creator AV decode/composition comparison."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping

import run_human_face_5s_creator_av_probe as legacy
import run_human_face_0p6mp_clipproj_probe as high


SCHEMA = "t8.minimax_h3.human_face_0p6mp_creator_av_probe.v1"
MIN_FREE_VRAM_MIB = high.MIN_FREE_VRAM_MIB
MIN_OBSERVED_HEADROOM_MIB = high.MIN_OBSERVED_HEADROOM_MIB


def _encode_review_arm_from_file(
    *, frames: Any, audio_path: Path, output_path: Path, ffmpeg: str
) -> None:
    """Avoid the unstable 431 MiB Windows stdin pipe used by the low-res helper."""

    raw_path = output_path.with_suffix(".rgb24.tmp")
    try:
        with raw_path.open("wb") as handle:
            for frame in frames:
                handle.write(frame.tobytes())
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
            f"{high.WIDTH}x{high.HEIGHT}",
            "-r",
            str(legacy.face.FPS),
            "-i",
            str(raw_path),
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
            "-threads",
            "1",
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
            str(legacy.AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=600
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"FFmpeg file-backed review encode failed ({completed.returncode}): {stderr}"
            )
    finally:
        raw_path.unlink(missing_ok=True)


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, check=False, timeout=600)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"FFmpeg PNG-sequence review encode failed ({completed.returncode}): {stderr}"
        )


def _video_encode_args() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-threads",
        "1",
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
        str(legacy.AUDIO_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
    ]


def _prepare_review_media_from_pngs(
    *, run_root: Path, run_id: str, ffmpeg: str, ffprobe: str
) -> dict[str, Any]:
    """Encode the existing PNG sequences without a high-volume rawvideo pipe."""

    media_root = run_root / "output" / "MiniMaxH3_HumanFaceCreator"
    combined_paths = legacy._sorted_pngs(media_root, f"{run_id}_combined")
    source_paths = legacy._sorted_pngs(media_root, f"{run_id}_source")
    if len(combined_paths) != legacy.OUTPUT_FRAMES or len(source_paths) != legacy.SOURCE_FRAMES:
        raise ValueError(
            f"unexpected frame counts: combined={len(combined_paths)}, source={len(source_paths)}"
        )
    with legacy.Image.open(combined_paths[0]) as image:
        combined_size = image.size
    with legacy.Image.open(source_paths[0]) as image:
        source_size = image.size
    if combined_size != (high.WIDTH, high.HEIGHT) or source_size != (
        high.WIDTH,
        high.HEIGHT,
    ):
        raise ValueError(
            f"unexpected PNG geometry: combined={combined_size}, source={source_size}"
        )

    combined_audio_path = legacy.face.base.shared._latest_file(
        media_root, f"{run_id}_combined*.flac"
    )
    source_audio_path = legacy.face.base.shared._latest_file(
        media_root, f"{run_id}_source*.flac"
    )
    combined_audio, combined_rate = legacy.sf.read(
        combined_audio_path, dtype="float32", always_2d=True
    )
    source_audio, source_rate = legacy.sf.read(
        source_audio_path, dtype="float32", always_2d=True
    )
    if (combined_rate, source_rate) != (legacy.AUDIO_RATE, legacy.AUDIO_RATE):
        raise ValueError("unexpected lossless audio sample rate")
    if len(combined_audio) != legacy.OUTPUT_AUDIO_SAMPLES or len(source_audio) != legacy.SOURCE_AUDIO_SAMPLES:
        raise ValueError(
            f"unexpected audio samples: combined={len(combined_audio)}, source={len(source_audio)}"
        )
    separate_audio = legacy.np.concatenate(
        [source_audio, source_audio[legacy.DROP_AUDIO_SAMPLES :]], axis=0
    )
    if len(separate_audio) != legacy.OUTPUT_AUDIO_SAMPLES:
        raise ValueError(f"separate audio accounting changed: {len(separate_audio)}")

    single_audio_path = run_root / "single_decode_combined.flac"
    separate_audio_path = run_root / "separate_decode_composed.flac"
    legacy.sf.write(
        single_audio_path, combined_audio, legacy.AUDIO_RATE, subtype="PCM_24"
    )
    legacy.sf.write(
        separate_audio_path, separate_audio, legacy.AUDIO_RATE, subtype="PCM_24"
    )
    candidate = run_root / "single_decode_candidate.mp4"
    control = run_root / "separate_decode_control.mp4"
    combined_pattern = str(media_root / f"{run_id}_combined_%05d_.png")
    source_pattern = str(media_root / f"{run_id}_source_%05d_.png")
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            str(legacy.face.FPS),
            "-start_number",
            "1",
            "-i",
            combined_pattern,
            "-i",
            str(single_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(legacy.OUTPUT_FRAMES),
            *_video_encode_args(),
            str(candidate),
        ]
    )
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            str(legacy.face.FPS),
            "-start_number",
            "1",
            "-i",
            source_pattern,
            "-framerate",
            str(legacy.face.FPS),
            "-start_number",
            str(legacy.DROP_VIDEO_FRAMES + 1),
            "-i",
            source_pattern,
            "-i",
            str(separate_audio_path),
            "-filter_complex",
            "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a:0",
            "-frames:v",
            str(legacy.OUTPUT_FRAMES),
            *_video_encode_args(),
            str(control),
        ]
    )
    candidate_report = legacy.face.base.shared.media_report(
        candidate, ffmpeg=ffmpeg, ffprobe=ffprobe
    )
    control_report = legacy.face.base.shared.media_report(
        control, ffmpeg=ffmpeg, ffprobe=ffprobe
    )
    return {
        "candidate": candidate_report,
        "control": control_report,
        "candidate_checks": legacy._review_media_checks(candidate_report),
        "control_checks": legacy._review_media_checks(control_report),
        "source_frames": len(source_paths),
        "combined_frames": len(combined_paths),
        "separate_frames": legacy.OUTPUT_FRAMES,
        "source_audio_samples": len(source_audio),
        "combined_audio_samples": len(combined_audio),
        "separate_audio_samples": len(separate_audio),
        "candidate_path": candidate,
        "control_path": control,
        "encoding_route": "direct_png_sequence_libx264",
    }


@contextmanager
def _configured_legacy() -> Iterator[None]:
    face_replacements = {
        "SCHEMA": high.SCHEMA,
        "WIDTH": high.WIDTH,
        "HEIGHT": high.HEIGHT,
        "_contract": high._contract,
        "_media_checks": high._media_checks,
    }
    previous_face = {
        name: getattr(legacy.face, name) for name in face_replacements
    }
    previous_schema = legacy.SCHEMA
    previous_encoder = legacy._encode_review_arm
    previous_preparer = legacy._prepare_review_media
    try:
        for name, value in face_replacements.items():
            setattr(legacy.face, name, value)
        legacy.SCHEMA = SCHEMA
        legacy._encode_review_arm = _encode_review_arm_from_file
        legacy._prepare_review_media = _prepare_review_media_from_pngs
        yield
    finally:
        for name, value in previous_face.items():
            setattr(legacy.face, name, value)
        legacy.SCHEMA = previous_schema
        legacy._encode_review_arm = previous_encoder
        legacy._prepare_review_media = previous_preparer


def build_prompt(*, run_id: str) -> dict[str, Any]:
    with _configured_legacy():
        return legacy.build_prompt(run_id=run_id)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    with _configured_legacy():
        report = legacy.preflight(args)
    report["resolution_acceptance"] = {
        "requested_megapixels": 0.6,
        "actual_megapixels_decimal": high.TARGET_MEGAPIXELS,
        "canvas": [high.WIDTH, high.HEIGHT],
        "multiple_of_32": True,
        "relative_aspect_error": high._contract()["relative_aspect_error"],
    }
    return report


def _finalize_headroom(result: dict[str, Any]) -> dict[str, Any]:
    minimum = int(result.get("gpu_monitor", {}).get("minimum_free_mib", -1))
    accepted = minimum >= MIN_OBSERVED_HEADROOM_MIB
    result.setdefault("checks", {})[
        "observed_minimum_free_vram_at_least_512_mib"
    ] = accepted
    result["headroom_acceptance"] = {
        "minimum_required_mib": MIN_OBSERVED_HEADROOM_MIB,
        "observed_minimum_free_mib": minimum,
        "passed": accepted,
    }
    result["contract"].update(high._contract())
    result["boundary"] = (
        "One fixed SHA-locked 1088x544 human-face latent reused twice. PASS establishes exact "
        "243-frame/10.125-second media mechanics and at least 512 MiB observed headroom for this "
        "one run only; human review remains authoritative."
    )
    if result.get("passed") and not accepted:
        result["passed"] = False
        result["status"] = "FAIL_OBSERVED_MEMORY_HEADROOM_GATE"
    run_root = Path(str(result["run_root"]))
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    with _configured_legacy():
        result = legacy.run_real_probe(args, preflight_report)
    return _finalize_headroom(result)


def recover_existing_run(
    args: argparse.Namespace, run_root: Path
) -> dict[str, Any]:
    """Finish media packaging after the known Windows large-stdin FFmpeg failure."""

    run_root = run_root.resolve()
    report_path = run_root / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    error = report.get("runtime_error") or {}
    if (
        not report.get("phase", {}).get("terminal", {}).get("type")
        == "execution_success"
        or error.get("type") != "RuntimeError"
        or "FFmpeg review encode failed" not in str(error.get("message"))
    ):
        raise ValueError("run is not an eligible completed-model FFmpeg recovery")
    prompt = json.loads((run_root / "prompt.json").read_text(encoding="utf-8"))
    if (
        int(prompt["8"]["inputs"]["width"]),
        int(prompt["8"]["inputs"]["height"]),
        int(prompt["8"]["inputs"]["length"]),
        int(prompt["9"]["inputs"]["steps"]),
    ) != (high.WIDTH, high.HEIGHT, legacy.SOURCE_FRAMES, 8):
        raise ValueError("existing run geometry is not the reviewed 0.6MP contract")
    with _configured_legacy():
        media = legacy._prepare_review_media(
            run_root=run_root,
            run_id=run_root.name,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
        blind = legacy._build_blind_package(
            run_root, media, legacy.face._input_path(args)
        )
    checks = {
        "asset_hashes": all(report.get("hash_checks", {}).values()),
        "one_isolated_process": len(report.get("process_ids", [])) == 1,
        "execution_success": True,
        "source_frames_124": media["source_frames"] == legacy.SOURCE_FRAMES,
        "both_review_arms_243_frames": media["combined_frames"]
        == legacy.OUTPUT_FRAMES
        and media["separate_frames"] == legacy.OUTPUT_FRAMES,
        "both_review_arms_324000_lossless_samples": media[
            "combined_audio_samples"
        ]
        == legacy.OUTPUT_AUDIO_SAMPLES
        and media["separate_audio_samples"] == legacy.OUTPUT_AUDIO_SAMPLES,
        "candidate_media_contract": all(media["candidate_checks"].values()),
        "control_media_contract": all(media["control_checks"].values()),
        "blind_package_created": bool(blind),
    }
    report.update(
        {
            "schema": SCHEMA,
            "status": "PASS" if all(checks.values()) else "FAIL_RECOVERY_MEDIA_CONTRACT",
            "passed": all(checks.values()),
            "media": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in media.items()
            },
            "blind": blind,
            "checks": checks,
            "runtime_error": None,
            "recovery": {
                "model_rerun": False,
                "reason": "replace_unstable_large_windows_ffmpeg_stdin_with_file_backed_rawvideo",
            },
        }
    )
    return _finalize_headroom(report)


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
        / "human-face-0p6mp-creator-av-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument(
        "--min-free-vram-mib", type=int, default=MIN_FREE_VRAM_MIB
    )
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1_200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--recover-run-root", type=Path)
    args = parser.parse_args(argv)
    if args.min_free_vram_mib < MIN_FREE_VRAM_MIB:
        parser.error(
            "--min-free-vram-mib cannot be lower than the reviewed 0.6MP Creator floor "
            f"({MIN_FREE_VRAM_MIB} MiB)"
        )
    args.arm = legacy.ARM
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.recover_run_root is not None:
        result = recover_existing_run(args, args.recover_run_root)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "passed": result.get("passed", False),
                    "run_root": result.get("run_root"),
                    "model_rerun": False,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.get("passed") else 1
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    latest = args.artifact_root / "latest_preflight.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.confirm_run or not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_real_run": report["ready_for_real_run"],
                    "real_run_started": False,
                    "preflight": str(latest.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0 if not args.confirm_run else 2
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result.get("passed", False),
                "run_root": result.get("run_root"),
                "minimum_free_mib": result.get("gpu_monitor", {}).get(
                    "minimum_free_mib"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
