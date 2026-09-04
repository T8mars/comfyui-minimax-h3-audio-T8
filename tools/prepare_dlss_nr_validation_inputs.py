#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import run_dlss_nr_validation as validation_tool
except ImportError:  # pragma: no cover - direct script execution
    import run_dlss_nr_validation as validation_tool


dlss = validation_tool.dlss
SCHEMA = "t8.dlss_nr.validation_inputs.v1"
SPEECH_PHRASE = validation_tool.SPEECH_PHRASE
SPEECH_FRAME_COUNT = 124
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
TARGET_RATE = 24
HARD_CUT_FRAME = SPEECH_FRAME_COUNT // 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    shown = resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved)
    return {
        "path": shown,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _strict_video_contract(path: Path) -> dict[str, Any]:
    video = validation_tool.FileBackedVideo(path)
    resolved, contract = dlss._file_source_contract(video)
    if resolved != path.resolve():
        raise RuntimeError("strict VIDEO contract resolved a different source")
    return contract


def _validate_speech_source(path: Path, phrase_confirmation: str) -> dict[str, Any]:
    if phrase_confirmation.strip() != SPEECH_PHRASE:
        raise ValueError(
            f"operator must confirm that the source clearly says {SPEECH_PHRASE!r}"
        )
    contract = _strict_video_contract(path)
    width = int(contract["width"])
    height = int(contract["height"])
    frame_count = int(contract["frame_count"])
    rate = float(contract["rate"])
    megapixels = width * height / 1_000_000.0
    if frame_count != SPEECH_FRAME_COUNT:
        raise ValueError(
            f"speech source must contain exactly {SPEECH_FRAME_COUNT} frames, got {frame_count}"
        )
    if not 0.40 <= megapixels <= 0.65:
        raise ValueError(
            f"speech source must be approximately 0.5 MP, got {width}x{height} ({megapixels:.6f} MP)"
        )
    if abs(rate - TARGET_RATE) > 1.0e-6:
        raise ValueError(f"speech source must be exactly {TARGET_RATE} fps, got {rate}")
    if not contract["audio_packets"] or not contract["audio_pcm"]:
        raise ValueError("speech source must contain a decodable audio stream")
    return contract


def _run(command: list[str], *, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: {result.stderr[-4000:]}"
        )
    return {
        "returncode": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "stderr_tail": result.stderr[-2000:],
    }


def _copy_or_link(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, target)
        method = "copy"
    if _sha256_file(source) != _sha256_file(target):
        raise RuntimeError("speech source changed while preparing the validation bundle")
    return method


def _extract_frame(source: Path, target: Path, frame_index: int) -> dict[str, Any]:
    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == frame_index:
                rgb = frame.to_ndarray(format="rgb24")
                Image.fromarray(rgb, mode="RGB").save(target, format="PNG", compress_level=6)
                return {
                    "source_frame_index": frame_index,
                    "width": int(frame.width),
                    "height": int(frame.height),
                    "megapixels": round(frame.width * frame.height / 1_000_000.0, 6),
                    "rgb_bridge": "decoded_rgb8_png",
                }
    raise ValueError(f"source does not contain requested frame {frame_index}")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.is_file():
        raise FileNotFoundError(f"fine-texture font does not exist: {path}")
    return ImageFont.truetype(str(path), size=size)


def _write_fine_texture_overlay(path: Path, font_path: Path) -> dict[str, Any]:
    image = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, mode="RGBA")
    draw.rounded_rectangle((42, 28, 626, 132), radius=8, fill=(3, 7, 12, 190))
    draw.text(
        (58, 43),
        "T8 DLSS-NR  0123456789  ABCDEFGHIJKLMN",
        font=_font(font_path, 21),
        fill=(250, 250, 250, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )
    draw.text(
        (58, 82),
        "Fine text: 1px lines / 2px checker / $%&@#",
        font=_font(font_path, 16),
        fill=(225, 235, 246, 255),
    )
    left, top, right, bottom = 660, 28, 932, 132
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(0, 0, 0, 210))
    for x in range(left, right + 1):
        color = (255, 255, 255, 235) if (x - left) % 4 < 2 else (20, 20, 20, 235)
        draw.line((x, top, x, top + 48), fill=color, width=1)
    for y in range(top + 56, bottom + 1):
        for x in range(left, right + 1):
            value = 238 if ((x - left) // 2 + (y - top - 56) // 2) % 2 == 0 else 18
            draw.point((x, y), fill=(value, value, value, 235))
    image.save(path, format="PNG", compress_level=6)
    return {
        "geometry": [TARGET_WIDTH, TARGET_HEIGHT],
        "text": [
            "T8 DLSS-NR  0123456789  ABCDEFGHIJKLMN",
            "Fine text: 1px lines / 2px checker / $%&@#",
        ],
        "patterns": ["alternating_2px_vertical_lines", "2px_checkerboard"],
        "purpose": "deterministic subtitle and fine-detail preservation target",
    }


def _video_encode_options() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(TARGET_RATE),
        "-fps_mode",
        "cfr",
        "-g",
        "48",
        "-bf",
        "0",
        "-threads",
        "1",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-movflags",
        "+faststart",
    ]


def _hard_cut_command(
    ffmpeg: Path, first: Path, second: Path, output: Path
) -> list[str]:
    normalize = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,format=yuv420p"
    )
    graph = (
        f"[0:v]trim=start_frame=0:end_frame={HARD_CUT_FRAME},setpts=PTS-STARTPTS,"
        f"{normalize}[first];"
        f"[1:v]trim=start_frame=0:end_frame={HARD_CUT_FRAME},setpts=PTS-STARTPTS,"
        f"{normalize}[second];"
        "[first][second]concat=n=2:v=1:a=0[outv]"
    )
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-i",
        str(first),
        "-i",
        str(second),
        "-filter_complex",
        graph,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        *_video_encode_options(),
        "-c:a",
        "copy",
        "-frames:v",
        str(SPEECH_FRAME_COUNT),
        str(output),
    ]


def _fine_texture_command(
    ffmpeg: Path, source: Path, overlay: Path, output: Path
) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-i",
        str(source),
        "-loop",
        "1",
        "-framerate",
        str(TARGET_RATE),
        "-i",
        str(overlay),
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[outv]",
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        *_video_encode_options(),
        "-c:a",
        "copy",
        "-frames:v",
        str(SPEECH_FRAME_COUNT),
        str(output),
    ]


def _hard_cut_screen(path: Path) -> dict[str, Any]:
    frames = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            thumbnail = frame.reformat(width=96, height=54, format="rgb24")
            frames.append(thumbnail.to_ndarray().astype(np.float32) / 255.0)
    if len(frames) != SPEECH_FRAME_COUNT:
        raise RuntimeError(
            f"hard-cut source decoded {len(frames)} frames, expected {SPEECH_FRAME_COUNT}"
        )
    deltas = [
        float(np.mean(np.abs(current - previous)))
        for previous, current in zip(frames[:-1], frames[1:], strict=True)
    ]
    strongest_cut_frame = int(np.argmax(deltas)) + 1
    join_delta = deltas[HARD_CUT_FRAME - 1]
    if join_delta < 0.08:
        raise RuntimeError(
            f"prepared hard-cut source is not mechanically strong enough: {join_delta:.8f}"
        )
    if strongest_cut_frame != HARD_CUT_FRAME:
        raise RuntimeError(
            "prepared hard-cut splice is not the strongest transition: "
            f"expected frame {HARD_CUT_FRAME}, got {strongest_cut_frame}"
        )
    return {
        "thumbnail_geometry": [96, 54],
        "expected_cut_frame": HARD_CUT_FRAME,
        "strongest_cut_frame": strongest_cut_frame,
        "cut_mean_absolute_delta": join_delta,
        "minimum_required_delta": 0.08,
        "mechanical_hard_cut": True,
    }


def _input_record(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        **_file_record(path),
        "video": {
            "width": int(contract["width"]),
            "height": int(contract["height"]),
            "frame_count": int(contract["frame_count"]),
            "frame_rate": float(contract["rate"]),
            "audio_stream_count": len(contract["audio_packets"]),
        },
    }


def prepare_validation_inputs(
    *,
    speech_video: Path,
    hard_cut_second_video: Path,
    output_dir: Path,
    ffmpeg: Path,
    font_file: Path,
    speech_phrase_confirmation: str,
) -> dict[str, Any]:
    speech = speech_video.resolve()
    second = hard_cut_second_video.resolve()
    executable = ffmpeg.resolve()
    font = font_file.resolve()
    if not speech.is_file():
        raise FileNotFoundError(speech)
    if not second.is_file():
        raise FileNotFoundError(second)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    speech_contract = _validate_speech_source(speech, speech_phrase_confirmation)
    second_contract = _strict_video_contract(second)
    if int(second_contract["frame_count"]) < HARD_CUT_FRAME:
        raise ValueError(
            f"second hard-cut source needs at least {HARD_CUT_FRAME} frames"
        )

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation input bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        speech_target = staging / "p3_speech_960x544_124f.mp4"
        speech_copy_method = _copy_or_link(speech, speech_target)
        image_target = staging / "p2_representative_frame_061.png"
        image_report = _extract_frame(speech_target, image_target, HARD_CUT_FRAME - 1)

        hard_cut_target = staging / "p3_hard_cut_960x544_124f.mp4"
        hard_cut_execution = _run(
            _hard_cut_command(executable, speech, second, hard_cut_target),
            label="hard-cut source preparation",
        )
        hard_cut_contract = _strict_video_contract(hard_cut_target)
        hard_cut_screen = _hard_cut_screen(hard_cut_target)

        overlay_target = staging / "fine_texture_overlay.png"
        overlay_report = _write_fine_texture_overlay(overlay_target, font)
        fine_target = staging / "p3_fine_texture_960x544_124f.mp4"
        fine_execution = _run(
            _fine_texture_command(executable, speech, overlay_target, fine_target),
            label="fine-texture source preparation",
        )
        fine_contract = _strict_video_contract(fine_target)
        for label, contract in (
            ("hard-cut", hard_cut_contract),
            ("fine-texture", fine_contract),
        ):
            if (
                int(contract["width"]),
                int(contract["height"]),
                int(contract["frame_count"]),
                float(contract["rate"]),
            ) != (TARGET_WIDTH, TARGET_HEIGHT, SPEECH_FRAME_COUNT, float(TARGET_RATE)):
                raise RuntimeError(f"prepared {label} source does not match the fixed contract")

        manifest = {
            "schema": SCHEMA,
            "status": "PREPARED_NOT_DLSS_TESTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gate_effect": {
                "p2_complete": False,
                "p3_complete": False,
                "p4_complete": False,
                "automatic_promotion": False,
            },
            "operator_confirmations": {
                "speech_phrase": SPEECH_PHRASE,
                "speech_phrase_clearly_audible": True,
                "hard_cut_is_intentional": True,
                "fine_texture_overlay_is_intentional": True,
            },
            "original_inputs": {
                "speech": _input_record(speech, speech_contract),
                "hard_cut_second": _input_record(second, second_contract),
                "font": _file_record(font),
                "ffmpeg": _file_record(executable),
            },
            "prepared_inputs": {
                "p2_image": {
                    **_file_record(image_target, relative_to=staging),
                    **image_report,
                },
                "speech": {
                    **_file_record(speech_target, relative_to=staging),
                    "copy_method": speech_copy_method,
                    "byte_identical_to_original": True,
                    "contract": _input_record(speech_target, speech_contract)["video"],
                },
                "hard_cut": {
                    **_file_record(hard_cut_target, relative_to=staging),
                    "contract": _input_record(hard_cut_target, hard_cut_contract)["video"],
                    "splice": {
                        "first_source_frames": [0, HARD_CUT_FRAME - 1],
                        "second_source_frames": [0, HARD_CUT_FRAME - 1],
                    },
                    "screen": hard_cut_screen,
                    "execution": hard_cut_execution,
                },
                "fine_texture": {
                    **_file_record(fine_target, relative_to=staging),
                    "contract": _input_record(fine_target, fine_contract)["video"],
                    "overlay": {
                        **_file_record(overlay_target, relative_to=staging),
                        **overlay_report,
                    },
                    "execution": fine_execution,
                },
            },
            "next_step": {
                "tool": "tools/run_dlss_nr_validation.py",
                "stage": "all",
                "requires_runtime_audit_ready": True,
                "requires_operator_license_acceptance": True,
                "input_manifest": "validation_inputs.json",
                "input_manifest_sha256": "validation_inputs.sha256",
                "arguments": {
                    "image": "p2_representative_frame_061.png",
                    "speech_video": "p3_speech_960x544_124f.mp4",
                    "hard_cut_video": "p3_hard_cut_960x544_124f.mp4",
                    "fine_texture_video": "p3_fine_texture_960x544_124f.mp4",
                    "confirm_speech_phrase": SPEECH_PHRASE,
                    "confirm_hard_cut_source": True,
                    "confirm_fine_texture_source": True,
                },
            },
            "limitations": [
                "No DLSS-NR process was started while preparing this bundle.",
                "The hard-cut clip is a deterministic concat of two separately generated H3 clips.",
                "The fine-detail target is a deterministic synthetic overlay on the speech H3 clip.",
                "Prepared inputs do not satisfy any real DLSS quality or human-review gate.",
            ],
        }
        manifest_path = staging / "validation_inputs.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_sha256 = _sha256_file(manifest_path)
        checksum_path = staging / "validation_inputs.sha256"
        with checksum_path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(f"{manifest_sha256}  validation_inputs.json\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _default_font() -> Path:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    return windows / "Fonts" / "consola.ttf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare hash-bound P2/P3 DLSS-NR validation inputs without running DLSS."
    )
    parser.add_argument("--speech-video", type=Path, required=True)
    parser.add_argument("--hard-cut-second-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg.exe")
    )
    parser.add_argument("--font-file", type=Path, default=_default_font())
    parser.add_argument("--confirm-speech-phrase", default="")
    args = parser.parse_args()
    report = prepare_validation_inputs(
        speech_video=args.speech_video,
        hard_cut_second_video=args.hard_cut_second_video,
        output_dir=args.output_dir,
        ffmpeg=args.ffmpeg,
        font_file=args.font_file,
        speech_phrase_confirmation=args.confirm_speech_phrase,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
