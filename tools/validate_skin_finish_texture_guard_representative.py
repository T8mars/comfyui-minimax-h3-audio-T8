#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_skin_p2_validation"
P0_REPORT = ROOT / "artifacts" / "skin-finish-p0-representative-20260824" / "validation_report.json"
OUTPUT = (
    ROOT
    / "artifacts"
    / "skin-finish-p2-texture-guard-representative-1920w-single-thread-20260824"
)
COMFY_ROOT = ROOT.parents[1]
INSTALL_ROOT = ROOT.parents[2]
FFMPEG = INSTALL_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = INSTALL_ROOT / "ffmpeg" / "bin" / "ffprobe.exe"
MODELS = COMFY_ROOT / "models"

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,nb_frames,r_frame_rate,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def _decode_frames(path: Path) -> torch.Tensor:
    frames = torch.empty((124, 544, 1088, 3), dtype=torch.float32)
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if count >= 124:
                raise RuntimeError("source has more than 124 decoded frames")
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (544, 1088, 3):
                raise RuntimeError(f"unexpected frame shape {array.shape}")
            frames[count].copy_(torch.from_numpy(array).float().div_(255.0))
            count += 1
    if count != 124:
        raise RuntimeError(f"decoded {count} frames instead of 124")
    return frames


def _decoded_pcm(path: Path) -> bytes:
    result = subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _audio_object(pcm: bytes) -> dict:
    interleaved = np.frombuffer(pcm, dtype="<f4").copy().reshape(-1, 2)
    return {
        "waveform": torch.from_numpy(interleaved.T.copy()).unsqueeze(0),
        "sample_rate": 32000,
    }


def _u8(frame: torch.Tensor) -> np.ndarray:
    return (
        frame[..., :3]
        .detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )


def _encode_review(source, raw, guarded, audio_path: Path, output: Path) -> None:
    command = [
        str(FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x320",
        "-r",
        "24",
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-threads",
        "1",
        "-x264-params",
        "threads=1:lookahead_threads=1:sliced_threads=0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for index in range(124):
            panels = [
                Image.fromarray(_u8(value[index])).resize(
                    (640, 320), resample=Image.Resampling.LANCZOS
                )
                for value in (source, raw, guarded)
            ]
            canvas = Image.new("RGB", (1920, 320), "black")
            for panel_index, panel in enumerate(panels):
                canvas.paste(panel, (panel_index * 640, 0))
            draw = ImageDraw.Draw(canvas)
            for x, label in (
                (0, "SOURCE"),
                (640, "RAW SKIN FINISH"),
                (1280, "P2 TEXTURE GUARD"),
            ):
                draw.rectangle((x, 0, x + 260, 24), fill=(0, 0, 0))
                draw.text((x + 6, 5), label, fill=(255, 255, 255))
            process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"review encode failed ({code}): {stderr[-4000:]}")


def _strict_decode(path: Path) -> int:
    result = subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    diagnostics = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode or diagnostics:
        raise RuntimeError(
            f"strict decode failed ({result.returncode}): {diagnostics[-4000:]}"
        )
    video = next(item for item in _probe(path)["streams"] if item["codec_type"] == "video")
    return int(video["nb_frames"])


def _contact(source, raw, guarded, output: Path) -> None:
    indices = [0, 62, 123]
    canvas = Image.new("RGB", (3264, 1632), "black")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        for column, (label, frame) in enumerate(
            (("SOURCE", source[index]), ("RAW", raw[index]), ("GUARDED", guarded[index]))
        ):
            x, y = column * 1088, row * 544
            canvas.paste(Image.fromarray(_u8(frame)), (x, y))
            draw.rectangle((x, y, x + 190, y + 24), fill=(0, 0, 0))
            draw.text((x + 6, y + 5), f"{label} F{index}", fill=(255, 255, 255))
    canvas.save(output, quality=92, subsampling=0)


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty evidence directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not P0_REPORT.is_file() or not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("P0 report and bundled FFmpeg/FFprobe are required")
    p0 = json.loads(P0_REPORT.read_text(encoding="utf-8"))
    source_path = Path(p0["source"]["path"])
    if _sha256(source_path) != p0["source"]["sha256"]:
        raise RuntimeError("P0 representative source SHA no longer matches")

    _load_package()
    import folder_paths

    folder_paths.models_dir = str(MODELS)
    from h3_audio_t8_skin_p2_validation.face_refine_advanced import (
        YUNET_2023MAR_RELATIVE,
        build_face_refine_plan,
    )
    from h3_audio_t8_skin_p2_validation.skin_finish import run_skin_finish
    from h3_audio_t8_skin_p2_validation.skin_finish_p2 import (
        guard_skin_finish_candidate,
    )

    started = time.perf_counter()
    frames = _decode_frames(source_path)
    pcm = _decoded_pcm(source_path)
    audio = _audio_object(pcm)
    plan, crops, preview, _, *_ = build_face_refine_plan(
        frames=frames,
        fps=24.0,
        detector_mode="local_opencv_yunet",
        detector_model=YUNET_2023MAR_RELATIVE,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.10,
        manual_roi_width=0.40,
        manual_roi_height=0.55,
        scene_cut_threshold=0.28,
        max_track_jump=0.18,
        max_gap_frames=4,
        smoothing_radius=2,
        crop_context_scale=3.0,
        canvas_size="384",
        require_h3_grid=True,
        analysis_chunk_frames=4,
    )
    del crops, preview
    gc.collect()
    p0_values = run_skin_finish(
        frames,
        preset="subtle",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        execution_mode="candidate_only",
        chunk_frames=4,
        audio=audio,
        mask_source="face_refine_plan",
        face_plan=plan,
        protect_features=True,
        minimum_mask_area=0.002,
        maximum_mask_area=0.45,
        mask_feather_px=3,
        proxy_long_side=640,
        accept_candidate=False,
    )
    raw, source, _, used, _, _, _, audio_out, p0_report = p0_values
    guard_values = guard_skin_finish_candidate(
        source,
        raw,
        used,
        shadow_protection=0.10,
        highlight_protection=0.94,
        transition_width=0.06,
        minimum_texture_ratio=0.78,
        minimum_reference_texture=0.003,
        maximum_new_clipped_fraction=0.0005,
        clipping_epsilon=1.0 / 255.0,
        texture_radius=1,
        chunk_frames=4,
        accept_candidate=False,
        audio=audio_out,
    )
    guarded, _, selected, guard_audio, effective, rejected, difference, guard_report = guard_values
    parsed_guard = json.loads(guard_report)

    review = OUTPUT / "source_raw_guarded_1088x544x124.mp4"
    contact = OUTPUT / "source_raw_guarded_contact.jpg"
    _encode_review(source, raw, guarded, source_path, review)
    _contact(source, raw, guarded, contact)
    strict_frames = _strict_decode(review)
    pcm_sha = hashlib.sha256(pcm).hexdigest().upper()
    review_pcm_sha = hashlib.sha256(_decoded_pcm(review)).hexdigest().upper()
    outside = effective <= 0.0
    outside_exact = torch.equal(guarded[..., :3][outside], source[..., :3][outside])
    mechanical_pass = all(
        (
            json.loads(p0_report)["status"] == "CANDIDATE_READY",
            parsed_guard["accepted_frame_count"] > 0,
            outside_exact,
            selected is source,
            guard_audio is audio,
            strict_frames == 124,
            pcm_sha == review_pcm_sha,
            bool(torch.isfinite(guarded).all()),
        )
    )
    report = {
        "schema": "t8.minimax_h3.skin_finish.texture_guard_representative/v1",
        "status": "PASS" if mechanical_pass else "FAIL",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "contract": "1088x544x124 at 24fps; 0.591872MP",
        },
        "p0_report": json.loads(p0_report),
        "texture_guard_report": parsed_guard,
        "mechanical_gates": {
            "outside_effective_mask_bit_exact": outside_exact,
            "source_selected_by_default": selected is source,
            "audio_same_python_object": guard_audio is audio,
            "review_decoded_frames": strict_frames,
            "review_strict_single_thread_decode": True,
            "source_review_decoded_pcm_exact": pcm_sha == review_pcm_sha,
            "finite": bool(torch.isfinite(guarded).all()),
            "difference_nonzero": int(torch.count_nonzero(difference)) > 0,
            "effective_mask_nonzero": int(torch.count_nonzero(effective)) > 0,
            "rejected_mask_nonzero": int(torch.count_nonzero(rejected)) > 0,
        },
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "outputs": {
            "review_video": str(review),
            "review_video_sha256": _sha256(review),
            "contact_sheet": str(contact),
            "contact_sheet_sha256": _sha256(contact),
        },
        "human_review_required": True,
        "boundary": (
            "One fixed CPU portrait run proves default mechanical gating only. It does not "
            "prove aesthetic preference, semantic skin parsing, multi-person safety, long-video "
            "continuity, HDR support or universal memory safety."
        ),
    }
    report_path = OUTPUT / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "accepted_frames": parsed_guard["accepted_frame_count"],
                "rejected_frames": parsed_guard["rejected_frame_count"],
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
