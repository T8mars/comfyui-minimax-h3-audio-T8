#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
PACKAGE_NAME = "h3_audio_t8_skin_validation"
DEFAULT_SOURCE = next(
    (
        ROOT
        / "artifacts"
        / "human-face-0p6mp-clipproj-runtime-v1"
        / "clipproj_4b"
    ).rglob("*_00001-audio.mp4")
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-p0-representative-20260824"
COMFY_ROOT = ROOT.parents[1]
INSTALL_ROOT = ROOT.parents[2]
FFMPEG = INSTALL_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = INSTALL_ROOT / "ffmpeg" / "bin" / "ffprobe.exe"
MODELS = COMFY_ROOT / "models"


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
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _ffprobe(path: Path) -> dict:
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


def _decode_frames(path: Path, expected: int, width: int, height: int) -> torch.Tensor:
    frames = torch.empty((expected, height, width, 3), dtype=torch.float32)
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if count >= expected:
                raise RuntimeError(f"source has more than {expected} decoded frames")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (height, width, 3):
                raise RuntimeError(f"unexpected frame shape {rgb.shape}")
            frames[count].copy_(torch.from_numpy(rgb).float().div_(255.0))
            count += 1
    if count != expected:
        raise RuntimeError(f"decoded {count} frames, expected {expected}")
    return frames


def _decoded_pcm(path: Path) -> bytes:
    result = subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
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
    waveform = torch.from_numpy(interleaved.T.copy()).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 32000}


def _memory() -> dict:
    try:
        import psutil

        info = psutil.Process().memory_info()._asdict()
        return {key: round(value / 2**20, 3) for key, value in info.items()}
    except Exception as error:
        return {"status": "unavailable", "detail": str(error)}


def _to_u8(frame: torch.Tensor) -> np.ndarray:
    return (
        frame[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .numpy()
    )


def _encode_review(source: torch.Tensor, candidate: torch.Tensor, audio_path: Path, output: Path):
    frame_count, height, width, _ = source.shape
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
        f"{width * 2}x{height}",
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
        "2",
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
        for index in range(frame_count):
            combined = np.concatenate(
                [_to_u8(source[index]), _to_u8(candidate[index])], axis=1
            )
            image = Image.fromarray(combined)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 125, 24), fill=(0, 0, 0))
            draw.text((6, 5), "SOURCE", fill=(255, 255, 255))
            draw.rectangle((width, 0, width + 250, 24), fill=(0, 0, 0))
            draw.text((width + 6, 5), "SKIN FINISH CANDIDATE", fill=(255, 255, 255))
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg review encode failed ({code}): {stderr[-4000:]}")


def _contact_sheet(source, candidate, used_mask, output: Path):
    indices = [0, int(source.shape[0]) // 2, int(source.shape[0]) - 1]
    height, width = int(source.shape[1]), int(source.shape[2])
    sheet = Image.new("RGB", (width * 3, height * 3), "black")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        mask = used_mask[index].detach().cpu().clamp(0.0, 1.0).numpy()
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask_rgb[..., 1] = np.rint(mask * 255.0).astype(np.uint8)
        for row, (label, array) in enumerate(
            (
                (f"SOURCE F{index}", _to_u8(source[index])),
                (f"CANDIDATE F{index}", _to_u8(candidate[index])),
                (f"USED MASK F{index}", mask_rgb),
            )
        ):
            x, y = column * width, row * height
            sheet.paste(Image.fromarray(array), (x, y))
            draw.rectangle((x, y, x + 180, y + 24), fill=(0, 0, 0))
            draw.text((x + 6, y + 5), label, fill=(255, 255, 255))
    sheet.save(output, quality=92, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("bundled ffmpeg/ffprobe is required")

    _load_package()
    import folder_paths

    folder_paths.models_dir = str(MODELS)
    from h3_audio_t8_skin_validation.face_refine_advanced import (
        YUNET_2023MAR_RELATIVE,
        build_face_refine_plan,
    )
    from h3_audio_t8_skin_validation.skin_finish import run_skin_finish

    source_probe = _ffprobe(source_path)
    video = next(item for item in source_probe["streams"] if item["codec_type"] == "video")
    if (int(video["width"]), int(video["height"]), int(video["nb_frames"])) != (
        1088,
        544,
        124,
    ):
        raise RuntimeError("representative contract requires exact 1088x544x124 source")

    memory_start = _memory()
    frames = _decode_frames(source_path, 124, 1088, 544)
    source_pcm = _decoded_pcm(source_path)
    audio = _audio_object(source_pcm)
    memory_after_decode = _memory()

    plan_started = time.perf_counter()
    plan, crops, preview, plan_report, *_ = build_face_refine_plan(
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
    plan_seconds = time.perf_counter() - plan_started
    del crops, preview
    gc.collect()
    memory_after_plan = _memory()

    values = run_skin_finish(
        frames,
        preset="subtle",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        tone_adjust=0.0,
        execution_mode="candidate_only",
        chunk_frames=4,
        audio=audio,
        mask_source="face_refine_plan",
        face_plan=plan,
        protect_features=True,
        minimum_mask_area=0.002,
        maximum_mask_area=0.45,
        mask_feather_px=3,
        temporal_mask_radius=0,
        proxy_long_side=640,
        accept_candidate=False,
    )
    candidate, source, selected, used, rejected, difference, state, audio_out, skin_report = values
    skin = json.loads(skin_report)
    memory_after_skin = _memory()

    review_video = output_root / "skin_finish_source_vs_candidate_1088x544x124.mp4"
    contact_sheet = output_root / "skin_finish_source_candidate_mask_contact_sheet.jpg"
    _encode_review(source, candidate, source_path, review_video)
    _contact_sheet(source, candidate, used, contact_sheet)
    review_probe = _ffprobe(review_video)
    review_pcm = _decoded_pcm(review_video)
    source_pcm_sha = hashlib.sha256(source_pcm).hexdigest().upper()
    review_pcm_sha = hashlib.sha256(review_pcm).hexdigest().upper()
    audio_object_exact = audio_out is audio
    selected_is_source = selected is frames
    used_area = used.float().mean(dim=(1, 2))
    mechanical_pass = all(
        (
            skin["status"] == "CANDIDATE_READY",
            skin["mechanical_gates"]["outside_mask_bit_exact"],
            skin["mechanical_gates"]["alpha_or_aux_channels_preserved"],
            skin["mechanical_gates"]["finite"],
            audio_object_exact,
            selected_is_source,
            source_pcm_sha == review_pcm_sha,
            int(next(x for x in review_probe["streams"] if x["codec_type"] == "video")["nb_frames"])
            == 124,
            float(skin["difference"]["mean_abs_rgb"]) > 0.0,
        )
    )
    report = {
        "schema": "t8.minimax_h3.skin_finish.representative_validation/v1",
        "status": "PASS" if mechanical_pass else "FAIL",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "probe": source_probe,
            "contract": "1088x544x124 at 24fps; 0.591872MP",
        },
        "fixed_parameters": skin["parameters"],
        "mask_route": {
            "source": "face_refine_plan",
            "detector": YUNET_2023MAR_RELATIVE,
            "detector_sha256": _sha256(MODELS / YUNET_2023MAR_RELATIVE),
            "plan_seconds": round(plan_seconds, 6),
            "plan": json.loads(plan_report),
            "used_area_fraction_min": round(float(used_area.min()), 8),
            "used_area_fraction_mean": round(float(used_area.mean()), 8),
            "used_area_fraction_max": round(float(used_area.max()), 8),
            "rejected_nonzero": int(torch.count_nonzero(rejected)),
        },
        "skin_finish": skin,
        "audio": {
            "same_python_object": audio_object_exact,
            "source_decoded_pcm_sha256": source_pcm_sha,
            "review_decoded_pcm_sha256": review_pcm_sha,
            "decoded_pcm_exact": source_pcm_sha == review_pcm_sha,
            "sample_rate": 32000,
            "channels": 2,
        },
        "selection": {
            "accept_candidate": False,
            "selected_is_exact_source_object": selected_is_source,
            "automatic_accept": state["automatic_accept"],
        },
        "memory_mib": {
            "start": memory_start,
            "after_decode": memory_after_decode,
            "after_face_plan_cleanup": memory_after_plan,
            "after_skin_finish": memory_after_skin,
        },
        "outputs": {
            "review_video": str(review_video),
            "review_video_sha256": _sha256(review_video),
            "review_probe": review_probe,
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": _sha256(contact_sheet),
        },
        "mechanical_pass": mechanical_pass,
        "human_review_required": True,
        "boundary": (
            "One fixed CPU 1088x544x124 close-face run proves only the local P0 execution, "
            "mask/source/audio gates and bounded-chunk mechanics. It does not prove visual "
            "preference, multi-person safety, long-video continuity, HDR support, universal "
            "16GB safety or that Skin Finish repairs facial structure or blur."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps({"status": report["status"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
