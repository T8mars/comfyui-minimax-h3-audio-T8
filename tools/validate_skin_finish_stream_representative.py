#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_skin_stream_validation"
P0_REPORT = ROOT / "artifacts" / "skin-finish-p0-representative-20260824" / "validation_report.json"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "skin-finish-p1-stream-representative-single-thread-20260824"
)
COMFY_ROOT = ROOT.parents[1]
INSTALL_ROOT = ROOT.parents[2]
FFMPEG = INSTALL_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = INSTALL_ROOT / "ffmpeg" / "bin" / "ffprobe.exe"

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


def _strict_decode(path: Path) -> None:
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )


def _decoded_pcm_sha256(path: Path) -> str:
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
    return hashlib.sha256(result.stdout).hexdigest().upper()


def _memory_mib() -> dict:
    try:
        import psutil

        info = psutil.Process().memory_info()._asdict()
        return {key: round(value / 2**20, 3) for key, value in info.items()}
    except Exception as error:
        return {"status": "unavailable", "detail": str(error)}


def _encode_review(source: Path, candidate: Path, output: Path, expected_frames: int) -> None:
    with av.open(str(source)) as source_container, av.open(str(candidate)) as candidate_container:
        source_stream = source_container.streams.video[0]
        width, height = int(source_stream.width), int(source_stream.height)
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
            str(source),
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
        count = 0
        try:
            source_frames = source_container.decode(source_stream)
            candidate_frames = candidate_container.decode(candidate_container.streams.video[0])
            for source_frame, candidate_frame in zip(source_frames, candidate_frames, strict=True):
                left = source_frame.to_ndarray(format="rgb24")
                right = candidate_frame.to_ndarray(format="rgb24")
                combined = np.concatenate((left, right), axis=1)
                image = Image.fromarray(combined)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 130, 28), fill=(0, 0, 0))
                draw.text((8, 7), "SOURCE", fill=(255, 255, 255))
                draw.rectangle((width, 0, width + 300, 28), fill=(0, 0, 0))
                draw.text((width + 8, 7), "TWO-PASS SKIN FINISH", fill=(255, 255, 255))
                process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
                count += 1
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        code = process.wait()
        if code:
            raise RuntimeError(f"review encode failed ({code}): {stderr[-4000:]}")
        if count != expected_frames:
            raise RuntimeError(f"review paired {count} frames, expected {expected_frames}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing validation directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("bundled ffmpeg/ffprobe is required")

    p0_report = json.loads(P0_REPORT.read_text(encoding="utf-8"))
    source_path = (args.source or Path(p0_report["source"]["path"])).resolve()
    expected_source_sha = p0_report["source"]["sha256"]
    if not source_path.is_file() or _sha256(source_path) != expected_source_sha:
        raise RuntimeError("representative source is missing or differs from the P0 evidence")

    _load_package()
    import folder_paths
    from comfy_api.latest import InputImpl
    from h3_audio_t8_skin_stream_validation.skin_finish_p1 import stream_skin_finish_video

    folder_paths.models_dir = str(COMFY_ROOT / "models")
    folder_paths.set_output_directory(str(output_root))
    source_video = InputImpl.VideoFromFile(str(source_path))
    source_probe = _ffprobe(source_path)
    video_stream = next(item for item in source_probe["streams"] if item["codec_type"] == "video")
    if (int(video_stream["width"]), int(video_stream["height"]), int(video_stream["nb_frames"])) != (
        1088,
        544,
        124,
    ):
        raise RuntimeError("representative contract requires exact 1088x544x124 source")

    before_memory = _memory_mib()
    started = time.perf_counter()
    output_video, saved_path, report_json, saved = stream_skin_finish_video(
        source_video,
        preset="subtle",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        detection_threshold=0.35,
        minimum_face_height_px=24.0,
        minimum_detail=0.010,
        bbox_ema_alpha=0.55,
        scene_cut_threshold=0.28,
        maximum_faces=4,
        mask_feather_px=3,
        proxy_long_side=640,
        chunk_frames=4,
        filename_prefix="node-output/stream_candidate_1088x544x124",
        crf=18.0,
        accept_candidate=True,
    )
    elapsed = time.perf_counter() - started
    after_memory = _memory_mib()
    stream_report = json.loads(report_json)
    candidate_path = Path(saved_path).resolve()
    if output_video.get_frame_count() != 124 or saved is None or not candidate_path.is_file():
        raise RuntimeError("stream node did not return the expected file-backed 124-frame VIDEO")

    _strict_decode(source_path)
    _strict_decode(candidate_path)
    candidate_probe = _ffprobe(candidate_path)
    source_pcm_sha = _decoded_pcm_sha256(source_path)
    candidate_pcm_sha = _decoded_pcm_sha256(candidate_path)
    review_path = output_root / "source_vs_two_pass_skin_finish_1088x544x124.mp4"
    _encode_review(source_path, candidate_path, review_path, 124)
    _strict_decode(review_path)
    review_probe = _ffprobe(review_path)

    mechanical_pass = all(
        (
            stream_report["status"] == "CANDIDATE_TWO_PASS_STREAM_FINALIZED",
            stream_report["execution"]["passes"] == 2,
            stream_report["execution"]["full_image_batch_materialized"] is False,
            stream_report["execution"]["peak_chunk_frames"] <= 4,
            stream_report["execution"]["source_proxy_equal_between_passes"] is True,
            stream_report["execution"]["outside_mask_bit_exact_before_encode"] is True,
            stream_report["audio"]["packet_payload_exact"] is True,
            stream_report["source_overwritten"] is False,
            stream_report["atomic_publish"] is True,
            _sha256(source_path) == expected_source_sha,
            source_pcm_sha == candidate_pcm_sha,
            int(next(x for x in candidate_probe["streams"] if x["codec_type"] == "video")["nb_frames"]) == 124,
        )
    )
    report = {
        "schema": "t8.minimax_h3.skin_finish.stream_representative_validation/v1",
        "status": "PASS" if mechanical_pass else "FAIL",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "probe": source_probe,
            "contract": "1088x544x124 at 24fps; 0.591872MP",
        },
        "stream_node_report": stream_report,
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "memory_mib_before": before_memory,
            "memory_mib_after": after_memory,
            "h3_model_loaded": False,
            "gpu_processing_requested": False,
            "stress_or_repeated_run": False,
        },
        "media_gates": {
            "source_strict_decode": True,
            "candidate_strict_decode": True,
            "review_strict_decode": True,
            "source_decoded_pcm_sha256": source_pcm_sha,
            "candidate_decoded_pcm_sha256": candidate_pcm_sha,
            "decoded_pcm_exact": source_pcm_sha == candidate_pcm_sha,
            "candidate_probe": candidate_probe,
            "review_probe": review_probe,
        },
        "outputs": {
            "candidate_video": str(candidate_path),
            "candidate_video_sha256": _sha256(candidate_path),
            "review_video": str(review_path),
            "review_video_sha256": _sha256(review_path),
        },
        "mechanical_pass": mechanical_pass,
        "human_review_required": True,
        "boundary": (
            "One fixed 0.592MP close-face file validates the actual pinned-YuNet two-pass "
            "bounded stream, exact decoded source audio and strict output media contract. "
            "It does not establish visual preference, multi-person/cross-shot quality, long-"
            "video continuity, semantic skin parsing, HDR, arbitrary codecs or universal 16GB safety."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps({"status": report["status"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
