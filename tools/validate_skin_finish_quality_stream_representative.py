#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
INSTALL_ROOT = ROOT.parents[2]
FFMPEG = INSTALL_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = INSTALL_ROOT / "ffmpeg" / "bin" / "ffprobe.exe"
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-speaking-material-audit-20260825"
    / "source_speaking_960x544_124f.mp4"
)
EXPECTED_SOURCE_SHA256 = (
    "0330B4F36641777024509CA76135638860F52CC1899FB3A4068A5C48F8F4295F"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-quality-stream-probe-20260825"
PACKAGE_NAME = "h3_audio_t8_skin_finish_quality_stream_validation"

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_package() -> None:
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


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True)


def _probe(path: Path) -> dict:
    result = _run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,nb_frames,r_frame_rate,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout.decode("utf-8"))


def _strict_decode(path: Path) -> None:
    _run(
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
        ]
    )


def _pcm_sha256(path: Path) -> str:
    result = _run(
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
        ]
    )
    return hashlib.sha256(result.stdout).hexdigest().upper()


def _memory_mib() -> dict:
    try:
        import psutil

        values = psutil.Process().memory_info()._asdict()
        return {key: round(value / 2**20, 3) for key, value in values.items()}
    except Exception as error:
        return {"status": "unavailable", "detail": str(error)}


def _make_five_frame_source(source: Path, target: Path) -> None:
    duration = 5.0 / 24.0
    _run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(source),
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-af",
            f"atrim=duration={duration:.9f},asetpts=PTS-STARTPTS",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite validation evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("bundled FFmpeg and FFprobe are required")
    if not source.is_file() or _sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned clear speaking source is missing or changed")

    os.environ["PATH"] = str(FFMPEG.parent) + os.pathsep + os.environ.get("PATH", "")
    clip = output_root / "source_speaking_960x544_5f.mp4"
    _make_five_frame_source(source, clip)
    _strict_decode(clip)
    source_probe = _probe(clip)
    video_stream = next(item for item in source_probe["streams"] if item["codec_type"] == "video")
    if (
        int(video_stream["width"]),
        int(video_stream["height"]),
        int(video_stream["nb_frames"]),
        video_stream["r_frame_rate"],
    ) != (960, 544, 5, "24/1"):
        raise RuntimeError("bounded probe source must be exact 960x544x5 at 24fps")

    _load_package()
    import folder_paths
    import torch
    from comfy_api.latest import InputImpl
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_stream_quality import (
        stream_skin_finish_quality_video,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    folder_paths.set_output_directory(str(output_root))
    source_video = InputImpl.VideoFromFile(str(clip))
    before = _memory_mib()
    started = time.perf_counter()
    output_video, saved_path, report_json, saved = stream_skin_finish_quality_video(
        source_video,
        preset="subtle",
        amount=0.30,
        texture_keep=0.95,
        shine_control=0.25,
        detection_threshold=0.35,
        minimum_face_height_px=32.0,
        minimum_detail=0.010,
        bbox_ema_alpha=0.55,
        scene_cut_threshold=0.28,
        maximum_faces=2,
        crop_expansion=1.45,
        minimum_class_probability=0.55,
        feature_protection_px=4,
        mask_feather_px=0,
        proxy_long_side=640,
        low_frequency_strength=1.0,
        source_detail_gain=1.0,
        separation_radius_percent=1.0,
        maximum_radius_px=32,
        shadow_protection=0.10,
        highlight_protection=0.94,
        minimum_texture_ratio=0.78,
        maximum_temporal_effect_jump=0.04,
        chunk_frames=2,
        filename_prefix="node-output/quality_stream_candidate_960x544_5f",
        crf=18.0,
        accept_candidate=True,
    )
    elapsed = time.perf_counter() - started
    after = _memory_mib()
    candidate = Path(saved_path).resolve()
    node_report = json.loads(report_json)
    if saved is None or not candidate.is_file() or output_video.get_frame_count() != 5:
        raise RuntimeError("quality stream did not publish an exact five-frame VIDEO")
    _strict_decode(candidate)
    candidate_probe = _probe(candidate)
    source_pcm = _pcm_sha256(clip)
    candidate_pcm = _pcm_sha256(candidate)
    summary = node_report["quality_pipeline"]["summary"]
    parser_report = node_report["quality_pipeline"]["parser"]
    execution = node_report["execution"]
    expected_chunks = math.ceil(5 / 2)
    mechanical_pass = all(
        (
            node_report["status"]
            in {
                "CANDIDATE_QUALITY_STREAM_FINALIZED",
                "CANDIDATE_QUALITY_STREAM_FINALIZED_WITH_SOURCE_FALLBACKS",
            },
            execution["passes"] == 2,
            execution["peak_chunk_frames"] <= 2,
            execution["chunk_processor_calls"] == expected_chunks,
            execution["full_image_batch_materialized"] is False,
            execution["full_semantic_mask_batch_materialized"] is False,
            execution["full_candidate_image_batch_materialized"] is False,
            execution["source_proxy_equal_between_passes"] is True,
            execution["outside_mask_bit_exact_before_encode"] is True,
            node_report["audio"]["packet_payload_exact"] is True,
            parser_report["loaded"] is True,
            parser_report["released_after_execute"] is True,
            parser_report["persistent_cache"] is False,
            summary["source_frame_count"] == 5,
            summary["peak_chunk_frames"] <= 2,
            summary["chunk_count"] == expected_chunks,
            source_pcm == candidate_pcm,
            int(
                next(
                    item
                    for item in candidate_probe["streams"]
                    if item["codec_type"] == "video"
                )["nb_frames"]
            )
            == 5,
        )
    )
    report = {
        "schema": "t8.minimax_h3.skin_finish.quality_stream_probe/v1",
        "status": "PASS_MECHANICAL" if mechanical_pass else "FAIL",
        "source": {
            "full_path": str(source),
            "full_sha256": _sha256(source),
            "probe_path": str(clip),
            "probe_sha256": _sha256(clip),
            "probe": source_probe,
            "contract": "960x544x5 at 24fps; first 0.208333 seconds of the clear speaking source",
        },
        "node_report": node_report,
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "memory_mib_before": before,
            "memory_mib_after": after,
            "torch_cpu_threads": 2,
            "h3_model_loaded": False,
            "sam_model_loaded": False,
            "cuda_processing_requested": False,
            "stress_or_repeated_run": False,
        },
        "media_gates": {
            "source_strict_decode": True,
            "candidate_strict_decode": True,
            "source_decoded_pcm_sha256": source_pcm,
            "candidate_decoded_pcm_sha256": candidate_pcm,
            "decoded_pcm_exact": source_pcm == candidate_pcm,
            "candidate_probe": candidate_probe,
        },
        "outputs": {
            "candidate_video": str(candidate),
            "candidate_video_sha256": _sha256(candidate),
        },
        "mechanical_pass": mechanical_pass,
        "human_review_required": True,
        "boundary": (
            "A single five-frame CPU-only probe verifies the real pinned ParseNet load/release, "
            "two-frame bounded quality stages, one-frame cross-chunk Safety Audit continuity, "
            "strict H.264 decode and exact source audio packet/PCM preservation. It is not a "
            "visual preference, long-video, stress, repeated-run, crossing-person, HDR or "
            "universal memory-safety result."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["runtime"]["elapsed_seconds"],
                "summary": summary,
                "output": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
