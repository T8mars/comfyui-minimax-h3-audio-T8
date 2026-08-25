#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = Path(
    r"F:\AI-T8-video-onekey\ComfyUI\output\minimax_h3_t8_long_video"
    r"\bg_relmat_unload_u32_qipao_drum_i1_m0_20260809_185358\assembled"
    r"\H3_Unseen_32s_qipao_drum_dance_Interval1_r0008_cosine_bridge.mp4"
)
EXPECTED_SOURCE_SHA256 = (
    "10CE6352F704700A3DBC24CBF19F503D1B6A6B244258FD6B14CCD98DF3D42BA0"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-quality-stream-long-32s-20260825"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_skin_finish_quality_stream_representative as common  # noqa: E402


class _RuntimeMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.peak_working_set_mib = 0.0
        self.samples = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self, *, emit: bool) -> None:
        memory = common._memory_mib()
        working = float(memory.get("wset", memory.get("rss", 0.0)))
        peak = float(memory.get("peak_wset", working))
        self.peak_working_set_mib = max(self.peak_working_set_mib, peak, working)
        self.samples += 1
        if emit:
            print(
                json.dumps(
                    {
                        "monitor": "skin_finish_quality_stream_32s",
                        "working_set_mib": working,
                        "peak_working_set_mib": self.peak_working_set_mib,
                        "samples": self.samples,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def _run(self) -> None:
        self._sample(emit=False)
        while not self.stop_event.wait(45.0):
            self._sample(emit=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        self._sample(emit=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite validation evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or _sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned 32-second H3 source is missing or changed")
    if not common.FFMPEG.is_file() or not common.FFPROBE.is_file():
        raise FileNotFoundError("bundled FFmpeg and FFprobe are required")
    os.environ["PATH"] = (
        str(common.FFMPEG.parent) + os.pathsep + os.environ.get("PATH", "")
    )
    common._strict_decode(source)
    source_probe = common._probe(source)
    video_stream = next(
        item for item in source_probe["streams"] if item["codec_type"] == "video"
    )
    if (
        int(video_stream["width"]),
        int(video_stream["height"]),
        int(video_stream["nb_frames"]),
        video_stream["r_frame_rate"],
    ) != (736, 416, 768, "24/1"):
        raise RuntimeError("long representative contract must be 736x416x768 at 24fps")

    common._load_package()
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
    source_video = InputImpl.VideoFromFile(str(source))
    before = common._memory_mib()
    monitor = _RuntimeMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
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
            filename_prefix="node-output/quality_stream_candidate_qipao_32s",
            crf=18.0,
            accept_candidate=True,
        )
    finally:
        monitor.stop()
    elapsed = time.perf_counter() - started
    after = common._memory_mib()
    candidate = Path(saved_path).resolve()
    node_report = json.loads(report_json)
    if saved is None or not candidate.is_file() or output_video.get_frame_count() != 768:
        raise RuntimeError("quality stream did not publish an exact 768-frame VIDEO")
    common._strict_decode(candidate)
    candidate_probe = common._probe(candidate)
    source_pcm = common._pcm_sha256(source)
    candidate_pcm = common._pcm_sha256(candidate)
    summary = node_report["quality_pipeline"]["summary"]
    parser_report = node_report["quality_pipeline"]["parser"]
    execution = node_report["execution"]
    expected_chunks = math.ceil(768 / 2)
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
            summary["source_frame_count"] == 768,
            summary["peak_chunk_frames"] <= 2,
            summary["chunk_count"] == expected_chunks,
            summary["semantic_ready_frame_count"] > 0,
            source_pcm == candidate_pcm,
            int(
                next(
                    item
                    for item in candidate_probe["streams"]
                    if item["codec_type"] == "video"
                )["nb_frames"]
            )
            == 768,
        )
    )
    report = {
        "schema": "t8.minimax_h3.skin_finish.quality_stream_long_validation/v1",
        "status": "PASS_MECHANICAL_HUMAN_REVIEW_PENDING" if mechanical_pass else "FAIL",
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "probe": source_probe,
            "contract": (
                "736x416x768 at 24fps; 32 seconds; unique assembled H3 qipao fan dance "
                "with close/far face scale, fast turns and fan occlusion"
            ),
        },
        "node_report": node_report,
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "memory_mib_before": before,
            "memory_mib_after": after,
            "observed_peak_working_set_mib": round(
                monitor.peak_working_set_mib, 3
            ),
            "monitor_samples": monitor.samples,
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
            "This is one unique 32-second final-file CPU validation of bounded memory, exact "
            "frame/audio preservation, source-safe fallbacks and cross-chunk audit continuity. "
            "It does not automatically establish better skin, lip/identity truth, multi-person "
            "fairness, HDR/high-bit-depth support, repeated-run stability or universal RAM safety."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path, flush=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["runtime"]["elapsed_seconds"],
                "peak_working_set_mib": report["runtime"][
                    "observed_peak_working_set_mib"
                ],
                "summary": summary,
                "output": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
