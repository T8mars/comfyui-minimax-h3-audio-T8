#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-oily-lora8-source-20260825"
    / "20260825-094343-e7088277"
    / "output"
    / "MiniMaxH3_SkinFinish"
    / "oily_lora8_speaking_00001_.mp4"
)
EXPECTED_SOURCE_SHA256 = (
    "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-oily-lora8-oil-control-stream-20260825"
)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_skin_finish_human_review as human_review  # noqa: E402
import validate_skin_finish_quality_stream_representative as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded Skin Finish oil_control file-stream validation on the "
            "pinned MiniMax H3 v1.0 eight-step oily speaking source."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required because the CPU ParseNet validation is intentionally non-trivial.",
    )
    args = parser.parse_args()
    if not args.confirm_run:
        raise RuntimeError("refusing to run without --confirm-run")

    source = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite validation evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or common._sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned v1.0 eight-step oily speaking source is missing or changed")
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
    source_contract = (
        int(video_stream["width"]),
        int(video_stream["height"]),
        int(video_stream["nb_frames"]),
        str(video_stream["r_frame_rate"]),
    )
    if source_contract != (960, 544, 124, "24/1"):
        raise RuntimeError(
            "oil-control representative contract must be 960x544x124 at 24fps"
        )

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
    started = time.perf_counter()
    output_video, saved_path, report_json, saved = stream_skin_finish_quality_video(
        source_video,
        preset="oil_control",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        detection_threshold=0.35,
        minimum_face_height_px=32.0,
        minimum_detail=0.010,
        bbox_ema_alpha=0.55,
        scene_cut_threshold=0.28,
        maximum_faces=1,
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
        filename_prefix="node-output/oily_lora8_oil_control_candidate",
        crf=16.0,
        accept_candidate=True,
    )
    elapsed = time.perf_counter() - started
    after = common._memory_mib()

    candidate = Path(saved_path).resolve()
    node_report = json.loads(report_json)
    expected_chunks = math.ceil(124 / 2)
    if saved is None or not candidate.is_file() or output_video.get_frame_count() != 124:
        raise RuntimeError("quality stream did not publish an exact 124-frame VIDEO")
    common._strict_decode(candidate)
    candidate_probe = common._probe(candidate)
    source_pcm = common._pcm_sha256(source)
    candidate_pcm = common._pcm_sha256(candidate)
    summary = node_report["quality_pipeline"]["summary"]
    parser_report = node_report["quality_pipeline"]["parser"]
    execution = node_report["execution"]
    candidate_frames = int(
        next(
            item
            for item in candidate_probe["streams"]
            if item["codec_type"] == "video"
        )["nb_frames"]
    )
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
            summary["source_frame_count"] == 124,
            summary["chunk_count"] == expected_chunks,
            summary["semantic_ready_frame_count"] > 0,
            summary["safety_audit_failed_frame_count"] == 0,
            source_pcm == candidate_pcm,
            candidate_frames == 124,
        )
    )
    if not mechanical_pass:
        raise RuntimeError("oil-control quality stream failed its mechanical gate")

    review = human_review.build_review(
        source, candidate, output_root / "blind-review"
    )
    report = {
        "schema": "h3_t8_skin_finish_oil_control_stream_validation/v1",
        "status": "PASS_MECHANICAL_HUMAN_REVIEW_PENDING",
        "source": {
            "path": str(source),
            "sha256": common._sha256(source),
            "probe": source_probe,
            "generation_contract": {
                "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                "lora_strength": 1.0,
                "sampling_steps": 8,
                "dialogue": "你在干嘛呢，我在这里呀，看看效果如何。",
            },
        },
        "skin_finish_contract": {
            "route": "bounded_quality_video_stream",
            "preset": "oil_control",
            "amount": 0.35,
            "texture_keep": 0.90,
            "shine_control": 0.35,
            "chunk_frames": 2,
            "automatic_accept": False,
            "comparison_requires_human_review": True,
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
            "source_frame_count": 124,
            "candidate_frame_count": candidate_frames,
            "decoded_pcm_exact": source_pcm == candidate_pcm,
            "decoded_pcm_sha256": source_pcm,
            "packet_payload_exact": node_report["audio"]["packet_payload_exact"],
        },
        "outputs": {
            "candidate": str(candidate),
            "candidate_sha256": common._sha256(candidate),
            "blind_review": review,
        },
        "mechanical_pass": True,
        "human_review_required": True,
        "claim_boundary": (
            "This validates one pinned oily speaking source, exact media preservation and "
            "bounded oil_control execution. It does not prove visual preference, identity, "
            "mouth semantics, multi-person behavior, arbitrary duration or universal RAM safety."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "candidate": str(candidate),
                "review": review["review"],
                "review_id": review["review_id"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
