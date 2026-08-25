#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as torch_functional

import build_skin_finish_human_review as human_review
import run_skin_finish_live_sam31_validation as live_base
import validate_skin_finish_representative as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-speaking-material-audit-20260825"
    / "source_speaking_960x544_124f.mp4"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-speaking-validation-20260825"
EXPECTED_DIALOGUE = "你在干嘛呢，我在这里呀，看看效果如何。"


def _strict_decode(path: Path, selector: str) -> None:
    subprocess.run(
        [
            str(base.FFMPEG),
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-map",
            selector,
            "-f",
            "null",
            "NUL",
        ],
        check=True,
    )


def _encode_candidate(
    frames: torch.Tensor,
    audio_source: Path,
    output: Path,
    *,
    fps: int,
) -> dict[str, Any]:
    if int(fps) != 24:
        raise ValueError("the speaking representative packet-copy route requires 24fps")
    import folder_paths
    from comfy_api.latest import InputImpl
    from h3_audio_t8_skin_validation.skin_finish_p1 import (
        finalize_skin_finish_video,
    )

    previous_output = Path(folder_paths.get_output_directory()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.set_output_directory(str(output.parent.resolve()))
        source_video = InputImpl.VideoFromFile(str(audio_source.resolve()))
        _, generated_path, report_json, _ = finalize_skin_finish_video(
            source_video,
            frames,
            filename_prefix=output.stem,
            crf=16.0,
            accept_candidate=True,
        )
    finally:
        folder_paths.set_output_directory(str(previous_output))
    generated = Path(generated_path).resolve()
    if generated != output.resolve():
        os.replace(generated, output.resolve())
    report = json.loads(report_json)
    if report.get("audio", {}).get("packet_payload_exact") is not True:
        raise RuntimeError("Skin Finish finalizer did not preserve every source audio packet")
    return report


def _contact_sheet(
    source: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    output: Path,
) -> None:
    indices = [0, 24, 48, 72, 96, int(source.shape[0]) - 1]
    tile_width, tile_height = 480, 272
    canvas = Image.new("RGB", (tile_width * len(indices), tile_height * 3), "black")
    draw = ImageDraw.Draw(canvas)
    for column, frame_index in enumerate(indices):
        arrays = [base._to_u8(source[frame_index]), base._to_u8(candidate[frame_index])]
        mask_rgb = np.zeros((int(source.shape[1]), int(source.shape[2]), 3), dtype=np.uint8)
        mask_rgb[..., 1] = np.rint(
            mask[frame_index].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
        ).astype(np.uint8)
        arrays.append(mask_rgb)
        for row, (label, array) in enumerate(
            zip(("SOURCE", "CANDIDATE", "USED SKIN MASK"), arrays, strict=True)
        ):
            image = Image.fromarray(array).resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            x, y = column * tile_width, row * tile_height
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + 190, y + 23), fill="black")
            draw.text((x + 5, y + 4), f"{label} F{frame_index}", fill="white")
    canvas.save(output, quality=92, subsampling=0)


def _iou(first: list[float], second: list[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(1.0e-8, first_area + second_area - intersection)


def _ellipse_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> torch.Tensor:
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    return (
        ((xx - float(center_x)) / max(1.0, float(radius_x))).square()
        + ((yy - float(center_y)) / max(1.0, float(radius_y))).square()
        <= 1.0
    )


def _mouth_temporal_diagnostics(
    source: torch.Tensor,
    candidate: torch.Tensor,
    face_plan: dict[str, Any],
    detections: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    frame_count, height, width, _ = source.shape
    source_crops: list[torch.Tensor] = []
    candidate_crops: list[torch.Tensor] = []
    protected_roi_delta: list[float] = []
    protected_roi_peak: list[float] = []
    detected_indices: list[int] = []

    for frame_index in range(frame_count):
        plan_box = face_plan["frames"][frame_index]["source_face_box_xyxy"]
        available = [item for item in detections[frame_index] if len(item.get("landmarks_xy", [])) == 5]
        if not available:
            continue
        detection = max(available, key=lambda item: _iou(item["box"], plan_box))
        if _iou(detection["box"], plan_box) < 0.20:
            continue
        landmarks = np.asarray(detection["landmarks_xy"], dtype=np.float32)
        box = [float(value) for value in detection["box"]]
        face_width = max(1.0, box[2] - box[0])
        face_height = max(1.0, box[3] - box[1])
        mouth_left, mouth_right = landmarks[3], landmarks[4]
        mouth_center = (mouth_left + mouth_right) * 0.5
        mouth_width = max(float(np.linalg.norm(mouth_right - mouth_left)), face_width * 0.12)
        mouth_mask = _ellipse_mask(
            height,
            width,
            float(mouth_center[0]),
            float(mouth_center[1]),
            mouth_width * 0.78,
            face_height * 0.065,
        )
        eye_mask = torch.zeros((height, width), dtype=torch.bool)
        for eye in landmarks[:2]:
            eye_mask |= _ellipse_mask(
                height,
                width,
                float(eye[0]),
                float(eye[1]),
                face_width * 0.050,
                face_height * 0.040,
            )
        protected = mouth_mask | eye_mask
        delta = (candidate[frame_index, ..., :3] - source[frame_index, ..., :3]).abs()
        selected = delta[protected]
        protected_roi_delta.append(float(selected.mean()))
        protected_roi_peak.append(float(selected.max()))

        x1 = max(0, int(np.floor(mouth_center[0] - mouth_width * 1.05)))
        x2 = min(width, int(np.ceil(mouth_center[0] + mouth_width * 1.05)))
        y1 = max(0, int(np.floor(mouth_center[1] - face_height * 0.11)))
        y2 = min(height, int(np.ceil(mouth_center[1] + face_height * 0.11)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        source_crop = source[frame_index, y1:y2, x1:x2, :3].movedim(-1, 0).unsqueeze(0)
        candidate_crop = candidate[frame_index, y1:y2, x1:x2, :3].movedim(-1, 0).unsqueeze(0)
        source_crops.append(
            torch_functional.interpolate(
                source_crop, size=(32, 64), mode="bilinear", align_corners=False
            )[0]
        )
        candidate_crops.append(
            torch_functional.interpolate(
                candidate_crop, size=(32, 64), mode="bilinear", align_corners=False
            )[0]
        )
        detected_indices.append(frame_index)

    source_motion: list[float] = []
    candidate_motion: list[float] = []
    for index in range(1, len(detected_indices)):
        if detected_indices[index] != detected_indices[index - 1] + 1:
            continue
        source_motion.append(float((source_crops[index] - source_crops[index - 1]).abs().mean()))
        candidate_motion.append(
            float((candidate_crops[index] - candidate_crops[index - 1]).abs().mean())
        )
    if len(source_motion) >= 3 and np.std(source_motion) > 1.0e-9 and np.std(candidate_motion) > 1.0e-9:
        motion_correlation = float(np.corrcoef(source_motion, candidate_motion)[0, 1])
    else:
        motion_correlation = None
    return {
        "detected_frame_count": len(detected_indices),
        "detected_frame_fraction": round(len(detected_indices) / max(1, frame_count), 8),
        "protected_eye_mouth_roi_mean_abs_delta_mean": round(
            float(np.mean(protected_roi_delta)) if protected_roi_delta else 0.0, 10
        ),
        "protected_eye_mouth_roi_mean_abs_delta_max": round(
            float(np.max(protected_roi_delta)) if protected_roi_delta else 0.0, 10
        ),
        "protected_eye_mouth_roi_peak_abs_delta": round(
            float(np.max(protected_roi_peak)) if protected_roi_peak else 0.0, 10
        ),
        "mouth_motion_pair_count": len(source_motion),
        "source_mouth_motion_mean": round(float(np.mean(source_motion)), 10)
        if source_motion
        else None,
        "candidate_mouth_motion_mean": round(float(np.mean(candidate_motion)), 10)
        if candidate_motion
        else None,
        "mouth_motion_mean_abs_difference": round(
            float(np.mean(np.abs(np.asarray(source_motion) - np.asarray(candidate_motion)))),
            10,
        )
        if source_motion
        else None,
        "mouth_motion_correlation": round(motion_correlation, 8)
        if motion_correlation is not None
        else None,
        "boundary": (
            "YuNet landmark ROIs and mouth-crop temporal differences are descriptive mechanical "
            "proxies only. They cannot prove phoneme correctness, lip sync, identity or beauty; "
            "the full speaking clip still requires human review."
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-skin-area", type=float, default=0.20)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not 0.20 <= float(args.maximum_skin_area) <= 0.50:
        raise ValueError("maximum_skin_area must stay within the reviewed 0.20..0.50 range")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not base.FFMPEG.is_file() or not base.FFPROBE.is_file():
        raise FileNotFoundError("bundled ffmpeg/ffprobe is required")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if str(base.COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(base.COMFY_ROOT))
    base._load_package()
    import folder_paths

    folder_paths.models_dir = str(base.MODELS)
    from h3_audio_t8_skin_validation.face_refine_advanced import (
        YUNET_2023MAR_RELATIVE,
        _detect_local_opencv_yunet,
        build_face_refine_plan,
    )
    from h3_audio_t8_skin_validation.skin_finish import run_skin_finish
    from h3_audio_t8_skin_validation.skin_finish_frequency import (
        separate_skin_finish_frequencies,
    )
    from h3_audio_t8_skin_validation.skin_finish_p2 import guard_skin_finish_candidate
    from h3_audio_t8_skin_validation.skin_finish_parser import (
        PARSENET_MODEL_NAME,
        run_semantic_skin_mask,
    )
    from h3_audio_t8_skin_validation.skin_finish_safety_audit import (
        audit_skin_finish_candidate,
    )

    started = time.perf_counter()
    probe = base._ffprobe(source_path)
    video = next(item for item in probe["streams"] if item["codec_type"] == "video")
    width = int(video["width"])
    height = int(video["height"])
    frame_count = int(video["nb_frames"])
    fps = int(str(video["r_frame_rate"]).split("/")[0])
    if (width, height, frame_count, fps) != (960, 544, 124, 24):
        raise RuntimeError(
            "speaking validation requires exact 960x544x124 at 24fps normalized source"
        )
    memory_start = base._memory()
    frames = base._decode_frames(source_path, frame_count, width, height)
    source_pcm = base._decoded_pcm(source_path)
    audio = base._audio_object(source_pcm)
    memory_after_decode = base._memory()

    plan, crops, plan_preview, plan_json, *_ = build_face_refine_plan(
        frames=frames,
        fps=24.0,
        detector_mode="local_opencv_yunet",
        detector_model=YUNET_2023MAR_RELATIVE,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.10,
        manual_roi_width=0.40,
        manual_roi_height=0.60,
        scene_cut_threshold=0.28,
        max_track_jump=0.18,
        max_gap_frames=4,
        smoothing_radius=2,
        crop_context_scale=2.5,
        canvas_size="384",
        require_h3_grid=True,
        analysis_chunk_frames=2,
    )
    del crops, plan_preview
    gc.collect()

    semantic_mask, semantic_preview, semantic_json = run_semantic_skin_mask(
        frames=frames,
        face_plan=plan,
        parser_model=PARSENET_MODEL_NAME,
        include_neck=False,
        crop_expansion=1.45,
        minimum_face_weight=0.35,
        minimum_class_probability=0.55,
        feature_protection_px=4,
        minimum_skin_area=0.0005,
        maximum_skin_area=float(args.maximum_skin_area),
        preview_count=6,
    )
    semantic = json.loads(semantic_json)
    del semantic_preview
    gc.collect()

    (
        raw_candidate,
        source,
        selected,
        used,
        skin_rejected,
        skin_difference,
        skin_state,
        audio_out,
        skin_json,
    ) = run_skin_finish(
        frames,
        preset="subtle",
        amount=0.30,
        texture_keep=0.95,
        shine_control=0.25,
        tone_adjust=0.0,
        execution_mode="candidate_only",
        chunk_frames=2,
        mask=semantic_mask,
        audio=audio,
        mask_source="external_exact",
        face_plan=None,
        protect_features=True,
        minimum_mask_area=0.0005,
        maximum_mask_area=float(args.maximum_skin_area),
        mask_feather_px=0,
        temporal_mask_radius=0,
        proxy_long_side=640,
        accept_candidate=False,
    )
    skin = json.loads(skin_json)
    if selected is not frames or audio_out is not audio:
        raise RuntimeError("Skin Finish changed the default source/audio selection contract")
    del skin_rejected, skin_difference, skin_state, semantic_mask
    gc.collect()

    (
        frequency_candidate,
        frequency_source,
        frequency_selected,
        frequency_audio,
        frequency_mask,
        frequency_rejected,
        frequency_difference,
        frequency_json,
    ) = separate_skin_finish_frequencies(
        source,
        raw_candidate,
        used,
        low_frequency_strength=1.0,
        source_detail_gain=1.0,
        separation_radius_percent=1.0,
        maximum_radius_px=32,
        minimum_mask_area=0.0001,
        maximum_mask_area=float(args.maximum_skin_area),
        maximum_new_clipped_fraction=0.0005,
        chunk_frames=2,
        accept_candidate=False,
        audio=audio_out,
    )
    frequency = json.loads(frequency_json)
    if frequency_audio is not audio:
        raise RuntimeError("Frequency Split changed the audio object")
    if frequency_source is not source or frequency_selected is not source:
        raise RuntimeError("Frequency Split changed the default source selection contract")
    del (
        raw_candidate,
        used,
        frequency_source,
        frequency_selected,
        frequency_rejected,
        frequency_difference,
    )
    gc.collect()

    (
        guarded_candidate,
        guard_source,
        guard_selected,
        guard_audio,
        guard_mask,
        guard_rejected,
        guard_difference,
        guard_json,
    ) = guard_skin_finish_candidate(
        source,
        frequency_candidate,
        frequency_mask,
        shadow_protection=0.10,
        highlight_protection=0.94,
        transition_width=0.06,
        minimum_texture_ratio=0.78,
        minimum_reference_texture=0.003,
        maximum_new_clipped_fraction=0.0005,
        texture_radius=1,
        chunk_frames=2,
        accept_candidate=False,
        audio=frequency_audio,
    )
    guard = json.loads(guard_json)
    if guard_audio is not audio:
        raise RuntimeError("Texture Guard changed the audio object")
    if guard_source is not source or guard_selected is not source:
        raise RuntimeError("Texture Guard changed the default source selection contract")
    del (
        frequency_candidate,
        frequency_mask,
        guard_source,
        guard_selected,
        guard_rejected,
        guard_difference,
    )
    gc.collect()

    (
        audit_selected,
        gated_candidate,
        audit_source,
        audit_audio,
        hard_gate_pass,
        failed_frames,
        audit_preview,
        audit_json,
    ) = audit_skin_finish_candidate(
        source,
        guarded_candidate,
        guard_mask,
        audit_scope="mask_only",
        temporal_policy="hard_gate",
        maximum_mean_abs_change=0.08,
        maximum_peak_abs_change=0.30,
        maximum_temporal_effect_jump=0.04,
        minimum_temporal_pixels=64,
        scene_cut_reset_threshold=0.20,
        accept_candidate=False,
        audio_source=audio,
        audio_passthrough=guard_audio,
    )
    audit = json.loads(audit_json)
    if audit_audio is not audio:
        raise RuntimeError("Safety Audit changed the default audio object")
    if audit_selected is not source or audit_source is not source:
        raise RuntimeError("Safety Audit changed the default source selection contract")
    del guarded_candidate, audit_selected, audit_source, audit_preview
    gc.collect()

    detections, detector_report = _detect_local_opencv_yunet(
        frames,
        YUNET_2023MAR_RELATIVE,
        0.35,
        "cpu",
    )
    mouth = _mouth_temporal_diagnostics(frames, gated_candidate, plan, detections)

    candidate_path = output_root / "skin_finish_speaking_candidate_960x544x124.mp4"
    contact_path = output_root / "source_candidate_mask_contact_sheet.jpg"
    finalizer = _encode_candidate(gated_candidate, source_path, candidate_path, fps=24)
    _strict_decode(candidate_path, "0:v:0")
    _strict_decode(candidate_path, "0:a:0")
    candidate_probe = base._ffprobe(candidate_path)
    candidate_pcm = base._decoded_pcm(candidate_path)
    source_packet = live_base._audio_packet_payload(source_path)
    candidate_packet = live_base._audio_packet_payload(candidate_path)
    _contact_sheet(frames, gated_candidate, guard_mask, contact_path)

    review = None
    if hard_gate_pass:
        review = human_review.build_review(
            source_path,
            candidate_path,
            output_root / "blind-review",
        )

    mechanical_pass = all(
        (
            semantic.get("status") == "READY",
            skin.get("status") == "CANDIDATE_READY",
            frequency.get("status") in {"PASS", "PASS_WITH_REJECTED_FRAMES"},
            guard.get("status") in {"PASS", "PASS_WITH_REJECTED_FRAMES"},
            hard_gate_pass,
            failed_frames == 0,
            candidate_pcm == source_pcm,
            source_packet == candidate_packet,
            int(
                next(
                    item
                    for item in candidate_probe["streams"]
                    if item["codec_type"] == "video"
                )["nb_frames"]
            )
            == frame_count,
            mouth["detected_frame_count"] == frame_count,
        )
    )
    report = {
        "schema": "h3_t8_skin_finish_speaking_representative/v1",
        "status": "PASS_MECHANICAL_HUMAN_REVIEW_PENDING" if mechanical_pass else "FAIL",
        "source": {
            "path": str(source_path),
            "sha256": base._sha256(source_path),
            "probe": probe,
            "contract": (
                "Aspect-preserving 1472x832 to 960x542 scale plus one-pixel top/bottom pad; "
                "960x544x124 at 24fps, 5.166667 seconds."
            ),
            "expected_dialogue": EXPECTED_DIALOGUE,
        },
        "execution": {
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_used": False,
            "h3_loaded": False,
            "sam_loaded": False,
            "pressure_or_repeat_test": False,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "maximum_skin_area": float(args.maximum_skin_area),
            "maximum_skin_area_basis": (
                "The default remains 0.20. A larger explicit value is allowed only when "
                "the source is a reviewed extreme close-up and the semantic mask preview "
                "shows no background leakage."
            ),
        },
        "face_plan": json.loads(plan_json),
        "detector_second_pass": detector_report,
        "semantic_mask": semantic,
        "skin_finish": skin,
        "frequency_split": frequency,
        "texture_guard": guard,
        "safety_audit": audit,
        "mouth_and_feature_diagnostics": mouth,
        "audio": {
            "same_python_object_through_chain": audit_audio is audio,
            "decoded_pcm_sha256": hashlib.sha256(source_pcm).hexdigest().upper(),
            "candidate_decoded_pcm_sha256": hashlib.sha256(candidate_pcm).hexdigest().upper(),
            "decoded_pcm_exact": candidate_pcm == source_pcm,
            "source_packet_payload": source_packet,
            "candidate_packet_payload": candidate_packet,
            "packet_payload_exact": source_packet == candidate_packet,
            "finalizer": finalizer,
        },
        "memory_mib": {
            "start": memory_start,
            "after_decode": memory_after_decode,
            "end": base._memory(),
        },
        "outputs": {
            "candidate": str(candidate_path),
            "candidate_sha256": base._sha256(candidate_path),
            "contact_sheet": str(contact_path),
            "contact_sheet_sha256": base._sha256(contact_path),
            "blind_review": review,
        },
        "mechanical_pass": mechanical_pass,
        "human_review_required": True,
        "claim_boundary": (
            "This single clear speaking close-up can close only source/mask/feature/audio and "
            "temporal-proxy mechanical gates. It cannot establish better skin, correct phonemes, "
            "universal lip sync, multi-person safety, long-video continuity or an automatic default."
        ),
    }
    report_path = output_root / "validation_report.json"
    _write_json(report_path, report)
    print(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "review": review,
                "mouth": mouth,
                "audio": report["audio"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
