from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as torch_functional

import comfy.nested_tensor

from .core import align_frame_count, nested_av_parts, split_noise_masks, video_latent_t
from .face_refine_advanced import (
    ANIME_FACE_V14_N_RELATIVE,
    MANUAL_DETECTOR,
    YUNET_2023MAR_RELATIVE,
    _boxes_for_crop_context,
    _crop_chunks,
    _detect_local_anime_onnx_exp,
    _detect_local_opencv_yunet,
    _detect_local_ultralytics,
    _draw_preview,
    _manual_box,
    _scene_ranges,
    _select_track,
    _validate_frames,
    _warp_crop_to_source,
    canonical_json,
    source_proxy_sha256,
)


PARITY_PLAN_SCHEMA = "h3_t8_face_refine_parity_plan/v1"
MANUAL512_RELATIVE_BASELINE_SCHEMA = "h3_t8_face_refine_manual512_relative_baseline/v1"
MANUAL512_RELATIVE_PROFILE = "manual512_relative_author_parity_v2"
PARITY_CANVAS_MODES = (
    "auto_capped_768",
    "auto_no_downscale",
    "manual_384",
    "manual_512",
    "manual_640",
    "manual_768",
)
ACTUAL_DETECTION_STATES = {
    "detected",
    "reacquired",
    "reacquired_unverified",
}


def _gaussian_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or int(window) <= 1:
        return values.copy()
    window = min(int(window), int(values.size))
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window -= 2
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode="reflect")
    x = np.arange(window, dtype=np.float64) - pad
    sigma = max(window / 6.0, 0.5)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _trajectory_jitter(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.abs(np.diff(values)).mean())


def _smooth_face_trajectory(
    boxes: list[list[float]],
    states: list[str],
    shot_ranges: list[tuple[int, int]],
    center_window: int,
    size_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    centers_x = np.asarray([(box[0] + box[2]) * 0.5 for box in boxes], dtype=np.float64)
    centers_y = np.asarray([(box[1] + box[3]) * 0.5 for box in boxes], dtype=np.float64)
    face_widths = np.asarray([max(1.0, box[2] - box[0]) for box in boxes], dtype=np.float64)
    face_heights = np.asarray([max(1.0, box[3] - box[1]) for box in boxes], dtype=np.float64)
    detection = np.asarray([state in ACTUAL_DETECTION_STATES for state in states], dtype=np.float64)

    smooth_x = centers_x.copy()
    smooth_y = centers_y.copy()
    smooth_width = face_widths.copy()
    smooth_height = face_heights.copy()
    smooth_weight = detection.copy()
    weight_window = max(9, int(center_window) // 2)
    for start, end in shot_ranges:
        section = slice(start, end + 1)
        smooth_x[section] = _gaussian_smooth(centers_x[section], center_window)
        smooth_y[section] = _gaussian_smooth(centers_y[section], center_window)
        smooth_width[section] = _gaussian_smooth(face_widths[section], size_window)
        smooth_height[section] = _gaussian_smooth(face_heights[section], size_window)
        smooth_weight[section] = _gaussian_smooth(detection[section], weight_window)
    smooth_weight = np.clip(smooth_weight, 0.0, 1.0)

    metrics = {
        "center_jitter_before_px_per_frame": (
            _trajectory_jitter(centers_x) + _trajectory_jitter(centers_y)
        )
        * 0.5,
        "center_jitter_after_px_per_frame": (
            _trajectory_jitter(smooth_x) + _trajectory_jitter(smooth_y)
        )
        * 0.5,
        "size_jitter_before_px_per_frame": _trajectory_jitter(face_heights),
        "size_jitter_after_px_per_frame": _trajectory_jitter(smooth_height),
        "center_smooth_window": int(center_window),
        "size_smooth_window": int(size_window),
        "weight_smooth_window": int(weight_window),
        "smooth_method": "gaussian_reflect",
    }
    return smooth_x, smooth_y, smooth_width, smooth_height, smooth_weight, metrics


def _fit_square_crop(
    center_x: float,
    center_y: float,
    face_height: float,
    width: int,
    height: int,
    crop_factor: float,
) -> list[float]:
    side = min(float(min(width, height)), max(32.0, float(face_height) * crop_factor))
    left = min(max(0.0, float(center_x) - side * 0.5), float(width) - side)
    top = min(max(0.0, float(center_y) - side * 0.5), float(height) - side)
    return [left, top, left + side, top + side]


def _resolve_parity_canvas(crop_boxes: list[list[float]], canvas_mode: str) -> int:
    if canvas_mode not in PARITY_CANVAS_MODES:
        raise ValueError(f"Unsupported canvas_mode: {canvas_mode}")
    if canvas_mode.startswith("manual_"):
        return int(canvas_mode.rsplit("_", 1)[1])
    largest_crop = max(box[3] - box[1] for box in crop_boxes)
    snapped = max(128, int(math.ceil(largest_crop / 32.0) * 32))
    if canvas_mode == "auto_capped_768":
        return min(snapped, 768)
    return min(snapped, 1344)


def _face_box_from_center(
    center_x: float,
    center_y: float,
    face_width: float,
    face_height: float,
    source_width: int,
    source_height: int,
) -> list[float]:
    left = max(0.0, float(center_x) - float(face_width) * 0.5)
    top = max(0.0, float(center_y) - float(face_height) * 0.5)
    right = min(float(source_width), float(center_x) + float(face_width) * 0.5)
    bottom = min(float(source_height), float(center_y) + float(face_height) * 0.5)
    return [left, top, max(left + 1.0, right), max(top + 1.0, bottom)]


def _upstream_parity_face_box_in_crop(
    face_width: float,
    face_height: float,
    crop_box: list[float],
    canvas: int,
) -> list[float]:
    """Reproduce the author's centred FaceDetailer-style mask rectangle."""

    crop_width = max(1e-6, crop_box[2] - crop_box[0])
    crop_height = max(1e-6, crop_box[3] - crop_box[1])
    face_canvas_width = float(face_width) / crop_width * float(canvas)
    face_canvas_height = float(face_height) / crop_height * float(canvas)
    return [
        float(canvas) * 0.5 - face_canvas_width * 0.5,
        float(canvas) * 0.5 - face_canvas_height * 0.5,
        float(canvas) * 0.5 + face_canvas_width * 0.5,
        float(canvas) * 0.5 + face_canvas_height * 0.5,
    ]


def _validate_parity_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != PARITY_PLAN_SCHEMA:
        raise ValueError(f"face_plan must use {PARITY_PLAN_SCHEMA}")
    source = plan.get("source")
    canvas = plan.get("canvas")
    records = plan.get("frames")
    if not isinstance(source, dict) or not isinstance(canvas, dict) or not isinstance(records, list):
        raise ValueError("face_plan is missing source, canvas, or frame records")
    if len(records) != int(source.get("frame_count", -1)):
        raise ValueError("face_plan frame records do not match source.frame_count")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected = hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
    if plan.get("plan_sha256") != expected:
        raise ValueError("face_plan hash mismatch; the plan may be stale or modified")
    return plan


def build_face_refine_parity_plan(
    frames: torch.Tensor,
    fps: float,
    detector_mode: str,
    detector_model: str,
    detector_device: str,
    confidence: float,
    manual_roi_x: float,
    manual_roi_y: float,
    manual_roi_width: float,
    manual_roi_height: float,
    scene_cut_threshold: float,
    max_track_jump: float,
    max_gap_frames: int,
    center_smooth_window: int,
    size_smooth_window: int,
    crop_factor: float,
    canvas_mode: str,
    require_h3_grid: bool,
    analysis_chunk_frames: int,
):
    frame_count, height, width = _validate_frames(frames)
    if abs(float(fps) - 24.0) > 0.01:
        raise ValueError(f"Face Refine Parity requires exact 24fps input; got {fps:.6g}fps")
    h3_aligned_frame_count = align_frame_count(frame_count)
    h3_alignment_tail_frames = h3_aligned_frame_count - frame_count
    if require_h3_grid and h3_alignment_tail_frames:
        raise ValueError(f"Face Refine Parity requires 17n+5 frames; got {frame_count}")
    if not require_h3_grid and h3_alignment_tail_frames > 1:
        raise ValueError(
            "Face Refine Parity relaxed grid mode only supports a one-frame alignment "
            f"tail; {frame_count} frames would require {h3_alignment_tail_frames}"
        )
    if frame_count > 362:
        raise ValueError("Tensor parity route is capped at 362 frames; split longer clips by shot")
    if int(center_smooth_window) < 1 or int(size_smooth_window) < 1:
        raise ValueError("smoothing windows must be positive")
    if float(crop_factor) < 1.0:
        raise ValueError("crop_factor must be at least 1.0")

    shot_ranges, scene_deltas = _scene_ranges(frames, float(scene_cut_threshold))
    if detector_mode == "manual_static_roi":
        box = _manual_box(
            width,
            height,
            manual_roi_x,
            manual_roi_y,
            manual_roi_width,
            manual_roi_height,
        )
        detections = [[{"box": list(box), "confidence": 1.0}] for _ in range(frame_count)]
        detector_report = {
            "backend": "manual_static_roi",
            "model": MANUAL_DETECTOR,
            "network_download": False,
            "identity_verification": False,
        }
    elif detector_mode == "local_opencv_yunet":
        detections, detector_report = _detect_local_opencv_yunet(
            frames, detector_model, confidence, detector_device
        )
        detector_report["identity_verification"] = False
    elif detector_mode == "local_anime_onnx_exp":
        detections, detector_report = _detect_local_anime_onnx_exp(
            frames, detector_model, confidence, detector_device
        )
        detector_report["identity_verification"] = False
    elif detector_mode == "local_ultralytics":
        # Ultralytics treats ndarray input as OpenCV/BGR.  The audited upstream node
        # explicitly reverses ComfyUI RGB here; keeping RGB shifted crop geometry.
        detections, detector_report = _detect_local_ultralytics(
            frames[..., [2, 1, 0]], detector_model, confidence, detector_device
        )
        detector_report["input_colour_space"] = "bgr_upstream_parity"
        detector_report["identity_verification"] = False
    else:
        raise ValueError(f"Unknown detector_mode: {detector_mode}")

    tracked_boxes, states, _, multi_face_frames = _select_track(
        detections,
        shot_ranges,
        width,
        height,
        float(max_track_jump),
        int(max_gap_frames),
    )
    if not any(box is not None for box in tracked_boxes):
        raise ValueError("No usable face boxes were found")
    fallback = _manual_box(
        width,
        height,
        manual_roi_x,
        manual_roi_y,
        manual_roi_width,
        manual_roi_height,
    )
    context_boxes = _boxes_for_crop_context(tracked_boxes, shot_ranges, fallback)
    smooth_x, smooth_y, smooth_w, smooth_h, smooth_weight, smooth_metrics = (
        _smooth_face_trajectory(
            context_boxes,
            states,
            shot_ranges,
            int(center_smooth_window),
            int(size_smooth_window),
        )
    )
    face_boxes = [
        _face_box_from_center(
            smooth_x[index],
            smooth_y[index],
            smooth_w[index],
            smooth_h[index],
            width,
            height,
        )
        for index in range(frame_count)
    ]
    crop_boxes = [
        _fit_square_crop(
            smooth_x[index],
            smooth_y[index],
            smooth_h[index],
            width,
            height,
            float(crop_factor),
        )
        for index in range(frame_count)
    ]
    canvas = _resolve_parity_canvas(crop_boxes, canvas_mode)
    crops = _crop_chunks(frames, crop_boxes, canvas, int(analysis_chunk_frames))
    preview, preview_indices = _draw_preview(frames, crop_boxes, states)

    actual_indices = [
        index for index, state in enumerate(states) if state in ACTUAL_DETECTION_STATES
    ]
    reference_index = max(
        actual_indices or list(range(frame_count)),
        key=lambda index: float(smooth_h[index]),
    )
    reference_crop = crops[reference_index : reference_index + 1]
    records = []
    for index in range(frame_count):
        shot_id = next(
            shot_index
            for shot_index, (start, end) in enumerate(shot_ranges)
            if start <= index <= end
        )
        records.append(
            {
                "frame_index": index,
                "shot_id": shot_id,
                "state": states[index],
                "detected": states[index] in ACTUAL_DETECTION_STATES,
                "source_face_box_xyxy": [round(value, 6) for value in face_boxes[index]],
                "source_face_height_px": round(float(smooth_h[index]), 6),
                "source_face_width_px": round(float(smooth_w[index]), 6),
                "source_crop_box_xyxy": [round(value, 6) for value in crop_boxes[index]],
                "crop_face_box_xyxy": [
                    round(value, 6)
                    for value in _upstream_parity_face_box_in_crop(
                        smooth_w[index], smooth_h[index], crop_boxes[index], canvas
                    )
                ],
                "parity_denoise_face_height_px": round(
                    float(crop_boxes[index][3] - crop_boxes[index][1])
                    / float(crop_factor),
                    6,
                ),
                "paste_weight": round(float(smooth_weight[index]), 6),
            }
        )

    magnifications = [canvas / max(1e-6, box[3] - box[1]) for box in crop_boxes]
    plan = {
        "schema": PARITY_PLAN_SCHEMA,
        "status": "parity_candidate_plan",
        "source": {
            "frame_count": frame_count,
            "h3_aligned_frame_count": h3_aligned_frame_count,
            "h3_alignment_tail_frames": h3_alignment_tail_frames,
            "h3_grid_aligned": h3_alignment_tail_frames == 0,
            "width": width,
            "height": height,
            "fps": float(fps),
            "proxy_sha256": source_proxy_sha256(frames),
        },
        "canvas": {
            "width": canvas,
            "height": canvas,
            "multiple": 32,
            "mode": canvas_mode,
        },
        "detector": detector_report,
        "shots": [
            {"shot_id": index, "start_frame": start, "end_frame": end}
            for index, (start, end) in enumerate(shot_ranges)
        ],
        "frames": records,
        "reference_frame_index": int(reference_index),
        "preview_frame_indices": preview_indices,
        "parity_defaults": {
            "require_h3_grid": bool(require_h3_grid),
            "center_smooth_window": int(center_smooth_window),
            "size_smooth_window": int(size_smooth_window),
            "crop_factor": float(crop_factor),
            "canvas_mode": canvas_mode,
            "ultralytics_input_colour_space": (
                "bgr_upstream_parity" if detector_mode == "local_ultralytics" else None
            ),
            "face_mask_geometry": "upstream_centered_smoothed_face_rect",
            "denoise_face_size_source": "source_crop_height_divided_by_crop_factor",
            "per_frame_denoise": {
                "strength_small_face": 0.8,
                "strength_large_face": 0.35,
                "scale_mode": "relative_to_clip",
                "face_px_small": 30.0,
                "face_px_large": 120.0,
                "gamma": 1.0,
                "smooth_frames": 9,
            },
            "stitch": {
                "paste_region": "face_only",
                "mask_dilation": 24,
                "feather_source_px": 24.0,
                "colour_match": 1.0,
                "blend": 1.0,
            },
        },
        "limits": {
            "h3_grid_required": bool(require_h3_grid),
            "explicit_alignment_tail_discard_required": h3_alignment_tail_frames > 0,
            "max_supported_alignment_tail_frames": 1,
            "single_pass_safe": len(shot_ranges) == 1,
            "identity_verified": False,
            "reference_crop_is_identity_proof": False,
            "automatic_accept": False,
            "audio_modified": False,
        },
        "metrics": {
            **smooth_metrics,
            "scene_cut_count": len(shot_ranges) - 1,
            "max_scene_delta": max(scene_deltas),
            "detected_frames": len(actual_indices),
            "lost_or_interpolated_frames": frame_count - len(actual_indices),
            "multi_face_frames": multi_face_frames,
            "face_height_min_px": float(smooth_h.min()),
            "face_height_mean_px": float(smooth_h.mean()),
            "face_height_max_px": float(smooth_h.max()),
            "crop_face_height_min_px": min(
                record["crop_face_box_xyxy"][3] - record["crop_face_box_xyxy"][1]
                for record in records
            ),
            "crop_face_height_mean_px": sum(
                record["crop_face_box_xyxy"][3] - record["crop_face_box_xyxy"][1]
                for record in records
            )
            / len(records),
            "crop_face_height_max_px": max(
                record["crop_face_box_xyxy"][3] - record["crop_face_box_xyxy"][1]
                for record in records
            ),
            "magnification_min": min(magnifications),
            "magnification_mean": sum(magnifications) / len(magnifications),
            "magnification_max": max(magnifications),
            "downscaled_crop_frames": sum(value < 1.0 for value in magnifications),
            "crop_tensor_estimated_mib": frame_count * canvas * canvas * 3 * 4 / 2**20,
        },
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    return (
        plan,
        crops,
        reference_crop,
        preview,
        canonical_json(plan),
        canvas,
        canvas,
        frame_count,
        int(reference_index),
    )


def inject_face_refine_parity_video_latent(
    positive,
    av_latent: dict,
    crops: torch.Tensor,
    video_vae,
    face_plan: dict,
    audio_policy: str,
    allow_multi_shot_exp: bool,
):
    plan = _validate_parity_plan(face_plan)
    frame_count, height, width = _validate_frames(crops, name="crops")
    source = plan["source"]
    canvas = plan["canvas"]
    if frame_count != int(source["frame_count"]):
        raise ValueError("crops frame count does not match face_plan")
    if (width, height) != (int(canvas["width"]), int(canvas["height"])):
        raise ValueError("crops canvas does not match face_plan")
    if len(plan["shots"]) > 1 and not allow_multi_shot_exp:
        raise ValueError("face_plan contains scene cuts; split the source into shot-local windows")

    video, audio = nested_av_parts(av_latent)
    aligned_frame_count = int(source.get("h3_aligned_frame_count", frame_count))
    alignment_tail_frames = int(
        source.get("h3_alignment_tail_frames", aligned_frame_count - frame_count)
    )
    if alignment_tail_frames not in (0, 1):
        raise ValueError(
            "Parity latent injection currently supports only an explicit zero- or "
            "one-frame H3 alignment tail"
        )
    expected_t = video_latent_t(aligned_frame_count)
    if int(video.shape[2]) != expected_t:
        raise ValueError(
            f"AV latent video time {video.shape[2]} does not match the explicit "
            f"{aligned_frame_count}-frame H3 grid ({expected_t}); implicit latent "
            "trim/pad is forbidden"
        )
    _, audio_mask = split_noise_masks(av_latent, video, audio)
    if audio_policy == "require_locked":
        if audio_mask is None:
            raise ValueError("require_locked needs a connected nested audio noise_mask")
        if int(torch.count_nonzero(audio_mask).item()) != 0:
            raise ValueError("require_locked refuses a nonzero audio noise_mask")
    elif audio_policy != "preserve_existing":
        raise ValueError(f"Unknown audio_policy: {audio_policy}")

    # Match the selected upstream path: encode the real source-frame batch directly.
    # Any actual latent mismatch remains fail-closed instead of being hidden by trim/pad.
    encode_crops = crops
    alignment_tail_input_policy = "encode_source_frames_directly_no_pixel_tail"
    encoded = video_vae.encode(encode_crops)
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
        raise ValueError("video_vae must return MiniMax H3 video latent [B,C,T,H,W]")
    if tuple(encoded.shape) != tuple(video.shape):
        raise ValueError(
            "Encoded parity crop latent does not exactly match the target AV latent; "
            f"got {tuple(encoded.shape)}, expected {tuple(video.shape)}"
        )
    encoded = encoded.to(device=video.device, dtype=video.dtype)
    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((encoded, audio))
    report = {
        "schema": "h3_t8_face_refine_parity_latent/v1",
        "status": "parity_video_latent_injected",
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "h3_aligned_frame_count": aligned_frame_count,
        "h3_alignment_tail_frames": alignment_tail_frames,
        "encoded_crop_frame_count": int(encode_crops.shape[0]),
        "alignment_tail_input_policy": alignment_tail_input_policy,
        "video_latent_shape": list(encoded.shape),
        "audio_latent_shape": list(audio.shape),
        "audio_policy": audio_policy,
        "audio_tensor_reused": output["samples"].unbind()[1].data_ptr() == audio.data_ptr(),
        "noise_mask_object_reused": output.get("noise_mask") is av_latent.get("noise_mask"),
        "implicit_temporal_fit": False,
        "automatic_accept": False,
    }
    return positive, output, canonical_json(report)


def apply_face_refine_per_frame_denoise(
    av_latent: dict,
    face_plan: dict,
    strength_small_face: float,
    strength_large_face: float,
    scale_mode: str,
    face_px_small: float,
    face_px_large: float,
    gamma: float,
    smooth_frames: int,
    video_mask_mode: str,
    require_locked_audio: bool,
):
    plan = _validate_parity_plan(face_plan)
    video, audio = nested_av_parts(av_latent)
    source = plan["source"]
    aligned_frame_count = int(
        source.get("h3_aligned_frame_count", source["frame_count"])
    )
    if int(video.shape[2]) != video_latent_t(aligned_frame_count):
        raise ValueError("AV latent time does not match parity plan")
    previous_video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    if require_locked_audio:
        if audio_mask is None:
            raise ValueError("Per-frame denoise requires an explicit locked audio mask")
        if int(torch.count_nonzero(audio_mask).item()) != 0:
            raise ValueError("Per-frame denoise refuses a nonzero audio mask")

    crop_factor = float(plan.get("parity_defaults", {}).get("crop_factor", 0.0))
    if crop_factor <= 0:
        raise ValueError("face_plan has no valid parity crop_factor")
    face = np.asarray(
        [
            (
                float(record["source_crop_box_xyxy"][3])
                - float(record["source_crop_box_xyxy"][1])
            )
            / crop_factor
            for record in plan["frames"]
        ],
        dtype=np.float64,
    )
    if scale_mode == "relative_to_clip":
        low, high = float(face.min()), float(face.max())
        relative_scale_degenerate = high <= low
    elif scale_mode == "absolute_px":
        low, high = float(face_px_small), float(face_px_large)
        relative_scale_degenerate = False
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    if scale_mode == "absolute_px" and high <= low:
        raise ValueError("face_px_large must be greater than face_px_small")
    if relative_scale_degenerate:
        normalized = np.full_like(face, 0.5)
    else:
        normalized = np.clip((face - low) / (high - low), 0.0, 1.0) ** float(gamma)
    strength = float(strength_small_face) + (
        float(strength_large_face) - float(strength_small_face)
    ) * normalized
    strength = np.clip(_gaussian_smooth(strength, int(smooth_frames)), 0.0, 1.0)

    curve = torch.from_numpy(strength).float().view(1, 1, -1)
    curve = torch_functional.interpolate(
        curve,
        size=int(video.shape[2]),
        mode="linear",
        align_corners=True,
    )
    curve = curve.view(1, 1, int(video.shape[2]), 1, 1).to(video.device)
    parity_mask = curve.expand(
        int(video.shape[0]),
        int(video.shape[1]),
        int(video.shape[2]),
        int(video.shape[3]),
        int(video.shape[4]),
    ).contiguous()
    parity_mask = parity_mask.to(dtype=video.dtype)
    if video_mask_mode == "replace_video_parity":
        output_video_mask = parity_mask
    elif video_mask_mode == "cap_existing":
        if previous_video_mask is None:
            output_video_mask = parity_mask
        else:
            existing = previous_video_mask.to(device=video.device, dtype=video.dtype)
            output_video_mask = torch.minimum(existing.expand_as(parity_mask), parity_mask)
    else:
        raise ValueError(f"Unknown video_mask_mode: {video_mask_mode}")

    if audio_mask is None:
        audio_mask = torch.zeros_like(audio)
    output = av_latent.copy()
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (output_video_mask, audio_mask)
    )
    output_audio_mask = output["noise_mask"].unbind()[1]
    report = {
        "schema": "h3_t8_face_refine_per_frame_denoise/v1",
        "status": "parity_per_frame_video_mask_applied",
        "plan_sha256": plan["plan_sha256"],
        "scale_mode": scale_mode,
        "requested_strength_small_face": float(strength_small_face),
        "requested_strength_large_face": float(strength_large_face),
        "face_height_min_px": float(face.min()),
        "face_height_max_px": float(face.max()),
        "threshold_low_px": low,
        "threshold_high_px": high,
        "relative_scale_degenerate": relative_scale_degenerate,
        "relative_constant_face_policy": (
            "midpoint_strength" if relative_scale_degenerate else "not_used"
        ),
        "pixel_frame_strength_min": float(strength.min()),
        "pixel_frame_strength_mean": float(strength.mean()),
        "pixel_frame_strength_max": float(strength.max()),
        "latent_time": int(video.shape[2]),
        "h3_aligned_frame_count": aligned_frame_count,
        "h3_alignment_tail_frames": int(
            source.get(
                "h3_alignment_tail_frames",
                aligned_frame_count - int(source["frame_count"]),
            )
        ),
        "gamma": float(gamma),
        "smooth_frames": int(smooth_frames),
        "face_size_source": "source_crop_height_divided_by_crop_factor",
        "video_mask_mode": video_mask_mode,
        "require_locked_audio": bool(require_locked_audio),
        "audio_mask_data_reused": output_audio_mask.data_ptr() == audio_mask.data_ptr(),
        "audio_mask_all_zero": int(torch.count_nonzero(output_audio_mask).item()) == 0,
        "audio_samples_modified": False,
    }
    return output, canonical_json(report)


def _gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    if int(feather) <= 0:
        return mask
    kernel_size = 2 * int(feather) + 1
    shortest = min(int(mask.shape[-2]), int(mask.shape[-1]))
    if shortest <= kernel_size:
        kernel_size = max(3, int(shortest / 2) | 1)
    sigma = max(kernel_size / 6.0, 0.5)
    axis = torch.arange(kernel_size, device=mask.device, dtype=torch.float32)
    axis -= kernel_size // 2
    kernel = torch.exp(-(axis**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).to(mask.dtype)
    pad = kernel_size // 2
    mask = torch_functional.conv2d(
        torch_functional.pad(mask, (pad, pad, 0, 0), mode="replicate"),
        kernel.view(1, 1, 1, kernel_size),
    )
    return torch_functional.conv2d(
        torch_functional.pad(mask, (0, 0, pad, pad), mode="replicate"),
        kernel.view(1, 1, kernel_size, 1),
    )


def _parity_canvas_mask(
    record: dict,
    canvas: int,
    paste_region: str,
    mask_dilation: int,
    feather_source_px: float,
    device: torch.device,
    feather_canvas_override: int | None = None,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, canvas, canvas), device=device, dtype=torch.float32)
    face = record["crop_face_box_xyxy"]
    left = float(face[0]) - int(mask_dilation)
    top = float(face[1]) - int(mask_dilation)
    right = float(face[2]) + int(mask_dilation)
    bottom = float(face[3]) + int(mask_dilation)
    if paste_region == "face_only":
        x0 = max(0, int(round(left)))
        y0 = max(0, int(round(top)))
        x1 = min(canvas, int(round(right)))
        y1 = min(canvas, int(round(bottom)))
        if x1 > x0 and y1 > y0:
            mask[0, 0, y0:y1, x0:x1] = 1.0
    elif paste_region == "face_ellipse":
        yy = torch.arange(canvas, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(canvas, device=device, dtype=torch.float32).view(1, -1)
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        radius_x = max(1.0, (right - left) * 0.5)
        radius_y = max(1.0, (bottom - top) * 0.5)
        mask[0, 0] = (
            ((xx - center_x) / radius_x).square()
            + ((yy - center_y) / radius_y).square()
            <= 1.0
        ).float()
    elif paste_region == "full_crop_exp":
        mask.fill_(1.0)
    else:
        raise ValueError(f"Unknown paste_region: {paste_region}")

    if feather_canvas_override is None:
        crop = record["source_crop_box_xyxy"]
        source_side = max(1.0, float(crop[3]) - float(crop[1]))
        feather_canvas = max(
            1,
            min(canvas // 3, int(round(float(feather_source_px) * canvas / source_side))),
        )
    else:
        feather_canvas = max(1, min(canvas // 3, int(feather_canvas_override)))
    return _gaussian_blur_mask(mask, feather_canvas).clamp(0.0, 1.0)[0, 0]


def _source_space_colour_match(
    refined: torch.Tensor,
    original: torch.Tensor,
    alpha: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Match the audited author's post-warp, source-coordinate colour operation."""

    if strength <= 0:
        return refined
    weights = alpha[..., None]
    denominator = weights.sum(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    original_mean = (original * weights).sum(dim=(0, 1), keepdim=True) / denominator
    refined_mean = (refined * weights).sum(dim=(0, 1), keepdim=True) / denominator
    original_std = (
        ((original - original_mean).square() * weights).sum(dim=(0, 1), keepdim=True)
        / denominator
    ).sqrt().clamp_min(1e-6)
    refined_std = (
        ((refined - refined_mean).square() * weights).sum(dim=(0, 1), keepdim=True)
        / denominator
    ).sqrt().clamp_min(1e-6)
    matched = (refined - refined_mean) * (original_std / refined_std) + original_mean
    return torch.lerp(refined, matched, float(strength)).clamp(0.0, 1.0)


def stitch_face_refine_parity_candidate(
    base_frames: torch.Tensor,
    refined_crops: torch.Tensor,
    face_plan: dict,
    paste_region: str,
    mask_dilation: int,
    feather_source_px: float,
    colour_match: float,
    blend: float,
    undetected_frames: str,
    max_face_mean_abs_delta: float,
    processing_device: str,
):
    plan = _validate_parity_plan(face_plan)
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    crop_count, crop_height, crop_width = _validate_frames(
        refined_crops, name="refined_crops"
    )
    source = plan["source"]
    canvas = int(plan["canvas"]["width"])
    if (frame_count, height, width) != (
        int(source["frame_count"]),
        int(source["height"]),
        int(source["width"]),
    ):
        raise ValueError("base_frames dimensions do not match face_plan")
    if source_proxy_sha256(base_frames) != source["proxy_sha256"]:
        raise ValueError("base_frames content fingerprint does not match face_plan")
    aligned_frame_count = int(source.get("h3_aligned_frame_count", frame_count))
    alignment_tail_frames = int(
        source.get("h3_alignment_tail_frames", aligned_frame_count - frame_count)
    )
    if alignment_tail_frames not in (0, 1):
        raise ValueError(
            "Parity stitch only supports an explicit zero- or one-frame H3 alignment tail"
        )
    if crop_count == aligned_frame_count and alignment_tail_frames == 1:
        refined_crops = refined_crops[:frame_count]
        crop_count = frame_count
        alignment_tail_discarded_frames = 1
    else:
        alignment_tail_discarded_frames = 0
    if crop_count != frame_count or (crop_width, crop_height) != (canvas, canvas):
        raise ValueError("refined_crops frame count or canvas does not match face_plan")
    if alignment_tail_discarded_frames != alignment_tail_frames:
        raise ValueError(
            "refined_crops must carry the exact explicit H3 alignment tail declared by face_plan"
        )
    if processing_device == "cuda_if_available" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif processing_device == "cpu_memory_safe":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown processing_device: {processing_device}")

    per_frame_mib = height * width * 3 * 4 / 2**20
    upstream_chunk_frames = max(1, min(32, int(1024 / max(per_frame_mib, 1e-6))))
    feather_canvas_by_frame = [0] * frame_count
    for chunk_start in range(0, frame_count, upstream_chunk_frames):
        chunk_end = min(frame_count, chunk_start + upstream_chunk_frames)
        midpoint = (chunk_start + chunk_end - 1) // 2
        midpoint_crop = plan["frames"][midpoint]["source_crop_box_xyxy"]
        midpoint_height = max(1.0, float(midpoint_crop[3]) - float(midpoint_crop[1]))
        feather_canvas = max(
            1,
            min(canvas // 3, int(round(float(feather_source_px) * canvas / midpoint_height))),
        )
        for frame_index in range(chunk_start, chunk_end):
            feather_canvas_by_frame[frame_index] = feather_canvas

    output_frames = []
    changed_masks = []
    fallback_masks = []
    scores = []
    fallback_indices = []
    for index, record in enumerate(plan["frames"]):
        base = base_frames[index, ..., :3].detach().to(device=device, dtype=torch.float32)
        crop = refined_crops[index, ..., :3].detach().to(device=device, dtype=torch.float32)
        alpha_canvas = _parity_canvas_mask(
            record,
            canvas,
            paste_region,
            int(mask_dilation),
            float(feather_source_px),
            device,
            feather_canvas_by_frame[index],
        )
        patch, source_alpha = _warp_crop_to_source(
            crop,
            alpha_canvas,
            record["source_crop_box_xyxy"],
            height,
            width,
        )
        patch = _source_space_colour_match(
            patch, base, source_alpha, float(colour_match)
        )
        denominator = source_alpha.sum().clamp_min(1.0)
        score = float(
            ((patch - base).abs().mean(dim=-1) * source_alpha)
            .sum()
            .div(denominator)
            .item()
        )
        scores.append(score)
        invalid = not math.isfinite(score) or score > float(max_face_mean_abs_delta)
        if invalid:
            fallback_indices.append(index)

        if undetected_frames == "composite_anyway":
            detection_weight = 1.0
        elif undetected_frames == "skip":
            detection_weight = 1.0 if record["detected"] else 0.0
        elif undetected_frames == "fade_out":
            detection_weight = float(record["paste_weight"])
        else:
            raise ValueError(f"Unknown undetected_frames: {undetected_frames}")
        if invalid:
            detection_weight = 0.0
        alpha = source_alpha * detection_weight * float(blend)
        blended = base * (1.0 - alpha[..., None]) + patch * alpha[..., None]
        blended = torch.where(alpha[..., None] > 0, blended, base)
        output_frames.append(
            blended.clamp(0.0, 1.0).to(device="cpu", dtype=base_frames.dtype)
        )
        changed_masks.append(alpha.to(device="cpu", dtype=base_frames.dtype))
        fallback_masks.append(
            torch.full((height, width), float(invalid), dtype=base_frames.dtype)
        )

    result = torch.stack(output_frames).to(base_frames.device)
    changed_mask = torch.stack(changed_masks).to(base_frames.device)
    fallback_mask = torch.stack(fallback_masks).to(base_frames.device)
    outside = changed_mask == 0
    outside_exact = bool(torch.equal(result[outside], base_frames[..., :3][outside]))
    if not outside_exact:
        raise RuntimeError("Parity stitch audit failed: pixels outside the mask changed")
    report = {
        "schema": "h3_t8_face_refine_parity_stitch/v1",
        "status": "parity_candidate_requires_review",
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "h3_aligned_frame_count": aligned_frame_count,
        "h3_alignment_tail_frames": alignment_tail_frames,
        "alignment_tail_discarded_frames": alignment_tail_discarded_frames,
        "paste_region": paste_region,
        "mask_dilation": int(mask_dilation),
        "feather_source_px": float(feather_source_px),
        "feather_canvas_policy": "upstream_chunk_midpoint_source_scale",
        "upstream_chunk_frames": upstream_chunk_frames,
        "feather_canvas_min": min(feather_canvas_by_frame),
        "feather_canvas_max": max(feather_canvas_by_frame),
        "colour_match": float(colour_match),
        "colour_match_space": "post_warp_source_coordinates",
        "blend": float(blend),
        "undetected_frames": undetected_frames,
        "fallback_frames": fallback_indices,
        "fallback_count": len(fallback_indices),
        "mean_face_delta": sum(scores) / max(1, len(scores)),
        "max_face_delta": max(scores),
        "mask_outside_bit_exact": outside_exact,
        "processing_device": str(device),
        "audio_modified": False,
        "identity_verified": False,
        "automatic_accept": False,
    }
    return (
        result,
        changed_mask,
        fallback_mask,
        len(fallback_indices),
        canonical_json(report),
    )


def _parse_baseline_report(value: str, *, name: str, schema: str) -> dict:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a connected non-empty JSON report")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{name} must use {schema}")
    return payload


def validate_face_refine_manual512_relative_baseline(
    candidate_frames: torch.Tensor,
    face_plan: dict,
    latent_report_json: str,
    denoise_report_json: str,
    stitch_report_json: str,
    profile: str,
    minimum_crop_face_height_px: float,
):
    """Fail closed unless the human-selected MANUAL512 REL mechanics are preserved."""

    if profile != MANUAL512_RELATIVE_PROFILE:
        raise ValueError(f"Unsupported Face Refine baseline profile: {profile}")
    plan = _validate_parity_plan(face_plan)
    frame_count, height, width = _validate_frames(
        candidate_frames, name="candidate_frames"
    )
    source = plan["source"]
    if (frame_count, height, width) != (
        int(source["frame_count"]),
        int(source["height"]),
        int(source["width"]),
    ):
        raise ValueError("candidate_frames dimensions do not match face_plan source")
    if not bool(torch.isfinite(candidate_frames).all()):
        raise ValueError("candidate_frames contains NaN or Inf")

    canvas = plan["canvas"]
    defaults = plan.get("parity_defaults", {})
    if canvas.get("mode") != "manual_512" or (
        int(canvas.get("width", -1)), int(canvas.get("height", -1))
    ) != (512, 512):
        raise ValueError(
            "manual512_relative_author_parity_v2 requires manual_512 and a 512x512 crop canvas"
        )
    if not math.isclose(float(defaults.get("crop_factor", -1.0)), 2.5, abs_tol=1e-6):
        raise ValueError("manual512_relative_author_parity_v2 requires crop_factor=2.5")
    if int(defaults.get("center_smooth_window", -1)) != 21 or int(
        defaults.get("size_smooth_window", -1)
    ) != 51:
        raise ValueError(
            "manual512_relative_author_parity_v2 requires 21/51 Gaussian trajectory smoothing"
        )
    detector = plan.get("detector", {})
    if detector.get("backend") != "local_ultralytics" or detector.get(
        "input_colour_space"
    ) != "bgr_upstream_parity":
        raise ValueError(
            "manual512_relative_author_parity_v2 requires the audited local YOLO BGR path"
        )
    if defaults.get("face_mask_geometry") != "upstream_centered_smoothed_face_rect":
        raise ValueError(
            "manual512_relative_author_parity_v2 requires the upstream-centred face mask"
        )
    if defaults.get("denoise_face_size_source") != (
        "source_crop_height_divided_by_crop_factor"
    ):
        raise ValueError(
            "manual512_relative_author_parity_v2 requires the audited crop-derived face size"
        )

    crop_face_heights = [
        float(record["crop_face_box_xyxy"][3])
        - float(record["crop_face_box_xyxy"][1])
        for record in plan["frames"]
    ]
    minimum_measured_height = min(crop_face_heights)
    if minimum_measured_height + 1e-4 < float(minimum_crop_face_height_px):
        raise ValueError(
            "manual512_relative_author_parity_v2 face scale contract failed: "
            f"minimum crop face height {minimum_measured_height:.3f}px is below "
            f"{float(minimum_crop_face_height_px):.3f}px"
        )

    latent = _parse_baseline_report(
        latent_report_json,
        name="latent_report_json",
        schema="h3_t8_face_refine_parity_latent/v1",
    )
    denoise = _parse_baseline_report(
        denoise_report_json,
        name="denoise_report_json",
        schema="h3_t8_face_refine_per_frame_denoise/v1",
    )
    stitch = _parse_baseline_report(
        stitch_report_json,
        name="stitch_report_json",
        schema="h3_t8_face_refine_parity_stitch/v1",
    )
    for report_name, payload in (
        ("latent", latent),
        ("denoise", denoise),
        ("stitch", stitch),
    ):
        if payload.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(f"{report_name} report belongs to a different face_plan")

    aligned_frame_count = int(source.get("h3_aligned_frame_count", frame_count))
    alignment_tail_frames = int(
        source.get("h3_alignment_tail_frames", aligned_frame_count - frame_count)
    )
    if alignment_tail_frames not in (0, 1):
        raise ValueError(
            "manual512_relative_author_parity_v2 only accepts zero or one alignment tail frame"
        )
    latent_contract = {
        "frame_count": frame_count,
        "h3_aligned_frame_count": aligned_frame_count,
        "h3_alignment_tail_frames": alignment_tail_frames,
        "encoded_crop_frame_count": frame_count,
        "alignment_tail_input_policy": "encode_source_frames_directly_no_pixel_tail",
        "audio_policy": "require_locked",
        "audio_tensor_reused": True,
        "noise_mask_object_reused": True,
        "implicit_temporal_fit": False,
    }
    for key, expected in latent_contract.items():
        if latent.get(key) != expected:
            raise ValueError(
                f"manual512_relative_author_parity_v2 requires latent {key}={expected!r}; "
                f"got {latent.get(key)!r}"
            )

    denoise_contract = {
        "scale_mode": "relative_to_clip",
        "video_mask_mode": "replace_video_parity",
        "require_locked_audio": True,
        "audio_mask_all_zero": True,
        "audio_samples_modified": False,
        "smooth_frames": 9,
        "face_size_source": "source_crop_height_divided_by_crop_factor",
    }
    for key, expected in denoise_contract.items():
        if denoise.get(key) != expected:
            raise ValueError(
                f"manual512_relative_author_parity_v2 requires denoise {key}={expected!r}; "
                f"got {denoise.get(key)!r}"
            )
    for key, expected in (
        ("requested_strength_small_face", 0.8),
        ("requested_strength_large_face", 0.35),
        ("gamma", 1.0),
    ):
        if not math.isclose(float(denoise.get(key, float("nan"))), expected, abs_tol=1e-6):
            raise ValueError(
                f"manual512_relative_author_parity_v2 requires denoise {key}={expected}; "
                f"got {denoise.get(key)!r}"
            )

    stitch_contract = {
        "paste_region": "face_only",
        "mask_dilation": 24,
        "undetected_frames": "fade_out",
        "mask_outside_bit_exact": True,
        "audio_modified": False,
        "fallback_count": 0,
        "feather_canvas_policy": "upstream_chunk_midpoint_source_scale",
        "colour_match_space": "post_warp_source_coordinates",
    }
    for key, expected in stitch_contract.items():
        if stitch.get(key) != expected:
            raise ValueError(
                f"manual512_relative_author_parity_v2 requires stitch {key}={expected!r}; "
                f"got {stitch.get(key)!r}"
            )
    if stitch.get("alignment_tail_discarded_frames") != alignment_tail_frames:
        raise ValueError(
            "manual512_relative_author_parity_v2 requires the stitch to discard the explicit "
            "H3 alignment tail and no other frames"
        )
    for key, expected in (
        ("feather_source_px", 24.0),
        ("colour_match", 1.0),
        ("blend", 1.0),
    ):
        if not math.isclose(float(stitch.get(key, float("nan"))), expected, abs_tol=1e-6):
            raise ValueError(
                f"manual512_relative_author_parity_v2 requires stitch {key}={expected}; "
                f"got {stitch.get(key)!r}"
            )

    magnification = [
        512.0
        / max(
            1e-6,
            float(record["source_crop_box_xyxy"][3])
            - float(record["source_crop_box_xyxy"][1]),
        )
        for record in plan["frames"]
    ]
    warnings = []
    if min(magnification) < 1.0:
        warnings.append(
            "Some source crops are downscaled into the 512 canvas; this is valid but is not "
            "the low-resolution magnification case used for the human-selected reference."
        )
    report = {
        "schema": MANUAL512_RELATIVE_BASELINE_SCHEMA,
        "status": "mechanical_baseline_matched_candidate_requires_human_review",
        "profile": profile,
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "h3_aligned_frame_count": aligned_frame_count,
        "h3_alignment_tail_frames": alignment_tail_frames,
        "alignment_tail_discarded_frames": alignment_tail_frames,
        "source_dimensions": [width, height],
        "crop_canvas": [512, 512],
        "crop_factor": 2.5,
        "scale_mode": "relative_to_clip",
        "crop_face_height_min_px": minimum_measured_height,
        "crop_face_height_mean_px": sum(crop_face_heights) / len(crop_face_heights),
        "crop_face_height_max_px": max(crop_face_heights),
        "minimum_crop_face_height_required_px": float(minimum_crop_face_height_px),
        "magnification_min": min(magnification),
        "magnification_mean": sum(magnification) / len(magnification),
        "magnification_max": max(magnification),
        "candidate_tensor_reused": True,
        "mechanical_baseline_matched": True,
        "human_selected_reference": {
            "filename": "face_refine_manual512_crop2p5_relative_to_clip_seed42_00001_.mp4",
            "sha256": "19EA5844643B962F6FD197E34705861916D69F7EA70F3E00A2DF022D6A017399",
            "scope": "one fixed local fixture, seed 42, reviewed by the user",
        },
        "quality_guaranteed": False,
        "identity_verified": False,
        "automatic_accept": False,
        "universal_16gb_safe": False,
        "warnings": warnings,
    }
    return candidate_frames, canonical_json(report)


def _normalize_full_frame_mask(
    mask: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("changed_mask must be a torch.Tensor")
    if mask.ndim == 4 and int(mask.shape[1]) == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if tuple(mask.shape) != (frame_count, height, width):
        raise ValueError(
            "changed_mask must have shape [frames,height,width] matching base_frames"
        )
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("changed_mask contains NaN or Inf")
    return mask.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)


def _expanded_source_face_roi(
    box: list[float],
    width: int,
    height: int,
    scale: float = 1.5,
) -> tuple[int, int, int, int]:
    center_x = (float(box[0]) + float(box[2])) * 0.5
    center_y = (float(box[1]) + float(box[3])) * 0.5
    box_width = max(1.0, float(box[2]) - float(box[0])) * float(scale)
    box_height = max(1.0, float(box[3]) - float(box[1])) * float(scale)
    left = max(0, int(math.floor(center_x - box_width * 0.5)))
    top = max(0, int(math.floor(center_y - box_height * 0.5)))
    right = min(int(width), int(math.ceil(center_x + box_width * 0.5)))
    bottom = min(int(height), int(math.ceil(center_y + box_height * 0.5)))
    if right - left < 5 or bottom - top < 5:
        raise ValueError(f"Face quality ROI is too small: {(left, top, right, bottom)}")
    return left, top, right, bottom


def _normalized_gray_face(
    frame: torch.Tensor,
    roi: tuple[int, int, int, int],
    size: int = 64,
) -> torch.Tensor:
    left, top, right, bottom = roi
    crop = frame[top:bottom, left:right, :3].permute(2, 0, 1).unsqueeze(0)
    crop = torch_functional.interpolate(
        crop,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    )
    weights = crop.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (crop * weights).sum(dim=1, keepdim=True)


def _global_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float()
    second = second.float()
    mean_first = first.mean()
    mean_second = second.mean()
    centered_first = first - mean_first
    centered_second = second - mean_second
    variance_first = centered_first.square().mean()
    variance_second = centered_second.square().mean()
    covariance = (centered_first * centered_second).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_first * mean_second + c1) * (2.0 * covariance + c2)
    denominator = (
        mean_first.square() + mean_second.square() + c1
    ) * (variance_first + variance_second + c2)
    return float((numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0).item())


def _laplacian_variance(gray: torch.Tensor) -> float:
    kernel = gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    response = torch_functional.conv2d(
        torch_functional.pad(gray, (1, 1, 1, 1), mode="replicate"), kernel
    )
    return float(response.var(unbiased=False).item())


def _accepted_runs(raw_accept: list[bool], minimum_run: int) -> list[bool]:
    accepted = [False] * len(raw_accept)
    start = 0
    while start < len(raw_accept):
        if not raw_accept[start]:
            start += 1
            continue
        end = start + 1
        while end < len(raw_accept) and raw_accept[end]:
            end += 1
        if end - start >= int(minimum_run):
            accepted[start:end] = [True] * (end - start)
        start = end
    return accepted


def _taper_accepted_runs(accepted: list[bool], edge_fade_frames: int) -> list[float]:
    weights = [0.0] * len(accepted)
    start = 0
    fade = max(0, int(edge_fade_frames))
    while start < len(accepted):
        if not accepted[start]:
            start += 1
            continue
        end = start + 1
        while end < len(accepted) and accepted[end]:
            end += 1
        for index in range(start, end):
            if fade == 0:
                weights[index] = 1.0
                continue
            edge_distance = min(index - start + 1, end - index)
            weights[index] = min(1.0, edge_distance / float(fade + 1))
        start = end
    return weights


def gate_face_refine_parity_candidate(
    base_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    changed_mask: torch.Tensor,
    face_plan: dict,
    min_structure_ssim: float,
    min_sharpness_ratio: float,
    max_sharpness_ratio: float,
    max_face_mean_abs_delta: float,
    max_residual_temporal_jitter: float,
    minimum_accept_run: int,
    edge_fade_frames: int,
):
    """Conservatively retain only candidate frames that pass source-relative proxies.

    This is deliberately not an identity or quality oracle.  It can reject obvious
    regressions and return the original source, but a passing frame still requires review.
    """

    plan = _validate_parity_plan(face_plan)
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    candidate_count, candidate_height, candidate_width = _validate_frames(
        candidate_frames, name="candidate_frames"
    )
    if (candidate_count, candidate_height, candidate_width) != (
        frame_count,
        height,
        width,
    ):
        raise ValueError("candidate_frames dimensions must exactly match base_frames")
    source = plan["source"]
    if (frame_count, height, width) != (
        int(source["frame_count"]),
        int(source["height"]),
        int(source["width"]),
    ):
        raise ValueError("base_frames dimensions do not match face_plan")
    if source_proxy_sha256(base_frames) != source["proxy_sha256"]:
        raise ValueError("base_frames content fingerprint does not match face_plan")
    if not bool(torch.isfinite(candidate_frames).all()):
        raise ValueError("candidate_frames contains NaN or Inf")
    if not (0.0 <= float(min_structure_ssim) <= 1.0):
        raise ValueError("min_structure_ssim must be within 0..1")
    if float(min_sharpness_ratio) < 0.0:
        raise ValueError("min_sharpness_ratio must be non-negative")
    if float(max_sharpness_ratio) < float(min_sharpness_ratio):
        raise ValueError("max_sharpness_ratio must be >= min_sharpness_ratio")
    if not (0.0 <= float(max_face_mean_abs_delta) <= 1.0):
        raise ValueError("max_face_mean_abs_delta must be within 0..1")
    if float(max_residual_temporal_jitter) < 0.0:
        raise ValueError("max_residual_temporal_jitter must be non-negative")
    if int(minimum_accept_run) < 1:
        raise ValueError("minimum_accept_run must be positive")
    if int(edge_fade_frames) < 0:
        raise ValueError("edge_fade_frames must be non-negative")

    base_cpu = base_frames[..., :3].detach().to(device="cpu", dtype=torch.float32)
    candidate_cpu = (
        candidate_frames[..., :3].detach().to(device="cpu", dtype=torch.float32)
    )
    mask_cpu = _normalize_full_frame_mask(changed_mask, frame_count, height, width)
    outside = (mask_cpu <= 0.0).unsqueeze(-1).expand_as(base_cpu)
    if not bool(torch.equal(candidate_cpu[outside], base_cpu[outside])):
        raise ValueError(
            "candidate_frames changed pixels outside changed_mask; quality gate refuses "
            "non-Parity or codec-round-tripped input"
        )

    frame_metrics = []
    raw_accept = []
    previous_residual = None
    previous_shot = None
    for index, record in enumerate(plan["frames"]):
        roi = _expanded_source_face_roi(
            record["source_face_box_xyxy"], width, height
        )
        base_gray = _normalized_gray_face(base_cpu[index], roi)
        candidate_gray = _normalized_gray_face(candidate_cpu[index], roi)
        structure = _global_ssim(base_gray, candidate_gray)
        left, top, right, bottom = roi
        face_delta = float(
            (
                candidate_cpu[index, top:bottom, left:right]
                - base_cpu[index, top:bottom, left:right]
            )
            .abs()
            .mean()
            .item()
        )
        source_sharpness = _laplacian_variance(base_gray)
        candidate_sharpness = _laplacian_variance(candidate_gray)
        sharpness_ratio = (candidate_sharpness + 1e-8) / (
            source_sharpness + 1e-8
        )
        residual = candidate_gray - base_gray
        shot_index = int(record.get("shot_index", 0))
        if previous_residual is None or previous_shot != shot_index:
            residual_jitter = 0.0
        else:
            residual_jitter = float((residual - previous_residual).abs().mean().item())
        previous_residual = residual
        previous_shot = shot_index

        reasons = []
        if record.get("state") not in ACTUAL_DETECTION_STATES:
            reasons.append("face_not_directly_detected")
        if structure < float(min_structure_ssim):
            reasons.append("structure_below_threshold")
        if face_delta > float(max_face_mean_abs_delta):
            reasons.append("face_delta_above_threshold")
        if sharpness_ratio < float(min_sharpness_ratio):
            reasons.append("sharpness_gain_below_threshold")
        if sharpness_ratio > float(max_sharpness_ratio):
            reasons.append("sharpness_ratio_above_artifact_ceiling")
        if residual_jitter > float(max_residual_temporal_jitter):
            reasons.append("residual_temporal_jitter_above_threshold")
        finite = all(
            math.isfinite(value)
            for value in (
                structure,
                face_delta,
                source_sharpness,
                candidate_sharpness,
                sharpness_ratio,
                residual_jitter,
            )
        )
        if not finite:
            reasons.append("non_finite_metric")
        frame_metrics.append(
            {
                "frame_index": index,
                "state": record.get("state"),
                "structure_ssim": round(structure, 7),
                "face_mean_abs_delta": round(face_delta, 7),
                "source_laplacian_variance": round(source_sharpness, 9),
                "candidate_laplacian_variance": round(candidate_sharpness, 9),
                "sharpness_ratio": round(sharpness_ratio, 7),
                "residual_temporal_jitter": round(residual_jitter, 7),
                "raw_accept": not reasons,
                "reasons": reasons,
            }
        )
        raw_accept.append(not reasons)

    accepted = _accepted_runs(raw_accept, int(minimum_accept_run))
    weights = _taper_accepted_runs(accepted, int(edge_fade_frames))
    for index, accepted_value in enumerate(accepted):
        frame_metrics[index]["accepted_after_run_filter"] = bool(accepted_value)
        frame_metrics[index]["blend_weight"] = round(float(weights[index]), 7)
        if raw_accept[index] and not accepted_value:
            frame_metrics[index]["reasons"].append("accepted_run_too_short")

    frame_weights = base_cpu.new_tensor(weights).view(frame_count, 1, 1)
    applied_mask = mask_cpu * frame_weights
    result_cpu = base_cpu + (candidate_cpu - base_cpu) * applied_mask.unsqueeze(-1)
    result_cpu = torch.where(
        applied_mask.unsqueeze(-1) > 0.0, result_cpu, base_cpu
    ).clamp(0.0, 1.0)
    output = result_cpu.to(device=base_frames.device, dtype=base_frames.dtype)
    applied_mask = applied_mask.to(device=base_frames.device, dtype=base_frames.dtype)
    rejected = base_frames.new_tensor([not value for value in accepted]).view(
        frame_count, 1, 1
    )
    rejected_mask = rejected.expand(frame_count, height, width)
    outside_applied = applied_mask <= 0.0
    outside_exact = bool(
        torch.equal(output[outside_applied], base_frames[..., :3][outside_applied])
    )
    if not outside_exact:
        raise RuntimeError("Face Refine quality gate changed a rejected source pixel")

    accepted_count = sum(bool(value) for value in accepted)
    raw_accept_count = sum(bool(value) for value in raw_accept)
    report = {
        "schema": "h3_t8_face_refine_proxy_quality_gate/v1",
        "status": (
            "no_frame_met_proxy_gate_source_returned"
            if accepted_count == 0
            else "proxy_gated_candidate_requires_human_review"
        ),
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "raw_accept_count": raw_accept_count,
        "accepted_after_run_filter_count": accepted_count,
        "rejected_count": frame_count - accepted_count,
        "thresholds": {
            "min_structure_ssim": float(min_structure_ssim),
            "min_sharpness_ratio": float(min_sharpness_ratio),
            "max_sharpness_ratio": float(max_sharpness_ratio),
            "max_face_mean_abs_delta": float(max_face_mean_abs_delta),
            "max_residual_temporal_jitter": float(max_residual_temporal_jitter),
            "minimum_accept_run": int(minimum_accept_run),
            "edge_fade_frames": int(edge_fade_frames),
        },
        "frames": frame_metrics,
        "changed_mask_outside_bit_exact": outside_exact,
        "source_returned_bit_exact": bool(
            accepted_count == 0 and torch.equal(output, base_frames[..., :3])
        ),
        "identity_verified": False,
        "quality_validated": False,
        "automatic_accept": False,
        "limitations": [
            "Source-relative SSIM, Laplacian variance, MAE and residual jitter are only rejection proxies.",
            "A passing frame is not proof of restored identity, naturalness or facial detail.",
            "The gate may conservatively return the original source for every frame.",
        ],
    }
    return (
        output,
        applied_mask,
        rejected_mask,
        accepted_count,
        frame_count - accepted_count,
        canonical_json(report),
    )


__all__ = [
    "ANIME_FACE_V14_N_RELATIVE",
    "PARITY_CANVAS_MODES",
    "PARITY_PLAN_SCHEMA",
    "YUNET_2023MAR_RELATIVE",
    "apply_face_refine_per_frame_denoise",
    "build_face_refine_parity_plan",
    "gate_face_refine_parity_candidate",
    "inject_face_refine_parity_video_latent",
    "stitch_face_refine_parity_candidate",
]
