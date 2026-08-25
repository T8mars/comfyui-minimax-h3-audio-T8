from __future__ import annotations

import gc
import json
import math
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as torch_functional

from .face_refine_advanced import (
    YUNET_2023MAR_RELATIVE,
    _detect_local_opencv_yunet,
)
from .multiface_refine_advanced import (
    ASSIGNMENT_SCHEMA,
    TRACK_PLAN_SCHEMA,
    _mask_at_source,
    _source_contract,
    _validate_hashed_dict,
)
from .skin_finish import (
    _interrupt_and_progress,
    _memory_snapshot,
    _progress_bar,
    _tensor_proxy_sha256,
)
from .skin_finish_p1 import (
    _detection_person_overlap,
    _quality_weight,
    _shot_for_frame,
)
from .skin_finish_parser import (
    PARSENET_CLASS_NAMES,
    PARSENET_MODEL_NAME,
    PARSENET_MODEL_RELATIVE,
    PARSENET_MODEL_SHA256,
    PARSENET_MODEL_SIZE,
    _ParserUnavailable,
    _build_preview,
    _load_pinned_parsenet,
    _parser_logits,
    _preview_indices,
    _semantic_local_masks,
    _square_crop_box,
)


SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA = (
    "h3_t8_skin_finish_multiface_semantic_mask/v1"
)
ALIGNMENT_POLICIES = (
    "five_point_strict",
    "five_point_then_profile_crop",
)

# Standard 512x512 FFHQ five-point template used by FaceXLib's restoration helper.
# YuNet's first/second eye and mouth corners are normalized by x coordinate before use,
# because its semantic left/right names are viewpoint-dependent while the template is not.
FFHQ_FIVE_POINT_TEMPLATE_512 = np.asarray(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


class _MultiParserUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _report_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_track_plan(frames: torch.Tensor, track_plan: dict) -> dict:
    try:
        plan = _validate_hashed_dict(track_plan, TRACK_PLAN_SCHEMA, "track_plan")
    except Exception as error:
        raise _MultiParserUnavailable(
            "ABSTAIN_TRACK_PLAN_MISSING_OR_INVALID", str(error)
        ) from error
    source = _source_contract(frames)
    if plan.get("status") != "sam31_shot_local_tracks_ready":
        raise _MultiParserUnavailable(
            "ABSTAIN_TRACK_PLAN_NOT_READY",
            "SAM3.1 track plan is validly hashed but is not in the ready state.",
        )
    expected = plan.get("source", {})
    if any(
        source[key] != expected.get(key)
        for key in ("frame_count", "height", "width", "proxy_sha256")
    ):
        raise _MultiParserUnavailable(
            "ABSTAIN_TRACK_PLAN_SOURCE_MISMATCH",
            "SAM3.1 track plan is bound to different source pixels or geometry.",
        )
    return plan


def _identity_labels(identity_assignment: dict | None, track_plan: dict) -> dict[str, str]:
    if identity_assignment is None:
        return {}
    try:
        assignment = _validate_hashed_dict(
            identity_assignment,
            ASSIGNMENT_SCHEMA,
            "identity_assignment",
        )
    except Exception as error:
        raise _MultiParserUnavailable(
            "ABSTAIN_IDENTITY_ASSIGNMENT_INVALID", str(error)
        ) from error
    if assignment.get("track_plan_sha256") != track_plan.get("sha256"):
        raise _MultiParserUnavailable(
            "ABSTAIN_IDENTITY_ASSIGNMENT_TRACK_MISMATCH",
            "identity_assignment belongs to a different SAM3.1 track plan.",
        )
    if assignment.get("source", {}).get("proxy_sha256") != track_plan.get(
        "source", {}
    ).get("proxy_sha256"):
        raise _MultiParserUnavailable(
            "ABSTAIN_IDENTITY_ASSIGNMENT_SOURCE_MISMATCH",
            "identity_assignment belongs to different source pixels.",
        )
    labels: dict[str, str] = {}
    for mapping in assignment.get("mappings", []):
        track_key = str(mapping.get("track_key", ""))
        character_id = str(mapping.get("character_id", ""))
        if track_key and character_id:
            if track_key in labels:
                raise _MultiParserUnavailable(
                    "ABSTAIN_IDENTITY_ASSIGNMENT_DUPLICATE_TRACK",
                    f"identity_assignment maps track {track_key!r} more than once.",
                )
            labels[track_key] = character_id
    return labels


def _normalized_five_points(detection: dict) -> np.ndarray:
    landmarks = detection.get("landmarks_xy")
    if not isinstance(landmarks, list) or len(landmarks) != 5:
        raise ValueError("YuNet detection does not contain five landmarks")
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("YuNet five landmarks are malformed or non-finite")
    eyes = points[:2][np.argsort(points[:2, 0])]
    mouth = points[3:5][np.argsort(points[3:5, 0])]
    normalized = np.stack((eyes[0], eyes[1], points[2], mouth[0], mouth[1]))
    if float(np.linalg.norm(normalized[1] - normalized[0])) < 4.0:
        raise ValueError("YuNet eye distance is too small for stable five-point alignment")
    return normalized.astype(np.float32, copy=False)


def _align_face(
    frame_rgb: np.ndarray,
    detection: dict,
    *,
    maximum_alignment_rms: float,
) -> tuple[torch.Tensor, np.ndarray, float]:
    try:
        import cv2
    except Exception as error:
        raise _MultiParserUnavailable(
            "ABSTAIN_OPENCV_AFFINE_UNAVAILABLE",
            f"Five-point alignment requires OpenCV: {error}",
        ) from error
    source_points = _normalized_five_points(detection)
    matrix, _ = cv2.estimateAffinePartial2D(
        source_points,
        FFHQ_FIVE_POINT_TEMPLATE_512,
        method=cv2.LMEDS,
    )
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("OpenCV could not estimate a finite five-point similarity transform")
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    if not math.isfinite(determinant) or determinant <= 1.0e-8:
        raise ValueError("Five-point transform is reflected or degenerate")
    projected = source_points @ linear.T + matrix[:, 2]
    rms = float(
        np.sqrt(np.mean(np.sum((projected - FFHQ_FIVE_POINT_TEMPLATE_512) ** 2, axis=1)))
        / 512.0
    )
    if not math.isfinite(rms) or rms > float(maximum_alignment_rms):
        raise ValueError(
            f"Five-point normalized alignment RMS {rms:.6f} exceeds "
            f"{float(maximum_alignment_rms):.6f}"
        )
    aligned = cv2.warpAffine(
        frame_rgb,
        matrix,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135.0 / 255.0, 133.0 / 255.0, 132.0 / 255.0),
    )
    inverse = cv2.invertAffineTransform(matrix)
    if inverse.shape != (2, 3) or not np.isfinite(inverse).all():
        raise ValueError("OpenCV returned a non-finite inverse face transform")
    aligned_tensor = torch.from_numpy(np.ascontiguousarray(aligned)).float().unsqueeze(0)
    return aligned_tensor, inverse.astype(np.float32), rms


def _warp_aligned_mask(
    mask: torch.Tensor,
    inverse: np.ndarray,
    *,
    width: int,
    height: int,
    nearest: bool,
) -> torch.Tensor:
    import cv2

    array = mask.detach().to(device="cpu", dtype=torch.float32).numpy()
    warped = cv2.warpAffine(
        array,
        inverse,
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return torch.from_numpy(np.ascontiguousarray(warped)).float().clamp_(0.0, 1.0)


def _frame_rgb_numpy(frame: torch.Tensor) -> np.ndarray:
    return (
        frame[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .contiguous()
        .numpy()
    )


def _profile_crop_face(
    frame: torch.Tensor,
    detection: dict,
    *,
    expansion: float,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    height, width = int(frame.shape[0]), int(frame.shape[1])
    crop_box = _square_crop_box(
        detection.get("box"),
        width,
        height,
        float(expansion),
    )
    if crop_box is None:
        raise ValueError("Profile fallback could not build a finite square face crop")
    x1, y1, x2, y2 = crop_box
    crop = (
        frame[y1:y2, x1:x2, :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .unsqueeze(0)
    )
    if crop.shape[1] < 8 or crop.shape[2] < 8:
        raise ValueError("Profile fallback face crop is too small")
    return crop, crop_box


def _project_profile_crop_mask(
    mask: torch.Tensor,
    crop_box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    nearest: bool,
) -> torch.Tensor:
    x1, y1, x2, y2 = crop_box
    resized = torch_functional.interpolate(
        mask.detach().to(device="cpu", dtype=torch.float32)[None, None],
        size=(y2 - y1, x2 - x1),
        mode="nearest" if nearest else "bilinear",
        align_corners=None if nearest else False,
    )[0, 0]
    projected = torch.zeros((height, width), dtype=torch.float32)
    projected[y1:y2, x1:x2] = resized
    return projected.clamp_(0.0, 1.0)


def run_multiface_semantic_skin_mask(
    frames: torch.Tensor,
    track_plan: dict | None,
    *,
    identity_assignment: dict | None = None,
    parser_model: str = PARSENET_MODEL_NAME,
    detection_threshold: float = 0.45,
    minimum_face_height_px: float = 32.0,
    minimum_detail: float = 0.010,
    minimum_person_overlap: float = 0.20,
    minimum_track_quality: float = 0.10,
    minimum_class_probability: float = 0.55,
    feature_protection_px: int = 3,
    include_neck: bool = False,
    minimum_skin_area_per_face: float = 0.00005,
    maximum_skin_area_per_frame: float = 0.35,
    maximum_alignment_rms: float = 0.08,
    alignment_policy: str = "five_point_strict",
    profile_crop_expansion: float = 1.45,
    minimum_ready_frame_fraction: float = 0.50,
    preview_count: int = 6,
):
    if parser_model != PARSENET_MODEL_NAME:
        raise ValueError(f"Unsupported parser_model: {parser_model}")
    if not 0.0 <= float(detection_threshold) <= 1.0:
        raise ValueError("detection_threshold must stay within 0..1")
    if float(minimum_face_height_px) < 8.0:
        raise ValueError("minimum_face_height_px must be at least 8")
    if not 0.0 <= float(minimum_person_overlap) <= 1.0:
        raise ValueError("minimum_person_overlap must stay within 0..1")
    if not 0.0 <= float(minimum_track_quality) <= 1.0:
        raise ValueError("minimum_track_quality must stay within 0..1")
    if not 0.0 <= float(minimum_class_probability) <= 1.0:
        raise ValueError("minimum_class_probability must stay within 0..1")
    if not 0.0 <= float(minimum_skin_area_per_face) < float(
        maximum_skin_area_per_frame
    ) <= 1.0:
        raise ValueError("skin area limits must satisfy 0 <= minimum < maximum <= 1")
    if not 0.0 < float(maximum_alignment_rms) <= 0.25:
        raise ValueError("maximum_alignment_rms must stay within 0..0.25")
    if alignment_policy not in ALIGNMENT_POLICIES:
        raise ValueError(f"Unknown alignment_policy: {alignment_policy}")
    if not 1.0 <= float(profile_crop_expansion) <= 3.0:
        raise ValueError("profile_crop_expansion must stay within 1.0..3.0")
    if not 0.0 <= float(minimum_ready_frame_fraction) <= 1.0:
        raise ValueError("minimum_ready_frame_fraction must stay within 0..1")

    source = _source_contract(frames)
    frame_count = int(source["frame_count"])
    height = int(source["height"])
    width = int(source["width"])
    preview_indices = _preview_indices(frame_count, int(preview_count))
    combined_mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
    feature_previews: dict[int, torch.Tensor] = {}
    frame_reports: list[dict[str, Any]] = []
    track_ready_counts: dict[str, int] = {}
    profile_crop_ready_counts: dict[str, int] = {}
    started = time.perf_counter()
    memory_before = _memory_snapshot()
    detector_report: dict[str, Any] = {}
    model = None
    model_path = None
    model_hash = None
    model_loaded = False
    model_unloaded = False
    status = "ABSTAIN_NOT_EXECUTED"
    detail = ""
    identity_labels: dict[str, str] = {}
    plan: dict | None = None
    progress = _progress_bar(frame_count)

    try:
        if track_plan is None:
            raise _MultiParserUnavailable(
                "ABSTAIN_TRACK_PLAN_MISSING_OR_INVALID",
                "A source-bound SAM3.1 multi-person track plan is required.",
            )
        plan = _validate_track_plan(frames, track_plan)
        identity_labels = _identity_labels(identity_assignment, plan)
        try:
            detections, detector_report = _detect_local_opencv_yunet(
                frames,
                YUNET_2023MAR_RELATIVE,
                float(detection_threshold),
                "cpu",
            )
        except Exception as error:
            raise _MultiParserUnavailable(
                "ABSTAIN_YUNET_DETECTION_UNAVAILABLE",
                f"Pinned YuNet detection failed before ParseNet load: {error}",
            ) from error
        try:
            model, model_path, model_hash = _load_pinned_parsenet()
        except _ParserUnavailable as error:
            raise _MultiParserUnavailable(error.status, error.detail) from error
        model_loaded = True

        for frame_index, candidates in enumerate(detections):
            _interrupt_and_progress(progress, frame_index, frame_count)
            shot = _shot_for_frame(plan, frame_index)
            shot_id = int(shot["shot_id"])
            shot_local = frame_index - int(shot["start_frame"])
            person_masks = [
                _mask_at_source(shot, shot_local, track_index, height, width)
                for track_index in range(int(shot["object_count"]))
            ]
            available = set(range(len(candidates)))
            frame_rgb = _frame_rgb_numpy(frames[frame_index])
            frame_feature = torch.zeros((height, width), dtype=torch.float32)
            per_track: list[dict[str, Any]] = []
            accepted_tracks = 0

            for track_index, person_mask in enumerate(person_masks):
                track_key = str(shot["track_keys"][track_index])
                ranked = []
                for candidate_index in available:
                    candidate = candidates[candidate_index]
                    overlap = _detection_person_overlap(candidate, person_mask)
                    if overlap < float(minimum_person_overlap):
                        continue
                    quality, metrics = _quality_weight(
                        frames[frame_index],
                        candidate,
                        overlap,
                        detection_threshold=float(detection_threshold),
                        minimum_face_height_px=float(minimum_face_height_px),
                        minimum_detail=float(minimum_detail),
                    )
                    face_box = candidate["box"]
                    area = max(0.0, face_box[2] - face_box[0]) * max(
                        0.0, face_box[3] - face_box[1]
                    )
                    ranked.append((quality, area, candidate_index, candidate, metrics))
                ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
                record: dict[str, Any] = {
                    "track_key": track_key,
                    "character_id": identity_labels.get(track_key),
                    "status": "ABSTAIN_NO_RELIABLE_FACE",
                    "reasons": [],
                }
                selected = (
                    ranked[0]
                    if ranked and ranked[0][0] >= float(minimum_track_quality)
                    else None
                )
                if selected is None:
                    record["reasons"].append("no_unique_yunet_face_meets_quality_gate")
                    per_track.append(record)
                    continue

                quality, _, candidate_index, candidate, metrics = selected
                available.remove(candidate_index)
                record["metrics"] = {
                    key: round(float(value), 8) for key, value in metrics.items()
                }
                record["source_face_box_xyxy"] = [
                    round(float(value), 5) for value in candidate["box"]
                ]
                alignment_name = "yunet_five_point_to_ffhq_512_similarity"
                alignment_rms: float | None = None
                alignment_fallback_reason: str | None = None
                profile_crop_box: tuple[int, int, int, int] | None = None
                try:
                    try:
                        aligned, inverse, alignment_rms = _align_face(
                            frame_rgb,
                            candidate,
                            maximum_alignment_rms=float(maximum_alignment_rms),
                        )
                    except _MultiParserUnavailable:
                        raise
                    except ValueError as error:
                        if alignment_policy != "five_point_then_profile_crop":
                            raise
                        alignment_fallback_reason = f"{type(error).__name__}: {error}"
                        aligned, profile_crop_box = _profile_crop_face(
                            frames[frame_index],
                            candidate,
                            expansion=float(profile_crop_expansion),
                        )
                        inverse = None
                        alignment_name = "profile_bbox_crop_fallback"
                    logits = _parser_logits(model, aligned)
                    local_skin, local_feature, semantic_stats = _semantic_local_masks(
                        logits,
                        include_neck=bool(include_neck),
                        minimum_class_probability=float(minimum_class_probability),
                        feature_protection_px=int(feature_protection_px),
                    )
                    if profile_crop_box is None:
                        full_skin = _warp_aligned_mask(
                            local_skin[0],
                            inverse,
                            width=width,
                            height=height,
                            nearest=False,
                        )
                        full_feature = _warp_aligned_mask(
                            local_feature[0],
                            inverse,
                            width=width,
                            height=height,
                            nearest=True,
                        )
                    else:
                        full_skin = _project_profile_crop_mask(
                            local_skin[0],
                            profile_crop_box,
                            width=width,
                            height=height,
                            nearest=False,
                        )
                        full_feature = _project_profile_crop_mask(
                            local_feature[0],
                            profile_crop_box,
                            width=width,
                            height=height,
                            nearest=True,
                        )
                except _MultiParserUnavailable:
                    raise
                except Exception as error:
                    record["status"] = "ABSTAIN_ALIGNMENT_OR_PARSE_FAILED"
                    record["reasons"].append(f"{type(error).__name__}: {error}")
                    per_track.append(record)
                    continue

                full_skin.mul_(person_mask.float()).mul_(float(quality)).clamp_(0.0, 1.0)
                full_feature.mul_(person_mask.float()).clamp_(0.0, 1.0)
                skin_area = float((full_skin > 0.05).sum()) / float(height * width)
                if skin_area < float(minimum_skin_area_per_face):
                    record["status"] = "ABSTAIN_SEMANTIC_SKIN_AREA_TOO_SMALL"
                    record["reasons"].append("semantic_skin_area_below_minimum")
                    per_track.append(record)
                    continue
                proposed = torch.maximum(combined_mask[frame_index], full_skin)
                combined_area = float((proposed > 0.05).sum()) / float(height * width)
                if combined_area > float(maximum_skin_area_per_frame):
                    record["status"] = "ABSTAIN_COMBINED_SKIN_AREA_TOO_LARGE"
                    record["reasons"].append("combined_semantic_skin_area_above_maximum")
                    per_track.append(record)
                    continue
                combined_mask[frame_index] = proposed
                frame_feature = torch.maximum(frame_feature, full_feature)
                accepted_tracks += 1
                track_ready_counts[track_key] = track_ready_counts.get(track_key, 0) + 1
                ready_record: dict[str, Any] = {
                    "status": "READY",
                    "alignment": alignment_name,
                    "skin_area_fraction": round(skin_area, 8),
                    "semantic_stats": semantic_stats,
                }
                if alignment_rms is not None:
                    ready_record["alignment_normalized_rms"] = round(alignment_rms, 8)
                if profile_crop_box is not None:
                    profile_crop_ready_counts[track_key] = (
                        profile_crop_ready_counts.get(track_key, 0) + 1
                    )
                    ready_record["alignment_fallback_reason"] = alignment_fallback_reason
                    ready_record["profile_crop_box_xyxy"] = list(profile_crop_box)
                    ready_record["profile_crop_expansion"] = float(
                        profile_crop_expansion
                    )
                record.update(ready_record)
                per_track.append(record)

            frame_area = float((combined_mask[frame_index] > 0.05).sum()) / float(
                height * width
            )
            if frame_index in preview_indices:
                feature_previews[frame_index] = frame_feature
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "shot_id": shot_id,
                    "detected_face_count": len(candidates),
                    "accepted_track_count": accepted_tracks,
                    "combined_skin_area_fraction": round(frame_area, 8),
                    "tracks": per_track,
                }
            )

        ready_frames = [
            item["frame_index"]
            for item in frame_reports
            if item["accepted_track_count"] > 0
        ]
        ready_fraction = len(ready_frames) / max(1, frame_count)
        if not ready_frames:
            status = "ABSTAIN_NO_RELIABLE_MULTIFACE_SEMANTIC_SKIN"
        elif ready_fraction < float(minimum_ready_frame_fraction):
            status = "ABSTAIN_READY_FRAME_FRACTION_BELOW_MINIMUM"
            detail = (
                f"Only {len(ready_frames)}/{frame_count} frames had a reliable semantic face; "
                f"minimum fraction is {float(minimum_ready_frame_fraction):.4f}."
            )
        else:
            status = "READY"
        _interrupt_and_progress(progress, frame_count, frame_count)
    except _MultiParserUnavailable as error:
        status = error.status
        detail = error.detail
    except Exception as error:
        status = "ABSTAIN_MULTIFACE_SEMANTIC_INFERENCE_FAILED"
        detail = f"{type(error).__name__}: {error}"
    finally:
        if model is not None:
            try:
                model.to(device="cpu")
            except Exception:
                pass
            del model
            model = None
            model_unloaded = True
        gc.collect()

    observed_ready_frame_indices = [
        item["frame_index"]
        for item in frame_reports
        if item["accepted_track_count"] > 0
    ]
    if status != "READY":
        combined_mask.zero_()
        feature_previews.clear()
    preview = _build_preview(frames, combined_mask, feature_previews, preview_indices)
    ready_frame_indices = (
        observed_ready_frame_indices
        if status == "READY"
        else []
    )
    report = {
        "schema": SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
        "status": status,
        "detail": detail,
        "source": source,
        "track_plan_sha256": str((plan or {}).get("sha256", "")),
        # Bind the report to the actual mask tensor used by downstream per-person
        # routing.  The source and track plan already use the project's bounded
        # proxy-digest contract; keeping the mask on the same contract avoids an
        # unbounded second copy of a long full-resolution float mask.
        "mask_proxy_sha256": _tensor_proxy_sha256(combined_mask),
        "identity_assignment": {
            "connected": identity_assignment is not None,
            "labels_are_suggestions_not_identity_proof": True,
            "mapped_track_count": len(identity_labels),
            "track_to_character": identity_labels,
        },
        "detector": detector_report,
        "parser": {
            "name": PARSENET_MODEL_NAME,
            "path": str(model_path) if model_path else str(PARSENET_MODEL_RELATIVE),
            "expected_size": PARSENET_MODEL_SIZE,
            "expected_sha256": PARSENET_MODEL_SHA256,
            "actual_sha256": model_hash,
            "loaded": model_loaded,
            "unloaded_after_execute": model_unloaded,
            "device": "cpu",
            "persistent_cache": False,
            "runtime_download": False,
        },
        "alignment": {
            "method": "OpenCV estimateAffinePartial2D LMEDS",
            "policy": str(alignment_policy),
            "source": "YuNet five landmarks",
            "target": "FaceXLib standard FFHQ five-point 512 template",
            "eye_and_mouth_points_sorted_by_image_x": True,
            "maximum_normalized_rms": float(maximum_alignment_rms),
            "profile_crop_expansion": float(profile_crop_expansion),
            "profile_crop_fallback_is_not_frontalization": True,
            "gap_landmarks_propagated": False,
        },
        "class_mapping": {
            str(index): name for index, name in enumerate(PARSENET_CLASS_NAMES)
        },
        "selection": {
            "ready_frame_indices": ready_frame_indices,
            "ready_frame_fraction": round(
                len(ready_frame_indices) / max(1, frame_count), 8
            ),
            "observed_ready_frame_indices_before_global_gate": (
                observed_ready_frame_indices
            ),
            "observed_ready_frame_fraction_before_global_gate": round(
                len(observed_ready_frame_indices) / max(1, frame_count), 8
            ),
            "track_ready_counts": track_ready_counts,
            "profile_crop_ready_counts": profile_crop_ready_counts,
            "per_frame_identity_scope": "shot_local_sam31_track",
            "cross_shot_character_labels": bool(identity_labels),
            "masks_intersected_with_their_person_track": True,
            "one_yunet_detection_cannot_be_reused_by_two_tracks": True,
            "automatic_accept": False,
        },
        "parameters": {
            "detection_threshold": float(detection_threshold),
            "minimum_face_height_px": float(minimum_face_height_px),
            "minimum_detail": float(minimum_detail),
            "minimum_person_overlap": float(minimum_person_overlap),
            "minimum_track_quality": float(minimum_track_quality),
            "minimum_class_probability": float(minimum_class_probability),
            "feature_protection_px": int(feature_protection_px),
            "include_neck": bool(include_neck),
            "minimum_skin_area_per_face": float(minimum_skin_area_per_face),
            "maximum_skin_area_per_frame": float(maximum_skin_area_per_frame),
            "alignment_policy": str(alignment_policy),
            "profile_crop_expansion": float(profile_crop_expansion),
            "minimum_ready_frame_fraction": float(minimum_ready_frame_fraction),
        },
        "frames": frame_reports,
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "product_boundary": (
            "Five-point alignment, optional profile crop fallback and per-track intersection "
            "reduce geometric and cross-person mask errors; the crop fallback does not "
            "frontalize a profile or prove its landmarks. They do not prove identity, skin "
            "quality, deblur, face reconstruction, cross-shot continuity or universal "
            "multi-person safety. Empty or partial evidence fails closed and requires human review."
        ),
    }
    return combined_mask, preview, _report_json(report)
