from __future__ import annotations

import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

import comfy.model_management

from .core import align_frame_count
from .face_refine_advanced import (
    YUNET_2023MAR_RELATIVE,
    _boxes_for_crop_context,
    _crop_chunks,
    _detect_local_opencv_yunet,
    _draw_preview,
    _model_root,
    _scene_ranges,
    _validate_frames,
    canonical_json,
    source_proxy_sha256,
)
from .face_refine_parity_advanced import (
    ACTUAL_DETECTION_STATES,
    PARITY_PLAN_SCHEMA,
    _face_box_from_center,
    _fit_square_crop,
    _resolve_parity_canvas,
    _smooth_face_trajectory,
    _upstream_parity_face_box_in_crop,
    _validate_parity_plan,
)


CHARACTER_PROFILE_SCHEMA = "h3_t8_multiface_character_profile/v1"
FACE_CAST_SCHEMA = "h3_t8_multiface_cast/v1"
TRACK_PLAN_SCHEMA = "h3_t8_sam31_multiface_track_plan/v1"
ASSIGNMENT_SCHEMA = "h3_t8_multiface_identity_assignment/v1"
COMPOSITE_SCHEMA = "h3_t8_multiface_composite/v1"

SAM31_EXPECTED_SHA256 = "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03"
SFACE_EXPECTED_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_MODEL_NAME = "face_recognition_sface_2021dec.onnx"

COLORS = (
    (0.12, 0.47, 0.71),
    (1.0, 0.50, 0.05),
    (0.17, 0.63, 0.17),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_hashed_dict(value: dict, schema: str, name: str) -> dict:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{name} must use {schema}")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _hash_json(_json_safe(unsigned)):
        raise ValueError(f"{name} hash mismatch; the object may be stale or modified")
    return value


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
            "tensor_sha256": _tensor_sha256(value),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _source_contract(frames: torch.Tensor) -> dict:
    count, height, width = _validate_frames(frames)
    return {
        "frame_count": count,
        "height": height,
        "width": width,
        "proxy_sha256": source_proxy_sha256(frames),
    }


def _sface_path() -> Path:
    path = _model_root() / "face_detection" / SFACE_MODEL_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing OpenCV Zoo SFace model: {path}. Install {SFACE_MODEL_NAME} in "
            "ComfyUI/models/face_detection."
        )
    return path


def _create_sface_recognizer():
    try:
        import cv2
    except Exception as error:
        raise RuntimeError("SFace identity suggestions require OpenCV") from error
    if not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError("This OpenCV build does not provide FaceRecognizerSF")
    path = _sface_path()
    return cv2.FaceRecognizerSF.create(str(path), "")


def _face_row(detection: dict) -> np.ndarray:
    landmarks = detection.get("landmarks_xy")
    if not isinstance(landmarks, list) or len(landmarks) != 5:
        raise ValueError("SFace requires five YuNet landmarks")
    left, top, right, bottom = [float(value) for value in detection["box"]]
    values = [left, top, right - left, bottom - top]
    for point in landmarks:
        values.extend([float(point[0]), float(point[1])])
    values.append(float(detection.get("confidence", 1.0)))
    return np.asarray(values, dtype=np.float32)


def _frame_bgr(frame: torch.Tensor) -> np.ndarray:
    rgb = (
        frame[..., :3]
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
    )
    return rgb[..., ::-1].copy()


def _sface_feature(frame: torch.Tensor, detection: dict, recognizer) -> torch.Tensor:
    aligned = recognizer.alignCrop(_frame_bgr(frame), _face_row(detection))
    feature = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("SFace returned a non-finite or zero identity feature")
    return torch.from_numpy(feature / norm)


def _mean_normalized(features: list[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("No usable identity features were found")
    value = torch.stack([item.float().cpu() for item in features]).mean(dim=0)
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-8:
        raise ValueError("Identity feature mean is non-finite or zero")
    return value / norm


def _face_area(detection: dict) -> float:
    left, top, right, bottom = [float(value) for value in detection["box"]]
    return max(0.0, right - left) * max(0.0, bottom - top)


def _select_reference_face(
    candidates: list[dict], reference_face_policy: str, image_index: int
) -> tuple[dict, dict]:
    if not candidates:
        raise ValueError(f"reference image {image_index} contains no YuNet face")
    ordered = sorted(candidates, key=_face_area, reverse=True)
    selected = ordered[0]
    selection = {
        "image_index": int(image_index),
        "policy": str(reference_face_policy),
        "detected_face_count": len(ordered),
        "selected_area_px2": _face_area(selected),
        "selected_confidence": float(selected.get("confidence", 0.0)),
        "dominant_face_auto_passed": len(ordered) == 1,
        "area_ratio_to_runner_up": None,
        "confidence_margin_to_runner_up": None,
    }
    if len(ordered) == 1:
        return selected, selection
    if reference_face_policy == "require_single_face":
        raise ValueError(
            f"reference image {image_index} contains {len(ordered)} YuNet detections; "
            "use dominant_face_auto for a clearly dominant person, crop the image, or "
            "select largest_face_exp"
        )
    if reference_face_policy == "largest_face_exp":
        return selected, selection

    runner_up = ordered[1]
    area_ratio = _face_area(selected) / max(_face_area(runner_up), 1e-8)
    confidence_margin = float(selected.get("confidence", 0.0)) - float(
        runner_up.get("confidence", 0.0)
    )
    selection["area_ratio_to_runner_up"] = area_ratio
    selection["confidence_margin_to_runner_up"] = confidence_margin
    selection["dominant_face_auto_passed"] = bool(
        area_ratio >= 1.8
        and confidence_margin >= 0.20
        and float(selected.get("confidence", 0.0)) >= 0.60
    )
    if not selection["dominant_face_auto_passed"]:
        raise ValueError(
            f"reference image {image_index} has {len(ordered)} ambiguous YuNet detections "
            f"(largest/runner-up area={area_ratio:.3f}, confidence margin="
            f"{confidence_margin:.3f}); crop to one character or explicitly select "
            "largest_face_exp"
        )
    return selected, selection


def build_multiface_character_profile(
    character_id: str,
    reference_images: torch.Tensor,
    rights_confirmed: bool | None = None,
    reference_face_policy: str = "dominant_face_auto",
):
    # Kept as an ignored compatibility argument for older direct callers and API prompts.
    # Current nodes do not expose or enforce an authorization-state widget.
    del rights_confirmed
    character_id = str(character_id).strip()
    if not character_id or len(character_id) > 64:
        raise ValueError("character_id must contain 1-64 visible characters")
    if reference_face_policy not in {
        "dominant_face_auto",
        "require_single_face",
        "largest_face_exp",
    }:
        raise ValueError(f"Unknown reference_face_policy: {reference_face_policy}")
    frame_count, _, _ = _validate_frames(reference_images, name="reference_images")
    detections, detector_report = _detect_local_opencv_yunet(
        reference_images,
        YUNET_2023MAR_RELATIVE,
        0.35,
        "cpu",
    )
    recognizer_path = _sface_path()
    recognizer_hash = _file_sha256(recognizer_path)
    recognizer = _create_sface_recognizer()
    features: list[torch.Tensor] = []
    selected_boxes = []
    selections = []
    try:
        for index, candidates in enumerate(detections):
            selected, selection = _select_reference_face(
                candidates, reference_face_policy, index
            )
            features.append(_sface_feature(reference_images[index], selected, recognizer))
            selected_boxes.append([round(float(value), 4) for value in selected["box"]])
            selections.append(selection)
    finally:
        del recognizer
        gc.collect()

    embedding = _mean_normalized(features)
    source = _source_contract(reference_images)
    profile = {
        "schema": CHARACTER_PROFILE_SCHEMA,
        "character_id": character_id,
        "reference_images": reference_images.detach(),
        "reference_source": source,
        "reference_face_boxes": selected_boxes,
        "reference_face_policy": reference_face_policy,
        "reference_face_selections": selections,
        "identity_embedding": embedding,
        "identity_backend": {
            "detector": "opencv_zoo_yunet_2023mar",
            "recognizer": "opencv_zoo_sface_2021dec",
            "recognizer_sha256": recognizer_hash,
            "official_opencv_zoo_match": recognizer_hash == SFACE_EXPECTED_SHA256,
            "model_identity_policy": "diagnostic_only_not_a_load_gate",
            "device": "cpu",
            "license": "Apache-2.0",
            "identity_is_suggestion_not_proof": True,
        },
    }
    profile["sha256"] = _hash_json(_json_safe(profile))
    report = {
        "schema": "h3_t8_multiface_character_profile_report/v1",
        "status": "in_memory_profile_ready",
        "character_id": character_id,
        "reference_count": frame_count,
        "reference_proxy_sha256": source["proxy_sha256"],
        "profile_sha256": profile["sha256"],
        "reference_face_policy": reference_face_policy,
        "reference_face_selections": selections,
        "detector": detector_report,
        "recognizer_model_sha256": recognizer_hash,
        "recognizer_official_match": recognizer_hash == SFACE_EXPECTED_SHA256,
        "persistent_biometric_storage": False,
        "identity_is_suggestion_not_proof": True,
    }
    return profile, reference_images[:1], canonical_json(report)


def merge_multiface_cast(profile: dict, previous_cast: dict | None = None):
    _validate_hashed_dict(profile, CHARACTER_PROFILE_SCHEMA, "profile")
    profiles = []
    if previous_cast is not None:
        cast = _validate_hashed_dict(previous_cast, FACE_CAST_SCHEMA, "previous_cast")
        profiles.extend(cast["profiles"])
    if len(profiles) >= 3:
        raise ValueError("Multi-person Face Refine supports at most three characters")
    ids = {item["character_id"] for item in profiles}
    if profile["character_id"] in ids:
        raise ValueError(f"Duplicate character_id: {profile['character_id']}")
    profiles.append(profile)
    cast = {
        "schema": FACE_CAST_SCHEMA,
        "profiles": profiles,
        "character_ids": [item["character_id"] for item in profiles],
        "maximum_characters": 3,
        "identity_backend": "opencv_zoo_sface_cpu",
    }
    cast["sha256"] = _hash_json(_json_safe(cast))

    previews = []
    for item in profiles:
        image = item["reference_images"][:1].movedim(-1, 1)
        image = torch_functional.interpolate(
            image.float(), size=(256, 256), mode="bilinear", align_corners=False
        ).movedim(1, -1)
        previews.append(image)
    contact_sheet = torch.cat(previews, dim=0)
    report = {
        "schema": "h3_t8_multiface_cast_report/v1",
        "status": "cast_ready",
        "cast_sha256": cast["sha256"],
        "character_ids": cast["character_ids"],
        "character_count": len(profiles),
        "maximum_characters": 3,
        "persistent_biometric_storage": False,
    }
    return cast, contact_sheet, canonical_json(report)


def _resize_for_analysis(frames: torch.Tensor, maximum_side: int) -> torch.Tensor:
    _, height, width = _validate_frames(frames)
    if maximum_side <= 0 or max(height, width) <= maximum_side:
        return frames
    scale = float(maximum_side) / float(max(height, width))
    out_h = max(32, int(round(height * scale)))
    out_w = max(32, int(round(width * scale)))
    return torch_functional.interpolate(
        frames[..., :3].movedim(-1, 1),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)


def _unpack_mask(packed: torch.Tensor) -> torch.Tensor:
    from comfy.ldm.sam3.tracker import unpack_masks

    return unpack_masks(packed).bool()


def _mask_stats(packed: torch.Tensor) -> list[dict]:
    frame_count, object_count = packed.shape[:2]
    output = []
    for object_index in range(object_count):
        areas = []
        centers = []
        for frame_index in range(frame_count):
            mask = _unpack_mask(packed[frame_index, object_index])
            area = int(mask.sum().item())
            areas.append(area)
            if area:
                coords = mask.nonzero(as_tuple=False)
                centers.append((frame_index, float(coords[:, 1].float().mean())))
        first = next((index for index, area in enumerate(areas) if area > 0), None)
        output.append(
            {
                "native_object_index": object_index,
                "active_frames": sum(area > 0 for area in areas),
                "first_active_frame": first,
                "median_area": float(np.median([area for area in areas if area] or [0])),
                "first_center_x": centers[0][1] if centers else float("inf"),
            }
        )
    return output


def _whole_device_snapshot() -> dict:
    snapshot = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            snapshot.update(
                {
                    "device_free_mib": round(float(free) / 2**20, 3),
                    "device_total_mib": round(float(total) / 2**20, 3),
                    "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
                    "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
                }
            )
        except Exception as error:
            snapshot["error"] = f"{type(error).__name__}: {error}"
    return snapshot


def _selectively_unload_model(model) -> dict:
    before = _whole_device_snapshot()
    before_loaded = len(comfy.model_management.loaded_models())
    comfy.model_management.unload_model_and_clones(
        model, unload_additional_models=True, all_devices=False
    )
    gc.collect()
    comfy.model_management.soft_empty_cache()
    after_loaded = len(comfy.model_management.loaded_models())
    return {
        "policy": "offload_sam31_after_track",
        "scope": "selected_model_and_clones",
        "global_unload_called": False,
        "loaded_model_count_before": before_loaded,
        "loaded_model_count_after": after_loaded,
        "before": before,
        "after": _whole_device_snapshot(),
    }


def _assert_native_sam31_capability(model) -> dict:
    try:
        diffusion_model = model.model.diffusion_model
        tracker = diffusion_model.tracker
    except AttributeError as error:
        raise ValueError(
            "The connected MODEL does not expose ComfyUI's native SAM3.1 tracker contract"
        ) from error

    tracker_class = type(tracker).__name__
    has_multiplex_tracking = callable(getattr(tracker, "track_video_with_detection", None))
    has_video_entrypoint = callable(getattr(diffusion_model, "forward_video", None))
    if tracker_class != "SAM31Tracker" or not has_multiplex_tracking or not has_video_entrypoint:
        raise ValueError(
            "This node requires the current ComfyUI native SAM3.1 multiplex checkpoint "
            f"(SAM31Tracker); received {tracker_class or 'unknown'}"
        )
    return {
        "tracker_class": tracker_class,
        "native_forward_video": True,
        "multiplex_text_detection": True,
        "contract_check": "class_and_callable_behavior_probe",
    }


def _run_native_track(images, model, conditioning, detection_threshold, max_people, detect_interval):
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack

    _assert_native_sam31_capability(model)
    output = SAM3_VideoTrack.execute(
        images=images,
        model=model,
        conditioning=conditioning,
        detection_threshold=float(detection_threshold),
        max_objects=int(max_people),
        detect_interval=int(detect_interval),
    )
    return output[0]


def _preview_track_plan(
    frames: torch.Tensor,
    plan: dict,
    preview_stride: int,
    identity_labels: dict[str, str | None] | None = None,
) -> torch.Tensor:
    import cv2

    frame_count = int(frames.shape[0])
    indices = sorted(set(range(0, frame_count, max(1, int(preview_stride)))) | {frame_count - 1})
    previews = []
    for global_index in indices:
        shot = next(
            item
            for item in plan["shots"]
            if int(item["start_frame"]) <= global_index <= int(item["end_frame"])
        )
        local_index = global_index - int(shot["start_frame"])
        frame = frames[global_index, ..., :3].detach().float().cpu().clone()
        labels = []
        for object_index in range(int(shot["object_count"])):
            mask = _unpack_mask(shot["packed_masks"][local_index, object_index]).float()
            mask = torch_functional.interpolate(
                mask[None, None], size=frame.shape[:2], mode="nearest"
            )[0, 0] > 0.5
            if not bool(mask.any()):
                continue
            color = torch.tensor(COLORS[object_index], dtype=frame.dtype)
            frame[mask] = frame[mask] * 0.55 + color * 0.45
            coords = mask.nonzero(as_tuple=False)
            track_key = f"{shot['shot_id']}:{object_index}"
            character = (identity_labels or {}).get(track_key)
            display_label = (
                f"{character} [{track_key}]" if character else f"S{shot['shot_id']}:T{object_index}"
            )
            labels.append(
                (
                    int(coords[:, 1].float().mean()),
                    int(coords[:, 0].float().mean()),
                    display_label,
                    tuple(int(value * 255) for value in COLORS[object_index]),
                )
            )
        image = frame.clamp(0.0, 1.0).mul(255.0).byte().numpy()
        for x, y, label, color in labels:
            cv2.putText(
                image,
                label,
                (max(0, x - 24), max(16, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        previews.append(torch.from_numpy(image).float() / 255.0)
    return torch.stack(previews)


def build_sam31_multiface_track_plan(
    frames: torch.Tensor,
    model,
    conditioning,
    fps: float,
    maximum_people: int,
    detection_threshold: float,
    detect_interval: int,
    scene_cut_threshold: float,
    analysis_max_side: int,
    preview_stride: int,
    release_policy: str,
):
    frame_count, height, width = _validate_frames(frames)
    if int(maximum_people) not in (2, 3):
        raise ValueError("maximum_people must be 2 or 3")
    if conditioning is None or len(conditioning) == 0:
        raise ValueError("SAM3.1 multi-person tracking requires text conditioning such as 'person'")
    if release_policy not in {"offload_sam31_after_track", "keep_loaded"}:
        raise ValueError(f"Unknown release_policy: {release_policy}")
    if abs(float(fps) - 24.0) > 0.01:
        raise ValueError(f"Multi-person Face Refine currently requires exact 24fps; got {fps}")

    analysis_frames = _resize_for_analysis(frames, int(analysis_max_side))
    _, analysis_h, analysis_w = _validate_frames(analysis_frames, name="analysis_frames")
    shot_ranges, scene_deltas = _scene_ranges(frames, float(scene_cut_threshold))
    shots = []
    release_report = {"policy": release_policy, "performed": False}
    native_capability = None
    try:
        native_capability = _assert_native_sam31_capability(model)
        for shot_id, (start, end) in enumerate(shot_ranges):
            result = _run_native_track(
                analysis_frames[start : end + 1],
                model,
                conditioning,
                detection_threshold,
                maximum_people,
                detect_interval,
            )
            packed = result.get("packed_masks")
            if packed is None:
                raise ValueError(f"SAM3.1 detected no people in shot {shot_id}")
            packed = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            stats = _mask_stats(packed)
            active = [item for item in stats if item["active_frames"] > 0]
            if not active:
                raise ValueError(f"SAM3.1 returned only empty tracks in shot {shot_id}")
            order = [
                item["native_object_index"]
                for item in sorted(
                    active,
                    key=lambda item: (
                        item["first_active_frame"] if item["first_active_frame"] is not None else 10**9,
                        item["first_center_x"],
                    ),
                )
            ][: int(maximum_people)]
            packed = packed[:, order]
            ordered_stats = [stats[index] for index in order]
            scores = list(result.get("scores", []))
            shots.append(
                {
                    "shot_id": shot_id,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": end - start + 1,
                    "object_count": len(order),
                    "track_keys": [f"{shot_id}:{index}" for index in range(len(order))],
                    "native_object_indices": order,
                    "scores": [float(scores[index]) if index < len(scores) else None for index in order],
                    "stats": ordered_stats,
                    "packed_masks": packed,
                    "packed_masks_sha256": _tensor_sha256(packed),
                    "mask_size": [int(packed.shape[-2]), int(packed.shape[-1] * 8)],
                }
            )
    finally:
        if release_policy == "offload_sam31_after_track":
            release_report = _selectively_unload_model(model)
            release_report["performed"] = True

    source = {
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "fps": float(fps),
        "proxy_sha256": source_proxy_sha256(frames),
    }
    plan = {
        "schema": TRACK_PLAN_SCHEMA,
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {
            "height": analysis_h,
            "width": analysis_w,
            "requested_max_side": int(analysis_max_side),
            "sam_internal_square_size": 1008,
            "input_downscale_does_not_reduce_fixed_sam_backbone_size": True,
        },
        "sam31": {
            "expected_checkpoint_sha256": SAM31_EXPECTED_SHA256,
            "capability": "multiplex_video_tracking_with_text_detection",
            "native_capability_probe": native_capability,
            "maximum_people": int(maximum_people),
            "detection_threshold": float(detection_threshold),
            "detect_interval": int(detect_interval),
            "track_identity_scope": "shot_local_only",
        },
        "shots": shots,
        "scene_cut_threshold": float(scene_cut_threshold),
        "scene_cut_count": len(shots) - 1,
        "max_scene_delta": max(scene_deltas),
        "release": release_report,
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    preview = _preview_track_plan(frames, plan, preview_stride)
    report = {
        "schema": "h3_t8_sam31_multiface_track_report/v1",
        "status": plan["status"],
        "track_plan_sha256": plan["sha256"],
        "source": source,
        "shot_count": len(shots),
        "objects_per_shot": [item["object_count"] for item in shots],
        "track_keys": [key for item in shots for key in item["track_keys"]],
        "analysis": plan["analysis"],
        "release": release_report,
        "warning": (
            "SAM colors and indices are shot-local tracks, not character identity. "
            "Use the identity assignment node and review every shot."
        ),
    }
    return plan, preview, canonical_json(report), len(shots), sum(item["object_count"] for item in shots)


def _parse_overrides(overrides_json: str, valid_characters: set[str]) -> dict[str, str]:
    text = str(overrides_json or "").strip()
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"manual_assignments_json is invalid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("manual_assignments_json must be an object like {'0:0':'Alice'}")
    output = {}
    for key, value in raw.items():
        key = str(key).strip()
        value = str(value).strip()
        parts = key.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid track key {key!r}; expected shot:track, for example 0:1")
        if value not in valid_characters:
            raise ValueError(f"Unknown character_id in manual assignment: {value!r}")
        output[key] = value
    return output


def _mask_at_source(shot: dict, local_frame: int, track_index: int, height: int, width: int):
    mask = _unpack_mask(shot["packed_masks"][local_frame, track_index]).float()
    return torch_functional.interpolate(
        mask[None, None], size=(height, width), mode="nearest"
    )[0, 0] > 0.5


def _detection_inside_mask(detection: dict, mask: torch.Tensor) -> bool:
    left, top, right, bottom = detection["box"]
    x = max(0, min(mask.shape[1] - 1, int(round((left + right) * 0.5))))
    y = max(0, min(mask.shape[0] - 1, int(round((top + bottom) * 0.5))))
    if bool(mask[y, x]):
        return True
    x1 = max(0, min(mask.shape[1] - 1, int(left)))
    y1 = max(0, min(mask.shape[0] - 1, int(top)))
    x2 = max(x1 + 1, min(mask.shape[1], int(np.ceil(right))))
    y2 = max(y1 + 1, min(mask.shape[0], int(np.ceil(bottom))))
    return float(mask[y1:y2, x1:x2].float().mean()) >= 0.20


def _track_identity_embedding(
    frames: torch.Tensor,
    shot: dict,
    track_index: int,
    samples: int,
    recognizer,
) -> tuple[torch.Tensor | None, list[int], list[list[float]]]:
    _, height, width = _validate_frames(frames)
    areas = []
    for local_index in range(int(shot["frame_count"])):
        mask = _unpack_mask(shot["packed_masks"][local_index, track_index])
        areas.append((int(mask.sum().item()), local_index))
    selected = [index for area, index in sorted(areas, reverse=True) if area > 0][: max(1, int(samples))]
    if not selected:
        return None, [], []
    batch = torch.stack([frames[int(shot["start_frame"]) + index] for index in selected])
    detections, _ = _detect_local_opencv_yunet(batch, YUNET_2023MAR_RELATIVE, 0.25, "cpu")
    features = []
    boxes = []
    used = []
    for batch_index, local_index in enumerate(selected):
        mask = _mask_at_source(shot, local_index, track_index, height, width)
        candidates = [item for item in detections[batch_index] if _detection_inside_mask(item, mask)]
        if not candidates:
            continue
        candidate = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
        features.append(_sface_feature(batch[batch_index], candidate, recognizer))
        boxes.append([round(float(value), 4) for value in candidate["box"]])
        used.append(int(shot["start_frame"]) + local_index)
    if not features:
        return None, [], []
    return _mean_normalized(features), used, boxes


def _best_one_to_one(scores: dict[str, dict[str, float]], tracks: list[str], characters: list[str]):
    if len(tracks) > len(characters):
        return {}
    best_total = -float("inf")
    best = {}
    for chosen in itertools.permutations(characters, len(tracks)):
        total = sum(scores[track][character] for track, character in zip(tracks, chosen, strict=True))
        if total > best_total:
            best_total = total
            best = dict(zip(tracks, chosen, strict=True))
    return best


def assign_multiface_identities(
    frames: torch.Tensor,
    track_plan: dict,
    face_cast: dict,
    identity_mode: str,
    manual_assignments_json: str,
    minimum_similarity: float,
    minimum_margin: float,
    identity_samples_per_track: int,
    strict_identity: bool,
    preview_stride: int,
):
    track_plan = _validate_hashed_dict(track_plan, TRACK_PLAN_SCHEMA, "track_plan")
    face_cast = _validate_hashed_dict(face_cast, FACE_CAST_SCHEMA, "face_cast")
    source = _source_contract(frames)
    if source["proxy_sha256"] != track_plan["source"]["proxy_sha256"]:
        raise ValueError("frames do not match the SAM3.1 track plan source")
    if identity_mode not in {"manual_only", "sface_cpu_suggest"}:
        raise ValueError(f"Unknown identity_mode: {identity_mode}")
    character_ids = list(face_cast["character_ids"])
    overrides = _parse_overrides(manual_assignments_json, set(character_ids))
    valid_track_keys = {key for shot in track_plan["shots"] for key in shot["track_keys"]}
    unknown_keys = sorted(set(overrides) - valid_track_keys)
    if unknown_keys:
        raise ValueError(f"Manual assignments reference unknown tracks: {unknown_keys}")

    profiles = {item["character_id"]: item for item in face_cast["profiles"]}
    mappings = []
    similarity_matrix: dict[str, dict[str, float]] = {}
    track_evidence = {}
    recognizer = None
    try:
        if identity_mode == "sface_cpu_suggest":
            recognizer = _create_sface_recognizer()
        for shot in track_plan["shots"]:
            shot_keys = list(shot["track_keys"])
            assigned = {key: overrides[key] for key in shot_keys if key in overrides}
            if len(set(assigned.values())) != len(assigned):
                raise ValueError(f"A character cannot occupy two tracks in shot {shot['shot_id']}")
            unresolved = [key for key in shot_keys if key not in assigned]
            available = [item for item in character_ids if item not in set(assigned.values())]

            if unresolved and identity_mode == "sface_cpu_suggest":
                for key in unresolved:
                    track_index = int(key.split(":")[1])
                    embedding, sample_frames, face_boxes = _track_identity_embedding(
                        frames,
                        shot,
                        track_index,
                        identity_samples_per_track,
                        recognizer,
                    )
                    track_evidence[key] = {
                        "sample_frames": sample_frames,
                        "face_boxes": face_boxes,
                        "embedding_available": embedding is not None,
                    }
                    if embedding is None:
                        similarity_matrix[key] = {character: -1.0 for character in character_ids}
                    else:
                        similarity_matrix[key] = {
                            character: float(torch.dot(embedding, profiles[character]["identity_embedding"]))
                            for character in character_ids
                        }
                proposal = _best_one_to_one(similarity_matrix, unresolved, available)
                for key, character in proposal.items():
                    score = similarity_matrix[key][character]
                    alternatives = sorted(
                        (value for name, value in similarity_matrix[key].items() if name != character),
                        reverse=True,
                    )
                    margin = score - (alternatives[0] if alternatives else -1.0)
                    if score >= float(minimum_similarity) and margin >= float(minimum_margin):
                        assigned[key] = character

            unresolved = [key for key in shot_keys if key not in assigned]
            if unresolved and strict_identity:
                raise ValueError(
                    "Identity assignment is unresolved for tracks "
                    f"{unresolved}. Review the colored preview and add manual JSON overrides."
                )
            for key in shot_keys:
                character = assigned.get(key)
                matrix = similarity_matrix.get(key, {})
                score = matrix.get(character) if character is not None else None
                values = sorted(matrix.values(), reverse=True)
                margin = values[0] - values[1] if len(values) > 1 else None
                mappings.append(
                    {
                        "track_key": key,
                        "shot_id": int(key.split(":")[0]),
                        "track_index": int(key.split(":")[1]),
                        "character_id": character,
                        "source": "manual_override" if key in overrides else (
                            "sface_cpu_suggestion" if character is not None else "unassigned"
                        ),
                        "similarity": score,
                        "margin": margin,
                    }
                )
    finally:
        del recognizer
        gc.collect()

    assignment = {
        "schema": ASSIGNMENT_SCHEMA,
        "status": "identity_assignment_ready",
        "source": track_plan["source"],
        "track_plan_sha256": track_plan["sha256"],
        "cast_sha256": face_cast["sha256"],
        "track_plan": track_plan,
        "face_cast": face_cast,
        "mappings": mappings,
        "track_evidence": track_evidence,
        "identity_mode": identity_mode,
        "minimum_similarity": float(minimum_similarity),
        "minimum_margin": float(minimum_margin),
        "strict_identity": bool(strict_identity),
        "identity_is_suggestion_not_proof": True,
        "automatic_accept": False,
    }
    assignment["sha256"] = _hash_json(_json_safe(assignment))
    mapped_ids = {item["track_key"]: item["character_id"] for item in mappings}
    preview_plan = dict(track_plan)
    preview = _preview_track_plan(frames, preview_plan, preview_stride, mapped_ids)
    report = {
        "schema": "h3_t8_multiface_identity_assignment_report/v1",
        "status": assignment["status"],
        "assignment_sha256": assignment["sha256"],
        "identity_mode": identity_mode,
        "mappings": mappings,
        "mapped_track_count": sum(value is not None for value in mapped_ids.values()),
        "unassigned_track_count": sum(value is None for value in mapped_ids.values()),
        "track_evidence": track_evidence,
        "manual_overrides": overrides,
        "persistent_biometric_storage": False,
        "warning": "SFace similarity is a CPU matching aid, not legal or biometric identity proof.",
    }
    return assignment, preview, canonical_json(report), len(mappings)


def _selected_mapping(assignment: dict, character_id: str, shot_id: int) -> dict:
    matches = [
        item
        for item in assignment["mappings"]
        if item["character_id"] == character_id and int(item["shot_id"]) == int(shot_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one mapping for character {character_id!r} in shot {shot_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _profile_for_character(assignment: dict, character_id: str) -> dict:
    profiles = [
        item
        for item in assignment["face_cast"]["profiles"]
        if item["character_id"] == character_id
    ]
    if len(profiles) != 1:
        raise ValueError(f"Unknown or duplicate character profile: {character_id!r}")
    return profiles[0]


def _resolve_multiface_crop_scale(
    crop_factor: float,
    canvas_mode: str,
    crop_scale_mode: str,
    target_face_px: float,
) -> tuple[float, float | None]:
    mode = str(crop_scale_mode)
    if mode == "legacy_crop_factor":
        factor = float(crop_factor)
        if not 1.2 <= factor <= 8.0:
            raise ValueError("crop_factor must be between 1.2 and 8.0")
        return factor, None
    if mode != "target_face_px":
        raise ValueError(f"Unknown crop_scale_mode: {crop_scale_mode}")
    if not str(canvas_mode).startswith("manual_"):
        raise ValueError(
            "target_face_px requires a manual canvas mode so the requested H3 face size "
            "is unambiguous"
        )
    canvas = int(str(canvas_mode).rsplit("_", 1)[1])
    target = float(target_face_px)
    if target <= 0.0 or target > float(canvas):
        raise ValueError(f"target_face_px must be greater than 0 and no larger than {canvas}")
    factor = float(canvas) / target
    if not 1.2 <= factor <= 8.0:
        raise ValueError(
            f"target_face_px={target:g} resolves to crop_factor={factor:.4f}; "
            "the supported crop-factor range is 1.2 to 8.0"
        )
    return factor, target


def build_multiface_repair_job(
    frames: torch.Tensor,
    identity_assignment: dict,
    character_id: str,
    shot_id: int,
    window_start_in_shot: int,
    window_frame_count: int,
    crop_factor: float,
    canvas_mode: str,
    center_smooth_window: int,
    size_smooth_window: int,
    identity_guard: str,
    minimum_similarity: float,
    analysis_chunk_frames: int,
    crop_scale_mode: str = "legacy_crop_factor",
    target_face_px: float = 300.0,
):
    assignment = _validate_hashed_dict(
        identity_assignment, ASSIGNMENT_SCHEMA, "identity_assignment"
    )
    source = _source_contract(frames)
    if source["proxy_sha256"] != assignment["source"]["proxy_sha256"]:
        raise ValueError("frames do not match the identity assignment source")
    character_id = str(character_id).strip()
    mapping = _selected_mapping(assignment, character_id, int(shot_id))
    profile = _profile_for_character(assignment, character_id)
    track_plan = assignment["track_plan"]
    shot = next(
        (item for item in track_plan["shots"] if int(item["shot_id"]) == int(shot_id)),
        None,
    )
    if shot is None:
        raise ValueError(f"Unknown shot_id: {shot_id}")
    count = int(window_frame_count)
    if count < 5 or align_frame_count(count) != count:
        raise ValueError("window_frame_count must follow the H3 17n+5 grid (5, 22, 39, ...)")
    if count > 124:
        raise ValueError("A multi-person repair job is capped at 124 frames; prefer 22-56")
    local_start = int(window_start_in_shot)
    shot_frame_count = int(shot["frame_count"])
    if local_start < 0 or local_start >= shot_frame_count:
        raise ValueError("Requested repair window starts outside the selected shot")
    source_count = min(count, shot_frame_count - local_start)
    alignment_context_pad_frames = count - source_count
    if source_count < 5:
        raise ValueError("The selected shot has fewer than 5 source frames after window_start")
    if alignment_context_pad_frames > 16:
        raise ValueError(
            "Requested repair window exceeds the selected shot by more than 16 frames; "
            "choose a shorter H3 17n+5 window or another shot-local start"
        )
    absolute_start = int(shot["start_frame"]) + local_start
    absolute_end = absolute_start + source_count - 1
    source_window = frames[absolute_start : absolute_end + 1]
    if alignment_context_pad_frames:
        tail = source_window[-1:].expand(alignment_context_pad_frames, -1, -1, -1).clone()
        window = torch.cat((source_window, tail), dim=0)
    else:
        window = source_window
    _, height, width = _validate_frames(window, name="repair_window")
    effective_crop_factor, requested_target_face_px = _resolve_multiface_crop_scale(
        crop_factor,
        canvas_mode,
        crop_scale_mode,
        target_face_px,
    )
    track_index = int(mapping["track_index"])
    masks = [
        _mask_at_source(
            shot,
            min(local_start + index, shot_frame_count - 1),
            track_index,
            height,
            width,
        )
        for index in range(count)
    ]
    detections, detector_report = _detect_local_opencv_yunet(
        window, YUNET_2023MAR_RELATIVE, 0.25, "cpu"
    )
    if identity_guard not in {"sface_cpu", "sam_track_only_exp"}:
        raise ValueError(f"Unknown identity_guard: {identity_guard}")
    recognizer = _create_sface_recognizer() if identity_guard == "sface_cpu" else None
    tracked_boxes: list[list[float] | None] = []
    states = []
    similarities = []
    try:
        for index, candidates in enumerate(detections):
            inside = [item for item in candidates if _detection_inside_mask(item, masks[index])]
            selected = None
            selected_similarity = None
            if inside and recognizer is not None:
                scored = []
                for candidate in inside:
                    feature = _sface_feature(window[index], candidate, recognizer)
                    similarity = float(torch.dot(feature, profile["identity_embedding"]))
                    scored.append((similarity, candidate))
                selected_similarity, selected = max(scored, key=lambda item: item[0])
                if selected_similarity < float(minimum_similarity):
                    selected = None
            elif inside:
                selected = max(inside, key=lambda item: float(item.get("confidence", 0.0)))
            tracked_boxes.append(list(selected["box"]) if selected is not None else None)
            states.append("detected" if selected is not None else "lost")
            similarities.append(selected_similarity)
    finally:
        del recognizer
        gc.collect()
    actual = [index for index, box in enumerate(tracked_boxes) if box is not None]
    if not actual:
        raise ValueError(
            "No identity-compatible face was localized inside the selected SAM3.1 person track"
        )
    fallback = list(tracked_boxes[actual[0]])
    shot_ranges = [(0, count - 1)]
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
            smooth_x[index], smooth_y[index], smooth_w[index], smooth_h[index], width, height
        )
        for index in range(count)
    ]
    crop_boxes = [
        _fit_square_crop(
            smooth_x[index],
            smooth_y[index],
            smooth_h[index],
            width,
            height,
            effective_crop_factor,
        )
        for index in range(count)
    ]
    canvas = _resolve_parity_canvas(crop_boxes, canvas_mode)
    crops = _crop_chunks(window, crop_boxes, canvas, int(analysis_chunk_frames))
    preview, preview_indices = _draw_preview(window, crop_boxes, states)
    reference_index = max(actual, key=lambda index: float(smooth_h[index]))
    source_reference = crops[reference_index : reference_index + 1]
    records = []
    for index in range(count):
        records.append(
            {
                "frame_index": index,
                "absolute_frame_index": min(absolute_start + index, absolute_end),
                "alignment_context_pad": index >= source_count,
                "shot_id": int(shot_id),
                "state": states[index],
                "detected": states[index] in ACTUAL_DETECTION_STATES,
                "source_face_box_xyxy": [round(float(value), 6) for value in face_boxes[index]],
                "source_face_height_px": round(float(smooth_h[index]), 6),
                "source_face_width_px": round(float(smooth_w[index]), 6),
                "source_crop_box_xyxy": [round(float(value), 6) for value in crop_boxes[index]],
                "crop_face_box_xyxy": [
                    round(float(value), 6)
                    for value in _upstream_parity_face_box_in_crop(
                        smooth_w[index], smooth_h[index], crop_boxes[index], canvas
                    )
                ],
                "parity_denoise_face_height_px": round(
                    float(crop_boxes[index][3] - crop_boxes[index][1])
                    / effective_crop_factor,
                    6,
                ),
                "paste_weight": round(float(smooth_weight[index]), 6),
                "identity_similarity": similarities[index],
            }
        )
    magnifications = [canvas / max(1e-6, box[3] - box[1]) for box in crop_boxes]
    crop_face_heights = [
        float(record["crop_face_box_xyxy"][3] - record["crop_face_box_xyxy"][1])
        for record in records
    ]
    source_boundary_limited_frames = sum(
        1
        for index, box in enumerate(crop_boxes)
        if float(box[3] - box[1]) + 1e-4
        < float(smooth_h[index]) * effective_crop_factor
    )
    plan = {
        "schema": PARITY_PLAN_SCHEMA,
        "status": "multiface_shot_local_parity_candidate_plan",
        "source": {
            "frame_count": count,
            "h3_aligned_frame_count": count,
            "h3_alignment_tail_frames": 0,
            "h3_grid_aligned": True,
            "width": width,
            "height": height,
            "fps": 24.0,
            "proxy_sha256": source_proxy_sha256(window),
        },
        "canvas": {
            "width": canvas,
            "height": canvas,
            "multiple": 32,
            "mode": canvas_mode,
        },
        "detector": {
            **detector_report,
            "backend": "sam31_person_track_plus_yunet_face_plus_sface_identity_guard",
            "sam31_track_plan_sha256": track_plan["sha256"],
            "identity_assignment_sha256": assignment["sha256"],
            "identity_guard": identity_guard,
            "identity_is_suggestion_not_proof": True,
        },
        "shots": [{"shot_id": 0, "start_frame": 0, "end_frame": count - 1}],
        "frames": records,
        "reference_frame_index": int(reference_index),
        "preview_frame_indices": preview_indices,
        "parity_defaults": {
            "require_h3_grid": True,
            "center_smooth_window": int(center_smooth_window),
            "size_smooth_window": int(size_smooth_window),
            "crop_factor": effective_crop_factor,
            "crop_scale_mode": str(crop_scale_mode),
            "requested_crop_factor": float(crop_factor),
            "target_face_px": requested_target_face_px,
            "canvas_mode": canvas_mode,
            "ultralytics_input_colour_space": None,
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
        "multiface": {
            "parent_source_proxy_sha256": source["proxy_sha256"],
            "identity_assignment_sha256": assignment["sha256"],
            "character_id": character_id,
            "track_key": mapping["track_key"],
            "source_shot_id": int(shot_id),
            "window_start_absolute": absolute_start,
            "window_end_absolute": absolute_end,
            "source_window_frame_count": source_count,
            "model_window_frame_count": count,
            "alignment_context_pad_frames": alignment_context_pad_frames,
            "source_window_proxy_sha256": source_proxy_sha256(source_window),
            "sequential_generation_required": True,
        },
        "limits": {
            "h3_grid_required": True,
            "explicit_alignment_tail_discard_required": False,
            "max_supported_alignment_tail_frames": 0,
            "max_supported_context_pad_frames": 16,
            "single_pass_safe": True,
            "identity_verified": False,
            "identity_guard_applied": identity_guard == "sface_cpu",
            "reference_crop_is_identity_proof": False,
            "automatic_accept": False,
            "audio_modified": False,
        },
        "metrics": {
            **smooth_metrics,
            "detected_frames": len(actual),
            "lost_or_interpolated_frames": count - len(actual),
            "face_height_min_px": float(smooth_h.min()),
            "face_height_mean_px": float(smooth_h.mean()),
            "face_height_max_px": float(smooth_h.max()),
            "magnification_min": min(magnifications),
            "magnification_mean": sum(magnifications) / len(magnifications),
            "magnification_max": max(magnifications),
            "crop_face_height_min_px": min(crop_face_heights),
            "crop_face_height_mean_px": sum(crop_face_heights) / len(crop_face_heights),
            "crop_face_height_max_px": max(crop_face_heights),
            "source_boundary_limited_frames": source_boundary_limited_frames,
            "crop_tensor_estimated_mib": count * canvas * canvas * 3 * 4 / 2**20,
        },
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    report = {
        "schema": "h3_t8_multiface_repair_job_report/v1",
        "status": plan["status"],
        "plan_sha256": plan["plan_sha256"],
        "character_id": character_id,
        "track_key": mapping["track_key"],
        "shot_id": int(shot_id),
        "absolute_window": [absolute_start, absolute_end],
        "source_window_frame_count": source_count,
        "model_window_frame_count": count,
        "alignment_context_pad_frames": alignment_context_pad_frames,
        "canvas": [canvas, canvas],
        "crop_scale_mode": str(crop_scale_mode),
        "effective_crop_factor": effective_crop_factor,
        "target_face_px": requested_target_face_px,
        "achieved_crop_face_height_px": {
            "min": min(crop_face_heights),
            "mean": sum(crop_face_heights) / len(crop_face_heights),
            "max": max(crop_face_heights),
        },
        "source_boundary_limited_frames": source_boundary_limited_frames,
        "identity_guard": identity_guard,
        "detected_frames": len(actual),
        "interpolated_frames": count - len(actual),
        "sequential_generation_required": True,
        "automatic_accept": False,
    }
    return (
        plan,
        window,
        crops,
        profile["reference_images"],
        source_reference,
        preview,
        canonical_json(report),
        absolute_start,
        count,
    )


def composite_multiface_candidate(
    base_frames: torch.Tensor,
    candidate_window: torch.Tensor,
    changed_mask: torch.Tensor,
    face_plan: dict,
    accept_candidate: bool,
    overlap_policy: str,
    previous_composite: dict | None = None,
):
    plan = _validate_parity_plan(face_plan)
    multiface = plan.get("multiface")
    if not isinstance(multiface, dict):
        raise ValueError("face_plan is not a multi-person repair job")
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    start = int(multiface["window_start_absolute"])
    end = int(multiface["window_end_absolute"])
    source_count = end - start + 1
    model_count = int(multiface.get("model_window_frame_count", source_count))
    candidate_count, candidate_h, candidate_w = _validate_frames(
        candidate_window, name="candidate_window"
    )
    if (candidate_count, candidate_h, candidate_w) != (model_count, height, width):
        raise ValueError("candidate_window shape does not match its absolute source window")
    if changed_mask.ndim != 3 or tuple(changed_mask.shape) != (model_count, height, width):
        raise ValueError("changed_mask must be [window_frames,H,W]")
    source_hash = source_proxy_sha256(base_frames)
    if source_hash != multiface["parent_source_proxy_sha256"]:
        raise ValueError("base_frames do not match the multi-person source")
    expected_source_window_hash = multiface.get(
        "source_window_proxy_sha256", plan["source"]["proxy_sha256"]
    )
    if source_proxy_sha256(base_frames[start : end + 1]) != expected_source_window_hash:
        raise ValueError("base_frames repair window does not match face_plan")
    if overlap_policy not in {"reject", "new_over_old_exp", "keep_old_exp"}:
        raise ValueError(f"Unknown overlap_policy: {overlap_policy}")

    if previous_composite is None:
        current = base_frames.clone()
        applied_mask = torch.zeros((frame_count, height, width), dtype=torch.bool)
        applied = []
    else:
        if not isinstance(previous_composite, dict) or previous_composite.get("schema") != COMPOSITE_SCHEMA:
            raise ValueError(f"previous_composite must use {COMPOSITE_SCHEMA}")
        if previous_composite.get("source_proxy_sha256") != source_hash:
            raise ValueError("previous_composite belongs to a different source")
        current = previous_composite["frames"].clone()
        applied_mask = previous_composite["applied_mask"].clone()
        applied = list(previous_composite["applied"])

    mask_values = changed_mask[:source_count].detach().to(device="cpu")
    if not bool(torch.isfinite(mask_values).all()):
        raise ValueError("changed_mask contains NaN or Inf")
    if bool((mask_values < 0).any()) or bool((mask_values > 1).any()):
        raise ValueError("changed_mask values must stay within 0..1")
    # Parity Stitch defines the bit-exact exterior as alpha == 0.  Do not use
    # an epsilon here: feathered masks legitimately contain tiny positive
    # alpha values, and their candidate pixels have already been blended.
    # Treating those values as exterior contradicts the producer contract and
    # incorrectly rejects otherwise audited second/third-person candidates.
    mask = mask_values > 0
    outside = ~mask
    base_window = base_frames[start : end + 1].detach().cpu()
    candidate_cpu = candidate_window[:source_count].detach().cpu()
    if not torch.equal(candidate_cpu[outside], base_window[outside]):
        raise ValueError("candidate_window changed pixels outside changed_mask")
    existing = applied_mask[start : end + 1]
    overlap = mask & existing
    overlap_pixels = int(overlap.sum().item())
    if overlap_pixels and overlap_policy == "reject":
        raise ValueError(
            f"Multi-person candidate overlaps {overlap_pixels} pixels already applied by another job"
        )
    effective = mask
    if overlap_policy == "keep_old_exp":
        effective = mask & ~existing
    if accept_candidate:
        target = current[start : end + 1]
        source_candidate = candidate_window[:source_count].to(
            device=target.device, dtype=target.dtype
        )
        effective_device = effective.to(device=target.device)
        current[start : end + 1] = torch.where(
            effective_device.unsqueeze(-1), source_candidate, target
        )
        applied_mask[start : end + 1] |= effective.cpu()
        applied.append(
            {
                "character_id": multiface["character_id"],
                "track_key": multiface["track_key"],
                "window": [start, end],
                "plan_sha256": plan["plan_sha256"],
                "changed_pixels": int(effective.sum().item()),
            }
        )
    state = {
        "schema": COMPOSITE_SCHEMA,
        "source_proxy_sha256": source_hash,
        "frames": current,
        "applied_mask": applied_mask,
        "applied": applied,
        "automatic_accept": False,
    }
    report = {
        "schema": "h3_t8_multiface_composite_report/v1",
        "status": "candidate_applied" if accept_candidate else "candidate_review_only",
        "character_id": multiface["character_id"],
        "track_key": multiface["track_key"],
        "absolute_window": [start, end],
        "source_window_frame_count": source_count,
        "model_window_frame_count": model_count,
        "alignment_context_pad_frames": model_count - source_count,
        "overlap_policy": overlap_policy,
        "overlap_pixels": overlap_pixels,
        "applied_job_count": len(applied),
        "audio_modified": False,
        "outside_mask_bit_exact": True,
    }
    return current, state, canonical_json(report), len(applied)
