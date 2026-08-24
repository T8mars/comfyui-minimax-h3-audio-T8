from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional

from .face_refine_advanced import (
    YUNET_2023MAR_RELATIVE,
    YUNET_2023MAR_SHA256,
    YUNET_2023MAR_SOURCE,
    _detect_local_opencv_yunet,
    _detector_report_name,
    _file_sha256,
    _resolve_detector_path,
)
from .multiface_refine_advanced import (
    TRACK_PLAN_SCHEMA,
    _mask_at_source,
    _validate_hashed_dict,
)
from .skin_finish import (
    _interrupt_and_progress,
    _prepare_mask,
    _process_chunk,
    _progress_bar,
    canonical_json,
    run_skin_finish,
)


SKIN_FINISH_SEQUENCE_STATE_SCHEMA = "h3_t8_skin_finish_sequence_state/v1"
SKIN_FINISH_SEQUENCE_REPORT_SCHEMA = "h3_t8_skin_finish_sequence_report/v1"
SKIN_FINISH_VIDEO_REPORT_SCHEMA = "h3_t8_skin_finish_video_finalize_report/v1"
SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA = "h3_t8_skin_finish_video_stream_report/v1"
SKIN_FINISH_VIDEO_STRICT_DECODE_POLICY = "ffmpeg_single_thread_xerror_v1"


def _hash_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_frames(frames: torch.Tensor, *, name: str = "frames") -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"{name} must be IMAGE [N,H,W,C], got {shape}")
    frame_count, height, width, channels = map(int, frames.shape)
    if frame_count < 1 or height < 2 or width < 2 or channels < 3:
        raise ValueError(f"{name} has an unsupported shape: {tuple(frames.shape)}")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool((frames < 0).any()) or bool((frames > 1).any()):
        raise ValueError(f"{name} must stay within 0..1")
    return frame_count, height, width, channels


def _frame_proxy_sha256(frame: torch.Tensor) -> str:
    proxy = torch_functional.interpolate(
        frame[..., :3].detach().float().movedim(-1, 0).unsqueeze(0),
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    )[0]
    digest = hashlib.sha256()
    digest.update(str(tuple(frame.shape)).encode("ascii"))
    digest.update(str(frame.dtype).encode("ascii"))
    digest.update(proxy.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_sequence_state(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SKIN_FINISH_SEQUENCE_STATE_SCHEMA:
        raise ValueError(
            f"previous_state must use {SKIN_FINISH_SEQUENCE_STATE_SCHEMA}"
        )
    unsigned = dict(value)
    signature = unsigned.pop("sha256", None)
    if signature != _hash_json(unsigned):
        raise ValueError("previous_state hash mismatch; it may be stale or modified")
    return value


def _shot_for_frame(track_plan: dict, absolute_frame: int) -> dict:
    for shot in track_plan["shots"]:
        if int(shot["start_frame"]) <= absolute_frame <= int(shot["end_frame"]):
            return shot
    raise ValueError(f"track_plan has no shot covering absolute frame {absolute_frame}")


def _detection_person_overlap(detection: dict, person_mask: torch.Tensor) -> float:
    left, top, right, bottom = [float(value) for value in detection["box"]]
    height, width = map(int, person_mask.shape)
    x1 = max(0, min(width - 1, int(math.floor(left))))
    y1 = max(0, min(height - 1, int(math.floor(top))))
    x2 = max(x1 + 1, min(width, int(math.ceil(right))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bottom))))
    return float(person_mask[y1:y2, x1:x2].float().mean())


def _face_detail_score(frame: torch.Tensor, box: list[float]) -> float:
    height, width = map(int, frame.shape[:2])
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, int(math.floor(left))))
    y1 = max(0, min(height - 1, int(math.floor(top))))
    x2 = max(x1 + 1, min(width, int(math.ceil(right))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bottom))))
    crop = frame[y1:y2, x1:x2, :3].detach().float().cpu()
    if int(crop.shape[0]) < 3 or int(crop.shape[1]) < 3:
        return 0.0
    luma = crop[..., 0] * 0.2126 + crop[..., 1] * 0.7152 + crop[..., 2] * 0.0722
    horizontal = (luma[:, 1:] - luma[:, :-1]).abs().mean()
    vertical = (luma[1:] - luma[:-1]).abs().mean()
    return float((horizontal + vertical) * 0.5)


def _profile_weight(detection: dict) -> float:
    landmarks = detection.get("landmarks_xy")
    if not isinstance(landmarks, list) or len(landmarks) != 5:
        return 0.70
    left_eye_x = float(landmarks[0][0])
    right_eye_x = float(landmarks[1][0])
    nose_x = float(landmarks[2][0])
    eye_span = abs(right_eye_x - left_eye_x)
    if eye_span < 1.0:
        return 0.35
    ratio = (nose_x - min(left_eye_x, right_eye_x)) / eye_span
    return max(0.30, min(1.0, 1.0 - abs(ratio - 0.50) / 0.55))


def _quality_weight(
    frame: torch.Tensor,
    detection: dict,
    overlap: float,
    *,
    detection_threshold: float,
    minimum_face_height_px: float,
    minimum_detail: float,
) -> tuple[float, dict[str, float]]:
    box = [float(value) for value in detection["box"]]
    face_height = max(0.0, box[3] - box[1])
    confidence = float(detection.get("confidence", 0.0))
    detail = _face_detail_score(frame, box)
    confidence_weight = max(
        0.0,
        min(1.0, (confidence - detection_threshold) / max(1.0e-6, 1.0 - detection_threshold)),
    )
    size_weight = max(0.0, min(1.0, face_height / max(1.0, minimum_face_height_px)))
    detail_weight = max(0.0, min(1.0, detail / max(1.0e-6, minimum_detail)))
    pose_weight = _profile_weight(detection)
    overlap_weight = max(0.0, min(1.0, overlap / 0.75))
    weight = confidence_weight * size_weight * detail_weight * pose_weight * overlap_weight
    return weight, {
        "confidence": confidence,
        "face_height_px": face_height,
        "detail_score": detail,
        "person_overlap": overlap,
        "profile_weight": pose_weight,
        "quality_weight": weight,
    }


def _inner_face_skin_mask(
    height: int,
    width: int,
    box: list[float],
    person_mask: torch.Tensor,
    *,
    weight: float,
    protect_features: bool,
    include_neck: bool,
) -> torch.Tensor:
    left, top, right, bottom = [float(value) for value in box]
    x1 = max(0, min(width - 1, int(math.floor(left))))
    y1 = max(0, min(height - 1, int(math.floor(top))))
    x2 = max(x1 + 1, min(width, int(math.ceil(right))))
    y2 = max(y1 + 1, min(height, int(math.ceil(bottom))))
    box_width = x2 - x1
    box_height = y2 - y1
    output = torch.zeros((height, width), dtype=torch.float32)
    if box_width < 4 or box_height < 4 or weight <= 0.0:
        return output
    yy = (torch.arange(box_height, dtype=torch.float32) + 0.5) / box_height
    xx = (torch.arange(box_width, dtype=torch.float32) + 0.5) / box_width
    ellipse = (
        ((xx[None, :] - 0.5) / 0.47) ** 2
        + ((yy[:, None] - 0.52) / 0.50) ** 2
    ) <= 1.0
    local = ellipse.float() * float(max(0.0, min(1.0, weight)))
    if protect_features:
        for center_x, center_y, radius_x, radius_y in (
            (0.32, 0.38, 0.16, 0.10),
            (0.68, 0.38, 0.16, 0.10),
            (0.50, 0.56, 0.11, 0.09),
            (0.50, 0.74, 0.24, 0.12),
        ):
            protected = (
                ((xx[None, :] - center_x) / radius_x) ** 2
                + ((yy[:, None] - center_y) / radius_y) ** 2
            ) <= 1.0
            local[protected] = 0.0
    local *= person_mask[y1:y2, x1:x2].float()
    output[y1:y2, x1:x2] = local
    if include_neck:
        neck_y1 = y2
        neck_y2 = min(height, y2 + max(2, int(round(box_height * 0.28))))
        neck_x1 = max(0, x1 + int(round(box_width * 0.30)))
        neck_x2 = min(width, x2 - int(round(box_width * 0.30)))
        if neck_y2 > neck_y1 and neck_x2 > neck_x1:
            output[neck_y1:neck_y2, neck_x1:neck_x2] = (
                person_mask[neck_y1:neck_y2, neck_x1:neck_x2].float()
                * float(max(0.0, min(1.0, weight)))
                * 0.55
            )
    return output


def _resolve_chunk_contract(
    frames: torch.Tensor,
    track_plan: dict,
    previous_state: dict | None,
    absolute_start_frame: int,
    maximum_overlap_frames: int,
) -> tuple[int, int]:
    frame_count, height, width, _ = _validate_frames(frames)
    source = track_plan["source"]
    if (height, width) != (int(source["height"]), int(source["width"])):
        raise ValueError("frames dimensions do not match track_plan source")
    absolute_start = int(absolute_start_frame)
    absolute_end = absolute_start + frame_count - 1
    if absolute_start < 0 or absolute_end >= int(source["frame_count"]):
        raise ValueError("chunk absolute frame range exceeds track_plan source")
    if previous_state is None:
        if absolute_start != 0:
            raise ValueError("the first stateful chunk must start at absolute frame 0")
        return 0, absolute_end
    if previous_state["track_plan_sha256"] != track_plan["sha256"]:
        raise ValueError("previous_state belongs to another SAM3.1 track plan")
    if (int(previous_state["height"]), int(previous_state["width"])) != (height, width):
        raise ValueError("previous_state geometry does not match this chunk")
    previous_end = int(previous_state["absolute_end_frame"])
    if absolute_start > previous_end + 1:
        raise ValueError("stateful chunks cannot contain an unreviewed frame gap")
    leading_overlap = max(0, previous_end - absolute_start + 1)
    if leading_overlap > int(maximum_overlap_frames):
        raise ValueError(
            f"chunk overlap is {leading_overlap} frames, above maximum_overlap_frames="
            f"{int(maximum_overlap_frames)}"
        )
    if leading_overlap >= frame_count:
        raise ValueError("chunk must contain at least one new frame after its overlap")
    recent = previous_state.get("recent_source_proxy_sha256", {})
    for local_index in range(leading_overlap):
        absolute_index = absolute_start + local_index
        expected = recent.get(str(absolute_index))
        if expected is None or expected != _frame_proxy_sha256(frames[local_index]):
            raise ValueError(
                f"overlap source mismatch at absolute frame {absolute_index}; "
                "the chunk may come from another video or edit"
            )
    return leading_overlap, absolute_end


def run_multiface_skin_finish(
    frames: torch.Tensor,
    track_plan: dict,
    *,
    absolute_start_frame: int = 0,
    previous_state: dict | None = None,
    preset: str = "subtle",
    amount: float = 0.35,
    texture_keep: float = 0.90,
    shine_control: float = 0.35,
    detection_threshold: float = 0.45,
    minimum_face_height_px: float = 24.0,
    minimum_detail: float = 0.010,
    bbox_ema_alpha: float = 0.55,
    max_missing_frames: int = 2,
    protect_features: bool = True,
    include_neck: bool = False,
    maximum_overlap_frames: int = 8,
    mask_feather_px: int = 3,
    proxy_long_side: int = 640,
    chunk_frames: int = 4,
    accept_candidate: bool = False,
    audio: dict | None = None,
) -> tuple:
    started = time.perf_counter()
    track_plan = _validate_hashed_dict(track_plan, TRACK_PLAN_SCHEMA, "track_plan")
    previous_state = _validate_sequence_state(previous_state)
    if preset not in {"subtle", "oil_control"}:
        raise ValueError("multi-person P1 supports only color-neutral subtle or oil_control")
    if not 0.0 <= float(bbox_ema_alpha) <= 1.0:
        raise ValueError("bbox_ema_alpha must stay within 0..1")
    if int(max_missing_frames) < 0 or int(max_missing_frames) > 8:
        raise ValueError("max_missing_frames must stay within 0..8")
    frame_count, height, width, _ = _validate_frames(frames)
    leading_overlap, absolute_end = _resolve_chunk_contract(
        frames,
        track_plan,
        previous_state,
        int(absolute_start_frame),
        int(maximum_overlap_frames),
    )
    emitted = frames[leading_overlap:]
    emitted_count = int(emitted.shape[0])
    emitted_absolute_start = int(absolute_start_frame) + leading_overlap
    detections, detector_report = _detect_local_opencv_yunet(
        emitted,
        YUNET_2023MAR_RELATIVE,
        float(detection_threshold),
        "cpu",
    )
    trajectories = json.loads(
        json.dumps((previous_state or {}).get("track_trajectories", {}))
    )
    raw_mask = torch.zeros((emitted_count, height, width), dtype=torch.float32)
    frame_records: list[dict[str, Any]] = []
    current_shot = None
    for local_index, candidates in enumerate(detections):
        absolute_index = emitted_absolute_start + local_index
        shot = _shot_for_frame(track_plan, absolute_index)
        shot_id = int(shot["shot_id"])
        if current_shot != shot_id:
            current_shot = shot_id
        shot_local = absolute_index - int(shot["start_frame"])
        person_masks = [
            _mask_at_source(shot, shot_local, track_index, height, width)
            for track_index in range(int(shot["object_count"]))
        ]
        available = set(range(len(candidates)))
        accepted_tracks = 0
        per_track = []
        for track_index, person_mask in enumerate(person_masks):
            track_key = str(shot["track_keys"][track_index])
            ranked = []
            for candidate_index in available:
                candidate = candidates[candidate_index]
                overlap = _detection_person_overlap(candidate, person_mask)
                if overlap < 0.20:
                    continue
                quality, metrics = _quality_weight(
                    emitted[local_index],
                    candidate,
                    overlap,
                    detection_threshold=float(detection_threshold),
                    minimum_face_height_px=float(minimum_face_height_px),
                    minimum_detail=float(minimum_detail),
                )
                area = max(0.0, candidate["box"][2] - candidate["box"][0]) * max(
                    0.0, candidate["box"][3] - candidate["box"][1]
                )
                ranked.append((quality, area, candidate_index, candidate, metrics))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            state = trajectories.get(track_key)
            selected = ranked[0] if ranked and ranked[0][0] >= 0.10 else None
            if selected is not None:
                quality, _, candidate_index, candidate, metrics = selected
                available.remove(candidate_index)
                observed = [float(value) for value in candidate["box"]]
                if state is not None and int(state.get("last_frame", -100)) == absolute_index - 1:
                    prior = [float(value) for value in state["smoothed_box_xyxy"]]
                    alpha = float(bbox_ema_alpha)
                    smoothed = [alpha * new + (1.0 - alpha) * old for old, new in zip(prior, observed, strict=True)]
                else:
                    smoothed = observed
                state = {
                    "smoothed_box_xyxy": smoothed,
                    "last_frame": absolute_index,
                    "last_detection_frame": absolute_index,
                    "last_quality_weight": float(quality),
                }
                trajectories[track_key] = state
                accepted_tracks += 1
                status = "detected"
            elif (
                state is not None
                and absolute_index - int(state.get("last_detection_frame", -100))
                <= int(max_missing_frames)
            ):
                missing = absolute_index - int(state["last_detection_frame"])
                quality = float(state.get("last_quality_weight", 0.0)) * (0.55**missing)
                smoothed = [float(value) for value in state["smoothed_box_xyxy"]]
                metrics = {
                    "confidence": 0.0,
                    "face_height_px": max(0.0, smoothed[3] - smoothed[1]),
                    "detail_score": 0.0,
                    "person_overlap": 1.0,
                    "profile_weight": 0.0,
                    "quality_weight": quality,
                }
                state["last_frame"] = absolute_index
                trajectories[track_key] = state
                status = "short_gap_decay"
            else:
                quality = 0.0
                smoothed = []
                metrics = {"quality_weight": 0.0}
                status = "abstain_no_reliable_face"
            if smoothed and quality > 0.0:
                face_mask = _inner_face_skin_mask(
                    height,
                    width,
                    smoothed,
                    person_mask,
                    weight=float(quality),
                    protect_features=bool(protect_features),
                    include_neck=bool(include_neck),
                )
                raw_mask[local_index] = torch.maximum(raw_mask[local_index], face_mask)
            per_track.append(
                {
                    "track_key": track_key,
                    "status": status,
                    "smoothed_box_xyxy": [round(value, 5) for value in smoothed],
                    "metrics": {key: round(float(value), 7) for key, value in metrics.items()},
                }
            )
        frame_records.append(
            {
                "absolute_frame": absolute_index,
                "shot_id": shot_id,
                "detected_face_count": len(candidates),
                "accepted_track_count": accepted_tracks,
                "tracks": per_track,
            }
        )

    candidate, source, selected, used, rejected, difference, _, audio_out, base_report_json = (
        run_skin_finish(
            emitted,
            preset=preset,
            amount=float(amount),
            texture_keep=float(texture_keep),
            shine_control=float(shine_control),
            tone_adjust=0.0,
            execution_mode="candidate_only",
            chunk_frames=int(chunk_frames),
            mask=raw_mask,
            audio=audio,
            mask_source="external_exact",
            protect_features=bool(protect_features),
            minimum_mask_area=0.00005,
            maximum_mask_area=0.35,
            mask_feather_px=int(mask_feather_px),
            temporal_mask_radius=0,
            proxy_long_side=int(proxy_long_side),
            accept_candidate=bool(accept_candidate),
        )
    )
    base_report = json.loads(base_report_json)
    recent: dict[str, str] = {}
    first_recent = max(
        int(absolute_start_frame),
        absolute_end - max(0, int(maximum_overlap_frames)) + 1,
    )
    for absolute_index in range(first_recent, absolute_end + 1):
        local_index = absolute_index - int(absolute_start_frame)
        recent[str(absolute_index)] = _frame_proxy_sha256(frames[local_index])
    sequence_state = {
        "schema": SKIN_FINISH_SEQUENCE_STATE_SCHEMA,
        "track_plan_sha256": track_plan["sha256"],
        "height": height,
        "width": width,
        "fps": float(track_plan["source"]["fps"]),
        "absolute_end_frame": absolute_end,
        "emitted_frame_count_total": int((previous_state or {}).get("emitted_frame_count_total", 0))
        + emitted_count,
        "track_trajectories": trajectories,
        "recent_source_proxy_sha256": recent,
        "color_policy": "same_color_neutral_parameters_for_all_tracks",
        "rgb_temporal_averaging": False,
    }
    sequence_state["sha256"] = _hash_json(sequence_state)
    report = {
        "schema": SKIN_FINISH_SEQUENCE_REPORT_SCHEMA,
        "status": base_report["status"],
        "track_plan_sha256": track_plan["sha256"],
        "chunk": {
            "absolute_input_range": [int(absolute_start_frame), absolute_end],
            "leading_overlap_frames_verified_and_discarded": leading_overlap,
            "absolute_emitted_range": [emitted_absolute_start, absolute_end],
            "emitted_frame_count": emitted_count,
            "gap_allowed": False,
        },
        "mask_contract": {
            "source": "existing_sam31_person_tracks_intersected_with_local_yunet_inner_face_regions",
            "sam31_reloaded": False,
            "feature_exclusion": bool(protect_features),
            "neck_included": bool(include_neck),
            "temporal_filter": "shot_local_causal_bbox_ema_only",
            "rgb_temporal_averaging": False,
            "small_profile_blur_occlusion_downweighted": True,
            "semantic_claim": "conservative tracked face-skin proxy; not pixel-semantic skin parsing",
        },
        "color_contract": {
            "per_person_hue_or_saturation_shift": False,
            "shared_color_neutral_parameters": True,
            "tone_adjust_forced": 0.0,
        },
        "detector": detector_report,
        "frames": frame_records,
        "base_skin_finish_report": base_report,
        "sequence_state_sha256": sequence_state["sha256"],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "automatic_accept": False,
    }
    return (
        candidate,
        source,
        selected,
        audio_out,
        used,
        rejected,
        difference,
        sequence_state,
        canonical_json(report),
        emitted_absolute_start,
        emitted_count,
    )


def _audio_packet_digest(container, streams) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    bytes_total = 0
    stream_indices = {stream.index for stream in streams}
    for packet in container.demux():
        if packet.stream.index not in stream_indices or packet.dts is None:
            continue
        payload = bytes(packet)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        count += 1
        bytes_total += len(payload)
    return {
        "packet_count": count,
        "payload_bytes": bytes_total,
        "payload_sha256": digest.hexdigest(),
    }


def _strict_validate_encoded_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required for strict Skin Finish video validation"
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-xerror",
            "-err_detect",
            "explode",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    diagnostics = (result.stderr or "").strip()
    if result.returncode or diagnostics:
        raise RuntimeError(
            "strict Skin Finish H.264 validation reported decode errors:\n"
            + diagnostics[-4000:]
        )


def _resolve_output_path(filename_prefix: str, width: int, height: int) -> tuple[Path, str, str]:
    import folder_paths

    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        str(filename_prefix), folder_paths.get_output_directory(), width, height
    )
    output = Path(full_folder) / f"{filename}_{counter:05}_.mp4"
    return output, output.name, subfolder


def _file_video_source_path(source_video) -> Path:
    if not hasattr(source_video, "get_stream_source"):
        raise ValueError("source_video must be a current ComfyUI file-backed VIDEO")
    source = source_video.get_stream_source()
    if isinstance(source, io.BytesIO):
        raise ValueError("BytesIO VIDEO is not supported by the two-pass file stream")
    source_path = Path(os.fspath(source)).resolve()
    if not source_path.is_file():
        raise ValueError(f"source VIDEO file no longer exists: {source_path}")
    if hasattr(source_video, "get_active_trim_window"):
        start_time, duration = source_video.get_active_trim_window()
        if float(start_time) != 0.0 or float(duration) != 0.0:
            raise ValueError("trimmed VIDEO is not supported by the two-pass file stream")
    return source_path


def _video_frame_proxy(frame: torch.Tensor) -> torch.Tensor:
    return torch_functional.interpolate(
        frame[..., :3].detach().float().movedim(-1, 0).unsqueeze(0),
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    )[0].movedim(0, -1).cpu().contiguous()


def _update_proxy_digest(digest, frame_index: int, proxy: torch.Tensor) -> None:
    digest.update(int(frame_index).to_bytes(8, "little", signed=False))
    digest.update(proxy.numpy().tobytes())


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(1.0e-6, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0e-6, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(1.0e-6, first_area + second_area - intersection)


def _create_pinned_yunet(width: int, height: int, confidence: float):
    path = _resolve_detector_path(YUNET_2023MAR_RELATIVE)
    model_hash = _file_sha256(path)
    if model_hash != YUNET_2023MAR_SHA256:
        raise ValueError(
            "the two-pass stream requires the pinned OpenCV Zoo YuNet 2023mar model; "
            f"expected {YUNET_2023MAR_SHA256}, got {model_hash}"
        )
    try:
        import cv2
    except Exception as error:
        raise RuntimeError(
            "two-pass Skin Finish requires OpenCV with FaceDetectorYN support"
        ) from error
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not provide FaceDetectorYN")
    model = cv2.FaceDetectorYN.create(
        str(path),
        "",
        (int(width), int(height)),
        float(confidence),
        0.30,
        5000,
    )
    report = {
        "backend": "local_opencv_yunet",
        "model": _detector_report_name(path),
        "model_sha256": model_hash,
        "official_opencv_zoo_match": True,
        "model_source": YUNET_2023MAR_SOURCE,
        "model_license": "MIT",
        "effective_device": "cpu",
        "network_download": False,
        "cached_after_analysis": False,
    }
    return model, report


def _detect_yunet_rgb(model, rgb) -> list[dict[str, Any]]:
    bgr = rgb[..., ::-1].copy()
    _, result = model.detect(bgr)
    detections: list[dict[str, Any]] = []
    if result is None:
        return detections
    height, width = map(int, rgb.shape[:2])
    for row in result:
        values = [float(value) for value in row.tolist()]
        if len(values) < 5:
            continue
        x, y, box_width, box_height = values[:4]
        left = max(0.0, min(float(width - 1), x))
        top = max(0.0, min(float(height - 1), y))
        right = max(left + 1.0, min(float(width), x + box_width))
        bottom = max(top + 1.0, min(float(height), y + box_height))
        item: dict[str, Any] = {
            "box": [left, top, right, bottom],
            "confidence": values[14] if len(values) >= 15 else values[-1],
        }
        if len(values) >= 14:
            item["landmarks_xy"] = [
                [values[index], values[index + 1]] for index in range(4, 14, 2)
            ]
        detections.append(item)
    return detections


def _smooth_stream_faces(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    *,
    alpha: float,
    next_track_slot: int,
) -> tuple[list[dict[str, Any]], int]:
    available = set(range(len(previous)))
    output: list[dict[str, Any]] = []
    for item in sorted(current, key=lambda value: float(value["weight"]), reverse=True):
        best_index = None
        best_iou = 0.0
        for index in available:
            score = _box_iou(item["box"], previous[index]["box"])
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index is not None and best_iou >= 0.10:
            prior = previous[best_index]
            available.remove(best_index)
            box = [
                float(alpha) * new + (1.0 - float(alpha)) * old
                for old, new in zip(prior["box"], item["box"], strict=True)
            ]
            track_slot = int(prior["track_slot"])
        else:
            box = [float(value) for value in item["box"]]
            track_slot = int(next_track_slot)
            next_track_slot += 1
        output.append(
            {
                "box": box,
                "weight": float(item["weight"]),
                "track_slot": track_slot,
            }
        )
    return output, next_track_slot


def _analyze_stream_faces(
    source_path: Path,
    *,
    expected_frame_count: int,
    width: int,
    height: int,
    detection_threshold: float,
    minimum_face_height_px: float,
    minimum_detail: float,
    bbox_ema_alpha: float,
    scene_cut_threshold: float,
    maximum_faces: int,
    progress,
) -> dict[str, Any]:
    import av

    model = None
    detector_report = None
    records: list[list[dict[str, Any]]] = []
    proxy_digest = hashlib.sha256()
    previous_proxy = None
    previous_faces: list[dict[str, Any]] = []
    next_track_slot = 0
    scene_cut_count = 0
    total_faces = 0
    try:
        model, detector_report = _create_pinned_yunet(
            width, height, float(detection_threshold)
        )
        with av.open(str(source_path), mode="r") as container:
            if not container.streams.video:
                raise ValueError("source VIDEO contains no video stream")
            video_stream = container.streams.video[0]
            for frame_index, av_frame in enumerate(container.decode(video_stream)):
                _interrupt_and_progress(progress, frame_index, expected_frame_count * 2)
                if av_frame.width != width or av_frame.height != height:
                    raise ValueError("source VIDEO changes geometry during the analysis pass")
                rgb = av_frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(rgb.copy()).to(dtype=torch.float32).div_(255.0)
                proxy = _video_frame_proxy(tensor)
                _update_proxy_digest(proxy_digest, frame_index, proxy)
                cut = False
                if previous_proxy is not None:
                    delta = float((proxy - previous_proxy).abs().mean())
                    cut = delta >= float(scene_cut_threshold)
                if cut:
                    previous_faces = []
                    scene_cut_count += 1
                candidates = []
                for detection in _detect_yunet_rgb(model, rgb):
                    quality, _ = _quality_weight(
                        tensor,
                        detection,
                        1.0,
                        detection_threshold=float(detection_threshold),
                        minimum_face_height_px=float(minimum_face_height_px),
                        minimum_detail=float(minimum_detail),
                    )
                    if quality >= 0.10:
                        candidates.append(
                            {
                                "box": [float(value) for value in detection["box"]],
                                "weight": float(quality),
                            }
                        )
                candidates.sort(key=lambda value: float(value["weight"]), reverse=True)
                candidates = candidates[: int(maximum_faces)]
                smoothed, next_track_slot = _smooth_stream_faces(
                    candidates,
                    previous_faces,
                    alpha=float(bbox_ema_alpha),
                    next_track_slot=next_track_slot,
                )
                records.append(smoothed)
                total_faces += len(smoothed)
                previous_faces = smoothed
                previous_proxy = proxy
    finally:
        del model
        gc.collect()
    if len(records) != int(expected_frame_count):
        raise ValueError(
            f"source VIDEO decoded {len(records)} frames during analysis, "
            f"expected {int(expected_frame_count)}"
        )
    return {
        "records": records,
        "source_proxy_sha256": proxy_digest.hexdigest(),
        "detector": detector_report,
        "scene_cut_count": scene_cut_count,
        "accepted_face_instances": total_faces,
        "unique_track_slots_proxy": next_track_slot,
    }


def stream_skin_finish_video(
    source_video,
    *,
    preset: str = "subtle",
    amount: float = 0.35,
    texture_keep: float = 0.90,
    shine_control: float = 0.35,
    detection_threshold: float = 0.45,
    minimum_face_height_px: float = 24.0,
    minimum_detail: float = 0.010,
    bbox_ema_alpha: float = 0.55,
    scene_cut_threshold: float = 0.28,
    maximum_faces: int = 4,
    mask_feather_px: int = 3,
    proxy_long_side: int = 640,
    chunk_frames: int = 4,
    filename_prefix: str = "MiniMaxH3/SkinFinish/stream_skin_finish",
    crf: float = 18.0,
    accept_candidate: bool = False,
):
    from comfy_api.latest import InputImpl

    if not bool(accept_candidate):
        report = {
            "schema": SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA,
            "status": "SOURCE_SELECTED_NO_ANALYSIS_OR_FILE_WRITTEN",
            "automatic_accept": False,
            "two_pass_executed": False,
            "full_image_batch_materialized": False,
        }
        return source_video, "", canonical_json(report), None
    if preset not in {"subtle", "oil_control"}:
        raise ValueError("two-pass stream supports only color-neutral subtle or oil_control")
    if not 0.0 <= float(amount) <= 1.0:
        raise ValueError("amount must stay within 0..1")
    if not 0.0 <= float(texture_keep) <= 1.0:
        raise ValueError("texture_keep must stay within 0..1")
    if not 0.0 <= float(shine_control) <= 1.0:
        raise ValueError("shine_control must stay within 0..1")
    if not 0.0 <= float(bbox_ema_alpha) <= 1.0:
        raise ValueError("bbox_ema_alpha must stay within 0..1")
    if not 0.0 < float(scene_cut_threshold) <= 1.0:
        raise ValueError("scene_cut_threshold must stay within 0..1")
    if not 1 <= int(maximum_faces) <= 12:
        raise ValueError("maximum_faces must stay within 1..12")
    chunk_size = max(1, min(32, int(chunk_frames)))
    source_path = _file_video_source_path(source_video)
    source_stat = source_path.stat()
    frame_count = int(source_video.get_frame_count())
    width, height = map(int, source_video.get_dimensions())
    if frame_count < 1 or width < 2 or height < 2:
        raise ValueError("source VIDEO has invalid frame count or geometry")
    if int(source_video.get_bit_depth()) > 8:
        raise ValueError("two-pass Skin Finish is SDR 8-bit only")

    import av

    with av.open(str(source_path), mode="r") as source_container:
        if not source_container.streams.video:
            raise ValueError("source VIDEO contains no video stream")
        video_stream = source_container.streams.video[0]
        if (int(video_stream.width), int(video_stream.height)) != (width, height):
            raise ValueError("ComfyUI VIDEO geometry differs from its encoded stream")
        if getattr(video_stream, "rotation", 0):
            raise ValueError("rotated VIDEO is not supported by the exact-geometry stream")
        transfer = str(getattr(video_stream.codec_context, "color_trc", "")).lower()
        if "smpte2084" in transfer or "arib-std-b67" in transfer:
            raise ValueError("HDR transfer functions remain outside the SDR stream contract")
        audio_streams = list(source_container.streams.audio)
        for stream in audio_streams:
            if stream.codec_context is None:
                raise ValueError("source contains an unreadable audio stream")
            if stream.codec.name not in {"aac", "mp3", "alac", "ac3", "eac3"}:
                raise ValueError(
                    f"audio codec {stream.codec.name!r} is not approved for MP4 packet-copy"
                )

    progress = _progress_bar(frame_count * 2)
    started = time.perf_counter()
    analysis = _analyze_stream_faces(
        source_path,
        expected_frame_count=frame_count,
        width=width,
        height=height,
        detection_threshold=float(detection_threshold),
        minimum_face_height_px=float(minimum_face_height_px),
        minimum_detail=float(minimum_detail),
        bbox_ema_alpha=float(bbox_ema_alpha),
        scene_cut_threshold=float(scene_cut_threshold),
        maximum_faces=int(maximum_faces),
        progress=progress,
    )
    current_stat = source_path.stat()
    if (current_stat.st_size, current_stat.st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise ValueError("source VIDEO changed between metadata and analysis passes")
    if int(analysis["accepted_face_instances"]) == 0:
        report = {
            "schema": SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA,
            "status": "ABSTAIN_NO_RELIABLE_FACE_NO_FILE_WRITTEN",
            "source_path": str(source_path),
            "analysis": {
                key: value for key, value in analysis.items() if key != "records"
            },
            "two_pass_executed": False,
            "analysis_pass_executed": True,
            "full_image_batch_materialized": False,
            "automatic_accept": False,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        return source_video, "", canonical_json(report), None

    output_path, saved_name, subfolder = _resolve_output_path(
        filename_prefix, width, height
    )
    temporary = output_path.with_name(
        f"{output_path.stem}.partial-{uuid.uuid4().hex}{output_path.suffix}"
    )
    source_audio_digest = hashlib.sha256()
    source_audio_count = 0
    source_audio_bytes = 0
    pass_two_proxy = hashlib.sha256()
    decoded_video_frames = 0
    encoded_video_frames = 0
    used_mask_frame_count = 0
    peak_chunk_frames = 0
    outside_mask_exact = True
    source_audio = None
    output_audio = None
    verified_frames = 0
    try:
        with av.open(str(source_path), mode="r") as source_container:
            video_stream = source_container.streams.video[0]
            audio_streams = list(source_container.streams.audio)
            rate = (
                Fraction(video_stream.average_rate)
                if video_stream.average_rate
                else Fraction(24)
            )
            with av.open(
                str(temporary),
                mode="w",
                format="mp4",
                options={"movflags": "use_metadata_tags+faststart"},
            ) as output_container:
                for key, value in source_container.metadata.items():
                    output_container.metadata[str(key)] = str(value)
                out_video = output_container.add_stream("libx264", rate=rate)
                out_video.width = width
                out_video.height = height
                out_video.pix_fmt = "yuv420p"
                out_video.codec_context.max_b_frames = 0
                out_video.codec_context.thread_count = 1
                out_video.options = {
                    "crf": str(float(crf)),
                    "preset": "medium",
                    "threads": "1",
                }
                out_video.codec_context.time_base = video_stream.time_base
                audio_map = {
                    stream: output_container.add_stream_from_template(stream, opaque=True)
                    for stream in audio_streams
                }
                pending_frames: list[torch.Tensor] = []
                pending_pts: list[Any] = []

                def flush_pending() -> None:
                    nonlocal encoded_video_frames
                    nonlocal used_mask_frame_count
                    nonlocal peak_chunk_frames
                    nonlocal outside_mask_exact
                    if not pending_frames:
                        return
                    source_chunk = torch.stack(pending_frames, dim=0)
                    peak_chunk_frames = max(peak_chunk_frames, int(source_chunk.shape[0]))
                    raw_mask = torch.zeros(
                        (int(source_chunk.shape[0]), height, width), dtype=torch.float32
                    )
                    person_plane = torch.ones((height, width), dtype=torch.float32)
                    for local_index in range(int(source_chunk.shape[0])):
                        absolute_index = encoded_video_frames + local_index
                        for face in analysis["records"][absolute_index]:
                            face_mask = _inner_face_skin_mask(
                                height,
                                width,
                                face["box"],
                                person_plane,
                                weight=float(face["weight"]),
                                protect_features=True,
                                include_neck=False,
                            )
                            raw_mask[local_index] = torch.maximum(
                                raw_mask[local_index], face_mask
                            )
                    used_mask, _, mask_report = _prepare_mask(
                        raw_mask,
                        frame_count=int(source_chunk.shape[0]),
                        height=height,
                        width=width,
                        minimum_area=0.00005,
                        maximum_area=0.35,
                        feather_px=int(mask_feather_px),
                        temporal_radius=0,
                        chunk_frames=chunk_size,
                    )
                    used_mask_frame_count += int(mask_report["accepted_frame_count"])
                    if int(mask_report["accepted_frame_count"]) > 0:
                        candidate = _process_chunk(
                            source_chunk,
                            used_mask,
                            preset=preset,
                            amount=float(amount),
                            texture_keep=float(texture_keep),
                            shine_control=float(shine_control),
                            tone_adjust=0.0,
                            proxy_long_side=int(proxy_long_side),
                        )
                    else:
                        candidate = source_chunk
                    outside = used_mask <= 0
                    if not torch.equal(
                        candidate[..., :3][outside], source_chunk[..., :3][outside]
                    ):
                        outside_mask_exact = False
                        raise RuntimeError("stream Skin Finish changed pixels outside its mask")
                    for local_index, frame_tensor in enumerate(candidate):
                        rgb = (
                            frame_tensor[..., :3]
                            .detach()
                            .clamp(0.0, 1.0)
                            .mul(255.0)
                            .byte()
                            .cpu()
                            .contiguous()
                            .numpy()
                        )
                        target = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        target.pts = pending_pts[local_index]
                        target.time_base = video_stream.time_base
                        for encoded in out_video.encode(target):
                            output_container.mux(encoded)
                    encoded_video_frames += int(source_chunk.shape[0])
                    pending_frames.clear()
                    pending_pts.clear()
                    _interrupt_and_progress(
                        progress, frame_count + encoded_video_frames, frame_count * 2
                    )

                streams = [video_stream, *audio_streams]
                for packet in source_container.demux(*streams):
                    if packet.stream == video_stream:
                        for source_frame in packet.decode():
                            if decoded_video_frames >= frame_count:
                                raise ValueError(
                                    "source VIDEO decoded more frames than its reported count"
                                )
                            rgb = source_frame.to_ndarray(format="rgb24")
                            tensor = (
                                torch.from_numpy(rgb.copy())
                                .to(dtype=torch.float32)
                                .div_(255.0)
                            )
                            proxy = _video_frame_proxy(tensor)
                            _update_proxy_digest(
                                pass_two_proxy, decoded_video_frames, proxy
                            )
                            pending_frames.append(tensor)
                            pending_pts.append(source_frame.pts)
                            decoded_video_frames += 1
                            if len(pending_frames) >= chunk_size:
                                flush_pending()
                    elif packet.stream in audio_map and packet.dts is not None:
                        payload = bytes(packet)
                        source_audio_digest.update(len(payload).to_bytes(8, "little"))
                        source_audio_digest.update(payload)
                        source_audio_count += 1
                        source_audio_bytes += len(payload)
                        packet.stream = audio_map[packet.stream]
                        output_container.mux(packet)
                flush_pending()
                for encoded in out_video.encode():
                    output_container.mux(encoded)
        if decoded_video_frames != frame_count or encoded_video_frames != frame_count:
            raise ValueError(
                "two-pass stream frame count differs from the ComfyUI VIDEO contract"
            )
        if pass_two_proxy.hexdigest() != analysis["source_proxy_sha256"]:
            raise RuntimeError("source pixels changed between analysis and processing passes")
        current_stat = source_path.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            raise RuntimeError("source VIDEO changed during the processing pass")
        source_audio = {
            "packet_count": source_audio_count,
            "payload_bytes": source_audio_bytes,
            "payload_sha256": source_audio_digest.hexdigest(),
        }
        with av.open(str(temporary), mode="r") as output_container:
            output_audio = _audio_packet_digest(
                output_container, list(output_container.streams.audio)
            )
        if source_audio != output_audio:
            raise RuntimeError(
                "output audio packet payloads differ from the source; refusing publication"
            )
        with av.open(str(temporary), mode="r") as output_container:
            if not output_container.streams.video:
                raise RuntimeError("stream output contains no decodable video stream")
            verified_stream = output_container.streams.video[0]
            if (int(verified_stream.width), int(verified_stream.height)) != (width, height):
                raise RuntimeError("stream output geometry differs from the source")
            for verified_frame in output_container.decode(verified_stream):
                if verified_frame.width != width or verified_frame.height != height:
                    raise RuntimeError("stream output changes geometry mid-stream")
                verified_frames += 1
        if verified_frames != frame_count:
            raise RuntimeError(
                f"stream output decodes {verified_frames} frames, expected {frame_count}"
            )
        _strict_validate_encoded_video(temporary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    report = {
        "schema": SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA,
        "status": "CANDIDATE_TWO_PASS_STREAM_FINALIZED",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "analysis": {key: value for key, value in analysis.items() if key != "records"},
        "execution": {
            "passes": 2,
            "analysis_pass": "all_frames_cpu_yunet_metadata_only",
            "processing_pass": "bounded_image_chunks_plus_incremental_h264_encode",
            "full_image_batch_materialized": False,
            "peak_chunk_frames": peak_chunk_frames,
            "chunk_frames_requested": chunk_size,
            "frame_metadata_records_retained": len(analysis["records"]),
            "used_mask_frame_count": used_mask_frame_count,
            "outside_mask_bit_exact_before_encode": outside_mask_exact,
            "source_proxy_equal_between_passes": True,
            "rgb_temporal_averaging": False,
        },
        "product_boundary": (
            "Pinned YuNet inner-face proxy with shared color-neutral parameters. It is not "
            "semantic skin parsing, identity tracking, deblur or facial reconstruction."
        ),
        "video": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "codec": "libx264",
            "encoder_threads": 1,
            "crf": float(crf),
            "strict_decoded_frame_count_verified": verified_frames,
            "strict_decode_policy": SKIN_FINISH_VIDEO_STRICT_DECODE_POLICY,
            "sdr_8bit_only": True,
        },
        "audio": {
            "method": "source_packet_payload_copy",
            "source": source_audio,
            "output": output_audio,
            "packet_payload_exact": True,
            "decoded_pcm_reencode": False,
        },
        "source_overwritten": False,
        "atomic_publish": True,
        "automatic_accept": False,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    saved_result = (saved_name, subfolder)
    return InputImpl.VideoFromFile(str(output_path)), str(output_path), canonical_json(report), saved_result


def finalize_skin_finish_video(
    source_video,
    processed_frames: torch.Tensor,
    *,
    filename_prefix: str = "MiniMaxH3/SkinFinish/skin_finish",
    crf: float = 18.0,
    accept_candidate: bool = False,
):
    from comfy_api.latest import InputImpl

    if not bool(accept_candidate):
        report = {
            "schema": SKIN_FINISH_VIDEO_REPORT_SCHEMA,
            "status": "SOURCE_SELECTED_NO_FILE_WRITTEN",
            "automatic_accept": False,
            "audio_packet_copy": "not_run",
        }
        return source_video, "", canonical_json(report), None
    if not hasattr(source_video, "get_stream_source"):
        raise ValueError("source_video must be a current ComfyUI file-backed VIDEO")
    source = source_video.get_stream_source()
    if isinstance(source, io.BytesIO):
        raise ValueError("BytesIO VIDEO is not supported because packet-copy verification needs a file")
    source_path = Path(os.fspath(source)).resolve()
    if not source_path.is_file():
        raise ValueError(f"source VIDEO file no longer exists: {source_path}")
    if hasattr(source_video, "get_active_trim_window"):
        start_time, duration = source_video.get_active_trim_window()
        if float(start_time) != 0.0 or float(duration) != 0.0:
            raise ValueError("trimmed VIDEO packet-copy is not supported; finalize the untrimmed source")
    frame_count, height, width, _ = _validate_frames(processed_frames, name="processed_frames")
    if int(source_video.get_bit_depth()) > 8:
        raise ValueError("P1 VIDEO finalization is SDR 8-bit only; HDR/10-bit remains a separate P2 path")
    if tuple(map(int, source_video.get_dimensions())) != (width, height):
        raise ValueError("processed_frames dimensions must match the unrotated source VIDEO")
    source_frame_count = int(source_video.get_frame_count())
    if source_frame_count != frame_count:
        raise ValueError(
            f"processed_frames has {frame_count} frames but source VIDEO reports {source_frame_count}"
        )
    output_path, saved_name, subfolder = _resolve_output_path(filename_prefix, width, height)
    temporary = output_path.with_name(
        f"{output_path.stem}.partial-{uuid.uuid4().hex}{output_path.suffix}"
    )
    started = time.perf_counter()
    source_audio = None
    output_audio = None
    decoded_video_frames = 0
    try:
        import av

        with av.open(str(source_path), mode="r") as source_container:
            video_stream = source_container.streams.video[0] if source_container.streams.video else None
            if video_stream is None:
                raise ValueError("source VIDEO contains no video stream")
            if getattr(video_stream, "rotation", 0):
                raise ValueError("rotated VIDEO is not supported by the exact-geometry P1 finalizer")
            audio_streams = list(source_container.streams.audio)
            for stream in audio_streams:
                if stream.codec_context is None:
                    raise ValueError("source contains an audio stream that cannot be packet-copied")
                if stream.codec.name not in {"aac", "mp3", "alac", "ac3", "eac3"}:
                    raise ValueError(
                        f"audio codec {stream.codec.name!r} is not approved for MP4 packet-copy"
                    )
            source_audio = _audio_packet_digest(source_container, audio_streams)

        with av.open(str(source_path), mode="r") as source_container:
            video_stream = source_container.streams.video[0]
            audio_streams = list(source_container.streams.audio)
            rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(24)
            with av.open(
                str(temporary),
                mode="w",
                format="mp4",
                options={"movflags": "use_metadata_tags+faststart"},
            ) as output_container:
                for key, value in source_container.metadata.items():
                    output_container.metadata[str(key)] = str(value)
                out_video = output_container.add_stream("libx264", rate=rate)
                out_video.width = width
                out_video.height = height
                out_video.pix_fmt = "yuv420p"
                out_video.codec_context.max_b_frames = 0
                out_video.codec_context.thread_count = 1
                out_video.options = {
                    "crf": str(float(crf)),
                    "preset": "medium",
                    "threads": "1",
                }
                out_video.codec_context.time_base = video_stream.time_base
                audio_map = {
                    stream: output_container.add_stream_from_template(stream, opaque=True)
                    for stream in audio_streams
                }
                streams = [video_stream, *audio_streams]
                for packet in source_container.demux(*streams):
                    if packet.stream == video_stream:
                        for source_frame in packet.decode():
                            if decoded_video_frames >= frame_count:
                                raise ValueError("source VIDEO decoded more frames than its reported count")
                            rgb = (
                                processed_frames[decoded_video_frames, ..., :3]
                                .detach()
                                .clamp(0.0, 1.0)
                                .mul(255.0)
                                .byte()
                                .cpu()
                                .contiguous()
                                .numpy()
                            )
                            target = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                            target.pts = source_frame.pts
                            target.time_base = video_stream.time_base
                            for encoded in out_video.encode(target):
                                output_container.mux(encoded)
                            decoded_video_frames += 1
                    elif packet.stream in audio_map and packet.dts is not None:
                        packet.stream = audio_map[packet.stream]
                        output_container.mux(packet)
                for encoded in out_video.encode():
                    output_container.mux(encoded)
        if decoded_video_frames != frame_count:
            raise ValueError(
                f"source VIDEO decoded {decoded_video_frames} frames, expected {frame_count}"
            )
        with av.open(str(temporary), mode="r") as output_container:
            output_audio = _audio_packet_digest(output_container, list(output_container.streams.audio))
        if source_audio != output_audio:
            raise RuntimeError(
                "output audio packet payloads differ from the source; refusing to publish the file"
            )
        verified_frames = 0
        with av.open(str(temporary), mode="r") as output_container:
            if not output_container.streams.video:
                raise RuntimeError("finalized file contains no decodable video stream")
            verified_stream = output_container.streams.video[0]
            if (int(verified_stream.width), int(verified_stream.height)) != (width, height):
                raise RuntimeError("finalized file geometry differs from processed_frames")
            for verified_frame in output_container.decode(verified_stream):
                if verified_frame.width != width or verified_frame.height != height:
                    raise RuntimeError("finalized file changes geometry mid-stream")
                verified_frames += 1
        if verified_frames != frame_count:
            raise RuntimeError(
                f"finalized file decodes {verified_frames} frames, expected {frame_count}"
            )
        _strict_validate_encoded_video(temporary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    report = {
        "schema": SKIN_FINISH_VIDEO_REPORT_SCHEMA,
        "status": "CANDIDATE_VIDEO_FINALIZED",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "video": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "codec": "libx264",
            "encoder_threads": 1,
            "crf": float(crf),
            "reencoded": True,
            "sdr_8bit_only": True,
            "strict_decoded_frame_count_verified": verified_frames,
            "strict_decode_policy": SKIN_FINISH_VIDEO_STRICT_DECODE_POLICY,
        },
        "audio": {
            "method": "source_packet_payload_copy",
            "provided": bool(source_audio and int(source_audio["packet_count"]) > 0),
            "source": source_audio,
            "output": output_audio,
            "packet_payload_exact": True,
            "decoded_pcm_reencode": False,
        },
        "atomic_publish": True,
        "source_overwritten": False,
        "automatic_accept": False,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    saved_result = (saved_name, subfolder)
    return InputImpl.VideoFromFile(str(output_path)), str(output_path), canonical_json(report), saved_result
