from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as torch_functional

import comfy.model_management
import comfy.nested_tensor

from .core import nested_av_parts, split_noise_masks, video_latent_t
from .sampling import setup_dual_clock_sampling


PLAN_SCHEMA = "h3_t8_face_refine_plan/v1"
MANUAL_DETECTOR = "<manual ROI - no detector>"
NO_LOCAL_DETECTOR = "<no local face detector found>"
CANVAS_OPTIONS = ("auto_512", "384", "512", "640", "768")
YUNET_2023MAR_RELATIVE = "face_detection/face_detection_yunet_2023mar.onnx"
YUNET_2023MAR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_2023MAR_SOURCE = (
    "https://github.com/opencv/opencv_zoo/blob/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
ANIME_FACE_V14_N_RELATIVE = "face_detection/anime_face_detect_v1.4_n.onnx"
ANIME_FACE_V14_N_SHA256 = "fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9"
ANIME_FACE_V14_N_SOURCE = (
    "https://huggingface.co/deepghs/anime_face_detection/blob/"
    "784dc4c0bb692351ddcdbe6131a050b17d3025d5/face_detect_v1.4_n/model.onnx"
)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _model_root() -> Path:
    import folder_paths

    return Path(folder_paths.models_dir).resolve()


def local_face_detector_options() -> list[str]:
    root = _model_root()
    candidates: set[str] = set()
    search_roots = [
        root / "face_detection",
        root / "yolo",
        root / "detection",
        root / "facedetection",
        root / "SVFR",
    ]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for suffix in ("*.pt", "*.onnx"):
            for path in search_root.rglob(suffix):
                lower = path.name.lower()
                if "face" not in lower and "yolo" not in lower:
                    continue
                candidates.add(path.resolve().relative_to(root).as_posix())
    ordered = sorted(candidates)
    if YUNET_2023MAR_RELATIVE in ordered:
        ordered.remove(YUNET_2023MAR_RELATIVE)
        ordered.insert(0, YUNET_2023MAR_RELATIVE)
    if ANIME_FACE_V14_N_RELATIVE in ordered:
        ordered.remove(ANIME_FACE_V14_N_RELATIVE)
        ordered.insert(
            1 if ordered and ordered[0] == YUNET_2023MAR_RELATIVE else 0,
            ANIME_FACE_V14_N_RELATIVE,
        )
    return ordered or [NO_LOCAL_DETECTOR]


def _resolve_detector_path(relative_name: str) -> Path:
    if relative_name in {"", MANUAL_DETECTOR, NO_LOCAL_DETECTOR}:
        raise ValueError(
            "automatic face detection requires a local detector under ComfyUI/models; "
            "no model is downloaded automatically"
        )
    root = _model_root()
    path = (root / str(relative_name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("detector_model must stay inside ComfyUI/models") from error
    if not path.is_file() or path.suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError(f"Local detector does not exist or is unsupported: {relative_name}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frames(frames: torch.Tensor, *, name: str = "frames") -> tuple[int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"{name} must be IMAGE [N,H,W,C], got {shape}")
    frame_count, height, width, channels = map(int, frames.shape)
    if frame_count < 1 or height < 2 or width < 2 or channels < 3:
        raise ValueError(f"{name} is empty or has an unsupported shape: {tuple(frames.shape)}")
    if not torch.isfinite(frames[..., :3]).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return frame_count, height, width


def _lowres_proxy(frames: torch.Tensor, size: int = 32, chunk_frames: int = 8) -> torch.Tensor:
    chunks = []
    for start in range(0, int(frames.shape[0]), max(1, int(chunk_frames))):
        chunk = frames[start : start + max(1, int(chunk_frames)), ..., :3]
        chunk = chunk.detach().float().movedim(-1, 1)
        chunk = torch_functional.interpolate(
            chunk,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0)


def source_proxy_sha256(frames: torch.Tensor) -> str:
    proxy = _lowres_proxy(frames, size=16)
    digest = hashlib.sha256()
    digest.update(str(tuple(frames.shape)).encode("ascii"))
    digest.update(str(frames.dtype).encode("ascii"))
    digest.update(proxy.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _scene_ranges(frames: torch.Tensor, threshold: float) -> tuple[list[tuple[int, int]], list[float]]:
    proxy = _lowres_proxy(frames, size=32)
    deltas = [0.0]
    if proxy.shape[0] > 1:
        values = (proxy[1:] - proxy[:-1]).abs().mean(dim=(1, 2, 3))
        deltas.extend(float(value) for value in values)
    cuts = [index for index, value in enumerate(deltas) if index > 0 and value >= threshold]
    starts = [0, *cuts]
    ends = [value - 1 for value in cuts] + [int(frames.shape[0]) - 1]
    return list(zip(starts, ends, strict=True)), deltas


def _manual_box(width: int, height: int, x: float, y: float, w: float, h: float) -> list[float]:
    left = max(0.0, min(float(width - 1), x * width))
    top = max(0.0, min(float(height - 1), y * height))
    right = max(left + 1.0, min(float(width), (x + w) * width))
    bottom = max(top + 1.0, min(float(height), (y + h) * height))
    return [left, top, right, bottom]


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(1e-6, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1e-6, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(1e-6, first_area + second_area - intersection)


def _box_center_size(box: list[float]) -> tuple[float, float, float, float]:
    return (
        (box[0] + box[2]) * 0.5,
        (box[1] + box[3]) * 0.5,
        max(1.0, box[2] - box[0]),
        max(1.0, box[3] - box[1]),
    )


def _detect_local_opencv_yunet(
    frames: torch.Tensor,
    detector_model: str,
    confidence: float,
    detector_device: str,
) -> tuple[list[list[dict]], dict]:
    path = _resolve_detector_path(detector_model)
    if path.suffix.lower() != ".onnx":
        raise ValueError("local_opencv_yunet requires a YuNet .onnx detector")
    try:
        import cv2
    except Exception as error:
        raise RuntimeError(
            "local_opencv_yunet requires OpenCV with FaceDetectorYN support"
        ) from error
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not provide FaceDetectorYN")

    frame_count, height, width = _validate_frames(frames)
    model = None
    detections: list[list[dict]] = []
    try:
        model = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (width, height),
            float(confidence),
            0.30,
            5000,
        )
        for frame in frames:
            rgb = (
                frame[..., :3]
                .detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            bgr = rgb[..., ::-1].copy()
            _, result = model.detect(bgr)
            frame_items = []
            if result is not None:
                for row in result:
                    values = [float(value) for value in row.tolist()]
                    if len(values) < 5:
                        continue
                    x, y, box_width, box_height = values[:4]
                    left = max(0.0, min(float(width - 1), x))
                    top = max(0.0, min(float(height - 1), y))
                    right = max(left + 1.0, min(float(width), x + box_width))
                    bottom = max(top + 1.0, min(float(height), y + box_height))
                    confidence_value = values[14] if len(values) >= 15 else values[-1]
                    item = {
                        "box": [left, top, right, bottom],
                        "confidence": float(confidence_value),
                    }
                    if len(values) >= 14:
                        item["landmarks_xy"] = [
                            [values[index], values[index + 1]] for index in range(4, 14, 2)
                        ]
                    frame_items.append(item)
            detections.append(frame_items)
    except Exception as error:
        raise RuntimeError(
            f"Local detector {path.name!r} could not run through OpenCV FaceDetectorYN. "
            "Use the pinned OpenCV Zoo YuNet 2023mar ONNX model or select another backend."
        ) from error
    finally:
        del model
        gc.collect()

    model_hash = _file_sha256(path)
    official_match = model_hash == YUNET_2023MAR_SHA256
    return detections, {
        "backend": "local_opencv_yunet",
        "model": path.relative_to(_model_root()).as_posix(),
        "model_sha256": model_hash,
        "official_opencv_zoo_match": official_match,
        "model_source": YUNET_2023MAR_SOURCE if official_match else None,
        "model_license": "MIT" if official_match else "user_supplied_unverified",
        "requested_device": detector_device,
        "effective_device": "cpu",
        "frame_count": frame_count,
        "network_download": False,
        "cached_after_execute": False,
        "detector_object_released": True,
        "process_global_allocator_release_guaranteed": False,
        "five_point_landmarks": True,
    }


def _numpy_nms_xyxy(boxes, scores, iou_threshold: float):
    import numpy as np

    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        inter_x1 = np.maximum(x1[index], x1[remaining])
        inter_y1 = np.maximum(y1[index], y1[remaining])
        inter_x2 = np.minimum(x2[index], x2[remaining])
        inter_y2 = np.minimum(y2[index], y2[remaining])
        intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(
            0.0, inter_y2 - inter_y1
        )
        union = areas[index] + areas[remaining] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[iou <= float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def _detect_local_anime_onnx_exp(
    frames: torch.Tensor,
    detector_model: str,
    confidence: float,
    detector_device: str,
) -> tuple[list[list[dict]], dict]:
    path = _resolve_detector_path(detector_model)
    if path.suffix.lower() != ".onnx":
        raise ValueError("local_anime_onnx_exp requires the pinned anime YOLOv8 .onnx detector")
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as error:
        raise RuntimeError(
            "local_anime_onnx_exp requires the optional onnxruntime package"
        ) from error

    frame_count, height, width = _validate_frames(frames)
    session = None
    detections: list[list[dict]] = []
    input_size = 640
    try:
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("expected one input and one output")
        input_name = inputs[0].name
        output_name = outputs[0].name

        for frame in frames:
            image = frame[..., :3].detach().clamp(0.0, 1.0).float().cpu()
            image = torch_functional.interpolate(
                image.movedim(-1, 0).unsqueeze(0),
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            tensor = image.contiguous().numpy().astype(np.float32, copy=False)[None]
            output = np.asarray(session.run([output_name], {input_name: tensor})[0])
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError(f"unexpected output shape {tuple(output.shape)}")
            predictions = output[0]
            if predictions.shape[0] == 5:
                predictions = predictions.T
            if predictions.ndim != 2 or predictions.shape[1] != 5:
                raise ValueError(f"unexpected YOLOv8 face tensor {tuple(predictions.shape)}")

            predictions = predictions[predictions[:, 4] >= float(confidence)]
            frame_items = []
            if predictions.size:
                center_x, center_y, box_width, box_height, scores = predictions.T
                boxes = np.stack(
                    [
                        center_x - box_width * 0.5,
                        center_y - box_height * 0.5,
                        center_x + box_width * 0.5,
                        center_y + box_height * 0.5,
                    ],
                    axis=1,
                )
                keep = _numpy_nms_xyxy(boxes, scores, 0.70)
                for box, score in zip(boxes[keep], scores[keep], strict=True):
                    left = max(0.0, min(float(width - 1), float(box[0]) / input_size * width))
                    top = max(0.0, min(float(height - 1), float(box[1]) / input_size * height))
                    right = max(left + 1.0, min(float(width), float(box[2]) / input_size * width))
                    bottom = max(top + 1.0, min(float(height), float(box[3]) / input_size * height))
                    frame_items.append(
                        {
                            "box": [left, top, right, bottom],
                            "confidence": float(score),
                        }
                    )
            detections.append(frame_items)
    except Exception as error:
        raise RuntimeError(
            f"Local anime detector {path.name!r} could not run through the isolated ONNX "
            "backend. Use the pinned deepghs v1.4 nano model; other ONNX layouts fail closed."
        ) from error
    finally:
        del session
        gc.collect()

    model_hash = _file_sha256(path)
    official_match = model_hash == ANIME_FACE_V14_N_SHA256
    return detections, {
        "backend": "local_anime_onnx_exp",
        "domain": "anime_only_experimental",
        "model": path.relative_to(_model_root()).as_posix(),
        "model_sha256": model_hash,
        "official_deepghs_match": official_match,
        "model_source": ANIME_FACE_V14_N_SOURCE if official_match else None,
        "model_license": "MIT" if official_match else "user_supplied_unverified",
        "requested_device": detector_device,
        "effective_device": "cpu",
        "frame_count": frame_count,
        "network_download": False,
        "cached_after_execute": False,
        "detector_object_released": True,
        "process_global_allocator_release_guaranteed": False,
        "five_point_landmarks": False,
        "warning": "Anime-only experimental detector; do not use its boxes as identity proof.",
    }


def _detect_local_ultralytics(
    frames: torch.Tensor,
    detector_model: str,
    confidence: float,
    detector_device: str,
) -> tuple[list[list[dict]], dict]:
    path = _resolve_detector_path(detector_model)
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError(
            "local_ultralytics requires the optional ultralytics package; "
            "use manual_static_roi when it is unavailable"
        ) from error

    device = "cpu"
    if detector_device == "cuda_auto" and torch.cuda.is_available():
        device = "0"
    model = None
    detections: list[list[dict]] = []
    try:
        model = YOLO(str(path))
        for frame in frames:
            image = (
                frame[..., :3]
                .detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            result = model.predict(
                source=image,
                conf=float(confidence),
                device=device,
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            frame_items = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().float().cpu()
                scores = boxes.conf.detach().float().cpu()
                for box, score in zip(xyxy.tolist(), scores.tolist(), strict=True):
                    frame_items.append({"box": [float(v) for v in box], "confidence": float(score)})
            detections.append(frame_items)
    except Exception as error:
        raise RuntimeError(
            f"Local detector {path.name!r} could not run through Ultralytics. "
            "The file may be TorchScript, YOLOv5-face, or another incompatible format. "
            "Use manual_static_roi or install a locally licensed Ultralytics-compatible face "
            "detector; this node will not download or convert one automatically."
        ) from error
    finally:
        del model
        gc.collect()
        if device != "cpu":
            comfy.model_management.soft_empty_cache()
    return detections, {
        "backend": "local_ultralytics",
        "model": path.relative_to(_model_root()).as_posix(),
        "device": "cuda_auto" if device != "cpu" else "cpu",
        "network_download": False,
        "cached_after_execute": False,
    }


def _select_track(
    detections: list[list[dict]],
    shot_ranges: list[tuple[int, int]],
    width: int,
    height: int,
    max_track_jump: float,
    max_gap_frames: int,
) -> tuple[list[list[float] | None], list[str], list[float], int]:
    boxes: list[list[float] | None] = [None] * len(detections)
    states = ["lost"] * len(detections)
    weights = [0.0] * len(detections)
    multi_face_frames = sum(1 for value in detections if len(value) > 1)
    diagonal = math.hypot(width, height)

    for shot_start, shot_end in shot_ranges:
        previous = None
        previous_center = None
        velocity = (0.0, 0.0)
        previous_index = None
        for index in range(shot_start, shot_end + 1):
            candidates = detections[index]
            chosen = None
            unverified_reacquire = False
            if candidates:
                if previous is None:
                    chosen = max(
                        candidates,
                        key=lambda item: (item["box"][2] - item["box"][0])
                        * (item["box"][3] - item["box"][1]),
                    )
                else:
                    px, py, pw, ph = _box_center_size(previous)
                    gap = max(1, index - int(previous_index))
                    predicted = (px + velocity[0] * gap, py + velocity[1] * gap)

                    def cost(item):
                        cx, cy, cw, ch = _box_center_size(item["box"])
                        predicted_distance = math.hypot(
                            cx - predicted[0], cy - predicted[1]
                        ) / diagonal
                        previous_distance = math.hypot(cx - px, cy - py) / diagonal
                        distance = min(predicted_distance, previous_distance)
                        size_cost = abs(math.log(max(cw * ch, 1.0) / max(pw * ph, 1.0)))
                        return distance + 0.20 * size_cost + 0.35 * (1.0 - _box_iou(previous, item["box"]))

                    candidate = min(candidates, key=cost)
                    cx, cy, _, _ = _box_center_size(candidate["box"])
                    candidate_jump = min(
                        math.hypot(cx - predicted[0], cy - predicted[1]) / diagonal,
                        math.hypot(cx - px, cy - py) / diagonal,
                    )
                    if candidate_jump <= max_track_jump:
                        chosen = candidate
                    elif (
                        len(candidates) == 1
                        and previous_index is not None
                        and index - int(previous_index) > max_gap_frames
                    ):
                        chosen = candidate
                        unverified_reacquire = True
            if chosen is None:
                continue
            current = [float(value) for value in chosen["box"]]
            boxes[index] = current
            reacquired = previous_index is not None and index - int(previous_index) > 1
            if unverified_reacquire:
                states[index] = "reacquired_unverified"
                weights[index] = 0.45
            else:
                states[index] = "reacquired" if reacquired else "detected"
                weights[index] = 0.85 if reacquired else 1.0
            center = _box_center_size(current)[:2]
            if previous_center is not None:
                elapsed = max(1, index - int(previous_index))
                instantaneous = (
                    (center[0] - previous_center[0]) / elapsed,
                    (center[1] - previous_center[1]) / elapsed,
                )
                velocity = (
                    0.65 * velocity[0] + 0.35 * instantaneous[0],
                    0.65 * velocity[1] + 0.35 * instantaneous[1],
                )
            previous_center = center
            previous = current
            previous_index = index

        detected = [index for index in range(shot_start, shot_end + 1) if boxes[index] is not None]
        for left_index, right_index in zip(detected, detected[1:]):
            gap = right_index - left_index - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            for offset in range(1, gap + 1):
                mix = offset / (gap + 1)
                boxes[left_index + offset] = [
                    (1.0 - mix) * first + mix * second
                    for first, second in zip(boxes[left_index], boxes[right_index], strict=True)
                ]
                states[left_index + offset] = "interpolated"
                weights[left_index + offset] = 0.55

        if detected:
            first, last = detected[0], detected[-1]
            for index in range(max(shot_start, first - max_gap_frames), first):
                boxes[index] = list(boxes[first])
                states[index] = "occluded"
                weights[index] = 0.30
            for index in range(last + 1, min(shot_end + 1, last + max_gap_frames + 1)):
                boxes[index] = list(boxes[last])
                states[index] = "occluded"
                weights[index] = 0.30
    return boxes, states, weights, multi_face_frames


def _boxes_for_crop_context(
    boxes: list[list[float] | None],
    shot_ranges: list[tuple[int, int]],
    fallback: list[float],
) -> list[list[float]]:
    output: list[list[float] | None] = [None if box is None else list(box) for box in boxes]
    for shot_start, shot_end in shot_ranges:
        reliable = [index for index in range(shot_start, shot_end + 1) if boxes[index] is not None]
        if not reliable:
            for index in range(shot_start, shot_end + 1):
                output[index] = list(fallback)
            continue
        for index in range(shot_start, shot_end + 1):
            if output[index] is not None:
                continue
            nearest = min(reliable, key=lambda candidate: abs(candidate - index))
            output[index] = list(boxes[nearest])
    return [list(box) for box in output]


def _smooth_boxes(
    boxes: list[list[float] | None],
    states: list[str],
    shot_ranges: list[tuple[int, int]],
    radius: int,
) -> list[list[float] | None]:
    if radius <= 0:
        return [None if box is None else list(box) for box in boxes]
    output = [None if box is None else list(box) for box in boxes]
    for shot_start, shot_end in shot_ranges:
        for index in range(shot_start, shot_end + 1):
            if boxes[index] is None or states[index] == "lost":
                continue
            values = [
                boxes[candidate]
                for candidate in range(max(shot_start, index - radius), min(shot_end, index + radius) + 1)
                if boxes[candidate] is not None and states[candidate] != "lost"
            ]
            if values:
                output[index] = [sum(value[axis] for value in values) / len(values) for axis in range(4)]
    return output


def _crop_box(face_box: list[float], width: int, height: int, context_scale: float) -> list[float]:
    cx, cy, face_width, face_height = _box_center_size(face_box)
    side = min(float(min(width, height)), max(32.0, max(face_width, face_height) * context_scale))
    left = min(max(0.0, cx - side * 0.5), width - side)
    top = min(max(0.0, cy - side * 0.5), height - side)
    return [left, top, left + side, top + side]


def _face_box_in_crop(face_box: list[float], crop_box: list[float], canvas: int) -> list[float]:
    left, top, right, bottom = crop_box
    width = max(1e-6, right - left)
    height = max(1e-6, bottom - top)
    return [
        max(0.0, min(float(canvas), (face_box[0] - left) / width * canvas)),
        max(0.0, min(float(canvas), (face_box[1] - top) / height * canvas)),
        max(0.0, min(float(canvas), (face_box[2] - left) / width * canvas)),
        max(0.0, min(float(canvas), (face_box[3] - top) / height * canvas)),
    ]


def _crop_chunks(
    frames: torch.Tensor,
    crop_boxes: list[list[float]],
    canvas: int,
    chunk_frames: int,
) -> torch.Tensor:
    frame_count, height, width, _ = frames.shape
    outputs = []
    for start in range(0, frame_count, max(1, chunk_frames)):
        end = min(frame_count, start + max(1, chunk_frames))
        chunk = frames[start:end, ..., :3].movedim(-1, 1).float()
        theta = chunk.new_zeros((end - start, 2, 3))
        for local_index, box in enumerate(crop_boxes[start:end]):
            left, top, right, bottom = box
            theta[local_index, 0, 0] = (right - left) / width
            theta[local_index, 1, 1] = (bottom - top) / height
            theta[local_index, 0, 2] = (left + right) / width - 1.0
            theta[local_index, 1, 2] = (top + bottom) / height - 1.0
        grid = torch_functional.affine_grid(
            theta,
            size=(end - start, 3, canvas, canvas),
            align_corners=False,
        )
        output = torch_functional.grid_sample(
            chunk,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        outputs.append(output.movedim(1, -1).to(dtype=frames.dtype, device=frames.device))
    return torch.cat(outputs, dim=0)


def _draw_preview(
    frames: torch.Tensor,
    boxes: list[list[float] | None],
    states: list[str],
    sample_count: int = 8,
) -> tuple[torch.Tensor, list[int]]:
    count, height, width, _ = frames.shape
    if count <= sample_count:
        indices = list(range(count))
    else:
        indices = sorted(
            {
                round(index * (count - 1) / max(1, sample_count - 1))
                for index in range(sample_count)
            }
        )
    preview = frames[indices, ..., :3].clone()
    thickness = max(1, round(min(width, height) / 256))
    colors = {
        "detected": preview.new_tensor([0.1, 1.0, 0.1]),
        "reacquired": preview.new_tensor([0.1, 0.8, 1.0]),
        "reacquired_unverified": preview.new_tensor([0.8, 0.25, 1.0]),
        "interpolated": preview.new_tensor([1.0, 0.75, 0.1]),
        "occluded": preview.new_tensor([1.0, 0.35, 0.1]),
        "lost": preview.new_tensor([1.0, 0.1, 0.1]),
    }
    for output_index, frame_index in enumerate(indices):
        box = boxes[frame_index]
        if box is None:
            continue
        left, top, right, bottom = [int(round(value)) for value in box]
        left, right = max(0, left), min(width - 1, right)
        top, bottom = max(0, top), min(height - 1, bottom)
        color = colors[states[frame_index]]
        preview[output_index, top : min(height, top + thickness), left : right + 1] = color
        preview[output_index, max(0, bottom - thickness + 1) : bottom + 1, left : right + 1] = color
        preview[output_index, top : bottom + 1, left : min(width, left + thickness)] = color
        preview[output_index, top : bottom + 1, max(0, right - thickness + 1) : right + 1] = color
    return preview, indices


def _validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"face_plan must use {PLAN_SCHEMA}")
    source = plan.get("source")
    canvas = plan.get("canvas")
    frames = plan.get("frames")
    if not isinstance(source, dict) or not isinstance(canvas, dict) or not isinstance(frames, list):
        raise ValueError("face_plan is missing source, canvas, or frame records")
    if len(frames) != int(source.get("frame_count", -1)):
        raise ValueError("face_plan frame records do not match source.frame_count")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected = hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
    if plan.get("plan_sha256") != expected:
        raise ValueError("face_plan hash mismatch; the plan may be stale or modified")
    return plan


def build_face_refine_plan(
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
    smoothing_radius: int,
    crop_context_scale: float,
    canvas_size: str,
    require_h3_grid: bool,
    analysis_chunk_frames: int,
):
    frame_count, height, width = _validate_frames(frames)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if abs(float(fps) - 24.0) > 0.01:
        raise ValueError(
            f"The tensor route requires an exact 24fps source; got {fps:.6g}fps. "
            "Resample explicitly before planning so video and audio timing stay auditable."
        )
    if require_h3_grid and frame_count % 17 != 5:
        raise ValueError(
            f"Face refine requires an exact H3 17n+5 sequence; got {frame_count} frames"
        )
    if frame_count > 362:
        raise ValueError(
            "The tensor route is capped at 362 frames to avoid multi-gigabyte IMAGE duplication; "
            "split long videos into accepted repair windows"
        )
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
            "network_download": False,
            "cached_after_execute": False,
            "identity_verification": False,
        }
    elif detector_mode == "local_opencv_yunet":
        detections, detector_report = _detect_local_opencv_yunet(
            frames,
            detector_model,
            confidence,
            detector_device,
        )
        detector_report["identity_verification"] = False
    elif detector_mode == "local_anime_onnx_exp":
        detections, detector_report = _detect_local_anime_onnx_exp(
            frames,
            detector_model,
            confidence,
            detector_device,
        )
        detector_report["identity_verification"] = False
    elif detector_mode == "local_ultralytics":
        detections, detector_report = _detect_local_ultralytics(
            frames,
            detector_model,
            confidence,
            detector_device,
        )
        detector_report["identity_verification"] = False
    else:
        raise ValueError(f"Unknown detector_mode: {detector_mode}")

    boxes, states, weights, multi_face_frames = _select_track(
        detections,
        shot_ranges,
        width,
        height,
        float(max_track_jump),
        int(max_gap_frames),
    )
    boxes = _smooth_boxes(boxes, states, shot_ranges, int(smoothing_radius))
    fallback = _manual_box(
        width,
        height,
        manual_roi_x,
        manual_roi_y,
        manual_roi_width,
        manual_roi_height,
    )
    if not any(box is not None for box in boxes):
        raise ValueError("No usable face boxes were found in any frame")
    crop_context_boxes = _boxes_for_crop_context(boxes, shot_ranges, fallback)
    crop_boxes = [
        _crop_box(box, width, height, crop_context_scale) for box in crop_context_boxes
    ]
    if canvas_size not in CANVAS_OPTIONS:
        raise ValueError(f"Unsupported canvas_size: {canvas_size}")
    canvas = 512 if canvas_size == "auto_512" else int(canvas_size)
    crops = _crop_chunks(frames, crop_boxes, canvas, int(analysis_chunk_frames))
    preview, preview_indices = _draw_preview(frames, boxes, states)

    frame_records = []
    for index, (box, crop_box, state, weight) in enumerate(
        zip(boxes, crop_boxes, states, weights, strict=True)
    ):
        shot_id = next(
            shot_index
            for shot_index, (start, end) in enumerate(shot_ranges)
            if start <= index <= end
        )
        face_box = crop_context_boxes[index]
        frame_records.append(
            {
                "frame_index": index,
                "shot_id": shot_id,
                "state": state,
                "source_face_box_xyxy": [round(value, 6) for value in face_box],
                "source_crop_box_xyxy": [round(value, 6) for value in crop_box],
                "crop_face_box_xyxy": [
                    round(value, 6) for value in _face_box_in_crop(face_box, crop_box, canvas)
                ],
                "paste_weight": round(float(weight), 6),
            }
        )

    source_hash = source_proxy_sha256(frames)
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "experimental_candidate_plan",
        "source": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": float(fps),
            "proxy_sha256": source_hash,
        },
        "canvas": {"width": canvas, "height": canvas, "multiple": 32},
        "detector": detector_report,
        "shots": [
            {"shot_id": index, "start_frame": start, "end_frame": end}
            for index, (start, end) in enumerate(shot_ranges)
        ],
        "frames": frame_records,
        "preview_frame_indices": preview_indices,
        "limits": {
            "single_pass_safe": len(shot_ranges) == 1,
            "identity_verified": False,
            "automatic_accept": False,
            "audio_modified": False,
        },
        "metrics": {
            "scene_cut_count": len(shot_ranges) - 1,
            "max_scene_delta": max(scene_deltas),
            "detected_frames": states.count("detected"),
            "reacquired_frames": states.count("reacquired"),
            "reacquired_unverified_frames": states.count("reacquired_unverified"),
            "interpolated_frames": states.count("interpolated"),
            "occluded_frames": states.count("occluded"),
            "lost_frames": states.count("lost"),
            "multi_face_frames": multi_face_frames,
            "crop_tensor_estimated_mib": frame_count * canvas * canvas * 3 * 4 / 2**20,
        },
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    return plan, crops, preview, canonical_json(plan), canvas, canvas, frame_count


def inject_face_refine_video_latent(
    positive,
    av_latent: dict,
    crops: torch.Tensor,
    video_vae,
    face_plan: dict,
    audio_policy: str,
    allow_multi_shot_exp: bool,
):
    plan = _validate_plan(face_plan)
    frame_count, height, width = _validate_frames(crops, name="crops")
    source = plan["source"]
    canvas = plan["canvas"]
    if frame_count != int(source["frame_count"]):
        raise ValueError("crops frame count does not match face_plan")
    if (width, height) != (int(canvas["width"]), int(canvas["height"])):
        raise ValueError("crops canvas does not match face_plan")
    if len(plan["shots"]) > 1 and not allow_multi_shot_exp:
        raise ValueError(
            "face_plan contains scene cuts; split it into shot-local H3 windows or explicitly "
            "enable allow_multi_shot_exp"
        )

    video, audio = nested_av_parts(av_latent)
    expected_t = video_latent_t(frame_count)
    if int(video.shape[2]) != expected_t:
        raise ValueError(
            f"AV latent video time {video.shape[2]} does not match {frame_count} frames ({expected_t}); "
            "implicit trim/pad is forbidden"
        )
    video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    if audio_policy == "require_locked":
        if audio_mask is None:
            raise ValueError("require_locked needs a connected nested audio noise_mask")
        if not torch.count_nonzero(audio_mask).eq(0):
            raise ValueError("require_locked refuses a nonzero audio noise_mask")
    elif audio_policy != "preserve_existing":
        raise ValueError(f"Unknown audio_policy: {audio_policy}")

    encoded = video_vae.encode(crops)
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
        raise ValueError("video_vae must return MiniMax H3 video latent [B,C,T,H,W]")
    if tuple(encoded.shape) != tuple(video.shape):
        raise ValueError(
            "Encoded face video latent does not exactly match the target AV latent; "
            f"got {tuple(encoded.shape)}, expected {tuple(video.shape)}"
        )
    encoded = encoded.to(device=video.device, dtype=video.dtype)
    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((encoded, audio))
    report = {
        "schema": "h3_t8_face_refine_conditioning_report/v1",
        "status": "candidate_latent_injected",
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "video_latent_shape": list(encoded.shape),
        "audio_latent_shape": list(audio.shape),
        "audio_policy": audio_policy,
        "audio_tensor_reused": output["samples"].unbind()[1].data_ptr() == audio.data_ptr(),
        "noise_mask_object_reused": output.get("noise_mask") is av_latent.get("noise_mask"),
        "implicit_temporal_fit": False,
        "automatic_accept": False,
    }
    return positive, output, canonical_json(report)


def setup_face_refine_sampling(
    model,
    av_latent: dict,
    steps: int,
    denoise: float,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
):
    if steps < 1:
        raise ValueError("steps must be at least one")
    if not 0.0 < denoise <= 1.0:
        raise ValueError("denoise must be in (0, 1]")
    total_steps = max(int(steps), int(steps / denoise))
    patched_model, sampler, full_sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        total_steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    )
    sigmas = full_sigmas[-(int(steps) + 1) :]
    if sigmas.shape[-1] != int(steps) + 1:
        raise RuntimeError("Face-refine sigma truncation produced an unexpected schedule length")
    report = {
        "schema": "h3_t8_face_refine_sampling/v1",
        "status": "experimental_low_denoise_schedule",
        "requested_model_calls": int(steps),
        "full_schedule_steps": total_steps,
        "denoise": float(denoise),
        "starting_sigma_video": float(sigmas[0]),
        "ending_sigma_video": float(sigmas[-1]),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "calibrated_strength": False,
        "sampling_py_modified": False,
    }
    return patched_model, sampler, sigmas, canonical_json(report)


def _canvas_alpha(
    record: dict,
    canvas: int,
    paste_region: str,
    feather_source_px: float,
    blend_strength: float,
) -> torch.Tensor:
    face = record["crop_face_box_xyxy"]
    crop = record["source_crop_box_xyxy"]
    crop_source_width = max(1e-6, crop[2] - crop[0])
    feather = max(0.5, feather_source_px * canvas / crop_source_width)
    yy, xx = torch.meshgrid(
        torch.arange(canvas, dtype=torch.float32),
        torch.arange(canvas, dtype=torch.float32),
        indexing="ij",
    )
    if paste_region == "ellipse":
        cx, cy = (face[0] + face[2]) * 0.5, (face[1] + face[3]) * 0.5
        rx = max(1.0, (face[2] - face[0]) * 0.58)
        ry = max(1.0, (face[3] - face[1]) * 0.62)
        distance = torch.sqrt(((xx - cx) / rx).square() + ((yy - cy) / ry).square())
        normalized_feather = feather / max(rx, ry)
        alpha = ((1.0 + normalized_feather - distance) / normalized_feather).clamp(0.0, 1.0)
    elif paste_region == "rectangle":
        left = face[0] - (face[2] - face[0]) * 0.08
        right = face[2] + (face[2] - face[0]) * 0.08
        top = face[1] - (face[3] - face[1]) * 0.08
        bottom = face[3] + (face[3] - face[1]) * 0.08
        inside = torch.minimum(
            torch.minimum(xx - left, right - xx),
            torch.minimum(yy - top, bottom - yy),
        )
        alpha = ((inside + feather) / feather).clamp(0.0, 1.0)
    elif paste_region == "full_crop_exp":
        edge = torch.minimum(
            torch.minimum(xx, canvas - 1 - xx),
            torch.minimum(yy, canvas - 1 - yy),
        )
        alpha = (edge / feather).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown paste_region: {paste_region}")
    return alpha * float(record["paste_weight"]) * float(blend_strength)


def _warp_crop_to_source(
    crop: torch.Tensor,
    alpha: torch.Tensor,
    crop_box: list[float],
    output_height: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, top, right, bottom = crop_box
    crop_width = max(1e-6, right - left)
    crop_height = max(1e-6, bottom - top)
    theta = crop.new_zeros((1, 2, 3))
    theta[0, 0, 0] = output_width / crop_width
    theta[0, 1, 1] = output_height / crop_height
    theta[0, 0, 2] = (output_width - left - right) / crop_width
    theta[0, 1, 2] = (output_height - top - bottom) / crop_height
    grid = torch_functional.affine_grid(
        theta,
        size=(1, 3, output_height, output_width),
        align_corners=False,
    )
    patch = torch_functional.grid_sample(
        crop.movedim(-1, 0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].movedim(0, -1)
    mask = torch_functional.grid_sample(
        alpha.to(device=crop.device, dtype=crop.dtype)[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    return patch, mask.clamp(0.0, 1.0)


def _color_match(
    refined: torch.Tensor,
    original: torch.Tensor,
    alpha: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0:
        return refined
    weights = alpha[..., None]
    denominator = weights.sum(dim=(0, 1)).clamp_min(1.0)
    refined_mean = (refined * weights).sum(dim=(0, 1)) / denominator
    original_mean = (original * weights).sum(dim=(0, 1)) / denominator
    refined_var = ((refined - refined_mean).square() * weights).sum(dim=(0, 1)) / denominator
    original_var = ((original - original_mean).square() * weights).sum(dim=(0, 1)) / denominator
    scale = (original_var.clamp_min(1e-5).sqrt() / refined_var.clamp_min(1e-5).sqrt()).clamp(0.5, 2.0)
    matched = (refined - refined_mean) * scale + original_mean
    return torch.lerp(refined, matched.clamp(0.0, 1.0), float(strength))


def stitch_face_refine_candidate(
    base_frames: torch.Tensor,
    refined_crops: torch.Tensor,
    face_plan: dict,
    paste_region: str,
    feather_source_px: float,
    blend_strength: float,
    color_match_strength: float,
    max_face_mean_abs_delta: float,
    fallback_neighbor_frames: int,
    processing_device: str,
):
    plan = _validate_plan(face_plan)
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    crop_count, crop_height, crop_width = _validate_frames(refined_crops, name="refined_crops")
    source = plan["source"]
    canvas = plan["canvas"]
    if (frame_count, height, width) != (
        int(source["frame_count"]),
        int(source["height"]),
        int(source["width"]),
    ):
        raise ValueError("base_frames dimensions do not match face_plan")
    if source_proxy_sha256(base_frames) != source["proxy_sha256"]:
        raise ValueError("base_frames content fingerprint does not match face_plan")
    if crop_count != frame_count or (crop_width, crop_height) != (
        int(canvas["width"]),
        int(canvas["height"]),
    ):
        raise ValueError("refined_crops frame count or canvas does not match face_plan")
    if processing_device == "cuda_if_available" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif processing_device == "cpu_memory_safe":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown processing_device: {processing_device}")

    candidate_frames = []
    alpha_frames = []
    raw_fallback = []
    scores = []
    for index, record in enumerate(plan["frames"]):
        base = base_frames[index, ..., :3].detach().to(device=device, dtype=torch.float32)
        crop = refined_crops[index, ..., :3].detach().to(device=device, dtype=torch.float32)
        original_crop = _crop_chunks(
            base_frames[index : index + 1],
            [record["source_crop_box_xyxy"]],
            int(canvas["width"]),
            1,
        )[0].to(device=device, dtype=torch.float32)
        alpha_canvas = _canvas_alpha(
            record,
            int(canvas["width"]),
            paste_region,
            feather_source_px,
            blend_strength,
        ).to(device)
        crop = _color_match(crop, original_crop, alpha_canvas, color_match_strength)
        denominator = alpha_canvas.sum().clamp_min(1.0)
        score = float(
            ((crop - original_crop).abs().mean(dim=-1) * alpha_canvas).sum().div(denominator).item()
        )
        invalid = not math.isfinite(score) or score > max_face_mean_abs_delta
        raw_fallback.append(
            invalid or record["state"] in {"lost", "reacquired_unverified"}
        )
        scores.append(score)
        patch, alpha = _warp_crop_to_source(
            crop,
            alpha_canvas,
            record["source_crop_box_xyxy"],
            height,
            width,
        )
        candidate_frames.append((base, patch))
        alpha_frames.append(alpha)

    fallback = list(raw_fallback)
    radius = max(0, int(fallback_neighbor_frames))
    for index, failed in enumerate(raw_fallback):
        if not failed:
            continue
        for neighbor in range(max(0, index - radius), min(frame_count, index + radius + 1)):
            fallback[neighbor] = True

    output = []
    changed_masks = []
    fallback_masks = []
    for index, ((base, patch), alpha) in enumerate(zip(candidate_frames, alpha_frames, strict=True)):
        if fallback[index]:
            alpha = torch.zeros_like(alpha)
        blended = base * (1.0 - alpha[..., None]) + patch * alpha[..., None]
        blended = torch.where(alpha[..., None] > 0, blended, base)
        output.append(blended.clamp(0.0, 1.0).to(device="cpu", dtype=base_frames.dtype))
        changed_masks.append(alpha.to(device="cpu", dtype=base_frames.dtype))
        fallback_masks.append(
            torch.full((height, width), float(fallback[index]), dtype=base_frames.dtype)
        )

    result = torch.stack(output).to(base_frames.device)
    changed_mask = torch.stack(changed_masks).to(base_frames.device)
    fallback_mask = torch.stack(fallback_masks).to(base_frames.device)
    outside = changed_mask == 0
    outside_exact = bool(torch.equal(result[outside], base_frames[..., :3][outside]))
    report = {
        "schema": "h3_t8_face_refine_stitch_audit/v1",
        "status": "candidate_requires_review",
        "plan_sha256": plan["plan_sha256"],
        "frame_count": frame_count,
        "fallback_frames": [index for index, value in enumerate(fallback) if value],
        "fallback_count": sum(fallback),
        "mean_face_delta": sum(scores) / max(1, len(scores)),
        "max_face_delta": max(scores),
        "mask_outside_bit_exact": outside_exact,
        "processing_device": str(device),
        "audio_modified": False,
        "identity_verified": False,
        "automatic_accept": False,
    }
    if not outside_exact:
        raise RuntimeError("Stitch audit failed: pixels outside the paste mask changed")
    return result, changed_mask, fallback_mask, sum(fallback), canonical_json(report)
