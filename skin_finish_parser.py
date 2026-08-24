from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as torch_functional

from .skin_finish import (
    _interrupt_and_progress,
    _memory_snapshot,
    _progress_bar,
    _validate_frames,
    canonical_json,
)


SKIN_FINISH_SEMANTIC_MASK_SCHEMA = "h3_t8_skin_finish_semantic_mask/v1"
PARSENET_MODEL_NAME = "facexlib_parsenet_v0.2.2_pinned"
PARSENET_MODEL_RELATIVE = Path("facedetection") / "parsing_parsenet.pth"
PARSENET_MODEL_SIZE = 85_331_193
PARSENET_MODEL_SHA256 = (
    "3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2"
)
PARSENET_MODEL_SOURCE = (
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
    "parsing_parsenet.pth"
)
PARSENET_CODE_SOURCE = "https://github.com/xinntao/facexlib"
PARSENET_MAPPING_SOURCE = (
    "https://github.com/switchablenorms/CelebAMask-HQ/blob/master/"
    "face_parsing/README.md"
)

# This is the ParseNet/CelebAMask-HQ order used by the pinned checkpoint. It is not
# the differently ordered BiSeNet label list shown by some FaceXLib examples.
PARSENET_CLASS_NAMES = (
    "background",
    "skin",
    "nose",
    "eyeglasses",
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "left_ear",
    "right_ear",
    "mouth",
    "upper_lip",
    "lower_lip",
    "hair",
    "hat",
    "earring",
    "necklace",
    "neck",
    "cloth",
)
SKIN_CLASS = 1
NECK_CLASS = 17
FEATURE_CLASSES = frozenset({2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18})


class _ParserUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_path() -> Path:
    try:
        import folder_paths
    except Exception as error:
        raise _ParserUnavailable(
            "ABSTAIN_COMFYUI_MODEL_ROOT_UNAVAILABLE",
            f"ComfyUI model root is unavailable: {error}",
        ) from error
    return Path(folder_paths.models_dir) / PARSENET_MODEL_RELATIVE


def _load_pinned_parsenet(path: Path | None = None):
    resolved = _model_path() if path is None else Path(path)
    if not resolved.is_file():
        raise _ParserUnavailable(
            "ABSTAIN_PARSENET_MODEL_MISSING",
            f"Install the pinned checkpoint at {resolved}; runtime download is disabled.",
        )
    actual_size = int(resolved.stat().st_size)
    if actual_size != PARSENET_MODEL_SIZE:
        raise _ParserUnavailable(
            "REJECT_PARSENET_MODEL_SIZE_MISMATCH",
            f"Expected {PARSENET_MODEL_SIZE} bytes, got {actual_size} at {resolved}.",
        )
    actual_hash = _file_sha256(resolved)
    if actual_hash.lower() != PARSENET_MODEL_SHA256:
        raise _ParserUnavailable(
            "REJECT_PARSENET_MODEL_HASH_MISMATCH",
            f"Expected SHA-256 {PARSENET_MODEL_SHA256}, got {actual_hash}.",
        )
    try:
        from facexlib.parsing.parsenet import ParseNet
    except Exception as error:
        raise _ParserUnavailable(
            "ABSTAIN_FACEXLIB_DEPENDENCY_MISSING",
            f"FaceXLib ParseNet is unavailable: {error}",
        ) from error
    try:
        state = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise _ParserUnavailable(
            "REJECT_SAFE_TORCH_LOAD_UNAVAILABLE",
            "This Torch build cannot load the pinned pickle checkpoint with weights_only=True.",
        ) from error
    except Exception as error:
        raise _ParserUnavailable(
            "REJECT_PARSENET_SAFE_LOAD_FAILED",
            f"Pinned ParseNet checkpoint failed safe loading: {error}",
        ) from error
    try:
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device="cpu", dtype=torch.float32)
    except Exception as error:
        raise _ParserUnavailable(
            "REJECT_PARSENET_STATE_DICT_MISMATCH",
            f"Pinned ParseNet state dict does not match the expected architecture: {error}",
        ) from error
    finally:
        del state
    return model, resolved, actual_hash


def _validate_bound_face_plan(frames: torch.Tensor, face_plan: dict) -> dict:
    try:
        from .face_refine_advanced import _validate_plan, source_proxy_sha256

        plan = _validate_plan(face_plan)
    except Exception as error:
        raise _ParserUnavailable(
            "ABSTAIN_FACE_PLAN_MISSING_OR_INVALID", str(error)
        ) from error
    frame_count, height, width, _ = _validate_frames(frames)
    source = plan["source"]
    if (
        int(source.get("frame_count", -1)) != frame_count
        or int(source.get("height", -1)) != height
        or int(source.get("width", -1)) != width
    ):
        raise _ParserUnavailable(
            "ABSTAIN_FACE_PLAN_GEOMETRY_MISMATCH",
            "face_plan source geometry does not match frames.",
        )
    if str(source.get("proxy_sha256", "")) != source_proxy_sha256(frames):
        raise _ParserUnavailable(
            "ABSTAIN_FACE_PLAN_SOURCE_MISMATCH",
            "face_plan is bound to different source pixels.",
        )
    return plan


def _square_crop_box(
    box: list[float], width: int, height: int, expansion: float
) -> tuple[int, int, int, int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    left, top, right, bottom = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    box_width = right - left
    box_height = bottom - top
    if box_width < 8.0 or box_height < 8.0:
        return None
    side = max(box_width, box_height) * float(expansion)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    x1 = max(0, int(math.floor(center_x - side * 0.5)))
    y1 = max(0, int(math.floor(center_y - side * 0.5)))
    x2 = min(width, int(math.ceil(center_x + side * 0.5)))
    y2 = min(height, int(math.ceil(center_y + side * 0.5)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def _parser_logits(model, crop: torch.Tensor) -> torch.Tensor:
    resized = torch_functional.interpolate(
        crop.movedim(-1, 1),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        output = model(resized.mul(2.0).sub(1.0))
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if tuple(logits.shape) != (1, 19, 512, 512):
        raise ValueError(f"ParseNet returned unexpected shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("ParseNet returned NaN or Inf")
    return logits.float()


def _semantic_local_masks(
    logits: torch.Tensor,
    *,
    include_neck: bool,
    minimum_class_probability: float,
    feature_protection_px: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    probabilities = logits.softmax(dim=1)
    labels = logits.argmax(dim=1)
    selected_classes = [SKIN_CLASS] + ([NECK_CLASS] if include_neck else [])
    selected_probability = probabilities[:, selected_classes].sum(dim=1)
    selected_label = torch.zeros_like(labels, dtype=torch.bool)
    for class_index in selected_classes:
        selected_label |= labels == class_index
    skin = selected_label & (selected_probability >= float(minimum_class_probability))

    feature = torch.zeros_like(labels, dtype=torch.bool)
    for class_index in FEATURE_CLASSES:
        feature |= labels == class_index
    radius = max(0, int(feature_protection_px))
    if radius:
        feature = (
            torch_functional.max_pool2d(
                feature.float().unsqueeze(1),
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            )[:, 0]
            > 0.0
        )
    skin &= ~feature
    maximum_probability = probabilities.max(dim=1).values
    class_counts = torch.bincount(labels.flatten(), minlength=19)
    return skin.float(), feature.float(), {
        "selected_pixel_count": int(skin.sum()),
        "protected_pixel_count": int(feature.sum()),
        "mean_selected_probability": (
            round(float(selected_probability[skin].mean()), 8) if bool(skin.any()) else 0.0
        ),
        "mean_argmax_probability": round(float(maximum_probability.mean()), 8),
        "class_pixel_counts": {
            PARSENET_CLASS_NAMES[index]: int(class_counts[index])
            for index in range(len(PARSENET_CLASS_NAMES))
            if int(class_counts[index]) > 0
        },
    }


def _preview_indices(frame_count: int, preview_count: int) -> list[int]:
    count = min(frame_count, max(1, int(preview_count)))
    if count == 1:
        return [0]
    return sorted(
        {round(index * (frame_count - 1) / (count - 1)) for index in range(count)}
    )


def _source_preview(frames: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return frames[indices, ..., :3].detach().to(device="cpu", dtype=torch.float32).clone()


def _build_preview(
    frames: torch.Tensor,
    skin_mask: torch.Tensor,
    feature_masks: dict[int, torch.Tensor],
    indices: list[int],
) -> torch.Tensor:
    preview = _source_preview(frames, indices)
    skin_colour = preview.new_tensor([0.10, 0.95, 0.25])
    feature_colour = preview.new_tensor([1.00, 0.15, 0.15])
    for preview_index, frame_index in enumerate(indices):
        skin = skin_mask[frame_index].clamp(0.0, 1.0).unsqueeze(-1)
        feature = feature_masks.get(frame_index)
        if feature is None:
            feature = torch.zeros_like(skin_mask[frame_index])
        feature = feature.clamp(0.0, 1.0).unsqueeze(-1)
        preview[preview_index] = preview[preview_index] * (1.0 - skin * 0.42) + (
            skin_colour * skin * 0.42
        )
        preview[preview_index] = preview[preview_index] * (1.0 - feature * 0.36) + (
            feature_colour * feature * 0.36
        )
    return preview.clamp(0.0, 1.0)


def _report_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def run_semantic_skin_mask(
    frames: torch.Tensor,
    face_plan: dict | None,
    parser_model: str = PARSENET_MODEL_NAME,
    include_neck: bool = False,
    crop_expansion: float = 1.45,
    minimum_face_weight: float = 0.35,
    minimum_class_probability: float = 0.55,
    feature_protection_px: int = 3,
    minimum_skin_area: float = 0.0005,
    maximum_skin_area: float = 0.25,
    preview_count: int = 6,
):
    frame_count, height, width, _ = _validate_frames(frames)
    if parser_model != PARSENET_MODEL_NAME:
        raise ValueError(f"Unsupported parser_model: {parser_model}")
    if not 1.0 <= float(crop_expansion) <= 3.0:
        raise ValueError("crop_expansion must stay within 1.0..3.0")
    if not 0.0 <= float(minimum_face_weight) <= 1.0:
        raise ValueError("minimum_face_weight must stay within 0..1")
    if not 0.0 <= float(minimum_class_probability) <= 1.0:
        raise ValueError("minimum_class_probability must stay within 0..1")
    if not 0.0 <= float(minimum_skin_area) < float(maximum_skin_area) <= 1.0:
        raise ValueError("skin area limits must satisfy 0 <= minimum < maximum <= 1")

    started = time.perf_counter()
    memory_before = _memory_snapshot()
    preview_indices = _preview_indices(frame_count, int(preview_count))
    skin_mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
    feature_previews: dict[int, torch.Tensor] = {}
    model = None
    model_path = None
    model_hash = None
    model_loaded = False
    model_unloaded = False
    plan = None
    frame_reports: list[dict] = []
    status = "ABSTAIN_NOT_EXECUTED"
    detail = ""
    progress = _progress_bar(frame_count)

    try:
        if face_plan is None:
            raise _ParserUnavailable(
                "ABSTAIN_FACE_PLAN_MISSING_OR_INVALID",
                "A source-bound H3 face_refine_plan is required.",
            )
        plan = _validate_bound_face_plan(frames, face_plan)
        model, model_path, model_hash = _load_pinned_parsenet()
        model_loaded = True
        for frame_index, record in enumerate(plan["frames"]):
            _interrupt_and_progress(progress, frame_index, frame_count)
            frame_status = "ABSTAIN"
            reasons: list[str] = []
            state = str(record.get("state", "lost"))
            weight = float(record.get("paste_weight", 0.0))
            crop_box = _square_crop_box(
                record.get("source_face_box_xyxy"),
                width,
                height,
                float(crop_expansion),
            )
            if state == "lost":
                reasons.append("face_track_lost")
            if weight < float(minimum_face_weight):
                reasons.append("face_weight_below_minimum")
            if crop_box is None:
                reasons.append("face_crop_invalid")
            local_feature_full = None
            area_fraction = 0.0
            semantic_stats = None
            if not reasons and crop_box is not None:
                x1, y1, x2, y2 = crop_box
                crop = (
                    frames[frame_index : frame_index + 1, y1:y2, x1:x2, :3]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                logits = _parser_logits(model, crop)
                local_skin, local_feature, semantic_stats = _semantic_local_masks(
                    logits,
                    include_neck=bool(include_neck),
                    minimum_class_probability=float(minimum_class_probability),
                    feature_protection_px=int(feature_protection_px),
                )
                target_size = (y2 - y1, x2 - x1)
                local_skin = torch_functional.interpolate(
                    local_skin.unsqueeze(1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                local_feature = torch_functional.interpolate(
                    local_feature.unsqueeze(1),
                    size=target_size,
                    mode="nearest",
                )[0, 0]
                local_skin = local_skin.mul(max(0.0, min(1.0, weight))).clamp(0.0, 1.0)
                area_fraction = float((local_skin > 0.05).sum()) / float(height * width)
                if area_fraction < float(minimum_skin_area):
                    reasons.append("semantic_skin_area_below_minimum")
                elif area_fraction > float(maximum_skin_area):
                    reasons.append("semantic_skin_area_above_maximum")
                else:
                    skin_mask[frame_index, y1:y2, x1:x2] = local_skin
                    frame_status = "READY"
                if frame_index in preview_indices:
                    local_feature_full = torch.zeros((height, width), dtype=torch.float32)
                    local_feature_full[y1:y2, x1:x2] = local_feature
            if local_feature_full is not None:
                feature_previews[frame_index] = local_feature_full
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "shot_id": int(record.get("shot_id", -1)),
                    "face_state": state,
                    "face_weight": round(weight, 8),
                    "status": frame_status,
                    "reasons": reasons,
                    "crop_box_xyxy": list(crop_box) if crop_box is not None else None,
                    "skin_area_fraction": round(area_fraction, 8),
                    "semantic_stats": semantic_stats,
                }
            )
        accepted = sum(item["status"] == "READY" for item in frame_reports)
        status = "READY" if accepted else "ABSTAIN_NO_RELIABLE_SEMANTIC_SKIN"
        _interrupt_and_progress(progress, frame_count, frame_count)
    except _ParserUnavailable as error:
        status = error.status
        detail = error.detail
    except Exception as error:
        status = "ABSTAIN_PARSENET_INFERENCE_FAILED"
        detail = f"{type(error).__name__}: {error}"
        skin_mask.zero_()
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

    if status != "READY":
        skin_mask.zero_()
        feature_previews.clear()
    preview = _build_preview(frames, skin_mask, feature_previews, preview_indices)
    if status == "READY":
        accepted_indices = [
            item["frame_index"] for item in frame_reports if item["status"] == "READY"
        ]
        rejected_indices = [
            item["frame_index"] for item in frame_reports if item["status"] != "READY"
        ]
    else:
        accepted_indices = []
        rejected_indices = list(range(frame_count))
    state = {
        "schema": SKIN_FINISH_SEMANTIC_MASK_SCHEMA,
        "status": status,
        "source_proxy_sha256": (
            str(plan.get("source", {}).get("proxy_sha256", "")) if plan else ""
        ),
        "face_plan_sha256": str(plan.get("plan_sha256", "")) if plan else "",
        "model_sha256": model_hash or "",
        "accepted_frame_indices": accepted_indices,
        "parameters": {
            "include_neck": bool(include_neck),
            "crop_expansion": float(crop_expansion),
            "minimum_face_weight": float(minimum_face_weight),
            "minimum_class_probability": float(minimum_class_probability),
            "feature_protection_px": int(feature_protection_px),
            "minimum_skin_area": float(minimum_skin_area),
            "maximum_skin_area": float(maximum_skin_area),
        },
    }
    state["sha256"] = hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
    report = {
        "schema": SKIN_FINISH_SEMANTIC_MASK_SCHEMA,
        "status": status,
        "detail": detail,
        "source": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "face_plan_sha256": state["face_plan_sha256"],
            "source_proxy_sha256": state["source_proxy_sha256"],
        },
        "model": {
            "name": PARSENET_MODEL_NAME,
            "path": str(model_path) if model_path else str(PARSENET_MODEL_RELATIVE),
            "expected_size": PARSENET_MODEL_SIZE,
            "expected_sha256": PARSENET_MODEL_SHA256,
            "actual_sha256": model_hash,
            "source": PARSENET_MODEL_SOURCE,
            "code_source": PARSENET_CODE_SOURCE,
            "mapping_source": PARSENET_MAPPING_SOURCE,
            "source_code_license": "MIT",
            "checkpoint_license": "not_explicitly_stated_by_the_release",
            "checkpoint_redistributed_by_t8": False,
            "runtime_download": False,
            "safe_weights_only_load_required": True,
            "loaded": model_loaded,
            "unloaded_after_execute": model_unloaded,
            "persistent_cache": False,
            "device": "cpu",
        },
        "class_mapping": {
            str(index): name for index, name in enumerate(PARSENET_CLASS_NAMES)
        },
        "selection": {
            "skin_classes": ["skin"] + (["neck"] if include_neck else []),
            "protected_classes": [
                PARSENET_CLASS_NAMES[index] for index in sorted(FEATURE_CLASSES)
            ],
            "ears_are_not_selected_by_default": True,
            "accepted_frame_count": len(accepted_indices),
            "accepted_frame_indices": accepted_indices,
            "rejected_frame_count": len(rejected_indices),
            "rejected_frame_indices": rejected_indices,
        },
        "parameters": state["parameters"],
        "frame_reports": frame_reports,
        "preview_frame_indices": preview_indices,
        "mechanical_gates": {
            "automatic_accept": False,
            "source_pixels_modified": False,
            "network_access_performed": False,
            "cuda_used": False,
            "output_finite": bool(torch.isfinite(skin_mask).all()),
            "output_range_valid": bool((skin_mask >= 0).all() and (skin_mask <= 1).all()),
            "model_released_if_loaded": (not model_loaded) or model_unloaded,
        },
        "alignment_boundary": (
            "The existing face_refine_plan stores source face boxes but not five-point landmarks. "
            "ParseNet therefore runs on an expanded upright square crop, not an affine-aligned "
            "face. Profile, rotation, occlusion and tiny faces may ABSTAIN or require review."
        ),
        "product_boundary": (
            "This node only proposes a semantic skin mask. It does not prove identity, skin "
            "quality, beauty, natural pores, deblurring or multi-person identity assignment."
        ),
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "state_sha256": state["sha256"],
        "human_review_required": True,
    }
    return skin_mask, preview, _report_json(report)
