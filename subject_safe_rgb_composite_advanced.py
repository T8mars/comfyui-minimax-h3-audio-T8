from __future__ import annotations

import json
from typing import Any

import torch


REPORT_SCHEMA = "h3_t8_subject_safe_rgb_composite/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _validate_frames(frames: torch.Tensor, *, name: str) -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if frames.ndim != 4:
        raise ValueError(f"{name} must use ComfyUI IMAGE layout [N,H,W,C]")
    count, height, width, channels = (int(value) for value in frames.shape)
    if count < 1 or height < 1 or width < 1 or channels < 3:
        raise ValueError(f"{name} must contain at least one RGB frame")
    if not frames.is_floating_point():
        raise TypeError(f"{name} must be a floating-point IMAGE tensor")
    return count, height, width, channels


def _normalise_mask(
    mask: torch.Tensor | None,
    *,
    name: str,
    frame_count: int,
    height: int,
    width: int,
    frame_policy: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, bool]:
    if mask is None:
        return None, False
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"{name} must use MASK layout [N,H,W]")
    if tuple(int(value) for value in mask.shape[1:]) != (height, width):
        raise ValueError(
            f"{name} geometry {tuple(mask.shape[1:])} does not match {(height, width)}"
        )
    broadcast = False
    if int(mask.shape[0]) == 1 and frame_count > 1:
        if frame_policy != "allow_single_broadcast_exp":
            raise ValueError(
                f"{name} has one frame but the videos have {frame_count}; "
                "provide a tracked per-frame mask or explicitly allow broadcast"
            )
        mask = mask.expand(frame_count, -1, -1)
        broadcast = True
    elif int(mask.shape[0]) != frame_count:
        raise ValueError(
            f"{name} frame count {int(mask.shape[0])} does not match {frame_count}"
        )
    if not mask.is_floating_point():
        mask = mask.float()
    return mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0), broadcast


def _mask_metrics(mask: torch.Tensor) -> dict[str, Any]:
    frame_count, height, width = (int(value) for value in mask.shape)
    flat = mask.reshape(frame_count, -1)
    support = flat > 0.0
    areas = support.float().mean(dim=1)
    weights = flat.sum(dim=1)

    yy = torch.arange(height, device=mask.device, dtype=torch.float32).view(1, height, 1)
    xx = torch.arange(width, device=mask.device, dtype=torch.float32).view(1, 1, width)
    safe_weights = weights.clamp_min(1.0)
    centroids_x = (mask * xx).sum(dim=(1, 2)) / safe_weights / max(1, width - 1)
    centroids_y = (mask * yy).sum(dim=(1, 2)) / safe_weights / max(1, height - 1)
    if frame_count > 1:
        jumps = torch.sqrt(
            torch.diff(centroids_x).square() + torch.diff(centroids_y).square()
        )
        max_jump = float(jumps.max().item())
    else:
        max_jump = 0.0
    return {
        "area_min": float(areas.min().item()),
        "area_max": float(areas.max().item()),
        "centroid_jump_max": max_jump,
        "empty_frames": int((weights <= 0.0).sum().item()),
    }


def compose_subject_safe_rgb(
    base_frames: torch.Tensor,
    refined_frames: torch.Tensor,
    subject_alpha: torch.Tensor,
    *,
    accept_candidate: bool = False,
    mask_mode: str = "input_alpha_exact",
    mask_frame_policy: str = "strict_exact",
    minimum_subject_area: float = 0.002,
    maximum_subject_area: float = 0.45,
    maximum_centroid_jump: float = 0.08,
    strictness: str = "fallback_on_contract_failure",
    chunk_frames: int = 4,
    protect_mask: torch.Tensor | None = None,
    audio: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, str]:
    """Composite a reviewed refinement only inside an explicit subject alpha.

    The implementation deliberately does not detect people, text, faces, camera motion,
    model files or quality.  Those decisions stay upstream and human-owned.  Pixels where
    the final alpha is exactly zero are copied from ``base_frames`` by construction.
    """

    if mask_mode not in {"input_alpha_exact", "threshold_binary"}:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if mask_frame_policy not in {"strict_exact", "allow_single_broadcast_exp"}:
        raise ValueError(f"unsupported mask_frame_policy: {mask_frame_policy}")
    if strictness not in {"fallback_on_contract_failure", "audit_only"}:
        raise ValueError(f"unsupported strictness: {strictness}")
    if not 0.0 <= float(minimum_subject_area) <= 1.0:
        raise ValueError("minimum_subject_area must be within 0..1")
    if not 0.0 <= float(maximum_subject_area) <= 1.0:
        raise ValueError("maximum_subject_area must be within 0..1")
    if float(minimum_subject_area) > float(maximum_subject_area):
        raise ValueError("minimum_subject_area must not exceed maximum_subject_area")
    if float(maximum_centroid_jump) < 0.0:
        raise ValueError("maximum_centroid_jump must be non-negative")
    if int(chunk_frames) < 1:
        raise ValueError("chunk_frames must be at least one")

    frame_count, height, width, channels = _validate_frames(base_frames, name="base_frames")
    refined_shape = _validate_frames(refined_frames, name="refined_frames")
    if refined_shape != (frame_count, height, width, channels):
        raise ValueError(
            "refined_frames must exactly match base_frames in frame count, geometry and channels"
        )

    alpha, alpha_broadcast = _normalise_mask(
        subject_alpha,
        name="subject_alpha",
        frame_count=frame_count,
        height=height,
        width=width,
        frame_policy=mask_frame_policy,
        device=base_frames.device,
    )
    assert alpha is not None
    protect, protect_broadcast = _normalise_mask(
        protect_mask,
        name="protect_mask",
        frame_count=frame_count,
        height=height,
        width=width,
        frame_policy=mask_frame_policy,
        device=base_frames.device,
    )
    if mask_mode == "threshold_binary":
        alpha = (alpha >= 0.5).to(dtype=torch.float32)
    if protect is not None:
        alpha = alpha * (1.0 - protect)

    metrics = _mask_metrics(alpha)
    failures: list[str] = []
    if metrics["empty_frames"]:
        failures.append(f"empty_alpha_frames={metrics['empty_frames']}")
    if metrics["area_min"] < float(minimum_subject_area):
        failures.append(
            f"alpha_area_min={metrics['area_min']:.8f}<minimum={float(minimum_subject_area):.8f}"
        )
    if metrics["area_max"] > float(maximum_subject_area):
        failures.append(
            f"alpha_area_max={metrics['area_max']:.8f}>maximum={float(maximum_subject_area):.8f}"
        )
    if metrics["centroid_jump_max"] > float(maximum_centroid_jump):
        failures.append(
            "centroid_jump_max="
            f"{metrics['centroid_jump_max']:.8f}>maximum={float(maximum_centroid_jump):.8f}"
        )

    refined = refined_frames.to(device=base_frames.device, dtype=base_frames.dtype)
    finite_failure = False
    for start in range(0, frame_count, int(chunk_frames)):
        stop = min(frame_count, start + int(chunk_frames))
        if not bool(torch.isfinite(refined[start:stop, ..., :3]).all().item()):
            finite_failure = True
            break
    if finite_failure:
        failures.append("refined_frames_contains_non_finite_rgb")

    fallback = bool(failures and strictness == "fallback_on_contract_failure")
    if fallback:
        candidate = base_frames
        used_alpha = torch.zeros_like(alpha)
    else:
        candidate = base_frames.clone()
        for start in range(0, frame_count, int(chunk_frames)):
            stop = min(frame_count, start + int(chunk_frames))
            base_rgb = base_frames[start:stop, ..., :3]
            refined_rgb = refined[start:stop, ..., :3]
            alpha_rgb = alpha[start:stop].to(dtype=base_rgb.dtype).unsqueeze(-1)
            blended = torch.lerp(base_rgb, refined_rgb, alpha_rgb)
            candidate[start:stop, ..., :3] = torch.where(
                alpha_rgb > 0.0,
                blended,
                base_rgb,
            )
        used_alpha = alpha

    selected = candidate if bool(accept_candidate) and not fallback else base_frames
    if fallback:
        status = "ABSTAIN_CONTRACT_FAILURE_SOURCE_RETURNED"
    elif bool(accept_candidate):
        status = "CANDIDATE_SELECTED_REQUIRES_HUMAN_REVIEW"
    else:
        status = "SOURCE_SELECTED_CANDIDATE_AVAILABLE"

    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "frames": frame_count,
        "width": width,
        "height": height,
        "channels": channels,
        "mask_mode": mask_mode,
        "mask_frame_policy": mask_frame_policy,
        "subject_alpha_single_frame_broadcast": alpha_broadcast,
        "protect_mask_connected": protect is not None,
        "protect_mask_single_frame_broadcast": protect_broadcast,
        "metrics": metrics,
        "limits": {
            "minimum_subject_area": float(minimum_subject_area),
            "maximum_subject_area": float(maximum_subject_area),
            "maximum_centroid_jump": float(maximum_centroid_jump),
        },
        "contract_failures": failures,
        "fallback_applied": fallback,
        "accept_candidate": bool(accept_candidate),
        "outside_zero_alpha_exact_by_construction": True,
        "audio_passthrough_same_object": True,
        "automatic_person_detection": False,
        "automatic_camera_gate": False,
        "automatic_quality_selection": False,
        "human_review_required": True,
        "boundary": (
            "Manual RGB composite only. It does not detect the target, protect text/faces, "
            "judge camera motion, or prove general quality. Provide a reviewed per-frame alpha."
        ),
    }
    return selected, candidate, base_frames, used_alpha, audio, _canonical_json(report)
