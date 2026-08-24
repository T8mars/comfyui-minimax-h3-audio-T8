from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional


SKIN_FINISH_STATE_SCHEMA = "h3_t8_skin_finish_state/v1"
SKIN_FINISH_REPORT_SCHEMA = "h3_t8_skin_finish_report/v1"
SKIN_FINISH_REVIEW_SCHEMA = "h3_t8_skin_finish_review/v1"
FACE_REFINE_PLAN_SCHEMA = "h3_t8_face_refine_plan/v1"

PRESET_CONFIG = {
    "subtle": {"smooth": 0.10, "tone_even": 0.35, "shine": 0.60},
    "oil_control": {"smooth": 0.12, "tone_even": 0.30, "shine": 1.25},
    "tone_even": {"smooth": 0.08, "tone_even": 0.85, "shine": 0.45},
    "soft_portrait": {"smooth": 0.42, "tone_even": 0.42, "shine": 0.65},
    "custom": {"smooth": 0.20, "tone_even": 0.50, "shine": 1.00},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _json_hash(value: Any) -> str:
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
        raise ValueError(f"{name} must stay within the ComfyUI IMAGE range 0..1")
    return frame_count, height, width, channels


def _normalize_mask(
    mask: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
    *,
    name: str = "mask",
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a MASK tensor")
    value = mask.detach()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"{name} must be [N,H,W], got {tuple(value.shape)}")
    if tuple(value.shape[1:]) != (height, width):
        value = torch_functional.interpolate(
            value.float().unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    if int(value.shape[0]) == 1 and frame_count > 1:
        value = value.expand(frame_count, -1, -1)
    if int(value.shape[0]) != frame_count:
        raise ValueError(
            f"{name} frame count must be 1 or {frame_count}, got {int(value.shape[0])}"
        )
    value = value.to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError(f"{name} values must stay within 0..1")
    return value


def _tensor_proxy_sha256(value: torch.Tensor, *, size: int = 32) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    if tensor.ndim == 4:
        proxy = torch_functional.interpolate(
            tensor[..., :3].movedim(-1, 1),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
    elif tensor.ndim == 3:
        proxy = torch_functional.interpolate(
            tensor.unsqueeze(1),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        proxy = tensor.reshape(1, 1, 1, -1)
    digest.update(proxy.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _audio_contract(audio: dict | None) -> dict:
    if audio is None:
        return {"provided": False, "passthrough": True}
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO value")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or not bool(torch.isfinite(waveform).all()):
        raise ValueError("audio waveform must be a finite tensor")
    cpu = waveform.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
    return {
        "provided": True,
        "passthrough": True,
        "sample_rate": int(audio["sample_rate"]),
        "shape": list(map(int, waveform.shape)),
        "dtype": str(waveform.dtype),
        "pcm_sha256": digest,
    }


def _memory_snapshot() -> dict:
    result: dict[str, Any] = {"pid": os.getpid()}
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                result.update(
                    {
                        "host_total_mib": round(status.ullTotalPhys / 2**20, 3),
                        "host_available_mib": round(status.ullAvailPhys / 2**20, 3),
                        "commit_limit_mib": round(status.ullTotalPageFile / 2**20, 3),
                        "commit_available_mib": round(status.ullAvailPageFile / 2**20, 3),
                    }
                )
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                result["process_rss_mib"] = round(counters.WorkingSetSize / 2**20, 3)
                result["process_commit_mib"] = round(counters.PagefileUsage / 2**20, 3)
        except Exception:
            result["host_measurement"] = "unavailable"
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            result.update(
                {
                    "cuda_free_mib": round(free / 2**20, 3),
                    "cuda_total_mib": round(total / 2**20, 3),
                    "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
                    "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
                }
            )
        except Exception:
            result["cuda_measurement"] = "unavailable"
    return result


def _face_plan_mask(
    frames: torch.Tensor,
    face_plan: dict | None,
    *,
    protect_features: bool,
) -> tuple[torch.Tensor | None, dict]:
    frame_count, height, width, _ = _validate_frames(frames)
    if not isinstance(face_plan, dict) or face_plan.get("schema") != FACE_REFINE_PLAN_SCHEMA:
        return None, {"status": "ABSTAIN_FACE_PLAN_MISSING_OR_INVALID"}
    try:
        from .face_refine_advanced import source_proxy_sha256

        expected = str(face_plan.get("source", {}).get("proxy_sha256", ""))
        if not expected or source_proxy_sha256(frames) != expected:
            return None, {"status": "ABSTAIN_FACE_PLAN_SOURCE_MISMATCH"}
    except Exception as error:
        return None, {
            "status": "ABSTAIN_FACE_PLAN_SOURCE_CHECK_FAILED",
            "detail": str(error),
        }
    records = face_plan.get("frames")
    if not isinstance(records, list) or len(records) != frame_count:
        return None, {"status": "ABSTAIN_FACE_PLAN_FRAME_COUNT_MISMATCH"}

    mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
    accepted = 0
    rejected = 0
    for index, record in enumerate(records):
        state = str(record.get("state", "lost"))
        weight = float(record.get("paste_weight", 0.0))
        box = record.get("source_face_box_xyxy")
        if state == "lost" or weight < 0.35 or not isinstance(box, list) or len(box) != 4:
            rejected += 1
            continue
        left, top, right, bottom = [float(item) for item in box]
        x1 = max(0, min(width - 1, int(math.floor(left))))
        y1 = max(0, min(height - 1, int(math.floor(top))))
        x2 = max(x1 + 1, min(width, int(math.ceil(right))))
        y2 = max(y1 + 1, min(height, int(math.ceil(bottom))))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 4 or box_height < 4:
            rejected += 1
            continue
        yy = (torch.arange(box_height, dtype=torch.float32) + 0.5) / box_height
        xx = (torch.arange(box_width, dtype=torch.float32) + 0.5) / box_width
        ellipse = (((xx[None, :] - 0.5) / 0.47) ** 2 + ((yy[:, None] - 0.52) / 0.50) ** 2) <= 1.0
        local = ellipse.float() * max(0.0, min(1.0, weight))
        if protect_features:
            for center_x, center_y, radius_x, radius_y in (
                (0.32, 0.38, 0.15, 0.09),
                (0.68, 0.38, 0.15, 0.09),
                (0.50, 0.56, 0.10, 0.08),
                (0.50, 0.74, 0.23, 0.11),
            ):
                protected = (
                    ((xx[None, :] - center_x) / radius_x) ** 2
                    + ((yy[:, None] - center_y) / radius_y) ** 2
                ) <= 1.0
                local[protected] = 0.0
        mask[index, y1:y2, x1:x2] = local
        accepted += 1
    return mask, {
        "status": "READY" if accepted else "ABSTAIN_FACE_PLAN_NO_RELIABLE_FRAMES",
        "accepted_frames": accepted,
        "rejected_frames": rejected,
        "feature_protection": bool(protect_features),
        "geometry": "face_box_inner_ellipse_with_approximate_feature_exclusion",
        "warning": (
            "Face-plan geometry is a conservative face-region proxy, not semantic skin parsing. "
            "Review the used mask before accepting the candidate."
        ),
    }


def _temporal_median(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius == 0 or int(mask.shape[0]) == 1:
        return mask
    output = torch.empty_like(mask)
    for index in range(int(mask.shape[0])):
        start = max(0, index - radius)
        end = min(int(mask.shape[0]), index + radius + 1)
        output[index] = mask[start:end].median(dim=0).values
    return output


def _prepare_mask(
    raw_mask: torch.Tensor,
    *,
    frame_count: int,
    height: int,
    width: int,
    minimum_area: float,
    maximum_area: float,
    feather_px: int,
    temporal_radius: int,
    chunk_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    value = _normalize_mask(raw_mask, frame_count, height, width)
    value = _temporal_median(value, int(temporal_radius))
    binary = value > 1.0e-5
    areas = binary.float().mean(dim=(1, 2))
    reliable = (areas >= float(minimum_area)) & (areas <= float(maximum_area))
    used = torch.zeros_like(value)
    rejected = torch.zeros_like(value)
    radius = max(0, int(feather_px))
    kernel = radius * 2 + 1
    chunk_size = max(1, int(chunk_frames))
    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        chunk = value[start:end]
        chunk_binary = binary[start:end]
        chunk_reliable = reliable[start:end, None, None]
        if radius:
            soft = torch_functional.avg_pool2d(
                chunk.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=radius,
            )[:, 0]
            # Never expand beyond the user's or plan's semantic region.
            soft = soft * chunk_binary
        else:
            soft = chunk
        used[start:end] = torch.where(chunk_reliable, soft, torch.zeros_like(soft))
        rejected[start:end] = torch.where(
            chunk_reliable, torch.zeros_like(chunk), chunk
        )
    accepted_indices = torch.nonzero(reliable, as_tuple=False).flatten().tolist()
    rejected_indices = torch.nonzero(~reliable, as_tuple=False).flatten().tolist()
    return used, rejected, {
        "minimum_area_fraction": float(minimum_area),
        "maximum_area_fraction": float(maximum_area),
        "area_fraction_min": round(float(areas.min()), 8),
        "area_fraction_mean": round(float(areas.mean()), 8),
        "area_fraction_max": round(float(areas.max()), 8),
        "accepted_frame_count": len(accepted_indices),
        "rejected_frame_count": len(rejected_indices),
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "feather_px_inside_only": radius,
        "temporal_mask_radius": int(temporal_radius),
    }


def _interrupt_and_progress(progress, completed: int, total: int) -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass
    if progress is not None:
        progress.update_absolute(completed, total)


def _progress_bar(total: int):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except Exception:
        return None


def _proxy_size(height: int, width: int, maximum_side: int) -> tuple[int, int]:
    maximum_side = max(64, int(maximum_side))
    scale = min(1.0, maximum_side / max(height, width))
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def _local_average(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    result = value
    for _ in range(2):
        result = torch_functional.avg_pool2d(
            result, kernel_size=kernel, stride=1, padding=radius
        )
    return result


def _process_chunk(
    chunk: torch.Tensor,
    mask: torch.Tensor,
    *,
    preset: str,
    amount: float,
    texture_keep: float,
    shine_control: float,
    tone_adjust: float,
    proxy_long_side: int,
) -> torch.Tensor:
    config = PRESET_CONFIG[preset]
    source = chunk.detach().to(device="cpu", dtype=torch.float32)
    rgb = source[..., :3]
    batch, height, width, _ = rgb.shape
    proxy_h, proxy_w = _proxy_size(height, width, proxy_long_side)
    proxy = torch_functional.interpolate(
        rgb.movedim(-1, 1),
        size=(proxy_h, proxy_w),
        mode="bilinear",
        align_corners=False,
    )
    radius = max(1, int(round(min(proxy_h, proxy_w) * 0.008)))
    base = _local_average(proxy, radius)
    luma_weights = torch.tensor(
        [0.2126, 0.7152, 0.0722], dtype=proxy.dtype
    ).view(1, 3, 1, 1)
    luma = (proxy * luma_weights).sum(dim=1, keepdim=True)
    base_luma = (base * luma_weights).sum(dim=1, keepdim=True)
    residual = luma - base_luma
    edge_weight = torch.exp(-residual.abs() / 0.075).clamp(0.08, 1.0)
    midtone = (1.0 - ((luma - 0.5).abs() * 1.8)).clamp(0.10, 1.0)
    dark_spot = (-residual).clamp_min(0.0)
    highlight = residual.clamp_min(0.0)
    tone_delta = dark_spot * float(config["tone_even"]) * 0.55
    shine_delta = highlight * float(config["shine"]) * float(shine_control) * 0.70
    exposure_delta = float(tone_adjust) * 0.08 * midtone
    colour_even = (base - proxy) * float(config["tone_even"]) * 0.16 * edge_weight
    smoothing = (
        (base - proxy)
        * float(config["smooth"])
        * max(0.0, 1.0 - float(texture_keep))
        * edge_weight
    )
    corrected_proxy = proxy + colour_even + smoothing
    corrected_proxy = corrected_proxy + (tone_delta - shine_delta + exposure_delta) * midtone
    correction = corrected_proxy - proxy
    correction = torch_functional.interpolate(
        correction,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    corrected = (rgb + correction * float(amount)).clamp(0.0, 1.0)
    alpha = mask.to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
    blended_rgb = rgb + (corrected - rgb) * alpha
    binary = alpha > 0
    blended_rgb = torch.where(binary, blended_rgb, rgb)
    if int(source.shape[-1]) == 3:
        return blended_rgb.to(dtype=chunk.dtype)
    output = source.clone()
    output[..., :3] = blended_rgb
    # Alpha and any future auxiliary channels are byte-for-byte preserved.
    return output.to(dtype=chunk.dtype)


def run_skin_finish(
    frames: torch.Tensor,
    *,
    preset: str = "subtle",
    amount: float = 0.35,
    texture_keep: float = 0.90,
    shine_control: float = 0.35,
    tone_adjust: float = 0.0,
    execution_mode: str = "candidate_only",
    chunk_frames: int = 4,
    mask: torch.Tensor | None = None,
    audio: dict | None = None,
    mask_source: str = "external_exact",
    face_plan: dict | None = None,
    protect_features: bool = True,
    minimum_mask_area: float = 0.002,
    maximum_mask_area: float = 0.45,
    mask_feather_px: int = 3,
    temporal_mask_radius: int = 0,
    proxy_long_side: int = 640,
    accept_candidate: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
    dict | None,
    str,
]:
    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(frames)
    if preset not in PRESET_CONFIG:
        raise ValueError(f"Unsupported preset: {preset}")
    if execution_mode not in {"candidate_only", "review_only", "bypass"}:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    if mask_source not in {"external_exact", "face_refine_plan"}:
        raise ValueError(f"Unsupported P0 mask_source: {mask_source}")
    if not 0.0 <= float(amount) <= 1.0:
        raise ValueError("amount must stay within 0..1")
    if not 0.0 <= float(texture_keep) <= 1.0:
        raise ValueError("texture_keep must stay within 0..1")
    if not 0.0 <= float(shine_control) <= 1.0:
        raise ValueError("shine_control must stay within 0..1")
    if not -1.0 <= float(tone_adjust) <= 1.0:
        raise ValueError("tone_adjust must stay within -1..1")
    if not 0.0 <= float(minimum_mask_area) < float(maximum_mask_area) <= 1.0:
        raise ValueError("mask area limits must satisfy 0 <= minimum < maximum <= 1")

    memory_before = _memory_snapshot()
    audio_report = _audio_contract(audio)
    source_report: dict[str, Any]
    raw_mask = None
    if mask_source == "external_exact":
        raw_mask = mask
        source_report = {
            "status": "READY" if mask is not None else "ABSTAIN_EXTERNAL_MASK_MISSING",
            "source": "external_exact",
            "semantic_claim": "user_or_upstream_supplied_skin_region",
        }
    else:
        raw_mask, source_report = _face_plan_mask(
            frames, face_plan, protect_features=bool(protect_features)
        )
        source_report["source"] = "face_refine_plan"

    zero_mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
    used_mask = zero_mask
    rejected_mask = zero_mask
    mask_report = {
        "accepted_frame_count": 0,
        "rejected_frame_count": frame_count,
    }
    status = "CANDIDATE_READY"
    findings: list[str] = []
    if execution_mode == "bypass":
        status = "BYPASS_EXACT"
        findings.append("execution_mode_bypass")
    elif raw_mask is None or not str(source_report.get("status", "")).startswith("READY"):
        status = str(source_report.get("status", "ABSTAIN_NO_RELIABLE_SKIN_MASK"))
        findings.append("no_reliable_skin_mask")
    else:
        used_mask, rejected_mask, mask_report = _prepare_mask(
            raw_mask,
            frame_count=frame_count,
            height=height,
            width=width,
            minimum_area=float(minimum_mask_area),
            maximum_area=float(maximum_mask_area),
            feather_px=int(mask_feather_px),
            temporal_radius=int(temporal_mask_radius),
            chunk_frames=int(chunk_frames),
        )
        if int(mask_report["accepted_frame_count"]) == 0:
            status = "ABSTAIN_NO_RELIABLE_SKIN_MASK"
            findings.append("all_frames_rejected_by_mask_area_gate")
    if execution_mode == "review_only":
        findings.append("review_only_forces_source_selection")

    progress = _progress_bar(frame_count)
    if status == "CANDIDATE_READY":
        # One preallocated CPU candidate avoids retaining every processed chunk and then
        # allocating a second full batch during torch.cat. Comfy IMAGE batches normally
        # arrive on CPU; a non-CPU source is copied only chunk by chunk.
        candidate = torch.empty(
            tuple(frames.shape), dtype=frames.dtype, device="cpu"
        )
        chunk_size = max(1, int(chunk_frames))
        try:
            for start in range(0, frame_count, chunk_size):
                end = min(frame_count, start + chunk_size)
                _interrupt_and_progress(progress, start, frame_count)
                candidate[start:end] = _process_chunk(
                    frames[start:end],
                    used_mask[start:end],
                    preset=preset,
                    amount=float(amount),
                    texture_keep=float(texture_keep),
                    shine_control=float(shine_control),
                    tone_adjust=float(tone_adjust),
                    proxy_long_side=int(proxy_long_side),
                )
                _interrupt_and_progress(progress, end, frame_count)
        finally:
            gc.collect()
    else:
        candidate = frames
        _interrupt_and_progress(progress, frame_count, frame_count)

    if not bool(torch.isfinite(candidate).all()):
        raise ValueError("Skin Finish candidate contains NaN or Inf")
    # Difference remains a complete IMAGE batch for downstream inspection, but fp16 is
    # sufficient for a visual audit and halves its retained RAM. Mechanical metrics and
    # exactness gates are calculated in bounded float32 chunks.
    difference = torch.empty(
        (frame_count, height, width, 3), dtype=torch.float16, device="cpu"
    )
    difference_sum = 0.0
    difference_count = 0
    difference_max = 0.0
    outside_exact = True
    alpha_preserved = True
    audit_chunk = max(1, int(chunk_frames))
    for start in range(0, frame_count, audit_chunk):
        end = min(frame_count, start + audit_chunk)
        source_chunk = frames[start:end].detach().to(device="cpu")
        candidate_chunk = candidate[start:end].detach().to(device="cpu")
        delta = (candidate_chunk[..., :3].float() - source_chunk[..., :3].float()).abs()
        difference[start:end] = delta.to(dtype=torch.float16)
        difference_sum += float(delta.double().sum())
        difference_count += int(delta.numel())
        difference_max = max(difference_max, float(delta.max()))
        outside_chunk = used_mask[start:end] <= 0
        if not torch.equal(
            candidate_chunk[..., :3][outside_chunk],
            source_chunk[..., :3][outside_chunk],
        ):
            outside_exact = False
        if channels > 3 and not torch.equal(
            candidate_chunk[..., 3:], source_chunk[..., 3:]
        ):
            alpha_preserved = False
    if not outside_exact:
        raise RuntimeError("Skin Finish changed pixels outside the used mask")
    if not alpha_preserved:
        raise RuntimeError("Skin Finish changed alpha or auxiliary channels")

    accepted = (
        bool(accept_candidate)
        and status == "CANDIDATE_READY"
        and execution_mode != "review_only"
    )
    selected = candidate if accepted else frames
    state = {
        "schema": SKIN_FINISH_STATE_SCHEMA,
        "status": status,
        "source_proxy_sha256": _tensor_proxy_sha256(frames),
        "mask_proxy_sha256": _tensor_proxy_sha256(used_mask),
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "preset": preset,
        "accepted_candidate": accepted,
        "automatic_accept": False,
    }
    state["sha256"] = _json_hash(state)
    memory_after = _memory_snapshot()
    report = {
        "schema": SKIN_FINISH_REPORT_SCHEMA,
        "status": status,
        "findings": findings,
        "product_boundary": (
            "Non-generative SDR skin finishing only. It does not restore identity, facial "
            "structure, missing pores, blur, occlusion or lip sync."
        ),
        "source": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": str(frames.dtype),
            "device": str(frames.device),
            "proxy_sha256": state["source_proxy_sha256"],
        },
        "mask_source": source_report,
        "mask_gate": mask_report,
        "parameters": {
            "preset": preset,
            "amount": float(amount),
            "texture_keep": float(texture_keep),
            "shine_control": float(shine_control),
            "tone_adjust": float(tone_adjust),
            "execution_mode": execution_mode,
            "chunk_frames": max(1, int(chunk_frames)),
            "proxy_long_side": int(proxy_long_side),
            "protect_features": bool(protect_features),
        },
        "mechanical_gates": {
            "finite": True,
            "shape_preserved": tuple(candidate.shape) == tuple(frames.shape),
            "outside_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": alpha_preserved,
            "source_overwrite_performed": False,
            "candidate_selected": accepted,
            "audio_object_passthrough": True,
        },
        "difference": {
            "mean_abs_rgb": round(difference_sum / max(1, difference_count), 10),
            "max_abs_rgb": round(difference_max, 10),
            "retained_dtype": str(difference.dtype),
        },
        "audio": audio_report,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "state_sha256": state["sha256"],
        "review_required": status == "CANDIDATE_READY",
    }
    return (
        candidate,
        frames,
        selected,
        used_mask,
        rejected_mask,
        difference,
        state,
        audio,
        canonical_json(report),
    )


def build_skin_finish_review(
    source_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    used_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    skin_finish_state: dict,
    gate_report_json: str,
    *,
    frame_index: int,
    comparison_position: float,
    accept_candidate: bool,
    audio_source: dict | None = None,
    audio_passthrough: dict | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict | None,
    str,
]:
    frame_count, height, width, _ = _validate_frames(source_frames, name="source_frames")
    if tuple(candidate_frames.shape) != tuple(source_frames.shape):
        raise ValueError("candidate_frames must match source_frames exactly")
    _validate_frames(candidate_frames, name="candidate_frames")
    used = _normalize_mask(used_mask, frame_count, height, width, name="used_mask")
    rejected = _normalize_mask(
        rejected_mask, frame_count, height, width, name="rejected_mask"
    )
    if not isinstance(skin_finish_state, dict) or skin_finish_state.get("schema") != SKIN_FINISH_STATE_SCHEMA:
        raise ValueError(f"skin_finish_state must use {SKIN_FINISH_STATE_SCHEMA}")
    expected_hash = skin_finish_state.get("sha256")
    payload = dict(skin_finish_state)
    payload.pop("sha256", None)
    if expected_hash != _json_hash(payload):
        raise ValueError("skin_finish_state hash mismatch")
    try:
        gate = json.loads(gate_report_json)
    except json.JSONDecodeError as error:
        raise ValueError("gate_report_json must contain valid JSON") from error
    if gate.get("schema") != SKIN_FINISH_REPORT_SCHEMA:
        raise ValueError(f"gate_report_json must use {SKIN_FINISH_REPORT_SCHEMA}")
    index = max(0, min(frame_count - 1, int(frame_index)))
    position = max(0.0, min(1.0, float(comparison_position)))
    split = max(0, min(width, int(round(width * position))))

    source_cpu = source_frames.detach().to(device="cpu")
    candidate_cpu = candidate_frames.detach().to(device="cpu")
    comparison = candidate_cpu[index].clone()
    comparison[:, :split] = source_cpu[index, :, :split]
    mask_visual = torch.zeros((height, width, 3), dtype=torch.float32)
    mask_visual[..., 1] = used[index]
    mask_visual[..., 0] = rejected[index]
    difference = (
        (candidate_cpu[index, ..., :3].float() - source_cpu[index, ..., :3].float())
        .abs()
        .mul(4.0)
        .clamp(0.0, 1.0)
    )

    active = used[index] > 0
    if bool(active.any()):
        coords = active.nonzero(as_tuple=False)
        y1, x1 = coords.min(dim=0).values.tolist()
        y2, x2 = (coords.max(dim=0).values + 1).tolist()
        pad = max(8, int(round(max(y2 - y1, x2 - x1) * 0.15)))
        y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
        y2, x2 = min(height, y2 + pad), min(width, x2 + pad)
    else:
        y1, x1, y2, x2 = 0, 0, height, width
    source_crop = source_cpu[index : index + 1, y1:y2, x1:x2]
    candidate_crop = candidate_cpu[index : index + 1, y1:y2, x1:x2]

    loop = []
    loop_indices = list(range(max(0, index - 2), min(frame_count, index + 3)))
    for loop_index in loop_indices:
        frame = candidate_cpu[loop_index].clone()
        frame[:, :split] = source_cpu[loop_index, :, :split]
        loop.append(frame)
    loop_preview = torch.stack(loop)

    audio_equal = True
    audio_status = "not_provided"
    if audio_source is not None or audio_passthrough is not None:
        if audio_source is None or audio_passthrough is None:
            audio_equal = False
            audio_status = "one_side_missing"
        else:
            first = _audio_contract(audio_source)
            second = _audio_contract(audio_passthrough)
            audio_equal = first == second
            audio_status = "pcm_exact" if audio_equal else "mismatch"
    candidate_allowed = gate.get("status") == "CANDIDATE_READY"
    accepted = bool(accept_candidate) and candidate_allowed and audio_equal
    selected = candidate_frames if accepted else source_frames
    review = {
        "schema": SKIN_FINISH_REVIEW_SCHEMA,
        "status": "ACCEPTED_CANDIDATE" if accepted else (
            "ABSTAIN_AUDIO_MISMATCH" if not audio_equal else "REVIEW_REQUIRED"
        ),
        "frame_index": index,
        "loop_frame_indices": loop_indices,
        "comparison_position": position,
        "comparison_left": "source",
        "comparison_right": "candidate",
        "crop_xyxy": [x1, y1, x2, y2],
        "audio_status": audio_status,
        "candidate_allowed_by_gate": candidate_allowed,
        "accept_candidate_requested": bool(accept_candidate),
        "accepted_candidate": accepted,
        "automatic_accept": False,
        "state_sha256": expected_hash,
    }
    return (
        selected,
        comparison.unsqueeze(0),
        source_crop,
        candidate_crop,
        mask_visual.unsqueeze(0),
        difference.unsqueeze(0),
        loop_preview,
        audio_passthrough,
        canonical_json(review),
    )
