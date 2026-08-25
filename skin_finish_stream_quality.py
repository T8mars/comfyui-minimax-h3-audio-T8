from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional

from .skin_finish import (
    _interrupt_and_progress,
    _memory_snapshot,
    _prepare_mask,
    _process_chunk,
    canonical_json,
)
from .skin_finish_frequency import separate_skin_finish_frequencies
from .skin_finish_p1 import stream_skin_finish_video
from .skin_finish_p2 import guard_skin_finish_candidate
from .skin_finish_parser import (
    PARSENET_MODEL_NAME,
    _load_pinned_parsenet,
    _parser_logits,
    _semantic_local_masks,
    _square_crop_box,
)
from .skin_finish_safety_audit import audit_skin_finish_candidate


SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA = (
    "h3_t8_skin_finish_quality_stream_report/v1"
)
SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA = (
    "h3_t8_skin_finish_quality_stream_ram_preflight/v1"
)
QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB = 2048.0
QUALITY_STREAM_REVIEWED_INCREMENTAL_PEAK_MIB = 1163.129


def _quality_stream_ram_preflight(
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed = dict(_memory_snapshot() if snapshot is None else snapshot)
    available = observed.get("host_available_mib")
    measurement_source = "native_memory_snapshot"
    if available is None and snapshot is None and os.name != "nt":
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                available = pages * page_size / float(2**20)
                observed["host_available_mib"] = round(float(available), 3)
                measurement_source = "posix_sysconf"
        except (AttributeError, OSError, TypeError, ValueError):
            available = None

    measurement_available = False
    if available is not None:
        try:
            available = float(available)
            measurement_available = math.isfinite(available) and available >= 0.0
        except (TypeError, ValueError):
            available = None
    if not measurement_available:
        available = None

    allowed = (
        True
        if available is None
        else available >= QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB
    )
    if available is None:
        status = "PROCEED_BOUNDED_ROUTE_MEMORY_MEASUREMENT_UNAVAILABLE"
    elif allowed:
        status = "PASS_REVIEWED_AVAILABLE_RAM_FLOOR"
    else:
        status = "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN"
    return {
        "schema": SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA,
        "status": status,
        "allowed": allowed,
        "measurement_available": measurement_available,
        "measurement_source": measurement_source,
        "available_mib": round(available, 3) if available is not None else None,
        "minimum_available_mib": QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB,
        "reviewed_incremental_peak_mib": (
            QUALITY_STREAM_REVIEWED_INCREMENTAL_PEAK_MIB
        ),
        "reviewed_floor_headroom_mib": round(
            QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB
            - QUALITY_STREAM_REVIEWED_INCREMENTAL_PEAK_MIB,
            3,
        ),
        "snapshot": observed,
        "boundary": (
            "The 2048 MiB floor is derived from one reviewed 32-second CPU run whose "
            "process working-set increase was about 1163.129 MiB. It is a pre-load "
            "ABSTAIN guard, not a universal RAM-safety guarantee. Unsupported host-memory "
            "measurement proceeds only because the route remains chunk bounded and reports "
            "the missing measurement explicitly."
        ),
    }


class _QualityChunkProcessor:
    """Keep ParseNet and only one bounded frame chunk alive during file streaming."""

    def __init__(
        self,
        *,
        preset: str,
        amount: float,
        texture_keep: float,
        shine_control: float,
        crop_expansion: float,
        minimum_class_probability: float,
        feature_protection_px: int,
        mask_feather_px: int,
        proxy_long_side: int,
        low_frequency_strength: float,
        source_detail_gain: float,
        separation_radius_percent: float,
        maximum_radius_px: int,
        shadow_protection: float,
        highlight_protection: float,
        minimum_texture_ratio: float,
        maximum_temporal_effect_jump: float,
    ) -> None:
        self.preset = str(preset)
        self.amount = float(amount)
        self.texture_keep = float(texture_keep)
        self.shine_control = float(shine_control)
        self.crop_expansion = float(crop_expansion)
        self.minimum_class_probability = float(minimum_class_probability)
        self.feature_protection_px = int(feature_protection_px)
        self.mask_feather_px = int(mask_feather_px)
        self.proxy_long_side = int(proxy_long_side)
        self.low_frequency_strength = float(low_frequency_strength)
        self.source_detail_gain = float(source_detail_gain)
        self.separation_radius_percent = float(separation_radius_percent)
        self.maximum_radius_px = int(maximum_radius_px)
        self.shadow_protection = float(shadow_protection)
        self.highlight_protection = float(highlight_protection)
        self.minimum_texture_ratio = float(minimum_texture_ratio)
        self.maximum_temporal_effect_jump = float(maximum_temporal_effect_jump)
        self.model = None
        self.model_path: Path | None = None
        self.model_hash = ""
        self.model_loaded = False
        self.model_released = False
        self.previous_source: torch.Tensor | None = None
        self.previous_candidate: torch.Tensor | None = None
        self.previous_mask: torch.Tensor | None = None
        self.stats: dict[str, Any] = {
            "chunk_count": 0,
            "peak_chunk_frames": 0,
            "source_frame_count": 0,
            "semantic_ready_frame_count": 0,
            "semantic_face_instance_count": 0,
            "semantic_face_rejection_count": 0,
            "source_only_frame_count": 0,
            "frequency_rejected_frame_count": 0,
            "texture_guard_rejected_frame_count": 0,
            "safety_audit_failed_chunk_count": 0,
            "safety_audit_failed_frame_count": 0,
            "maximum_temporal_effect_jump": 0.0,
        }

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self.model, self.model_path, self.model_hash = _load_pinned_parsenet()
        self.model_loaded = True

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.to(device="cpu")
            except Exception:
                pass
            del self.model
            self.model = None
            self.model_released = True
        self.previous_source = None
        self.previous_candidate = None
        self.previous_mask = None
        gc.collect()

    def _semantic_mask(
        self,
        source_chunk: torch.Tensor,
        face_records: list[list[dict[str, Any]]],
    ) -> torch.Tensor:
        self._ensure_model()
        frame_count, height, width, _ = map(int, source_chunk.shape)
        raw_mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
        for local_index, faces in enumerate(face_records):
            frame_ready = False
            for face in faces:
                _interrupt_and_progress(None, 0, 1)
                crop_box = _square_crop_box(
                    face.get("box"),
                    width,
                    height,
                    self.crop_expansion,
                )
                if crop_box is None:
                    self.stats["semantic_face_rejection_count"] += 1
                    continue
                x1, y1, x2, y2 = crop_box
                crop = (
                    source_chunk[local_index : local_index + 1, y1:y2, x1:x2, :3]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                logits = _parser_logits(self.model, crop)
                local_skin, _, _ = _semantic_local_masks(
                    logits,
                    include_neck=False,
                    minimum_class_probability=self.minimum_class_probability,
                    feature_protection_px=self.feature_protection_px,
                )
                local_skin = torch_functional.interpolate(
                    local_skin.unsqueeze(1),
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                local_skin.mul_(max(0.0, min(1.0, float(face.get("weight", 0.0)))))
                area_fraction = float((local_skin > 0.05).sum()) / float(height * width)
                if not 0.00005 <= area_fraction <= 0.35:
                    self.stats["semantic_face_rejection_count"] += 1
                    continue
                raw_mask[local_index, y1:y2, x1:x2] = torch.maximum(
                    raw_mask[local_index, y1:y2, x1:x2],
                    local_skin,
                )
                self.stats["semantic_face_instance_count"] += 1
                frame_ready = True
            if frame_ready:
                self.stats["semantic_ready_frame_count"] += 1
            else:
                self.stats["source_only_frame_count"] += 1
        return raw_mask

    def __call__(
        self,
        source_chunk: torch.Tensor,
        face_records: list[list[dict[str, Any]]],
        absolute_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count, height, width, _ = map(int, source_chunk.shape)
        if len(face_records) != frame_count:
            raise RuntimeError("quality stream face metadata does not match its frame chunk")
        self.stats["chunk_count"] += 1
        self.stats["peak_chunk_frames"] = max(
            int(self.stats["peak_chunk_frames"]), frame_count
        )
        self.stats["source_frame_count"] += frame_count

        raw_mask = self._semantic_mask(source_chunk, face_records)
        used_mask, _, mask_report = _prepare_mask(
            raw_mask,
            frame_count=frame_count,
            height=height,
            width=width,
            minimum_area=0.00005,
            maximum_area=0.35,
            feather_px=self.mask_feather_px,
            temporal_radius=0,
            chunk_frames=frame_count,
        )
        if int(mask_report["accepted_frame_count"]) == 0:
            output = source_chunk
            effective_mask = torch.zeros_like(used_mask)
        else:
            raw_candidate = _process_chunk(
                source_chunk,
                used_mask,
                preset=self.preset,
                amount=self.amount,
                texture_keep=self.texture_keep,
                shine_control=self.shine_control,
                tone_adjust=0.0,
                proxy_long_side=self.proxy_long_side,
            )
            (
                frequency_candidate,
                frequency_source,
                frequency_selected,
                _,
                frequency_mask,
                frequency_rejected,
                frequency_difference,
                frequency_json,
            ) = separate_skin_finish_frequencies(
                source_chunk,
                raw_candidate,
                used_mask,
                low_frequency_strength=self.low_frequency_strength,
                source_detail_gain=self.source_detail_gain,
                separation_radius_percent=self.separation_radius_percent,
                maximum_radius_px=self.maximum_radius_px,
                minimum_mask_area=0.00005,
                maximum_mask_area=0.35,
                maximum_new_clipped_fraction=0.0005,
                chunk_frames=frame_count,
                accept_candidate=False,
                audio=None,
            )
            frequency_report = json.loads(frequency_json)
            self.stats["frequency_rejected_frame_count"] += int(
                frequency_report["rejected_frame_count"]
            )
            if frequency_source is not source_chunk or frequency_selected is not source_chunk:
                raise RuntimeError("quality stream frequency stage changed source selection")
            (
                guarded_candidate,
                guard_source,
                guard_selected,
                _,
                guard_mask,
                guard_rejected,
                guard_difference,
                guard_json,
            ) = guard_skin_finish_candidate(
                source_chunk,
                frequency_candidate,
                frequency_mask,
                shadow_protection=self.shadow_protection,
                highlight_protection=self.highlight_protection,
                transition_width=0.06,
                minimum_texture_ratio=self.minimum_texture_ratio,
                minimum_reference_texture=0.003,
                maximum_new_clipped_fraction=0.0005,
                texture_radius=1,
                chunk_frames=frame_count,
                accept_candidate=False,
                audio=None,
            )
            guard_report = json.loads(guard_json)
            self.stats["texture_guard_rejected_frame_count"] += int(
                guard_report["rejected_frame_count"]
            )
            if guard_source is not source_chunk or guard_selected is not source_chunk:
                raise RuntimeError("quality stream guard stage changed source selection")

            if self.previous_source is None:
                audit_source = source_chunk
                audit_candidate = guarded_candidate
                audit_mask = guard_mask
                leading = 0
            else:
                audit_source = torch.cat((self.previous_source, source_chunk), dim=0)
                audit_candidate = torch.cat(
                    (self.previous_candidate, guarded_candidate), dim=0
                )
                audit_mask = torch.cat((self.previous_mask, guard_mask), dim=0)
                leading = 1
            if bool((audit_mask > 0).any()):
                (
                    audit_selected,
                    gated_candidate,
                    audit_source_output,
                    _,
                    hard_gate_pass,
                    failed_frame_count,
                    audit_preview,
                    audit_json,
                ) = audit_skin_finish_candidate(
                    audit_source,
                    audit_candidate,
                    audit_mask,
                    audit_scope="mask_only",
                    temporal_policy="hard_gate",
                    maximum_mean_abs_change=0.08,
                    maximum_peak_abs_change=0.30,
                    maximum_temporal_effect_jump=self.maximum_temporal_effect_jump,
                    minimum_temporal_pixels=64,
                    scene_cut_reset_threshold=0.20,
                    accept_candidate=False,
                    audio_source=None,
                    audio_passthrough=None,
                )
                audit_report = json.loads(audit_json)
                self.stats["maximum_temporal_effect_jump"] = max(
                    float(self.stats["maximum_temporal_effect_jump"]),
                    float(
                        audit_report["summary"][
                            "maximum_observed_temporal_effect_jump"
                        ]
                    ),
                )
                current_failed = sum(
                    int(index) >= leading
                    for index in audit_report["summary"]["failed_frame_indices"]
                )
                self.stats["safety_audit_failed_frame_count"] += current_failed
                if not hard_gate_pass:
                    self.stats["safety_audit_failed_chunk_count"] += 1
                    output = source_chunk
                    effective_mask = torch.zeros_like(guard_mask)
                else:
                    output = gated_candidate[leading:]
                    effective_mask = guard_mask
                del (
                    audit_selected,
                    audit_source_output,
                    audit_preview,
                    failed_frame_count,
                )
            else:
                output = source_chunk
                effective_mask = torch.zeros_like(guard_mask)

            del (
                raw_candidate,
                frequency_candidate,
                frequency_source,
                frequency_selected,
                frequency_mask,
                frequency_rejected,
                frequency_difference,
                guarded_candidate,
                guard_source,
                guard_selected,
                guard_rejected,
                guard_difference,
            )

        outside = effective_mask <= 0
        if not torch.equal(output[..., :3][outside], source_chunk[..., :3][outside]):
            raise RuntimeError("quality stream changed pixels outside its effective mask")
        self.previous_source = source_chunk[-1:].detach().clone()
        self.previous_candidate = output[-1:].detach().clone()
        self.previous_mask = effective_mask[-1:].detach().clone()
        gc.collect()
        return output, effective_mask

    def report(self) -> dict[str, Any]:
        return {
            "executed": bool(int(self.stats["chunk_count"])),
            "parser": {
                "name": PARSENET_MODEL_NAME,
                "path": str(self.model_path) if self.model_path else "",
                "sha256": self.model_hash,
                "device": "cpu",
                "loaded": self.model_loaded,
                "released_after_execute": self.model_released,
                "persistent_cache": False,
                "network_access": False,
            },
            "parameters": {
                "crop_expansion": self.crop_expansion,
                "minimum_class_probability": self.minimum_class_probability,
                "feature_protection_px": self.feature_protection_px,
                "mask_feather_px": self.mask_feather_px,
                "low_frequency_strength": self.low_frequency_strength,
                "source_detail_gain": self.source_detail_gain,
                "separation_radius_percent": self.separation_radius_percent,
                "maximum_radius_px": self.maximum_radius_px,
                "shadow_protection": self.shadow_protection,
                "highlight_protection": self.highlight_protection,
                "minimum_texture_ratio": self.minimum_texture_ratio,
                "maximum_temporal_effect_jump": self.maximum_temporal_effect_jump,
            },
            "summary": dict(self.stats),
        }


def stream_skin_finish_quality_video(
    source_video,
    *,
    preset: str = "subtle",
    amount: float = 0.30,
    texture_keep: float = 0.95,
    shine_control: float = 0.25,
    detection_threshold: float = 0.45,
    minimum_face_height_px: float = 32.0,
    minimum_detail: float = 0.010,
    bbox_ema_alpha: float = 0.55,
    scene_cut_threshold: float = 0.28,
    maximum_faces: int = 4,
    crop_expansion: float = 1.45,
    minimum_class_probability: float = 0.55,
    feature_protection_px: int = 4,
    mask_feather_px: int = 0,
    proxy_long_side: int = 640,
    low_frequency_strength: float = 1.0,
    source_detail_gain: float = 1.0,
    separation_radius_percent: float = 1.0,
    maximum_radius_px: int = 32,
    shadow_protection: float = 0.10,
    highlight_protection: float = 0.94,
    minimum_texture_ratio: float = 0.78,
    maximum_temporal_effect_jump: float = 0.04,
    chunk_frames: int = 2,
    filename_prefix: str = "MiniMaxH3/SkinFinish/quality_stream",
    crf: float = 18.0,
    accept_candidate: bool = False,
):
    if not 1.0 <= float(crop_expansion) <= 3.0:
        raise ValueError("crop_expansion must stay within 1.0..3.0")
    if not 0.0 <= float(minimum_class_probability) <= 1.0:
        raise ValueError("minimum_class_probability must stay within 0..1")
    if not 0 <= int(feature_protection_px) <= 64:
        raise ValueError("feature_protection_px must stay within 0..64")
    if not 0 <= int(mask_feather_px) <= 8:
        raise ValueError("quality stream mask_feather_px must stay within 0..8")
    if not 0.0 <= float(maximum_temporal_effect_jump) <= 1.0:
        raise ValueError("maximum_temporal_effect_jump must stay within 0..1")

    if not bool(accept_candidate):
        video, path, report_json, saved = stream_skin_finish_video(
            source_video,
            accept_candidate=False,
        )
        report = json.loads(report_json)
        report["schema"] = SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA
        report["quality_pipeline"] = {
            "executed": False,
            "parser_loaded": False,
            "reason": "source_selected_by_default",
        }
        report["resource_preflight"] = {
            "schema": SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA,
            "status": "SKIPPED_SOURCE_SELECTED_BY_DEFAULT",
            "allowed": False,
            "measurement_performed": False,
        }
        return video, path, canonical_json(report), saved

    resource_preflight = _quality_stream_ram_preflight()
    if not bool(resource_preflight["allowed"]):
        video, path, report_json, saved = stream_skin_finish_video(
            source_video,
            accept_candidate=False,
        )
        report = json.loads(report_json)
        report["schema"] = SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA
        report["status"] = str(resource_preflight["status"])
        report["quality_pipeline"] = {
            "executed": False,
            "parser_loaded": False,
            "reason": "insufficient_system_ram_preflight",
        }
        report["resource_preflight"] = resource_preflight
        report["automatic_accept"] = False
        report["human_review_required"] = False
        return video, path, canonical_json(report), saved

    processor = _QualityChunkProcessor(
        preset=preset,
        amount=amount,
        texture_keep=texture_keep,
        shine_control=shine_control,
        crop_expansion=crop_expansion,
        minimum_class_probability=minimum_class_probability,
        feature_protection_px=feature_protection_px,
        mask_feather_px=mask_feather_px,
        proxy_long_side=proxy_long_side,
        low_frequency_strength=low_frequency_strength,
        source_detail_gain=source_detail_gain,
        separation_radius_percent=separation_radius_percent,
        maximum_radius_px=maximum_radius_px,
        shadow_protection=shadow_protection,
        highlight_protection=highlight_protection,
        minimum_texture_ratio=minimum_texture_ratio,
        maximum_temporal_effect_jump=maximum_temporal_effect_jump,
    )
    result = None
    try:
        result = stream_skin_finish_video(
            source_video,
            preset=preset,
            amount=amount,
            texture_keep=texture_keep,
            shine_control=shine_control,
            detection_threshold=detection_threshold,
            minimum_face_height_px=minimum_face_height_px,
            minimum_detail=minimum_detail,
            bbox_ema_alpha=bbox_ema_alpha,
            scene_cut_threshold=scene_cut_threshold,
            maximum_faces=maximum_faces,
            mask_feather_px=mask_feather_px,
            proxy_long_side=proxy_long_side,
            chunk_frames=chunk_frames,
            filename_prefix=filename_prefix,
            crf=crf,
            accept_candidate=True,
            _chunk_processor=processor,
            _stream_report_label="parsenet_frequency_guard_safety",
        )
    finally:
        processor.close()
    assert result is not None
    video, path, report_json, saved = result
    report = json.loads(report_json)
    quality = processor.report()
    base_status = str(report.get("status", ""))
    report["schema"] = SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA
    if saved is None or not path:
        report["status"] = base_status or "ABSTAIN_NO_OUTPUT_FILE_WRITTEN"
    else:
        report["status"] = (
            "CANDIDATE_QUALITY_STREAM_FINALIZED_WITH_SOURCE_FALLBACKS"
            if int(quality["summary"]["source_only_frame_count"])
            or int(quality["summary"]["frequency_rejected_frame_count"])
            or int(quality["summary"]["texture_guard_rejected_frame_count"])
            or int(quality["summary"]["safety_audit_failed_chunk_count"])
            or int(quality["summary"]["safety_audit_failed_frame_count"])
            else "CANDIDATE_QUALITY_STREAM_FINALIZED"
        )
    report["quality_pipeline"] = quality
    report["resource_preflight"] = resource_preflight
    report["product_boundary"] = (
        "Bounded CPU ParseNet semantic skin masking, non-generative Skin Finish, source-detail "
        "frequency recombination, Texture Guard and cross-chunk source-relative Safety Audit. "
        "It cannot deblur, reconstruct identity, prove lip sync or choose aesthetic quality."
    )
    execution = report.setdefault("execution", {})
    execution["full_semantic_mask_batch_materialized"] = False
    execution["full_candidate_image_batch_materialized"] = False
    report["automatic_accept"] = False
    report["human_review_required"] = True
    return video, path, canonical_json(report), saved
