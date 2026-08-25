#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-oily-lora8-source-20260825"
    / "20260825-094343-e7088277"
    / "output"
    / "MiniMaxH3_SkinFinish"
    / "oily_lora8_speaking_00001_.mp4"
)
EXPECTED_SOURCE_SHA256 = (
    "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-oil-control-calibration-20260825-v3"

CALIBRATION_ARMS = OrderedDict(
    (
        (
            "current",
            {"amount": 0.35, "texture_keep": 0.90, "shine_control": 0.35},
        ),
        (
            "balanced",
            {"amount": 0.55, "texture_keep": 0.94, "shine_control": 0.60},
        ),
        (
            "strong",
            {"amount": 0.70, "texture_keep": 0.96, "shine_control": 0.75},
        ),
    )
)

SPECULAR_ROUTES = OrderedDict(
    (
        (
            "ordinary_balanced",
            {
                **CALIBRATION_ARMS["balanced"],
                "highlight_detail_suppression": 0.0,
            },
        ),
        (
            "specular_balanced_035",
            {
                **CALIBRATION_ARMS["balanced"],
                "highlight_detail_suppression": 0.35,
            },
        ),
        (
            "specular_balanced_065",
            {
                **CALIBRATION_ARMS["balanced"],
                "highlight_detail_suppression": 0.65,
            },
        ),
    )
)

UPPER_BOUND_ROUTES = OrderedDict(
    (
        (
            "ordinary_upper_bound",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 0.0,
            },
        ),
        (
            "specular_upper_bound_065",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 0.65,
            },
        ),
        (
            "specular_upper_bound_100",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 1.0,
            },
        ),
    )
)

BROAD_UPPER_BOUND_ROUTES = OrderedDict(
    (
        (
            "ordinary_broad_upper_bound",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 0.0,
                "separation_radius_percent": 3.0,
                "positive_detail_threshold": 0.004,
            },
        ),
        (
            "specular_broad_upper_bound_065",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 0.65,
                "separation_radius_percent": 3.0,
                "positive_detail_threshold": 0.004,
            },
        ),
        (
            "specular_broad_upper_bound_100",
            {
                "amount": 1.0,
                "texture_keep": 1.0,
                "shine_control": 1.0,
                "highlight_detail_suppression": 1.0,
                "separation_radius_percent": 3.0,
                "positive_detail_threshold": 0.004,
            },
        ),
    )
)

SPECULAR_ROUTE_PROFILES = {
    "balanced": SPECULAR_ROUTES,
    "broad_upper_bound": BROAD_UPPER_BOUND_ROUTES,
    "upper_bound": UPPER_BOUND_ROUTES,
}

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_skin_finish_quality_stream_representative as common  # noqa: E402


def _effective_highlight_residual_fraction(parameters: dict[str, float]) -> float:
    # Mirrors skin_finish._process_chunk for the oil_control preset:
    # highlight * preset.shine(1.25) * shine_control * 0.70 * amount.
    return (
        1.25
        * float(parameters["shine_control"])
        * 0.70
        * float(parameters["amount"])
    )


def _select_bin_peaks(scores: list[float], bins: int = 6) -> list[int]:
    if not scores:
        raise ValueError("scores must not be empty")
    if not 1 <= int(bins) <= len(scores):
        raise ValueError("bins must be within 1..len(scores)")
    selected: list[int] = []
    for bin_index in range(int(bins)):
        start = bin_index * len(scores) // int(bins)
        end = (bin_index + 1) * len(scores) // int(bins)
        if end <= start:
            raise RuntimeError("calibration bin is empty")
        selected.append(max(range(start, end), key=lambda index: (scores[index], -index)))
    return selected


def _validate_arms() -> None:
    previous = -1.0
    for name, parameters in CALIBRATION_ARMS.items():
        if set(parameters) != {"amount", "texture_keep", "shine_control"}:
            raise RuntimeError(f"calibration arm {name!r} has an invalid parameter set")
        if not all(0.0 <= float(value) <= 1.0 for value in parameters.values()):
            raise RuntimeError(f"calibration arm {name!r} is outside the public node range")
        fraction = _effective_highlight_residual_fraction(parameters)
        if fraction <= previous:
            raise RuntimeError("calibration highlight strength must increase monotonically")
        previous = fraction
    for profile_name, routes in SPECULAR_ROUTE_PROFILES.items():
        route_strengths = [
            float(parameters["highlight_detail_suppression"])
            for parameters in routes.values()
        ]
        if route_strengths != sorted(route_strengths) or route_strengths[0] != 0.0:
            raise RuntimeError("specular calibration routes must start at zero and increase")
        first_route = routes[next(iter(routes))]
        raw_contract = {
            key: float(value)
            for key, value in first_route.items()
            if key != "highlight_detail_suppression"
        }
        for parameters in routes.values():
            if not 0.0 <= float(parameters["highlight_detail_suppression"]) <= 1.0:
                raise RuntimeError("specular calibration strength is outside 0..1")
            for key, value in raw_contract.items():
                if float(parameters[key]) != value:
                    raise RuntimeError(
                        f"specular calibration profile {profile_name!r} must keep one raw arm"
                    )


def _oily_proxy_score(rgb, faces: list[dict]) -> float:
    import numpy as np

    if not faces:
        return float("-inf")
    face = max(
        faces,
        key=lambda item: max(0.0, float(item["box"][2]) - float(item["box"][0]))
        * max(0.0, float(item["box"][3]) - float(item["box"][1])),
    )
    height, width = rgb.shape[:2]
    left, top, right, bottom = [float(value) for value in face["box"]]
    left = max(0, min(width - 1, int(math.floor(left))))
    top = max(0, min(height - 1, int(math.floor(top))))
    right = max(left + 1, min(width, int(math.ceil(right))))
    bottom = max(top + 1, min(height, int(math.ceil(bottom))))
    crop = rgb[top:bottom, left:right]
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 16 or crop_w < 16:
        return float("-inf")
    luma = (
        crop[..., 0].astype(np.float32) * 0.2126
        + crop[..., 1].astype(np.float32) * 0.7152
        + crop[..., 2].astype(np.float32) * 0.0722
    ) / 255.0
    zones = (
        luma[int(crop_h * 0.08) : int(crop_h * 0.36), int(crop_w * 0.20) : int(crop_w * 0.80)],
        luma[int(crop_h * 0.48) : int(crop_h * 0.78), int(crop_w * 0.08) : int(crop_w * 0.38)],
        luma[int(crop_h * 0.48) : int(crop_h * 0.78), int(crop_w * 0.62) : int(crop_w * 0.92)],
    )
    values = np.concatenate([zone.reshape(-1) for zone in zones if zone.size])
    if not values.size:
        return float("-inf")
    return float(np.quantile(values, 0.99) - np.quantile(values, 0.75))


def _decode_selected(source: Path, records: list[list[dict]], bins: int):
    import av
    import numpy as np

    frames: list[np.ndarray] = []
    scores: list[float] = []
    with av.open(str(source), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(rgb)
            scores.append(_oily_proxy_score(rgb, records[index]))
    if len(frames) != len(records):
        raise RuntimeError("decoded frames do not match the pinned face-analysis records")
    indices = _select_bin_peaks(scores, bins=bins)
    selected_frames = [frames[index] for index in indices]
    selected_records = [records[index] for index in indices]
    selected_scores = [scores[index] for index in indices]
    return indices, selected_frames, selected_records, selected_scores


def _delta_metrics(source, candidate, mask) -> dict:
    import torch
    import torch.nn.functional as torch_functional

    source_rgb = source[..., :3].float()
    candidate_rgb = candidate[..., :3].float()
    active = mask > 0.10
    outside = mask <= 0.0
    delta = candidate_rgb - source_rgb
    selected = delta[active]
    if not int(selected.numel()):
        raise RuntimeError("calibration arm has no active semantic skin pixels")
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
    source_luma = (source_rgb * luma_weights).sum(dim=-1)
    candidate_luma = (candidate_rgb * luma_weights).sum(dim=-1)
    bright_deltas: list[torch.Tensor] = []
    for frame_index in range(int(source.shape[0])):
        local = active[frame_index]
        values = source_luma[frame_index][local]
        if int(values.numel()) < 16:
            continue
        threshold = torch.quantile(values, 0.90)
        bright = local & (source_luma[frame_index] >= threshold)
        bright_deltas.append(candidate_luma[frame_index][bright] - source_luma[frame_index][bright])
    bright_delta = torch.cat(bright_deltas) if bright_deltas else torch.zeros(1)
    source_high = source_luma.unsqueeze(1) - torch_functional.avg_pool2d(
        source_luma.unsqueeze(1), 3, stride=1, padding=1
    )
    candidate_high = candidate_luma.unsqueeze(1) - torch_functional.avg_pool2d(
        candidate_luma.unsqueeze(1), 3, stride=1, padding=1
    )
    high_mask = active.unsqueeze(1)
    source_texture = torch.sqrt((source_high[high_mask].square()).mean()).clamp_min(1e-8)
    candidate_texture = torch.sqrt((candidate_high[high_mask].square()).mean())
    outside_exact = bool(torch.equal(candidate_rgb[outside], source_rgb[outside]))
    return {
        "masked_mean_abs_rgb_change": round(float(selected.abs().mean()), 8),
        "masked_peak_abs_rgb_change": round(float(selected.abs().max()), 8),
        "brightest_skin_decile_mean_luma_delta": round(float(bright_delta.mean()), 8),
        "brightest_skin_decile_p10_luma_delta": round(
            float(torch.quantile(bright_delta, 0.10)), 8
        ),
        "texture_ratio_proxy": round(float(candidate_texture / source_texture), 8),
        "outside_effective_mask_bit_exact": outside_exact,
    }


def _arm_metrics(source, candidate, mask, guard_report: dict, audit_report: dict) -> dict:
    metrics = _delta_metrics(source, candidate, mask)
    metrics.update(
        {
        "texture_guard_status": str(guard_report["status"]),
        "texture_guard_rejected_frames": list(guard_report["rejected_frame_indices"]),
        "safety_status": str(audit_report["status"]),
        "safety_failed_frames": list(audit_report["summary"]["failed_frame_indices"]),
        }
    )
    return metrics


def _face_crop(image, faces: list[dict], expansion: float = 1.65):
    from PIL import Image

    height, width = image.shape[:2]
    if not faces:
        return Image.fromarray(image).resize((256, 256), Image.Resampling.LANCZOS)
    face = max(
        faces,
        key=lambda item: (float(item["box"][2]) - float(item["box"][0]))
        * (float(item["box"][3]) - float(item["box"][1])),
    )
    left, top, right, bottom = [float(value) for value in face["box"]]
    side = max(right - left, bottom - top) * float(expansion)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    x1 = max(0, int(math.floor(center_x - side * 0.5)))
    y1 = max(0, int(math.floor(center_y - side * 0.5)))
    x2 = min(width, int(math.ceil(center_x + side * 0.5)))
    y2 = min(height, int(math.ceil(center_y + side * 0.5)))
    return Image.fromarray(image[y1:y2, x1:x2]).resize(
        (256, 256), Image.Resampling.LANCZOS
    )


def _write_contact_sheet(
    path: Path,
    rows: OrderedDict[str, list],
    records: list[list[dict]],
    indices: list[int],
    scores: list[float],
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    cell = 256
    left_margin = 150
    header = 42
    canvas = Image.new(
        "RGB",
        (left_margin + cell * len(indices), header + cell * len(rows)),
        (24, 27, 32),
    )
    draw = ImageDraw.Draw(canvas)
    for column, (index, score) in enumerate(zip(indices, scores, strict=True)):
        draw.text(
            (left_margin + column * cell + 8, 12),
            f"frame {index}  oily-proxy {score:.3f}",
            fill=(235, 238, 243),
        )
    for row_index, (label, images) in enumerate(rows.items()):
        y = header + row_index * cell
        draw.text((12, y + 116), label.upper(), fill=(255, 255, 255))
        for column, image in enumerate(images):
            array = np.asarray(image, dtype=np.uint8)
            crop = _face_crop(array, records[column])
            canvas.paste(crop, (left_margin + column * cell, y))
    canvas.save(path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the ordinary and specular-aware frequency routes on six representative "
            "frames from the pinned MiniMax H3 v1.0 eight-step speaking source."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--representative-frames", type=int, default=6)
    parser.add_argument(
        "--calibration-profile",
        choices=sorted(SPECULAR_ROUTE_PROFILES),
        default="balanced",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    _validate_arms()
    routes = SPECULAR_ROUTE_PROFILES[str(args.calibration_profile)]
    plan = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "representative_frames": int(args.representative_frames),
        "calibration_profile": str(args.calibration_profile),
        "routes": {
            name: {
                **parameters,
                "effective_highlight_residual_fraction": round(
                    _effective_highlight_residual_fraction(parameters), 8
                ),
            }
            for name, parameters in routes.items()
        },
        "loads_h3": False,
        "loads_sam": False,
        "full_video_candidates": False,
        "stress_or_repeat": False,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0
    if not 3 <= int(args.representative_frames) <= 8:
        raise ValueError("representative-frames must stay within 3..8")

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or common._sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned v1.0 eight-step oily source is missing or changed")
    if not common.FFMPEG.is_file() or not common.FFPROBE.is_file():
        raise FileNotFoundError("bundled FFmpeg and FFprobe are required")
    os.environ["PATH"] = str(common.FFMPEG.parent) + os.pathsep + os.environ.get("PATH", "")
    common._strict_decode(source)

    common._load_package()
    import av
    import folder_paths
    import numpy as np
    import torch

    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish import (
        _prepare_mask,
        _process_chunk,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_frequency import (
        separate_skin_finish_frequencies,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_specular_frequency import (
        separate_skin_finish_specular_frequencies,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p1 import (
        _analyze_stream_faces,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p2 import (
        guard_skin_finish_candidate,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_safety_audit import (
        audit_skin_finish_candidate,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_stream_quality import (
        _QualityChunkProcessor,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        frame_count = int(stream.frames)
    if (width, height, frame_count) != (960, 544, 124):
        raise RuntimeError("calibration source contract must be 960x544x124")

    before = common._memory_mib()
    started = time.perf_counter()
    analysis = _analyze_stream_faces(
        source,
        expected_frame_count=frame_count,
        width=width,
        height=height,
        detection_threshold=0.35,
        minimum_face_height_px=32.0,
        minimum_detail=0.010,
        bbox_ema_alpha=0.55,
        scene_cut_threshold=0.28,
        maximum_faces=1,
        progress=None,
    )
    indices, arrays, selected_records, scores = _decode_selected(
        source, analysis["records"], int(args.representative_frames)
    )
    source_frames = torch.from_numpy(np.stack(arrays)).float().div_(255.0)

    parser = _QualityChunkProcessor(
        preset="oil_control",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        crop_expansion=1.45,
        minimum_class_probability=0.55,
        feature_protection_px=4,
        mask_feather_px=0,
        proxy_long_side=640,
        low_frequency_strength=1.0,
        source_detail_gain=1.0,
        separation_radius_percent=1.0,
        maximum_radius_px=32,
        shadow_protection=0.10,
        highlight_protection=0.94,
        minimum_texture_ratio=0.78,
        maximum_temporal_effect_jump=0.04,
    )
    try:
        raw_mask = parser._semantic_mask(source_frames, selected_records)
    finally:
        parser.close()
    used_mask, _, mask_report = _prepare_mask(
        raw_mask,
        frame_count=int(source_frames.shape[0]),
        height=height,
        width=width,
        minimum_area=0.00005,
        maximum_area=0.35,
        feather_px=0,
        temporal_radius=0,
        chunk_frames=int(source_frames.shape[0]),
    )
    if int(mask_report["accepted_frame_count"]) != int(source_frames.shape[0]):
        raise RuntimeError("not all calibration frames produced a reliable semantic skin mask")

    rows: OrderedDict[str, list] = OrderedDict()
    rows["source"] = arrays
    arm_reports = {}
    for name, parameters in routes.items():
        raw_candidate = _process_chunk(
            source_frames,
            used_mask,
            preset="oil_control",
            amount=float(parameters["amount"]),
            texture_keep=float(parameters["texture_keep"]),
            shine_control=float(parameters["shine_control"]),
            tone_adjust=0.0,
            proxy_long_side=640,
        )
        route_strength = float(parameters["highlight_detail_suppression"])
        frequency_function = (
            separate_skin_finish_frequencies
            if route_strength == 0.0
            else separate_skin_finish_specular_frequencies
        )
        frequency_kwargs = {
            "low_frequency_strength": 1.0,
            "source_detail_gain": 1.0,
            "separation_radius_percent": 1.0,
            "maximum_radius_px": 32,
            "minimum_mask_area": 0.00005,
            "maximum_mask_area": 0.35,
            "maximum_new_clipped_fraction": 0.0005,
            "chunk_frames": int(source_frames.shape[0]),
            "accept_candidate": False,
            "audio": None,
        }
        frequency_kwargs["separation_radius_percent"] = float(
            parameters.get("separation_radius_percent", 1.0)
        )
        if route_strength > 0.0:
            frequency_kwargs.update(
                {
                    "highlight_detail_suppression": route_strength,
                    "highlight_start": 0.60,
                    "highlight_end": 0.92,
                    "positive_detail_threshold": float(
                        parameters.get("positive_detail_threshold", 0.008)
                    ),
                    "treatment_intent_scale": 0.004,
                    "maximum_specular_delta": 0.04,
                }
            )
        frequency_candidate, _, _, _, frequency_mask, _, _, frequency_json = (
            frequency_function(
                source_frames,
                raw_candidate,
                used_mask,
                **frequency_kwargs,
            )
        )
        guarded, _, _, _, guard_mask, _, _, guard_json = guard_skin_finish_candidate(
            source_frames,
            frequency_candidate,
            frequency_mask,
            shadow_protection=0.10,
            highlight_protection=0.94,
            transition_width=0.06,
            minimum_texture_ratio=0.78,
            minimum_reference_texture=0.003,
            maximum_new_clipped_fraction=0.0005,
            texture_radius=1,
            chunk_frames=int(source_frames.shape[0]),
            accept_candidate=False,
            audio=None,
        )
        _, gated, _, _, hard_gate_pass, _, _, audit_json = audit_skin_finish_candidate(
            source_frames,
            guarded,
            guard_mask,
            audit_scope="mask_only",
            temporal_policy="report_only",
            maximum_mean_abs_change=0.08,
            maximum_peak_abs_change=0.30,
            maximum_temporal_effect_jump=0.04,
            minimum_temporal_pixels=64,
            scene_cut_reset_threshold=0.20,
            accept_candidate=False,
            audio_source=None,
            audio_passthrough=None,
        )
        guard_report = json.loads(guard_json)
        audit_report = json.loads(audit_json)
        if not hard_gate_pass:
            raise RuntimeError(f"calibration arm {name!r} failed a static hard gate")
        if not bool(torch.isfinite(gated).all()):
            raise RuntimeError(f"calibration arm {name!r} produced non-finite values")
        raw_metrics = _delta_metrics(source_frames, raw_candidate, used_mask)
        frequency_metrics = _delta_metrics(
            source_frames, frequency_candidate, frequency_mask
        )
        final_metrics = _arm_metrics(
            source_frames, gated, guard_mask, guard_report, audit_report
        )
        raw_mean = max(float(raw_metrics["masked_mean_abs_rgb_change"]), 1.0e-12)
        arm_reports[name] = {
            "parameters": dict(parameters),
            "effective_highlight_residual_fraction": round(
                _effective_highlight_residual_fraction(parameters), 8
            ),
            "metrics": {
                "raw_process": raw_metrics,
                "after_frequency_split": frequency_metrics,
                "final_guarded": final_metrics,
                "final_to_raw_mean_change_ratio": round(
                    float(final_metrics["masked_mean_abs_rgb_change"]) / raw_mean, 8
                ),
            },
            "frequency_status": json.loads(frequency_json)["status"],
            "frequency_route": (
                "ordinary_frequency_split"
                if route_strength == 0.0
                else "specular_aware_frequency_split"
            ),
        }
        rows[name] = [
            tensor.mul(255.0).round().clamp(0, 255).byte().numpy()
            for tensor in gated[..., :3]
        ]

    contact_sheet = output / "face_contact_sheet_source_ordinary_specular.png"
    _write_contact_sheet(contact_sheet, rows, selected_records, indices, scores)
    after = common._memory_mib()
    report = {
        "schema": "h3_t8_skin_finish_oil_control_calibration/v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_READY_FOR_HUMAN_SELECTION",
        "source": {
            "path": str(source),
            "sha256": common._sha256(source),
            "contract": {"width": width, "height": height, "frames": frame_count, "fps": 24},
        },
        "selection": {
            "method": "six_equal_time_bins_then_max_forehead_cheek_highlight_proxy",
            "frame_indices": indices,
            "oily_proxy_scores": [round(float(score), 8) for score in scores],
            "semantic_mask_report": mask_report,
        },
        "calibration_profile": str(args.calibration_profile),
        "arms": arm_reports,
        "runtime": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "memory_mib_before": before,
            "memory_mib_after": after,
            "torch_cpu_threads": 2,
            "h3_loaded": False,
            "sam_loaded": False,
            "parsenet_loaded_once_for_selected_frames": True,
            "full_video_candidates_generated": False,
            "stress_or_repeat": False,
        },
        "outputs": {
            "face_contact_sheet": str(contact_sheet),
            "face_contact_sheet_sha256": common._sha256(contact_sheet),
        },
        "automatic_selection": False,
        "claim_boundary": (
            "This six-frame calibration compares visible strength and static hard gates only. "
            "Non-contiguous selected frames cannot validate temporal stability, full-video mouth "
            "motion, identity, aesthetics or universal memory safety."
        ),
    }
    report_path = output / "calibration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    (output / "README.md").write_text(
        "# Skin Finish oil-control calibration\n\n"
        "Rows are SOURCE followed by the three route names recorded in calibration_report.json. "
        "Columns are six independently selected high-highlight frames across the 124-frame "
        "source. All three candidate rows use the exact same raw Skin Finish parameters within "
        "the selected calibration profile; only the frequency route changes. This is a "
        "labelled calibration sheet, not a blind test. Inspect highlight reduction, waxiness, "
        "eyes, lips and texture before choosing any route for a full-video temporal review.\n\n"
        "No H3 or SAM model was loaded; no full-video candidate or stress run was made.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "contact_sheet": str(contact_sheet),
                "selected_frames": indices,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
