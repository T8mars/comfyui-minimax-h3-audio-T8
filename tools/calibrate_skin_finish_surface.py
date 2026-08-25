from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import calibrate_skin_finish_oil_control as calibration
import validate_skin_finish_quality_stream_representative as common


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = calibration.DEFAULT_SOURCE
EXPECTED_SOURCE_SHA256 = calibration.EXPECTED_SOURCE_SHA256
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-surface-calibration-20260825-v5"
SURFACE_ARMS = OrderedDict(
    [
        (
            "localized_highlight_oil_v3",
            {
                "amount": 0.90,
                "surface_smoothing": 0.25,
                "texture_keep": 0.96,
                "highlight_compression": 0.90,
                "broad_highlight_compression": 0.90,
                "broad_highlight_start": 0.68,
                "broad_highlight_end": 0.94,
                "blemish_balance": 0.10,
                "surface_radius_percent": 2.5,
            },
        ),
    ]
)


def _as_uint8_rows(value):
    return [
        frame.mul(255.0).round().clamp(0, 255).byte().numpy()
        for frame in value[..., :3]
    ]


def _boundary_diagnostics(source, candidate, mask) -> dict:
    import torch.nn.functional as torch_functional

    active = mask > 0.10
    eroded = (
        -torch_functional.max_pool2d(
            -active.float().unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2,
        )
    ).squeeze(1) > 0.99
    boundary = active & ~eroded
    delta = (candidate[..., :3].float() - source[..., :3].float()).abs().mean(
        dim=-1
    )

    def _mean(region):
        return round(float(delta[region].mean()), 8) if bool(region.any()) else 0.0

    boundary_mean = _mean(boundary)
    interior_mean = _mean(eroded)
    return {
        "two_pixel_boundary_mean_abs_rgb_change": boundary_mean,
        "interior_mean_abs_rgb_change": interior_mean,
        "boundary_to_interior_change_ratio": round(
            boundary_mean / max(interior_mean, 1.0e-8), 8
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current bounded Quality Stream baseline with one revised clean-room "
            "localized Surface Finish candidate on six pinned representative frames."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    plan = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "representative_frames": 6,
        "arms": SURFACE_ARMS,
        "loads_h3": False,
        "loads_sam": False,
        "runs_full_video": False,
        "stress_or_repeat": False,
        "torch_cpu_threads": 2,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or common._sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned v1.0 eight-step source is missing or changed")
    if not common.FFMPEG.is_file() or not common.FFPROBE.is_file():
        raise FileNotFoundError("bundled FFmpeg and FFprobe are required")
    os.environ["PATH"] = str(common.FFMPEG.parent) + os.pathsep + os.environ.get(
        "PATH", ""
    )
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
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_surface import (
        finish_skin_surface,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        width, height, frame_count = (
            int(stream.width),
            int(stream.height),
            int(stream.frames),
        )
    if (width, height, frame_count) != (960, 544, 124):
        raise RuntimeError("surface calibration source must be 960x544x124")

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
    indices, arrays, selected_records, scores = calibration._decode_selected(
        source, analysis["records"], 6
    )
    source_frames = torch.from_numpy(np.stack(arrays)).float().div_(255.0)
    mask_parser = _QualityChunkProcessor(
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
        raw_mask = mask_parser._semantic_mask(source_frames, selected_records)
    finally:
        mask_parser.close()
    used_mask, _, mask_report = _prepare_mask(
        raw_mask,
        frame_count=6,
        height=height,
        width=width,
        minimum_area=0.00005,
        maximum_area=0.35,
        feather_px=0,
        temporal_radius=0,
        chunk_frames=6,
    )
    if int(mask_report["accepted_frame_count"]) != 6:
        raise RuntimeError("not all calibration frames produced a reliable semantic mask")

    baseline_raw = _process_chunk(
        source_frames,
        used_mask,
        preset="oil_control",
        amount=0.35,
        texture_keep=0.90,
        shine_control=0.35,
        tone_adjust=0.0,
        proxy_long_side=640,
    )
    baseline_frequency, _, _, _, baseline_mask, _, _, _ = (
        separate_skin_finish_frequencies(
            source_frames,
            baseline_raw,
            used_mask,
            low_frequency_strength=1.0,
            source_detail_gain=1.0,
            separation_radius_percent=1.0,
            maximum_radius_px=32,
            minimum_mask_area=0.00005,
            maximum_mask_area=0.35,
            chunk_frames=6,
            accept_candidate=False,
            audio=None,
        )
    )

    candidates = OrderedDict([("current_quality_stream", (baseline_frequency, baseline_mask, {}))])
    for name, parameters in SURFACE_ARMS.items():
        candidate, _, _, _, effective, _, _, report_json = finish_skin_surface(
            source_frames,
            used_mask,
            **parameters,
            minimum_mask_area=0.00005,
            maximum_mask_area=0.35,
            chunk_frames=2,
            accept_candidate=False,
            audio=None,
        )
        candidates[name] = (candidate, effective, json.loads(report_json))

    rows = OrderedDict([("source", arrays)])
    arm_reports = {}
    for name, (candidate, candidate_mask, internal_report) in candidates.items():
        guarded, _, _, _, guard_mask, _, _, guard_json = guard_skin_finish_candidate(
            source_frames,
            candidate,
            candidate_mask,
            shadow_protection=0.10,
            highlight_protection=0.94,
            transition_width=0.06,
            minimum_texture_ratio=0.78,
            minimum_reference_texture=0.003,
            maximum_new_clipped_fraction=0.0005,
            texture_radius=1,
            chunk_frames=6,
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
        arm_reports[name] = {
            "parameters": (
                {
                    "preset": "oil_control",
                    "amount": 0.35,
                    "texture_keep": 0.90,
                    "shine_control": 0.35,
                    "frequency": "1.0/1.0/1%",
                }
                if name == "current_quality_stream"
                else SURFACE_ARMS[name]
            ),
            "internal_report": internal_report,
            "pre_guard_metrics": calibration._delta_metrics(
                source_frames, candidate, candidate_mask
            ),
            "final_guarded_metrics": calibration._arm_metrics(
                source_frames, gated, guard_mask, guard_report, audit_report
            ),
            "boundary_diagnostics": _boundary_diagnostics(
                source_frames, gated, guard_mask
            ),
            "hard_gate_pass": bool(hard_gate_pass),
        }
        rows[name] = _as_uint8_rows(gated)

    contact_sheet = output / "face_contact_sheet_source_current_surface.png"
    calibration._write_contact_sheet(
        contact_sheet, rows, selected_records, indices, scores
    )
    report = {
        "schema": "h3_t8_skin_finish_surface_calibration/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_READY_FOR_HUMAN_SELECTION",
        "source": {
            "path": str(source),
            "sha256": common._sha256(source),
            "contract": {
                "width": width,
                "height": height,
                "frames": frame_count,
                "fps": 24,
            },
        },
        "selection": {
            "frame_indices": indices,
            "oily_proxy_scores": [round(float(score), 8) for score in scores],
            "semantic_mask_report": mask_report,
        },
        "arms": arm_reports,
        "runtime": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "memory_mib_before": before,
            "memory_mib_after": common._memory_mib(),
            "torch_cpu_threads": 2,
            "h3_loaded": False,
            "sam_loaded": False,
            "full_video_candidates_generated": False,
            "stress_or_repeat": False,
        },
        "outputs": {
            "face_contact_sheet": str(contact_sheet),
            "face_contact_sheet_sha256": common._sha256(contact_sheet),
        },
        "automatic_selection": False,
        "claim_boundary": (
            "Six non-contiguous frames can gate static visibility and preservation only. They "
            "cannot validate temporal flicker, speech, identity, aesthetic preference or a "
            "full-video memory envelope."
        ),
    }
    report_path = output / "calibration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "contact_sheet": str(contact_sheet),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
