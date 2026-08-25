from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
UPSTREAM_COMMIT = "e7d5facafd95c97190fcf54171960f25c21b3043"
EXPECTED_UPSTREAM_FILE_SHA256 = (
    "1F8D8EBD44FEA4C75A0C77D2798173A525B2CCBFDEAFE60F0C82F74B3CB7FDF6"
)
DEFAULT_UPSTREAM_FILE = Path(
    r"C:\Users\Administrator\AppData\Local\Temp"
    r"\h3-t8-cinestyle-audit-e7d5fac\py\vfx_beauty.py"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-cinestyle-reference-20260825"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_pinned_upstream(path: Path):
    if not path.is_file() or _sha256(path) != EXPECTED_UPSTREAM_FILE_SHA256:
        raise RuntimeError(
            "the pinned CineStyle vfx_beauty.py audit file is missing or changed"
        )
    spec = importlib.util.spec_from_file_location(
        "h3_t8_cinestyle_reference_vfx_beauty", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned CineStyle audit file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_uint8_rows(value):
    return [
        frame.mul(255.0).round().clamp(0, 255).byte().numpy()
        for frame in value[..., :3]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one six-frame CPU-only reference audit of the pinned CineStyle Beauty "
            "defaults using T8's exact semantic skin mask. No upstream code is vendored."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--upstream-file", type=Path, default=DEFAULT_UPSTREAM_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    plan = {
        "source": str(args.source.resolve()),
        "upstream_file": str(args.upstream_file.resolve()),
        "upstream_commit": UPSTREAM_COMMIT,
        "output": str(args.output.resolve()),
        "representative_frames": 6,
        "loads_h3": False,
        "loads_sam": False,
        "runs_full_video": False,
        "stress_or_repeat": False,
        "cpu_only": True,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0

    source = args.source.resolve()
    upstream_file = args.upstream_file.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite reference evidence: {output}")
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
        width, height, frame_count = (
            int(stream.width),
            int(stream.height),
            int(stream.frames),
        )
    if (width, height, frame_count) != (960, 544, 124):
        raise RuntimeError("reference source contract must be 960x544x124")

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
        raise RuntimeError("not all reference frames produced a reliable semantic mask")

    upstream = _load_pinned_upstream(upstream_file)
    upstream._preferred_device = lambda _fallback: torch.device("cpu")
    upstream_output = upstream._run_beauty(
        source_frames,
        used_mask,
        None,
        torch.tensor([6.0, 0.0, 3.0], dtype=torch.float32),
        10.0,
        10.0,
        15.0,
        0.8,
        0.5,
        0.0,
        0.0,
        30.0,
        0.2,
        100.0,
        0.0,
        progress=None,
    )[..., :3]
    t8_max = _process_chunk(
        source_frames,
        used_mask,
        preset="oil_control",
        amount=1.0,
        texture_keep=1.0,
        shine_control=1.0,
        tone_adjust=0.0,
        proxy_long_side=640,
    )
    guarded, _, _, _, guard_mask, _, _, guard_json = guard_skin_finish_candidate(
        source_frames,
        upstream_output,
        used_mask,
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
    rows = OrderedDict(
        [
            ("source", arrays),
            ("cinestyle_default_raw", _as_uint8_rows(upstream_output)),
            ("cinestyle_default_guarded", _as_uint8_rows(gated)),
            ("t8_oil_control_max_raw", _as_uint8_rows(t8_max)),
        ]
    )
    contact_sheet = output / "face_contact_sheet_source_cinestyle_t8.png"
    calibration._write_contact_sheet(
        contact_sheet, rows, selected_records, indices, scores
    )
    report = {
        "schema": "h3_t8_skin_finish_cinestyle_reference_audit/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "REFERENCE_READY_FOR_HUMAN_INSPECTION",
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
        "upstream": {
            "repository": "https://github.com/chflame163/ComfyUI_CineStyle",
            "commit": UPSTREAM_COMMIT,
            "file_sha256": _sha256(upstream_file),
            "code_vendored": False,
            "execution_device": "cpu",
            "parameters": {
                "weights": "6.0,0.0,3.0",
                "blur_m": 10.0,
                "sigma": 10.0,
                "threshold": 15.0,
                "r_spots_blend": 0.8,
                "r_h_blend": 0.5,
                "strength": 0.0,
                "blur_h": 0.0,
                "blur_s": 30.0,
                "o_amount": 0.2,
                "sat_amount": 100.0,
                "hue_amount": 0.0,
            },
        },
        "selection": {
            "frame_indices": indices,
            "oily_proxy_scores": [round(float(score), 8) for score in scores],
            "semantic_mask_report": mask_report,
            "same_exact_t8_mask_for_all_rows": True,
        },
        "metrics": {
            "cinestyle_default_raw": calibration._delta_metrics(
                source_frames, upstream_output, used_mask
            ),
            "cinestyle_default_guarded": calibration._arm_metrics(
                source_frames, gated, guard_mask, guard_report, audit_report
            ),
            "t8_oil_control_max_raw": calibration._delta_metrics(
                source_frames, t8_max, used_mask
            ),
        },
        "hard_gate_pass": bool(hard_gate_pass),
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
        "claim_boundary": (
            "This is an audit-only execution of a separately downloaded, pinned upstream file. "
            "It compares six non-contiguous frames and cannot prove video temporal stability, "
            "aesthetic preference, licensing provenance, or that T8 may copy the upstream passes."
        ),
    }
    report_path = output / "reference_report.json"
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
                "hard_gate_pass": bool(hard_gate_pass),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
