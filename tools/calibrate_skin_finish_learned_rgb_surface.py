from __future__ import annotations

import argparse
import gc
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import calibrate_skin_finish_learned_detail as learned_detail
import calibrate_skin_finish_oil_control as calibration
import validate_skin_finish_quality_stream_representative as common


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = learned_detail.DEFAULT_SOURCE
EXPECTED_SOURCE_SHA256 = learned_detail.EXPECTED_SOURCE_SHA256
GFPGAN_PATH = learned_detail.GFPGAN_PATH
GFPGAN_SHA256 = learned_detail.GFPGAN_SHA256
PINNED_FRAME_INDICES = learned_detail.PINNED_FRAME_INDICES
PINNED_OILY_SCORES = learned_detail.PINNED_OILY_SCORES
DEFAULT_SINGLE_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-learned-rgb-surface-probe-20260825-v1"
)
DEFAULT_SIX_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-learned-rgb-surface-calibration-20260825-v1"
)
FUSION_PARAMETERS = OrderedDict(
    [
        ("amount", 0.55),
        ("surface_radius_px", 10),
        ("maximum_proposal_low_rgb_delta", 0.18),
        ("candidate_rgb_delta_cap", 0.10),
        ("minimum_masked_mean_abs_change", 0.005),
        ("maximum_masked_mean_abs_change", 0.055),
        ("maximum_peak_abs_change", 0.105),
        ("minimum_texture_ratio", 0.88),
        ("maximum_texture_ratio", 1.15),
        ("maximum_new_clipped_fraction", 0.002),
        ("minimum_mask_area", 0.0001),
        ("maximum_mask_area", 0.50),
        ("chunk_frames", 1),
    ]
)
MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE = 0.012
MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE = 0.90


def _warp_rgb_delta_to_source(
    source_frame,
    aligned_source,
    aligned_candidate,
    aligned_effective_mask,
    inverse,
    *,
    rgb_delta_cap: float,
):
    import cv2
    import numpy as np
    import torch

    height, width = map(int, source_frame.shape[:2])
    aligned_delta = (
        aligned_candidate[0, ..., :3].float() - aligned_source[0, ..., :3].float()
    ).numpy()
    warped_delta = cv2.warpAffine(
        aligned_delta,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    warped_mask = cv2.warpAffine(
        aligned_effective_mask[0].float().numpy(),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    source = torch.from_numpy(np.ascontiguousarray(source_frame)).float().div_(255.0)
    delta = torch.from_numpy(np.ascontiguousarray(warped_delta)).float()
    raw_peak = delta.abs().amax(dim=-1, keepdim=True)
    limiter = (
        torch.full_like(raw_peak, float(rgb_delta_cap))
        / raw_peak.clamp_min(1.0e-8)
    ).clamp_max(1.0)
    limited_delta = delta * limiter
    raw_candidate = source + limited_delta
    candidate = raw_candidate.clamp(0.0, 1.0)
    effective = torch.from_numpy(np.ascontiguousarray(warped_mask)).float().clamp_(0.0, 1.0)
    candidate = torch.where(effective.unsqueeze(-1) > 0.0, candidate, source)
    return candidate, effective, {
        "raw_peak_abs_rgb_change_before_limit": round(float(raw_peak.max()), 8),
        "rgb_delta_limited_pixel_fraction": round(
            float((limiter[..., 0] < 1.0).float().mean()), 8
        ),
        "rgb_delta_cap": float(rgb_delta_cap),
        "new_clipped_pixel_fraction": round(
            float(
                (
                    ((candidate <= 0.0) | (candidate >= 1.0)).any(dim=-1)
                    & ~((source <= 0.0) | (source >= 1.0)).any(dim=-1)
                    & (effective > 0.0)
                )
                .float()
                .mean()
            ),
            8,
        ),
    }


def _sface_similarity(source, candidate, source_detection, detector, recognizer):
    import numpy as np
    import torch

    from h3_audio_t8_skin_finish_quality_stream_validation.multiface_refine_advanced import (
        _sface_feature,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p1 import (
        _detect_yunet_rgb,
    )

    candidate_array = (
        candidate.mul(255.0).round().clamp(0, 255).byte().numpy()
    )
    candidate_detection = learned_detail._largest_detection(
        _detect_yunet_rgb(detector, np.ascontiguousarray(candidate_array))
    )
    source_feature = _sface_feature(source, source_detection, recognizer)
    candidate_feature = _sface_feature(candidate, candidate_detection, recognizer)
    return float(torch.dot(source_feature, candidate_feature)), candidate_detection


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe bounded low-frequency GFPGAN RGB skin-surface transfer on one pinned frame. "
            "Six-frame mode exists only for the next gate and must not run before the single "
            "frame is visibly different and identity-safe."
        )
    )
    parser.add_argument("--mode", choices=("single", "six"), default="single")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-single-gate-passed", action="store_true")
    args = parser.parse_args()
    indices = (66,) if args.mode == "single" else PINNED_FRAME_INDICES
    output = (
        args.output
        if args.output is not None
        else DEFAULT_SINGLE_OUTPUT
        if args.mode == "single"
        else DEFAULT_SIX_OUTPUT
    )
    plan = {
        "mode": args.mode,
        "source": str(args.source.resolve()),
        "output": str(output.resolve()),
        "frame_indices": list(indices),
        "learned_model": str(GFPGAN_PATH),
        "learned_model_sha256": GFPGAN_SHA256,
        "fusion_parameters": FUSION_PARAMETERS,
        "minimum_full_frame_masked_mean_change": MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE,
        "minimum_sface_source_candidate_cosine": MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
        "loads_h3": False,
        "loads_sam": False,
        "runs_full_video": False,
        "stress_or_repeat": False,
        "automatic_accept": False,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "six" and not args.confirm_single_gate_passed:
        raise RuntimeError(
            "ABSTAIN_SINGLE_FRAME_GATE_NOT_CONFIRMED: six-frame work is forbidden until the "
            "single-frame candidate is visibly different and passes identity/mechanical gates"
        )

    source_path = args.source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")
    if not source_path.is_file() or common._sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned eight-step source is missing or changed")
    if not GFPGAN_PATH.is_file() or learned_detail._file_sha256(GFPGAN_PATH) != GFPGAN_SHA256:
        raise RuntimeError("the installed GFPGAN v1.4 checkpoint is missing or changed")
    output.mkdir(parents=True, exist_ok=True)
    common._load_package()

    import comfy.model_management
    import comfy.utils
    import folder_paths
    import numpy as np
    import torch
    from spandrel import ModelLoader

    from h3_audio_t8_skin_finish_quality_stream_validation.multiface_refine_advanced import (
        _create_sface_recognizer,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_learned_rgb_surface import (
        fuse_learned_low_frequency_rgb_surface,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_multiface_parser import (
        _align_face,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p1 import (
        _create_pinned_yunet,
        _detect_yunet_rgb,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_parser import (
        _load_pinned_parsenet,
        _parser_logits,
        _semantic_local_masks,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 4096 * 2**20:
            raise RuntimeError(
                "ABSTAIN_INSUFFICIENT_FREE_VRAM: GFPGAN probe requires 4096 MiB free"
            )
    effective_device = torch.device(device)
    effective_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    arrays = learned_detail._decode_indices(source_path, indices)
    detector = parser_model = descriptor = state = recognizer = None
    before = common._memory_mib()
    started = time.perf_counter()
    model_inference_seconds = 0.0
    aligned_source_rows = []
    aligned_proposal_rows = []
    aligned_candidate_rows = []
    full_candidate_rows = []
    records = []
    frame_reports = []
    all_gates_passed = True
    try:
        detector, detector_report = _create_pinned_yunet(960, 544, 0.35)
        parser_model, parser_path, parser_sha = _load_pinned_parsenet()
        recognizer = _create_sface_recognizer()
        state = comfy.utils.load_torch_file(str(GFPGAN_PATH), safe_load=True)
        descriptor = ModelLoader().load_from_state_dict(state).eval()
        del state
        state = None
        descriptor.model.to(device=effective_device, dtype=effective_dtype)

        for frame_index, array in zip(indices, arrays, strict=True):
            detection = learned_detail._largest_detection(
                _detect_yunet_rgb(detector, array)
            )
            records.append([detection])
            frame_rgb = np.ascontiguousarray(array).astype(np.float32) / 255.0
            source_full = torch.from_numpy(frame_rgb.copy()).float()
            aligned, inverse, alignment_rms = _align_face(
                frame_rgb,
                detection,
                maximum_alignment_rms=0.08,
            )
            logits = _parser_logits(parser_model, aligned)
            aligned_skin, _, semantic_stats = _semantic_local_masks(
                logits,
                include_neck=False,
                minimum_class_probability=0.55,
                feature_protection_px=4,
            )
            model_input = aligned[..., :3].movedim(-1, 1).to(
                device=effective_device, dtype=effective_dtype
            ).contiguous()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                proposal_nchw = descriptor(model_input)
            if device == "cuda":
                torch.cuda.synchronize()
            model_inference_seconds += time.perf_counter() - inference_started
            proposal = proposal_nchw.float().cpu().movedim(1, -1)
            candidate, effective, rejected, _, fusion_json = (
                fuse_learned_low_frequency_rgb_surface(
                    aligned,
                    proposal,
                    aligned_skin,
                    **FUSION_PARAMETERS,
                )
            )
            fusion_report = json.loads(fusion_json)
            full_candidate, full_effective, full_composition = (
                _warp_rgb_delta_to_source(
                    array,
                    aligned,
                    candidate,
                    effective,
                    inverse,
                    rgb_delta_cap=float(FUSION_PARAMETERS["candidate_rgb_delta_cap"]),
                )
            )
            full_delta = (full_candidate - source_full).abs()
            active = full_effective > 0.10
            outside = full_effective <= 0.0
            if not torch.equal(full_candidate[outside], source_full[outside]):
                raise RuntimeError("full-frame RGB-surface candidate changed mask exterior")
            full_mean = float(full_delta[active].mean()) if bool(active.any()) else 0.0
            full_peak = float(full_delta[active].max()) if bool(active.any()) else 0.0
            similarity, candidate_detection = _sface_similarity(
                source_full,
                full_candidate,
                detection,
                detector,
                recognizer,
            )
            gates = {
                "fusion_passed": fusion_report["status"]
                == "PASS_REQUIRES_HUMAN_REVIEW"
                and not bool(rejected.any()),
                "full_frame_effect_visible": full_mean
                >= MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE,
                "full_frame_rgb_cap_passed": full_peak
                <= float(FUSION_PARAMETERS["candidate_rgb_delta_cap"]) + 1.0e-6,
                "sface_identity_gate_passed": similarity
                >= MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
                "outside_effective_mask_bit_exact": True,
            }
            frame_passed = all(gates.values())
            all_gates_passed &= frame_passed
            aligned_source_rows.append(
                aligned[0].mul(255.0).round().clamp(0, 255).byte().numpy()
            )
            aligned_proposal_rows.append(
                proposal[0].mul(255.0).round().clamp(0, 255).byte().numpy()
            )
            aligned_candidate_rows.append(
                candidate[0].mul(255.0).round().clamp(0, 255).byte().numpy()
            )
            full_candidate_rows.append(
                full_candidate.mul(255.0).round().clamp(0, 255).byte().numpy()
            )
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "gate_status": "PASS" if frame_passed else "ABSTAIN",
                    "gates": gates,
                    "alignment_rms": round(float(alignment_rms), 8),
                    "detector_confidence": round(float(detection["confidence"]), 8),
                    "candidate_detector_confidence": round(
                        float(candidate_detection["confidence"]), 8
                    ),
                    "sface_source_candidate_cosine": round(similarity, 8),
                    "semantic_stats": semantic_stats,
                    "fusion": fusion_report,
                    "full_frame_effective_area_fraction": round(
                        float(active.float().mean()), 8
                    ),
                    "full_frame_masked_mean_abs_rgb_change": round(full_mean, 8),
                    "full_frame_masked_peak_abs_rgb_change": round(full_peak, 8),
                    "full_frame_composition": full_composition,
                }
            )
            del model_input, proposal_nchw, proposal, logits, candidate, effective
    finally:
        del detector, parser_model, descriptor, state, recognizer
        gc.collect()
        if torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    aligned_sheet = output / "aligned_source_gfpgan_proposal_low_rgb_candidate.png"
    learned_detail._write_aligned_sheet(
        aligned_sheet,
        OrderedDict(
            [
                ("source aligned", aligned_source_rows),
                ("GFPGAN proposal - not pasted", aligned_proposal_rows),
                ("low-frequency RGB candidate", aligned_candidate_rows),
            ]
        ),
        indices,
    )
    full_sheet = output / "full_frame_source_vs_low_rgb_candidate.png"
    calibration._write_contact_sheet(
        full_sheet,
        OrderedDict([("source", arrays), ("low RGB candidate", full_candidate_rows)]),
        records,
        list(indices),
        [PINNED_OILY_SCORES[index] for index in indices],
    )
    report = {
        "schema": "h3_t8_skin_finish_learned_rgb_surface_calibration/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SINGLE_FRAME_GATE_PASSED_REQUIRES_HUMAN_REVIEW"
            if args.mode == "single" and all_gates_passed
            else "SIX_FRAME_GATE_PASSED_REQUIRES_HUMAN_REVIEW"
            if args.mode == "six" and all_gates_passed
            else "ABSTAIN_MECHANICAL_OR_VISIBILITY_GATE_FAILED"
        ),
        "plan": plan,
        "source_sha256": common._sha256(source_path),
        "model": {
            "name": "GFPGANv1.4",
            "path": str(GFPGAN_PATH),
            "sha256": GFPGAN_SHA256,
            "proposal_geometry_or_high_frequency_pasted": False,
            "only_low_frequency_rgb_surface_transferred": True,
        },
        "identity": {
            "backend": "opencv_zoo_sface_2021dec_cpu",
            "minimum_cosine": MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
            "boundary": "A model-specific safety signal, not identity or biometric proof.",
            "released": True,
        },
        "parser": {"path": str(parser_path), "sha256": parser_sha, "released": True},
        "detector": detector_report,
        "runtime": {
            "device": device,
            "dtype": str(effective_dtype),
            "model_inference_seconds": round(model_inference_seconds, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "memory_before": before,
            "memory_after": common._memory_mib(),
            "h3_loaded": False,
            "sam_loaded": False,
            "full_video_generated": False,
            "stress_or_repeat": False,
            "models_released_after_execute": True,
        },
        "frames": frame_reports,
        "outputs": {
            "aligned_contact_sheet": str(aligned_sheet),
            "aligned_contact_sheet_sha256": common._sha256(aligned_sheet),
            "full_frame_contact_sheet": str(full_sheet),
            "full_frame_contact_sheet_sha256": common._sha256(full_sheet),
        },
        "automatic_selection": False,
        "claim_boundary": (
            "Passing this gate means only that one pinned frame is visibly changed, bounded and "
            "SFace-close to its own source. It does not establish temporal stability, general "
            "skin-quality benefit or a production-ready node."
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
                "aligned_contact_sheet": str(aligned_sheet),
                "full_frame_contact_sheet": str(full_sheet),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
