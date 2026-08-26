from __future__ import annotations

import argparse
import gc
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import calibrate_skin_finish_learned_detail as learned_detail
import calibrate_skin_finish_learned_mid_surface as mid_surface
import calibrate_skin_finish_learned_rgb_surface as low_rgb
import calibrate_skin_finish_oil_control as calibration
import validate_skin_finish_quality_stream_representative as common


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = learned_detail.DEFAULT_SOURCE
EXPECTED_SOURCE_SHA256 = learned_detail.EXPECTED_SOURCE_SHA256
GFPGAN_PATH = learned_detail.GFPGAN_PATH
GFPGAN_SHA256 = learned_detail.GFPGAN_SHA256
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-learned-skin-reconstruction-probe-20260825-v1"
)
FUSION_PARAMETERS = OrderedDict(
    [
        ("amount", 0.70),
        ("proposal_prefilter_radius_px", 1),
        ("maximum_proposal_component_delta", 0.25),
        ("candidate_rgb_delta_cap", 0.12),
        ("flat_edge_low", 0.012),
        ("flat_edge_high", 0.035),
        ("aligned_edge_cosine_low", 0.70),
        ("aligned_edge_cosine_high", 0.90),
        ("minimum_structural_gradient_cosine", 0.92),
        ("minimum_masked_mean_abs_change", 0.025),
        ("maximum_masked_mean_abs_change", 0.080),
        ("minimum_texture_ratio", 0.55),
        ("maximum_texture_ratio", 1.30),
        ("maximum_new_clipped_fraction", 0.002),
        ("minimum_mask_area", 0.0001),
        ("maximum_mask_area", 0.50),
        ("chunk_frames", 1),
    ]
)
SFACE_OFFICIAL_COSINE_THRESHOLD = 0.363
SFACE_ADDITIONAL_SAFETY_MARGIN = 0.20
MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE = 0.563
MINIMUM_SFACE_IMPROVEMENT_OVER_RAW_PROPOSAL = 0.02
MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE = 0.025


def _save_rgb(path: Path, value) -> None:
    import numpy as np
    from PIL import Image

    array = value
    if hasattr(value, "detach"):
        array = value.detach().mul(255.0).round().clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(np.ascontiguousarray(array).astype(np.uint8), mode="RGB").save(
        path, format="PNG", optimize=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed CPU-only semantic-skin GFPGAN RGB reconstruction probe. "
            "This deliberately higher-risk route saves full-resolution evidence and never "
            "auto-accepts or runs video."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    frame_index = 66
    plan = {
        "mode": "single",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "frame_indices": [frame_index],
        "learned_model": str(GFPGAN_PATH),
        "learned_model_sha256": GFPGAN_SHA256,
        "fusion_parameters": FUSION_PARAMETERS,
        "identity_gate": {
            "official_sface_cosine_threshold": SFACE_OFFICIAL_COSINE_THRESHOLD,
            "additional_safety_margin": SFACE_ADDITIONAL_SAFETY_MARGIN,
            "minimum_source_candidate_cosine": MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
            "minimum_improvement_over_raw_gfpgan_proposal": (
                MINIMUM_SFACE_IMPROVEMENT_OVER_RAW_PROPOSAL
            ),
        },
        "minimum_full_frame_masked_mean_change": (
            MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE
        ),
        "saves_full_resolution_evidence": True,
        "loads_h3": False,
        "loads_sam": False,
        "runs_full_video": False,
        "stress_or_repeat": False,
        "automatic_accept": False,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0

    source_path = args.source.resolve()
    output = args.output.resolve()
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
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_learned_skin_reconstruction import (
        fuse_bounded_semantic_skin_reconstruction,
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
    array = learned_detail._decode_indices(source_path, (frame_index,))[0]
    detector = aligned_detector = parser_model = descriptor = state = recognizer = None
    before = common._memory_mib()
    started = time.perf_counter()
    model_inference_seconds = 0.0
    try:
        detector, detector_report = _create_pinned_yunet(960, 544, 0.35)
        aligned_detector, aligned_detector_report = _create_pinned_yunet(512, 512, 0.35)
        parser_model, parser_path, parser_sha = _load_pinned_parsenet()
        recognizer = _create_sface_recognizer()
        state = comfy.utils.load_torch_file(str(GFPGAN_PATH), safe_load=True)
        descriptor = ModelLoader().load_from_state_dict(state).eval()
        del state
        state = None
        descriptor.model.to(device=effective_device, dtype=effective_dtype)

        detection = learned_detail._largest_detection(_detect_yunet_rgb(detector, array))
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
        model_inference_seconds = time.perf_counter() - inference_started
        proposal = proposal_nchw.float().cpu().movedim(1, -1)
        candidate, effective, rejected, _, fusion_json = (
            fuse_bounded_semantic_skin_reconstruction(
                aligned,
                proposal,
                aligned_skin,
                **FUSION_PARAMETERS,
            )
        )
        fusion_report = json.loads(fusion_json)
        full_candidate, full_effective, full_composition = (
            low_rgb._warp_rgb_delta_to_source(
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
            raise RuntimeError("full-frame skin reconstruction changed mask exterior")
        full_mean = float(full_delta[active].mean()) if bool(active.any()) else 0.0
        full_peak = float(full_delta[active].max()) if bool(active.any()) else 0.0

        source_feature, source_aligned_detection = mid_surface._aligned_sface_feature(
            aligned, aligned_detector, recognizer
        )
        proposal_feature, proposal_detection = mid_surface._aligned_sface_feature(
            proposal, aligned_detector, recognizer
        )
        candidate_feature, candidate_aligned_detection = (
            mid_surface._aligned_sface_feature(candidate, aligned_detector, recognizer)
        )
        raw_proposal_similarity = float(torch.dot(source_feature, proposal_feature))
        aligned_candidate_similarity = float(torch.dot(source_feature, candidate_feature))
        full_candidate_similarity, full_candidate_detection = low_rgb._sface_similarity(
            source_full,
            full_candidate,
            detection,
            detector,
            recognizer,
        )
        identity_improvement = aligned_candidate_similarity - raw_proposal_similarity
        gates = {
            "fusion_passed": fusion_report["status"]
            == "PASS_REQUIRES_IDENTITY_AND_HUMAN_REVIEW"
            and not bool(rejected.any()),
            "full_frame_effect_visible": full_mean
            >= MINIMUM_FULL_FRAME_MASKED_MEAN_CHANGE,
            "full_frame_rgb_cap_passed": full_peak
            <= float(FUSION_PARAMETERS["candidate_rgb_delta_cap"]) + 1.0e-6,
            "aligned_sface_official_plus_margin_passed": aligned_candidate_similarity
            >= MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
            "full_frame_sface_official_plus_margin_passed": full_candidate_similarity
            >= MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
            "sface_improved_over_raw_proposal": identity_improvement
            >= MINIMUM_SFACE_IMPROVEMENT_OVER_RAW_PROPOSAL,
            "outside_effective_mask_bit_exact": True,
        }
        all_gates_passed = all(gates.values())

        aligned_source_path = output / "aligned_source_512.png"
        aligned_proposal_path = output / "aligned_raw_gfpgan_proposal_512.png"
        aligned_candidate_path = output / "aligned_skin_reconstruction_candidate_512.png"
        full_source_path = output / "full_source_960x544.png"
        full_candidate_path = output / "full_skin_reconstruction_candidate_960x544.png"
        _save_rgb(aligned_source_path, aligned[0])
        _save_rgb(aligned_proposal_path, proposal[0])
        _save_rgb(aligned_candidate_path, candidate[0])
        _save_rgb(full_source_path, array)
        _save_rgb(full_candidate_path, full_candidate)

        aligned_sheet = output / "aligned_source_gfpgan_proposal_skin_reconstruction.png"
        learned_detail._write_aligned_sheet(
            aligned_sheet,
            OrderedDict(
                [
                    ("source aligned", [array for array in [aligned[0].mul(255.0).round().clamp(0, 255).byte().numpy()]]),
                    ("GFPGAN proposal - not pasted whole", [proposal[0].mul(255.0).round().clamp(0, 255).byte().numpy()]),
                    ("semantic skin reconstruction", [candidate[0].mul(255.0).round().clamp(0, 255).byte().numpy()]),
                ]
            ),
            (frame_index,),
        )
        full_sheet = output / "full_frame_source_vs_skin_reconstruction.png"
        calibration._write_contact_sheet(
            full_sheet,
            OrderedDict(
                [
                    ("source", [array]),
                    (
                        "semantic skin reconstruction",
                        [full_candidate.mul(255.0).round().clamp(0, 255).byte().numpy()],
                    ),
                ]
            ),
            [[detection]],
            [frame_index],
            [learned_detail.PINNED_OILY_SCORES[frame_index]],
        )

        frame_report = {
            "frame_index": frame_index,
            "gate_status": "PASS" if all_gates_passed else "ABSTAIN",
            "gates": gates,
            "alignment_rms": round(float(alignment_rms), 8),
            "detector_confidence": round(float(detection["confidence"]), 8),
            "full_candidate_detector_confidence": round(
                float(full_candidate_detection["confidence"]), 8
            ),
            "aligned_source_detector_confidence": round(
                float(source_aligned_detection["confidence"]), 8
            ),
            "aligned_raw_proposal_detector_confidence": round(
                float(proposal_detection["confidence"]), 8
            ),
            "aligned_candidate_detector_confidence": round(
                float(candidate_aligned_detection["confidence"]), 8
            ),
            "sface_source_raw_proposal_cosine": round(raw_proposal_similarity, 8),
            "sface_source_aligned_candidate_cosine": round(
                aligned_candidate_similarity, 8
            ),
            "sface_source_full_candidate_cosine": round(
                full_candidate_similarity, 8
            ),
            "sface_candidate_improvement_over_raw_proposal": round(
                identity_improvement, 8
            ),
            "semantic_stats": semantic_stats,
            "fusion": fusion_report,
            "full_frame_effective_area_fraction": round(float(active.float().mean()), 8),
            "full_frame_masked_mean_abs_rgb_change": round(full_mean, 8),
            "full_frame_masked_peak_abs_rgb_change": round(full_peak, 8),
            "full_frame_composition": full_composition,
        }
    finally:
        del detector, aligned_detector, parser_model, descriptor, state, recognizer
        gc.collect()
        if torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    output_files = [
        aligned_source_path,
        aligned_proposal_path,
        aligned_candidate_path,
        full_source_path,
        full_candidate_path,
        aligned_sheet,
        full_sheet,
    ]
    report = {
        "schema": "h3_t8_skin_finish_learned_skin_reconstruction_calibration/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SINGLE_FRAME_GATE_PASSED_REQUIRES_HUMAN_REVIEW"
            if all_gates_passed
            else "ABSTAIN_MECHANICAL_VISIBILITY_OR_IDENTITY_GATE_FAILED"
        ),
        "plan": plan,
        "source_sha256": common._sha256(source_path),
        "model": {
            "name": "GFPGANv1.4",
            "path": str(GFPGAN_PATH),
            "sha256": GFPGAN_SHA256,
            "whole_face_proposal_pasted": False,
            "generated_semantic_skin_rgb_can_be_transferred": True,
        },
        "identity": {
            "backend": "opencv_zoo_sface_2021dec_cpu",
            "official_cosine_threshold": SFACE_OFFICIAL_COSINE_THRESHOLD,
            "additional_safety_margin": SFACE_ADDITIONAL_SAFETY_MARGIN,
            "minimum_cosine": MINIMUM_SFACE_SOURCE_CANDIDATE_COSINE,
            "minimum_improvement_over_raw_proposal": (
                MINIMUM_SFACE_IMPROVEMENT_OVER_RAW_PROPOSAL
            ),
            "boundary": "Model-specific research gates, not identity or biometric proof.",
            "released": True,
        },
        "parser": {"path": str(parser_path), "sha256": parser_sha, "released": True},
        "detector": detector_report,
        "aligned_detector": aligned_detector_report,
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
        "frames": [frame_report],
        "outputs": {
            path.stem: {"path": str(path), "sha256": common._sha256(path)}
            for path in output_files
        },
        "automatic_selection": False,
        "risk_boundary": (
            "This route can transfer generated skin RGB and texture. Semantic feature exclusion, "
            "structure gating and caps reduce but do not remove hallucination, identity or "
            "temporal risk. Passing one frame cannot authorize a node or video claim."
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
                "full_source": str(full_source_path),
                "full_candidate": str(full_candidate_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
