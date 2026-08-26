from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
GFPGAN_PATH = COMFY_ROOT / "models" / "facerestore_models" / "GFPGANv1.4.pth"
GFPGAN_SHA256 = "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"
PINNED_FRAME_INDICES = (16, 20, 60, 66, 86, 119)
PINNED_OILY_SCORES = {
    16: 0.10139453,
    20: 0.08966595,
    60: 0.12249494,
    66: 0.14067555,
    86: 0.13648236,
    119: 0.12865484,
}
DEFAULT_SINGLE_OUTPUT = ROOT / "artifacts" / "skin-finish-learned-surface-probe-20260825-v6"
DEFAULT_SIX_OUTPUT = ROOT / "artifacts" / "skin-finish-learned-surface-calibration-20260825-v6"
FUSION_PARAMETERS = OrderedDict(
    [
        ("amount", 0.70),
        ("surface_amount", 0.45),
        ("surface_radius_px", 10),
        ("maximum_surface_mismatch", 0.12),
        ("maximum_surface_luma_delta", 0.035),
        ("chroma_amount", 0.20),
        ("maximum_chroma_component_delta", 0.04),
        ("candidate_rgb_delta_cap", 0.10),
        ("detail_radius_px", 2),
        ("energy_radius_px", 5),
        ("maximum_detail_gain", 1.80),
        ("maximum_linear_luma_delta", 0.025),
        ("low_frequency_tolerance", 0.025),
        ("minimum_source_detail_rms", 0.00075),
        ("minimum_source_luma", 0.005),
        ("maximum_texture_ratio", 1.45),
        ("maximum_mean_abs_change", 0.025),
        ("maximum_peak_abs_change", 0.12),
        ("minimum_mask_area", 0.0001),
        ("maximum_mask_area", 0.50),
        ("chunk_frames", 1),
    ]
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_indices(source: Path, indices: tuple[int, ...]):
    import av

    selected = set(indices)
    frames = {}
    with av.open(str(source), mode="r") as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in selected:
                frames[frame_index] = frame.to_ndarray(format="rgb24")
    missing = sorted(selected - set(frames))
    if missing:
        raise RuntimeError(f"pinned source is missing selected frames {missing}")
    return [frames[index] for index in indices]


def _largest_detection(detections: list[dict]) -> dict:
    if not detections:
        raise RuntimeError("YuNet did not detect a face on a pinned calibration frame")
    usable = [item for item in detections if len(item.get("landmarks_xy", [])) == 5]
    if not usable:
        raise RuntimeError("YuNet detections do not contain five landmarks")
    return max(
        usable,
        key=lambda item: (
            float(item["box"][2] - item["box"][0])
            * float(item["box"][3] - item["box"][1]),
            float(item.get("confidence", 0.0)),
        ),
    )


def _warp_scale_to_source(
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

    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_dichromatic import (
        _linear_to_srgb,
        _srgb_to_linear,
    )

    height, width = map(int, source_frame.shape[:2])
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
        1, 3, 1, 1
    )
    aligned_source_linear = _srgb_to_linear(aligned_source.movedim(-1, 1))
    aligned_candidate_linear = _srgb_to_linear(aligned_candidate.movedim(-1, 1))
    aligned_source_luma = (aligned_source_linear * weights).sum(dim=1, keepdim=True)
    aligned_candidate_luma = (aligned_candidate_linear * weights).sum(
        dim=1, keepdim=True
    )
    scale_delta = torch.where(
        aligned_source_luma > 1.0e-8,
        aligned_candidate_luma / aligned_source_luma.clamp_min(1.0e-8) - 1.0,
        torch.zeros_like(aligned_source_luma),
    )[0, 0]
    scale_delta *= aligned_effective_mask[0]
    warped_scale_delta = cv2.warpAffine(
        scale_delta.numpy(),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_mask = cv2.warpAffine(
        aligned_effective_mask[0].numpy(),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    source = torch.from_numpy(np.ascontiguousarray(source_frame)).float().div_(255.0)
    source_linear = _srgb_to_linear(source.movedim(-1, 0).unsqueeze(0))
    requested_scale = 1.0 + torch.from_numpy(warped_scale_delta).view(1, 1, height, width)
    maximum_channel = source_linear.amax(dim=1, keepdim=True)
    maximum_scale = torch.where(
        maximum_channel > 1.0e-8,
        maximum_channel.reciprocal(),
        torch.ones_like(maximum_channel),
    )
    applied_scale = torch.minimum(requested_scale.clamp_min(0.0), maximum_scale)
    raw_candidate = _linear_to_srgb(source_linear * applied_scale)[0].movedim(0, -1)
    raw_delta = raw_candidate - source
    raw_peak = raw_delta.abs().amax(dim=-1, keepdim=True)
    limiter = (
        torch.full_like(raw_peak, float(rgb_delta_cap))
        / raw_peak.clamp_min(1.0e-8)
    ).clamp_max(1.0)
    candidate = source + raw_delta * limiter
    effective = torch.from_numpy(np.ascontiguousarray(warped_mask)).float().clamp_(0.0, 1.0)
    candidate = torch.where(effective.unsqueeze(-1) > 0.0, candidate, source)
    return candidate, effective, {
        "raw_peak_abs_rgb_change_before_limit": round(float(raw_peak.max()), 8),
        "rgb_delta_limited_pixel_fraction": round(
            float((limiter[..., 0] < 1.0).float().mean()), 8
        ),
        "rgb_delta_cap": float(rgb_delta_cap),
    }


def _write_aligned_sheet(path: Path, rows: OrderedDict[str, list], indices: tuple[int, ...]):
    import numpy as np
    from PIL import Image, ImageDraw

    cell = 256
    left = 175
    header = 38
    canvas = Image.new(
        "RGB",
        (left + cell * len(indices), header + cell * len(rows)),
        (24, 27, 32),
    )
    draw = ImageDraw.Draw(canvas)
    for column, frame_index in enumerate(indices):
        draw.text((left + column * cell + 8, 10), f"frame {frame_index}", fill="white")
    for row_index, (label, images) in enumerate(rows.items()):
        y = header + row_index * cell
        draw.text((10, y + 116), label.upper(), fill="white")
        for column, image in enumerate(images):
            array = np.asarray(image, dtype=np.uint8)
            canvas.paste(
                Image.fromarray(array).resize((cell, cell), Image.Resampling.LANCZOS),
                (left + column * cell, y),
            )
    canvas.save(path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pinned GFPGAN proposal probe or one fixed six-frame calibration. "
            "The learned RGB is never pasted into the source."
        )
    )
    parser.add_argument("--mode", choices=("single", "six"), default="single")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-run", action="store_true")
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
        "loads_h3": False,
        "loads_sam": False,
        "runs_full_video": False,
        "stress_or_repeat": False,
        "automatic_accept": False,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0

    source = args.source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")
    if not source.is_file() or common._sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned eight-step source is missing or changed")
    if not GFPGAN_PATH.is_file() or _file_sha256(GFPGAN_PATH) != GFPGAN_SHA256:
        raise RuntimeError("the installed GFPGAN v1.4 checkpoint is missing or changed")
    output.mkdir(parents=True, exist_ok=True)
    common._load_package()

    import comfy.model_management
    import comfy.utils
    import folder_paths
    import numpy as np
    import torch
    from spandrel import ModelLoader

    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_learned_detail import (
        fuse_proposal_guided_skin_detail,
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
    arrays = _decode_indices(source, indices)
    detector = parser_model = descriptor = state = None
    before = common._memory_mib()
    started = time.perf_counter()
    model_inference_seconds = 0.0
    aligned_source_rows = []
    aligned_proposal_rows = []
    aligned_candidate_rows = []
    full_candidate_rows = []
    records = []
    frame_reports = []
    try:
        detector, detector_report = _create_pinned_yunet(960, 544, 0.35)
        parser_model, parser_path, parser_sha = _load_pinned_parsenet()
        state = comfy.utils.load_torch_file(str(GFPGAN_PATH), safe_load=True)
        descriptor = ModelLoader().load_from_state_dict(state).eval()
        del state
        state = None
        descriptor.model.to(device=effective_device, dtype=effective_dtype)

        for frame_index, array in zip(indices, arrays, strict=True):
            detection = _largest_detection(_detect_yunet_rgb(detector, array))
            records.append([detection])
            frame_rgb = np.ascontiguousarray(array).astype(np.float32) / 255.0
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
                fuse_proposal_guided_skin_detail(
                    aligned,
                    proposal,
                    aligned_skin,
                    **FUSION_PARAMETERS,
                )
            )
            fusion_report = json.loads(fusion_json)
            if fusion_report["status"] != "PASS" or bool(rejected.any()):
                failure_path = output / f"rejected_fusion_frame_{frame_index}.json"
                failure_path.write_text(
                    json.dumps(
                        {
                            "status": "REJECTED_BEFORE_FULL_FRAME_COMPOSITION",
                            "frame_index": frame_index,
                            "fusion": fusion_report,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                raise RuntimeError(
                    f"learned-detail fusion rejected pinned frame {frame_index}: "
                    f"{fusion_report['status']}; report={failure_path}"
                )
            full_candidate, full_effective, full_composition = _warp_scale_to_source(
                array,
                aligned,
                candidate,
                effective,
                inverse,
                rgb_delta_cap=float(FUSION_PARAMETERS["candidate_rgb_delta_cap"]),
            )
            source_full = torch.from_numpy(array.copy()).float().div_(255.0)
            full_delta = (full_candidate - source_full).abs()
            active = full_effective > 0.10
            if not bool(active.any()):
                raise RuntimeError("aligned learned-detail mask did not map back to source")
            outside = full_effective <= 0.0
            if not torch.equal(full_candidate[outside], source_full[outside]):
                raise RuntimeError("full-frame learned-detail candidate changed mask exterior")
            final_full_peak = float(full_delta[active].max())
            if final_full_peak > float(FUSION_PARAMETERS["candidate_rgb_delta_cap"]) + 1.0e-6:
                raise RuntimeError(
                    "full-frame learned-detail candidate exceeded its RGB delta cap"
                )
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
                    "alignment_rms": round(float(alignment_rms), 8),
                    "detector_confidence": round(float(detection["confidence"]), 8),
                    "semantic_stats": semantic_stats,
                    "fusion": fusion_report,
                    "full_frame_effective_area_fraction": round(
                        float((full_effective > 0.10).float().mean()), 8
                    ),
                    "full_frame_masked_mean_abs_rgb_change": round(
                        float(full_delta[active].mean()), 8
                    ),
                    "full_frame_masked_peak_abs_rgb_change": round(
                        final_full_peak, 8
                    ),
                    "full_frame_composition": full_composition,
                    "outside_effective_mask_bit_exact": True,
                }
            )
            del model_input, proposal_nchw, proposal, logits, candidate, effective
    finally:
        del detector, parser_model, descriptor, state
        gc.collect()
        if torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    aligned_sheet = output / "aligned_source_gfpgan_proposal_source_detail.png"
    _write_aligned_sheet(
        aligned_sheet,
        OrderedDict(
            [
                ("source aligned", aligned_source_rows),
                ("GFPGAN proposal - not pasted", aligned_proposal_rows),
                ("source-phase detail candidate", aligned_candidate_rows),
            ]
        ),
        indices,
    )
    full_sheet = output / "full_frame_source_vs_source_detail.png"
    calibration._write_contact_sheet(
        full_sheet,
        OrderedDict([("source", arrays), ("learned detail candidate", full_candidate_rows)]),
        records,
        list(indices),
        [PINNED_OILY_SCORES[index] for index in indices],
    )
    report = {
        "schema": "h3_t8_skin_finish_learned_detail_calibration/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SINGLE_FRAME_PROBE_COMPLETE" if args.mode == "single" else "SIX_FRAME_CALIBRATION_COMPLETE",
        "plan": plan,
        "source_sha256": common._sha256(source),
        "model": {
            "name": "GFPGANv1.4",
            "path": str(GFPGAN_PATH),
            "sha256": GFPGAN_SHA256,
            "loader": "ComfyUI bundled Spandrel ImageModelDescriptor",
            "license_boundary": (
                "GFPGAN repository is Apache-2.0 with third-party notices; this project does not "
                "redistribute the checkpoint or vendor GFPGAN architecture code."
            ),
            "proposal_rgb_pasted": False,
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
            "A single frame gates model loading and geometry only; six frames gate static "
            "visibility and preservation only. Neither establishes identity fidelity, temporal "
            "stability, general skin quality or a production-ready node."
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
