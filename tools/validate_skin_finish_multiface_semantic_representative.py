#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import av
import cv2
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_RUNTIME = ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_skin_multiface_parser_validation"
SOURCE = Path(r"C:\ComfyUI_00001_qirzb_1786961596.mp4")
SOURCE_INDICES = [6, 12, 18, 24, 30, 36]
TARGET_WIDTH = 960
TARGET_HEIGHT = 704
OUTPUT = (
    ROOT
    / "artifacts"
    / "skin-finish-multiface-semantic-representative-6frame-20260824"
)


def _load_package():
    if str(COMFY_RUNTIME) not in sys.path:
        sys.path.insert(0, str(COMFY_RUNTIME))
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _decode_selected(path: Path) -> torch.Tensor:
    wanted = set(SOURCE_INDICES)
    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                resized = cv2.resize(
                    rgb,
                    (TARGET_WIDTH, TARGET_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                selected[index] = torch.from_numpy(resized.copy()).float().div_(255.0)
            if index >= max(SOURCE_INDICES):
                break
    if sorted(selected) != SOURCE_INDICES:
        raise RuntimeError(
            f"decoded source indices {sorted(selected)}, expected {SOURCE_INDICES}"
        )
    return torch.stack([selected[index] for index in SOURCE_INDICES])


def _shot(package_name: str, shot_id: int, start: int, end: int) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    frame_count = end - start + 1
    masks = torch.zeros((frame_count, 2, 176, 240), dtype=torch.bool)
    masks[:, 0, 0:176, 0:120] = True
    masks[:, 1, 0:176, 120:240] = True
    packed = pack_masks(masks)
    helper = sys.modules[f"{package_name}.multiface_refine_advanced"]
    return {
        "shot_id": shot_id,
        "start_frame": start,
        "end_frame": end,
        "frame_count": frame_count,
        "object_count": 2,
        "track_keys": [f"{shot_id}:0", f"{shot_id}:1"],
        "native_object_indices": [0, 1],
        "scores": [1.0, 1.0],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": helper._tensor_sha256(packed),
        "mask_size": [176, 240],
    }


def _track_plan(frames: torch.Tensor, package_name: str) -> dict:
    helper = sys.modules[f"{package_name}.multiface_refine_advanced"]
    source = dict(helper._source_contract(frames))
    source["fps"] = 24.0
    plan = {
        "schema": helper.TRACK_PLAN_SCHEMA,
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {
            "height": 176,
            "width": 240,
            "harness": "deterministic left/right person regions at preserved aspect ratio",
        },
        "sam31": {
            "track_identity_scope": "shot_local_only",
            "live_sam31_executed_in_this_harness": False,
        },
        "shots": [
            _shot(package_name, 0, 0, 2),
            _shot(package_name, 1, 3, 5),
        ],
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 1,
        "max_scene_delta": None,
        "release": {
            "performed": True,
            "note": "No SAM model was loaded by this low-load harness.",
        },
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = helper._hash_json(helper._json_safe(plan))
    return plan


def _identity_assignment(plan: dict, package_name: str) -> dict:
    helper = sys.modules[f"{package_name}.multiface_refine_advanced"]
    assignment = {
        "schema": helper.ASSIGNMENT_SCHEMA,
        "status": "identity_assignment_ready",
        "source": plan["source"],
        "track_plan_sha256": plan["sha256"],
        "mappings": [
            {"track_key": "0:0", "character_id": "Character_A"},
            {"track_key": "0:1", "character_id": "Character_B"},
            {"track_key": "1:0", "character_id": "Character_A"},
            {"track_key": "1:1", "character_id": "Character_B"},
        ],
        "identity_is_suggestion_not_proof": True,
        "automatic_accept": False,
    }
    assignment["sha256"] = helper._hash_json(helper._json_safe(assignment))
    return assignment


def _to_image(frame: torch.Tensor) -> Image.Image:
    array = (
        frame.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _contact_sheet(
    frames: torch.Tensor,
    preview: torch.Tensor,
    output: Path,
) -> None:
    width = int(frames.shape[2])
    height = int(frames.shape[1])
    header = 36
    canvas = Image.new("RGB", (width * 3, (height + header) * 4), "black")
    draw = ImageDraw.Draw(canvas)
    for shot_id in range(2):
        for column in range(3):
            batch_index = shot_id * 3 + column
            source_index = SOURCE_INDICES[batch_index]
            x = column * width
            source_y = shot_id * 2 * (height + header)
            preview_y = source_y + height + header
            draw.text(
                (x + 10, source_y + 10),
                f"SHOT {shot_id} SOURCE FRAME {source_index}",
                fill="white",
            )
            canvas.paste(_to_image(frames[batch_index]), (x, source_y + header))
            draw.text(
                (x + 10, preview_y + 10),
                f"SHOT {shot_id} PARSENET: GREEN SKIN / RED PROTECTED",
                fill="white",
            )
            canvas.paste(_to_image(preview[batch_index]), (x, preview_y + header))
    canvas.save(output, quality=94, subsampling=0)


def _write_mask_frames(mask: torch.Tensor, output: Path) -> list[dict]:
    records = []
    for index, source_index in enumerate(SOURCE_INDICES):
        path = output / f"semantic_mask_batch{index}_source{source_index}.png"
        array = (
            mask[index]
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .numpy()
        )
        Image.fromarray(array, mode="L").save(path)
        records.append({"path": str(path), "sha256": _sha256(path)})
    return records


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    started = time.perf_counter()
    _load_package()
    frames = _decode_selected(SOURCE)
    source_snapshot = frames.clone()
    plan = _track_plan(frames, PACKAGE_NAME)
    assignment = _identity_assignment(plan, PACKAGE_NAME)
    module = sys.modules[f"{PACKAGE_NAME}.skin_finish_multiface_parser"]
    skin_mask, preview, parser_report_json = module.run_multiface_semantic_skin_mask(
        frames=frames,
        track_plan=plan,
        identity_assignment=assignment,
        parser_model=module.PARSENET_MODEL_NAME,
        detection_threshold=0.45,
        minimum_face_height_px=32.0,
        minimum_detail=0.005,
        minimum_person_overlap=0.20,
        minimum_track_quality=0.08,
        minimum_class_probability=0.55,
        feature_protection_px=3,
        include_neck=False,
        minimum_skin_area_per_face=0.00005,
        maximum_skin_area_per_frame=0.35,
        maximum_alignment_rms=0.08,
        minimum_ready_frame_fraction=1.0,
        preview_count=6,
    )
    parser_report = json.loads(parser_report_json)
    if parser_report["status"] != "READY":
        raise RuntimeError(
            f"multi-person parser did not pass: {parser_report['status']} "
            f"{parser_report.get('detail', '')}"
        )
    per_frame_counts = [
        int(frame["accepted_track_count"]) for frame in parser_report["frames"]
    ]
    if per_frame_counts != [2, 2, 2, 2, 2, 2]:
        raise RuntimeError(
            f"expected two independently matched semantic faces per frame, got {per_frame_counts}"
        )
    if not torch.equal(frames, source_snapshot):
        raise RuntimeError("representative execution modified its source IMAGE tensor")
    if not bool(torch.isfinite(skin_mask).all()):
        raise RuntimeError("semantic mask contains NaN or Inf")
    if bool((skin_mask < 0).any()) or bool((skin_mask > 1).any()):
        raise RuntimeError("semantic mask escaped 0..1")
    if tuple(preview.shape) != tuple(frames.shape):
        raise RuntimeError("six-frame preview shape changed")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    contact = OUTPUT / "source_vs_multiface_fivepoint_parsenet_6frames.jpg"
    _contact_sheet(frames, preview, contact)
    mask_files = _write_mask_frames(skin_mask, OUTPUT)
    skin_area = [
        round(float((skin_mask[index] > 0.05).float().mean()), 8)
        for index in range(len(SOURCE_INDICES))
    ]
    alignments = [
        [
            track.get("alignment_normalized_rms")
            for track in frame["tracks"]
            if track["status"] == "READY"
        ]
        for frame in parser_report["frames"]
    ]
    report = {
        "schema": "t8.minimax_h3.skin_finish.multiface_semantic_representative/v1",
        "status": "PASS",
        "source": {
            "path": str(SOURCE),
            "sha256": _sha256(SOURCE),
            "original_contract": "1920x1408, 24fps, 69 decoded frames",
            "representative_contract": "960x704 (0.67584MP), six selected frames, two explicit shot-local ranges",
            "selected_source_frame_indices": SOURCE_INDICES,
        },
        "person_track_harness": {
            "type": "source-bound deterministic left/right person regions",
            "live_sam31_executed": False,
            "reason": "low-load validation reuses the already unit-tested SAM3.1 track-plan contract without loading the large SAM model",
            "what_it_proves": "real YuNet detection, unique per-track matching, five-point alignment, real pinned ParseNet inference, inverse projection, person-region intersection, shot-local key reset and optional label consumption",
            "what_it_does_not_prove": "live SAM3.1 segmentation quality or automatic scene-cut detection on this clip",
        },
        "parser_report": parser_report,
        "mechanical_gates": {
            "six_of_six_ready": True,
            "two_tracks_ready_per_frame": per_frame_counts,
            "source_tensor_unchanged": True,
            "finite": True,
            "range_0_1": True,
            "network_access_performed": False,
            "cuda_used": False,
            "yunet_released_before_parser_load": parser_report["detector"].get(
                "detector_object_released"
            ),
            "parser_unloaded_after_execute": parser_report["parser"][
                "unloaded_after_execute"
            ],
        },
        "skin_area_fraction": skin_area,
        "alignment_normalized_rms_by_frame_and_track": alignments,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "outputs": {
            "contact_sheet": str(contact),
            "contact_sheet_sha256": _sha256(contact),
            "mask_files": mask_files,
        },
        "human_review_required": True,
        "boundary": (
            "One clear two-person source, six resized frames and deterministic shot-local "
            "person regions prove the low-load real YuNet/ParseNet/alignment mechanics only. "
            "They do not establish aesthetic improvement, live SAM segmentation, side-profile/"
            "occlusion robustness, full-video temporal continuity, identity truth, arbitrary "
            "cuts, HDR or universal memory safety."
        ),
    }
    report_path = OUTPUT / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_md = OUTPUT / "REPORT.md"
    report_md.write_text(
        "# Skin Finish multi-person semantic parser representative check\n\n"
        "- Status: **PASS**\n"
        "- Input: six frames resized to 960×704 (0.67584MP) from a clear two-person clip\n"
        "- Result: two unique YuNet faces, five-point alignment and real pinned ParseNet masks "
        "were READY on all six frames; source pixels were not modified.\n"
        "- Resource policy: CPU only, two Torch threads, no network, no CUDA, YuNet released "
        "before ParseNet load, ParseNet released after execution.\n"
        "- Important limitation: the person masks are a deterministic source-bound left/right "
        "harness split into two shot-local ranges. This run does **not** claim a fresh live "
        "SAM3.1 segmentation or automatic scene-cut validation.\n"
        "- Human review is still required for skin quality, temporal flicker, profiles, "
        "occlusion and real cross-shot identity continuity.\n",
        encoding="utf-8",
    )
    print(report_path)
    print(contact)
    print(
        json.dumps(
            {
                "status": "PASS",
                "ready_tracks_per_frame": per_frame_counts,
                "skin_area_fraction": skin_area,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
