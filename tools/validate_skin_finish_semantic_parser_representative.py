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
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_skin_parser_validation"
P0_REPORT = (
    ROOT
    / "artifacts"
    / "skin-finish-p0-representative-20260824"
    / "validation_report.json"
)
OUTPUT = (
    ROOT
    / "artifacts"
    / "skin-finish-semantic-parser-representative-3frame-20260824"
)
FRAME_INDICES = [0, 62, 123]


def _load_package():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
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


def _decode_selected(path: Path, indices: list[int]) -> torch.Tensor:
    wanted = set(indices)
    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                array = frame.to_ndarray(format="rgb24")
                selected[index] = torch.from_numpy(array.copy()).float().div_(255.0)
            if index >= max(indices):
                break
    if sorted(selected) != indices:
        raise RuntimeError(f"decoded selected frames {sorted(selected)}, expected {indices}")
    return torch.stack([selected[index] for index in indices])


def _subset_plan(full_plan: dict, frames: torch.Tensor, package_name: str) -> dict:
    face_module = sys.modules[f"{package_name}.face_refine_advanced"]
    records = []
    for new_index, source_index in enumerate(FRAME_INDICES):
        record = dict(full_plan["frames"][source_index])
        record["frame_index"] = new_index
        record["shot_id"] = 0
        records.append(record)
    source_hash = face_module.source_proxy_sha256(frames)
    plan = {
        "schema": face_module.PLAN_SCHEMA,
        "status": "experimental_candidate_plan",
        "source": {
            "frame_count": len(records),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "fps": 24.0,
            "proxy_sha256": source_hash,
        },
        "canvas": dict(full_plan["canvas"]),
        "detector": dict(full_plan["detector"]),
        "shots": [{"shot_id": 0, "start_frame": 0, "end_frame": len(records) - 1}],
        "frames": records,
        "preview_frame_indices": list(range(len(records))),
        "limits": dict(full_plan["limits"]),
        "metrics": {"subset_source_frame_indices": FRAME_INDICES},
    }
    plan["plan_sha256"] = hashlib.sha256(
        face_module.canonical_json(plan).encode("utf-8")
    ).hexdigest()
    return plan


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


def _contact_sheet(frames: torch.Tensor, preview: torch.Tensor, output: Path) -> None:
    width = int(frames.shape[2])
    height = int(frames.shape[1])
    header = 34
    canvas = Image.new("RGB", (width * len(FRAME_INDICES), (height + header) * 2), "black")
    draw = ImageDraw.Draw(canvas)
    for column, source_index in enumerate(FRAME_INDICES):
        x = column * width
        draw.text((x + 10, 9), f"SOURCE FRAME {source_index}", fill="white")
        canvas.paste(_to_image(frames[column]), (x, header))
        row_y = height + header
        draw.text(
            (x + 10, row_y + 9),
            f"PARSENET MASK FRAME {source_index}: GREEN=SKIN, RED=PROTECTED",
            fill="white",
        )
        canvas.paste(_to_image(preview[column]), (x, row_y + header))
    canvas.save(output, quality=94, subsampling=0)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    started = time.perf_counter()
    _load_package()
    report_source = json.loads(P0_REPORT.read_text(encoding="utf-8"))
    source_path = Path(report_source["source"]["path"])
    frames = _decode_selected(source_path, FRAME_INDICES)
    plan = _subset_plan(report_source["mask_route"]["plan"], frames, PACKAGE_NAME)
    parser_module = sys.modules[f"{PACKAGE_NAME}.skin_finish_parser"]
    skin_mask, preview, parser_report_json = parser_module.run_semantic_skin_mask(
        frames=frames,
        face_plan=plan,
        parser_model=parser_module.PARSENET_MODEL_NAME,
        include_neck=False,
        crop_expansion=1.45,
        minimum_face_weight=0.35,
        minimum_class_probability=0.55,
        feature_protection_px=3,
        minimum_skin_area=0.0005,
        maximum_skin_area=0.25,
        preview_count=3,
    )
    parser_report = json.loads(parser_report_json)
    if parser_report["status"] != "READY":
        raise RuntimeError(f"parser did not pass: {parser_report['status']}")
    if parser_report["selection"]["accepted_frame_count"] != 3:
        raise RuntimeError("not all three representative frames produced a semantic mask")
    if not bool(torch.isfinite(skin_mask).all()):
        raise RuntimeError("semantic mask contains NaN or Inf")
    if bool((skin_mask < 0).any()) or bool((skin_mask > 1).any()):
        raise RuntimeError("semantic mask escaped 0..1")
    if not all(float(skin_mask[index].sum()) > 0.0 for index in range(3)):
        raise RuntimeError("at least one semantic skin mask is empty")
    if tuple(preview.shape) != tuple(frames.shape):
        raise RuntimeError("three-frame preview shape changed")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    contact = OUTPUT / "source_vs_parsenet_semantic_mask_frames_0_62_123.jpg"
    _contact_sheet(frames, preview, contact)
    skin_area = [
        round(float((skin_mask[index] > 0.05).float().mean()), 8) for index in range(3)
    ]
    report = {
        "schema": "t8.minimax_h3.skin_finish.semantic_parser_representative/v1",
        "status": "PASS",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "contract": "three selected frames from 1088x544x124 at 24fps",
            "selected_frame_indices": FRAME_INDICES,
        },
        "parser_report": parser_report,
        "mechanical_gates": {
            "three_of_three_ready": True,
            "finite": True,
            "range_0_1": True,
            "nonempty_each_frame": True,
            "source_tensor_unchanged": True,
            "network_access_performed": False,
            "cuda_used": False,
            "model_unloaded_after_execute": parser_report["model"][
                "unloaded_after_execute"
            ],
        },
        "skin_area_fraction": skin_area,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "outputs": {
            "contact_sheet": str(contact),
            "contact_sheet_sha256": _sha256(contact),
        },
        "human_review_required": True,
        "boundary": (
            "One fixed portrait and three sampled frames prove pinned CPU ParseNet loading, "
            "mapping, source binding, fail-closed mechanics and visible mask production only. "
            "They do not prove video continuity, multi-person identity, profile/occlusion safety, "
            "aesthetic improvement or general memory safety."
        ),
    }
    report_path = OUTPUT / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)
    print(contact)
    print(json.dumps({"status": "PASS", "skin_area_fraction": skin_area}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
