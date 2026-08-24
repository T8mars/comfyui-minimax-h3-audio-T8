#!/usr/bin/env python3
"""Build one two-pair final blind page for the long human-face replacement media.

The page combines the 10.125-second Creator decode/composition pair and the 5.167-second ClipProj
4B/8B pair. It verifies the SHA-locked shared I2VA contract and the existing runtime PASS reports,
then delegates media hashing, per-pair contract checks and private A/B mapping to the immutable blind
review builder. No model is loaded and no existing review package is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import build_external_bridge_blind_review as blind_builder


SCHEMA = "t8.minimax_h3.human_face_replacement_final_review.v1"
REVIEW_ID = "human-face-replacement-final-20260824"
BLIND_SEED = 2608245004
EXPECTED_IMAGE_SHA256 = (
    "34E67512265DA29076075030B62BA93EC304210A09171FF68E1F44894D15A36C"
)
COMMON_CONTRACT_FIELDS = (
    "prompt",
    "seed",
    "input_image",
    "input_image_sha256",
    "width",
    "height",
    "frame_count",
    "fps",
    "steps",
    "shift_video",
    "shift_audio",
    "task_type",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _common_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("runtime report is missing a contract object")
    missing = [field for field in COMMON_CONTRACT_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"runtime contract is missing fields: {missing}")
    return {field: contract[field] for field in COMMON_CONTRACT_FIELDS}


def validate_reports(
    creator: Mapping[str, Any],
    clipproj_4b: Mapping[str, Any],
    clipproj_8b: Mapping[str, Any],
) -> dict[str, Any]:
    reports = {
        "creator": creator,
        "clipproj_4b": clipproj_4b,
        "clipproj_8b": clipproj_8b,
    }
    for name, report in reports.items():
        if report.get("status") != "PASS" or report.get("passed") is not True:
            raise ValueError(f"{name} runtime report is not PASS")
    common = {name: _common_contract(report) for name, report in reports.items()}
    if common["creator"] != common["clipproj_4b"] or common["creator"] != common[
        "clipproj_8b"
    ]:
        raise ValueError("Creator and ClipProj replacement contracts differ")
    contract = common["creator"]
    if contract["input_image_sha256"] != EXPECTED_IMAGE_SHA256:
        raise ValueError("replacement input-image SHA changed")
    if (
        int(contract["width"]),
        int(contract["height"]),
        int(contract["frame_count"]),
        int(contract["fps"]),
        int(contract["steps"]),
        float(contract["shift_video"]),
        float(contract["shift_audio"]),
        str(contract["task_type"]),
    ) != (512, 256, 124, 24, 8, 12.0, 3.0, "I2VA"):
        raise ValueError("replacement generation geometry or sampler contract changed")

    creator_media = creator.get("media", {})
    if not (
        creator_media.get("candidate_checks", {}).get("strict_decode")
        and creator_media.get("control_checks", {}).get("strict_decode")
        and int(creator_media.get("combined_frames", 0)) == 243
        and int(creator_media.get("separate_frames", 0)) == 243
        and int(creator_media.get("combined_audio_samples", 0)) == 324_000
        and int(creator_media.get("separate_audio_samples", 0)) == 324_000
    ):
        raise ValueError("Creator long-media contract is incomplete")
    for name, report in (
        ("clipproj_4b", clipproj_4b),
        ("clipproj_8b", clipproj_8b),
    ):
        if not report.get("media", {}).get("strict_decode_passed"):
            raise ValueError(f"{name} strict media contract is not PASS")
    return contract


def build_manifest(
    creator: Mapping[str, Any],
    clipproj_4b: Mapping[str, Any],
    clipproj_8b: Mapping[str, Any],
    *,
    reference_image: Path,
) -> dict[str, Any]:
    contract = validate_reports(creator, clipproj_4b, clipproj_8b)
    creator_media = creator["media"]
    return {
        "schema": blind_builder.MANIFEST_SCHEMA,
        "review_id": REVIEW_ID,
        "page_title": "MiniMax H3 最终近景真人双组盲评",
        "page_intro": (
            "两组均按反馈改为清晰近景真人、8步和至少5秒。第一组10.125秒，请重点看/听5.17秒接缝；"
            "第二组5.167秒，请完整比较人脸、口型、动作与中文音轨。每组先选择是否可判断，"
            "两组全部完成后只需导出一个JSON。"
        ),
        "export_filename": "human_face_replacement_final_blind_review.json",
        "analysis_generalization": (
            "Two fixed SHA-locked human-face I2VA comparisons: one reused-latent Creator decode/"
            "composition pair and one ClipProj 4B/8B pair. One seed and one portrait cannot establish "
            "universal quality, audio noninferiority, spoken-text accuracy or 16GiB safety. Native "
            "32B is excluded until its evidence-derived resource gate passes."
        ),
        "pairs": [
            {
                "pair_id": "creator-human-face-long-final",
                "label": "Creator 10.125秒近景真人接缝",
                "task_type": "I2VA AV Review",
                "prompt": contract["prompt"],
                "control": str(Path(creator_media["control_path"]).resolve()),
                "candidate": str(Path(creator_media["candidate_path"]).resolve()),
                "control_method": "Separate VAE decode then media composition",
                "candidate_method": "Native latent concat then one VAE decode",
                "reference_images": [str(reference_image.resolve())],
                "reference_metrics": ["first_frame", "identity"],
            },
            {
                "pair_id": "clipproj-human-face-4b-vs-8b-final",
                "label": "ClipProj 5.167秒近景真人 4B/8B",
                "task_type": "I2VA AV Review",
                "prompt": contract["prompt"],
                "control": str(Path(clipproj_4b["media"]["path"]).resolve()),
                "candidate": str(Path(clipproj_8b["media"]["path"]).resolve()),
                "control_method": "ClipProj 4B",
                "candidate_method": "ClipProj 8B",
                "reference_images": [str(reference_image.resolve())],
                "reference_metrics": ["first_frame", "identity"],
            },
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creator-report", type=Path, required=True)
    parser.add_argument("--clipproj-4b-report", type=Path, required=True)
    parser.add_argument("--clipproj-8b-report", type=Path, required=True)
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI\input\10A.jpg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "artifacts" / "human-face-replacement-final-review-20260824",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = args.reference_image.resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if _sha256_file(reference) != EXPECTED_IMAGE_SHA256:
        raise ValueError("reference-image SHA changed")
    manifest = build_manifest(
        _load_json(args.creator_report),
        _load_json(args.clipproj_4b_report),
        _load_json(args.clipproj_8b_report),
        reference_image=reference,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    key = blind_builder.build_package(
        manifest, args.output_dir, args.output_dir / "review", blind_seed=BLIND_SEED
    )
    key_path = args.output_dir / "review" / "blind_key.json"
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "review_id": REVIEW_ID,
        "pair_count": len(manifest["pairs"]),
        "manifest": str(manifest_path.resolve()),
        "review_page": str((args.output_dir / "review" / "blind_review.html").resolve()),
        "private_key": str(key_path.resolve()),
        "private_key_sha256": _sha256_file(key_path),
        "key_pair_count": len(key["pairs"]),
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
