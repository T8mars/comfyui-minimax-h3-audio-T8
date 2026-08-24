#!/usr/bin/env python3
"""Build the final two-pair 0.6MP human-face blind review page."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import build_external_bridge_blind_review as blind_builder
import run_human_face_0p6mp_clipproj_probe as high
import run_human_face_5s_creator_av_probe as creator_contract


SCHEMA = "t8.minimax_h3.human_face_0p6mp_final_review.v1"
REVIEW_ID = "human-face-0p6mp-final-20260824"
BLIND_SEED = 2608245005


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _shared_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("runtime report is missing contract")
    fields = (
        "prompt",
        "seed",
        "input_image",
        "input_image_sha256",
        "width",
        "height",
        "pixels",
        "frame_count",
        "fps",
        "steps",
        "shift_video",
        "shift_audio",
        "task_type",
    )
    missing = [field for field in fields if field not in contract]
    if missing:
        raise ValueError(f"runtime contract is missing fields: {missing}")
    return {field: contract[field] for field in fields}


def validate_reports(
    creator: Mapping[str, Any],
    clipproj_4b: Mapping[str, Any],
    clipproj_8b: Mapping[str, Any],
) -> dict[str, Any]:
    reports = (creator, clipproj_4b, clipproj_8b)
    if any(report.get("status") != "PASS" or report.get("passed") is not True for report in reports):
        raise ValueError("all three 0.6MP runtime reports must be PASS")
    contracts = [_shared_contract(report) for report in reports]
    if contracts[1:] != contracts[:-1]:
        raise ValueError("0.6MP review contracts differ")
    contract = contracts[0]
    expected = high._contract()
    if any(contract[field] != expected[field] for field in contract):
        raise ValueError("0.6MP generation contract changed")
    if any(
        int(report.get("gpu_monitor", {}).get("minimum_free_mib", -1))
        < high.MIN_OBSERVED_HEADROOM_MIB
        for report in reports
    ):
        raise ValueError("one runtime report failed the observed 512 MiB headroom gate")

    media = creator.get("media", {})
    if not (
        media.get("candidate_checks", {}).get("strict_decode")
        and media.get("control_checks", {}).get("strict_decode")
        and int(media.get("combined_frames", 0)) == creator_contract.OUTPUT_FRAMES
        and int(media.get("separate_frames", 0)) == creator_contract.OUTPUT_FRAMES
        and int(media.get("combined_audio_samples", 0))
        == creator_contract.OUTPUT_AUDIO_SAMPLES
        and int(media.get("separate_audio_samples", 0))
        == creator_contract.OUTPUT_AUDIO_SAMPLES
    ):
        raise ValueError("Creator 0.6MP media contract is incomplete")
    for name, report in (("4B", clipproj_4b), ("8B", clipproj_8b)):
        if not report.get("media", {}).get("strict_decode_passed"):
            raise ValueError(f"ClipProj {name} strict decode is not PASS")
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
        "page_title": "MiniMax H3 0.6MP近景真人双组盲评",
        "page_intro": (
            "本轮按反馈提升为1088×544（591,872像素，约0.592MP），同一参考图、提示词、seed、"
            "124帧与8步。第一组10.125秒，请重点看/听5.17秒接缝；第二组5.167秒，请比较人脸、"
            "口型、动作和中文音轨。两组分别盲选，完成后导出一个JSON。"
        ),
        "export_filename": "human_face_0p6mp_final_blind_review.json",
        "analysis_generalization": (
            "Two fixed SHA-locked 1088x544 human-face I2VA comparisons. One portrait, prompt and "
            "seed cannot establish universal quality, audio noninferiority or general 16GB safety."
        ),
        "pairs": [
            {
                "pair_id": "creator-human-face-0p6mp",
                "label": "Creator 10.125秒 0.6MP接缝",
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
                "pair_id": "clipproj-human-face-0p6mp",
                "label": "ClipProj 5.167秒 0.6MP 4B/8B",
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
        default=repo / "artifacts" / "human-face-0p6mp-final-review-20260824",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = args.reference_image.resolve()
    if _sha256(reference) != high.legacy.INPUT_IMAGE_SHA256:
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
        "review_page": str((args.output_dir / "review" / "blind_review.html").resolve()),
        "private_key": str(key_path.resolve()),
        "private_key_sha256": _sha256(key_path),
        "key_pair_count": len(key["pairs"]),
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
