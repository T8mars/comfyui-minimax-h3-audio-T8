#!/usr/bin/env python3
"""Build one controlled 0.6MP ClipProj-4B versus native-32B blind review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import build_external_bridge_blind_review as blind_builder
import run_human_face_0p6mp_clipproj_probe as high


SCHEMA = "t8.minimax_h3.human_face_0p6mp_4b_vs_native_review.v1"
REVIEW_ID = "human-face-0p6mp-4b-vs-native-20260824"
BLIND_SEED = 2608245006
MIN_NORMALIZATION_SSIM = 0.98
SSIM_RE = re.compile(r"All:([0-9.]+)")


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


def _contract(report: Mapping[str, Any]) -> dict[str, Any]:
    value = report.get("contract")
    if not isinstance(value, Mapping):
        raise ValueError("runtime report is missing contract")
    return dict(value)


def _ssim(source: Path, normalized: Path, *, ffmpeg: str) -> float:
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "info",
            "-i",
            str(source),
            "-i",
            str(normalized),
            "-lavfi",
            "ssim",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    matches = SSIM_RE.findall(completed.stderr)
    if completed.returncode != 0 or not matches:
        raise ValueError("unable to measure source/normalized SSIM")
    return float(matches[-1])


def validate_reports(
    clipproj_4b: Mapping[str, Any],
    native_32b: Mapping[str, Any],
    normalized_4b: Mapping[str, Any],
    normalized_native: Mapping[str, Any],
) -> dict[str, Any]:
    if clipproj_4b.get("status") != "PASS" or clipproj_4b.get("passed") is not True:
        raise ValueError("0.6MP ClipProj 4B runtime report must be PASS")
    if not clipproj_4b.get("media", {}).get("strict_decode_passed"):
        raise ValueError("0.6MP ClipProj 4B source must strictly decode")

    terminal = native_32b.get("phase", {}).get("terminal", {})
    native_checks = native_32b.get("checks", {})
    false_checks = {name for name, passed in native_checks.items() if passed is not True}
    if (
        terminal.get("type") != "execution_success"
        or native_32b.get("runtime_error") is not None
        or false_checks != {"strict_decode"}
    ):
        raise ValueError("native 32B may fail only the original H.264 strict-decode check")
    if int(native_32b.get("gpu_monitor", {}).get("minimum_free_mib", -1)) < 512:
        raise ValueError("native 32B failed the observed 512 MiB headroom gate")

    contract = _contract(clipproj_4b)
    if _contract(native_32b) != contract or contract != high._contract():
        raise ValueError("0.6MP generation contracts differ")
    for name, media in (("4B", normalized_4b), ("native 32B", normalized_native)):
        checks = high._media_checks(media)
        if not all(checks.values()):
            raise ValueError(f"normalized {name} media contract is incomplete: {checks}")
    source_audio = (
        clipproj_4b["media"]["decoded_audio"]["sha256"],
        native_32b["media"]["decoded_audio"]["sha256"],
    )
    normalized_audio = (
        normalized_4b["decoded_audio"]["sha256"],
        normalized_native["decoded_audio"]["sha256"],
    )
    if normalized_audio != source_audio:
        raise ValueError("matched re-encode changed one or both decoded audio tracks")
    return contract


def build_manifest(
    contract: Mapping[str, Any],
    *,
    normalized_4b: Path,
    normalized_native: Path,
    reference_image: Path,
) -> dict[str, Any]:
    return {
        "schema": blind_builder.MANIFEST_SCHEMA,
        "review_id": REVIEW_ID,
        "page_title": "MiniMax H3 0.6MP真人近景：4B与原生32B盲评",
        "page_intro": (
            "只评这一组：同一参考图、提示词、seed、1088×544、124帧、8步。两边均用相同的"
            "单线程x264参数做匹配重编码，音频保持原AAC码流；请比较人脸、口型、动作、中文音轨"
            "和提示词遵循，然后导出JSON。"
        ),
        "export_filename": "human_face_0p6mp_4b_vs_native_blind_review.json",
        "analysis_generalization": (
            "One fixed SHA-locked 1088x544 I2VA portrait comparison. A tie or preference cannot "
            "establish universal encoder equivalence, superiority or general 16GB safety."
        ),
        "pairs": [
            {
                "pair_id": "clipproj-4b-vs-native-32b-0p6mp",
                "label": "0.6MP 5.167秒：紧凑编码器与原生编码器",
                "task_type": "I2VA AV Review",
                "prompt": contract["prompt"],
                "control": str(normalized_native.resolve()),
                "candidate": str(normalized_4b.resolve()),
                "control_method": "Native MiniMax H3 32B text encoder",
                "candidate_method": "ClipProj 4B text encoder",
                "reference_images": [str(reference_image.resolve())],
                "reference_metrics": ["first_frame", "identity"],
            }
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clipproj-4b-report", type=Path, required=True)
    parser.add_argument("--native-32b-report", type=Path, required=True)
    parser.add_argument("--normalized-4b", type=Path, required=True)
    parser.add_argument("--normalized-native", type=Path, required=True)
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI\input\10A.jpg"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "artifacts" / "human-face-0p6mp-4b-vs-native-20260824",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip4 = _load_json(args.clipproj_4b_report.resolve())
    native = _load_json(args.native_32b_report.resolve())
    normalized_4b = args.normalized_4b.resolve()
    normalized_native = args.normalized_native.resolve()
    reference = args.reference_image.resolve()
    if _sha256(reference) != high.legacy.INPUT_IMAGE_SHA256:
        raise ValueError("reference-image SHA changed")
    media4 = high.legacy.base.shared.media_report(
        normalized_4b, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
    )
    media_native = high.legacy.base.shared.media_report(
        normalized_native, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
    )
    contract = validate_reports(clip4, native, media4, media_native)
    ssim4 = _ssim(Path(clip4["media"]["path"]), normalized_4b, ffmpeg=args.ffmpeg)
    ssim_native = _ssim(
        Path(native["media"]["path"]), normalized_native, ffmpeg=args.ffmpeg
    )
    if min(ssim4, ssim_native) < MIN_NORMALIZATION_SSIM:
        raise ValueError("matched re-encode SSIM fell below the reviewed 0.98 floor")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        "schema": f"{SCHEMA}.normalization",
        "status": "PASS",
        "encoder": "libx264 preset=medium crf=18 threads=1 pix_fmt=yuv420p; audio=copy",
        "minimum_ssim": MIN_NORMALIZATION_SSIM,
        "arms": {
            "clipproj_4b": {
                "source": clip4["media"]["path"],
                "source_sha256": clip4["media"]["file_sha256"],
                "normalized": str(normalized_4b),
                "normalized_sha256": _sha256(normalized_4b),
                "ssim_all": ssim4,
                "strict_decode": media4["strict_decode_passed"],
                "decoded_audio_sha256": media4["decoded_audio"]["sha256"],
            },
            "native_32b": {
                "source": native["media"]["path"],
                "source_sha256": native["media"]["file_sha256"],
                "source_strict_decode": native["media"]["strict_decode_passed"],
                "source_strict_diagnostic": native["media"]["strict_decode"]["video"][
                    "diagnostic"
                ],
                "normalized": str(normalized_native),
                "normalized_sha256": _sha256(normalized_native),
                "ssim_all": ssim_native,
                "strict_decode": media_native["strict_decode_passed"],
                "decoded_audio_sha256": media_native["decoded_audio"]["sha256"],
            },
        },
        "boundary": (
            "The original native-32B MP4 is preserved and had one H.264 packet error while all "
            "124 frames decoded. Both arms were then identically re-encoded to prevent an encode-"
            "path confound. This repairs transport only and does not recover unknown pixels."
        ),
    }
    (args.output_dir / "normalization_report.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        contract,
        normalized_4b=normalized_4b,
        normalized_native=normalized_native,
        reference_image=reference,
    )
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
        "pair_count": 1,
        "review_page": str((args.output_dir / "review" / "blind_review.html").resolve()),
        "private_key": str(key_path.resolve()),
        "private_key_sha256": _sha256(key_path),
        "normalization_report": str((args.output_dir / "normalization_report.json").resolve()),
        "key_pair_count": len(key["pairs"]),
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
