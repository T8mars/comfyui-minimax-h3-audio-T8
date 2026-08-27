#!/usr/bin/env python3
"""Build the fixed four-arm Audio Refine mechanical comparison without model work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import run_audio_refine_phase2 as phase2


SCHEMA = "t8.minimax_h3.audio_refine.phase2.four_arm_mechanical.v1"


def _latest_report(
    root: Path, predicate: Callable[[Mapping[str, Any]], bool]
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in root.rglob("validation_report.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and predicate(value):
            candidates.append((path.stat().st_mtime, path, value))
    if not candidates:
        raise FileNotFoundError(f"no matching PASS validation report below {root}")
    _mtime, path, value = max(candidates, key=lambda item: item[0])
    return path, value


def _row(
    *,
    name: str,
    route: str,
    first_nfe: int,
    refine_nfe: int,
    audio_denoise: float | None,
    report_path: Path,
    media: Mapping[str, Any],
    pcm: Mapping[str, Any],
    minimum_free_vram_mib: int | None,
    human_note: str,
) -> dict[str, Any]:
    probe = media.get("probe") or {}
    streams = probe.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    return {
        "arm": name,
        "route": route,
        "first_pass_nfe": first_nfe,
        "refine_nfe": refine_nfe,
        "total_nfe": first_nfe + refine_nfe,
        "audio_denoise": audio_denoise,
        "report_path": str(report_path),
        "media_path": str(media.get("path") or ""),
        "strict_decode_passed": bool(media.get("strict_decode_passed")),
        "width": video.get("width"),
        "height": video.get("height"),
        "frames": int(video.get("nb_frames") or 0),
        "fps": video.get("avg_frame_rate"),
        "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": audio.get("channels"),
        "decoded_video_sha256": (media.get("decoded_video") or {}).get("sha256"),
        "decoded_audio_sha256": (media.get("decoded_audio") or {}).get("sha256"),
        "left_rms": pcm.get("left_rms"),
        "right_rms": pcm.get("right_rms"),
        "peak_absolute": pcm.get("peak_absolute"),
        "clipping_suspected": pcm.get("clipping_suspected"),
        "pcm_contract_passed": pcm.get("passed"),
        "minimum_free_vram_mib": minimum_free_vram_mib,
        "human_note": human_note,
    }


def build_report(*, artifact_root: Path, ffmpeg: str) -> dict[str, Any]:
    quality_path, quality = _latest_report(
        artifact_root / "audio-refine-quality-pair-20260826",
        lambda report: report.get("status") == "PASS",
    )
    ordinary_path, ordinary = _latest_report(
        artifact_root / "audio-refine-phase2-20260826",
        lambda report: report.get("status") == "PASS"
        and report.get("arm") == "base_ordinary8",
    )
    base_refine_path, base_refine = _latest_report(
        artifact_root / "audio-refine-phase2-20260826",
        lambda report: report.get("status") == "PASS"
        and report.get("arm") == "base_refine4",
    )

    turbo_original_media = quality["original_media"]
    same_turbo_media = quality["refined_media"]
    turbo_original_pcm = phase2.decoded_pcm_contract(
        Path(turbo_original_media["path"]), ffmpeg=ffmpeg
    )
    same_turbo_pcm = phase2.decoded_pcm_contract(
        Path(same_turbo_media["path"]), ffmpeg=ffmpeg
    )
    quality_minimum = (quality.get("gpu") or {}).get("monitor", {}).get(
        "minimum_free_mib"
    )
    ordinary_media = ordinary["media"]["ordinary8"]
    ordinary_pcm = ordinary["decoded_pcm_contracts"]["ordinary8"]
    ordinary_minimum = (ordinary.get("gpu") or {}).get("monitor", {}).get(
        "minimum_free_mib"
    )
    base_media = base_refine["media"]["candidate"]
    base_pcm = base_refine["decoded_pcm_contracts"]["candidate"]
    base_minimum = (base_refine.get("gpu") or {}).get("generation", {}).get(
        "minimum_free_mib"
    )

    reviewed_note = (
        "one reviewer: same-Turbo Refine4 was slightly preferred; original was quieter; "
        "manual gate remains required"
    )
    rows = [
        _row(
            name="turbo4_original",
            route="Turbo4 original",
            first_nfe=4,
            refine_nfe=0,
            audio_denoise=None,
            report_path=quality_path,
            media=turbo_original_media,
            pcm=turbo_original_pcm,
            minimum_free_vram_mib=quality_minimum,
            human_note=reviewed_note,
        ),
        _row(
            name="base_ordinary8",
            route="base without Turbo, ordinary sampling",
            first_nfe=8,
            refine_nfe=0,
            audio_denoise=None,
            report_path=ordinary_path,
            media=ordinary_media,
            pcm=ordinary_pcm,
            minimum_free_vram_mib=ordinary_minimum,
            human_note="not yet compared by the user",
        ),
        _row(
            name="turbo4_same_turbo_refine4",
            route="Turbo4 then the same Turbo stack",
            first_nfe=4,
            refine_nfe=4,
            audio_denoise=0.5,
            report_path=quality_path,
            media=same_turbo_media,
            pcm=same_turbo_pcm,
            minimum_free_vram_mib=quality_minimum,
            human_note=reviewed_note,
        ),
        _row(
            name="turbo4_base_refine4",
            route="Turbo4 then base without Turbo",
            first_nfe=4,
            refine_nfe=4,
            audio_denoise=0.5,
            report_path=base_refine_path,
            media=base_media,
            pcm=base_pcm,
            minimum_free_vram_mib=base_minimum,
            human_note=(
                "candidate is mechanically eligible but about 12 dB louder than its original; "
                "quality gate ABSTAINED and retained the original"
            ),
        ),
    ]
    comparable = all(
        row["strict_decode_passed"]
        and row["width"] == 1056
        and row["height"] == 608
        and row["frames"] == 124
        and row["fps"] == "24/1"
        and row["audio_sample_rate"] == 32000
        and row["audio_channels"] == 2
        and row["pcm_contract_passed"]
        for row in rows
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixed_contract": {
            "prompt": phase2.PROMPT,
            "seed": 2608260404,
            "width": 1056,
            "height": 608,
            "frames": 124,
            "fps": 24.0,
        },
        "four_arm_mechanical_contract_passed": comparable,
        "rows": rows,
        "decision": (
            "MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED"
            if comparable
            else "FAIL_MECHANICAL_CONTRACT"
        ),
        "scientific_boundary": (
            "Equal total NFE is not equal training distribution. Signal metrics and one prior "
            "single-reviewer result do not select a universal winner."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MiniMax H3 Audio Refine 四臂机械汇总",
        "",
        f"结论：`{report['decision']}`。四条均为 1056×608、124帧、24fps、32kHz双声道。",
        "",
        "| 路线 | NFE | denoise | RMS(L) | 峰值 | 最低空闲显存 | 人工/质量门 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        denoise = "-" if row["audio_denoise"] is None else row["audio_denoise"]
        lines.append(
            "| {arm} | {nfe} | {denoise} | {rms:.6f} | {peak:.6f} | {vram} MiB | {note} |".format(
                arm=row["arm"],
                nfe=row["total_nfe"],
                denoise=denoise,
                rms=float(row["left_rms"]),
                peak=float(row["peak_absolute"]),
                vram=row["minimum_free_vram_mib"],
                note=row["human_note"],
            )
        )
    lines.extend(
        [
            "",
            "注意：总 NFE 相同不代表训练分布相同；机械通过和信号指标不能替代听感。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=project_root / "artifacts")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "artifacts" / "audio-refine-four-arm-20260827",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    report = build_report(artifact_root=args.artifact_root, ffmpeg=args.ffmpeg)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "four_arm_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "README.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "output": str(args.output_root.resolve())}, ensure_ascii=False))
    return 0 if report["four_arm_mechanical_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
