#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import uuid
from typing import Any, Iterable

import av
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
PACKAGE = "h3_audio_t8_pkg"
REPORT_SCHEMA = "t8.dlss_nr.p4_candidate_generation.v1"
MANIFEST_SCHEMA = "t8.dlss_nr.blind_manifest.v1"
P3_ANALYSIS_SCHEMA = "t8.dlss_nr.p3_human_review_analysis.v1"
VALIDATION_SCHEMA = "t8.dlss_nr.real_validation.v1"
CLIP_TYPES = {
    "speech": "speech",
    "hard_cut": "hard_cut",
    "fine_texture": "fine_texture",
}
LABELS = {
    "speech": "124-frame H3 character speech source",
    "hard_cut": "Hard-cut source",
    "fine_texture": "Subtitle and fine-texture source",
}


def _load_project_module(name: str):
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _run(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {stderr[-4000:]}")
    return completed


def _decode_frames(path: Path) -> tuple[torch.Tensor, Fraction]:
    decoded: list[torch.Tensor] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate or stream.base_rate)
        for frame in container.decode(stream):
            array = frame.to_ndarray(format="rgb24")
            decoded.append(torch.from_numpy(array.copy()))
    if not decoded:
        raise RuntimeError(f"no video frames decoded from {path}")
    frames = torch.stack(decoded).to(torch.float32).div_(255.0)
    return frames.contiguous(), rate


def _atomic_frame_encode(
    frames: Iterable[torch.Tensor],
    *,
    count: int,
    width: int,
    height: int,
    rate: Fraction,
    target: Path,
    ffmpeg: str,
    lanczos_2x: bool = False,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.mp4")
    input_width = width // 2 if lanczos_2x else width
    input_height = height // 2 if lanczos_2x else height
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{input_width}x{input_height}",
        "-r",
        f"{rate.numerator}/{rate.denominator}",
        "-i",
        "pipe:0",
        "-an",
    ]
    if lanczos_2x:
        command += ["-vf", f"scale={width}:{height}:flags=lanczos"]
    command += [
        "-frames:v",
        str(count),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "8",
        "-x264-params",
        "ref=1:bframes=0:threads=1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdin is not None
    assert process.stderr is not None
    written = 0
    try:
        for frame in frames:
            array = (
                frame.detach()
                .float()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )
            expected = (input_height, input_width, 3)
            if array.shape != expected:
                raise ValueError(f"frame {written} has shape {array.shape}, expected {expected}")
            process.stdin.write(np.ascontiguousarray(array).tobytes())
            written += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f"FFmpeg frame encode failed ({returncode}): {stderr[-4000:]}")
        if written != count:
            raise RuntimeError(f"encoded {written} frames, expected {count}")
        os.replace(temporary, target)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
        temporary.unlink(missing_ok=True)


def _atomic_lanczos(source: Path, target: Path, ffmpeg: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.mp4")
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "scale=iw*2:ih*2:flags=lanczos",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "8",
            "-x264-params",
            "ref=1:bframes=0:threads=1",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    os.replace(temporary, target)


def _strict_decode(path: Path, ffmpeg: str) -> None:
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )


def _contract(path: Path) -> dict[str, Any]:
    dlss = _load_project_module("dlss_nr_advanced")
    source_video = _FileBackedVideo(path)
    resolved, value = dlss._file_source_contract(source_video)
    if resolved != path.resolve():
        raise RuntimeError(f"media contract resolved a different path for {path}")
    return value


class _FileBackedVideo:
    def __init__(self, path: Path):
        self.path = path.resolve()
        with av.open(str(self.path), mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError("P4 media must contain exactly one video stream")
            stream = container.streams.video[0]
            if stream.average_rate is None:
                raise ValueError("P4 media has no declared frame rate")
            self.width = int(stream.width)
            self.height = int(stream.height)
            self.rate = float(stream.average_rate)
            self.frame_count = sum(1 for _frame in container.decode(stream))

    def get_stream_source(self):
        return str(self.path)

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_frame_count(self):
        return self.frame_count

    def get_dimensions(self):
        return self.width, self.height

    def get_bit_depth(self):
        return 8

    def get_frame_rate(self):
        return self.rate


def _contract_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": int(value["width"]),
        "height": int(value["height"]),
        "frame_count": int(value["frame_count"]),
        "fps": float(Fraction(value["rate"])),
    }


def _validate_2x(source: Path, candidate: Path, *, ffmpeg: str) -> dict[str, Any]:
    _strict_decode(candidate, ffmpeg)
    source_contract = _contract(source)
    candidate_contract = _contract(candidate)
    expected = (int(source_contract["width"]) * 2, int(source_contract["height"]) * 2)
    actual = (int(candidate_contract["width"]), int(candidate_contract["height"]))
    if actual != expected:
        raise RuntimeError(f"{candidate} is {actual}, expected exact 2x {expected}")
    if int(candidate_contract["frame_count"]) != int(source_contract["frame_count"]):
        raise RuntimeError(f"{candidate} frame count differs from source")
    if Fraction(candidate_contract["rate"]) != Fraction(source_contract["rate"]):
        raise RuntimeError(f"{candidate} frame rate differs from source")
    return _contract_summary(candidate_contract)


def _is_valid_2x(source: Path, candidate: Path, *, ffmpeg: str) -> bool:
    if not candidate.is_file():
        return False
    try:
        _validate_2x(source, candidate, ffmpeg=ffmpeg)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _validated_inputs(validation: dict[str, Any], p3_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    if validation.get("schema") != VALIDATION_SCHEMA:
        raise ValueError(f"validation schema must be {VALIDATION_SCHEMA}")
    if p3_analysis.get("schema") != P3_ANALYSIS_SCHEMA:
        raise ValueError(f"P3 analysis schema must be {P3_ANALYSIS_SCHEMA}")
    decision = p3_analysis.get("decision", {})
    if decision.get("p3_fixed_material_gate") != "PASS" or decision.get(
        "eligible_to_build_p4_comparison"
    ) is not True:
        raise ValueError("P3 human gate has not passed")
    runs = validation.get("p3", {}).get("runs", [])
    analysis = {row["clip_id"]: row for row in p3_analysis.get("clips", [])}
    result = []
    for run in runs:
        clip_id = run.get("clip_id")
        if clip_id not in CLIP_TYPES or clip_id not in analysis:
            raise ValueError("P3 validation/analysis clips must be speech, hard_cut and fine_texture")
        if run.get("mechanical_pass") is not True or analysis[clip_id].get(
            "fixed_clip_pass"
        ) is not True:
            raise ValueError(f"P3 clip {clip_id} has not passed both gates")
        source = Path(run["source"]["path"]).resolve()
        candidate = Path(run["candidate"]["path"]).resolve()
        source_hashes = {
            _sha256(source),
            run["source"]["sha256"],
            analysis[clip_id]["source_sha256"],
        }
        if len(source_hashes) != 1:
            raise ValueError(f"P3 source hash differs for {clip_id}")
        candidate_hashes = {
            _sha256(candidate),
            run["candidate"]["sha256"],
            analysis[clip_id]["candidate_sha256"],
        }
        if len(candidate_hashes) != 1:
            raise ValueError(f"P3 DLSS candidate hash differs for {clip_id}")
        result.append({"clip_id": clip_id, "source": source, "dlss_nr": candidate})
    if {row["clip_id"] for row in result} != set(CLIP_TYPES):
        raise ValueError("P3 validation is missing a required clip type")
    return result


def _bind_method(source_sha: str, path: Path, profile: str, base: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(base).as_posix(),
        "source_sha256": source_sha,
        "candidate_sha256": _sha256(path),
        "profile": profile,
    }


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serially prepare the hash-bound four-method DLSS-NR P4 candidates."
    )
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--p3-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--realbasicvsr-model", type=Path, required=True)
    parser.add_argument("--flashvsr-model-dir", type=Path, required=True)
    parser.add_argument("--flashvsr-mode", choices=("tiny", "tiny_long"), default="tiny_long")
    parser.add_argument(
        "--flashvsr-spatial",
        choices=("full_frame", "adaptive_tiles"),
        default="adaptive_tiles",
    )
    parser.add_argument("--flashvsr-tile-size", type=int, default=256)
    parser.add_argument("--flashvsr-tile-overlap", type=int, default=24)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    validation_path = args.validation_report.resolve()
    p3_path = args.p3_analysis.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    p3_analysis = json.loads(p3_path.read_text(encoding="utf-8"))
    if p3_analysis.get("source_files", {}).get("validation_report_sha256", "").lower() != _sha256(
        validation_path
    ):
        raise ValueError("P3 analysis is not bound to this validation report")
    inputs = _validated_inputs(validation, p3_analysis)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "RUNNING_STRICTLY_SERIAL",
        "strictly_serial": True,
        "parallel_or_stress_execution": False,
        "source_files": {
            "validation_report": str(validation_path),
            "validation_report_sha256": _sha256(validation_path),
            "p3_analysis": str(p3_path),
            "p3_analysis_sha256": _sha256(p3_path),
        },
        "profiles": {
            "lanczos": "lanczos_2x",
            "realbasicvsr": "conservative",
            "flashvsr": "quality_locked",
            "dlss_nr": "standard",
        },
        "implementation": {
            "realbasicvsr": "native-size restoration at strength 0.30, then Lanczos 2x",
            "flashvsr": (
                f"{args.flashvsr_mode}, published fixed 2.0/3.0/11 quality budget, "
                f"{args.flashvsr_spatial} spatial route, tile size "
                f"{args.flashvsr_tile_size}, overlap {args.flashvsr_tile_overlap}, "
                "and staged memory"
            ),
        },
        "clips": [],
    }
    _write_json_atomic(output / "candidate_generation_report.json", report)

    method_paths: dict[str, dict[str, Path]] = {}
    for row in inputs:
        clip_id = row["clip_id"]
        clip_dir = output / clip_id
        source_target = clip_dir / "source.mp4"
        if not source_target.is_file() or _sha256(source_target) != _sha256(row["source"]):
            _copy_atomic(row["source"], source_target)
        paths = {
            "source": source_target,
            "lanczos": clip_dir / "lanczos_2x.mp4",
            "realbasicvsr": clip_dir / "realbasicvsr_conservative_2x.mp4",
            "flashvsr": clip_dir / "flashvsr_quality_locked_2x.mp4",
            "dlss_nr": clip_dir / "dlss_nr_standard_2x.mp4",
        }
        method_paths[clip_id] = paths
        if not args.resume or not _is_valid_2x(
            source_target, paths["lanczos"], ffmpeg=args.ffmpeg
        ):
            print(f"[P4] {clip_id}: Lanczos 2x", flush=True)
            _atomic_lanczos(source_target, paths["lanczos"], args.ffmpeg)
        _validate_2x(source_target, paths["lanczos"], ffmpeg=args.ffmpeg)
        if not paths["dlss_nr"].is_file() or _sha256(paths["dlss_nr"]) != _sha256(
            row["dlss_nr"]
        ):
            _copy_atomic(row["dlss_nr"], paths["dlss_nr"])
        _validate_2x(source_target, paths["dlss_nr"], ffmpeg=args.ffmpeg)

    realbasic = _load_project_module("realbasicvsr_advanced")
    realbasic_reports: dict[str, Any] = {}
    for row in inputs:
        clip_id = row["clip_id"]
        paths = method_paths[clip_id]
        if args.resume and _is_valid_2x(
            paths["source"], paths["realbasicvsr"], ffmpeg=args.ffmpeg
        ):
            continue
        print(f"[P4] {clip_id}: RealBasicVSR conservative", flush=True)
        frames, rate = _decode_frames(paths["source"])
        audio_sentinel = {"source": str(paths["source"]), "identity": True}
        candidate, _, returned_audio, run_report = realbasic.restore_realbasicvsr(
            frames,
            audio_sentinel,
            model_path=args.realbasicvsr_model.resolve(),
            model_name=args.realbasicvsr_model.name,
            output_mode="native_size_restore",
            strength=0.30,
            chunk_frames=2,
            overlap_frames=1,
            precision="auto",
            checkpoint_branch="prefer_ema",
            release_policy="clear_after",
        )
        if returned_audio is not audio_sentinel:
            raise RuntimeError("RealBasicVSR did not return the exact input audio object")
        height, width = int(candidate.shape[1]), int(candidate.shape[2])
        _atomic_frame_encode(
            candidate,
            count=int(candidate.shape[0]),
            width=width * 2,
            height=height * 2,
            rate=rate,
            target=paths["realbasicvsr"],
            ffmpeg=args.ffmpeg,
            lanczos_2x=True,
        )
        realbasic_reports[clip_id] = json.loads(run_report)
        del frames, candidate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _validate_2x(paths["source"], paths["realbasicvsr"], ffmpeg=args.ffmpeg)

    flash = _load_project_module("flashvsr_advanced")
    missing_flash = [
        row
        for row in inputs
        if not (
            args.resume
            and _is_valid_2x(
                method_paths[row["clip_id"]]["source"],
                method_paths[row["clip_id"]]["flashvsr"],
                ffmpeg=args.ffmpeg,
            )
        )
    ]
    flash_reports: dict[str, Any] = {}
    if missing_flash:
        print(f"[P4] load FlashVSR {args.flashvsr_mode}", flush=True)
        handle, load_report = flash.load_flashvsr_model(
            model_dir=args.flashvsr_model_dir.resolve(),
            model_name=args.flashvsr_model_dir.name,
            mode=args.flashvsr_mode,
            precision="bf16",
        )
        for index, row in enumerate(missing_flash):
            clip_id = row["clip_id"]
            paths = method_paths[clip_id]
            print(f"[P4] {clip_id}: FlashVSR Quality Locked", flush=True)
            frames, rate = _decode_frames(paths["source"])
            plan, plan_report = flash.build_flashvsr_plan(
                frames,
                quality_profile="quality_locked",
                spatial_strategy=args.flashvsr_spatial,
                memory_policy="staged",
                base_attention_budget=2.0,
                kv_retention=3.0,
                local_radius=11,
                tile_size=args.flashvsr_tile_size,
                tile_overlap=args.flashvsr_tile_overlap,
            )
            audio_sentinel = {"source": str(paths["source"]), "identity": True}
            candidate, _, returned_audio, run_report = flash.restore_flashvsr(
                handle,
                plan,
                frames,
                audio_sentinel,
                scale=2,
                seed=26090404,
                color_fix=True,
                release_policy="clear_after" if index + 1 == len(missing_flash) else "keep_loaded",
            )
            if returned_audio is not audio_sentinel:
                raise RuntimeError("FlashVSR did not return the exact input audio object")
            _atomic_frame_encode(
                candidate,
                count=int(candidate.shape[0]),
                width=int(candidate.shape[2]),
                height=int(candidate.shape[1]),
                rate=rate,
                target=paths["flashvsr"],
                ffmpeg=args.ffmpeg,
            )
            flash_reports[clip_id] = {
                "load": json.loads(load_report),
                "plan": json.loads(plan_report),
                "run": json.loads(run_report),
            }
            del frames, candidate
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _validate_2x(paths["source"], paths["flashvsr"], ffmpeg=args.ffmpeg)

    manifest_clips = []
    report_clips = []
    for row in inputs:
        clip_id = row["clip_id"]
        paths = method_paths[clip_id]
        source_sha = _sha256(paths["source"])
        methods = {
            "lanczos": _bind_method(source_sha, paths["lanczos"], "lanczos_2x", output),
            "realbasicvsr": _bind_method(
                source_sha, paths["realbasicvsr"], "conservative", output
            ),
            "flashvsr": _bind_method(
                source_sha, paths["flashvsr"], "quality_locked", output
            ),
            "dlss_nr": _bind_method(source_sha, paths["dlss_nr"], "standard", output),
        }
        manifest_clips.append(
            {
                "clip_id": clip_id,
                "label": LABELS[clip_id],
                "clip_type": CLIP_TYPES[clip_id],
                "source": paths["source"].relative_to(output).as_posix(),
                "source_sha256": source_sha,
                "methods": methods,
            }
        )
        report_clips.append(
            {
                "clip_id": clip_id,
                "source_sha256": source_sha,
                "source_contract": _contract_summary(_contract(paths["source"])),
                "methods": {
                    name: {
                        "sha256": methods[name]["candidate_sha256"],
                        "contract": _validate_2x(
                            paths["source"], paths[name], ffmpeg=args.ffmpeg
                        ),
                    }
                    for name in ("lanczos", "realbasicvsr", "flashvsr", "dlss_nr")
                },
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "review_id": "dlss-nr-p4-v13-final-20260904",
        "bitrate_kbps": 12000,
        "clips": manifest_clips,
    }
    _write_json_atomic(output / "p4_manifest.json", manifest)
    report.update(
        {
            "status": "P4_CANDIDATES_READY_FOR_BLIND_PACKAGING",
            "clips": report_clips,
            "realbasicvsr_reports": realbasic_reports,
            "flashvsr_reports": flash_reports,
            "manifest": str(output / "p4_manifest.json"),
            "manifest_sha256": _sha256(output / "p4_manifest.json"),
        }
    )
    _write_json_atomic(output / "candidate_generation_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clips": len(report_clips),
                "manifest": report["manifest"],
                "manifest_sha256": report["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
