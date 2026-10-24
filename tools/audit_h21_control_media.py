"""Audit explicitly completed owned cases, without queueing, GPU or media repair.

History/graph, raw events, full RGB/PCM, frozen external clock inspection and
strict system FFmpeg are separate evidence. No quality or speed pass is issued.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def rational(value):
    return Fraction(value["num"], value["den"])


def contained(root, relative):
    value = Path(root, str(relative).replace("\\", "/")).resolve(strict=True)
    if not value.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("Audit path escaped its explicit owned root")
    return value


def completed_case(manifest, attempt, case_name):
    path = Path(manifest).resolve(strict=True)
    plan = read(path)
    root = Path(plan["run_root"]).resolve(strict=True)
    if path != root / "manifest.json" or plan.get("schema") != "t8.h21.control_probe.v1":
        raise ValueError("Expected an actual contained owned H21 manifest")
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("Select an exact owned attempt UUID")
    case, = [row for row in plan["cases"] if row["name"] == case_name]
    graph_path = contained(root, case["prompt"])
    if digest(graph_path) != plan["sealed_files"][graph_path.relative_to(root).as_posix()]["sha256"]:
        raise ValueError("Frozen case graph SHA changed")
    folder = contained(root, "attempts/" + attempt + "/executions/" + case_name)
    history = read(folder / "history.json")
    if history.get("status", {}).get("status_str") != "success" or not history["status"].get("completed"):
        raise ValueError("This case lacks authoritative completed successful history")
    graph = read(graph_path)
    if canonical(history["prompt"][2]) != canonical(graph):
        raise ValueError("Actual history graph differs from frozen selected graph")
    prompt_id = history["prompt"][1]
    submitted = read(folder / "submitted.json")
    events = read(folder / "observation/terminal.json")
    if submitted["prompt_id"] != prompt_id or events["prompt_id"] != prompt_id:
        raise ValueError("Case prompt IDs disagree across independent evidence")
    movies = []
    for node_id, node in graph.items():
        if node["class_type"] != "MiniMaxH3SafeAVSaveT8Advanced":
            continue
        for entry in history.get("outputs", {}).get(node_id, {}).get("images", []):
            if entry.get("type") != "output":
                raise ValueError("Safe AV output is outside the owned output store")
            movie = contained(root / "output", entry["subfolder"] + "/" + entry["filename"])
            movies.append(movie)
    if len(movies) != 1:
        raise ValueError("Choose a completed case with one actual final Safe AV movie")
    return plan, case, folder, movies[0], events


def audit(manifest, attempt, case_name, target, *, frames, width, height, fps, ffmpeg):
    if any(type(value) is not int or value <= 0 for value in (frames, width, height, fps)):
        raise ValueError("Expected delivery dimensions/frame count/fps must be positive integers")
    plan, case, folder, movie, events = completed_case(manifest, attempt, case_name)
    target = Path(target).resolve()
    repository = Path(__file__).resolve().parents[1]
    if target.exists() or not target.is_relative_to(repository / "artifacts"):
        raise ValueError("Use a new private audit directory, never overwrite evidence")
    target.mkdir(parents=True)
    root, frozen = Path(plan["run_root"]), Path(plan["package_root"]) / "h3_t8"
    original_sha = digest(movie)
    candidate = target / "candidate.mp4"
    shutil.copyfile(movie, candidate)
    if digest(candidate) != original_sha:
        raise ValueError("Candidate is not an exact original media byte copy")
    used_sources = {}
    for name in ("director_media.py", "director_media_worker.py"):
        path = contained(frozen, name)
        sha = digest(path)
        if sha != plan["sealed_files"][path.relative_to(root).as_posix()]["sha256"]:
            raise ValueError("Frozen clock inspector differs from the run snapshot")
        used_sources[str(path)] = sha
    spec = importlib.util.spec_from_file_location("h21_frozen_media_audit", frozen / "director_media.py")
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    clock = media.run_media("external", str(candidate), timeout=180)
    write(target / "frozen-clock.json", clock)
    info, decoded = clock["media"], clock["decoded_clock"]
    if (info["width"], info["height"], info["frames"], info["fps"]) != (width, height, frames, fps):
        raise ValueError("Complete decoded media differs from explicitly expected delivery")
    video, audio = decoded["video"], decoded["audio"]
    if (not decoded["fully_decoded"] or decoded["normalization_performed"]
            or not decoded["zero_origin_film_compatible"]
            or rational(video["first_time"]) != 0
            or rational(video["last_time"]) != Fraction(frames - 1, fps)
            or rational(video["end_time"]) != Fraction(frames, fps)):
        raise ValueError("Full actual video PTS do not match the unchanged requested CFR clock")
    if audio is None or (audio["sample_rate"], audio["channels"]) != (32000, 2):
        raise ValueError("Completed native AV needs its actual stereo32kHz audio")
    overhang = rational(audio["end_time"]) - Fraction(frames, fps)
    if rational(audio["first_time"]) != 0 or abs(overhang) > Fraction(2048, 32000):
        raise ValueError("Actual audio PTS exceed explicit AAC padding allowance")
    for index in (0, frames // 2, frames - 1):
        media.run_media("frame", str(candidate), frame_number=index,
                        output_png=target / ("frame-" + str(index).zfill(3) + ".png"))
    inspector = repository / "tools/audit_owned_short_media.py"
    frozen_inspector = target / "audit_owned_short_media.py"
    shutil.copyfile(inspector, frozen_inspector)
    inspector_sha = digest(frozen_inspector)
    if inspector_sha != digest(inspector):
        raise ValueError("Full RGB/PCM inspector changed while freezing")
    command = [sys.executable, "-I", "-B", "-X", "utf8", str(inspector),
               "--root", str(target), "--output", str(target / "full-av"),
               "--clip", str(candidate), "--frames", str(frames),
               "--width", str(width), "--height", str(height), "--audio-policy", "required"]
    # Preserve the exact inspector source privately, and verify the executed
    # source before/after. Its established artifacts containment stays intact.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180)
    write(target / "full-av-process.json", {"command": command, "exit_code": result.returncode,
          "stdout": result.stdout, "stderr": result.stderr, "source_sha256": inspector_sha})
    if result.returncode or digest(inspector) != inspector_sha:
        raise RuntimeError("Full RGB/PCM audit failed or its inspector changed; original retained")
    strict = [str(ffmpeg), "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
              "-threads", "1", "-hwaccel", "none", "-i", str(candidate),
              "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"]
    process = subprocess.run(strict, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180)
    write(target / "strict-decode.json", {"command": strict, "exit_code": process.returncode,
          "stdout": process.stdout, "stderr": process.stderr})
    if process.returncode or process.stderr:
        raise RuntimeError("Strict full AV decode failed; original retained")
    reports = read(folder / "reports.json")
    if reports["missing_or_invalid"]:
        raise ValueError("Actual runtime reports are incomplete, not an audited completion")
    if digest(candidate) != original_sha or digest(movie) != original_sha:
        raise ValueError("Original or exact candidate bytes changed during independent audit")
    receipt = {"schema": "t8.h21.completed_control_media.v1", "case": case_name,
               "status": "mechanical_AV_clock_verified_NOT_quality_accepted",
               "original": str(movie), "candidate": str(candidate), "media_sha256": original_sha,
               "manifest_sha256": digest(manifest), "history_sha256": digest(folder / "history.json"),
               "prompt_sha256": digest(root / case["prompt"]),
               "event_evidence": events, "runtime_reports": reports,
               "frozen_clock_sources": used_sources, "full_av": read(target / "full-av/report.json"),
               "actual_audio_padding_seconds": str(overhang), "padding_removed": False,
               "source_Core_all_weight_postscan": "owned controller final postscan separate; not inferred here",
               "human_qualified": False, "quality_accepted": False, "GPU_used_for_audit": False,
               "media_reencoded_or_repaired": False, "published": False}
    write(target / "audit.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.attempt, args.case, args.output,
                   frames=args.frames, width=args.width, height=args.height, fps=args.fps, ffmpeg=args.ffmpeg)
    print(json.dumps({"case": result["case"], "status": result["status"]}))


if __name__ == "__main__":
    main()
