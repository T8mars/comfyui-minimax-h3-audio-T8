#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_longform_tool"


def _load_package():
    if PACKAGE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE_NAME,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE_NAME] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    speech = __import__(f"{PACKAGE_NAME}.speech", fromlist=["make_voice_profile"])
    extended = __import__(f"{PACKAGE_NAME}.speech_extended", fromlist=["start_or_resume_longform"])
    return speech, extended


def _plan(speech, segment_count: int):
    profile, _, _ = speech.make_voice_profile(
        "described_voice", "state_probe", "a neutral test voice", "English", False
    )
    return {
        "schema": speech.SPEECH_PLAN_SCHEMA,
        "kind": "speech",
        "profiles": {profile["speaker_id"]: profile},
        "segments": [
            {
                "index": index,
                "speaker_id": profile["speaker_id"],
                "text": f"Long form state validation segment {index + 1}.",
                "language": "English",
                "direction": "neutral",
                "emotion": "neutral",
                "emotion_intensity": 0.5,
                "space": "studio",
                "pause_before_seconds": 0.0,
                "overlap_seconds": 0.0,
                "gain_db": 0.0,
                "pan": 0.0,
            }
            for index in range(segment_count)
        ],
        "chunking": "validation_fixed",
        "timing_status": "planned_only_until_audio_is_rendered",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate 32s/2m/10m speech manifest resume and exact assembly without claiming H3 quality.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--durations", type=int, nargs="+", default=[32, 120, 600])
    parser.add_argument("--segment-seconds", type=int, default=4)
    args = parser.parse_args()
    speech, extended = _load_package()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extended.folder_paths.get_output_directory = lambda: str(args.output_dir.resolve())
    sample_rate = 32000
    samples = args.segment_seconds * sample_rate
    time_axis = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = (0.02 * torch.sin(2 * torch.pi * 220.0 * time_axis)).view(1, 1, -1)
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    results = []
    for duration in args.durations:
        if duration % args.segment_seconds:
            raise ValueError("each duration must be divisible by segment_seconds")
        segment_count = duration // args.segment_seconds
        plan = _plan(speech, segment_count)
        job_id = f"state_{duration}s_{time.time_ns()}"
        started = time.perf_counter()
        resume_count = 0
        cancelled_once = False
        while True:
            session, _, index, text, _ = extended.start_or_resume_longform(plan, job_id)
            resume_count += 1
            if index < 0:
                if session["state"] == "cancelled":
                    extended.control_longform_session(job_id, "clear_cancel")
                    continue
                break
            if not cancelled_once and index >= max(1, segment_count // 2):
                extended.control_longform_session(job_id, "request_cancel")
                cancelled, _, cancelled_index, _, _ = extended.start_or_resume_longform(plan, job_id)
                if cancelled["state"] != "cancelled" or cancelled_index != -1:
                    raise RuntimeError("cooperative cancellation was not durable")
                extended.control_longform_session(job_id, "clear_cancel")
                cancelled_once = True
                continue
            extended.accept_longform_segment(
                session,
                plan,
                index,
                audio,
                text,
                1.0,
                0.0,
                True,
            )
            # Intentionally discard the returned session every segment. The next loop
            # reconstructs state solely from the durable manifest, simulating restart.
        composed, timeline_json, _, _ = extended.compose_longform_session(plan, job_id, 0.0, -1.0)
        expected_samples = duration * sample_rate
        actual_samples = int(composed["waveform"].shape[-1])
        timeline = json.loads(timeline_json)
        results.append(
            {
                "duration_seconds": duration,
                "segment_count": segment_count,
                "resume_count": resume_count,
                "cancel_clear_cycle_pass": cancelled_once,
                "expected_samples": expected_samples,
                "actual_samples": actual_samples,
                "sample_error": actual_samples - expected_samples,
                "hash_verification_pass": timeline["longform"]["segment_hashes_verified"],
                "runtime_seconds": time.perf_counter() - started,
                "mechanical_gate_pass": actual_samples == expected_samples,
            }
        )
    report = {
        "schema": "minimax_h3_t8_speech_longform_state_validation_v1",
        "results": results,
        "all_mechanical_gates_pass": all(result["mechanical_gate_pass"] for result in results),
        "boundary": (
            "This validates atomic manifests, per-segment restart, cooperative between-segment cancel, "
            "hash verification and sample-exact assembly with synthetic audio. It does not validate "
            "H3 voice continuity, ASR, identity drift, GPU memory, or in-flight sampler interruption."
        ),
    }
    output = args.output_dir / "longform_state_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_mechanical_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
