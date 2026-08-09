#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent
CASES = {
    "baseline": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "natural", "pitch": "natural", "energy": "natural"},
    "pace_slow": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "slow", "pitch": "natural", "energy": "natural"},
    "pace_fast": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "fast", "pitch": "natural", "energy": "natural"},
    "pitch_low": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "natural", "pitch": "low", "energy": "natural"},
    "pitch_high": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "natural", "pitch": "high", "energy": "natural"},
    "energy_low": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "natural", "pitch": "natural", "energy": "low"},
    "energy_high": {"emotion": "neutral", "prompt_intensity": 0.5, "pace": "natural", "pitch": "natural", "energy": "high"},
    "tender_low": {"emotion": "tender", "prompt_intensity": 0.2, "pace": "natural", "pitch": "natural", "energy": "natural"},
    "tender_high": {"emotion": "tender", "prompt_intensity": 0.8, "pace": "natural", "pitch": "natural", "energy": "natural"},
}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _validator():
    return _module("h3_t8_performance_vram", ROOT / "validate_h3_vram.py")


def _metric_function():
    helper = _module("h3_t8_performance_metrics", ROOT / "validate_speech_multilingual.py")
    return helper._load_metrics()


def _find_one(prompt: dict, class_type: str) -> tuple[str, dict]:
    matches = [(node_id, node) for node_id, node in prompt.items() if node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {class_type}, found {len(matches)}")
    return matches[0]


def _build_prompt(base: dict, case_name: str, seed: int) -> tuple[dict, str]:
    prompt = deepcopy(base)
    plan_id, plan = _find_one(prompt, "MiniMaxH3SpeechPlanT8")
    studio_id, studio = _find_one(prompt, "MiniMaxH3SpeechStudioT8")
    _, save = _find_one(prompt, "SaveAudio")
    next_id = str(max(int(node_id) for node_id in prompt) + 1)
    settings = CASES[case_name]
    prompt[next_id] = {
        "inputs": {
            "segment_index": -1,
            **settings,
            "nonverbal_direction": "",
            "speech_plan": [plan_id, 0],
        },
        "class_type": "MiniMaxH3SpeechPerformanceT8",
    }
    studio["inputs"]["speech_plan"] = [next_id, 0]
    studio["inputs"]["seed"] = int(seed)
    studio["inputs"]["verify_mode"] = "off"
    studio["inputs"]["release_policy"] = "unload_all_models"
    save["inputs"]["filename_prefix"] = f"MiniMaxH3_T8_Speech/performance_matrix/{case_name}"
    return prompt, str(plan["inputs"]["text"])


def _history(server: str, prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{server.rstrip('/')}/history/{prompt_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))[prompt_id]


def _saved_audio(history: dict, comfy_output: Path) -> Path:
    for output in history.get("outputs", {}).values():
        for item in output.get("audio", []):
            if item.get("type") == "output":
                path = (comfy_output / item.get("subfolder", "") / item["filename"]).resolve()
                if path.is_file():
                    return path
    raise FileNotFoundError("successful prompt did not expose a local SaveAudio output")


def _audio_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    rms = math.sqrt(float(np.mean(np.square(mono, dtype=np.float64))) + 1e-12)
    tensor = torch.from_numpy(mono).unsqueeze(0)
    pitch = torchaudio.functional.detect_pitch_frequency(
        tensor,
        int(sample_rate),
        frame_time=0.01,
        win_length=15,
        freq_low=50,
        freq_high=500,
    ).squeeze(0)
    voiced = pitch[(pitch >= 50.0) & (pitch <= 500.0)]
    return {
        "sample_rate": int(sample_rate),
        "samples": int(audio.shape[0]),
        "duration_seconds": float(audio.shape[0] / sample_rate),
        "rms_dbfs": float(20.0 * math.log10(max(rms, 1e-12))),
        "median_f0_hz": float(torch.median(voiced).item()) if voiced.numel() else None,
        "voiced_pitch_frames": int(voiced.numel()),
    }


def _transcribe(model, path: Path, beam_size: int) -> dict:
    segments, info = model.transcribe(
        str(path),
        language="en",
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
    )
    segments = list(segments)
    words = [word for segment in segments for word in (segment.words or []) if word.word.strip()]
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    if words:
        speech_seconds = max(1e-6, float(words[-1].end) - float(words[0].start))
    else:
        speech_seconds = 0.0
    return {
        "text": text,
        "detected_language": str(info.language or ""),
        "word_count": len(words),
        "speech_seconds": speech_seconds,
        "words_per_second": (len(words) / speech_seconds) if speech_seconds else 0.0,
    }


async def _generate(args, validator, names: list[str]):
    base = validator.load_api_prompt(args.workflow)
    generated = []
    for index, name in enumerate(names):
        prompt, expected = _build_prompt(base, name, args.seed)
        runtime = await validator.collect_run(
            prompt,
            server=args.server,
            device_index=args.device_index,
            poll_interval=args.poll_interval,
            baseline_seconds=args.baseline_seconds,
            timeout_seconds=args.timeout,
            preview_method="none",
        )
        item = {
            "case": name,
            "settings": CASES[name],
            "expected_text": expected,
            "runtime": {
                "prompt_id": runtime["prompt_id"],
                "status": runtime["status"],
                "duration_seconds": runtime["duration_seconds"],
                "vram_summary": runtime["summary"],
            },
        }
        if runtime["status"] == "success":
            item["audio_path"] = str(_saved_audio(_history(args.server, runtime["prompt_id"]), args.comfy_output))
        generated.append(item)
        print(f"[{index + 1}/{len(names)}] {name}: {runtime['status']}", flush=True)
        await asyncio.sleep(args.between_runs_seconds)
    return generated


def _monotonic(results: dict, names: tuple[str, str, str], field: str) -> dict:
    values = [results[name].get(field) for name in names]
    valid = all(isinstance(value, (int, float)) for value in values)
    return {
        "cases": list(names),
        "field": field,
        "values": values,
        "strictly_increasing": bool(valid and values[0] < values[1] < values[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and acoustically probe H3 prompt-level performance controls.")
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--seed", type=int, default=2608107001)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--between-runs-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    names = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")

    generated = asyncio.run(_generate(args, _validator(), names))
    from faster_whisper import WhisperModel

    asr = WhisperModel(str(args.asr_model.resolve()), device="cpu", compute_type="int8", cpu_threads=8)
    metric_fn = _metric_function()
    by_name = {}
    for item in generated:
        if "audio_path" not in item:
            by_name[item["case"]] = item
            continue
        path = Path(item["audio_path"])
        stats = _audio_stats(path)
        transcription = _transcribe(asr, path, args.beam_size)
        item["acoustics"] = stats
        item["transcription"] = transcription
        item["text_metrics"] = metric_fn(item["expected_text"], transcription["text"])
        item["words_per_second"] = transcription["words_per_second"]
        item["median_f0_hz"] = stats["median_f0_hz"]
        item["rms_dbfs"] = stats["rms_dbfs"]
        by_name[item["case"]] = item

    gates = {}
    required = set(names)
    if {"pace_slow", "baseline", "pace_fast"} <= required:
        gates["pace_monotonic"] = _monotonic(by_name, ("pace_slow", "baseline", "pace_fast"), "words_per_second")
    if {"pitch_low", "baseline", "pitch_high"} <= required:
        gates["pitch_monotonic"] = _monotonic(by_name, ("pitch_low", "baseline", "pitch_high"), "median_f0_hz")
    if {"energy_low", "baseline", "energy_high"} <= required:
        gates["energy_monotonic"] = _monotonic(by_name, ("energy_low", "baseline", "energy_high"), "rms_dbfs")
    report = {
        "schema": "minimax_h3_t8_speech_performance_matrix_v1",
        "seed": args.seed,
        "cases": generated,
        "acoustic_monotonic_gates": gates,
        "all_requested_generations_succeeded": all(item["runtime"]["status"] == "success" for item in generated),
        "calibration_status": "pilot_only_not_calibrated",
        "scientific_boundary": (
            "A same-seed acoustic pilot can reject non-monotonic controls, but it cannot establish emotion perception, "
            "naturalness, speaker preservation, or a stable numeric mapping. Emotion and intensity still require "
            "multi-seed blinded human ratings or an independently validated classifier."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": gates}, ensure_ascii=False, indent=2))
    return 0 if report["all_requested_generations_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
