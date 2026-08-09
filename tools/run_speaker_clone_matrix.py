#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen

import soundfile


ROOT = Path(__file__).resolve().parent


def _validator():
    spec = importlib.util.spec_from_file_location("h3_t8_clone_vram_validator", ROOT / "validate_h3_vram.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _history(server: str, prompt_id: str) -> dict:
    with urlopen(f"{server.rstrip('/')}/history/{prompt_id}", timeout=30) as response:
        value = json.load(response)
    return value[prompt_id]


def _prepare_prompt(template: dict, source: dict, index: int, args) -> dict:
    prompt = deepcopy(template)
    reference_path = Path(source["reference_audio"])
    duration = soundfile.info(str(reference_path)).duration
    if duration < 2.0:
        raise ValueError(f"speaker {source['speaker_id']} reference is shorter than 2 seconds")
    prompt["5"]["inputs"]["audio"] = f"h3_voiceval_{reference_path.name}"
    prompt["6"]["inputs"].update(
        {
            "speaker_id": f"librispeech_{source['speaker_id']}",
            "voice_description": "the same licensed LibriSpeech adult speaker, with natural clear diction",
            "reference_duration_seconds": min(15.0, duration),
        }
    )
    prompt["7"]["inputs"].update(
        {
            "text": args.target_text,
            "acting_direction": "neutral, clear, conversational, and naturally paced",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
        }
    )
    prompt["8"]["inputs"].update(
        {
            "seed": args.seed_start + index,
            "render_seconds": args.render_seconds,
            "steps": args.steps,
            "verify_mode": "trim_exact_target",
            "min_similarity": 0.0,
            "release_policy": "unload_all_models",
        }
    )
    prompt["9"]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3_T8_Speech/identity_matrix/speaker_{source['speaker_id']}"
    )
    return prompt


async def _run(args, validator):
    template = validator.load_api_prompt(args.template)
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    results = []
    for index, source in enumerate(sources["speakers"]):
        prompt = _prepare_prompt(template, source, index, args)
        runtime = await validator.collect_run(
            prompt,
            server=args.server,
            device_index=0,
            poll_interval=0.1,
            baseline_seconds=1.0,
            timeout_seconds=args.timeout,
            preview_method="none",
        )
        history = _history(args.server, runtime["prompt_id"])
        audio_items = history.get("outputs", {}).get("9", {}).get("audio", [])
        generated = None
        if audio_items:
            item = audio_items[0]
            generated = (
                args.comfy_output
                / item.get("subfolder", "")
                / item["filename"]
            ).resolve()
        summary = runtime["summary"]
        total = runtime["server_snapshot"]["devices"][0]["vram_total"]
        peak = int(summary["peak_vram_used_bytes"])
        results.append(
            {
                "speaker_id": source["speaker_id"],
                "source_id": source["source_id"],
                "reference_audio": source["reference_audio"],
                "generated_audios": [str(generated)] if generated else [],
                "status": runtime["status"],
                "duration_seconds": runtime["duration_seconds"],
                "peak_vram_used_mib": peak / (1024**2),
                "minimum_headroom_mib": (total - peak) / (1024**2),
                "prompt_id": runtime["prompt_id"],
                "seed": args.seed_start + index,
            }
        )
        print(
            f"[{index + 1}/{len(sources['speakers'])}] speaker={source['speaker_id']} "
            f"status={runtime['status']} headroom={(total - peak)/(1024**2):.1f}MiB",
            flush=True,
        )
        if runtime["status"] != "success" or generated is None:
            break
        await asyncio.sleep(args.between_runs_seconds)
    return sources, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a licensed multi-speaker Ref2VA clone evaluation matrix.")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-text", default="The morning train is waiting beside the river.")
    parser.add_argument("--seed-start", type=int, default=2608103601)
    parser.add_argument("--render-seconds", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--between-runs-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    validator = _validator()
    sources, results = asyncio.run(_run(args, validator))
    report = {
        "schema": "minimax_h3_t8_clone_generation_matrix_v1",
        "dataset": sources["dataset"],
        "dataset_revision": sources["dataset_revision"],
        "license": sources["license"],
        "target_text": args.target_text,
        "steps": args.steps,
        "render_seconds": args.render_seconds,
        "speakers": results,
        "all_success": len(results) == len(sources["speakers"]) and all(result["status"] == "success" for result in results),
        "high_fidelity_clone_claim": False,
        "boundary": "Generation success and embedding separation do not replace blinded human ABX.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
