#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics

import numpy as np
import soundfile
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMForXVector


def _audio(path: Path) -> tuple[torch.Tensor, int]:
    values, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(values.T.copy()).mean(dim=0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform, 16000


def _embedding(extractor, model, path: Path) -> torch.Tensor:
    waveform, sample_rate = _audio(path)
    inputs = extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        embedding = model(**inputs).embeddings[0].to(dtype=torch.float32)
    return torch.nn.functional.normalize(embedding, dim=0)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _copy_trial_audio(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate genuine/impostor speaker similarity and build a blinded ABX package."
    )
    parser.add_argument("manifest", type=Path, help="JSON {speakers:[{speaker_id,reference_audio,generated_audios:[...]}]}")
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-speakers", type=int, default=10)
    parser.add_argument("--impostor-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-genuine-pass-rate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    speakers = payload.get("speakers") if isinstance(payload, dict) else None
    if not isinstance(speakers, list) or not speakers:
        raise ValueError("manifest must contain a non-empty speakers list")
    if len(speakers) < 2:
        raise ValueError("speaker identity evaluation requires at least two speakers")
    ids = [str(entry["speaker_id"]) for entry in speakers]
    if len(ids) != len(set(ids)):
        raise ValueError("speaker_id values must be unique")

    extractor = AutoFeatureExtractor.from_pretrained(str(args.speaker_model.resolve()), local_files_only=True)
    model = WavLMForXVector.from_pretrained(str(args.speaker_model.resolve()), local_files_only=True)
    model.eval()
    references = {}
    generated = {}
    paths = {}
    for entry in speakers:
        speaker_id = str(entry["speaker_id"])
        reference_path = Path(entry["reference_audio"])
        if not reference_path.is_absolute():
            reference_path = (args.manifest.parent / reference_path).resolve()
        generated_paths = []
        for value in entry.get("generated_audios", []):
            path = Path(value)
            if not path.is_absolute():
                path = (args.manifest.parent / path).resolve()
            generated_paths.append(path)
        if not reference_path.is_file() or not generated_paths or any(not path.is_file() for path in generated_paths):
            raise FileNotFoundError(f"speaker {speaker_id} has a missing reference or generated file")
        references[speaker_id] = _embedding(extractor, model, reference_path)
        generated[speaker_id] = [_embedding(extractor, model, path) for path in generated_paths]
        paths[speaker_id] = {"reference": reference_path, "generated": generated_paths}

    genuine = []
    impostor = []
    per_speaker = {}
    for speaker_id in ids:
        own = [float(torch.dot(references[speaker_id], vector)) for vector in generated[speaker_id]]
        wrong = [
            float(torch.dot(references[other_id], vector))
            for other_id in ids
            if other_id != speaker_id
            for vector in generated[speaker_id]
        ]
        genuine.extend(own)
        impostor.extend(wrong)
        per_speaker[speaker_id] = {
            "genuine_scores": own,
            "impostor_scores": wrong,
            "genuine_mean": statistics.fmean(own),
            "impostor_mean": statistics.fmean(wrong) if wrong else None,
        }

    threshold = _percentile(impostor, args.impostor_percentile) if impostor else 1.0
    genuine_pass_rate = sum(score > threshold for score in genuine) / max(1, len(genuine))
    enough_speakers = len(ids) >= args.minimum_speakers
    machine_gate_pass = enough_speakers and genuine_pass_rate >= args.minimum_genuine_pass_rate

    rng = random.Random(args.seed)
    blind_dir = args.output_dir / "abx_blind"
    answers = []
    public_rows = []
    for trial_index, speaker_id in enumerate(ids):
        generated_index = rng.randrange(len(paths[speaker_id]["generated"]))
        impostor_id = rng.choice([value for value in ids if value != speaker_id])
        correct_is_a = bool(rng.randrange(2))
        a_source = paths[speaker_id]["reference"] if correct_is_a else paths[impostor_id]["reference"]
        b_source = paths[impostor_id]["reference"] if correct_is_a else paths[speaker_id]["reference"]
        x_source = paths[speaker_id]["generated"][generated_index]
        trial_name = f"trial_{trial_index + 1:03d}"
        _copy_trial_audio(a_source, blind_dir / f"{trial_name}_A{a_source.suffix.lower()}")
        _copy_trial_audio(b_source, blind_dir / f"{trial_name}_B{b_source.suffix.lower()}")
        _copy_trial_audio(x_source, blind_dir / f"{trial_name}_X{x_source.suffix.lower()}")
        public_rows.append({"trial": trial_name, "listener_choice_A_or_B": "", "confidence_1_to_5": "", "notes": ""})
        answers.append(
            {
                "trial": trial_name,
                "correct": "A" if correct_is_a else "B",
                "speaker_id": speaker_id,
                "impostor_id": impostor_id,
                "generated_index": generated_index,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "abx_listener_sheet.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    (args.output_dir / "abx_answer_key.json").write_text(
        json.dumps({"seed": args.seed, "answers": answers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "schema": "minimax_h3_t8_speaker_identity_validation_v1",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "speaker_model": str(args.speaker_model.resolve()),
        "speaker_count": len(ids),
        "minimum_required_speakers": args.minimum_speakers,
        "genuine_count": len(genuine),
        "impostor_count": len(impostor),
        "impostor_percentile": args.impostor_percentile,
        "calibrated_threshold": threshold,
        "genuine_pass_rate_above_impostor_threshold": genuine_pass_rate,
        "minimum_genuine_pass_rate": args.minimum_genuine_pass_rate,
        "representative_set_gate_pass": machine_gate_pass,
        "human_abx_status": "package_created_not_listened",
        "per_speaker": per_speaker,
        "denial": (
            "Do not claim high-fidelity cloning from embedding scores alone. At least three blinded "
            "listeners must complete the ABX sheet and the predeclared identity gates must pass."
        ),
    }
    (args.output_dir / "identity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if machine_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
