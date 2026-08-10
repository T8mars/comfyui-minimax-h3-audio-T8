from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Mapping
import unicodedata

import numpy as np
import torch
import torchaudio

import folder_paths

from .core import validate_audio


ASR_LANGUAGE_CODES = {
    "auto": None,
    "Arabic": "ar",
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
}

_ASR_MODEL = None
_ASR_MODEL_PATH: str | None = None
_ASR_LOCK = threading.Lock()
_SPEAKER_MODEL = None
_SPEAKER_EXTRACTOR = None
_SPEAKER_MODEL_PATH: str | None = None
_SPEAKER_LOCK = threading.Lock()
_UNIT_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]",
    flags=re.IGNORECASE,
)


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def normalized_asr_units(text: str) -> list[str]:
    return _UNIT_PATTERN.findall(str(text or "").lower().replace("’", "'"))


def _word_unit_stream(words: list[dict]) -> tuple[list[str], list[int]]:
    heard_units: list[str] = []
    owners: list[int] = []
    for word_index, word in enumerate(words):
        units = normalized_asr_units(word.get("word", ""))
        heard_units.extend(units)
        owners.extend([word_index] * len(units))
    return heard_units, owners


def _levenshtein(expected: list[str], heard: list[str]) -> int:
    row = list(range(len(heard) + 1))
    for expected_index, expected_unit in enumerate(expected, 1):
        next_row = [expected_index]
        for heard_index, heard_unit in enumerate(heard, 1):
            next_row.append(
                min(
                    next_row[-1] + 1,
                    row[heard_index] + 1,
                    row[heard_index - 1] + (expected_unit != heard_unit),
                )
            )
        row = next_row
    return row[-1]


def transcript_metrics(expected: str, heard: str) -> dict:
    expected_units = normalized_asr_units(expected)
    heard_units = normalized_asr_units(heard)
    distance = _levenshtein(expected_units, heard_units)
    expected_nfkc = unicodedata.normalize("NFKC", str(expected or "")).casefold()
    heard_nfkc = unicodedata.normalize("NFKC", str(heard or "")).casefold()
    expected_characters = [character for character in expected_nfkc if character.isalnum()]
    heard_characters = [character for character in heard_nfkc if character.isalnum()]
    character_distance = _levenshtein(expected_characters, heard_characters)
    contains_cjk = bool(
        re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", expected_nfkc)
    )
    primary_name = "CER" if contains_cjk else "WER"
    return {
        "expected_units": len(expected_units),
        "heard_units": len(heard_units),
        "edit_distance": distance,
        "word_or_character_error_rate": distance / max(1, len(expected_units)),
        "primary_metric": primary_name,
        "primary_error_rate": distance / max(1, len(expected_units)),
        "character_error_rate": character_distance / max(1, len(expected_characters)),
        "character_edit_distance": character_distance,
        "expected_characters": len(expected_characters),
        "heard_characters": len(heard_characters),
        "normalized_similarity": max(
            0.0,
            1.0 - distance / max(1, len(expected_units), len(heard_units)),
        ),
    }


def exact_target_word_spans(words: list[dict], expected: str) -> list[dict]:
    expected_units = normalized_asr_units(expected)
    if not expected_units:
        raise ValueError("expected_text cannot be empty")
    heard_units, owners = _word_unit_stream(words)
    if len(heard_units) < len(expected_units):
        return []
    spans = []
    for start in range(len(heard_units) - len(expected_units) + 1):
        end = start + len(expected_units)
        if heard_units[start:end] != expected_units:
            continue
        first_word_index = owners[start]
        last_word_index = owners[end - 1]
        first = words[first_word_index]
        last = words[last_word_index]
        spans.append(
            {
                "start_seconds": float(first["start"]),
                "end_seconds": float(last["end"]),
                "start_unit_index": start,
                "end_unit_index": end,
                "start_word_index": first_word_index,
                "end_word_index": last_word_index + 1,
            }
        )
    return spans


def exact_target_word_bounds(words: list[dict], expected: str) -> tuple[float, float] | None:
    spans = exact_target_word_spans(words, expected)
    if not spans:
        return None
    return spans[0]["start_seconds"], spans[0]["end_seconds"]


def _candidate_asr_roots() -> list[Path]:
    roots = [
        Path(folder_paths.models_dir) / "TTS",
        Path(folder_paths.models_dir) / "asr",
    ]
    configured = os.environ.get("H3_T8_ASR_MODEL_DIR", "").strip()
    if configured:
        roots.insert(0, Path(configured).expanduser())
    return roots


def resolve_asr_model_directory(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(
            "ASR verification requires a faster-whisper CTranslate2 model directory; "
            "provide an absolute path or a directory name under ComfyUI/models/TTS"
        )
    candidate = Path(raw).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        root / candidate for root in _candidate_asr_roots()
    ]
    for path in candidates:
        resolved = path.resolve()
        if (resolved / "config.json").is_file() and (resolved / "model.bin").is_file():
            return resolved
    raise ValueError(
        f"faster-whisper model directory {raw!r} is missing config.json or model.bin"
    )


def resolve_speaker_model_directory(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(
            "speaker verification requires a local WavLM X-Vector model directory"
        )
    candidate = Path(raw).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        root / candidate for root in _candidate_asr_roots()
    ]
    for path in candidates:
        resolved = path.resolve()
        if (
            (resolved / "config.json").is_file()
            and (resolved / "preprocessor_config.json").is_file()
            and (resolved / "pytorch_model.bin").is_file()
        ):
            return resolved
    raise ValueError(
        f"speaker model directory {raw!r} is missing the local WavLM X-Vector files"
    )


def _audio_to_asr_array(audio: Mapping) -> np.ndarray:
    waveform, sample_rate = validate_audio(audio, "speech_audio")
    mono = torch.nan_to_num(
        waveform.detach().to(device="cpu", dtype=torch.float32)
    ).mean(dim=1)
    if sample_rate != 16000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16000)
    return mono[0].contiguous().numpy()


def _load_asr_model(model_path: Path, cpu_threads: int):
    global _ASR_MODEL, _ASR_MODEL_PATH
    canonical = str(model_path)
    if _ASR_MODEL is not None and _ASR_MODEL_PATH == canonical:
        return _ASR_MODEL, True
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is optional and not installed in this ComfyUI Python environment"
        ) from exc
    _ASR_MODEL = WhisperModel(
        canonical,
        device="cpu",
        compute_type="int8",
        cpu_threads=int(cpu_threads),
    )
    _ASR_MODEL_PATH = canonical
    return _ASR_MODEL, False


def _release_asr_model() -> bool:
    global _ASR_MODEL, _ASR_MODEL_PATH
    released = _ASR_MODEL is not None
    _ASR_MODEL = None
    _ASR_MODEL_PATH = None
    gc.collect()
    return released


def _load_speaker_model(model_path: Path):
    global _SPEAKER_EXTRACTOR, _SPEAKER_MODEL, _SPEAKER_MODEL_PATH
    canonical = str(model_path)
    if _SPEAKER_MODEL is not None and _SPEAKER_MODEL_PATH == canonical:
        return _SPEAKER_EXTRACTOR, _SPEAKER_MODEL, True
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except ImportError as exc:
        raise RuntimeError(
            "transformers with WavLMForXVector is required for optional speaker verification"
        ) from exc
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        canonical,
        local_files_only=True,
    )
    speaker_model = WavLMForXVector.from_pretrained(
        canonical,
        local_files_only=True,
    ).to(device="cpu")
    speaker_model.eval()
    _SPEAKER_EXTRACTOR = extractor
    _SPEAKER_MODEL = speaker_model
    _SPEAKER_MODEL_PATH = canonical
    return _SPEAKER_EXTRACTOR, _SPEAKER_MODEL, False


def _release_speaker_model() -> bool:
    global _SPEAKER_EXTRACTOR, _SPEAKER_MODEL, _SPEAKER_MODEL_PATH
    released = _SPEAKER_MODEL is not None
    _SPEAKER_EXTRACTOR = None
    _SPEAKER_MODEL = None
    _SPEAKER_MODEL_PATH = None
    gc.collect()
    return released


def _speaker_array(audio: Mapping) -> np.ndarray:
    waveform, sample_rate = validate_audio(audio, "speaker_audio")
    mono = torch.nan_to_num(
        waveform.detach().to(device="cpu", dtype=torch.float32)
    ).mean(dim=1)
    if sample_rate != 16000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16000)
    return mono[0].contiguous().numpy()


def _speaker_cosine(extractor, model, reference_audio: Mapping, generated_audio: Mapping):
    arrays = [_speaker_array(reference_audio), _speaker_array(generated_audio)]
    inputs = extractor(
        arrays,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return float(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0))


def _transcribe(
    model,
    audio: Mapping,
    language: str,
    beam_size: int,
) -> dict:
    array = _audio_to_asr_array(audio)
    language_code = ASR_LANGUAGE_CODES.get(language)
    if language not in ASR_LANGUAGE_CODES:
        raise ValueError(f"unsupported ASR language: {language}")
    segments, info = model.transcribe(
        array,
        language=language_code,
        beam_size=int(beam_size),
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    realized = list(segments)
    words = []
    for segment in realized:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            words.append(
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": str(word.word).strip(),
                }
            )
    return {
        "text": " ".join(segment.text.strip() for segment in realized).strip(),
        "language": str(info.language),
        "language_probability": float(info.language_probability),
        "duration_seconds": float(info.duration),
        "words": words,
    }


def analyze_dialogue_boundary(
    audio: Mapping,
    expected_text: str,
    asr_model_directory: str,
    language: str = "auto",
    beam_size: int = 5,
    cpu_threads: int = 8,
    unload_after_analyze: bool = True,
    tail_activity_threshold_dbfs: float = -45.0,
):
    waveform, sample_rate = validate_audio(audio, "dialogue_mix")
    expected_text = str(expected_text or "").strip()
    if not expected_text:
        raise ValueError("expected_text cannot be empty")
    if language not in ASR_LANGUAGE_CODES:
        raise ValueError(f"unsupported ASR language: {language}")
    if not 1 <= int(beam_size) <= 20:
        raise ValueError("beam_size must be between 1 and 20")
    if not 1 <= int(cpu_threads) <= 64:
        raise ValueError("cpu_threads must be between 1 and 64")
    if not -120.0 <= float(tail_activity_threshold_dbfs) <= 0.0:
        raise ValueError("tail_activity_threshold_dbfs must be between -120 and 0")

    model_path = resolve_asr_model_directory(asr_model_directory)
    with _ASR_LOCK:
        model, reused = _load_asr_model(model_path, int(cpu_threads))
        released = False
        try:
            result = _transcribe(model, audio, language, int(beam_size))
        finally:
            if unload_after_analyze:
                released = _release_asr_model()

    words = result["words"]
    heard_units, _owners = _word_unit_stream(words)
    spans = exact_target_word_spans(words, expected_text)
    unique_target_found = len(spans) == 1
    start_seconds = 0.0
    end_seconds = 0.0
    before_units: list[str] = []
    after_units: list[str] = []
    tail_signal = {
        "measured": False,
        "warning": "No unique exact ASR target boundary was available.",
    }
    if unique_target_found:
        span = spans[0]
        start_seconds = span["start_seconds"]
        end_seconds = span["end_seconds"]
        before_units = heard_units[: span["start_unit_index"]]
        after_units = heard_units[span["end_unit_index"] :]
        end_sample = min(waveform.shape[-1], round(end_seconds * sample_rate))
        tail = waveform.detach().to(device="cpu", dtype=torch.float32)[..., end_sample:]
        if tail.numel():
            rms = float(tail.square().mean().sqrt())
            threshold = 10.0 ** (float(tail_activity_threshold_dbfs) / 20.0)
            active_ratio = float((tail.abs().mean(dim=1) >= threshold).float().mean())
            tail_signal = {
                "measured": True,
                "start_sample": end_sample,
                "sample_count": int(tail.shape[-1]),
                "duration_seconds": tail.shape[-1] / sample_rate,
                "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
                "activity_threshold_dbfs": float(tail_activity_threshold_dbfs),
                "samples_above_threshold_ratio": active_ratio,
                "is_not_a_speech_classifier": True,
            }
        else:
            tail_signal = {
                "measured": True,
                "start_sample": end_sample,
                "sample_count": 0,
                "duration_seconds": 0.0,
                "rms_dbfs": -240.0,
                "activity_threshold_dbfs": float(tail_activity_threshold_dbfs),
                "samples_above_threshold_ratio": 0.0,
                "is_not_a_speech_classifier": True,
            }

    clean_exact = unique_target_found and not before_units and not after_units
    if len(spans) == 0:
        status = "target_not_found"
    elif len(spans) > 1:
        status = "ambiguous_multiple_exact_targets"
    elif clean_exact:
        status = "clean_exact_target"
    else:
        status = "unique_target_with_lexical_extras"
    metrics = transcript_metrics(expected_text, result["text"])
    report = {
        "schema": "minimax_h3_t8_dialogue_boundary_v1",
        "status": status,
        "unique_exact_target_found": unique_target_found,
        "clean_exact_target": clean_exact,
        "exact_target_span_count": len(spans),
        "expected_text": expected_text,
        "transcript": result["text"],
        "transcript_metrics": metrics,
        "target_bounds": (
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "start_sample": round(start_seconds * sample_rate),
                "end_sample": round(end_seconds * sample_rate),
            }
            if unique_target_found
            else None
        ),
        "lexical_extras": {
            "before_units": before_units,
            "after_units": after_units,
            "before_count": len(before_units),
            "after_count": len(after_units),
        },
        "tail_signal_activity": tail_signal,
        "asr": {
            "engine": "faster-whisper",
            "device": "cpu",
            "compute_type": "int8",
            "model_directory": str(model_path),
            "model_reused": reused,
            "detected_language": result["language"],
            "language_probability": result["language_probability"],
            "unload": {
                "requested": bool(unload_after_analyze),
                "released": bool(released),
                "scope": "optional_cpu_asr_only",
            },
        },
        "claims": {
            "audio_was_modified": False,
            "boundary_is_exact_model_ground_truth": False,
            "tail_energy_proves_speech": False,
        },
        "limitations": [
            "The boundary is accepted only when ASR finds exactly one contiguous exact target-unit sequence.",
            "ASR can miss mumbling or hallucinated non-lexical vocalization.",
            "Tail energy reports signal activity, not whether that signal is speech, music, ambience, or effects.",
        ],
    }
    return (
        result["text"],
        bool(unique_target_found),
        bool(clean_exact),
        float(start_seconds),
        float(end_seconds),
        int(len(before_units)),
        int(len(after_units)),
        _json(report),
    )


def _trim_audio_to_bounds(
    audio: Mapping,
    bounds: tuple[float, float],
    pre_padding_seconds: float,
    post_padding_seconds: float,
) -> tuple[dict, dict]:
    waveform, sample_rate = validate_audio(audio, "speech_audio")
    start = max(0, round((bounds[0] - pre_padding_seconds) * sample_rate))
    end = min(
        waveform.shape[-1],
        round((bounds[1] + post_padding_seconds) * sample_rate),
    )
    if end <= start:
        raise ValueError("ASR target bounds produced an empty trim")
    trimmed = waveform[..., start:end].detach().to(device="cpu", dtype=torch.float32).clone()
    fade = min(round(0.012 * sample_rate), trimmed.shape[-1] // 2)
    if fade:
        phase = torch.linspace(0.0, math.pi / 2.0, fade, dtype=trimmed.dtype)
        curve = phase.sin().square().to(trimmed.device)
        trimmed[..., :fade] *= curve
        trimmed[..., -fade:] *= curve.flip(0)
    return {
        "waveform": trimmed.contiguous(),
        "sample_rate": int(sample_rate),
    }, {
        "start_sample": start,
        "end_sample": end,
        "start_seconds": start / sample_rate,
        "end_seconds": end / sample_rate,
        "pre_padding_seconds": float(pre_padding_seconds),
        "post_padding_seconds": float(post_padding_seconds),
    }


def _apply_peak_limit(audio: Mapping, peak_limit_dbfs: float) -> tuple[dict, dict]:
    waveform, sample_rate = validate_audio(audio, "speech_audio")
    if not -24.0 <= float(peak_limit_dbfs) <= 0.0:
        raise ValueError("peak_limit_dbfs must be between -24 and 0")
    peak_before = float(waveform.detach().abs().amax())
    limit = 10.0 ** (float(peak_limit_dbfs) / 20.0)
    gain = min(1.0, limit / max(peak_before, 1e-12))
    output = {
        "waveform": (
            waveform.detach().to(device="cpu", dtype=torch.float32) * gain
        ).contiguous(),
        "sample_rate": int(sample_rate),
    }
    return output, {
        "limit_dbfs": float(peak_limit_dbfs),
        "peak_before": peak_before,
        "gain": gain,
        "attenuation_db": 20.0 * math.log10(max(gain, 1e-12)),
        "applied": gain < 1.0,
    }


def verify_speech_audio(
    audio: Mapping,
    expected_text: str,
    verify_mode: str = "off",
    asr_model_directory: str = "",
    language: str = "auto",
    min_similarity: float = 0.85,
    beam_size: int = 5,
    cpu_threads: int = 8,
    unload_after_verify: bool = True,
    strict: bool = False,
    pre_padding_seconds: float = 0.12,
    post_padding_seconds: float = 0.25,
    reference_audio: Mapping | None = None,
    speaker_check_mode: str = "off",
    speaker_model_directory: str = "",
    min_speaker_similarity: float = 0.86,
    unload_speaker_after_verify: bool = True,
    peak_limit_dbfs: float = -1.0,
):
    validate_audio(audio, "speech_audio")
    verify_mode = str(verify_mode)
    if verify_mode not in {"off", "verify_only", "trim_exact_target"}:
        raise ValueError("verify_mode must be off, verify_only, or trim_exact_target")
    speaker_check_mode = str(speaker_check_mode)
    if speaker_check_mode not in {"off", "report_cosine", "require_threshold"}:
        raise ValueError(
            "speaker_check_mode must be off, report_cosine, or require_threshold"
        )
    expected_text = str(expected_text or "").strip()
    if verify_mode != "off" and not expected_text:
        raise ValueError("expected_text cannot be empty")
    if verify_mode == "off" and speaker_check_mode == "off":
        output, peak_report = _apply_peak_limit(audio, peak_limit_dbfs)
        report = {
            "schema": "minimax_h3_t8_speech_verify_v1",
            "status": "disabled",
            "warning": "No ASR or speaker-identity verification was run.",
            "peak_limit": peak_report,
        }
        return output, "", 0.0, 0.0, False, _json(report)
    if not 0.0 <= float(min_similarity) <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    if not 1 <= int(beam_size) <= 20:
        raise ValueError("beam_size must be between 1 and 20")
    if not 1 <= int(cpu_threads) <= 64:
        raise ValueError("cpu_threads must be between 1 and 64")
    if min(pre_padding_seconds, post_padding_seconds) < 0.0:
        raise ValueError("ASR trim padding cannot be negative")

    output = dict(audio)
    transcript = ""
    text_similarity = 0.0
    text_accepted = verify_mode == "off"
    text_report = {"status": "not_run"}
    if verify_mode != "off":
        model_path = resolve_asr_model_directory(asr_model_directory)
        with _ASR_LOCK:
            model, reused = _load_asr_model(model_path, int(cpu_threads))
            released = False
            try:
                raw_result = _transcribe(model, audio, language, int(beam_size))
                bounds = exact_target_word_bounds(raw_result["words"], expected_text)
                trim_report = {
                    "applied": False,
                    "exact_target_span_found": bounds is not None,
                }
                final_result = raw_result
                if verify_mode == "trim_exact_target" and bounds is not None:
                    output, trim_details = _trim_audio_to_bounds(
                        audio,
                        bounds,
                        float(pre_padding_seconds),
                        float(post_padding_seconds),
                    )
                    trim_report.update({"applied": True, **trim_details})
                    final_result = _transcribe(model, output, language, int(beam_size))
                metrics = transcript_metrics(expected_text, final_result["text"])
                transcript = final_result["text"]
                text_similarity = float(metrics["normalized_similarity"])
                text_accepted = text_similarity >= float(min_similarity)
                text_report = {
                    "status": "accepted" if text_accepted else "rejected",
                    "verify_mode": verify_mode,
                    "expected_text": expected_text,
                    "raw_transcript": raw_result["text"],
                    "transcript": transcript,
                    "metrics": metrics,
                    "minimum_similarity": float(min_similarity),
                    "trim": trim_report,
                    "asr": {
                        "engine": "faster-whisper",
                        "device": "cpu",
                        "compute_type": "int8",
                        "model_directory": str(model_path),
                        "model_reused": reused,
                        "detected_language": final_result["language"],
                        "language_probability": final_result["language_probability"],
                    },
                }
            finally:
                if unload_after_verify:
                    released = _release_asr_model()
            text_report["asr"]["unload"] = {
                "requested": bool(unload_after_verify),
                "released": bool(released),
                "scope": "optional_cpu_asr_only",
            }

    output, peak_report = _apply_peak_limit(output, peak_limit_dbfs)
    speaker_similarity = 0.0
    speaker_accepted = speaker_check_mode == "off"
    speaker_report = {"status": "not_run"}
    if speaker_check_mode != "off":
        if reference_audio is None:
            raise ValueError("speaker verification requires connected reference_audio")
        if not 0.0 <= float(min_speaker_similarity) <= 1.0:
            raise ValueError("min_speaker_similarity must be between 0 and 1")
        speaker_path = resolve_speaker_model_directory(speaker_model_directory)
        with _SPEAKER_LOCK:
            extractor, speaker_model, speaker_reused = _load_speaker_model(speaker_path)
            speaker_released = False
            try:
                speaker_similarity = _speaker_cosine(
                    extractor,
                    speaker_model,
                    reference_audio,
                    output,
                )
            finally:
                if unload_speaker_after_verify:
                    speaker_released = _release_speaker_model()
        speaker_accepted = speaker_similarity >= float(min_speaker_similarity)
        speaker_report = {
            "status": (
                "threshold_pass" if speaker_accepted else "threshold_fail"
            ),
            "mode": speaker_check_mode,
            "cosine_similarity": speaker_similarity,
            "threshold": float(min_speaker_similarity),
            "threshold_is_dataset_dependent": True,
            "model": {
                "engine": "transformers.WavLMForXVector",
                "device": "cpu",
                "model_directory": str(speaker_path),
                "model_reused": speaker_reused,
            },
            "unload": {
                "requested": bool(unload_speaker_after_verify),
                "released": bool(speaker_released),
                "scope": "optional_cpu_speaker_model_only",
            },
        }

    accepted = text_accepted and (
        speaker_accepted if speaker_check_mode == "require_threshold" else True
    )
    report = {
        "schema": "minimax_h3_t8_speech_verify_v1",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "peak_limit": peak_report,
        "text_verification": text_report,
        "speaker_verification": speaker_report,
        "limitations": [
            "ASR verifies spoken text but not speaker identity.",
            "Speaker cosine is model- and dataset-dependent; report_cosine does not gate acceptance.",
            "A single cosine observation is not enough to claim high-fidelity voice cloning.",
        ],
    }
    if strict and not accepted:
        raise ValueError(
            "speech verification failed: "
            f"text_similarity={text_similarity:.3f}, "
            f"speaker_similarity={speaker_similarity:.3f}"
        )
    return (
        output,
        transcript,
        text_similarity,
        speaker_similarity,
        bool(accepted),
        _json(report),
    )
