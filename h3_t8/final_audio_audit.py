"""ASR on the caller's retained final PCM, without gain/trim/retry side effects."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from .core import validate_audio
from .speech_verification import (
    transcribe_audio,
    transcript_metrics,
    normalized_asr_units,
    _release_asr_model,
)


def pcm_identity(audio):
    waveform = audio["waveform"].detach().cpu().contiguous()
    return {
        "sample_rate": audio["sample_rate"],
        "shape": list(waveform.shape),
        "dtype": str(waveform.dtype),
        "sha256": hashlib.sha256(
            waveform.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def audit_final_audio(
    audio,
    expected_text,
    model_directory,
    language="auto",
    windows=(),
    beam_size=5,
    cpu_threads=2,
    unload_after=True,
    cancel_check=None,
    transcription_scope="whole_final_audio",
):
    waveform, rate = validate_audio(audio, "final retained audio")
    if rate <= 0 or not torch.isfinite(waveform).all():
        raise ValueError(
            "Final retained PCM requires a positive sample rate and finite samples"
        )
    if not isinstance(expected_text, str):
        raise TypeError("Expected dialogue must be text")
    if not isinstance(windows, (list, tuple)):
        raise ValueError("Final-time dialogue windows must be a list")
    duration = audio["waveform"].shape[-1] / audio["sample_rate"]
    parsed = []
    for row in windows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("Each final-time window needs start, end and text")
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Each dialogue window needs finite start and end seconds"
            ) from error
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end <= duration
        ):
            raise ValueError(
                "Dialogue windows must lie within the actual final PCM duration"
            )
        parsed.append({"start": start, "end": end, "text": row["text"]})
    before = pcm_identity(audio)
    if transcription_scope not in ("whole_final_audio", "explicit_final_windows"):
        raise ValueError("Choose whole final audio or explicit final-time windows")
    if transcription_scope == "explicit_final_windows" and not parsed:
        raise ValueError("Windowed ASR needs explicit final-time dialogue windows")

    def transcribe(selected, release):
        identity = pcm_identity(selected)
        asr_input = dict(selected)
        asr_input["waveform"] = selected["waveform"].detach().clone()
        value = transcribe_audio(
            asr_input,
            model_directory,
            language,
            beam_size,
            cpu_threads,
            release,
            cancel_check,
        )
        if pcm_identity(asr_input) != identity:
            raise ValueError(
                "ASR consumer changed PCM; the caller's original was preserved in a separate buffer"
            )
        return value

    window_results = []
    if transcription_scope == "whole_final_audio":
        result = transcribe(audio, unload_after)
    else:
        try:
            for index, row in enumerate(parsed):
                start, end = round(row["start"] * rate), round(row["end"] * rate)
                if end <= start:
                    raise ValueError(
                        "A final-time ASR window contains no PCM samples after endpoint rounding"
                    )
                selected = {**audio, "waveform": waveform[..., start:end]}
                value = transcribe(selected, False)
                window_results.append(
                    {
                        **value,
                        "source_start_sample": start,
                        "source_end_sample": end,
                        "time_origin": start / rate,
                    }
                )
            result = {
                "text": " ".join(
                    x["text"]
                    for x in sorted(window_results, key=lambda x: x["time_origin"])
                ),
                "words": [],
                "untimed_words": [],
                "window_transcripts": window_results,
            }
            for value in window_results:
                result["words"].extend(
                    {
                        **w,
                        "start": w["start"] + value["time_origin"],
                        "end": w["end"] + value["time_origin"],
                    }
                    for w in value.get("words", [])
                )
                result["untimed_words"].extend(value.get("untimed_words", []))
        finally:
            if unload_after:
                _release_asr_model()
    if pcm_identity(audio) != before:
        raise ValueError(
            "ASR consumer changed PCM; the caller's original was preserved in a separate buffer"
        )
    words = result.get("words", [])
    timed = all(
        isinstance(w, dict)
        and isinstance(w.get("word"), str)
        and isinstance(w.get("start"), (int, float))
        and isinstance(w.get("end"), (int, float))
        and math.isfinite(w["start"])
        and math.isfinite(w["end"])
        and 0 <= w["start"] <= w["end"]
        for w in words
    )
    has_untimed = bool(result.get("untimed_words")) or not timed
    per_window = []
    for index, row in enumerate(parsed):
        expected_units = normalized_asr_units(row["text"])
        heard = (
            " ".join(
                w["word"]
                for w in words
                if row["start"] <= (w["start"] + w["end"]) / 2 < row["end"]
            )
            if timed
            else ""
        )
        if window_results:
            heard = window_results[index]["text"]
        per_window.append(
            {
                **row,
                "heard": heard,
                "status": "untimed_words_present"
                if has_untimed
                else "timed_ASR_estimate"
                if expected_units
                else "no_expected_transcript",
                "metrics": None
                if has_untimed or not expected_units
                else transcript_metrics(row["text"], heard),
            }
        )
    ordered = sorted(parsed, key=lambda row: row["start"])
    overlapping = any(b["start"] < a["end"] for a, b in zip(ordered, ordered[1:]))
    report = {
        "schema": "t8.final_audio_audit.v1",
        "status": "offline_ASR_estimate",
        "PCM": before,
        "PCM_object_preserved": True,
        "gain_or_limiter_applied": False,
        "trim_applied": False,
        "duration_seconds": duration,
        "transcript": result,
        "transcription_scope": transcription_scope,
        "ASR_request_count": len(window_results) if window_results else 1,
        "global_metrics": transcript_metrics(expected_text, result["text"])
        if normalized_asr_units(expected_text) and not (window_results and overlapping)
        else None,
        "dialogue_windows": per_window,
        "window_metric_aggregation": False,
        "overlapping_windows": overlapping,
        "discarded_generation_overlaps_scored": False,
        "role": "Caller-connected final retained PCM only; source/drive/voice references are not substituted",
        "automatic_retry": False,
        "human_listening_accepted": False,
        "calibration_note": "ASR may omit repeated dialogue or use different Chinese simplified/traditional characters. "
        "These are recognition estimates, not generation-failure or automatic-retry decisions. "
        "Explicit window mode uses the same selected local model, once per supplied final-time window.",
    }
    return audio, json.dumps(report, ensure_ascii=False)
