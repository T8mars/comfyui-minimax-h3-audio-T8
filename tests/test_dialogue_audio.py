from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.dialogue_audio import build_dialogue_safe_master
from h3_audio_t8_pkg import speech_verification
from h3_audio_t8_pkg.speech_verification import (
    analyze_dialogue_boundary,
    exact_target_word_spans,
)
from h3_audio_t8_pkg.timed_audio_latent import build_timed_audio_bed_lock
from helpers import FakeAudioVAE


def make_audio(seconds: float, sample_rate: int = 32000, value: float = 0.1):
    samples = round(seconds * sample_rate)
    return {
        "waveform": torch.full((1, 2, samples), value, dtype=torch.float32),
        "sample_rate": sample_rate,
    }


def word_rows(text: str, start: float = 0.0, step: float = 0.25):
    rows = []
    cursor = start
    for word in text.split():
        rows.append({"start": cursor, "end": cursor + step * 0.8, "word": word})
        cursor += step
    return rows


def test_exact_target_spans_reports_repetition_instead_of_guessing():
    words = word_rows("hello world pause hello world")
    spans = exact_target_word_spans(words, "hello world")
    assert len(spans) == 2
    assert spans[0]["start_word_index"] == 0
    assert spans[1]["start_word_index"] == 3


def test_boundary_analyzer_reports_unique_target_and_tail_extras_without_editing(monkeypatch):
    expected = "the door is open"
    words = word_rows("noise the door is open muttering continues", start=0.1)
    original = make_audio(3.0, value=0.2)
    original_ptr = original["waveform"].data_ptr()

    monkeypatch.setattr(
        speech_verification,
        "resolve_asr_model_directory",
        lambda value: Path("X:/verified-model"),
    )
    monkeypatch.setattr(
        speech_verification,
        "_load_asr_model",
        lambda path, threads: (object(), False),
    )
    monkeypatch.setattr(
        speech_verification,
        "_transcribe",
        lambda model, audio, language, beam_size: {
            "text": "noise the door is open muttering continues",
            "language": "en",
            "language_probability": 1.0,
            "duration_seconds": 3.0,
            "words": words,
        },
    )
    monkeypatch.setattr(speech_verification, "_release_asr_model", lambda: True)

    result = analyze_dialogue_boundary(
        original,
        expected,
        "verified-model",
        "English",
    )
    transcript, unique, clean, start, end, before, after, report_json = result
    report = json.loads(report_json)

    assert transcript.startswith("noise")
    assert unique is True
    assert clean is False
    assert start == pytest.approx(words[1]["start"])
    assert end == pytest.approx(words[4]["end"])
    assert before == 1
    assert after == 2
    assert report["status"] == "unique_target_with_lexical_extras"
    assert report["tail_signal_activity"]["is_not_a_speech_classifier"] is True
    assert report["claims"]["audio_was_modified"] is False
    assert original["waveform"].data_ptr() == original_ptr


def test_boundary_analyzer_rejects_ambiguous_repeated_target(monkeypatch):
    words = word_rows("hello world hello world")
    monkeypatch.setattr(
        speech_verification,
        "resolve_asr_model_directory",
        lambda value: Path("X:/verified-model"),
    )
    monkeypatch.setattr(
        speech_verification,
        "_load_asr_model",
        lambda path, threads: (object(), False),
    )
    monkeypatch.setattr(
        speech_verification,
        "_transcribe",
        lambda model, audio, language, beam_size: {
            "text": "hello world hello world",
            "language": "en",
            "language_probability": 1.0,
            "duration_seconds": 2.0,
            "words": words,
        },
    )
    monkeypatch.setattr(speech_verification, "_release_asr_model", lambda: True)

    result = analyze_dialogue_boundary(
        make_audio(2.0),
        "hello world",
        "verified-model",
        "English",
    )
    assert result[1] is False
    assert result[2] is False
    assert result[3] == result[4] == 0.0
    assert json.loads(result[-1])["status"] == "ambiguous_multiple_exact_targets"


def test_safe_master_preserves_background_after_verified_speech_ends():
    speech = make_audio(1.0, value=0.2)
    music = make_audio(5.0, value=0.05)
    ambience = make_audio(5.0, value=0.02)
    sfx_wave = torch.zeros((1, 2, 5 * 32000), dtype=torch.float32)
    sfx_wave[..., 4 * 32000 : 4 * 32000 + 200] = 0.3
    sfx = {"waveform": sfx_wave, "sample_rate": 32000}

    master, speech_stem, background, report_json = build_dialogue_safe_master(
        speech,
        True,
        5.0,
        music_audio=music,
        ambience_audio=ambience,
        sfx_audio=sfx,
    )
    report = json.loads(report_json)

    assert master["waveform"].shape[-1] == 5 * 32000
    assert torch.count_nonzero(speech_stem["waveform"][..., 32000:]) == 0
    assert torch.all(background["waveform"][..., 2 * 32000 : 3 * 32000] != 0)
    assert float(background["waveform"][..., 4 * 32000 :].abs().max()) > 0.3
    assert torch.equal(master["waveform"], speech_stem["waveform"] + background["waveform"])
    assert report["speech"]["action"] == "placed_without_trimming_or_time_stretch"
    assert report["claims"]["master_tail_was_truncated_at_dialogue_end"] is False
    assert report["exact_sample_error"] == 0


def test_safe_master_refuses_unverified_speech_and_overflow():
    with pytest.raises(ValueError, match="requires a true accepted signal"):
        build_dialogue_safe_master(make_audio(1.0), False, 5.0)
    with pytest.raises(ValueError, match="beyond target master"):
        build_dialogue_safe_master(make_audio(5.0), True, 5.0, speech_start_seconds=0.1)


def test_safe_master_never_silently_fits_background_stems():
    with pytest.raises(ValueError, match="choose an explicit fit policy"):
        build_dialogue_safe_master(
            make_audio(1.0),
            True,
            5.0,
            music_audio=make_audio(2.0),
        )

    master, _speech, background, report_json = build_dialogue_safe_master(
        make_audio(1.0),
        True,
        5.0,
        music_fit_policy="pad_or_trim",
        music_audio=make_audio(2.0, value=0.1),
    )
    report = json.loads(report_json)
    assert master["waveform"].shape[-1] == 5 * 32000
    assert torch.count_nonzero(background["waveform"][..., 2 * 32000 :]) == 0
    assert report["beds"]["music"]["action"].startswith("padded_")


def test_safe_master_loop_policy_is_explicit_and_sample_exact():
    master, speech, background, report_json = build_dialogue_safe_master(
        make_audio(0.5, value=0.1),
        True,
        3.0,
        music_fit_policy="loop_crossfade",
        loop_crossfade_seconds=0.1,
        music_audio=make_audio(1.0, value=0.02),
    )
    report = json.loads(report_json)
    assert master["waveform"].shape == speech["waveform"].shape == background["waveform"].shape
    assert master["waveform"].shape[-1] == 96000
    assert report["beds"]["music"]["action"] == "looped_with_crossfade"
    assert report["beds"]["music"]["repeat_count"] > 1


def make_av_latent(audio_t: int = 207):
    video = torch.randn((1, 24, 37, 4, 4))
    audio = torch.randn((1, 32, 2, audio_t))
    video_mask = torch.full_like(video, 0.6)
    audio_mask = torch.full_like(audio, 0.5)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        "kept_metadata": "yes",
    }


def test_timed_bed_lock_preserves_video_and_locks_tail_without_guessing():
    source = make_av_latent()
    source_video, _source_audio = source["samples"].unbind()
    source_video_mask, _source_audio_mask = source["noise_mask"].unbind()
    # FakeAudioVAE emits one latent step per 800 samples. 207 steps is exact.
    bed = make_audio(207 * 800 / 32000, value=0.03)

    output, audio_output, report_json = build_timed_audio_bed_lock(
        source,
        bed,
        FakeAudioVAE(),
        tail_lock_start_seconds=2.001,
    )
    out_video, out_audio = output["samples"].unbind()
    out_video_mask, out_audio_mask = output["noise_mask"].unbind()
    report = json.loads(report_json)

    assert out_video.data_ptr() == source_video.data_ptr()
    assert torch.equal(out_video_mask, source_video_mask)
    assert torch.all(out_audio == 0.25)
    # ceil(2.001 * 40) = 81: the boundary is never quantized earlier.
    assert torch.all(out_audio_mask[..., :81] == 0.5)
    assert torch.all(out_audio_mask[..., 81:] == 0.0)
    assert torch.equal(audio_output["samples"], out_audio)
    assert output["kept_metadata"] == "yes"
    assert report["facts"]["requested_start_step_ceil"] == 81
    assert report["facts"]["existing_audio_mask_was_cap"] is True
    assert report["claims"]["source_separation_performed"] is False
    assert report["claims"]["decoded_tail_quality_verified"] is False


def test_timed_bed_lock_strict_default_rejects_duration_mismatch():
    with pytest.raises(ValueError, match="different H3 latent duration"):
        build_timed_audio_bed_lock(
            make_av_latent(),
            make_audio(1.0),
            FakeAudioVAE(),
            tail_lock_start_seconds=0.5,
        )


def test_timed_bed_lock_explicit_fit_and_transition_are_reported():
    source = make_av_latent()
    source.pop("noise_mask")
    output, _audio_output, report_json = build_timed_audio_bed_lock(
        source,
        make_audio(1.0),
        FakeAudioVAE(),
        tail_lock_start_seconds=0.5,
        transition_seconds=0.05,
        audio_latent_fit_policy="fit_reported",
    )
    _video_mask, audio_mask = output["noise_mask"].unbind()
    report = json.loads(report_json)

    assert report["facts"]["audio_latent_fit_action"].startswith("zero_padded_")
    assert report["facts"]["transition_steps"] == 2
    assert report["facts"]["fully_locked_start_step"] == 22
    assert torch.all(audio_mask[..., :20] == 1.0)
    assert torch.all(audio_mask[..., 20] == 1.0)
    assert torch.all(audio_mask[..., 21] == 0.5)
    assert torch.all(audio_mask[..., 22:] == 0.0)


def test_timed_bed_lock_never_increases_tail_freedom():
    with pytest.raises(ValueError, match="cannot exceed"):
        build_timed_audio_bed_lock(
            make_av_latent(),
            make_audio(207 * 800 / 32000),
            FakeAudioVAE(),
            tail_lock_start_seconds=1.0,
            head_denoise_strength=0.2,
            tail_denoise_strength=0.3,
        )
