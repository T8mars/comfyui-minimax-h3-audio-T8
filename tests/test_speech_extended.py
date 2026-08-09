from __future__ import annotations

import json

import torch

from h3_audio_t8_pkg.speech import make_speech_plan, make_voice_profile
from h3_audio_t8_pkg.speech_extended import (
    accept_longform_segment,
    apply_performance_direction,
    compose_longform_session,
    control_longform_session,
    delete_voice_profile,
    fit_audio_for_adr,
    load_voice_profile,
    save_voice_profile,
    speech_manifest_fingerprint,
    start_or_resume_longform,
)
from h3_audio_t8_pkg.speech_reliability import SpeechGuardRegistry
from h3_audio_t8_pkg.speech_verification import transcript_metrics


def _audio(seconds=1.0, sample_rate=32000, frequency=220.0):
    samples = round(seconds * sample_rate)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = (0.1 * torch.sin(2.0 * torch.pi * frequency * time)).view(1, 1, -1)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _plan(text="第一段测试。第二段测试。"):
    profile, _, _ = make_voice_profile(
        "described_voice",
        "speaker_1",
        "a calm adult voice",
        "Chinese",
        False,
    )
    plan, _ = make_speech_plan(
        text,
        profile,
        "Chinese",
        "natural",
        "neutral",
        0.5,
        "close",
        "language_aware",
        5,
        8,
    )
    assert len(plan["segments"]) == 2
    return profile, plan


class _Runtime:
    def __init__(self):
        self.releases = []

    def request_release(self, policy):
        self.releases.append(policy)


def test_prompt_guard_releases_only_when_finalize_was_skipped():
    events = []
    runtime = _Runtime()
    registry = SpeechGuardRegistry(events.append)
    guard = registry.arm("prompt-a", "unload_all_models", runtime)
    event = registry.on_prompt_end("prompt-a")
    assert runtime.releases == ["unload_all_models"]
    assert event["release_requested"] is True
    assert events[-1]["scope"] == "global_comfyui_models"

    guard = registry.arm("prompt-b", "clear_execution_cache", runtime)
    assert registry.complete(guard)["completed"] is True
    assert registry.on_prompt_end("prompt-b") is None
    assert runtime.releases == ["unload_all_models"]


def test_voice_library_is_explicit_exact_and_recoverable(monkeypatch, tmp_path):
    from h3_audio_t8_pkg import speech_extended

    monkeypatch.setattr(speech_extended.folder_paths, "get_user_directory", lambda: str(tmp_path))
    reference = _audio(2.2)
    profile, _, _ = make_voice_profile(
        "reference_voice",
        "speaker_ref",
        "a clear adult voice",
        "Chinese",
        True,
        reference,
    )
    saved, report = save_voice_profile(profile, "consented_voice")
    loaded, load_report = load_voice_profile("consented_voice")
    assert saved["reference_sha256"] == loaded["reference_sha256"]
    assert torch.equal(saved["reference_audio"]["waveform"], loaded["reference_audio"]["waveform"])
    assert json.loads(report)["consent_record"]["rights_confirmed"] is True
    assert json.loads(load_report)["operation"] == "loaded"
    deleted = json.loads(delete_voice_profile("consented_voice", True))
    assert deleted["permanently_deleted"] is False
    assert ".trash" in deleted["recoverable_path"]


def test_performance_direction_is_labelled_uncalibrated():
    _, plan = _plan()
    output, report = apply_performance_direction(
        plan, -1, "tender", 0.7, "slow", "low", "restrained", "one soft sigh"
    )
    assert all(segment["performance_controls"]["control_kind"] == "uncalibrated_prompt_direction" for segment in output["segments"])
    assert json.loads(report)["control_kind"] == "uncalibrated_prompt_direction"


def test_adr_fit_is_exact_to_one_sample_and_refuses_unsafe_rate():
    fitted, report = fit_audio_for_adr(_audio(1.0), 1.05, "safe_time_stretch", 0.9, 1.1, 0.0)
    assert fitted["waveform"].shape[-1] == round(1.05 * 32000)
    assert json.loads(report)["exact_sample_error"] == 0
    try:
        fit_audio_for_adr(_audio(1.0), 2.0, "safe_time_stretch", 0.9, 1.1, 0.0)
    except ValueError as error:
        assert "outside the explicit safe range" in str(error)
    else:
        raise AssertionError("unsafe ADR rate should be rejected")


def test_longform_manifest_resume_cancel_and_compose(monkeypatch, tmp_path):
    from h3_audio_t8_pkg import speech_extended

    monkeypatch.setattr(speech_extended.folder_paths, "get_output_directory", lambda: str(tmp_path))
    _, plan = _plan()
    session, _, index, text, report = start_or_resume_longform(plan, "job-a")
    assert index == 0 and text == plan["segments"][0]["text"]
    assert json.loads(report)["accepted_segments"] == 0
    initial_fingerprint = speech_manifest_fingerprint("job-a")

    chunk = _audio(1.0)
    accept_longform_segment(session, plan, 0, chunk, text, 1.0, 0.0, True)
    assert speech_manifest_fingerprint("job-a") != initial_fingerprint
    resumed, _, index, _, _ = start_or_resume_longform(plan, "job-a")
    assert index == 1
    cancelled = json.loads(control_longform_session("job-a", "request_cancel"))
    assert cancelled["cancel_requested"] is True
    cancelled_session, _, index, _, _ = start_or_resume_longform(plan, "job-a")
    assert cancelled_session["state"] == "cancelled" and index == -1
    control_longform_session("job-a", "clear_cancel")
    resumed, _, index, _, _ = start_or_resume_longform(plan, "job-a")
    accept_longform_segment(resumed, plan, index, chunk, plan["segments"][index]["text"], 1.0, 0.0, True)
    complete, _, index, _, _ = start_or_resume_longform(plan, "job-a")
    assert complete["state"] == "complete" and index == -1
    audio, timeline, srt, vtt = compose_longform_session(plan, "job-a", 0.0, -1.0)
    assert audio["waveform"].shape[-1] == 64000
    assert json.loads(timeline)["longform"]["segment_hashes_verified"] is True
    assert "第一段测试" in srt and vtt.startswith("WEBVTT")


def test_multilingual_metrics_reports_cer_and_wer_explicitly():
    chinese = transcript_metrics("你好世界", "你好世间")
    english = transcript_metrics("hello world", "hello there")
    assert chinese["primary_metric"] == "CER"
    assert chinese["character_error_rate"] == 0.25
    assert english["primary_metric"] == "WER"
    assert english["primary_error_rate"] == 0.5
