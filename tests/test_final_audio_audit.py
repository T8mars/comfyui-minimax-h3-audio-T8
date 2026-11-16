import json

import pytest
import torch

from h3_audio_t8_pkg import final_audio_audit as audit


def audio():
    return {
        "waveform": torch.linspace(-1.3, 1.3, 32000 * 4).reshape(1, 1, -1),
        "sample_rate": 32000,
    }


def transcript():
    return {
        "text": "你好 你好",
        "words": [
            {"word": "你好", "start": 0.1, "end": 0.5},
            {"word": "你好", "start": 2.1, "end": 2.5},
        ],
        "untimed_words": [],
    }


def test_final_pcm_is_identical_even_above_full_scale_and_ASR_gets_a_private_copy(
    monkeypatch,
):
    source = audio()
    before = audit.pcm_identity(source)
    calls = []

    def transcribe(selected, *args):
        assert (
            selected is not source
            and selected["waveform"].data_ptr() != source["waveform"].data_ptr()
        )
        calls.append(selected)
        return transcript()

    monkeypatch.setattr(audit, "transcribe_audio", transcribe)
    output, text = audit.audit_final_audio(
        source,
        "你好，你好",
        "local-existing",
        windows=[
            {"start": 0.0, "end": 2.0, "text": "你好"},
            {"start": 2.0, "end": 4.0, "text": "你好"},
        ],
    )
    assert output is source and audit.pcm_identity(source) == before
    assert float(source["waveform"].abs().max()) > 1.0
    report = json.loads(text)
    assert len(calls) == 1 and [r["heard"] for r in report["dialogue_windows"]] == [
        "你好",
        "你好",
    ]
    assert not report["gain_or_limiter_applied"] and not report["trim_applied"]
    assert not report["window_metric_aggregation"] and not report["automatic_retry"]


def test_word_on_shared_boundary_is_assigned_only_once(monkeypatch):
    monkeypatch.setattr(
        audit,
        "transcribe_audio",
        lambda *_: dict(
            text="word",
            words=[{"word": "word", "start": 1.0, "end": 3.0}],
            untimed_words=[],
        ),
    )
    _, text = audit.audit_final_audio(
        audio(),
        "word",
        "local-existing",
        windows=[
            {"start": 0.0, "end": 2.0, "text": ""},
            {"start": 2.0, "end": 4.0, "text": "word"},
        ],
    )
    assert [r["heard"] for r in json.loads(text)["dialogue_windows"]] == ["", "word"]


def test_untimed_transcript_keeps_global_estimate_but_does_not_fake_window_alignment(
    monkeypatch,
):
    monkeypatch.setattr(
        audit,
        "transcribe_audio",
        lambda *_: dict(text="hello", words=[], untimed_words=[{"word": "hello"}]),
    )
    _, text = audit.audit_final_audio(
        audio(),
        "hello",
        "local-existing",
        windows=[{"start": 0.0, "end": 4.0, "text": "hello"}],
    )
    report = json.loads(text)
    assert (
        report["global_metrics"] is not None
        and report["dialogue_windows"][0]["metrics"] is None
    )


def test_overlapping_expected_windows_are_not_summed_as_independent_failures(
    monkeypatch,
):
    monkeypatch.setattr(audit, "transcribe_audio", lambda *_: transcript())
    _, text = audit.audit_final_audio(
        audio(),
        "你好你好",
        "local-existing",
        windows=[
            {"start": 0.0, "end": 3.0, "text": "你好"},
            {"start": 2.0, "end": 4.0, "text": "你好"},
        ],
    )
    report = json.loads(text)
    assert report["overlapping_windows"] and not report["window_metric_aggregation"]


def test_mutating_or_failing_consumer_never_changes_original_audio(monkeypatch):
    source = audio()
    before = audit.pcm_identity(source)

    def mutating(value, *args):
        value["waveform"].zero_()
        return transcript()

    monkeypatch.setattr(audit, "transcribe_audio", mutating)
    with pytest.raises(ValueError, match="original was preserved"):
        audit.audit_final_audio(source, "你好", "local-existing")
    assert audit.pcm_identity(source) == before

    def failing(*args):
        raise InterruptedError("actual cancellation")

    monkeypatch.setattr(audit, "transcribe_audio", failing)
    with pytest.raises(InterruptedError, match="actual cancellation"):
        audit.audit_final_audio(source, "你好", "local-existing")
    assert audit.pcm_identity(source) == before


def test_actual_invalid_time_and_pcm_fail_before_ASR(monkeypatch):
    def forbidden(*args):
        raise AssertionError("ASR must not run")

    monkeypatch.setattr(audit, "transcribe_audio", forbidden)
    with pytest.raises(ValueError, match="actual final PCM duration"):
        audit.audit_final_audio(
            audio(),
            "hello",
            "local-existing",
            windows=[{"start": 0, "end": 5, "text": "hello"}],
        )
    source = audio()
    source["waveform"][..., 0] = float("nan")
    with pytest.raises(ValueError, match="finite samples"):
        audit.audit_final_audio(source, "hello", "local-existing")


def test_schema_has_no_mutating_peak_or_trim_controls():
    from h3_audio_t8_pkg.nodes_final_audio_audit import MiniMaxH3FinalAudioAuditEXPT8

    names = {x.id for x in MiniMaxH3FinalAudioAuditEXPT8.define_schema().inputs}
    assert not {"peak_limit_dbfs", "trim", "retry"}.intersection(names)


def test_explicit_window_mode_uses_same_model_per_window_and_restores_global_offsets(
    monkeypatch,
):
    calls = []

    def transcribe(value, model, *args):
        calls.append((model, value["waveform"].shape[-1]))
        return {
            "text": "hello",
            "words": [{"word": "hello", "start": 0.1, "end": 0.5}],
            "untimed_words": [],
        }

    monkeypatch.setattr(audit, "transcribe_audio", transcribe)
    source = audio()
    identity = audit.pcm_identity(source)
    # User list order does not dictate playback chronology.
    output, text = audit.audit_final_audio(
        source,
        "hello hello",
        "same-existing-model",
        windows=[
            {"start": 2.0, "end": 4.0, "text": "hello"},
            {"start": 0.0, "end": 2.0, "text": "hello"},
        ],
        transcription_scope="explicit_final_windows",
    )
    report = json.loads(text)
    assert calls == [("same-existing-model", 64000)] * 2
    assert (
        report["ASR_request_count"] == 2
        and report["transcript"]["text"] == "hello hello"
    )
    assert sorted(w["start"] for w in report["transcript"]["words"]) == [0.1, 2.1]
    assert output is source and audit.pcm_identity(output) == identity
    assert [r["heard"] for r in report["dialogue_windows"]] == ["hello", "hello"]
    assert report["global_metrics"]["primary_error_rate"] == 0.0


def test_window_mode_requires_windows_and_cancellation_leaves_pcm_unchanged(
    monkeypatch,
):
    source = audio()
    before = audit.pcm_identity(source)
    with pytest.raises(ValueError, match="needs explicit"):
        audit.audit_final_audio(
            source, "", "same", transcription_scope="explicit_final_windows"
        )

    def cancel(*args):
        raise InterruptedError("window cancelled")

    monkeypatch.setattr(audit, "transcribe_audio", cancel)
    with pytest.raises(InterruptedError, match="window cancelled"):
        audit.audit_final_audio(
            source,
            "hello",
            "same",
            windows=[{"start": 0.0, "end": 2.0, "text": "hello"}],
            transcription_scope="explicit_final_windows",
        )
    assert audit.pcm_identity(source) == before
