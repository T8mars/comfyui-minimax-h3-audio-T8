from __future__ import annotations

import json
from pathlib import Path

import torch

from h3_audio_t8_pkg.audio_integrity_advanced import (
    analyze_audio_integrity,
    analyze_audio_perceptual_drift,
    audit_speaker_routing,
)
from h3_audio_t8_pkg.speech import make_dialogue_plan, make_voice_profile


def _tone(seconds=1.0, sample_rate=32000, frequency=220.0, amplitude=0.1):
    samples = round(seconds * sample_rate)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = (amplitude * torch.sin(2.0 * torch.pi * frequency * time)).view(1, 1, -1)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _reference_profile(speaker_id, description, frequency):
    return make_voice_profile(
        "reference_voice",
        speaker_id,
        description,
        "Chinese",
        True,
        _tone(2.2, frequency=frequency),
        highpass_60hz=False,
        peak_limit_minus_3_dbfs=False,
    )[0]


def test_clean_audio_passes_and_reports_exact_av_boundary_without_mutation():
    audio = _tone(seconds=124 / 24)
    output, passed, decision, samples, duration, delta_ms, report_json = analyze_audio_integrity(
        audio, video_frame_count=124, fps=24.0
    )
    report = json.loads(report_json)
    assert output is audio
    assert passed is True and decision == "PASS"
    assert samples == round(124 / 24 * 32000)
    assert duration == samples / 32000
    assert abs(delta_ms) < 0.02
    assert report["audio_mutated"] is False
    assert report["checks"]["audio_video_boundary"]["delta_samples"] == 0


def test_audio_integrity_abstains_on_pop_dc_clipping_and_boundary_mismatch():
    audio = _tone(seconds=1.0)
    waveform = audio["waveform"].clone()
    waveform[..., 0] = 1.0
    waveform[..., 16000:] += 0.2
    waveform[..., 20000:20100] = 1.0
    audio = {"waveform": waveform, "sample_rate": 32000}
    _, passed, decision, _, _, _, report_json = analyze_audio_integrity(
        audio, video_frame_count=48, fps=24.0
    )
    codes = {item["code"] for item in json.loads(report_json)["findings"]}
    assert passed is False and decision == "ABSTAIN"
    assert "suspected_opening_pop_or_cut" in codes
    assert "suspected_dc_jump" in codes
    assert "clipping_ratio_exceeded" in codes
    assert "audio_video_boundary_mismatch" in codes


def test_speech_like_short_window_mean_changes_do_not_masquerade_as_dc_offset():
    sample_rate = 32000
    samples = sample_rate * 3
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    carrier = torch.sin(2.0 * torch.pi * 173.0 * time)
    syllable_envelope = 0.08 + 0.22 * torch.sin(2.0 * torch.pi * 3.7 * time).square()
    waveform = (carrier * syllable_envelope).view(1, 1, -1)
    _, passed, decision, *_rest, report_json = analyze_audio_integrity(
        {"waveform": waveform, "sample_rate": sample_rate}
    )
    report = json.loads(report_json)
    assert passed is True and decision == "PASS"
    assert report["checks"]["dc_discontinuity"]["suspected"] is False
    assert report["checks"]["dc_discontinuity"]["context_ms_per_side"] >= 100.0


def test_audio_integrity_flags_exact_tail_to_head_copy_as_suspicious():
    audio = _tone(seconds=1.0, frequency=237.0)
    waveform = audio["waveform"].clone()
    count = round(0.25 * 32000)
    waveform[..., -count:] = waveform[..., :count]
    audio = {"waveform": waveform, "sample_rate": 32000}
    _, passed, decision, _, _, _, report_json = analyze_audio_integrity(audio)
    report = json.loads(report_json)
    assert passed is False and decision == "ABSTAIN"
    assert report["checks"]["tail_to_head_similarity"]["suspected"] is True


def _aligned_reference_and_drift_candidate(sample_rate=32000):
    seconds = 5.0
    samples = round(seconds * sample_rate)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    reference = (
        0.14 * torch.sin(2.0 * torch.pi * 220.0 * time)
        + 0.07 * torch.sin(2.0 * torch.pi * 880.0 * time)
        + 0.04 * torch.sin(2.0 * torch.pi * 3000.0 * time)
    )
    candidate = reference.clone()
    start = round(1.4 * sample_rate)
    end = round(3.5 * sample_rate)
    candidate[start:end] = 0.05 * torch.sin(
        2.0 * torch.pi * 220.0 * time[start:end]
    )
    return (
        {"waveform": reference.view(1, 1, -1), "sample_rate": sample_rate},
        {"waveform": candidate.view(1, 1, -1), "sample_rate": sample_rate},
    )


def test_perceptual_drift_audit_finds_persistent_reference_relative_change():
    reference, candidate = _aligned_reference_and_drift_candidate()
    output, passed, decision, max_spectral, max_level, start, end, report_json = (
        analyze_audio_perceptual_drift(reference, candidate)
    )
    report = json.loads(report_json)
    assert output is candidate
    assert passed is False and decision == "ABSTAIN"
    assert max_spectral >= 0.30
    assert max_level >= 4.0
    assert 0.8 <= start <= 1.8
    assert 3.0 <= end <= 4.1
    assert report["candidate_audio_mutated"] is False
    assert {
        item["code"] for item in report["findings"]
    } == {"reference_relative_acoustic_drift"}


def test_perceptual_drift_audit_accepts_identical_and_mild_blend():
    reference, candidate = _aligned_reference_and_drift_candidate()
    identical = analyze_audio_perceptual_drift(reference, reference)
    assert identical[1:3] == (True, "PASS")
    assert identical[3:7] == (0.0, 0.0, -1.0, -1.0)

    blend_waveform = reference["waveform"] * 0.9
    blend = {"waveform": blend_waveform, "sample_rate": 32000}
    result = analyze_audio_perceptual_drift(reference, blend)
    assert result[1:3] == (True, "PASS")
    assert json.loads(result[-1])["checks"]["reference_relative_acoustic_drift"][
        "sections"
    ] == []


def test_perceptual_drift_audit_abstains_without_silent_resampling():
    reference = _tone(seconds=2.0, sample_rate=32000)
    candidate = _tone(seconds=2.0, sample_rate=24000)
    result = analyze_audio_perceptual_drift(reference, candidate)
    report = json.loads(result[-1])
    assert result[1:3] == (False, "ABSTAIN")
    assert {item["code"] for item in report["findings"]} == {"sample_rate_mismatch"}
    assert report["checks"]["reference_relative_acoustic_drift"]["evaluated"] is False


def test_speaker_routing_compiles_unique_reference_audio_ordinals():
    profile_a = _reference_profile("actor_a", "an adult female voice, low and calm", 210.0)
    profile_b = _reference_profile("actor_b", "an adult female voice, bright and quick", 310.0)
    plan, _ = make_dialogue_plan(
        "actor_a: 你在干嘛呢\nactor_b: 我在这里呀",
        "speaker_lines",
        {"profile_0": profile_a, "profile_1": profile_b},
        "Chinese",
        "close",
    )
    output, passed, decision, binding_json, report_json = audit_speaker_routing(plan)
    bindings = json.loads(binding_json)["bindings"]
    assert output is plan
    assert passed is True and decision == "PASS"
    assert [item["audio_tag"] for item in bindings] == ["<Audio 1>", "<Audio 2>"]
    assert json.loads(report_json)["plan_mutated"] is False


def test_speaker_routing_abstains_on_duplicate_reference_and_vocalization():
    profile_a = _reference_profile("actor_a", "an adult female voice", 240.0)
    profile_b = dict(profile_a)
    profile_b["speaker_id"] = "actor_b"
    plan, _ = make_dialogue_plan(
        "actor_a: （喘息）你在干嘛呢\nactor_b: 哈哈哈哈",
        "speaker_lines",
        {"profile_0": profile_a, "profile_1": profile_b},
        "Chinese",
        "close",
    )
    _, passed, decision, _, report_json = audit_speaker_routing(plan)
    codes = {item["code"] for item in json.loads(report_json)["findings"]}
    assert passed is False and decision == "ABSTAIN"
    assert "duplicate_reference_audio_assignment" in codes
    assert "unstructured_vocalization_in_spoken_text" in codes


def test_audio_audit_frontend_workflows_are_importable_and_documented():
    root = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "05-speech-dialogue"
    expected = {
        "2026-08-22_H3_Audio_Integrity_Audit_Advanced.json": (
            "MiniMaxH3AudioIntegrityAuditT8Advanced",
            [124, 24.0, 40.0, 250.0, 0.15, 0.02, 0.985, 0.001, 21.0],
        ),
        "2026-08-22_H3_Speaker_Routing_Audit_Advanced.json": (
            "MiniMaxH3SpeakerRoutingAuditT8Advanced",
            [True, 0.75],
        ),
        "2026-08-22_H3_Audio_Perceptual_Drift_Audit_Advanced.json": (
            "MiniMaxH3AudioPerceptualDriftAuditT8Advanced",
            [500.0, 100.0, -50.0, 0.30, 4.0, 3, 21.0],
        ),
    }
    for filename, (node_type, widgets) in expected.items():
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        audit = next(node for node in nodes.values() if node["type"] == node_type)
        assert audit["widgets_values"] == widgets
        assert "MarkdownNote" in {node["type"] for node in nodes.values()}
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] == link_type
