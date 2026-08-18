from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.speech_verification as speech_verification
from h3_audio_t8_pkg.nodes_speech_exp import MiniMaxH3SpeechStudioT8
from h3_audio_t8_pkg.speech import (
    assemble_speech_audio,
    build_speech_conditioning,
    decode_speech_audio,
    make_dialogue_plan,
    make_speech_plan,
    make_voice_profile,
    render_frame_count,
    select_dialogue_turn,
    split_speech_text,
    validate_spoken_text,
)
from h3_audio_t8_pkg.speech_verification import (
    exact_target_word_bounds,
    transcript_metrics,
    verify_speech_audio,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def sine_audio(seconds=6.0, sample_rate=48000, channels=1, frequency=220.0):
    samples = round(seconds * sample_rate)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = 0.2 * torch.sin(2.0 * math.pi * frequency * time)
    return {
        "waveform": waveform.view(1, 1, -1).expand(1, channels, -1).clone(),
        "sample_rate": sample_rate,
    }


def described_profile(speaker_id="speaker_a"):
    profile, prepared, report = make_voice_profile(
        "described_voice",
        speaker_id,
        "a warm adult voice with clear natural diction",
        "English",
        False,
    )
    assert prepared is None
    assert json.loads(report)["mode"] == "described_voice"
    return profile


def reference_profile(speaker_id="speaker_a"):
    profile, prepared, report = make_voice_profile(
        "reference_voice",
        speaker_id,
        "the same speaker as the connected licensed reference",
        "English",
        True,
        sine_audio(),
        0.5,
        4.0,
        True,
        True,
    )
    return profile, prepared, json.loads(report)


def one_segment_plan(profile, text="The lantern is still burning.", language="English"):
    plan, _ = make_speech_plan(
        text,
        profile,
        language,
        "quiet and sincere",
        "tender",
        0.6,
        "studio",
        "single_segment",
        18,
        24,
    )
    return plan


def test_reference_profile_requires_rights_and_prepares_bounded_h3_audio():
    with pytest.raises(ValueError, match="right and consent"):
        make_voice_profile(
            "reference_voice",
            "speaker",
            "a real adult speaker",
            "English",
            False,
            sine_audio(),
        )

    profile, prepared, report = reference_profile()
    assert profile["mode"] == "reference_voice"
    assert profile["rights_confirmed"] is True
    assert prepared["sample_rate"] == 32000
    assert prepared["waveform"].shape == (1, 2, 4 * 32000)
    assert len(profile["reference_sha256"]) == 64
    assert report["reference"]["selected_start_seconds"] == 0.5
    assert report["limitations"]


def test_reference_profile_rejects_too_short_crop_and_nonfinite_is_sanitized():
    with pytest.raises(ValueError, match="at least 2.0 seconds"):
        make_voice_profile(
            "reference_voice",
            "speaker",
            "licensed voice",
            "English",
            True,
            sine_audio(seconds=1.5),
        )

    audio = sine_audio(seconds=2.1, sample_rate=32000)
    audio["waveform"][..., 10] = float("nan")
    _, prepared, report = make_voice_profile(
        "reference_voice",
        "speaker",
        "licensed voice",
        "English",
        True,
        audio,
        highpass_60hz=False,
        peak_limit_minus_3_dbfs=False,
    )
    assert torch.isfinite(prepared["waveform"]).all()
    assert report


def test_reference_crop_never_reports_or_selects_past_connected_audio():
    profile, prepared, report = make_voice_profile(
        "reference_voice",
        "speaker",
        "licensed voice",
        "English",
        True,
        sine_audio(seconds=3.0, sample_rate=32000),
        reference_start_seconds=0.5,
        reference_duration_seconds=15.0,
        highpass_60hz=False,
        peak_limit_minus_3_dbfs=False,
    )
    assert profile["reference_facts"]["duration_seconds"] == pytest.approx(2.5)
    assert prepared["waveform"].shape[-1] == 2.5 * 32000
    assert json.loads(report)["reference"]["requested_duration_seconds"] == 15.0


def test_asr_metrics_and_exact_target_bounds_do_not_fuzzy_guess():
    expected = "The door is open now. I'll wait beside the window."
    words = [
        {"start": 0.0, "end": 0.4, "word": "noise"},
        {"start": 5.78, "end": 5.98, "word": "The"},
        {"start": 5.98, "end": 6.20, "word": "door"},
        {"start": 6.20, "end": 6.40, "word": "is"},
        {"start": 6.40, "end": 6.70, "word": "open"},
        {"start": 6.70, "end": 7.50, "word": "now"},
        {"start": 8.84, "end": 9.00, "word": "I'll"},
        {"start": 9.00, "end": 9.20, "word": "wait"},
        {"start": 9.20, "end": 9.45, "word": "beside"},
        {"start": 9.45, "end": 9.60, "word": "the"},
        {"start": 9.60, "end": 9.96, "word": "window"},
    ]
    assert exact_target_word_bounds(words, expected) == (5.78, 9.96)
    assert exact_target_word_bounds(words, "The door was open") is None
    assert transcript_metrics(expected, expected)["normalized_similarity"] == 1.0
    contaminated = "unrequested words " + expected
    assert transcript_metrics(expected, contaminated)["word_or_character_error_rate"] > 0


def test_asr_exact_target_mode_trims_and_retranscribes(monkeypatch):
    expected = "The door is open now. I'll wait beside the window."
    raw_words = [
        {"start": 0.0, "end": 0.4, "word": "noise"},
        *[
            {"start": 5.78 + index * 0.3, "end": 6.04 + index * 0.3, "word": word}
            for index, word in enumerate(
                ["The", "door", "is", "open", "now", "I'll", "wait", "beside", "the", "window"]
            )
        ],
    ]
    clean_words = raw_words[1:]
    calls = []

    def fake_transcribe(model, audio, language, beam_size):
        calls.append(audio["waveform"].shape[-1])
        if len(calls) == 1:
            return {
                "text": "unrequested words " + expected,
                "language": "en",
                "language_probability": 1.0,
                "duration_seconds": 10.125,
                "words": raw_words,
            }
        return {
            "text": expected,
            "language": "en",
            "language_probability": 1.0,
            "duration_seconds": audio["waveform"].shape[-1] / 32000,
            "words": clean_words,
        }

    monkeypatch.setattr(
        speech_verification,
        "resolve_asr_model_directory",
        lambda value: Path("X:/verified-model"),
    )
    monkeypatch.setattr(
        speech_verification, "_load_asr_model", lambda path, threads: (object(), False)
    )
    monkeypatch.setattr(speech_verification, "_transcribe", fake_transcribe)
    monkeypatch.setattr(speech_verification, "_release_asr_model", lambda: True)

    output, transcript, similarity, speaker_similarity, accepted, report_json = verify_speech_audio(
        make_audio(10.125, 32000, value=0.1),
        expected,
        "trim_exact_target",
        "verified-model",
        "English",
        0.85,
        5,
        4,
        True,
        False,
        0.12,
        0.25,
    )
    report = json.loads(report_json)
    assert len(calls) == 2
    assert output["waveform"].shape[-1] < 10.125 * 32000
    assert transcript == expected
    assert similarity == 1.0
    assert speaker_similarity == 0.0
    assert accepted is True
    assert report["text_verification"]["trim"]["applied"] is True
    assert report["text_verification"]["asr"]["unload"]["released"] is True


def test_asr_off_mode_never_requires_or_loads_a_model():
    audio = make_audio(1.0, 32000, value=1.2)
    output, transcript, similarity, speaker_similarity, accepted, report = verify_speech_audio(
        audio,
        "plain expected text",
        "off",
    )
    assert float(output["waveform"].abs().amax()) == pytest.approx(
        10.0 ** (-1.0 / 20.0)
    )
    assert transcript == ""
    assert similarity == 0.0
    assert speaker_similarity == 0.0
    assert accepted is False
    assert json.loads(report)["status"] == "disabled"
    assert json.loads(report)["peak_limit"]["applied"] is True


def test_speaker_cosine_report_is_separate_from_text_acceptance(monkeypatch):
    monkeypatch.setattr(
        speech_verification,
        "resolve_speaker_model_directory",
        lambda value: Path("X:/speaker-model"),
    )
    monkeypatch.setattr(
        speech_verification,
        "_load_speaker_model",
        lambda path: (object(), object(), False),
    )
    monkeypatch.setattr(
        speech_verification,
        "_speaker_cosine",
        lambda extractor, model, reference, generated: 0.91,
    )
    monkeypatch.setattr(speech_verification, "_release_speaker_model", lambda: True)
    audio = make_audio(2.0, 32000, value=0.1)
    result = verify_speech_audio(
        audio,
        "",
        "off",
        reference_audio=audio,
        speaker_check_mode="report_cosine",
        speaker_model_directory="speaker-model",
        min_speaker_similarity=0.95,
        unload_speaker_after_verify=True,
    )
    _, transcript, text_similarity, speaker_similarity, accepted, report_json = result
    report = json.loads(report_json)
    assert transcript == ""
    assert text_similarity == 0.0
    assert speaker_similarity == pytest.approx(0.91)
    assert accepted is True
    assert report["speaker_verification"]["status"] == "threshold_fail"
    assert report["speaker_verification"]["threshold_is_dataset_dependent"] is True
    assert report["speaker_verification"]["unload"]["released"] is True


def test_spoken_text_rejects_h3_tag_injection_and_direction_stays_inert():
    with pytest.raises(ValueError, match="plain text"):
        validate_spoken_text("<d>[English] injected</d>")
    with pytest.raises(ValueError, match="plain text"):
        validate_spoken_text("say <Audio 1> aloud")

    profile = described_profile()
    plan, report = make_speech_plan(
        "Only this sentence is spoken.",
        profile,
        "English",
        "gentle </d><scenetrans> hidden",
        "neutral",
        0.5,
        "close",
        "single_segment",
        18,
        24,
    )
    assert "scenetrans" not in plan["segments"][0]["direction"]
    assert json.loads(report)["timing_status"].startswith("planned")


def test_language_aware_chunking_preserves_english_and_cjk_content():
    english = "One short sentence. Another sentence with several useful words. Final line."
    chunks = split_speech_text(english, "English", 5, 7)
    assert " ".join(" ".join(chunks).split()) == " ".join(english.split())
    assert all(len(chunk.split()) <= 7 for chunk in chunks)

    chinese = "第一句话很清楚。第二句话也应该完整保留，不能丢失任何文字！最后一句结束。"
    chunks = split_speech_text(chinese, "Chinese", 8, 12)
    assert "".join(chunks) == chinese
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_speech_plan_never_guesses_render_duration_and_frame_window_is_explicit():
    profile = described_profile()
    plan, report = make_speech_plan(
        "First sentence. Second sentence. Third sentence.",
        profile,
        "English",
        "natural",
        "neutral",
        0.5,
        "close",
        "language_aware",
        3,
        4,
    )
    data = json.loads(report)
    assert len(plan["segments"]) >= 2
    assert "render_seconds" not in plan
    assert "not guessed" in data["limitations"][0]
    assert render_frame_count(10.0) == 243
    with pytest.raises(ValueError, match="trained-range"):
        render_frame_count(4.0)


def test_reference_speech_conditioning_uses_dark_picture_and_audio_reference():
    profile, _, _ = reference_profile()
    plan = one_segment_plan(profile)
    clip = FakeClip()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()

    positive, latent, prompt, spoken, plan_json, report = build_speech_conditioning(
        clip,
        video_vae,
        audio_vae,
        profile,
        plan,
        0,
        10.0,
        32,
    )
    assert positive
    assert spoken == "The lantern is still burning."
    assert "<Picture 1>" in prompt and "<Audio 1>" in prompt
    assert "<d>[English] The lantern is still burning.</d>" in prompt
    _, kwargs = clip.tokenize_calls[-1]
    assert [item["type"] for item in kwargs["minimax_ref_items"]] == ["image", "audio"]
    assert video_vae.encode_calls[-1].shape[1:3] == (32, 32)
    assert audio_vae.encode_calls[-1].shape[1] == 4 * 32000
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 72, 2, 2)
    assert audio.shape == (1, 32, 2, 405)
    assert json.loads(plan_json)["profiles"]["speaker_a"]["reference_sha256"]
    assert json.loads(report)["task_type"] == "Ref2VA"


def test_described_speech_conditioning_has_no_fake_reference_or_duplicate_model_load():
    profile = described_profile()
    plan = one_segment_plan(profile)
    clip = FakeClip()
    output = build_speech_conditioning(
        clip,
        FakeVideoVAE(),
        FakeAudioVAE(),
        profile,
        plan,
        0,
        10.0,
        64,
    )
    prompt = output[2]
    _, kwargs = clip.tokenize_calls[-1]
    assert "minimax_ref_items" not in kwargs
    assert "<Audio 1>" not in prompt
    assert json.loads(output[-1])["task_type"] == "T2VA"


def test_speech_decode_skips_video_vae_and_reports_unverified_qa():
    profile = described_profile()
    plan = one_segment_plan(profile)
    latent = build_speech_conditioning(
        FakeClip(), FakeVideoVAE(), FakeAudioVAE(), profile, plan, 0, 10.0, 32
    )[1]
    audio, report = decode_speech_audio(latent, FakeAudioVAE(), "none")
    assert audio["sample_rate"] == 32000
    data = json.loads(report)
    assert data["transcript_status"] == "not_run"
    assert data["speaker_identity_status"] == "not_run"
    assert data["trim"]["applied"] is False


def test_dialogue_plan_binds_speakers_and_turn_selection_is_independent():
    first = described_profile("alice")
    second = described_profile("bob")
    plan, report = make_dialogue_plan(
        "S1: 你终于来了。\nS2: 路上有点堵，但我没忘记约定。",
        "speaker_lines",
        {"voice_profile_0": first, "voice_profile_1": second},
        "Chinese",
        "across_table",
    )
    assert plan["kind"] == "dialogue"
    assert [turn["speaker_id"] for turn in plan["segments"]] == ["alice", "bob"]
    assert json.loads(report)["render_strategy"] == "one_turn_at_a_time"

    profile, turn_plan, text, turn_report = select_dialogue_turn(plan, 1)
    assert profile["speaker_id"] == "bob"
    assert turn_plan["segments"][0]["index"] == 0
    assert text.startswith("路上")
    assert json.loads(turn_report)["turn_index"] == 1


def test_structured_dialogue_assembles_on_exact_samples_with_real_boundaries():
    first = described_profile("alice")
    second = described_profile("bob")
    script = json.dumps(
        {
            "turns": [
                {"speaker": "alice", "text": "First.", "pan": -0.5},
                {
                    "speaker": "bob",
                    "text": "Second.",
                    "pause_before_seconds": 0.25,
                    "pan": 0.5,
                },
            ]
        }
    )
    plan, _ = make_dialogue_plan(
        script,
        "json",
        {"voice_profile_0": first, "voice_profile_1": second},
        "English",
        "studio",
    )
    output, timeline_json, srt, vtt = assemble_speech_audio(
        plan,
        {
            "audio_segment_0": make_audio(1.0, 32000, value=0.1),
            "audio_segment_1": make_audio(1.0, 32000, value=0.1),
        },
        32000,
        0.0,
        -1.0,
    )
    timeline = json.loads(timeline_json)
    assert output["waveform"].shape == (1, 2, 72000)
    assert timeline["total_samples"] == 72000
    assert timeline["timeline"][0]["start_sample"] == 0
    assert timeline["timeline"][1]["start_sample"] == 40000
    assert timeline["subtitle_status"].startswith("actual_audio_boundaries")
    assert "alice: First." in srt
    assert vtt.startswith("WEBVTT")


def test_speech_studio_expands_native_comfy_graph_without_loader_nodes():
    profile = described_profile()
    plan = one_segment_plan(profile)
    result = MiniMaxH3SpeechStudioT8.execute(
        model=object(),
        clip=object(),
        video_vae=object(),
        audio_vae=object(),
        voice_profile=profile,
        speech_plan=plan,
        segment_index=0,
        seed=123,
        render_seconds=10.0,
        resolution=32,
        steps=20,
        sampler_name="res_multistep",
        scheduler="simple",
        shift_video=12.0,
        shift_audio=3.0,
        trim_mode="none",
        release_policy="clear_execution_cache",
    )
    assert result.expand
    class_types = {node["class_type"] for node in result.expand.values()}
    assert class_types == {
        "MiniMaxH3SpeechConditioningT8",
        "MiniMaxH3DualClockSamplerT8",
        "BasicGuider",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "MiniMaxH3SpeechDecodeT8",
        "MiniMaxH3SpeechVerifyT8",
        "MiniMaxH3SpeechFinalizeT8",
        "MiniMaxH3SpeechGuardT8",
    }
    assert not {"UNETLoader", "CLIPLoader", "VAELoader"} & class_types


def test_described_speech_api_example_reuses_models_and_keeps_stock_baseline():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "speech_described_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert {node["class_type"] for node in workflow.values()} == {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3VoiceProfileT8",
        "MiniMaxH3SpeechPlanT8",
        "MiniMaxH3SpeechStudioT8",
        "SaveAudio",
    }
    studio = workflow["7"]["inputs"]
    assert studio["model"] == ["1", 0]
    assert studio["clip"] == ["2", 0]
    assert studio["video_vae"] == ["3", 0]
    assert studio["audio_vae"] == ["4", 0]
    assert studio["steps"] == 20
    assert studio["sampler_name"] == "res_multistep"
    assert studio["scheduler"] == "simple"
    assert studio["resolution"] == 32
    assert studio["release_policy"] == "unload_all_models"
    assert all("Lora" not in node["class_type"] for node in workflow.values())


def test_reference_speech_api_example_requires_rights_and_ref2va():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "speech_reference_clone_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["1"]["inputs"]["unet_name"].startswith("minimax_h3_ref2va_")
    assert workflow["6"]["inputs"]["voice_mode"] == "reference_voice"
    assert workflow["6"]["inputs"]["rights_confirmed"] is True
    assert workflow["6"]["inputs"]["reference_audio"] == ["5", 0]
    assert workflow["8"]["inputs"]["steps"] == 20
    assert workflow["8"]["inputs"]["sampler_name"] == "res_multistep"
    assert workflow["8"]["inputs"]["scheduler"] == "simple"
    assert workflow["8"]["inputs"]["release_policy"] == "unload_all_models"
    assert workflow["8"]["inputs"]["verify_mode"] == "trim_exact_target"
    assert workflow["8"]["inputs"]["unload_asr_after_verify"] is True
    assert workflow["8"]["inputs"]["speaker_check_mode"] == "report_cosine"
    assert workflow["8"]["inputs"]["unload_speaker_after_verify"] is True


def test_dialogue_example_generates_turns_independently_before_sample_exact_mix():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "speech_dialogue_two_speaker_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["7"]["class_type"] == "MiniMaxH3DialogueScriptT8"
    assert workflow["8"]["class_type"] == "MiniMaxH3DialogueTurnSelectT8"
    assert workflow["9"]["class_type"] == "MiniMaxH3DialogueTurnSelectT8"
    assert workflow["10"]["inputs"]["voice_profile"] == ["8", 0]
    assert workflow["11"]["inputs"]["voice_profile"] == ["9", 0]
    assert workflow["10"]["inputs"]["verify_mode"] == "trim_exact_target"
    assert workflow["11"]["inputs"]["verify_mode"] == "trim_exact_target"
    assert workflow["12"]["class_type"] == "MiniMaxH3SpeechAssembleT8"
    assert workflow["12"]["inputs"]["peak_limit_dbfs"] == -1.0
    assert workflow["13"]["inputs"]["release_policy"] == "unload_all_models"


SPEECH_FRONTEND_WORKFLOWS = (
    "2026-08-10_H3_Speech_Described_Stock20_EXP.json",
    "2026-08-10_H3_Speech_Reference_Clone_Stock20_EXP.json",
    "2026-08-09_H3_Speech_Dialogue_Two_Speaker_Stock20_EXP.json",
    "2026-08-10_H3_Speech_Performance_ADR_Stock20_EXP.json",
    "2026-08-10_H3_Speech_LongForm_Resume_Stock20_EXP.json",
    "2026-08-10_H3_Speech_LongForm_Compose_EXP.json",
    "2026-08-10_H3_Speech_Voice_Library_Save_EXP.json",
    "2026-08-10_H3_Speech_Voice_Library_Load_EXP.json",
    "2026-08-10_H3_Speech_Voice_Library_Delete_EXP.json",
    "2026-08-10_H3_Speech_VRAM_Preflight_EXP.json",
    "2026-08-10_H3_Speech_LongForm_Control_EXP.json",
    "2026-08-09_H3_Speech_Joint_Dialogue_Stock20_EXP.json",
)


def load_speech_frontend_workflow(filename):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / filename
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename", SPEECH_FRONTEND_WORKFLOWS)
def test_speech_frontend_workflow_has_bidirectionally_consistent_links(filename):
    workflow = load_speech_frontend_workflow(filename)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == (max(links) if links else 0)

    for link_id, (_, source_id, source_slot, target_id, target_slot, link_type) in links.items():
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert link_id in source["links"]
        assert target["link"] == link_id
        assert source["type"] == target["type"] == link_type

    for node in nodes.values():
        for output in node.get("outputs", []):
            assert set(output.get("links") or []) <= links.keys()
        for node_input in node.get("inputs", []):
            if node_input.get("link") is not None:
                assert node_input["link"] in links


def test_speech_frontend_workflow_presets_keep_validated_exp_boundaries():
    described = load_speech_frontend_workflow(
        "2026-08-10_H3_Speech_Described_Stock20_EXP.json"
    )
    reference = load_speech_frontend_workflow(
        "2026-08-10_H3_Speech_Reference_Clone_Stock20_EXP.json"
    )
    dialogue = load_speech_frontend_workflow(
        "2026-08-09_H3_Speech_Dialogue_Two_Speaker_Stock20_EXP.json"
    )

    described_nodes = {node["id"]: node for node in described["nodes"]}
    assert described_nodes[1]["widgets_values"][0].startswith("minimax_h3_fl2va_")
    assert described_nodes[7]["widgets_values"][3:7] == [
        32,
        20,
        "res_multistep",
        "simple",
    ]
    assert described_nodes[7]["widgets_values"][10] == "off"
    assert described_nodes[7]["widgets_values"][20] == "unload_all_models"

    reference_nodes = {node["id"]: node for node in reference["nodes"]}
    assert reference_nodes[1]["widgets_values"][0].startswith("minimax_h3_ref2va_")
    assert reference_nodes[5]["widgets_values"] == ["speech_reference.flac"]
    assert reference_nodes[6]["widgets_values"][0] == "reference_voice"
    assert reference_nodes[6]["widgets_values"][4] is True
    assert reference_nodes[8]["widgets_values"][3:7] == [
        32,
        20,
        "res_multistep",
        "simple",
    ]
    assert reference_nodes[8]["widgets_values"][10:12] == [
        "trim_exact_target",
        "faster-whisper-small.en-d1d751a5",
    ]
    assert reference_nodes[8]["widgets_values"][15:17] == [
        "report_cosine",
        "wavlm-base-plus-sv-feb593a6",
    ]
    assert reference_nodes[8]["widgets_values"][20] == "unload_all_models"

    dialogue_nodes = {node["id"]: node for node in dialogue["nodes"]}
    assert dialogue_nodes[7]["type"] == "MiniMaxH3DialogueScriptT8"
    assert {item["name"] for item in dialogue_nodes[7]["inputs"][-2:]} == {
        "voice_profiles.voice_profile_0",
        "voice_profiles.voice_profile_1",
    }
    for node_id in (10, 11):
        assert dialogue_nodes[node_id]["widgets_values"][3:7] == [
            32,
            20,
            "res_multistep",
            "simple",
        ]
        assert dialogue_nodes[node_id]["widgets_values"][10] == "trim_exact_target"
        assert dialogue_nodes[node_id]["widgets_values"][20] == "keep_loaded"
    assert {item["name"] for item in dialogue_nodes[12]["inputs"][-2:]} == {
        "audio_segments.audio_segment_0",
        "audio_segments.audio_segment_1",
    }
    assert dialogue_nodes[13]["widgets_values"] == ["unload_all_models"]


def test_speech_reliability_api_examples_keep_explicit_safety_boundaries():
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api"
    performance = json.loads((root / "speech_performance_adr_api.json").read_text(encoding="utf-8"))
    assert performance["7"]["class_type"] == "MiniMaxH3SpeechPerformanceT8"
    assert performance["8"]["inputs"]["speech_plan"] == ["7", 0]
    assert performance["8"]["inputs"]["release_policy"] == "unload_all_models"
    assert performance["9"]["class_type"] == "MiniMaxH3SpeechADRFitT8"
    assert performance["9"]["inputs"]["minimum_rate"] == 0.9
    assert performance["9"]["inputs"]["maximum_rate"] == 1.1

    longform = json.loads((root / "speech_longform_resume_api.json").read_text(encoding="utf-8"))
    assert longform["7"]["class_type"] == "MiniMaxH3SpeechLongFormStartT8"
    assert longform["8"]["inputs"]["segment_index"] == ["7", 2]
    assert longform["8"]["inputs"]["voice_profile"] == ["7", 1]
    assert longform["8"]["inputs"]["release_policy"] == "unload_all_models"
    assert longform["9"]["class_type"] == "MiniMaxH3SpeechLongFormAcceptT8"
    assert longform["9"]["inputs"]["accepted"] == ["8", 7]

    joint = json.loads((root / "speech_joint_dialogue_exp_api.json").read_text(encoding="utf-8"))
    assert joint["10"]["class_type"] == "MiniMaxH3SpeechGuardT8"
    assert joint["11"]["inputs"]["speech_guard"] == ["10", 0]
    assert joint["17"]["inputs"]["speech_guard"] == ["10", 0]
    assert joint["17"]["inputs"]["release_policy"] == "unload_all_models"
    assert joint["5"]["inputs"]["audio"] == "speech_reference_a.flac"
    assert joint["6"]["inputs"]["audio"] == "speech_reference_b.flac"
