"""Real PCM/codec consumer regressions; synthetic stems are not speech quality evidence."""
import json
import math

import av
import pytest
import torch

from h3_audio_t8_pkg.dialogue_audio import build_dialogue_safe_master
from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg.local_sfx import prepare_local_sfx_candidates
from h3_audio_t8_pkg.nodes_local_sfx import MiniMaxH3LocalSFXCandidatesT8


def audio(waveform, rate=48000):
    return {"waveform": waveform.reshape(1, 1, -1), "sample_rate": rate}


def stems(rate=48000):
    time = torch.arange(rate*2) / rate
    # A deterministic modulated harmonic stem, explicitly not verified speech.
    voice = audio(.08 * torch.sin(2*math.pi*173*time) * (.5 + .5*torch.sin(2*math.pi*3*time)), rate)
    background = audio(.02 * torch.sin(2*math.pi*710*time), rate)
    external = audio(.3 * torch.sin(2*math.pi*1100*time), rate)
    h3 = audio(.006 * torch.sin(2*math.pi*420*time), rate)
    source = audio(voice["waveform"] + background["waveform"], rate)
    return source, voice, external, h3


@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_pcm_window_choice_matches_rms_keeps_voice_and_outside_bit_exact(rate):
    source, voice, external, h3 = stems(rate)
    originals = [x["waveform"].clone() for x in (source, voice, external, h3)]
    selected, external_preview, h3_preview, raw = prepare_local_sfx_candidates(
        source, external, .25, 1.25, protected_voice_audio=voice, h3_sfx_audio=h3, fade_seconds=0)
    report = json.loads(raw)
    assert selected is source
    start, end = report["start_sample"], report["end_sample"]
    for name, preview, candidate in (("external", external_preview, external), ("h3", h3_preview, h3)):
        gain = report["candidates"][name]["match_gain"]
        expected = voice["waveform"][..., start:end] + candidate["waveform"][..., :end-start] * gain
        assert torch.equal(preview["waveform"][..., start:end], expected)
        assert torch.equal(preview["waveform"][..., :start], source["waveform"][..., :start])
        assert torch.equal(preview["waveform"][..., end:], source["waveform"][..., end:])
        assert report["candidates"][name]["matched_window_rms_before_fade"] == pytest.approx(report["source_residual_rms"], rel=1e-6)
    chosen, real_preview, _, _ = prepare_local_sfx_candidates(
        source, external, .25, 1.25, protected_voice_audio=voice, selection="external", fade_seconds=0)
    assert chosen is real_preview  # No second confirmation or hidden approval.
    for before, after in zip(originals, (source, voice, external, h3)):
        assert torch.equal(before, after["waveform"])
    assert not any(report["claims"].values())


def test_crossfade_edges_and_absent_h3_are_explicit():
    source, voice, external, _ = stems()
    selected, preview, absent, raw = prepare_local_sfx_candidates(source, external, .5, 1., protected_voice_audio=voice)
    report = json.loads(raw)
    assert selected is source and absent is source
    assert report["candidates"]["h3"]["status"] == "absent"
    for index in (24000, 47999):
        assert torch.equal(preview["waveform"][..., index], source["waveform"][..., index])
    with pytest.raises(ValueError, match="absent"):
        prepare_local_sfx_candidates(source, external, .5, 1., selection="h3")


def test_optional_voice_does_not_claim_separation_or_voice_protection():
    source, _, external, _ = stems()
    _, _, _, raw = prepare_local_sfx_candidates(source, external, .5, 1.)
    report = json.loads(raw)
    assert report["protected_voice"] is None and report["voice_stem_gain"] is None
    assert report["claims"]["source_separation_performed"] is False


def test_explicit_h3_choice_needs_no_extra_accept_switch():
    source, voice, external, h3 = stems()
    selected, _, preview, raw = prepare_local_sfx_candidates(source, external, .5, 1.,
        protected_voice_audio=voice, h3_sfx_audio=h3, selection="h3")
    assert selected is preview and json.loads(raw)["selection"] == "h3"
    with pytest.raises(ValueError, match="external PCM"):
        prepare_local_sfx_candidates(source, None, .5, 1.)


@pytest.mark.parametrize("failure", ["rate", "shape", "dtype", "nonfinite"])
def test_protected_voice_never_resampled_or_truncated(failure):
    source, voice, external, _ = stems()
    if failure == "rate":
        voice["sample_rate"] = 44100
    elif failure == "shape":
        voice["waveform"] = voice["waveform"][..., :-1]
    elif failure == "dtype":
        voice["waveform"] = voice["waveform"].double()
    else:
        voice["waveform"][..., 100] = float("nan")
    with pytest.raises(ValueError, match="voice|Protected"):
        prepare_local_sfx_candidates(source, external, .5, 1., protected_voice_audio=voice)


@pytest.mark.parametrize("kwargs", [dict(start_seconds=-.1), dict(end_seconds=2.1), dict(end_seconds=.2),
    dict(fade_seconds=.6), dict(candidate_start_seconds=2.), dict(start_seconds=float("nan"))])
def test_actual_window_errors_are_not_silently_padded(kwargs):
    source, voice, external, _ = stems()
    settings = dict(start_seconds=.25, end_seconds=1.25, protected_voice_audio=voice)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        prepare_local_sfx_candidates(source, external, **settings)


def test_silence_requires_explicit_target_and_nonfinite_candidate_is_rejected():
    source, voice, external, _ = stems()
    with pytest.raises(ValueError, match="Source residual is silent"):
        prepare_local_sfx_candidates(voice, external, .5, 1., protected_voice_audio=voice)
    selected, _, _, raw = prepare_local_sfx_candidates(voice, external, .5, 1., protected_voice_audio=voice,
        level_policy="target_rms", target_rms_dbfs=-28, selection="external")
    assert selected is not voice and json.loads(raw)["target_rms"] == pytest.approx(10**(-28/20))
    external["waveform"][..., 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        prepare_local_sfx_candidates(source, external, .5, 1.)


def test_headroom_scales_only_candidate_without_limiting_voice_or_source():
    voice = audio(torch.full((48000,), .8))
    source = audio(torch.full((48000,), .9))
    external = audio(torch.ones(48000))
    settings = dict(protected_voice_audio=voice, fade_seconds=0, level_policy="target_rms",
                    target_rms_dbfs=-6, selection="external")
    unbounded, _, _, raw = prepare_local_sfx_candidates(source, external, .25, .75, **settings)
    assert json.loads(raw)["candidates"]["external"]["exceeds_pcm_full_scale"]
    assert float(unbounded["waveform"].max()) > 1
    limited, _, _, raw = prepare_local_sfx_candidates(source, external, .25, .75,
        headroom_policy="attenuate_candidate", **settings)
    row = json.loads(raw)["candidates"]["external"]
    assert row["headroom_gain"] < 1 and not row["exceeds_pcm_full_scale"]
    expected = .8 + 10**(-6/20) * row["headroom_gain"]
    torch.testing.assert_close(limited["waveform"][..., 12000:36000], torch.full((1, 1, 24000), expected))
    assert torch.equal(limited["waveform"][..., :12000], source["waveform"][..., :12000])
    assert torch.equal(voice["waveform"], torch.full_like(voice["waveform"], .8))


@pytest.mark.parametrize("amplitude", [0., .0004, .1])
def test_ceiling_keeps_quiet_candidate_native_and_only_reduces_louder_pcm(amplitude):
    rate = 32000
    source = audio(torch.full((rate,), .02), rate)
    candidate = audio(torch.full((rate,), amplitude), rate)
    original = source["waveform"].clone()
    target = 10**(-40/20)
    selected, _, _, raw = MiniMaxH3LocalSFXCandidatesT8.execute(source, candidate, .25, .75,
        protected_voice_audio=source, selection="external", level_policy="target_rms_ceiling",
        target_rms_dbfs=-40, fade_seconds=0).result
    row = json.loads(raw)["candidates"]["external"]
    expected = .02 + min(amplitude, target)
    torch.testing.assert_close(selected["waveform"][..., 8000:24000],
        torch.full((1,1,16000), expected))
    assert row["match_gain"] <= 1
    assert row["upward_gain_prevented"] == (0 < amplitude < target)
    assert torch.equal(selected["waveform"][..., :8000], original[..., :8000])
    assert torch.equal(selected["waveform"][..., 24000:], original[..., 24000:])
    assert torch.equal(source["waveform"], original)


def test_headroom_reports_impossible_protected_stem_without_altering_it():
    voice = audio(torch.full((48000,), 1.2))
    source = audio(torch.full((48000,), .2))
    external = audio(torch.ones(48000))
    with pytest.raises(ValueError, match="protected voice"):
        prepare_local_sfx_candidates(source, external, .25, .75, protected_voice_audio=voice,
            fade_seconds=0, headroom_policy="attenuate_candidate", selection="external")
    assert torch.equal(voice["waveform"], torch.full_like(voice["waveform"], 1.2))


def test_real_pcm_wave_mixer_local_candidate_and_three_arm_mp4(tmp_path):
    source, voice, external, h3 = stems(44100)
    # Actual existing mixer consumer creates the original and protected stems.
    mixed, protected, background, _ = build_dialogue_safe_master(
        voice, True, 2., output_sample_rate=48000, sfx_audio=source)
    pcm_file = tmp_path / "external-synthetic.wav"
    with av.open(str(pcm_file), "w") as container:
        stream = container.add_stream("pcm_f32le", rate=44100)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(external["waveform"][0].numpy(), format="flt", layout="mono")
        frame.sample_rate = 44100
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(pcm_file)) as container:
        frames = [torch.from_numpy(frame.to_ndarray().copy()) for frame in container.decode(audio=0)]
    loaded = audio(torch.cat(frames, -1), 44100)
    selected, ext_preview, h3_preview, raw = MiniMaxH3LocalSFXCandidatesT8.execute(mixed, loaded, .25, 1.25,
        protected_voice_audio=protected, h3_sfx_audio=h3, candidate_start_seconds=.125).result
    report = json.loads(raw)
    assert report["candidates"]["external"]["resampled"]
    assert report["candidates"]["external"]["candidate_local_start_sample"] == 6000
    images = torch.linspace(0, 1, 48*32*64*3).reshape(48, 32, 64, 3)
    delivery = [save_h3_av_safe(images, sound, tmp_path/f"{name}.mp4")
                for name, sound in (("original", selected), ("external", ext_preview), ("h3", h3_preview))]
    assert len({row["source_rgb8_sha256"] for row in delivery}) == 1
    assert len({row["source_pcm_f32_sha256"] for row in delivery}) == 3
    assert all(row["delivery_verification"]["full_decode_verified"] for row in delivery)
    assert all(row["delivery_verification"]["logical_audio_samples"] == 96000 for row in delivery)
    (tmp_path / "consumer-evidence.json").write_text(json.dumps({"sfx": report, "delivery": delivery,
        "synthetic_inputs": True, "trained_model_used": False, "human_reviewed": False}, indent=2), encoding="utf-8")
