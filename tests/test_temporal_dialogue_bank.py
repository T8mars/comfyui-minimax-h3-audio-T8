"""Actual v5 window consumption, not a generated speech/quality claim."""
from dataclasses import replace
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.modular_sampling import chunked_v5
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from h3_audio_t8_pkg.modular_sampling.temporal_chunked_v5 import sample_scoped_v5_window
from h3_audio_t8_pkg.temporal_dialogue_bank import prepare_window_bank, select_window_conditioning
from h3_audio_t8_pkg.temporal_dialogue_scope import build_dialogue_plan
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE
from test_chunked_two_pass_parity import _plan
from test_modular_chunked_v5 import make_v5_harness
from test_temporal_dialogue_scope import user_plan


@pytest.fixture
def scoped_harness(monkeypatch):
    calls, _source, _positive, noise, sampler, sigmas, _old_plan = make_v5_harness(monkeypatch)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 87, 2, 2), torch.zeros(1, 32, 2, 501)))}
    plan = _plan(temporal_chunk_frames=187, temporal_overlap_frames=34)
    clip = FakeClip()
    recipe = build_conditioning(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="Native media", width=64, height=64, length=294, audio_mode="native",
        return_text_recipe=True)[-1]["text_recipe"]
    bank, report = prepare_window_bank(recipe, user_plan(), source, plan)
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    return calls, source, noise, sampler, sigmas, plan, clip, bank, lifted, prepared, report


def run(h, index, previous=None, **updates):
    _calls, source, noise, sampler, sigmas, plan, _clip, bank, lifted, prepared, _report = h
    kwargs = dict(model=object(), source=source, lifted=lifted, prepared=prepared,
                  plan=plan, noise=noise, sampler=sampler, sigmas=sigmas,
                  bank=bank, index=index, previous=previous)
    kwargs.update(updates)
    return sample_scoped_v5_window(**kwargs)


def test_bank_encodes_before_sampling_and_actual_windows_consume_different_speech(scoped_harness, monkeypatch):
    h = scoped_harness
    calls, source, _noise, _sampler, _sigmas, _plan_value, clip, bank, _lifted, _prepared, report = h
    assert report["native_encodes"] == 2 and not report["sampling_executed"]
    assert len(clip.tokenize_calls) == 3  # original media plus two precompiled windows
    observed, pieces = [], []
    original = legacy.sample_piece

    def sample(piece, positive, *args, **kwargs):
        observed.append(positive[0][1]["tokens"]["prompt"])
        pieces.append(piece)
        return original(piece, positive, *args, **kwargs)

    monkeypatch.setattr(legacy, "sample_piece", sample)
    first, base, scoped, _ = run(h, 0)
    first_video, first_audio = first["samples"].unbind()
    first_identity = _input_identity(first)
    final, _base, _scoped, report_json = run(h, 1, scoped)
    assert "我没看过。" in observed[0] and "不敢。" not in observed[0]
    assert "我没看过。" not in observed[1] and "一次都没有？" in observed[1] and "不敢。" in observed[1]
    assert _input_identity(first) == first_identity
    v, a = final["samples"].unbind()
    assert torch.equal(v[:, :, :first_video.shape[2]], first_video)
    assert torch.equal(a[..., :first_audio.shape[-1]], first_audio)
    assert a.shape[-1] == 501 and not torch.equal(a, source["samples"].tensors[1])
    assert bool((pieces[1]["noise_mask"].tensors[1][..., :57] == 0).all())
    assert len(clip.tokenize_calls) == 3 and calls == {"lift": 1, "noise": 1, "sample": 2, "sampler_bind": 2}
    assert json.loads(report_json)["scoped_native_conditioning_consumed"]
    assert base.index == 0 and bank.encoded[1].report["prepared_dialogue_events"]


def test_scoped_previous_rejects_legacy_result_or_changed_bank_before_sampling(scoped_harness):
    h = scoped_harness
    _out, base, scoped, _report = run(h, 0)
    for bad in (base, replace(scoped, bank_sha256="f" * 64)):
        with pytest.raises(ValueError, match="bank/policy"):
            run(h, 1, bad)
    assert h[0]["sample"] == 1


def test_changed_real_audio_prefix_or_condition_cannot_masquerade_as_scoped(scoped_harness):
    h = scoped_harness
    _out, _base, scoped, _report = run(h, 0)
    bad_positive = [[tensor, {**metadata, "unknown_user_metadata": "changed"}]
                    for tensor, metadata in h[7].encoded[1].positive]
    with pytest.raises(ValueError, match="CONDITIONING"):
        run(h, 1, scoped, positive=bad_positive)
    scoped.base.output_latent["samples"].tensors[1][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="content changed"):
        run(h, 1, scoped)
    assert h[0]["sample"] == 1


def test_bank_rejects_changed_source_plan_encoded_values_or_ownership(scoped_harness):
    h = scoped_harness
    _calls, source, _noise, _sampler, _sigmas, plan, _clip, bank, _lifted, _prepared, _report = h
    with pytest.raises(ValueError, match="Chunk Plan"):
        select_window_conditioning(bank, source, {**plan, "temporal_overlap_frames": 17}, 0)
    with pytest.raises(ValueError, match="index"):
        select_window_conditioning(bank, source, plan, True)
    bank.encoded[0].positive[0][0].add_(1)
    with pytest.raises(ValueError, match="encoded native"):
        select_window_conditioning(bank, source, plan, 0)


def test_bank_native_prepared_spans_follow_audio_ordinal_remapping():
    from helpers import make_audio
    recipe = build_conditioning(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="Native media", width=64, height=64, length=294, audio_mode="native",
        drive_audio=make_audio(12), add_source_as_reference=True,
        ref_videos={"ref_video_1": torch.zeros(48, 64, 64, 3)},
        ref_video_audios={"ref_video_audio_1": make_audio(2)}, return_text_recipe=True)[-1]["text_recipe"]
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 87, 2, 2), torch.zeros(1, 32, 2, 501)))}
    dialogue = build_dialogue_plan("<Audio 1>保留音色", [
        {"event_id": "one", "speaker": "<Audio 1> woman", "utterance": "<Audio 1>你好。",
         "start_seconds": 10, "end_seconds": 11}], 294)
    bank, _ = prepare_window_bank(recipe, dialogue, source, _plan(temporal_chunk_frames=187))
    encoded = bank.encoded[-1]
    event = encoded.report["prepared_dialogue_events"][0]
    span = encoded.prepared_prompt[event["prompt_char_start"]:event["prompt_char_end"]]
    assert span == "<d><Audio 2>你好。</d>" and "<Audio 1>" not in encoded.prepared_prompt


def test_foreign_encoder_metadata_is_retained_live_not_certified_or_blocked(scoped_harness):
    class ForeignClip(FakeClip):
        def __init__(self):
            super().__init__()
            self.callback = lambda: None

        def encode_from_tokens_scheduled(self, tokens):
            output = super().encode_from_tokens_scheduled(tokens)
            output[0][1]["foreign_callback"] = self.callback
            return output

    h = list(scoped_harness)
    clip = ForeignClip()
    recipe = build_conditioning(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="Native media", width=64, height=64, length=294, audio_mode="native",
        return_text_recipe=True)[-1]["text_recipe"]
    bank, report = prepare_window_bank(recipe, user_plan(), h[1], h[5])
    h[6], h[7] = clip, bank
    assert not report["native_metadata_portable"] and report["warnings"]
    assert all(item.positive[0][1]["foreign_callback"] is clip.callback for item in bank.encoded)
    _, _, first, _ = run(h, 0)
    run(h, 1, first)
    rebuilt, _ = prepare_window_bank(recipe, user_plan(), h[1], h[5])
    assert rebuilt.sha256 != bank.sha256  # no invented portable provider identity
    bank.encoded[1].positive[0][1]["foreign_callback"] = lambda: None
    with pytest.raises(ValueError, match="encoded native"):
        select_window_conditioning(bank, h[1], h[5], 1)
