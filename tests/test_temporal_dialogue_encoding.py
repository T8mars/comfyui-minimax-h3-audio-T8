"""Native media/text re-encoding behavior, not a speech quality test."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from h3_audio_t8_pkg import semantic_bridge
from h3_audio_t8_pkg.temporal_dialogue_encoding import encode_scoped_window
from h3_audio_t8_pkg.temporal_dialogue_scope import build_dialogue_plan
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio
from test_temporal_dialogue_scope import event, windows


def args(**updates):
    value = dict(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                 prompt="Stable visual scene", width=128, height=128, length=294,
                 audio_mode="native", add_source_as_reference=False)
    value.update(updates)
    return value


def test_capture_is_explicit_and_default_conditioning_outputs_are_unchanged():
    values = args(first_frame=torch.zeros(1, 64, 64, 3))
    old = build_conditioning(**values)
    detailed = build_conditioning(**values, return_details=True)
    captured = build_conditioning(**values, return_text_recipe=True)
    assert len(old) == 6 and len(detailed) == len(captured) == 7
    assert "text_recipe" not in detailed[-1]
    assert _input_identity(old) == _input_identity(captured[:6])
    assert set(captured[-1]) - set(detailed[-1]) == {"text_recipe"}


@pytest.mark.parametrize("media", ["text", "first_last", "references", "hybrid"])
def test_raw_media_tokenization_is_reused_without_another_vae_encode(media):
    values = args()
    if media in {"first_last", "hybrid"}:
        values.update(first_frame=torch.zeros(1, 64, 64, 3),
                      last_frame=torch.ones(1, 64, 64, 3))
    if media in {"references", "hybrid"}:
        values.update(ref_images={"ref_image_1": torch.ones(1, 64, 64, 3)},
                      ref_videos={"ref_video_1": torch.zeros(48, 64, 64, 3)},
                      ref_video_audios={"ref_video_audio_1": make_audio(2)},
                      ref_audios={"ref_audio_1": make_audio(1)})
    source = build_conditioning(**values, return_text_recipe=True)
    recipe = source[-1]["text_recipe"]
    before = (len(values["video_vae"].encode_calls), len(values["audio_vae"].encode_calls))
    plan = build_dialogue_plan("保留人物、衣着、杯子和动作。", [event("later", 10.1, 10.9)], 294)
    output = [encode_scoped_window(recipe, plan, w) for w in windows()]
    assert before == (len(values["video_vae"].encode_calls), len(values["audio_vae"].encode_calls))
    assert len(values["clip"].tokenize_calls) == 3
    assert "不敢。" not in values["clip"].tokenize_calls[1][0]
    assert "不敢。" in values["clip"].tokenize_calls[2][0]
    original_kwargs = values["clip"].tokenize_calls[0][1]
    assert _input_identity(original_kwargs) == _input_identity(values["clip"].tokenize_calls[1][1])
    for positive, compiled, prepared, report in output:
        assert positive[0][1]["tokens"]["prompt"] == prepared
        assert report["native_text_reencoded"] and not report["embedding_sliced"]
        assert not report["sampling_executed"] and not report["quality_qualified"]
        assert positive[0][1]["t8_temporal_dialogue_scope"]["text_sha256"] == compiled.sha256
        for key in ("minimax_keyframes", "minimax_refs"):
            if key in source[0][0][1]:
                assert positive[0][1][key] is source[0][0][1][key]


def test_speaker_source_audio_is_remapped_with_the_original_native_recipe():
    values = args(drive_audio=make_audio(12), add_source_as_reference=True,
                  ref_videos={"ref_video_1": torch.zeros(48, 64, 64, 3)},
                  ref_video_audios={"ref_video_audio_1": make_audio(2)})
    recipe = build_conditioning(**values, return_text_recipe=True)[-1]["text_recipe"]
    assert recipe.source_audio_ordinal == 2
    plan = build_dialogue_plan("<Audio 1>保持音色。", [event("later", 10, 11, speaker="Audio 1")], 294)
    _, _, prepared, _ = encode_scoped_window(recipe, plan, windows()[1])
    assert prepared.count("<Audio 2>") == 2
    assert "<Audio 1>" not in prepared


def test_bridge_reapplied_once_to_fresh_native_tokens_not_to_original_bridge_output(monkeypatch):
    calls = []
    bridge = SimpleNamespace(active=True)
    def apply(positive, provider, **kwargs):
        assert provider is bridge
        calls.append(kwargs)
        return [[tensor + 1, {**metadata, "bridge_fixture": True}]
                for tensor, metadata in positive], {"fixture_applied": True}
    monkeypatch.setattr(semantic_bridge, "apply_bridge", apply)
    source = build_conditioning(**args(semantic_bridge=bridge), return_text_recipe=True)
    recipe = source[-1]["text_recipe"]
    plan = build_dialogue_plan("Stable scene", [], 294)
    for window in windows():
        positive, _, _, report = encode_scoped_window(recipe, plan, window)
        assert torch.equal(positive[0][0], torch.ones_like(positive[0][0]))
        assert report["semantic_bridge"] == {"fixture_applied": True}
    assert len(calls) == 3  # Original full encoding plus two fresh window encodes.
    assert torch.equal(source[0][0][0], torch.ones_like(source[0][0][0]))


def test_changed_media_metadata_or_source_timeline_rejected_before_encode():
    values = args(first_frame=torch.zeros(1, 64, 64, 3))
    recipe = build_conditioning(**values, return_text_recipe=True)[-1]["text_recipe"]
    plan = build_dialogue_plan("Stable scene", [], 294)
    with pytest.raises(ValueError, match="CONDITIONING alone"):
        encode_scoped_window([[torch.zeros(1), {}]], plan, windows()[0])
    with pytest.raises(ValueError, match="content changed"):
        encode_scoped_window(replace(recipe, width=256), plan, windows()[0])
    with pytest.raises(ValueError, match="source timeline"):
        encode_scoped_window(recipe, build_dialogue_plan("Stable scene", [], 300), windows(total_frames=300)[0])
    recipe.metadata["minimax_keyframes"][0]["latent"].add_(1)
    with pytest.raises(ValueError, match="content changed"):
        encode_scoped_window(recipe, plan, windows()[0])
    assert len(values["clip"].tokenize_calls) == 1
