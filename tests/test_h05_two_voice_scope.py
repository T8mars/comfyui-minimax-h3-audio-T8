"""Two routed voice anchors + actual snapped v5 text; not pretrained voice QA."""
import hashlib
import json

import comfy.nested_tensor
import pytest
import torch

from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio
from test_chunked_two_pass_parity import _plan
from h3_audio_t8_pkg import reference_runtime as runtime
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.temporal_dialogue_bank import prepare_window_bank
from h3_audio_t8_pkg.temporal_dialogue_scope import build_dialogue_plan


class DistinctVoiceVAE(FakeAudioVAE):
    def encode(self, audio):
        output = super().encode(audio)
        return output * 0 + audio.mean()


@pytest.fixture
def two_voices(monkeypatch):
    video, audio, clip = FakeVideoVAE(), DistinctVoiceVAE(), FakeClip()

    def identity(value, role):
        assert value in (video, audio)
        return {"scope": runtime.PRODUCER_SCOPE,
                "sha256": hashlib.sha256((role + "synthetic-two-voice-v1").encode()).hexdigest()}

    monkeypatch.setattr(runtime, "native_producer_identity", identity)
    scene, _ = runtime.create_package("Scene", frames=torch.zeros(1, 32, 32, 3),
                                     video_vae=video, width=32, height=32)
    a, _ = runtime.create_package("A", kind="audio", audio=make_audio(.1, value=.1), audio_vae=audio)
    b, _ = runtime.create_package("B", kind="audio", audio=make_audio(.1, value=-.2), audio_vae=audio)
    return video, audio, clip, [scene, a, b]


def capture(two_voices, order=("Scene", "A", "B")):
    video, audio, clip, packages = two_voices
    refs, mapping = runtime.capture_set(packages, json.dumps([
        {"role_id": role, "visual": role == "Scene", "voice": role != "Scene"} for role in order]))
    output = build_conditioning(clip, video, audio, "<Picture 1> with two voices <Audio 1> and <Audio 2>.",
        64, 64, 294, task_type="Ref2VA", audio_mode="native", add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, prepared_reference_set=refs, return_text_recipe=True)
    return refs, mapping, output


def bank(output):
    dialogue = build_dialogue_plan("<Picture 1>: two women. A uses <Audio 1>, B uses <Audio 2>.", [
        {"event_id": "A_crossing", "speaker": "A, <Audio 1>", "utterance": "你好，今天真不错。",
         "start_seconds": 6.2, "end_seconds": 8.2},
        {"event_id": "B_later", "speaker": "B, <Audio 2>", "utterance": "是啊，我们出去走走吧。",
         "start_seconds": 9.1, "end_seconds": 10.6}], 294)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 87, 2, 2), torch.zeros(1, 32, 2, 501)))}
    return prepare_window_bank(output[-1]["text_recipe"], dialogue, source,
                               _plan(temporal_chunk_frames=187, temporal_overlap_frames=34))


def test_two_voices_keep_actual_ordinals_across_onset_continuation_without_media_reencode(two_voices):
    video, audio, clip, packages = two_voices
    _refs, _mapping, output = capture(two_voices)
    media = json.loads(output[4])
    assert media["pictures"] == {"1": "role:Scene/visual"}
    assert media["audios"] == {"1": "role:A/voice", "2": "role:B/voice"}
    assert media["source_audio_ordinal"] is None and output[2] is None
    assert "noise_mask" not in output[1]  # Voice anchors do not freeze unfinished generated audio.
    before = (len(video.encode_calls), len(audio.encode_calls))
    result, report = bank(output)
    assert before == (1, 2) == (len(video.encode_calls), len(audio.encode_calls))
    assert report["native_encodes"] == 2 and len(clip.tokenize_calls) == 3
    first, second = result.encoded
    assert (first.compiled.window.start_frame, first.compiled.window.end_frame) == (0, 187)
    assert (second.compiled.window.start_frame, second.compiled.window.owned_start_frame,
            second.compiled.window.end_frame) == (153, 187, 294)
    assert [(item.event_id, item.state) for item in first.compiled.dialogue] == [("A_crossing", "onset")]
    assert [(item.event_id, item.state) for item in second.compiled.dialogue] == [
        ("A_crossing", "continuation"), ("B_later", "onset")]
    assert "是啊，我们出去走走吧。" not in first.prepared_prompt
    assert "你好，今天真不错。" in first.prepared_prompt and "你好，今天真不错。" in second.prepared_prompt
    assert "Continue the already-started line" in second.prepared_prompt
    for encoded in result.encoded:
        refs = encoded.positive[0][1]["minimax_refs"]
        assert [item["kind"] for item in refs] == ["image", "audio", "audio"]
        assert refs[1]["audio_latent"] is packages[1].tensors["voice_latent"]
        assert refs[2]["audio_latent"] is packages[2].tensors["voice_latent"]
        assert not torch.equal(refs[1]["audio_latent"], refs[2]["audio_latent"])
        assert not encoded.report["sampling_executed"]


def test_rerouting_voice_order_is_explicit_and_changes_frozen_bank_identity(two_voices):
    _refs, _mapping, original = capture(two_voices)
    old_bank, _ = bank(original)
    _refs, _mapping, swapped = capture(two_voices, ("Scene", "B", "A"))
    media = json.loads(swapped[4])
    assert media["audios"] == {"1": "role:B/voice", "2": "role:A/voice"}
    swapped_bank, _ = bank(swapped)
    assert swapped_bank.sha256 != old_bank.sha256
    assert swapped_bank.encoded[0].content_identity != old_bank.encoded[0].content_identity
    assert torch.equal(original[0][0][1]["minimax_refs"][1]["audio_latent"],
                       swapped[0][0][1]["minimax_refs"][2]["audio_latent"])
