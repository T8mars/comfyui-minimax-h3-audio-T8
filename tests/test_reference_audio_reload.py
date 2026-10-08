"""Fresh native audio wrapper reload: real identities, tiny weights, no encode."""
import json

import pytest
import torch

from helpers import FakeClip
from h3_audio_t8_pkg import reference_runtime as runtime
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.core import ensure_h3_audio_vae_non_aligned_crop_compat
from h3_audio_t8_pkg.reference_package import NORMALIZATION, capture_package, load_package, save_package
from test_progressive_producers import component


def voice_package():
    created = component("audio_vae")
    assert created.crop_input is True
    assert ensure_h3_audio_vae_non_aligned_crop_compat(created)
    producer = runtime.portable_producer(created, "audio_vae")
    latent = torch.linspace(-1, 1, 192).reshape(1, 32, 2, 3)
    package = capture_package([dict(id="voice", role_id="speaker_A", kind="audio",
        latent="voice", grounding=None, normalization=NORMALIZATION, producer=producer,
        source=dict(scope="actual_decoded_pcm_tensor", sha256="1" * 64))], {"voice": latent})
    return package, producer


def test_fresh_native_audio_reload_then_apply_keeps_exact_assets_and_encodes_only_qwen(tmp_path):
    package, producer = voice_package()
    path = tmp_path / "voice.safetensors"
    saved = save_package(package, path, confirmed=True)
    restored = load_package(path, expected_sha256=saved["sha256"])
    refs, _ = runtime.capture_set([restored], '[{"role_id":"speaker_A","visual":false,"voice":true}]')
    fresh = component("audio_vae")
    assert fresh.crop_input is True and runtime.portable_producer(fresh, "audio_vae") != producer
    original = restored.manifest_json, restored.tensors["voice"].clone(), path.read_bytes()
    inputs = refs.conditioning_inputs(None, fresh)
    assert fresh.crop_input is False and inputs["actual_producers"]["audio_vae"] == producer
    assert inputs["refs"][0]["audio_latent"] is restored.tensors["voice"]
    clip = FakeClip()
    output = build_conditioning(clip, component("video_vae"), fresh, '<Audio 1> says "你好。"',
        32, 32, 124, audio_mode="native", prepared_reference_set=refs, return_text_recipe=True)
    assert len(clip.tokenize_calls) == 1 and output[2] is None
    assert json.loads(output[4])["audios"] == {"1": "role:speaker_A/voice"}
    assert restored.manifest_json == original[0] and torch.equal(restored.tensors["voice"], original[1])
    assert path.read_bytes() == original[2] and not torch.cuda.is_initialized()


def test_audio_reload_still_rejects_changed_weights_before_qwen():
    package, _ = voice_package()
    refs, _ = runtime.capture_set([package], '[{"role_id":"speaker_A","visual":false,"voice":true}]')
    fresh = component("audio_vae")
    with torch.no_grad():
        fresh.first_stage_model.probe.add_(1)
    clip = FakeClip()
    with pytest.raises(ValueError, match="encoder differs"):
        build_conditioning(clip, component("video_vae"), fresh, "<Audio 1>", 32, 32, 124,
                           prepared_reference_set=refs)
    assert not clip.tokenize_calls


def test_deselected_voice_and_wrong_vae_are_not_mutated_or_certified():
    package, _ = voice_package()
    fresh = component("audio_vae")
    refs, _ = runtime.capture_set([package], '[{"role_id":"speaker_A","visual":false,"voice":false}]')
    assert refs.conditioning_inputs(None, fresh)["actual_producers"] == {}
    assert fresh.crop_input is True
    selected, _ = runtime.capture_set([package], '[{"role_id":"speaker_A","visual":false,"voice":true}]')
    video = component("video_vae")
    with pytest.raises(ValueError, match="wrong native H3 network"):
        selected.conditioning_inputs(None, video)
    assert video.crop_input is True
