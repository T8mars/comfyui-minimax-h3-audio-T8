"""Tiny synthetic encoders + real Core processor/tokenizer, not pretrained QA."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_reference_runtime import CountingClip
from h3_audio_t8_pkg import reference_runtime as runtime
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.nodes_qwen_reference_view import MiniMaxH3ReferenceQwenViewEXPT8
from h3_audio_t8_pkg.qwen_reference_view import (
    QwenImageView, observed_encoding, select_view, verify_view_inputs, view_dimensions,
)
from h3_audio_t8_pkg.reference_package import tensor_record
from h3_audio_t8_pkg.temporal_dialogue_encoding import validate_native_text_recipe


@pytest.fixture
def scene(monkeypatch):
    video, audio = FakeVideoVAE(), FakeAudioVAE()

    def identity(value, role):
        assert value in (video, audio)
        return {"scope": runtime.PRODUCER_SCOPE,
                "sha256": hashlib.sha256((role+str(getattr(value, "revision", 0))).encode()).hexdigest()}

    monkeypatch.setattr(runtime, "native_producer_identity", identity)
    rgb = torch.linspace(0, 1, 96 * 160 * 3).reshape(1, 96, 160, 3)
    package, _ = runtime.create_package("A", frames=rgb, width=160, height=96,
        audio=make_audio(.1), video_vae=video, audio_vae=audio)
    refs, _ = runtime.capture_set([package], '[{"role_id":"A","visual":true,"voice":true}]')
    return video, audio, package, refs


def selected(refs, **kwargs):
    return select_view(refs, mode="image_short_edge", short_edge=64,
                       max_long_edge=128, max_pixels=8192, **kwargs)


def build(scene, refs, clip=None, **kwargs):
    video, audio, _, _ = scene
    return build_conditioning(clip or CountingClip(), video, audio, "<Picture 1> speaks.",
        32, 32, 124, audio_mode="native", prepared_reference_set=refs,
        return_text_recipe=True, **kwargs)


def test_original_exact_identity_and_output_default(scene):
    video, audio, package, refs = scene
    same, report = select_view(refs)
    assert same is refs and report["policy"] is None
    inputs = same.conditioning_inputs(video, audio)
    assert "qwen_view" not in inputs and inputs["qwen_ref_items"][0]["data"] is package.tensors["visual_rgb"]
    out = build(scene, same)
    assert "t8_qwen_reference_view" not in out[0][0][1] and "qwen_reference_view=" not in out[5]
    changed, _ = selected(refs)
    restored, _ = select_view(changed)
    assert restored.qwen_view is None and restored.packages is refs.packages


def test_only_qwen_shrinks_latent_voice_source_producer_stay_exact(scene):
    video, audio, package, refs = scene
    manifest = package.manifest_json
    before = {key: tensor_record(value) for key, value in package.tensors.items()}
    smaller, report = selected(refs)
    inputs = smaller.conditioning_inputs(video, audio)
    verify_view_inputs(inputs)
    assert inputs["qwen_ref_items"][0]["data"].shape == (1, 64, 106, 3)
    assert inputs["refs"][0]["latent"] is package.tensors["visual_latent"]
    assert inputs["refs"][1]["audio_latent"] is package.tensors["voice_latent"]
    assert inputs["qwen_ref_items"][1] == {"type": "audio"}
    assert smaller.packages is refs.packages and refs.qwen_view is None
    assert package.manifest_json == manifest and before == {k: tensor_record(v) for k, v in package.tensors.items()}
    assert len(video.encode_calls) == len(audio.encode_calls) == 1
    assert report == inputs["qwen_view"] and not report["VAE_latents_reencoded"]


def test_short_edge_not_long_edge_caps_and_no_upscale():
    strategy = QwenImageView(512, 2048, 1048576, "a"*64)
    assert view_dimensions(1280, 768, strategy) == (853, 512)
    assert view_dimensions(768, 1280, strategy) == (512, 853)
    assert view_dimensions(64, 96, strategy) == (64, 96)
    assert view_dimensions(1280, 768, replace(strategy, max_long_edge=640)) == (640, 384)
    dims = view_dimensions(1280, 768, replace(strategy, max_pixels=65536))
    assert dims[0] * dims[1] <= 65536 and dims[0] <= 1280 and dims[1] <= 768
    with pytest.raises(ValueError, match="collapse"):
        view_dimensions(2048, 32, replace(strategy, max_long_edge=512))


@pytest.mark.parametrize("changes", [
    {"short_edge": True}, {"short_edge": 31}, {"max_long_edge": 32},
    {"max_pixels": float("nan")}, {"max_pixels": 2**40}, {"source_set_sha256": "label"},
])
def test_invalid_budgets_rejected_before_resize(changes):
    with pytest.raises(ValueError, match="bounded integer"):
        replace(QwenImageView(64, 128, 8192, "a"*64), **changes).validate()


def test_small_images_retain_tensor_not_requantized(scene):
    video, audio, package, refs = scene
    smaller, _ = select_view(refs, mode="image_short_edge")
    assert smaller.conditioning_inputs(video, audio)["qwen_ref_items"][0]["data"] is package.tensors["visual_rgb"]


def test_offscreen_video_timestamps_and_ordinals_unchanged(scene):
    video, audio, a, _ = scene
    b, _ = runtime.create_package("B", frames=torch.ones((1, 96, 160, 3)),
        width=160, height=96, video_vae=video)
    v, _ = runtime.create_package("V", kind="video", frames=torch.ones((22, 32, 32, 3)),
        width=32, height=32, frame_limit=22, video_vae=video)
    refs, _ = runtime.capture_set([a, b, v], json.dumps([
        {"role_id": "A", "visual": False, "voice": True},
        {"role_id": "V", "visual": True, "voice": False},
        {"role_id": "B", "visual": True, "voice": False}]))
    smaller, report = selected(refs)
    old, new = refs.conditioning_inputs(video, audio), smaller.conditioning_inputs(video, audio)
    assert old["mapping"] == new["mapping"] and old["packed_reference_rows"] == new["packed_reference_rows"]
    assert [x["type"] for x in new["qwen_ref_items"]] == ["audio", "video", "image"]
    assert new["qwen_ref_items"][1]["timestamps"] == old["qwen_ref_items"][1]["timestamps"]
    assert torch.equal(new["qwen_ref_items"][1]["data"], old["qwen_ref_items"][1]["data"])
    assert [(r["role_id"], r["routed_picture_ordinal"]) for r in report["members"]] == [("B", 1)]


def test_fresh_encode_once_recipe_and_actual_keyframe_picture_ordinals(scene):
    video, audio, package, refs = scene
    smaller, _ = selected(refs)
    clip = CountingClip()
    first, last = torch.zeros((1, 32, 32, 3)), torch.ones((1, 32, 32, 3))
    final = make_audio(.2)
    out = build(scene, smaller, clip, first_frame=first, last_frame=last, final_audio=final)
    assert clip.encode_count == len(clip.tokenize_calls) == 1 and out[2] is final
    values = out[0][0][1]
    assert values["minimax_refs"][-2]["latent"] is package.tensors["visual_latent"]
    assert len(video.encode_calls) == 3 and len(audio.encode_calls) == 1  # only original + first/last
    recipe = out[-1]["text_recipe"]
    validate_native_text_recipe(recipe)
    items = recipe.tokenize_kwargs["minimax_ref_items"]
    assert items[0]["data"].shape == items[1]["data"].shape == (1, 32, 32, 3)
    assert items[2]["data"].shape == (1, 64, 106, 3)
    report = json.loads(out[5].split("qwen_reference_view=", 1)[1])
    assert report["native_picture_ordinals"] == [{"role_id": "A", "member_id": "visual", "ordinal": 3}]
    assert report["processor_grid_thw"] is None and not report["encoding_observation"]["observed"]
    assert recipe.metadata["t8_qwen_reference_view"] == values["t8_qwen_reference_view"]
    with torch.no_grad():
        items[2]["data"].add_(.1)
    with pytest.raises(ValueError, match="content changed"):
        validate_native_text_recipe(recipe)


def test_original_pixel_producer_role_and_derived_mutation_detected(scene):
    video, audio, package, refs = scene
    smaller, _ = selected(refs)
    with pytest.raises(ValueError, match="source set changed"):
        replace(smaller, roles_json='[{"role_id":"A","visual":true,"voice":false}]').conditioning_inputs(video, audio)
    inputs = smaller.conditioning_inputs(video, audio)
    inputs["qwen_ref_items"][0]["data"][0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match="pixels changed"):
        verify_view_inputs(inputs)
    video.revision = 1
    with pytest.raises(ValueError, match="encoder differs"):
        smaller.conditioning_inputs(video, audio)
    del video.revision
    package.tensors["visual_rgb"][0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match="content changed"):
        smaller.conditioning_inputs(video, audio)


def test_changed_derived_pixels_inside_qwen_rejected(scene):
    smaller, _ = selected(scene[-1])

    class MutatingClip(CountingClip):
        def encode_from_tokens_scheduled(self, tokens):
            tokens["kwargs"]["minimax_ref_items"][0]["data"].add_(.1)
            return super().encode_from_tokens_scheduled(tokens)

    with pytest.raises(ValueError, match="pixels changed"):
        build(scene, smaller, MutatingClip())


def test_native_processor_observation_and_existing_prefix_cache_key(scene):
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from h3_audio_t8_pkg.qwen_prefix_cache_advanced import fingerprint_tokens
    smaller, _ = selected(scene[-1])
    video, audio = scene[:2]
    native = MiniMaxH3Tokenizer()
    old = native.tokenize_with_weights("<Picture 1> speaks.",
        minimax_ref_items=scene[-1].conditioning_inputs(video, audio)["qwen_ref_items"])
    new_items = smaller.conditioning_inputs(video, audio)["qwen_ref_items"]
    new = native.tokenize_with_weights("<Picture 1> speaks.", minimax_ref_items=new_items)
    assert fingerprint_tokens(old, "synthetic-model") != fingerprint_tokens(new, "synthetic-model")
    assert fingerprint_tokens(new, "synthetic-model") == fingerprint_tokens(
        native.tokenize_with_weights("<Picture 1> speaks.", minimax_ref_items=new_items), "synthetic-model")
    patches, grid = process_qwen2vl_images(new_items[0]["data"], patch_size=16,
        image_mean=[.5]*3, image_std=[.5]*3)
    assert grid.tolist() == [[1, 4, 6]] and patches.shape == (24, 1536)
    observation = observed_encoding([[torch.zeros((1, 8, 8)),
                                    {"minimax_token_tags": torch.tensor([1, 0, 0, 0, 0, 0, 0, 1])}]])
    assert observation["conditionings"][0]["vision_tag_positions_including_delimiters"] == 6
    assert not observation["per_image_processor_grid_available"]


def test_selector_schema_execution_no_extra_encode_and_original_six_unchanged(scene):
    from h3_audio_t8_pkg.nodes_reference_package import NODES as old_nodes
    assert len(old_nodes) == 6
    node = MiniMaxH3ReferenceQwenViewEXPT8
    schema = node.define_schema().get_v1_info(node)
    assert schema.input["required"]["mode"][1]["default"] == "original"
    assert schema.output == ["T8_H3_REFERENCE_SET", "STRING"]
    result = node.execute(scene[-1], "image_short_edge", 64, 128, 8192).result
    assert result[0].qwen_view is not None and json.loads(result[1])["members"][0]["resized"]
    assert len(scene[0].encode_calls) == len(scene[1].encode_calls) == 1
    with pytest.raises(ValueError, match="actual routed"):
        select_view("caller label")
