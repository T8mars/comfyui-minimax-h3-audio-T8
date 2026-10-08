"""Observed native media/recipe contracts with tiny encoders, not weight/GPU QA."""
from dataclasses import replace
import json

import pytest
import torch

from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_reference_runtime import CountingClip
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.nodes_visual_marker import MiniMaxH3VisualMarkerConditioningEXPT8
from h3_audio_t8_pkg.reference_package import tensor_record
from h3_audio_t8_pkg.temporal_dialogue_encoding import validate_native_text_recipe
from h3_audio_t8_pkg.visual_marker import DEFAULT_MARKERS, prepare_marker_prompt, prepare_markers
from h3_audio_t8_pkg.visual_marker_binding import insert_marker_reference


def material(mode="render_rectangles"):
    rgb = torch.full((1, 64, 96, 3), .25)
    marked, plan, _ = prepare_markers(rgb, DEFAULT_MARKERS, mode=mode)
    recipe = prepare_marker_prompt(plan, "A quiet room. <d>你好，今天真不错。</d>")[3]
    return rgb, marked, recipe


def build(recipe, *, other_images=None, first_frame=None, last_frame=None,
          policy="marked_vae_and_qwen", marker_position=1, clean_image=None, clip=None, **kwargs):
    video, audio, encoder = FakeVideoVAE(), FakeAudioVAE(), clip or CountingClip()
    images, binding = insert_marker_reference(recipe, other_images or [], marker_position=marker_position,
        policy=policy, clean_image=clean_image)
    result = build_conditioning(encoder, video, audio, recipe.base_prompt, 64, 64, 124,
        audio_mode="native", ref_images={f"ref_image_{i}": value for i, value in enumerate(images)},
        first_frame=first_frame, last_frame=last_frame, visual_marker_binding=binding,
        return_text_recipe=True, **kwargs)
    return result, encoder, video, audio, binding


def test_actual_multi_image_keyframes_offset_binds_before_one_fresh_native_encode():
    clean, _, recipe = material()
    out, clip, video, _, _ = build(recipe, other_images=[clean], marker_position=2,
        first_frame=clean, last_frame=clean)
    native = out[-1]["text_recipe"]
    receipt = native.metadata["t8_visual_marker_binding"]
    assert receipt["native_picture_ordinal"] == 4 and receipt["keyframe_picture_count"] == 2
    assert receipt["binding_verified"] and receipt["generated_legend_binding_verified"]
    assert "<Picture 4>" in clip.tokenize_calls[0][0] and "<Picture 4>" in out[3]
    assert clip.encode_count == 1 and len(video.encode_calls) == 4
    assert json.loads(out[4])["pictures"]["4"] == "ref_image_2"
    assert native.tokenize_kwargs["minimax_ref_items"][3]["type"] == "image"
    assert receipt["per_image_processor_grid"] is None and not receipt["generated_coordinate_lock"]
    validate_native_text_recipe(native)


def test_split_actual_clean_to_vae_marked_to_qwen_no_second_vae_encode():
    clean, marked, recipe = material()
    out, clip, video, audio, _ = build(recipe, policy="clean_vae_marked_qwen")
    receipt = out[-1]["text_recipe"].metadata["t8_visual_marker_binding"]
    assert len(video.encode_calls) == 1 and len(audio.encode_calls) == 0 and clip.encode_count == 1
    from h3_audio_t8_pkg.conditioning import _resize_reference_image
    expected, _, _ = _resize_reference_image(clean, 64, 64, "match")
    assert torch.equal(video.encode_calls[0], expected)
    qwen = clip.tokenize_calls[0][1]["minimax_ref_items"][0]["data"]
    assert not torch.equal(qwen, video.encode_calls[0])
    assert receipt["actual_qwen_rgb"] == tensor_record(qwen)
    assert receipt["actual_vae_rgb"] == tensor_record(video.encode_calls[0])
    assert receipt["source_marked_rgb"] == tensor_record(marked)
    assert receipt["geometry_observation"] == "same_source_rendered_rectangles"
    assert torch.equal(clean, recipe.plan.clean_image)
    assert not receipt["marker_removal_guaranteed"]


def test_default_same_marked_input_both_encoders_and_old_native_latents_exact():
    _, _, recipe = material()
    out, clip, video, _, _ = build(recipe)
    qwen = clip.tokenize_calls[0][1]["minimax_ref_items"][0]["data"]
    assert qwen is video.encode_calls[0]
    old = build_conditioning(CountingClip(), FakeVideoVAE(), FakeAudioVAE(), out[3],
        64, 64, 124, audio_mode="native", ref_images={"ref_image_0": recipe.plan.marked_image})
    for part, old_part in zip(out[1]["samples"].unbind(), old[1]["samples"].unbind(), strict=True):
        assert torch.equal(part, old_part)
    assert "t8_visual_marker_binding" not in old[0][0][1]
    assert out[2] is None


def test_reference_video_audio_and_source_final_are_not_replaced():
    _, _, recipe = material()
    soundtrack, drive, final = make_audio(.1), make_audio(.2), make_audio(.3)
    video_frames = torch.full((48, 64, 64, 3), .4)
    out, clip, _, _, _ = build(recipe, drive_audio=drive, final_audio=final,
        ref_videos={"ref_video_0": video_frames}, ref_video_audios={"ref_video_audio_0": soundtrack},
        ref_audios={"ref_audio_0": soundtrack})
    mapping = json.loads(out[4])
    assert out[2] is final and len(mapping["videos"]) == 1 and len(mapping["audios"]) == 3
    items = clip.tokenize_calls[0][1]["minimax_ref_items"]
    assert [x["type"] for x in items] == ["image", "audio", "video", "audio", "audio"]
    assert items[2]["timestamps"] == [0., .5, 1., 1.5]


def test_hand_drawn_split_requires_actual_clean_rgb_not_automatic_erasure():
    clean, _, recipe = material("provided_marked")
    with pytest.raises(ValueError, match="actual clean image"):
        build(recipe, policy="clean_vae_marked_qwen")
    out, _, _, _, _ = build(recipe, policy="clean_vae_marked_qwen", clean_image=clean)
    receipt = out[-1]["text_recipe"].metadata["t8_visual_marker_binding"]
    assert receipt["geometry_observation"] == "user_declared_same_size_not_registered"
    with pytest.raises(ValueError, match="dimensions differ"):
        build(recipe, policy="clean_vae_marked_qwen", clean_image=clean[:, :, :64])


def test_rendered_split_cannot_claim_an_unrelated_same_size_clean_source():
    clean, _, recipe = material()
    with pytest.raises(ValueError, match="actual recorded clean source"):
        build(recipe, policy="clean_vae_marked_qwen", clean_image=clean + .1)
    with pytest.raises(ValueError, match="only belongs"):
        build(recipe, clean_image=clean)


def test_original_manual_text_not_rewritten_despite_actual_media_binding():
    clean, _, recipe = material()
    manual = prepare_marker_prompt(recipe.plan, "Original <Picture 1> prompt.", mode="manual_original")[3]
    out, clip, _, _, _ = build(manual, first_frame=clean)
    receipt = out[-1]["text_recipe"].metadata["t8_visual_marker_binding"]
    assert clip.tokenize_calls[0][0] == "Original <Picture 1> prompt."
    assert receipt["native_picture_ordinal"] == 2 and not receipt["generated_legend_binding_verified"]


def test_bad_insertion_or_changed_actual_binding_source_is_rejected():
    _, _, recipe = material()
    with pytest.raises(ValueError, match="real insertion slot"):
        insert_marker_reference(recipe, [], marker_position=2)
    images, binding = insert_marker_reference(recipe, [])
    images[0] = images[0].clone()
    images[0][0, 0, 0, 0] = .5
    with pytest.raises(ValueError, match="not the marker binding source"):
        build_conditioning(CountingClip(), FakeVideoVAE(), FakeAudioVAE(), "", 64, 64, 124,
            ref_images={"ref_image_0": images[0]}, visual_marker_binding=binding)
    with pytest.raises(ValueError, match="selection changed"):
        replace(binding, raw_image_index=2).verify()


def test_qwen_mutation_is_a_real_error_and_recipe_receipt_is_content_bound():
    _, _, recipe = material()

    class MutatingClip(CountingClip):
        def encode_from_tokens_scheduled(self, tokens):
            tokens["kwargs"]["minimax_ref_items"][0]["data"][0, 0, 0, 0] += .1
            return super().encode_from_tokens_scheduled(tokens)

    with pytest.raises(ValueError, match="Qwen pixels changed"):
        build(recipe, clip=MutatingClip())
    out, _, _, _, _ = build(recipe)
    native = out[-1]["text_recipe"]
    native.metadata["t8_visual_marker_binding"]["native_picture_ordinal"] = 9
    with pytest.raises(ValueError, match="content changed"):
        validate_native_text_recipe(native)


def test_adapter_native_schema_and_execute_return_real_recipe_and_binding():
    _, _, recipe = material()
    result = MiniMaxH3VisualMarkerConditioningEXPT8.execute(recipe, CountingClip(), FakeVideoVAE(),
        FakeAudioVAE(), 64, 64, 124)
    assert json.loads(result[-1])["binding_verified"]
    validate_native_text_recipe(result[-2])
    info = MiniMaxH3VisualMarkerConditioningEXPT8.define_schema()
    assert info.is_experimental and len(info.outputs) == 8
    assert "prompt_recipe" in [x.id for x in info.inputs] and "prompt" not in [x.id for x in info.inputs]
