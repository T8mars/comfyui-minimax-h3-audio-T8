"""Real valid-choice and input-integrity counterexamples for new MC operators."""
import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import mc_seam_experimental as mc
from test_h21_mc_seam import latent, parts, preserved


def test_join_single_context_batch_can_broadcast_without_blending_distinct_batches():
    source, previous = latent(batch=2), latent(batch=1, seed=13)
    before = parts(source)[0].clone()
    output, report = mc.join_match(source, previous, enabled=True, min_excess=0,
                                  tuning="manual", match_detail=False, match_spread=False)
    expected = []
    for batch in range(2):
        single = {"samples": NestedTensor(tuple(value[batch:batch + 1] for value in parts(source)))}
        value, _ = mc.join_match(single, previous, enabled=True, min_excess=0,
                                 tuning="manual", match_detail=False, match_spread=False)
        expected.append(parts(value)[0])
    assert torch.equal(parts(output)[0], torch.cat(expected))
    assert parts(output)[1] is parts(source)[1]
    assert report["context_batch_broadcast"]
    assert torch.equal(parts(source)[0], before)
    preserved(source, output)


@pytest.mark.parametrize("original_dtype,refined_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32)])
def test_refine_keeps_real_PyTorch_mixed_float_promotion_then_original_output_dtype(original_dtype, refined_dtype):
    source, refined = latent(), latent(seed=14)
    source["samples"].tensors[0] = parts(source)[0].to(original_dtype)
    refined["samples"].tensors[0] = parts(refined)[0].to(refined_dtype)
    before, target = parts(source)[0], parts(refined)[0]
    expected = before.clone()
    for index, weight in enumerate(mc.refine_weights(before.shape[2], 22, 22)):
        if weight == 1:
            expected[:, :, index] = target[:, :, index]
        elif weight > 0:
            expected[:, :, index] = (1 - weight) * before[:, :, index] + weight * target[:, :, index]
    output, report = mc.refine_blend(source, refined, enabled=True)
    assert torch.equal(parts(output)[0], expected)
    assert parts(output)[0].dtype == original_dtype
    assert parts(output)[1] is parts(source)[1]
    assert report["refined_video_dtype"] == str(refined_dtype)
    preserved(source, output)


@pytest.mark.parametrize("different_shape", [False, True])
def test_explicit_refined_audio_is_selected_not_mixed_with_original_or_dtype_gated(different_shape):
    source, refined = latent(), latent(seed=15)
    sound = parts(refined)[1].double()
    if different_shape:
        sound = torch.linspace(-1, 1, 1 * 7 * 2 * 97, dtype=torch.float64).reshape(1, 7, 2, 97)
    refined["samples"].tensors[1] = sound
    output, report = mc.refine_blend(source, refined, enabled=True, audio="refined")
    assert parts(output)[1] is sound
    assert report["audio_changed"] and report["selected_audio_shape"] == list(sound.shape)
    assert parts(source)[1] is not sound
    preserved(source, output)


@pytest.mark.parametrize("table,strength", [([1., .5, .7, 0.], .5), ([1., .8, .6, .4, 0.], -2.)])
def test_explicit_computable_sigma_choice_is_not_rejected_by_widget_intent(table, strength):
    source = torch.tensor(table, dtype=torch.float64)
    before = source.clone()
    expected, power = list(table), 1 / (1 + strength)
    start, end = max(1, len(table) - 1 - 3), len(table) - 1
    anchor = table[start - 1]
    for index in range(start, end):
        expected[index] = anchor * (table[index] / anchor) ** power
    for index in range(start, end):
        if expected[index] >= expected[index - 1]:
            expected[index] = expected[index - 1] * .999
    output, report = mc.sigma_tail(source, enabled=True, strength=strength)
    assert torch.equal(output, source.new_tensor(expected))
    assert torch.equal(source, before)
    assert report["warning"] and not report["portable_cache_identity"]
    assert report["input_nonincreasing"] == all(a >= b for a, b in zip(table, table[1:]))
    assert report["positive_power"] == (power > 0)


def test_bad_nested_shallow_copy_does_not_overwrite_the_original_container():
    class BrokenCopy(NestedTensor):
        def __copy__(self):
            return self
    source, refined = latent(kind=BrokenCopy), latent(kind=BrokenCopy, seed=16)
    original_container = source["samples"]
    original_video, original_sound = parts(source)
    before, storage = original_video.clone(), original_container.tensors
    error, output = None, None
    try:
        output, _ = mc.refine_blend(source, refined, enabled=True)
    except ValueError as actual_error:
        error = actual_error
    assert source["samples"] is original_container and original_container.tensors is storage
    assert parts(source)[0] is original_video and parts(source)[1] is original_sound
    assert torch.equal(original_video, before)
    if error is not None:
        assert "copy" in str(error) and "original" in str(error) or "fresh" in str(error)
    else:
        assert output["samples"] is not original_container
        assert parts(output)[1] is original_sound
        preserved(source, output)
