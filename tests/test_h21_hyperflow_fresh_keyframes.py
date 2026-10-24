"""Real HyperFlow 8+4/partial4+4 around the complete installed learned3D.

The DiT is actual random tiny Core50+2, video Encoder is actual miniature Core,
and text/adapter/FakeAudioVAE are explicit synthetic fixtures. No GPU or media
quality claim. Neither sampling nor learned upscaling is replaced by a double.
"""
import json

from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import pytest
import torch

from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_director_keyframe_dual_runtime import (
    MODEL_NAME, conditioning, tensor_sha,
)
from test_h21_director_keyframe_dual_runtime import full_learned_checkpoint as full_learned_checkpoint
from test_h21_hyperflow_keyframe_identity import real_base
from test_h21_hyperflow_keyframe_runtime import check_original


def stage(model, positive, source, name, seed):
    prepared, sampler, sigmas, context, _ = fresh.build_stage(model, source, name)
    before = [value.clone() for value in source["samples"].unbind()]
    try:
        result = sample_stage(RandomNoise.execute(seed).result[0],
            BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, source, context)[2]
    finally:
        _restore_stage_objects(prepared)
    receipt = result.verify()
    assert receipt["verified_recipe_completion"] is True
    assert receipt["portable_identity"] is True, receipt["request"]["model"]
    assert receipt["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] == list(range(context.start, context.end))
    assert receipt["execution"]["denoiser_evaluations"] == context.steps
    assert all(torch.equal(a, b) for a, b in zip(before, source["samples"].unbind(), strict=True))
    return result


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("relay", [False, True])
def test_real_fresh_stage_middle_frames_full_learned3d_and_typed_storage(
        monkeypatch, tmp_path, partial, relay, request):
    base, weights, original = real_base(monkeypatch)
    low_owner = fresh.install(base, weights)[0]
    low_vae, high_vae = RealTinyEncoderOwner(), RealTinyEncoderOwner()
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    image_before = image.clone()
    low = conditioning(low_owner, low_vae, image, 32, relay)
    low_name = fresh.STAGES[1 if partial else 0]
    low_result = stage(low[0], low[1], low[2], low_name, 21102196)
    lifted_source, selected = fresh.lift_input(low_result)
    assert json.loads(selected)["selected_socket"] == ("denoised_output" if partial else "output")
    assert lifted_source is (low_result.denoised_output if partial else low_result.output)
    low_before = [value.clone() for value in lifted_source["samples"].unbind()]
    entry, path, calls = request.getfixturevalue("full_learned_checkpoint")
    count_before = len(calls)
    lifted, width, height, raw = learned.learned_upscale_h3_av_latent(
        lifted_source, MODEL_NAME, "target_dimensions", 2., .004096, 64, 64,
        "preserve_source", 1.05, "fp32", "offload_after")
    assert (width, height) == (64, 64)
    assert len(calls) == count_before + 1
    assert calls[-1]["input_shape"] == [1, 24, 7, 2, 2]
    assert calls[-1]["output_shape"] == [1, 24, 7, 4, 4]
    assert json.loads(raw)["model"]["sha256"] == entry.sha256
    assert path.stat().st_size == 690_592_672
    assert lifted["samples"].unbind()[1] is lifted_source["samples"].unbind()[1]
    assert all(torch.equal(a, b) for a, b in zip(low_before, lifted_source["samples"].unbind(), strict=True))
    high_owner = fresh.install(base, weights)[0]
    high = conditioning(high_owner, high_vae, image, width, relay)
    reconciled, high_positive, _ = learned.reconcile_two_pass_h3_latent(lifted, high[2], high[1], "auto")
    assert reconciled["samples"].unbind()[1] is lifted["samples"].unbind()[1]
    high_name = fresh.STAGES[3 if partial else 2]
    high_result = stage(high[0], high_positive, reconciled, high_name, 21102197)
    assert high_result.output["samples"].unbind()[0].shape == (1, 24, 7, 4, 4)
    assert len(low_vae.calls) == len(high_vae.calls) == 3
    assert [tuple(value.shape[1:3]) for value in low_vae.calls] == [(32, 32)] * 3
    assert [tuple(value.shape[1:3]) for value in high_vae.calls] == [(64, 64)] * 3
    for value, name in ((low_result, low_name), (high_result, high_name)):
        artifact, digest, _ = save_stage(value, tmp_path, name)
        loaded = load_stage(tmp_path, artifact, digest, name)[3]
        assert loaded.receipt_json == value.receipt_json
        for actual, expected in zip(loaded.output["samples"].unbind(), value.output["samples"].unbind(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(image, image_before)
    check_original(base, original)
    assert not torch.cuda.is_initialized()
    request.node.user_properties.append(("actual_hyperflow_learned_evidence", json.dumps({
        "recipe": "partial4plus4" if partial else "full8plus4", "relay": relay,
        "nfe": [4 if partial else 8, 4], "low_stage": low_name, "high_stage": high_name,
        "learned_sha256": entry.sha256, "learned_bytes": path.stat().st_size,
        "actual_lift": calls[-1], "high_video_sha256": tensor_sha(high_result.output["samples"].unbind()[0]),
        "high_audio_sha256": tensor_sha(high_result.output["samples"].unbind()[1]),
        "trained_dit_text_or_gpu_quality": False, "cuda_initialized": False}, sort_keys=True)))
