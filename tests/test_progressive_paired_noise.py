"""Noise mathematics and real tiny Core LOW/HIGH consumers, not image quality."""
import asyncio
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg.progressive_noise import haar_half, prepare_paired_noise
from h3_audio_t8_pkg.nodes_progressive_noise import MiniMaxH3ProgressivePairedNoiseEXPT8 as Node
from h3_audio_t8_pkg.modular_sampling import progressive_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from test_modular_progressive_stages import inputs, initialized
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


def pair(dtype=torch.float32, indices=None):
    batch = 1 if indices is None else len(indices)
    audio = torch.zeros(batch, 32, 2, 8, dtype=dtype)
    high = {"samples": NestedTensor((torch.zeros(batch, 24, 2, 4, 8, dtype=dtype), audio))}
    low = {"samples": NestedTensor((torch.zeros(batch, 24, 2, 2, 4, dtype=dtype), audio.clone()))}
    if indices is not None:
        high["batch_index"] = indices
        low["batch_index"] = list(indices)
    return low, high


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("indices", [None, [2, 0, 2]])
def test_independent_is_exact_core_noise_with_private_rng(dtype, indices):
    low, high = pair(dtype, indices)
    lo, hi, text = prepare_paired_noise(low, high, 2**64-1)
    expected_low = Noise_RandomNoise(2**64-1).generate_noise(low)
    high_video = {"samples": high["samples"].unbind()[0]}
    if indices is not None:
        high_video["batch_index"] = indices
    expected_high = Noise_RandomNoise(0).generate_noise(high_video)
    before = torch.random.get_rng_state().clone()
    actual_low = lo.generate_noise(low)
    # Existing HIGH handoff passes only samples; captured indices must survive.
    actual_high = hi.generate_noise({"samples": high_video["samples"]})
    for actual, expected in zip(actual_low.unbind(), expected_low.unbind()):
        assert torch.equal(actual, expected)
    assert torch.equal(actual_high, expected_high)
    assert torch.equal(before, torch.random.get_rng_state())
    assert json.loads(text)["high_seed"] == 0


def test_haar_projects_master_and_keeps_audio_draw_unscaled():
    low, high = pair()
    lo, hi, _ = prepare_paired_noise(low, high, 31, "haar")
    bl, _, _ = prepare_paired_noise(low, high, 31, "bilinear")
    master = Noise_RandomNoise(31).generate_noise(high)
    expected_video, expected_audio = master.unbind()
    before = torch.random.get_rng_state().clone()
    video, audio = lo.generate_noise(low).unbind()
    expected = (expected_video[..., 0::2, 0::2] + expected_video[..., 0::2, 1::2]
                + expected_video[..., 1::2, 0::2] + expected_video[..., 1::2, 1::2]) / 2
    torch.testing.assert_close(video, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(audio, expected_audio)
    assert torch.equal(audio, bl.generate_noise(low).unbind()[1])
    assert torch.equal(expected_video, hi.generate_noise({"samples": high["samples"].unbind()[0]}))
    assert torch.equal(before, torch.random.get_rng_state())
    # Caller modification of a returned tensor cannot poison a reused provider.
    video.fill_(999)
    assert not torch.equal(video, lo.generate_noise(low).unbind()[0])


def test_haar_unit_variance_is_not_bilinear_attenuation():
    generator = torch.Generator(device="cpu").manual_seed(123)
    master = torch.randn((1, 1, 2, 512, 512), generator=generator)
    projected = haar_half(master, (256, 256))
    assert abs(projected.mean().item()) < .02
    assert abs(projected.var().item() - 1) < .02
    # For iid noise, a block's sum/2 has correlation 1/2 with each constituent.
    covariance = (projected * master[..., ::2, ::2]).mean().item()
    assert abs(covariance - .5) < .02


def test_invalid_geometry_and_changed_inputs_fail_before_sampling():
    low, high = pair()
    odd_high = {"samples": NestedTensor((torch.zeros(1, 24, 2, 6, 8), high["samples"].unbind()[1]))}
    with pytest.raises(ValueError, match="exactly half"):
        prepare_paired_noise(low, odd_high, 1, "haar")
    lo, hi, _ = prepare_paired_noise(low, high, 1, "haar")
    with pytest.raises(ValueError, match="geometry"):
        hi.generate_noise({"samples": torch.zeros(1, 24, 2, 4, 10)})
    with pytest.raises(ValueError, match="batch_index"):
        lo.generate_noise({**low, "batch_index": [1]})
    indexed_low, indexed_high = pair(indices=[1])
    indexed, _, _ = prepare_paired_noise(indexed_low, indexed_high, 1)
    with pytest.raises(ValueError, match="batch_index"):
        indexed.generate_noise({"samples": indexed_low["samples"]})
    with pytest.raises(ValueError, match="unsigned"):
        prepare_paired_noise(low, high, True)


@pytest.mark.parametrize("mode", ["independent", "haar", "bilinear"])
@pytest.mark.parametrize("kind", ["t2va", "i2va", "avatar"])
def test_public_noise_nodes_drive_both_real_core_stages(stub_lifter, mode, kind):
    case = (inputs(source=initialized("avatar"), mode="initialized_av_exp")
            if kind == "avatar" else inputs(kind))
    original = stages.snapshot(case["source"])
    lo, hi, text = Node.execute(case["low_source"], case["source"], 7, mode).result
    report = json.loads(text)
    assert report["sampling_calls"] == 0 and report["noise_materialized"] is False
    boundary, low_report = nodes.MiniMaxH3ProgressiveLowStageEXPT8.execute(
        case["model"], case["sampler"], lo, case["plan"], case["low_source"], case["lp"], case["ln"]).result
    assert json.loads(low_report)["execution"]["actual_apply_calls"] == 2
    lift_input, _, _, plan, _ = nodes.MiniMaxH3ProgressiveLiftInputEXPT8.execute(
        boundary, case["model"], case["sampler"]).result
    video, audio = lift_input["samples"].unbind()
    lifted, _ = stages.legacy._lift_video(video, audio, plan, {"test_double": True})
    high_state, _, _ = nodes.MiniMaxH3ProgressiveHighHandoffEXPT8.execute(
        boundary, case["model"], case["sampler"], {"samples": NestedTensor((lifted, audio))},
        case["source"], hi).result
    output, high_report, completed = nodes.MiniMaxH3ProgressiveHighStageEXPT8.execute(
        high_state, case["model"], case["sampler"], case["hp"], case["hn"], 7).result
    assert json.loads(high_report)["execution"]["actual_apply_calls"] == 2
    assert completed.verify()["sampling"] == json.loads(high_report)
    assert stages.snapshot(case["source"]) == original
    assert all(torch.isfinite(p).all() for p in output["samples"].unbind())
    assert not case["model"].wrappers
    if kind == "avatar":
        torch.testing.assert_close(output["samples"].unbind()[1], case["source"]["samples"].unbind()[1], rtol=0, atol=1e-6)
    if mode == "independent":
        expected, _ = stages.legacy.sample_progressive_h3(
            case["model"], case["positive"], case["negative"], case["source"], case["sampler"], case["sigmas"],
            upscaler_model="test", seed=7, low_evaluations=case["low"], task=case["task"], input_mode=case["mode"])
        for left, right in zip(expected["samples"].unbind(), output["samples"].unbind()):
            assert torch.equal(left, right)


def test_registration_appends_to_all_published_node_ids():
    previous = asyncio.run(package._PikaTwoPassExtension().get_node_list())
    current = asyncio.run(package.comfy_entrypoint().get_node_list())
    assert current[:len(previous)] == previous
    assert current[len(previous)] is Node
    assert len({cls.define_schema().node_id for cls in current}) == len(current)
