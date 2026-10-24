"""Actual private diagnostic qualification; no trained model or new GPU."""
import importlib.util
import json
from pathlib import Path
import sys

import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg import multikeyframe_advanced as multi
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects
from h3_audio_t8_pkg.modular_sampling.hyperflow_identity import model_identity
from test_h21_hyperflow_keyframe_identity import actual_base, actual_encoder
from test_h21_hyperflow_keyframe_runtime import prepare, check_original

HERE = Path(__file__).resolve().parent / 'fixtures/h21_observers'
spec = importlib.util.spec_from_file_location("_h21_D_consumption_profile", HERE / "observer.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
actual_base, actual_encoder = actual_base, actual_encoder


def profiler(network):
    return observer.ConsumptionProfiler(network, hyperflow_runtime=runtime, advanced_conditioning=multi)


def pair(base, weights, encoder, nonuniform=False):
    return [prepare(base, weights, encoder, relay=False, nonuniform=nonuniform) for _ in range(2)]


def run(low, high, before_head=None, before_tail=None):
    callbacks = []
    model, positive, source = low[:3]
    head = effects.bind_head(model, source, positive, [], 4)
    head_identity = model_identity(head[0])
    if before_head:
        before_head(head[0])
    boundary = stages.sample_head(head[0], source, positive, [], comfy.sample.prepare_noise(source["samples"], 21102195),
        seed=21102195, callback=lambda step, _p, _x, total: callbacks.append((step, total)))
    assert model_identity(head[0]) == head_identity
    tail = effects.bind_tail(boundary, high[0], high[1], [])
    tail_identity = model_identity(tail[0])
    if before_tail:
        before_tail(tail[0])
    output, text = stages.sample_tail(boundary, tail[0], tail[1], [], seed=21102195,
        callback=lambda step, _p, _x, total: callbacks.append((step, total)))
    assert model_identity(tail[0]) == tail_identity
    return output, boundary.verify(), json.loads(text), callbacks, (head_identity, tail_identity)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_profile_real_HF_HEAD_TAIL_identity_raw_consumption_and_exact_next_request(
        actual_base, actual_encoder, tmp_path, nonuniform):
    base, weights, original = actual_base
    low, high = pair(base, weights, actual_encoder, nonuniform)
    baseline = run(low, high)
    low, high = pair(base, weights, actual_encoder, nonuniform)
    before_source = [value.clone() for value in low[2]["samples"].unbind()]
    before_patches = dict(low[0].object_patches)
    network = base.model.diffusion_model
    hook_state = {name: (dict(module._forward_pre_hooks), dict(module._forward_hooks)) for name, module in network.named_modules()}
    assert sys.getprofile() is None
    with profiler(network) as observed:
        actual = run(low, high)
    assert sys.getprofile() is None
    report = observed.report()
    (tmp_path / "actual-consumption.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    assert report["complete_target_coverage"], report["diagnostics"]
    assert len(report["records"]) == 8
    assert [row["HF_absolute_interval"] for row in report["records"]] == list(range(8))
    assert all([row["frame_index"] for row in record["keyframes"]] == [0, 1, 21] for record in report["records"])
    assert all([part["frame_index"] for part in record["actual_keyframe_projection_segments"]] == [0, 1, 21]
               for record in report["records"])
    assert all(row["execution"]["portable_identity"] is True for row in actual[1:3])
    assert actual[3] == baseline[3] == [(index, 8) for index in range(8)]
    assert actual[4] == baseline[4]
    assert actual[1]["execution"] == baseline[1]["execution"]
    assert actual[2]["execution"] == baseline[2]["execution"]
    for value, expected in zip(actual[0]["samples"].unbind(), baseline[0]["samples"].unbind(), strict=True):
        assert torch.equal(value, expected)
    for before, after in zip(before_source, low[2]["samples"].unbind(), strict=True):
        assert torch.equal(before, after)
    assert low[0].object_patches == before_patches
    assert hook_state == {name: (dict(module._forward_pre_hooks), dict(module._forward_hooks)) for name, module in network.named_modules()}
    low, high = pair(base, weights, actual_encoder, nonuniform)
    next_result = run(low, high)
    assert all(torch.equal(a, b) for a, b in zip(next_result[0]["samples"].unbind(), baseline[0]["samples"].unbind(), strict=True))
    check_original(base, original)
    assert not torch.cuda.is_initialized()


def test_module_hook_design_really_loses_portable_identity_not_filtered(actual_base, actual_encoder):
    base, weights, original = actual_base
    low = pair(base, weights, actual_encoder)[0]
    selected = effects.bind_head(low[0], low[2], low[1], [], 4)[0]
    assert model_identity(selected)["portable_cache_reuse"] is True
    calls = []
    handle = base.model.diffusion_model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        with pytest.raises(ValueError, match="shared live hooks"):
            model_identity(selected)
        boundary = stages.sample_head(selected, low[2], low[1], [], comfy.sample.prepare_noise(low[2]["samples"], 21102195), seed=21102195)
        assert boundary.verify()["execution"]["portable_identity"] is False
        assert len(calls) == 4
    finally:
        handle.remove()
    check_original(base, original)


def test_profile_restores_on_actual_sampler_interruption_and_next_plain(actual_base, actual_encoder):
    base, weights, original = actual_base
    low = pair(base, weights, actual_encoder)[0]
    network = base.model.diffusion_model
    model = effects.bind_head(low[0], low[2], low[1], [], 4)[0]
    before = model_identity(model)

    def stop(step, _p, _x, _total):
        if step == 1:
            raise RuntimeError("owned actual sampler interruption")

    with pytest.raises(RuntimeError, match="owned actual sampler interruption"):
        with profiler(network):
            stages.sample_head(model, low[2], low[1], [], comfy.sample.prepare_noise(low[2]["samples"], 21102195), seed=21102195, callback=stop)
    assert sys.getprofile() is None
    assert model_identity(model) == before
    retry = run(*pair(base, weights, actual_encoder))
    assert retry[1]["execution"]["portable_identity"] is True
    check_original(base, original)


def test_prior_profiler_is_preserved_not_replaced(actual_base):
    base, _weights, _original = actual_base

    def existing(_frame, _event, _value):
        return None

    sys.setprofile(existing)
    try:
        with pytest.raises(ValueError, match="Existing thread profiler retained"):
            with profiler(base.model.diffusion_model):
                raise AssertionError("must not enter")
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize("size", [32, 64])
def test_original_native_sampler_actual_low_high_shapes_same_math_no_extra_encodes(size, monkeypatch):
    from comfy.ldm.minimax import model as native
    from comfy.ldm.modules import attention
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    from h3_audio_t8_pkg import sampling
    from test_fast_h3_v2_core_sampler import model as real_tiny_model
    from test_h21_control_regions import RealTinyEncoderOwner
    from test_h21_director_keyframe_dual_runtime import conditioning
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    base, vae = real_tiny_model(), RealTinyEncoderOwner()
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    built = conditioning(base, vae, image, size, False)
    count = len(vae.calls)

    def sample():
        model, sampler, sigmas = sampling.setup_dual_clock_sampling(built[0], built[2], 2, 12., 3.)
        guider = BasicGuider.execute(model, built[1]).result[0]
        return SamplerCustomAdvanced.execute(RandomNoise.execute(21102191).result[0], guider, sampler, sigmas, built[2]).result[0]

    baseline = sample()
    with profiler(base.model.diffusion_model) as actual:
        output = sample()
    report = actual.report()
    assert report["complete_target_coverage"], report["diagnostics"]
    assert len(report["records"]) == 2
    assert all(row["video_shape"][-2:] == [size // 16, size // 16] for row in report["records"])
    assert len(vae.calls) == count
    assert all(torch.equal(a, b) for a, b in zip(output["samples"].unbind(), baseline["samples"].unbind(), strict=True))
    assert sys.getprofile() is None and not torch.cuda.is_initialized()
