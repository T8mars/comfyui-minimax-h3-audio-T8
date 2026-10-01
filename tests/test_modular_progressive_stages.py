"""Real Core tiny LOW/HIGH parity; interpolation is NOT the learned asset.

The independent oracle remains the unchanged legacy full executor. No pretrained
weights, CUDA or trained media quality are certified by these CPU checks.
"""
import copy
from dataclasses import replace
import json

import pytest
import torch
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
from comfy.weight_adapter.lora import LoRAAdapter

from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg import progressive_sampling_runtime as legacy
from h3_audio_t8_pkg.sampling import native_flow_sigmas, setup_dual_clock_sampling
from test_progressive_sampling_runtime import tiny_model, latent, conditioning
import test_progressive_sampling_runtime as runtime_fixtures

stub_lifter = runtime_fixtures.stub_lifter


def inputs(task="t2va", steps=4, low=2, source=None, mode="empty", model=None, sigmas=None):
    source = latent() if source is None else source
    model = tiny_model() if model is None else model
    sampler = comfy.samplers.ksampler("euler")
    sigmas = native_flow_sigmas(steps, 12.) if sigmas is None else sigmas
    positive, negative = conditioning(task), conditioning()
    plan = stages.build_plan(source, sigmas, low, .5, task, mode)
    low_source = stages.prepare_low_source(source, plan, mode)
    lp, hp = legacy.prepare_stage_conditioning(positive, plan, positive=True)
    ln, hn = legacy.prepare_stage_conditioning(negative, plan, positive=False)
    return dict(source=source, model=model, sampler=sampler, sigmas=sigmas, positive=positive,
                negative=negative, plan=plan, low_source=low_source, lp=lp, hp=hp, ln=ln, hn=hn,
                mode=mode, task=task, low=low)


def low_only(case, seed=7, **kwargs):
    noise = comfy.sample.prepare_noise(case["low_source"]["samples"], seed)
    return stages.sample_low(case["model"], case["sampler"], case["plan"], case["low_source"],
                             case["lp"], case["ln"], noise, seed=seed, **kwargs)


def restart(case, boundary, model=None, seed=7, **kwargs):
    model = case["model"] if model is None else model
    template = stages.lift_input(boundary, model, case["sampler"])
    video, audio = template["samples"].unbind()
    # Same labelled interpolation test double as the independent legacy oracle.
    lifted, _ = legacy._lift_video(video, audio, case["plan"], {"test_double": True})
    high_noise = comfy.sample.prepare_noise(lifted, (seed + 1) % 2**64)
    return stages.prepare_high(boundary, model, case["sampler"],
        {"samples": comfy.nested_tensor.NestedTensor((lifted, audio))}, case["source"], high_noise, **kwargs)


def high_only(case, state, model=None, **kwargs):
    return stages.sample_high(state, case["model"] if model is None else model, case["sampler"],
                              case["hp"], case["hn"], seed=7, **kwargs)


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("steps,low", [(2, 1), (4, 3), (8, 4), (8, 6)])
@pytest.mark.parametrize("noise_scale", [.7, 1., 1.3])
def test_actual_separated_stages_equal_unchanged_legacy(stub_lifter, task, steps, low, noise_scale):
    case = inputs(task, steps, low)
    case["model"].model.model_sampling.noise_scale = noise_scale
    original, _ = legacy.sample_progressive_h3(case["model"], case["positive"], case["negative"],
        case["source"], case["sampler"], case["sigmas"], upscaler_model="test", seed=7,
        low_evaluations=low, task=task)
    notifications = []
    def callback(step, _p, _s, total):
        notifications.append((step, total))
    boundary = low_only(case, callback=callback)
    output, report = high_only(case, restart(case, boundary), callback=callback)
    for left, right in zip(original["samples"].unbind(), output["samples"].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert notifications == [(i, steps) for i in range(steps)]
    receipt, report = boundary.verify(), json.loads(report)
    assert receipt["execution"]["actual_apply_calls"] == low
    assert report["execution"]["actual_apply_calls"] == steps - low
    assert report["low_executed"] is False
    assert receipt["portable_identity"] is True
    assert case["model"].wrappers == {}


def initialized(kind):
    generator = torch.Generator().manual_seed(23)
    source = {"samples": comfy.nested_tensor.NestedTensor([
        torch.randn(x.shape, generator=generator) * .2 for x in latent()["samples"].unbind()])}
    video, audio = source["samples"].unbind()
    if kind == "none":
        return source
    vm = torch.ones_like(video[:, :1])
    am = torch.ones_like(audio[:, :1])
    if kind == "avatar":
        am.zero_()
    elif kind == "binary":
        vm[..., :4] = 0
        am[..., :4] = 0
    elif kind == "fractional":
        vm[..., :4] = .4
        am[..., :4] = .3
    elif kind == "zero":
        vm.zero_()
        am.zero_()
    source["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
    return source


@pytest.mark.parametrize("kind", ["none", "ones", "zero", "binary", "fractional", "avatar"])
def test_initialized_source_and_native_masks_equal_legacy_composed(stub_lifter, kind):
    case = inputs(source=initialized(kind), mode="initialized_av_exp", sigmas=native_flow_sigmas(5, 12.)[1:])
    before = stages.snapshot(case["source"])
    expected, _ = legacy.sample_progressive_h3(case["model"], case["positive"], case["negative"],
        case["source"], case["sampler"], case["sigmas"], upscaler_model="test", seed=7,
        low_evaluations=2, input_mode=case["mode"])
    boundary = low_only(case)
    output, _ = high_only(case, restart(case, boundary))
    for left, right in zip(expected["samples"].unbind(), output["samples"].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if kind == "avatar":
        torch.testing.assert_close(output["samples"].unbind()[1], case["source"]["samples"].unbind()[1], rtol=0, atol=1e-6)
    assert stages.snapshot(case["source"]) == before


def test_native_selected_sampling_object_is_restored(stub_lifter):
    model = tiny_model()
    original_sampling = model.model.model_sampling
    selected, sampler, sigmas = setup_dual_clock_sampling(model, latent(), 4, 8., 2., "euler")
    case = inputs(model=selected, sigmas=sigmas)
    case["sampler"] = sampler
    expected, _ = legacy.sample_progressive_h3(selected, case["positive"], case["negative"],
        case["source"], sampler, sigmas, upscaler_model="test", seed=7, low_evaluations=2)
    boundary = low_only(case)
    assert model.model.model_sampling is original_sampling
    result, _ = high_only(case, restart(case, boundary))
    for a, b in zip(expected["samples"].unbind(), result["samples"].unbind()):
        assert torch.equal(a, b)
    assert model.model.model_sampling is original_sampling


def test_boundary_holds_model_space_video_and_evolving_audio():
    case = inputs()
    captured = {}
    def observe(step, prediction, state, total):
        if step == 1:
            captured["prediction"] = [v.detach().clone() for v in prediction.unbind()]
            captured["state"] = [v.detach().clone() for v in state.unbind()]
    noise = comfy.sample.prepare_noise(case["low_source"]["samples"], 7)
    boundary = low_only(case, callback=observe)
    p = case["plan"]
    audio = captured["state"][1] + (captured["state"][1] - captured["prediction"][1]) * ((p.resume_sigma - p.prediction_sigma) / p.prediction_sigma)
    assert torch.equal(boundary.tensors["clean_video"], captured["prediction"][0])
    assert torch.equal(boundary.tensors["audio_next"], audio)
    assert not torch.equal(audio, captured["prediction"][1])
    assert torch.equal(boundary.tensors["anchor_audio_noise"], noise.unbind()[1])


def test_changed_high_model_and_conditions_do_not_execute_low(stub_lifter, monkeypatch):
    case = inputs()
    boundary = low_only(case)
    low_receipt = boundary.receipt_json
    high = tiny_model()
    parameter, weight = next(iter(high.model.named_parameters()))
    assert high.add_patches({parameter: ("diff", (torch.ones_like(weight) * .02,))})
    # Hard fail on any attempt to call a whole executor or execute LOW again.
    def forbidden(*args, **kwargs):
        raise AssertionError("LOW/full executor must be absent")
    monkeypatch.setattr(stages, "sample_low", forbidden)
    monkeypatch.setattr(legacy, "sample_progressive_h3", forbidden)
    case["hp"][0][0] = torch.ones_like(case["hp"][0][0]) * .4
    output, text = high_only(case, restart(case, boundary, high), high)
    assert output["samples"].is_nested
    assert json.loads(text)["execution"]["callbacks"] == [0, 1]
    assert boundary.receipt_json == low_receipt


@pytest.mark.parametrize("phase", ["low", "high"])
def test_cancel_preserves_original_and_retry(stub_lifter, phase):
    case = inputs(source=initialized("avatar"), mode="initialized_av_exp")
    before = copy.deepcopy(case["model"].model_options)
    def cancel(*args):
        raise RuntimeError("modular stage cancelled")
    state = restart(case, low_only(case)) if phase == "high" else None
    with pytest.raises(RuntimeError, match="modular stage cancelled"):
        high_only(case, state, callback=cancel) if state is not None else low_only(case, callback=cancel)
    assert case["model"].model_options == before
    assert not case["model"].wrappers
    high_only(case, state) if state is not None else low_only(case)


@pytest.mark.parametrize("field", ["clean_video", "audio_next", "anchor_audio_noise"])
def test_boundary_tamper_rejected(field):
    boundary = low_only(inputs())
    changed = dict(boundary.tensors)
    changed[field] = changed[field] + .01
    with pytest.raises(ValueError, match="contents changed"):
        replace(boundary, tensors=changed).verify()


def test_unknown_model_wrapper_executes_and_is_not_portable():
    case = inputs()
    called = []
    def wrapper(execute, args):
        called.append(True)
        return execute(args["input"], args["timestep"], **args["c"])
    case["model"].set_model_unet_function_wrapper(wrapper)
    result = low_only(case)
    assert len(called) == 2
    assert result.verify()["portable_identity"] is False
    assert case["model"].model_options["model_function_wrapper"] is wrapper


def test_high_lora_is_real_preserved_and_changes_only_high(stub_lifter, monkeypatch):
    case = inputs()
    boundary = low_only(case)
    plain, _ = high_only(case, restart(case, boundary))
    high = tiny_model()
    key, weight = next((k, w) for k, w in high.model.named_parameters() if w.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .4,
        torch.ones(1, weight.shape[1]) * .4, 1., None, None, None))
    assert high.add_patches({key: adapter}, .7)
    before = list(high.patches[key])
    with monkeypatch.context() as patch:
        patch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("LOW reran"))
        output, report = high_only(case, restart(case, boundary, high), high)
    assert high.patches[key] == before and not case["model"].patches
    assert any(not torch.equal(a, b) for a, b in zip(plain["samples"].unbind(), output["samples"].unbind()))
    assert json.loads(report)["portable_identity"] is True


def test_high_schedule_changes_without_low_execution(stub_lifter, monkeypatch):
    case = inputs()
    boundary = low_only(case)
    resume = case["plan"].resume_sigma
    tail = torch.tensor([resume, resume * .8, resume * .4, 0.])
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("LOW reran"))
    state = restart(case, boundary, high_sigmas=tail)
    seen = []
    _, report = high_only(case, state, callback=lambda i, _p, _s, n: seen.append((i, n)))
    assert seen == [(2, 5), (3, 5), (4, 5)]
    assert json.loads(report)["execution"]["actual_apply_calls"] == 3
    assert boundary.verify()["request"]["plan"]["high_evaluations"] == 2


@pytest.mark.parametrize("tail", [[.3, .1, 0.], [.8, 0.], [1., 0.], [float("nan"), 0.]])
def test_high_cannot_change_frozen_boundary_sigma(stub_lifter, tail):
    case = inputs()
    with pytest.raises(ValueError, match="resume sigma"):
        restart(case, low_only(case), high_sigmas=torch.tensor(tail))


@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("reserve", [0, 511, True, 1024.5])
def test_resource_reserve_not_silently_removed(stub_lifter, phase, reserve):
    case = inputs()
    state = restart(case, low_only(case)) if phase == "high" else None
    with pytest.raises(ValueError, match="reserve"):
        high_only(case, state, reserve_vram_mib=reserve) if state else low_only(case, reserve_vram_mib=reserve)


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_resource_failure_stops_current_stage_without_retry(monkeypatch, failure_at):
    case = inputs()
    count = []
    def check(device, reserve):
        count.append(reserve)
        if len(count) == failure_at:
            raise RuntimeError("test resource limit")
        return {"device": str(device)}
    monkeypatch.setattr(legacy, "_resource_snapshot", check)
    with pytest.raises(RuntimeError, match="test resource limit"):
        low_only(case)
    assert count == [1024 * 1024**2] * failure_at
    assert not case["model"].wrappers


def test_high_retains_user_latent_metadata_and_detects_mutation(stub_lifter):
    case = inputs()
    case["source"]["user_note"] = {"meaning": "retain", "tensor": torch.arange(3)}
    state = restart(case, low_only(case))
    output, _ = high_only(case, state)
    assert output["user_note"] is case["source"]["user_note"]
    case["source"]["user_note"]["tensor"][0] = 99
    with pytest.raises(ValueError, match="restart changed"):
        state.verify()


@pytest.mark.parametrize("what", ["contract", "restart", "anchor", "anchor_noise", "mask"])
def test_high_restart_tamper_rejected(stub_lifter, what):
    case = inputs(source=initialized("avatar"), mode="initialized_av_exp")
    state = restart(case, low_only(case))
    if what == "contract":
        data = json.loads(state.contract_json)
        data["coordinates"]["sampling"]["noise_scale"] = 9.
        state = replace(state, contract_json=json.dumps(data))
    else:
        with torch.inference_mode():
            state.tensors[what].unbind()[0].add_(.01)
    with pytest.raises(ValueError):
        high_only(case, state)


@pytest.mark.parametrize("change", [{"task": "invalid"}, {"requested_low_scale": .1},
                                    {"low_width": 32}, {"high_evaluations": 0}])
def test_plan_math_cannot_be_overridden_with_descriptive_fields(change):
    case = inputs()
    with pytest.raises(ValueError):
        stages.validate_plan(replace(case["plan"], **change))


def test_sampler_mutation_invalidates_completion():
    case = inputs()
    def changed(*args):
        case["sampler"].extra_options = {"s_churn": 0}
    with pytest.raises(ValueError, match="sampler changed"):
        low_only(case, callback=changed)


def test_explicit_low_keeps_its_own_video_and_generation_mask():
    case = inputs(source=initialized("binary"), mode="initialized_av_exp")
    high_video, audio = case["source"]["samples"].unbind()
    low_video = torch.ones(1, 24, 2, 2, 4) * .25
    explicit = {"samples": comfy.nested_tensor.NestedTensor((low_video, audio)),
                "noise_mask": torch.ones(2, 4)}
    result = stages.prepare_low_source(case["source"], case["plan"], case["mode"], explicit_low=explicit)
    assert result["samples"].unbind()[0] is low_video
    assert result["samples"].unbind()[1] is audio
    assert bool((result["noise_mask"].unbind()[0] == 1).all())
    assert not torch.equal(high_video[..., :2, :4], low_video)
    # This primitive deliberately does not certify an accepted-parent chain.
    explicit["samples"] = comfy.nested_tensor.NestedTensor((low_video, audio + 1))
    with pytest.raises(ValueError, match="source audio differ"):
        stages.prepare_low_source(case["source"], case["plan"], case["mode"], explicit_low=explicit)
