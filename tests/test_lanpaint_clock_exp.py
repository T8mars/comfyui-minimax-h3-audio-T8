"""New clock integration checks. Real installed GPL dependency; no GPU/model download."""
import asyncio
import copy
import json
from types import FunctionType

import pytest
import torch
import comfy.samplers
import comfy.sampler_helpers
from comfy.ldm.minimax import model as core_h3
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import lanpaint_clock_exp as clock
from h3_audio_t8_pkg.nodes_lanpaint_clock_exp import MiniMaxH3LanPaintClockSamplerEXPT8
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


@pytest.fixture
def installed_backend():
    import nodes
    if "LanPaint_SamplerCustomAdvanced" not in nodes.NODE_CLASS_MAPPINGS:
        pytest.skip("Real external LanPaint dependency not loaded; no fake sampler fallback")
    return clock._reviewed_backend()


def core_functions():
    return (comfy.samplers.CFGGuider.outer_sample, comfy.samplers.CFGGuider.predict_noise,
            comfy.samplers.KSAMPLER.sample, comfy.sampler_helpers.prepare_mask)


def test_derivative_matches_actual_Core_autograd_at_endpoints_and_different_shifts():
    for from_shift, to_shift in ((12., 3.), (3., 12.), (1., 1.), (7., 7.), (2.5, .4)):
        sigma = torch.tensor([0., 1e-9, 1e-4, .1, .5, .99, 1.], dtype=torch.float64, requires_grad=True)
        mapped = core_h3.time_shift_sigma(sigma, from_shift, to_shift)
        derivative, = torch.autograd.grad(mapped.sum(), sigma, create_graph=True)
        actual = clock.time_shift_slope(sigma, from_shift, to_shift)
        torch.testing.assert_close(actual, derivative, rtol=1e-12, atol=1e-12)
        assert actual.dtype == sigma.dtype and actual.device == sigma.device
        assert actual.requires_grad and torch.isfinite(actual).all()
    for shift in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            clock.time_shift_slope(torch.tensor(.5), shift, 3.)
    assert not torch.cuda.is_initialized()


def test_actual_dependency_is_isolated_and_restores_Core_on_exception(installed_backend):
    backend, original, report = installed_backend
    assert backend is not original
    assert backend.LanPaint is original.LanPaint  # actual installed algorithm, not copied into T8
    assert original.time_shift_sigma is None and original.time_shift_slope is None
    assert backend.time_shift_sigma is core_h3.time_shift_sigma
    assert backend.time_shift_slope is clock.time_shift_slope
    assert report["Core_time_shift_slope_present"] is False
    before = core_functions()
    with pytest.raises(RuntimeError, match="intentional exception"):
        with backend.override_sample_function():
            assert comfy.samplers.KSAMPLER.sample is backend.KSAMPLER.sample
            assert original._override_active is False
            raise RuntimeError("intentional exception")
    assert core_functions() == before
    assert not backend._override_active and not original._override_active
    assert original.time_shift_sigma is None and not torch.cuda.is_initialized()


def test_actual_tiny_Core_LanPaint_AV_executes_audio_clock_and_preserves_caller(installed_backend):
    backend, original, _ = installed_backend
    bare, source = model(), source_latent(masked=True)
    # Keep audio frozen in the actual nested mask; generated PCM still is not
    # promised exact after VAE/export. This test does not qualify trained edits.
    source["noise_mask"].unbind()[1].zero_()
    source_before = [item.clone() for item in source["samples"].unbind()]
    masks_before = [item.clone() for item in source["noise_mask"].unbind()]
    events = []

    def selected_hook(args):
        events.append(float(args["sigma"].flatten()[0]))
        return args["denoised"]

    bare.set_model_sampler_post_cfg_function(selected_hook)
    key, weight = next((key, item) for key, item in bare.model.state_dict().items() if item.ndim == 2)
    bare.add_patches({key: ("diff", [torch.full_like(weight, .001)])}, strength_patch=.2)
    patches = bare.patches[key]
    options = copy.deepcopy(bare.model_options)
    guider = BasicGuider.execute(bare, conditioning()).result[0]
    original_conds = guider.original_conds
    before = core_functions()
    full_schedule = native_flow_sigmas(4, 12.)
    # The external VE conversion is singular at sigma=1. Preserve that real
    # refusal; a source-edit workflow explicitly selects a partial schedule,
    # rather than silently clamping the caller's full-noise starting value.
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        clock.sample_lanpaint_clock(RandomNoise.execute(26100803).result[0],
            guider, comfy.samplers.ksampler("euler"), full_schedule, source)
    output, denoised, raw = clock.sample_lanpaint_clock(RandomNoise.execute(26100803).result[0],
        guider, comfy.samplers.ksampler("euler"), full_schedule[1:], source,
        inner_steps=2, guidance_lambda=5., step_size=.2, prompt_mode="Image First")
    report = json.loads(raw)
    assert report["audio_clock_observed"] and report["clock_map_calls"] == 3
    assert report["clock_slope_calls"] == 3 and report["outer_schedule_steps"] == 3
    assert not report["human_edit_quality_accepted"] and not report["generic_patch_stack_qualified"]
    for result in (output, denoised):
        values = result["samples"].unbind()
        assert [item.shape for item in values] == [item.shape for item in source_before]
        assert all(torch.isfinite(item).all() for item in values)
    assert events and bare.patches[key] is patches
    assert bare.model_options == options and guider.original_conds is original_conds
    assert guider.model_patcher is bare and not hasattr(bare, "LanPaint_NumSteps")
    assert core_functions() == before and not backend._override_active
    assert original.time_shift_sigma is None and original.time_shift_slope is None
    assert backend.time_shift_slope is clock.time_shift_slope
    for old, actual in zip(source_before, source["samples"].unbind(), strict=True):
        assert torch.equal(old, actual)
    for old, actual in zip(masks_before, source["noise_mask"].unbind(), strict=True):
        assert torch.equal(old, actual)
    assert not torch.cuda.is_initialized()


def test_explicit_new_node_and_bad_input_do_not_change_existing_LanPaint_schema(installed_backend):
    from h3_audio_t8_pkg import comfy_entrypoint
    from h3_audio_t8_pkg.nodes_lanpaint_av_advanced import MiniMaxH3LanPaintAVPrepareT8Advanced
    schema = MiniMaxH3LanPaintClockSamplerEXPT8.GET_SCHEMA()
    assert schema.is_experimental
    fields = {item.id: item for item in schema.inputs}
    assert fields["inner_steps"].default == 5 and fields["step_size"].default == .2
    assert fields["prompt_mode"].default == "Image First"
    old = {item.id: item for item in MiniMaxH3LanPaintAVPrepareT8Advanced.GET_SCHEMA().inputs}
    assert old["require_lanpaint_sampler"].default is True and old["frame_policy"].default == "strict"
    identifiers = [kind.GET_SCHEMA().node_id for kind in asyncio.run(comfy_entrypoint().get_node_list())]
    assert identifiers[-1] == schema.node_id and len(identifiers) == len(set(identifiers))
    with pytest.raises(ValueError, match="nested H3 AV"):
        clock.sample_lanpaint_clock(None, None, None, torch.tensor([.5, 0.]), {"samples": torch.zeros(1)})
    assert not torch.cuda.is_initialized()


def test_actual_node_propagates_model_exception_and_restores_both_namespaces(installed_backend):
    backend, original, _ = installed_backend
    bare, source = model(), source_latent(masked=True)

    def failing_hook(_args):
        raise RuntimeError("real selected hook failure")

    bare.set_model_sampler_post_cfg_function(failing_hook)
    options = copy.deepcopy(bare.model_options)
    guider = BasicGuider.execute(bare, conditioning()).result[0]
    before = core_functions()
    with pytest.raises(RuntimeError, match="real selected hook failure"):
        clock.sample_lanpaint_clock(RandomNoise.execute(42).result[0], guider,
            comfy.samplers.ksampler("euler"), native_flow_sigmas(4, 12.)[1:], source,
            inner_steps=2)
    assert core_functions() == before and bare.model_options == options
    assert not hasattr(bare, "LanPaint_NumSteps") and not backend._override_active
    assert original.time_shift_sigma is None and original.time_shift_slope is None
    assert backend.time_shift_sigma is core_h3.time_shift_sigma and backend.time_shift_slope is clock.time_shift_slope
    assert not torch.cuda.is_initialized()


def test_live_Core_clock_replacement_is_rejected_without_executing_foreign_code(installed_backend, monkeypatch):
    native = core_h3.time_shift_sigma
    constants = list(native.__code__.co_consts)
    index = next(index for index, value in enumerate(constants) if type(value) is float)
    constants[index] += 1.
    changed = FunctionType(native.__code__.replace(co_consts=tuple(constants)), native.__globals__)
    with monkeypatch.context() as patch:
        patch.setattr(core_h3, "time_shift_sigma", changed)
        with pytest.raises(RuntimeError, match="live Core H3 time function differs"):
            clock._reviewed_backend()
    calls = []

    class ForeignConstant:
        def __eq__(self, _other):
            calls.append("unsafe equality executed")
            raise AssertionError("Foreign constant must not be evaluated")

    constants[index] = ForeignConstant()
    changed = FunctionType(native.__code__.replace(co_consts=tuple(constants)), native.__globals__)
    with monkeypatch.context() as patch:
        patch.setattr(core_h3, "time_shift_sigma", changed)
        with pytest.raises(RuntimeError, match="live Core H3 time function differs"):
            clock._reviewed_backend()
    assert calls == [] and core_h3.time_shift_sigma is native
    assert not torch.cuda.is_initialized()
