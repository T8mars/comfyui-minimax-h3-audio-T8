"""Real MODEL/wrapper lifecycle; tiny random Core is not trained GPU acceptance."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.h3_projection_chunks import ProjectionPlan
from h3_audio_t8_pkg import projection_chunks_advanced as feature
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def cpu_inference(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def sample(selected, seed=211021):
    source = latent()
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 2, 12.0, 3.0, "dual_clock_euler", "native_flow"
    )
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    return SamplerCustomAdvanced.execute(
        RandomNoise.execute(seed).result[0], guider, sampler, sigmas, source
    ).result[0]


def assert_av_close(actual, expected):
    for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=8e-5, atol=1e-5)


def test_actual_two_step_sampler_calls_selected_scope_then_restores_plain_request():
    bare = model()
    methods = {name: dict(vars(module)).get("forward")
               for name, module in bare.model.named_modules()}
    expected = sample(bare)
    selected, setup = feature.configure_projection_chunks(bare, ProjectionPlan(rows=7, query_rows=9))
    assert setup["status"] == "selected_not_executed"
    assert not bare.wrappers and bare.get_attachment(feature.KEY) is None
    actual = sample(selected)
    assert_av_close(actual, expected)
    report = feature.audit_projection_chunks(selected)
    assert report["completed_diffusion_calls"] == 2
    assert report["calls"]["norm_qkv.fused"]["max_rows"] <= 7
    assert report["portable_cache_identity"] is False
    assert report["request_active"] is False
    for name, module in bare.model.named_modules():
        current = vars(module).get("forward")
        old = methods[name]
        assert current is old or (getattr(current, "__func__", None) is type(module).forward
                                  and getattr(current, "__self__", None) is module)
    assert_av_close(sample(bare), expected)


def test_shared_base_low_high_plans_and_ordered_lora_are_independent():
    bare = model()
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    patch = ("diff", (torch.full_like(bare.model_state_dict()[key], 0.001),))
    bare.add_patches({key: patch}, strength_patch=-0.25)
    patches = deepcopy(bare.patches)
    low, _ = feature.configure_projection_chunks(bare, ProjectionPlan(rows=5), phase="LOW")
    high, _ = feature.configure_projection_chunks(bare, ProjectionPlan(rows=11), phase="HIGH")
    sample(low)
    sample(high)
    lreport, hreport = feature.audit_projection_chunks(low), feature.audit_projection_chunks(high)
    assert (lreport["phase"], hreport["phase"]) == ("LOW", "HIGH")
    assert lreport["calls"]["norm_qkv.fused"]["max_rows"] <= 5
    assert 5 < hreport["calls"]["norm_qkv.fused"]["max_rows"] <= 11
    assert lreport["completed_diffusion_calls"] == hreport["completed_diffusion_calls"] == 2
    assert list(low.patches) == list(high.patches) == list(patches)
    assert low.patches[key][0][0] == high.patches[key][0][0] == -0.25


def test_pre_run_resets_actual_counts_cleanup_keeps_last_audit():
    selected, _ = feature.configure_projection_chunks(model(), ProjectionPlan(rows=8))
    sample(selected)
    first = feature.audit_projection_chunks(selected)
    sample(selected)
    second = feature.audit_projection_chunks(selected)
    assert first["completed_diffusion_calls"] == second["completed_diffusion_calls"] == 2
    assert second["request_number"] > first["request_number"]
    assert second["cleanup_observed"] and not second["request_active"]


def test_unknown_network_api_retains_executor_without_guessing_model_name():
    selected, _ = feature.configure_projection_chunks(model(), ProjectionPlan(rows=3))
    wrapper = selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)[0]
    calls, answer = [], object()

    class Executor:
        class_obj = SimpleNamespace(new_model=True)

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return answer

    assert wrapper(Executor(), 7, unknown=False) is answer
    assert calls == [((7,), {"unknown": False})]
    report = feature.audit_projection_chunks(selected)
    assert report["partial"] and report["completed_diffusion_calls"] == 1
    assert report["calls"] == {} and not report["portable_cache_identity"]


def test_real_foreign_forward_error_is_not_retried_or_swallowed():
    bare = model()
    block = bare.model.diffusion_model.blocks[0]
    original = block.forward
    calls = []

    def foreign(*args, **kwargs):
        calls.append(1)
        raise TypeError("real foreign computation failure")

    block.forward = foreign
    selected, _ = feature.configure_projection_chunks(bare, ProjectionPlan(rows=5))
    with pytest.raises(TypeError, match="real foreign computation failure"):
        sample(selected)
    assert calls == [1]
    assert block.forward is foreign
    report = feature.audit_projection_chunks(selected)
    assert report["failed_diffusion_calls"] == 1
    assert not report["request_active"]
    block.forward = original
    assert all(torch.isfinite(x).all() for x in sample(bare)["samples"].unbind())


def test_disabled_returns_original_and_does_not_add_any_model_owner():
    bare = model()
    result, report = feature.configure_projection_chunks(bare, enabled=False)
    assert result is bare and report["status"] == "disabled"
    assert not bare.wrappers and not bare.callbacks
    assert feature.audit_projection_chunks(bare)["status"] == "not_selected"


def test_reconfigure_replaces_only_exact_own_lifecycle_and_keeps_other_owner():
    bare = model()
    called = []

    def foreign(executor, *args, **kwargs):
        called.append(1)
        return executor(*args, **kwargs)

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "foreign", foreign)
    selected, _ = feature.configure_projection_chunks(bare, ProjectionPlan(rows=4))
    replaced, _ = feature.configure_projection_chunks(selected, ProjectionPlan(rows=10))
    assert len(replaced.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)) == 1
    assert replaced.get_wrappers(WrappersMP.DIFFUSION_MODEL, "foreign") == [foreign]
    assert len(replaced.get_callbacks(CallbacksMP.ON_PRE_RUN, feature.KEY)) == 1
    sample(replaced)
    assert len(called) == 2
    assert feature.audit_projection_chunks(replaced)["plan"]["rows"] == 10
    assert feature.audit_projection_chunks(selected)["completed_diffusion_calls"] == 0


def test_own_mutated_plan_is_real_binding_error_before_executor():
    plan = ProjectionPlan(rows=4)
    selected, _ = feature.configure_projection_chunks(model(), plan)
    object.__setattr__(plan, "rows", 0)
    with pytest.raises(ValueError, match="binding"):
        sample(selected)


def test_report_is_detached_json_data_not_live_telemetry_alias():
    selected, _ = feature.configure_projection_chunks(model(), ProjectionPlan(rows=6))
    sample(selected)
    report = feature.audit_projection_chunks(selected)
    report["calls"]["norm_qkv.fused"]["calls"] = -999
    assert feature.audit_projection_chunks(selected)["calls"]["norm_qkv.fused"]["calls"] > 0
    json.dumps(feature.audit_projection_chunks(selected), allow_nan=False)


def test_actual_last_forward_cannot_return_success_after_binding_mutation():
    bare, plan, seen = model(), ProjectionPlan(rows=5), []

    def mutate_after_last_final(*_args):
        seen.append(1)
        if len(seen) == 2:
            object.__setattr__(plan, "rows", 6)

    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate_after_last_final)
    selected, _ = feature.configure_projection_chunks(bare, plan)
    try:
        with pytest.raises(ValueError, match="binding"):
            sample(selected)
    finally:
        handle.remove()
        object.__setattr__(plan, "rows", 5)
    assert len(seen) == 2
    assert feature.audit_projection_chunks(selected)["failed_diffusion_calls"] == 1
    assert all(torch.isfinite(x).all() for x in sample(bare)["samples"].unbind())
