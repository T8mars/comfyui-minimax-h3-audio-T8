"""Independent actual Core cache-owner review; approximate is not bit-exact.

Exact assertions below only apply to unknown user paths which must delegate
unchanged, not to the explicitly selected native approximation itself.
"""

from types import FunctionType, MethodType

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention

from h3_audio_t8_pkg import reference_step_cache_advanced as feature
from test_h21_reference_step_cache import sample, two_layers, positive, equal
from test_fast_h3_v2_core_sampler import model


@pytest.fixture(autouse=True)
def actual_CPU_only(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["pre", "post"])
def test_actual_global_module_hooks_keep_full_original_rows_and_calls(kind):
    base, pos = two_layers(), positive()
    target = base.model.diffusion_model.blocks[0].norm1
    calls = []

    def pre(module, args):
        if module is target:
            calls.append(args[0].shape[0])
            return (args[0] + args[0].mean() * 0.25,) + args[1:]

    def post(module, args, output):
        if module is target:
            calls.append(args[0].shape[0])
            return output + output.mean() * 0.25

    handle = (
        torch.nn.modules.module.register_module_forward_pre_hook(pre)
        if kind == "pre"
        else torch.nn.modules.module.register_module_forward_hook(post)
    )
    try:
        expected = sample(base, pos)
        expected_calls = list(calls)
        calls.clear()
        selected, paired, _ = feature.configure_reference_step_cache(
            base, pos, "approximate"
        )
        actual = sample(selected, paired)
        assert calls == expected_calls, (
            f"Unknown global {kind} hook must keep full original rows; actual={calls}, expected={expected_calls}"
        )
        assert equal(actual, expected)
        report = feature.audit_reference_step_cache(selected)
        assert report["partial"] and report["approximate_block_hits"] == 0
    finally:
        handle.remove()


def test_actual_RMSNorm_inplace_code_change_is_not_authenticated_by_same_function_identity(
    monkeypatch,
):
    base, pos = two_layers(), positive()
    live = torch.nn.RMSNorm.forward
    owner = __import__(live.__module__, fromlist=["RMSNorm"])
    original = FunctionType(
        live.__code__,
        live.__globals__,
        live.__name__,
        live.__defaults__,
        live.__closure__,
    )
    calls = []
    monkeypatch.setattr(owner, "_h21_r5_original_rms", original, raising=False)
    monkeypatch.setattr(owner, "_h21_r5_rms_calls", calls, raising=False)

    def changed(self, value):
        _h21_r5_rms_calls.append((id(self), value.shape[0]))  # noqa:F821
        output = _h21_r5_original_rms(self, value)  # noqa:F821
        return output + output.mean() * 0.25

    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = sample(base, pos)
    expected_calls = list(calls)
    calls.clear()
    selected, paired, _ = feature.configure_reference_step_cache(
        base, pos, "approximate"
    )
    actual = sample(selected, paired)
    assert calls == expected_calls, (
        "Changed actual RMSNorm owner must not be bypassed or split/repeated"
    )
    assert equal(actual, expected)
    report = feature.audit_reference_step_cache(selected)
    assert report["partial"] and report["approximate_block_hits"] == 0


def test_actual_module_call_impl_owner_is_retained_on_full_original_rows():
    # One real Core block isolates this unknown owner. A second native block
    # would be allowed to approximate, making whole-output equality irrelevant.
    base, pos = model(), positive()
    module = base.model.diffusion_model.blocks[0].norm1
    original = module._call_impl
    calls = []

    def foreign(self, *args, **kwargs):
        calls.append(args[0].shape[0])
        output = original(*args, **kwargs)
        return output + output.mean() * 0.25

    module._call_impl = MethodType(foreign, module)
    expected = sample(base, pos)
    expected_calls = list(calls)
    calls.clear()
    selected, paired, _ = feature.configure_reference_step_cache(
        base, pos, "approximate"
    )
    actual = sample(selected, paired)
    assert calls == expected_calls
    assert equal(actual, expected)
    report = feature.audit_reference_step_cache(selected)
    assert report["partial"] and report["approximate_block_hits"] == 0
    assert module._call_impl.__func__ is foreign


def test_real_attention_wrapper_closed_function_change_keeps_full_queries(monkeypatch):
    base, pos = two_layers(), positive()
    wrapped = attention.attention_pytorch
    cells = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__, strict=True))
    live = cells["func"].cell_contents
    original = FunctionType(
        live.__code__,
        live.__globals__,
        live.__name__,
        live.__defaults__,
        live.__closure__,
    )
    calls = []
    monkeypatch.setattr(
        attention, "_h21_r5_original_attention", original, raising=False
    )
    monkeypatch.setattr(attention, "_h21_r5_attention_calls", calls, raising=False)

    def changed(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        _h21_r5_attention_calls.append((q.shape[-2], k.shape[-2]))  # noqa:F821
        output = _h21_r5_original_attention(  # noqa:F821
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )  # noqa:F821
        return output + q.mean()

    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = sample(base, pos)
    expected_calls = list(calls)
    calls.clear()
    selected, paired, _ = feature.configure_reference_step_cache(
        base, pos, "approximate"
    )
    actual = sample(selected, paired)
    assert calls == expected_calls, (
        "Unknown dense attention must receive full original Q/K and not be treated as a native row-local backend"
    )
    assert equal(actual, expected)
    report = feature.audit_reference_step_cache(selected)
    assert report["partial"] and report["approximate_block_hits"] == 0
