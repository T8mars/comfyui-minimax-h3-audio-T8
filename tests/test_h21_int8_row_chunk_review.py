"""Independent actual INT8 owner/helper review, without GPU qualification."""
from dataclasses import replace

import pytest
import torch

from comfy_kitchen.backends.eager import quantization as eager_quant
from comfy_kitchen import constraints
from comfy_kitchen.registry import registry

from h3_audio_t8_pkg import int8_row_chunk as row
from test_h21_int8_row_chunk import LinearRoot, actual_int8_mlp
from test_h21_quant_execution import int8_module


@pytest.fixture(autouse=True)
def actual_CPU_only():
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fused", [False, True])
def test_actual_backend_helper_override_remains_one_full_original_call(tmp_path, monkeypatch, fused):
    if fused:
        root = actual_int8_mlp(tmp_path)
    else:
        source, _ = int8_module(tmp_path)
        root = LinearRoot(source.control)
    generator = torch.Generator().manual_seed(21102231)
    x = (torch.randn(17, 32, generator=generator) * .2
         + torch.linspace(-3., 4., 17)[:, None]).bfloat16()
    original, calls = eager_quant._apply_input_act, []

    def foreign(values, *args, **kwargs):
        calls.append(values.shape[0])
        answer = original(values, *args, **kwargs)
        return answer + answer.mean() * .5

    monkeypatch.setattr(eager_quant, "_apply_input_act", foreign)
    expected = root(x)
    original_calls = list(calls)
    calls.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == original_calls
    assert observed.report()["partial"]
    assert not observed.report()["portable_cache_identity"]


def test_actual_registered_custom_op_kernel_override_keeps_full_original(tmp_path):
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    generator = torch.Generator().manual_seed(21102232)
    x = (torch.randn(17, 32, generator=generator) * .2
         + torch.linspace(-3., 4., 17)[:, None]).bfloat16()
    definition = eager_quant._op_int8_linear
    original = definition._backend_fns.get("cpu", definition._backend_fns.get(None))
    assert callable(original), "Read the actual CustomOpDef kernel, not a fake replacement"
    calls = []

    def foreign(values, *args, **kwargs):
        calls.append(values.shape[0])
        answer = original(values, *args, **kwargs)
        return answer + answer.mean() * .5

    # Only this CPU test process's public CustomOpDef registration is changed.
    # Restore the original kernel even when a strong counterexample fails.
    definition.register_kernel("cpu")(foreign)
    try:
        expected = root(x)
        assert calls == [17]
        calls.clear()
        with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
            actual = root(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert calls == [17]
        assert observed.report()["partial"]
    finally:
        definition.register_kernel("cpu")(original)


@pytest.mark.parametrize("fused", [False, True])
def test_actual_row_dependent_shape_constraint_preserves_valid_full_call(
    tmp_path, monkeypatch, fused
):
    original = registry._constraints[("eager", "int8_linear")]
    x_constraint = replace(
        original.params["x"], shape_rules=(constraints.DivisibleBy(0, 17),)
    )
    monkeypatch.setitem(
        registry._constraints,
        ("eager", "int8_linear"),
        replace(original, params={**original.params, "x": x_constraint}),
    )
    if fused:
        root = actual_int8_mlp(tmp_path)
    else:
        source, _ = int8_module(tmp_path)
        root = LinearRoot(source.control)
    x = torch.randn((17, 32), generator=torch.Generator().manual_seed(21102233)).bfloat16()
    expected = root(x)  # Actual registered full-row API is valid.
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert observed.report()["partial"]


def test_unknown_actual_parameter_validator_is_not_called_extra(tmp_path, monkeypatch):
    original, calls = constraints.ParamConstraint.check_dtype, []

    def foreign(self, value):
        calls.append(id(value))
        return original(self, value)

    monkeypatch.setattr(constraints.ParamConstraint, "check_dtype", foreign)
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    x = torch.randn((17, 32), generator=torch.Generator().manual_seed(21102234)).bfloat16()
    calls.clear()  # Do not include genuine fixture weight-quantization validators.
    expected = root(x)
    expected_calls = list(calls)
    calls.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == expected_calls
    assert observed.report()["partial"]


def test_actual_nested_functional_activation_owner_keeps_full_original_call(
    tmp_path, monkeypatch
):
    root = actual_int8_mlp(tmp_path)
    x = (torch.randn((17, 32), generator=torch.Generator().manual_seed(21102236)) * .2
         + torch.linspace(-3., 4., 17)[:, None]).bfloat16()
    original, calls = torch.nn.functional.silu, []

    def foreign(values, *args, **kwargs):
        calls.append(values.shape[0])
        answer = original(values, *args, **kwargs)
        return answer + answer.mean() * .5

    monkeypatch.setattr(torch.nn.functional, "silu", foreign)
    expected = root(x)
    expected_calls = list(calls)
    calls.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == expected_calls
    assert observed.report()["partial"]


@pytest.mark.parametrize("fused", [False, True])
def test_unknown_callable_class_helper_retains_original_row_coupled_behavior(
    tmp_path, monkeypatch, fused
):
    if fused:
        root = actual_int8_mlp(tmp_path)
    else:
        source, _ = int8_module(tmp_path)
        root = LinearRoot(source.control)
    x = (torch.randn((17, 32), generator=torch.Generator().manual_seed(21102235)) * .2
         + torch.linspace(-3., 4., 17)[:, None]).bfloat16()
    original, calls = eager_quant._apply_input_act, []

    class CrossRowCallable:
        def __new__(cls, values, *args, **kwargs):
            calls.append(values.shape[0])
            answer = original(values, *args, **kwargs)
            return answer + answer.mean() * .5

    monkeypatch.setattr(eager_quant, "_apply_input_act", CrossRowCallable)
    expected = root(x)
    expected_calls = list(calls)
    calls.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as observed:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == expected_calls
    assert observed.report()["partial"]
