"""Actual CPU INT8 paths; never a CUDA, speed or trained-media claim."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from comfy import ops
import comfy.utils
from comfy.ldm.minimax import model as core
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor import base as tensor_base
from comfy_kitchen.tensor import int8 as tensor_int8
import comfy_kitchen as kitchen
from comfy_kitchen.registry import registry
from comfy_kitchen import constraints
from comfy.patcher_extension import WrappersMP

from h3_audio_t8_pkg import int8_row_chunk as row
from h3_audio_t8_pkg import int8_row_chunk_advanced as feature
from h3_audio_t8_pkg.h3_projection_chunks import ProjectionPlan, projection_scope

from test_h21_quant_execution import int8_module


@pytest.fixture(autouse=True)
def CPU_only():
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


class ObserveActualOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()
        self.quantized = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        self.ops[str(function)] += 1
        if any(isinstance(value, QuantizedTensor) for value in args):
            self.quantized.append(
                (
                    str(function),
                    [tuple(value.shape) for value in args if torch.is_tensor(value)],
                )
            )
        return function(*args, **({} if kwargs is None else kwargs))


def test_actual_quantized_module_operator_boundary_is_visible_without_global_install(
    tmp_path,
):
    root, _ = int8_module(tmp_path)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212101)
    ).bfloat16()
    expected = root.control(x)
    with ObserveActualOps() as observed:
        actual = root.control(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert observed.quantized, dict(observed.ops)
    assert any(
        name in ("aten.linear.default", "aten.mm.default", "aten.addmm.default")
        for name, _ in observed.quantized
    )


def test_actual_fused_swiglu_bypasses_quantized_aten_and_public_tensor_operator(
    tmp_path,
):
    root, _ = int8_module(tmp_path)
    x = torch.randn(
        (17, 64), generator=torch.Generator().manual_seed(212102)
    ).bfloat16()
    expected = root.control(ops._eager_input_act(x, "swiglu"))
    with ObserveActualOps() as observed:
        actual = ops.linear_input_act(root.control, x, "swiglu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not observed.quantized, observed.quantized
    assert not any("comfy_kitchen.int8_linear" in name for name in observed.ops)
    assert observed.ops, "Actual eager backend did not execute tensor arithmetic"


class LinearRoot(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.control = linear

    def forward(self, x):
        return self.control(x)


@pytest.mark.parametrize("rows", [1, 3, 8, 17, 10000])
def test_actual_core_int8_operator_rows_are_equivalent_and_next_plain_unmodified(
    tmp_path, rows
):
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212103)
    ).bfloat16()
    expected = root(x)
    raw, scale = (
        root.control.weight._qdata.clone(),
        root.control.weight._params.scale.clone(),
    )
    with row.int8_row_scope(root, row.Int8RowPlan(rows=rows)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.calls, runtime.report()
    assert max(item["max_rows"] for item in runtime.calls.values()) <= min(rows, 17)
    assert runtime.report()["portable_cache_identity"] is False
    assert torch.equal(root.control.weight._qdata, raw)
    assert torch.equal(root.control.weight._params.scale, scale)
    torch.testing.assert_close(root(x), expected, rtol=0, atol=0)


def actual_int8_mlp(tmp_path):
    generator = torch.Generator().manual_seed(212104)
    layer = core.MLP(
        32, 32, dtype=torch.bfloat16, device="cpu", operations=ops.manual_cast
    )
    first, _ = int8_module(
        tmp_path, weight=torch.randn((64, 32), generator=generator), bias=False
    )
    second, _ = int8_module(
        tmp_path, weight=torch.randn((32, 32), generator=generator), bias=False
    )
    layer.fc1, layer.fc2 = first.control, second.control
    return layer


def actual_int8_dtype(tmp_path, weight, dtype):
    source, payload = int8_module(tmp_path, weight=weight, bias=False)
    detected = comfy.utils.detect_layer_quantization(payload, "")
    actual = ops.mixed_precision_ops(detected, dtype).Linear(
        weight.shape[1], weight.shape[0], bias=False, device="cpu"
    )
    loaded = actual.load_state_dict(
        {key.removeprefix("control."): value for key, value in payload.items()},
        strict=True,
    )
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert actual.weight._params.orig_dtype is dtype
    assert torch.equal(actual.weight._qdata, source.control.weight._qdata)
    return actual


def test_actual_native_mlp_fused_swiglu_is_separately_chunked_and_restored(tmp_path):
    root = actual_int8_mlp(tmp_path)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212105)
    ).bfloat16()
    expected = root(x)
    existed = "forward" in vars(root)
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.calls, runtime.report()
    assert any("fused_int8" in name for name in runtime.calls), runtime.report()
    assert any("aten." in name for name in runtime.calls), runtime.report()
    assert all(item["max_rows"] <= 3 for item in runtime.calls.values())
    assert ("forward" in vars(root)) is existed
    torch.testing.assert_close(root(x), expected, rtol=0, atol=0)


def test_actual_module_scale_and_stateful_weight_processing_execute_once(tmp_path):
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    root.control.register_parameter(
        "pre_quant_scale",
        torch.nn.Parameter(torch.linspace(0.25, 1.75, 32), requires_grad=False),
    )
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212106)
    ).bfloat16()
    expected, hooks = root(x), []
    handle = root.control.register_forward_pre_hook(
        lambda _m, values: hooks.append(values[0].shape[0])
    )
    try:
        with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
            actual = root(x)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert hooks == [17] and runtime.calls
    calls = []

    def stateful(weight):
        calls.append(1)
        return weight + len(calls) * 0.01

    root.control.weight_function = [stateful]
    expected = root(x)
    calls.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)):
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [1]


def test_existing_core_fused_runtime_scale_bypass_is_not_reported_as_fixed(tmp_path):
    root = actual_int8_mlp(tmp_path)
    root.fc2.register_parameter(
        "pre_quant_scale",
        torch.nn.Parameter(torch.linspace(0.25, 1.75, 32), requires_grad=False),
    )
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212107)
    ).bfloat16()
    expected = root(x)
    proper_module_path = root.fc2(ops._eager_input_act(root.fc1(x), "swiglu"))
    assert not torch.equal(expected, proper_module_path)
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.report()["partial"]
    assert not any("fused_int8" in name for name in runtime.calls)


def test_unknown_actual_backend_keeps_one_original_call(tmp_path, monkeypatch):
    import comfy_kitchen.backends.eager as eager

    original, seen = eager.int8_linear, []

    def foreign(**kwargs):
        seen.append(kwargs["x"].shape[0])
        return original(**kwargs) + len(seen) * 0.01

    monkeypatch.setattr(eager, "int8_linear", foreign)
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212108)
    ).bfloat16()
    expected = root(x)
    seen.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [17] and runtime.report()["partial"]


def test_unknown_constraint_rule_not_executed_extra_for_optional_inspection(
    tmp_path, monkeypatch
):
    original, seen = registry._constraints[("eager", "int8_linear")], []

    def rule(_kwargs):
        seen.append(1)
        return constraints.ValidationResult.ok()

    monkeypatch.setitem(
        registry._constraints,
        ("eager", "int8_linear"),
        replace(original, call_rules=(rule,)),
    )
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212109)
    ).bfloat16()
    expected = root(x)
    seen.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [1] and runtime.report()["partial"]


def test_nested_low_high_disabled_and_next_plain_keep_separate_actual_counts(tmp_path):
    root = actual_int8_mlp(tmp_path)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212110)
    ).bfloat16()
    expected = root(x)
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3), phase="LOW") as low:
        torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
        before = deepcopy(low.calls)
        with row.int8_row_scope(root, row.Int8RowPlan(rows=7), phase="HIGH") as high:
            torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
        with row.int8_row_scope(root, row.Int8RowPlan(rows=0)) as off:
            torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
        assert not off.calls and low.calls == before
    assert max(item["max_rows"] for item in low.calls.values()) <= 3
    assert 3 < max(item["max_rows"] for item in high.calls.values()) <= 7
    assert not any("forward" in vars(module) for module in root.modules())
    torch.testing.assert_close(root(x), expected, rtol=0, atol=0)


def test_true_threads_share_lease_but_plain_request_has_no_context_or_global_hooks(
    tmp_path,
):
    root = actual_int8_mlp(tmp_path)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212111)
    ).bfloat16()
    expected, waiting, entered = root(x), threading.Event(), threading.Event()

    def plain():
        with torch.inference_mode():
            return root(x)

    def selected():
        waiting.set()
        with (
            torch.inference_mode(),
            row.int8_row_scope(root, row.Int8RowPlan(rows=7)) as runtime,
        ):
            entered.set()
            result = root(x)
        return result, runtime

    with ThreadPoolExecutor(max_workers=2) as pool:
        with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as low:
            future = pool.submit(selected)
            assert waiting.wait(2) and not entered.wait(0.1)
            torch.testing.assert_close(
                pool.submit(plain).result(timeout=10), expected, rtol=0, atol=0
            )
            assert not low.calls
            torch.testing.assert_close(root(x), expected, rtol=0, atol=0)
        actual, high = future.result(timeout=10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert high.calls and entered.is_set()
    assert not any("forward" in vars(module) for module in root.modules())


def test_actual_forward_failure_is_not_retried_and_scope_restores_next_request(
    tmp_path,
):
    root = actual_int8_mlp(tmp_path)
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212112)
    ).bfloat16()
    expected, seen = root(x), []

    def fail(_weight):
        seen.append(1)
        raise RuntimeError("actual selected INT8 weight callback failure")

    root.fc1.weight_function = [fail]
    with pytest.raises(
        RuntimeError, match="actual selected INT8 weight callback failure"
    ):
        with row.int8_row_scope(root, row.Int8RowPlan(rows=3)):
            root(x)
    assert seen == [1] and "forward" not in vars(root)
    root.fc1.weight_function = []
    torch.testing.assert_close(root(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_R1_R2_lease_and_actual_native_block_both_orders(tmp_path, reverse):
    from test_h21_projection_chunks import block, block_inputs, run_block

    layer = block()
    for key in ("fc1", "fc2"):
        loaded = actual_int8_dtype(
            tmp_path, getattr(layer.mlp, key).weight, torch.float32
        )
        setattr(layer.mlp, key, loaded)
    values = block_inputs()
    expected = run_block(layer, values)
    first = (
        projection_scope(layer, ProjectionPlan(rows=5))
        if reverse
        else row.int8_row_scope(layer, row.Int8RowPlan(rows=3))
    )
    second = (
        row.int8_row_scope(layer, row.Int8RowPlan(rows=3))
        if reverse
        else projection_scope(layer, ProjectionPlan(rows=5))
    )
    with first as a, second as b:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    r2 = b if reverse else a
    assert any("aten." in name for name in r2.calls), r2.report()
    assert any("fused_int8" in name for name in r2.calls), r2.report()
    assert not any("forward" in vars(module) for module in layer.modules())


def test_real_MODEL_disabled_and_foreign_same_key_identity_are_preserved():
    from test_fast_h3_v2_core_sampler import model

    base = model()
    result, report = feature.configure_int8_row_chunks(base, row.Int8RowPlan(rows=0))
    assert result is base and report["status"] == "disabled" and not base.wrappers

    class Foreign:
        __hash__ = None

        def __eq__(self, _other):
            raise AssertionError("User callable cannot be compared by value")

        def __call__(self, executor, *args, **kwargs):
            return executor(*args, **kwargs)

    foreign = Foreign()
    base.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, feature.KEY, foreign)
    first, _ = feature.configure_int8_row_chunks(base, row.Int8RowPlan(rows=3))
    second, _ = feature.configure_int8_row_chunks(first, row.Int8RowPlan(rows=7))
    owned = second.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)
    assert len(owned) == 2 and owned[0] is foreign
    assert feature.audit_int8_row_chunks(second)["selected_wrapper_present"]
    assert len(base.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)) == 1


def actual_int8_model(tmp_path):
    from test_fast_h3_v2_core_sampler import model

    selected = model()
    network = selected.model.diffusion_model
    for name, module in tuple(network.named_modules()):
        if isinstance(module, torch.nn.Linear):
            replacement = actual_int8_dtype(tmp_path, module.weight, torch.float32)
            if module.bias is not None:
                replacement.register_parameter(
                    "bias",
                    torch.nn.Parameter(
                        module.bias.detach().clone(), requires_grad=False
                    ),
                )
            parent, _, field = name.rpartition(".")
            setattr(network.get_submodule(parent), field, replacement)
    return selected


@pytest.mark.parametrize("targets", ["all", "mlp"])
def test_actual_two_step_AV_sampler_hits_selected_INT8_targets_and_resets_request(
    tmp_path, targets
):
    from test_h21_projection_model import sample

    base = actual_int8_model(tmp_path)
    expected = sample(base)
    selected, setup = feature.configure_int8_row_chunks(
        base, row.Int8RowPlan(rows=7, targets=targets)
    )
    assert setup["status"] == "selected_not_executed" and not base.wrappers
    actual = sample(selected)
    for value, reference in zip(
        actual["samples"].unbind(), expected["samples"].unbind(), strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    report = feature.audit_int8_row_chunks(selected)
    assert report["completed_diffusion_calls"] == 2 and report["calls"], report
    assert any("fused_int8" in name for name in report["calls"]), report
    if targets == "all":
        for target in (
            "video_patch_proj",
            "audio_patch_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "final_layer.video_out",
            "final_layer.audio_out",
        ):
            assert any(target in name for name in report["calls"]), (target, report)
    else:
        assert all("mlp" in name for name in report["calls"]), report
    assert report["cleanup_observed"] and not report["request_active"]
    assert not report["portable_cache_identity"] and not report["performance_qualified"]
    sample(selected)
    second = feature.audit_int8_row_chunks(selected)
    assert second["request_number"] > report["request_number"]
    assert second["completed_diffusion_calls"] == 2
    plain = sample(base)
    for value, reference in zip(
        plain["samples"].unbind(), expected["samples"].unbind(), strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_actual_ordered_lora_shared_base_LOW_HIGH_branches_preserve_choices_and_outputs(
    tmp_path,
):
    from comfy.weight_adapter.lora import LoRAAdapter
    from test_h21_projection_model import sample

    base = actual_int8_model(tmp_path)
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = base.model_state_dict()[key]
    for up, down, strength in ((0.03, 0.02, -0.35), (0.05, -0.04, 0.6)):
        adapter = LoRAAdapter(
            {str(strength)},
            (
                torch.full((weight.shape[0], 2), up),
                torch.full((2, weight.shape[1]), down),
                2.0,
                None,
                None,
                None,
            ),
        )
        assert base.add_patches({key: adapter}, strength_patch=strength) == [key]
    ordered = list(base.patches[key])
    expected = sample(base)
    low, _ = feature.configure_int8_row_chunks(
        base, row.Int8RowPlan(rows=5), phase="LOW"
    )
    high, _ = feature.configure_int8_row_chunks(
        base, row.Int8RowPlan(rows=11), phase="HIGH"
    )
    assert low.model is high.model is base.model
    for branch in (low, high):
        assert [item[0] for item in branch.patches[key]] == [
            item[0] for item in ordered
        ]
        assert all(
            item[1] is previous[1]
            for item, previous in zip(branch.patches[key], ordered, strict=True)
        )
        actual = sample(branch)
        for value, reference in zip(
            actual["samples"].unbind(), expected["samples"].unbind(), strict=True
        ):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    lreport, hreport = (
        feature.audit_int8_row_chunks(low),
        feature.audit_int8_row_chunks(high),
    )
    assert (lreport["phase"], hreport["phase"]) == ("LOW", "HIGH")
    assert (
        lreport["completed_diffusion_calls"]
        == hreport["completed_diffusion_calls"]
        == 2
    )
    assert max(item["max_rows"] for item in lreport["calls"].values()) <= 5
    assert 5 < max(item["max_rows"] for item in hreport["calls"].values()) <= 11


@pytest.mark.parametrize("activation", [None, "swiglu", "rms_norm"])
@pytest.mark.parametrize("convrot", [False, True])
def test_real_public_fused_INT8_multidimensional_residual_scale_and_convrot(
    tmp_path, activation, convrot
):
    generator = torch.Generator().manual_seed(212113)
    weight = QuantizedTensor.from_float(
        torch.randn((32, 32), generator=generator).bfloat16(),
        "TensorWiseINT8Layout",
        is_weight=True,
        per_channel=True,
        stochastic_rounding=0,
        convrot=convrot,
        convrot_groupsize=16,
    )
    x = torch.randn(
        (2, 9, 64 if activation == "swiglu" else 32), generator=generator
    ).bfloat16()
    values = dict(
        x=x,
        weight=weight._qdata,
        weight_scale=weight._params.scale,
        out_dtype=torch.bfloat16,
        convrot=convrot,
        convrot_groupsize=16,
        input_act=activation,
        input_act_weight=torch.linspace(0.5, 1.5, 32)
        if activation == "rms_norm"
        else None,
        input_act_eps=1e-6,
        residual=torch.randn((2, 9, 32), generator=generator).bfloat16(),
        residual_scale=torch.linspace(0.1, 0.9, 32).bfloat16(),
    )
    expected = kitchen.int8_linear(**values)
    runtime = row.Int8Runtime(torch.nn.Module(), row.Int8RowPlan(rows=4), "primitive")
    actual = row._public_fused(runtime, "fused", **values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.calls["fused:fused_int8"]["max_rows"] <= 4
    assert actual.shape == (2, 9, 32)


def test_unknown_public_Kitchen_callable_retained_once_without_signature_assumption(
    monkeypatch,
):
    generator = torch.Generator().manual_seed(212114)
    weight = QuantizedTensor.from_float(
        torch.randn((32, 32), generator=generator),
        "TensorWiseINT8Layout",
        is_weight=True,
        stochastic_rounding=0,
    )
    original, seen = kitchen.int8_linear, []

    def foreign(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs) + kwargs["x"].mean()

    monkeypatch.setattr(kitchen, "int8_linear", foreign)
    values = dict(
        x=torch.randn((17, 32), generator=generator),
        weight=weight._qdata,
        weight_scale=weight._params.scale,
        out_dtype=torch.float32,
    )
    expected = foreign(**values)
    seen.clear()
    runtime = row.Int8Runtime(torch.nn.Module(), row.Int8RowPlan(rows=3), "primitive")
    actual = row._public_fused(runtime, "fused", **values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [1] and runtime.report()["partial"] and not runtime.calls


def test_product_never_replaces_public_global_function_or_private_dispatch_table(
    tmp_path,
):
    root = actual_int8_mlp(tmp_path)
    public, core_fused = kitchen.int8_linear, ops.linear_input_act
    table = {
        operator: dict(mapping)
        for operator, mapping in tensor_base._LAYOUT_DISPATCH_TABLE.items()
    }
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212115)
    ).bfloat16()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)):
        root(x)
        assert kitchen.int8_linear is public and ops.linear_input_act is core_fused
        assert set(table) == set(tensor_base._LAYOUT_DISPATCH_TABLE)
        for operator, entries in table.items():
            assert set(entries) == set(tensor_base._LAYOUT_DISPATCH_TABLE[operator])
            assert all(
                tensor_base._LAYOUT_DISPATCH_TABLE[operator][key] is value
                for key, value in entries.items()
            )
    assert kitchen.int8_linear is public and ops.linear_input_act is core_fused


def test_nested_different_unselected_MODEL_is_not_inherited_into_operator_scope(
    tmp_path,
):
    primary = actual_int8_mlp(tmp_path)
    source, _ = int8_module(tmp_path)
    other = LinearRoot(source.control)

    class Composite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.primary = primary
            object.__setattr__(self, "external_MODEL", other)

        def forward(self, x):
            return self.primary(x) + self.external_MODEL(x)

    network = Composite()
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212116)
    ).bfloat16()
    expected = network(x)
    with row.int8_row_scope(network, row.Int8RowPlan(rows=3)) as runtime:
        actual = network(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.calls and all(name.startswith("primary") for name in runtime.calls)
    assert "operator:full_original_owner_or_layout" in runtime.delegated
    assert "forward" not in vars(other)


def test_same_physical_network_recursive_unselected_call_keeps_original_inner_operators(
    tmp_path,
):
    primary = actual_int8_mlp(tmp_path)

    class Recursive(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.primary = primary

        def forward(self, x, inner=False):
            value = self.primary(x)
            return value if inner else value + self(x, inner=True)

    network = Recursive()
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212117)
    ).bfloat16()
    expected = network(x)
    with row.int8_row_scope(network, row.Int8RowPlan(rows=3)) as runtime:
        actual = network(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Only outer invocation counts: each projection has 17 rows, not 34.
    assert runtime.calls and all(
        item["rows"] == 17 for item in runtime.calls.values()
    ), runtime.report()
    assert runtime.report()["partial"]


def test_last_actual_diffusion_binding_change_detected_after_restoration_and_retry(
    tmp_path,
):
    from test_h21_projection_model import sample

    base, plan = actual_int8_model(tmp_path), row.Int8RowPlan(rows=5)
    selected, _ = feature.configure_int8_row_chunks(base, plan)
    seen = []

    def mutate(*_):
        seen.append(1)
        if len(seen) == 2:
            object.__setattr__(plan, "rows", 6)

    handle = base.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match="binding changed during"):
            sample(selected)
    finally:
        handle.remove()
        object.__setattr__(plan, "rows", 5)
    report = feature.audit_int8_row_chunks(selected)
    assert seen == [1, 1] and report["failed_diffusion_calls"] == 1
    assert not report["request_active"]
    assert not any(
        "forward" in vars(module) for module in base.model.diffusion_model.modules()
    )
    sample(selected)
    assert feature.audit_int8_row_chunks(selected)["completed_diffusion_calls"] == 2


def test_unknown_executor_and_zero_execution_audit_do_not_invent_completion():
    from test_fast_h3_v2_core_sampler import model

    selected, _ = feature.configure_int8_row_chunks(model(), row.Int8RowPlan(rows=3))
    assert feature.audit_int8_row_chunks(selected)["completed_diffusion_calls"] == 0
    wrapper = selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)[0]
    seen, answer = [], object()

    class Executor:
        class_obj = SimpleNamespace(other_owner=True)

        def __call__(self, value):
            seen.append(value)
            return answer

    assert wrapper(Executor(), 7) is answer
    report = feature.audit_int8_row_chunks(selected)
    assert (
        seen == [7] and report["partial"] and report["completed_diffusion_calls"] == 1
    )
    assert not report["calls"] and not report["portable_cache_identity"]


def test_actual_ordinary_Stage_keeps_nonportable_identity_and_real_two_step_execution(
    tmp_path,
):
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
    from h3_audio_t8_pkg.modular_sampling import native_explicit
    from h3_audio_t8_pkg.modular_sampling.results import (
        sample_stage,
        selected_model_identity,
    )
    from test_progressive_sampling_runtime import conditioning, latent

    selected, _ = feature.configure_int8_row_chunks(
        actual_int8_model(tmp_path), row.Int8RowPlan(rows=7)
    )
    source = latent()
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 2, 12.0, 3.0
    )
    prepared, sampler, sigmas, context, _ = native_explicit.bind_stage(
        prepared, sampler, sigmas, source
    )
    assert selected_model_identity(prepared)["portable_cache_reuse"] is False
    noise = RandomNoise.execute(212118).result[0]
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    result, _, stage, _ = sample_stage(noise, guider, sampler, sigmas, source, context)
    receipt = stage.verify()
    assert receipt["execution"]["callbacks"] == [0, 1]
    assert (
        receipt["portable_identity"] is False
        and receipt["cache_reuse_authorized"] is False
    )
    assert feature.audit_int8_row_chunks(selected)["completed_diffusion_calls"] == 2
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())


def test_model_without_actual_INT8_calls_does_not_claim_optimization():
    from test_fast_h3_v2_core_sampler import model
    from test_h21_projection_model import sample

    base = model()
    expected = sample(base)
    selected, _ = feature.configure_int8_row_chunks(base, row.Int8RowPlan(rows=3))
    actual = sample(selected)
    for value, reference in zip(
        actual["samples"].unbind(), expected["samples"].unbind(), strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    report = feature.audit_int8_row_chunks(selected)
    assert report["status"] == "observed_no_eligible_calls" and not report["calls"]
    assert (
        report["completed_diffusion_calls"] == 2 and not report["performance_qualified"]
    )


def test_actual_INT8_kernel_shape_error_propagates_without_fallback_and_retry_is_plain(
    tmp_path,
):
    root = actual_int8_mlp(tmp_path)
    bad = torch.randn((17, 31)).bfloat16()
    with pytest.raises(ValueError, match="inner dimensions must match"):
        with row.int8_row_scope(root, row.Int8RowPlan(rows=3)):
            root(bad)
    assert "forward" not in vars(root)
    good = torch.randn((17, 32)).bfloat16()
    expected = root(good)
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)):
        actual = root(good)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [-1, True, 2.5, float("nan")])
def test_invalid_row_configuration_is_real_input_error_not_model_name_gate(rows):
    with pytest.raises(ValueError, match="nonnegative integer"):
        row.Int8RowPlan(rows=rows)
    assert row.Int8RowPlan(rows=10**12).rows == 10**12


def test_actual_layout_registry_owner_not_inferred_from_layout_string_or_base_handler(
    tmp_path, monkeypatch
):
    source, _ = int8_module(tmp_path)
    root = LinearRoot(source.control)
    operator = torch.ops.aten.linear.default
    original = tensor_base._LAYOUT_DISPATCH_TABLE[operator][
        tensor_int8.TensorWiseINT8Layout
    ]
    seen = []

    class OtherLayout(tensor_int8.TensorWiseINT8Layout):
        pass

    def foreign(qtensor, args, kwargs):
        seen.append(args[0].shape[0])
        return original(qtensor, args, kwargs) + args[0].mean()

    monkeypatch.setitem(tensor_base.LAYOUTS, "TensorWiseINT8Layout", OtherLayout)
    monkeypatch.setitem(
        tensor_base._LAYOUT_DISPATCH_TABLE[operator], OtherLayout, foreign
    )
    x = torch.randn(
        (17, 32), generator=torch.Generator().manual_seed(212119)
    ).bfloat16()
    expected = root(x)
    seen.clear()
    with row.int8_row_scope(root, row.Int8RowPlan(rows=3)) as runtime:
        actual = root(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [17] and runtime.report()["partial"]
