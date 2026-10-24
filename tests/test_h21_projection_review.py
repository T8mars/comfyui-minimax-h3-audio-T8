"""Independent adversarial R1 review using actual tiny Core modules/CPU calls.

These cases are not trained-model, GPU peak, speed, quality or portability
qualification. In-place runtime monkeypatches are private to each CPU test and
restored; production files, frozen snapshots and prior tests are never edited.
"""

import importlib.util
from pathlib import Path
import sys
from types import FunctionType, MethodType
import uuid

import pytest
import torch
from comfy import ops
from comfy.ldm.minimax import model as core
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrappersMP

from h3_audio_t8_pkg import h3_projection_chunks as chunks
from h3_audio_t8_pkg import projection_chunks_advanced as feature
from h3_audio_t8_pkg import nodes_projection_chunks as projection_nodes
from test_fast_h3_v2_core_sampler import model
from test_h21_projection_chunks import block, block_inputs, initialize, run_block
from test_h21_projection_model import sample
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def actual_CPU_only(monkeypatch):
    monkeypatch.setattr(core, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def final_layer(heads=1):
    return initialize(
        core.FinalLayer(
            12,
            4,
            3,
            2,
            1e-6,
            pdd_steps=heads,
            dtype=torch.float32,
            device="cpu",
            operations=ops.manual_cast,
        )
    )


def final_inputs():
    x, t, _, _ = block_inputs()
    return (
        x,
        t,
        (0, 12, 0),
        (12, 19, 1),
        torch.tensor(0.6),
        torch.tensor([1.0, 0.8, 0.6, 0.2, 0.0]),
        (12.0, 3.0),
    )


def test_same_key_foreign_callable_identity_is_not_compared_by_value():
    calls = []

    class Foreign:
        __hash__ = None

        def __eq__(self, _other):
            raise AssertionError("A valid user wrapper must never be compared by value")

        def __call__(self, executor, *args, **kwargs):
            calls.append(1)
            return executor(*args, **kwargs)

    bare, foreign = model(), Foreign()
    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, feature.KEY, foreign)
    selected, _ = feature.configure_projection_chunks(
        bare, chunks.ProjectionPlan(rows=5)
    )
    assert selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)[0] is foreign
    assert feature.audit_projection_chunks(selected)["selected_wrapper_present"] is True
    sample(selected)
    assert calls == [1, 1]
    assert feature.audit_projection_chunks(selected)["completed_diffusion_calls"] == 2


@pytest.mark.parametrize("heads", [1, 32])
def test_final_row_coupled_norm_hook_keeps_original_full_stream_semantics(heads):
    final, values, seen = final_layer(heads), final_inputs(), []

    def row_coupled(_module, inputs, output):
        seen.append(inputs[0].shape[0])
        return output + inputs[0].mean(0)

    handle = final.norm.register_forward_hook(row_coupled)
    try:
        expected = final(*values)
        with chunks.projection_scope(final, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = final(*values)
    finally:
        handle.remove()
    maximum = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
    assert maximum < 5e-6, (
        f"real final numerical error={maximum}; hook input rows={seen}"
    )
    assert seen == [12, 7, 12, 7]
    assert runtime.report()["partial"] is True


def test_final_adaln_hook_is_not_repeated_per_output_chunk():
    final, values, seen = final_layer(), final_inputs(), []
    handle = final.adaln_proj.register_forward_hook(lambda *_: seen.append(1))
    try:
        expected = final(*values)
        with chunks.projection_scope(final, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = final(*values)
    finally:
        handle.remove()
    assert seen == [1, 1], (
        f"Actual user adaln hook was called {len(seen)} times, expected exactly one per request"
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert runtime.report()["partial"] is True


def test_final_foreign_row_coupled_norm_forward_is_retained_once_per_stream(
    monkeypatch,
):
    final, values, seen = final_layer(), final_inputs(), []
    previous = final.norm.forward

    def foreign(self, x):
        seen.append(x.shape[0])
        return previous(x) + x.mean(0)

    selected = MethodType(foreign, final.norm)
    monkeypatch.setattr(final.norm, "forward", selected)
    expected = final(*values)
    with chunks.projection_scope(final, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = final(*values)
    maximum = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
    assert maximum < 5e-6, f"foreign final norm numerical error={maximum}; rows={seen}"
    assert final.norm.forward is selected and seen == [12, 7, 12, 7]
    assert runtime.report()["partial"] is True


def test_native_attention_inplace_code_mutation_is_not_misidentified_as_original(
    monkeypatch,
):
    layer, values = block(), block_inputs()
    live = core.Attention.forward
    original = FunctionType(
        live.__code__,
        live.__globals__,
        live.__name__,
        live.__defaults__,
        live.__closure__,
    )
    monkeypatch.setattr(
        layer.attn,
        "_review_original_attention",
        MethodType(original, layer.attn),
        raising=False,
    )

    def changed(self, x, rope_freqs=None, transformer_options={}):
        return (
            self._review_original_attention(
                x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )
            * 1.07
        )

    # Preserve the native source filename and function object; identity/path-only
    # checks must not mistake changed executable instructions for the original.
    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Changed real Attention.forward was bypassed, max error={error}"
    )
    assert runtime.report()["partial"] is True


def test_native_modulation_helper_inplace_code_mutation_is_not_misidentified(
    monkeypatch,
):
    layer, values = block(), block_inputs()
    live = core._mod_scale_shift
    original = FunctionType(
        live.__code__,
        live.__globals__,
        live.__name__,
        live.__defaults__,
        live.__closure__,
    )
    monkeypatch.setattr(
        core, "_h21_review_original_modulation", original, raising=False
    )

    def changed(h, shift, scale, segments):
        return _h21_review_original_modulation(h, shift, scale, segments) + 0.1  # noqa: F821

    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Changed actual Core modulation helper was bypassed, max error={error}"
    )
    assert runtime.report()["partial"] is True


def test_filename_only_row_contract_does_not_treat_full_matrix_LayerNorm_as_RMSNorm():
    root = torch.nn.Module()
    # LayerNorm.forward and RMSNorm.forward share normalization.py. That path
    # alone does not prove that this selected operator is row-local.
    root.video_patch_proj = torch.nn.LayerNorm((19, 12))
    x = block_inputs()[0]
    expected = root.video_patch_proj(x)
    with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = root.video_patch_proj(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.report()["partial"] is True


def test_unknown_stateful_weight_transform_keeps_its_actual_full_call_semantics():
    root = torch.nn.Module()
    root.video_patch_proj = initialize(ops.manual_cast.Linear(12, 8))
    calls = []

    def stateful(weight):
        calls.append(1)
        return weight + len(calls) * 0.01

    root.video_patch_proj.weight_function = [stateful]
    x = block_inputs()[0]
    expected = root.video_patch_proj(x)
    calls.clear()
    with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = root.video_patch_proj(x)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Unknown selected weight callable semantics changed, max error={error}; actual calls={len(calls)}"
    )
    assert calls == [1]
    assert runtime.report()["partial"] is True


@pytest.mark.parametrize("target", ["attention", "modulation"])
def test_modification_before_fresh_primitive_import_is_not_captured_as_native(
    monkeypatch, target
):
    layer, values = block(), block_inputs()
    live = core.Attention.forward if target == "attention" else core._mod_scale_shift
    original = FunctionType(
        live.__code__,
        live.__globals__,
        live.__name__,
        live.__defaults__,
        live.__closure__,
    )
    if target == "attention":
        monkeypatch.setattr(
            layer.attn,
            "_review_original_attention",
            MethodType(original, layer.attn),
            raising=False,
        )

        def changed(self, x, rope_freqs=None, transformer_options={}):
            return (
                self._review_original_attention(
                    x, rope_freqs=rope_freqs, transformer_options=transformer_options
                )
                * 1.07
            )
    else:
        monkeypatch.setattr(
            core, "_h21_review_original_modulation", original, raising=False
        )

        def changed(h, shift, scale, segments):
            return _h21_review_original_modulation(h, shift, scale, segments) + 0.1  # noqa: F821

    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = run_block(layer, values)
    alias = "h3_audio_t8_pkg._projection_review_fresh_" + uuid.uuid4().hex
    specification = importlib.util.spec_from_file_location(alias, Path(chunks.__file__))
    fresh = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, alias, fresh)
    specification.loader.exec_module(fresh)
    # The changed live function exists BEFORE this independently imported
    # primitive captures its constants. Import-time snapshots are not stock-
    # source authentication, and must not authorize a reimplementation.
    with fresh.projection_scope(layer, fresh.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Pre-import changed {target} was mistaken for native; max error={error}"
    )
    assert runtime.report()["partial"] is True


def test_shared_network_registry_never_invokes_unknown_module_equality():
    class Foreign(torch.nn.Module):
        __hash__ = object.__hash__

        def __eq__(self, _other):
            raise AssertionError(
                "Physical network lease must use identity, not foreign equality"
            )

        def forward(self, x):
            return x + 1

    network = Foreign()
    x = torch.tensor([1.0])
    for _ in range(2):
        with chunks.projection_scope(network, chunks.ProjectionPlan(rows=3)) as runtime:
            assert network(x).item() == 2.0
        assert runtime.report()["partial"] is True


@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_legitimate_global_module_hook_keeps_full_query_projection_call(hook_kind):
    layer, values, seen = block(), block_inputs(), []
    target = layer.attn.qkv_proj
    if hook_kind == "pre":

        def hook(module, inputs):
            if module is target:
                seen.append(inputs[0].shape[0])
                return (inputs[0] + inputs[0].mean(0), *inputs[1:])

        handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
    else:

        def hook(module, inputs, output):
            if module is target:
                seen.append(inputs[0].shape[0])
                return output + output.mean(0)

        handle = torch.nn.modules.module.register_module_forward_hook(hook)
    try:
        expected = run_block(layer, values)
        with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = run_block(layer, values)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=4e-6)
    assert seen == [19, 19]
    assert runtime.report()["partial"] is True


def test_real_ordinary_stage_executes_projection_without_portable_identity_promotion():
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
    from h3_audio_t8_pkg.modular_sampling import native_explicit
    from h3_audio_t8_pkg.modular_sampling.results import (
        sample_stage,
        selected_model_identity,
    )

    selected, _ = feature.configure_projection_chunks(
        model(), chunks.ProjectionPlan(rows=7)
    )
    source = latent()
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 2, 12.0, 3.0
    )
    prepared, sampler, sigmas, context, _ = native_explicit.bind_stage(
        prepared, sampler, sigmas, source
    )
    assert selected_model_identity(prepared)["portable_cache_reuse"] is False
    noise = RandomNoise.execute(21102199).result[0]
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    result, _, stage_result, _ = sample_stage(
        noise, guider, sampler, sigmas, source, context
    )
    receipt = stage_result.verify()
    assert receipt["execution"]["callbacks"] == [0, 1]
    assert receipt["portable_identity"] is False
    assert receipt["cache_reuse_authorized"] is False
    assert feature.audit_projection_chunks(selected)["completed_diffusion_calls"] == 2
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())


def test_next_real_pre_run_clears_old_counts_and_ordering_edge_does_not_invent_new_completion():
    selected, _ = feature.configure_projection_chunks(
        model(), chunks.ProjectionPlan(rows=6)
    )
    old_output = sample(selected)
    previous = feature.audit_projection_chunks(selected)
    assert previous["completed_diffusion_calls"] == 2
    selected.pre_run()
    try:
        text = projection_nodes.MiniMaxH3ProjectionChunksAuditEXPT8.execute(
            selected, old_output
        ).result[0]
        import json

        report = json.loads(text)
        assert report["completed_diffusion_calls"] == 0
        assert report["request_number"] > previous["request_number"]
        assert report["status"] == "selected_not_executed"
        assert not report["portable_cache_identity"] and not report["quality_accepted"]
    finally:
        selected.cleanup()


def test_disabled_node_on_same_original_input_restores_plain_actual_sampling():
    bare = model()
    expected = sample(bare)
    enabled, _ = projection_nodes.MiniMaxH3ProjectionChunksEXPT8.execute(
        bare, rows=5
    ).result
    sample(enabled)
    disabled, _ = projection_nodes.MiniMaxH3ProjectionChunksEXPT8.execute(
        bare, enabled=False
    ).result
    assert disabled is bare
    actual = sample(disabled)
    for value, reference in zip(
        actual["samples"].unbind(), expected["samples"].unbind(), strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert feature.audit_projection_chunks(disabled)["status"] == "not_selected"


def test_foreign_same_key_wrapper_survives_own_reconfiguration_without_equality():
    class Foreign:
        __hash__ = None

        def __eq__(self, _other):
            raise AssertionError("No equality during own reconfiguration")

        def __call__(self, executor, *args, **kwargs):
            return executor(*args, **kwargs)

    bare, foreign = model(), Foreign()
    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, feature.KEY, foreign)
    first, _ = feature.configure_projection_chunks(bare, chunks.ProjectionPlan(rows=5))
    second, _ = feature.configure_projection_chunks(
        first, chunks.ProjectionPlan(rows=9)
    )
    wrappers = second.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)
    assert len(wrappers) == 2 and wrappers[0] is foreign
    sample(second)
    report = feature.audit_projection_chunks(second)
    assert report["plan"]["rows"] == 9 and report["completed_diffusion_calls"] == 2
    assert feature.audit_projection_chunks(first)["completed_diffusion_calls"] == 0


def test_actual_ops_dynamic_cast_delegate_is_not_assumed_row_local(monkeypatch):
    root = torch.nn.Module()
    root.video_patch_proj = initialize(ops.manual_cast.Linear(12, 8))
    selected = root.video_patch_proj
    original = selected.forward_comfy_cast_weights
    calls = []

    def foreign(self, x):
        calls.append(x.shape[0])
        return original(x + x.mean(0))

    method = MethodType(foreign, selected)
    monkeypatch.setattr(selected, "forward_comfy_cast_weights", method)
    x = block_inputs()[0]
    expected = selected(x)
    with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = selected(x)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Real selected ops dynamic cast delegate split; max error={error}; rows={calls}"
    )
    assert calls == [19, 19] and selected.forward_comfy_cast_weights is method
    assert runtime.report()["partial"] is True


def test_real_dense_attention_closed_function_code_change_retains_full_queries(
    monkeypatch,
):
    layer, values = block(), block_inputs()
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
    monkeypatch.setattr(attention, "_h21_review_original_sdpa", original, raising=False)

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
        output = _h21_review_original_sdpa(  # noqa: F821
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )  # noqa: F821
        return output + q.mean()

    monkeypatch.setattr(
        live,
        "__code__",
        changed.__code__.replace(co_filename=live.__code__.co_filename),
    )
    expected = run_block(layer, values)
    with chunks.projection_scope(
        layer, chunks.ProjectionPlan(rows=3, query_rows=4)
    ) as runtime:
        actual = run_block(layer, values)
    error = float((actual - expected).abs().max())
    assert error < 5e-6, (
        f"Changed actual dense attention closure was query-sliced; max error={error}"
    )
    assert runtime.query_spans == []
    assert runtime.report()["partial"] is True


def test_real_last_forward_error_is_not_replaced_by_post_binding_check_and_retry_recovers():
    bare, plan, calls = model(), chunks.ProjectionPlan(rows=5), []

    def fail_last(*_):
        calls.append(1)
        if len(calls) == 2:
            object.__setattr__(plan, "rows", 6)
            raise RuntimeError("actual user final callback failure")

    selected, _ = feature.configure_projection_chunks(bare, plan)
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(fail_last)
    try:
        with pytest.raises(RuntimeError, match="actual user final callback failure"):
            sample(selected)
    finally:
        handle.remove()
        object.__setattr__(plan, "rows", 5)
    assert calls == [1, 1]
    failed = feature.audit_projection_chunks(selected)
    assert failed["failed_diffusion_calls"] == 1 and not failed["request_active"]
    assert failed["errors"][-1] == "RuntimeError: actual user final callback failure"
    sample(selected)
    recovered = feature.audit_projection_chunks(selected)
    assert recovered["failed_diffusion_calls"] == 0
    assert recovered["completed_diffusion_calls"] == 2
    assert recovered["request_number"] > failed["request_number"]
    assert not recovered["portable_cache_identity"]


def test_unknown_executor_remains_callable_but_own_post_mutation_cannot_return_success():
    plan = chunks.ProjectionPlan(rows=5)
    selected, _ = feature.configure_projection_chunks(model(), plan)
    wrapper = selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)[0]
    seen = []

    class Executor:
        class_obj = object()

        def __call__(self, value):
            seen.append(value)
            object.__setattr__(plan, "rows", 6)
            return value

    try:
        with pytest.raises(ValueError, match="binding changed during"):
            wrapper(Executor(), 7)
    finally:
        object.__setattr__(plan, "rows", 5)
    assert seen == [7]
    report = feature.audit_projection_chunks(selected)
    assert report["failed_diffusion_calls"] == 1
    assert report["completed_diffusion_calls"] == 0
    assert not report["portable_cache_identity"]


@pytest.mark.parametrize("target", ["attention", "qkv", "final_norm"])
def test_actual_module_call_override_is_not_bypassed_or_chunk_reentered(
    monkeypatch, target
):
    if target == "final_norm":
        root, values = final_layer(), final_inputs()
        selected = root.norm
    else:
        root, values = block(), block_inputs()
        selected = root.attn if target == "attention" else root.attn.qkv_proj
    original_class, rows = type(selected), []

    class ForeignCall(original_class):
        # Inherit the exact source-authenticated forward unchanged, but retain
        # the actual legitimate user-selected module call semantics too.
        def __call__(self, x, *args, **kwargs):
            rows.append(x.shape[0])
            if target == "attention":
                return super().__call__(x, *args, **kwargs) * 1.07
            return super().__call__(x + x.mean(0), *args, **kwargs)

    monkeypatch.setattr(selected, "__class__", ForeignCall)
    if target == "final_norm":
        expected = root(*values)
        with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = root(*values)
        error = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
        assert rows == [12, 7, 12, 7], (
            f"Actual final custom __call__ stream rows changed: {rows}"
        )
    else:
        expected = run_block(root, values)
        with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = run_block(root, values)
        error = float((actual - expected).abs().max())
        assert rows == [19, 19], (
            f"Actual {target} custom __call__ was bypassed/reentered: {rows}; maxerror={error}"
        )
    assert error < 5e-6, f"Actual {target} module __call__ math changed: {error}"
    assert runtime.report()["partial"] is True
