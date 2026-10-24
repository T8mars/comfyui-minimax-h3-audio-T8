"""Actual tiny Core projection math and real request isolation, never GPU media."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from types import MethodType

import pytest
import torch
from torch import nn

from comfy import ops
from comfy.ldm.minimax import model as core
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter.lora import LoRAAdapter

from h3_audio_t8_pkg import h3_projection_chunks as chunks
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.prepared_backend.resource_guard import SerialProbeLease


@pytest.fixture(scope="module", autouse=True)
def cpu_lease(tmp_path_factory):
    with SerialProbeLease(tmp_path_factory.mktemp("r1-cpu") / "projection.serial.lock"):
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(autouse=True)
def inference(monkeypatch):
    monkeypatch.setattr(core, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def initialize(module):
    generator = torch.Generator().manual_seed(211041)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.07)
        for layer in module.modules():
            if isinstance(layer, nn.RMSNorm):
                layer.weight.fill_(1)
    module.requires_grad_(False)
    return module


def block():
    return initialize(
        core.DiTBlock(
            12,
            2,
            8,
            16,
            4,
            1e-6,
            1e-6,
            dtype=torch.float32,
            device="cpu",
            operations=ops.manual_cast,
        )
    )


def block_inputs(vector=True, rope=True):
    generator = torch.Generator().manual_seed(211042)
    x = torch.randn((19, 12), generator=generator)
    t = torch.randn((3, 4), generator=generator)
    row = torch.tensor([0, 2, 3, 5, 1, 7, 8]) if vector else 2
    segments = [(0, 3, 0), (3, 10, row), (10, 19, 5)]
    angles = torch.randn((19, 6), generator=generator)
    return (
        x,
        t,
        segments,
        core.rope_rotation_table(angles, torch.float32) if rope else None,
    )


def run_block(layer, values, options=None):
    x, t, segments, rope = values
    return layer(
        x.clone(),
        t,
        segments,
        rope,
        transformer_options={} if options is None else options,
    )


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("stage", [False, True])
def test_native_fused_bounded_norm_qkv_full_kv_global_rope_and_mod_rows(
    vector, rope, stage
):
    layer, values = block(), block_inputs(vector, rope)
    before = {module: "forward" in module.__dict__ for module in layer.modules()}
    expected = run_block(layer, values)
    with chunks.projection_scope(
        layer, chunks.ProjectionPlan(rows=3, query_rows=4, stage_queries=stage)
    ) as runtime:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert runtime.max_live_norm_rows <= 3
    assert runtime.calls["norm_qkv.fused"]["rows"] == 19
    assert runtime.query_spans == [
        (0, 4, 19),
        (4, 8, 19),
        (8, 12, 19),
        (12, 16, 19),
        (16, 19, 19),
    ]
    assert runtime.persistent_kv_bytes == 2 * 19 * 2 * 8 * 4
    assert bool(runtime.staged_q_bytes) is stage
    assert runtime.report()["portable_cache_identity"] is False
    assert all(
        ("forward" in module.__dict__) is present for module, present in before.items()
    )
    torch.testing.assert_close(run_block(layer, values), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive", "broadcast"])
def test_dense_query_mask_slices_use_full_kv_and_absolute_query_order(mask_kind):
    generator = torch.Generator().manual_seed(211043)
    q, k, v = [torch.randn((1, 2, 11, 8), generator=generator) for _ in range(3)]
    mask = None
    if mask_kind == "bool":
        mask = torch.ones(11, 11, dtype=torch.bool)
        mask[4:, :2] = False
    elif mask_kind == "additive":
        mask = torch.randn((1, 2, 11, 11), generator=generator) * 0.3
    elif mask_kind == "broadcast":
        mask = torch.randn((1, 1, 1, 11), generator=generator)
    expected = attention.attention_pytorch(q, k, v, 2, mask=mask, skip_reshape=True)
    runtime = chunks.ProjectionRuntime(chunks.ProjectionPlan(query_rows=3), "HIGH")
    actual = chunks.dense_attention(runtime, q, k, v, 2, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(span[2] == 11 for span in runtime.query_spans)
    truncated = attention.attention_pytorch(
        q[:, :, :3], k[:, :, :3], v[:, :, :3], 2, skip_reshape=True
    )
    assert not torch.allclose(actual[:, :3], truncated)


def test_real_relay_receives_full_q_and_keeps_nonzero_temporal_bias():
    layer, values = block(), block_inputs()
    seen = []
    route = {
        "seq_len": 19,
        "query_segments": (
            {
                "start": 3,
                "end": 19,
                "query_times": torch.arange(16, dtype=torch.float32),
            },
        ),
        "events": (
            {
                "text_key_start": 0,
                "text_key_end": 3,
                "midpoint": 2.0,
                "window": 0.2,
                "sigma": 0.7,
            },
        ),
    }

    def override(_original, q, k, v, heads, **kwargs):
        seen.append((q.shape[-2], k.shape[-2]))
        return relay.route_prompt_relay_attention(
            q, k, v, heads, query_chunk_rows=4, **kwargs
        )

    options = {
        "optimized_attention_override": override,
        relay.PROMPT_RELAY_RUNTIME_KEY: route,
    }
    expected = run_block(layer, values, options)
    unmasked = run_block(layer, values)
    with chunks.projection_scope(
        layer, chunks.ProjectionPlan(rows=3, query_rows=2, stage_queries=True)
    ) as runtime:
        actual = run_block(layer, values, options)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert not torch.allclose(actual, unmasked, atol=1e-7)
    assert seen == [(19, 19), (19, 19)]
    assert "attention:full_query_route" in runtime.delegated
    assert "query_staging:full_query_route" in runtime.delegated
    assert runtime.query_spans == [] and runtime.staged_q_bytes == 0


def test_foreign_attention_and_module_owner_delegate_without_shape_guess():
    layer, values = block(), block_inputs()
    previous = layer.attn.forward
    calls = []

    def foreign(self, x, *args, **kwargs):
        calls.append(x.shape[0])
        return previous(x, *args, **kwargs) * 1.03

    selected = MethodType(foreign, layer.attn)
    layer.attn.forward = selected
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert calls == [19, 19]
    assert layer.attn.forward is selected
    assert runtime.report()["partial"] is True


def test_foreign_core_attention_container_route_still_receives_full_queries(
    monkeypatch,
):
    layer, values = block(), block_inputs()
    seen = []

    def container(q, k, v, heads, **kwargs):
        q, k, v = q.take(), k.take(), v.take()
        seen.append((q.shape[-2], k.shape[-2]))
        kwargs.pop("preferred_attention", None)
        return attention.attention_pytorch(
            q, k, v, heads, **{**kwargs, "_inside_attn_wrapper": True}
        )

    monkeypatch.setattr(attention.attention_pytorch, "container_function", container)
    expected = run_block(layer, values)
    with chunks.projection_scope(
        layer, chunks.ProjectionPlan(rows=3, query_rows=2)
    ) as runtime:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert seen == [(19, 19), (19, 19)]
    assert runtime.query_spans == []
    assert "attention:full_query_route" in runtime.delegated


def test_falsey_unhashable_attention_override_is_not_mistaken_for_no_override():
    layer, values = block(), block_inputs()
    seen = []

    class Foreign:
        __hash__ = None

        def __bool__(self):
            return False

        def __eq__(self, _other):
            raise AssertionError("An unknown owner must not be compared by value")

        def __call__(self, original, q, k, v, heads, **kwargs):
            seen.append(q.shape[-2])
            return original(q, k, v, heads, **kwargs) * 1.01

    options = {"optimized_attention_override": Foreign()}
    expected = run_block(layer, values, options)
    with chunks.projection_scope(
        layer, chunks.ProjectionPlan(rows=3, query_rows=2, stage_queries=True)
    ) as runtime:
        actual = run_block(layer, values, options)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert seen == [19, 19]
    assert runtime.query_spans == [] and runtime.report()["partial"] is True


def tiny_network():
    module = initialize(
        core.MiniMaxH3Model(
            hidden_size=12,
            num_layers=2,
            token_refiner_num_layers=1,
            num_attention_heads=2,
            attention_head_dim=8,
            ffn_hidden_size=16,
            text_dim=8,
            timestep_input_dim=4,
            time_embed_hidden_size=12,
            time_embed_dim=4,
            rope_inv_freq_len=1,
            dtype=torch.float32,
            device="cpu",
            operations=ops.manual_cast,
        )
    )
    module.rope.inv_freq.fill_(1)
    return module


def network_inputs():
    generator = torch.Generator().manual_seed(211044)
    x = [
        torch.randn((1, 24, 2, 4, 4), generator=generator),
        torch.randn((1, 32, 2, 3), generator=generator),
    ]
    context = torch.randn((1, 3, 8), generator=generator)
    keyframe = {
        "resolved_frame_index": 0,
        "latent": x[0][:, :, :1].clone(),
        "audio_latent": x[1][:, :, :, :1].clone(),
    }
    refs = [
        {"kind": "image", "latent_h": 4, "latent_w": 4},
        {"kind": "audio", "ref_audio_t": 1},
    ]
    payload = {
        "keyframes": [keyframe],
        "refs": refs,
        "cond_video_latents": [x[0][:, :, :1].clone(), x[0][:, :, :1].clone()],
        "cond_audio_latents": [x[1][:, :, :, :1].clone(), x[1][:, :, :, :1].clone()],
    }
    video_mask = torch.ones(1, 1, 2, 4, 4)
    video_mask[:, :, 0, :2] = 0.3
    video_mask[:, :, 1, :2] = 0
    return x, torch.tensor([600.0]), context, payload, video_mask


def forward_network(network, values):
    x, timestep, context, payload, mask = values
    return network(
        [item.clone() for item in x],
        timestep,
        context,
        transformer_options={"sample_sigmas": torch.tensor([1.0, 0.6, 0.2, 0.0])},
        minimax_payload=deepcopy(payload),
        denoise_mask=mask,
        audio_denoise_mask=torch.tensor([[[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]]]),
    )


def test_actual_full_core_embedding_refs_first_mask_and_fp32_final_roundtrip():
    network, values = tiny_network(), network_inputs()
    expected = forward_network(network, values)
    original_parameters = {
        key: value.clone() for key, value in network.state_dict().items()
    }
    with chunks.projection_scope(
        network, chunks.ProjectionPlan(rows=3, query_rows=4)
    ) as runtime:
        actual = forward_network(network, values)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=3e-5, atol=4e-6)
    assert runtime.calls["video_patch_proj"]["max_rows"] <= 3
    assert runtime.calls["audio_patch_proj"]["max_rows"] <= 3
    assert runtime.calls["condition_proj"]["rows"] == 3
    assert runtime.calls["final.delegate"]["max_rows"] <= 3
    assert all(
        torch.equal(value, original_parameters[key])
        for key, value in network.state_dict().items()
    )


@pytest.mark.parametrize("heads", [1, 32])
@pytest.mark.parametrize("audio_empty", [False, True])
def test_real_original_final_pdd_absolute_schedule_vector_rows_and_empty_range(
    heads, audio_empty
):
    final = initialize(
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
    x, t, _, _ = block_inputs()
    va = (7, 19, torch.tensor([0, 1, 2] * 4))
    aa = (
        3,
        3 if audio_empty else 7,
        torch.tensor([], dtype=torch.long) if audio_empty else 2,
    )
    args = (
        x,
        t,
        va,
        aa,
        torch.tensor(0.6),
        torch.tensor([1.0, 0.8, 0.6, 0.2, 0.0]),
        (12.0, 3.0),
    )
    expected = final(*args)
    with chunks.projection_scope(final, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = final(*args)
    for value, reference in zip(actual, expected):
        assert value.dtype is torch.float32
        torch.testing.assert_close(value, reference, rtol=2e-5, atol=3e-6)
    assert runtime.calls["final.delegate"]["max_rows"] <= 3


def test_real_core_lora_weight_merge_and_weight_functions_are_not_bypassed():
    network = tiny_network()
    base = nn.Module()
    base.diffusion_model = network
    base.device = torch.device("cpu")
    patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = network.blocks[0].attn.qkv_proj.weight
    before = weight.clone()
    adapter = LoRAAdapter(
        {"owned"},
        (
            torch.full((weight.shape[0], 2), 0.02),
            torch.full((2, weight.shape[1]), 0.04),
            2.0,
            None,
            None,
            None,
        ),
    )
    assert patcher.add_patches({target: adapter}, 0.7) == [target]
    patcher.patch_model()
    try:
        values = block_inputs()
        layer = network.blocks[0]
        merged_expected = run_block(layer, values)
        with chunks.projection_scope(network, chunks.ProjectionPlan(rows=3)) as merged:
            merged_actual = run_block(layer, values)
        torch.testing.assert_close(merged_actual, merged_expected, rtol=3e-5, atol=3e-6)
        assert merged.calls["norm_qkv.fused"]["max_rows"] <= 3
        transforms = []

        def transform(value):
            transforms.append(True)
            return value + 0.001

        layer.attn.qkv_proj.weight_function = [transform]
        expected = run_block(layer, values)
        transforms.clear()
        with chunks.projection_scope(
            network, chunks.ProjectionPlan(rows=3)
        ) as delegated:
            actual = run_block(layer, values)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert transforms == [True]  # Unknown callbacks keep the original call count.
        assert delegated.report()["partial"] is True
        assert not torch.equal(layer.attn.qkv_proj.weight, before)
        layer.attn.qkv_proj.weight_function = []
    finally:
        patcher.unpatch_model()
    torch.testing.assert_close(
        network.blocks[0].attn.qkv_proj.weight, before, rtol=0, atol=0
    )


def test_actual_int8_module_owned_input_scale_and_forward_hooks_execute(tmp_path):
    from test_h21_quant_execution import int8_module

    source, _ = int8_module(tmp_path)
    layer = source.control
    scale = torch.linspace(0.25, 1.75, 32)
    layer.register_parameter(
        "pre_quant_scale", nn.Parameter(scale, requires_grad=False)
    )
    root = nn.Module()
    root.video_patch_proj = layer
    x = torch.randn(
        (11, 32), generator=torch.Generator().manual_seed(211045)
    ).bfloat16()
    expected = layer(x)
    calls = []
    handle = layer.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = layer(x)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [True]  # Existing module hook gets its original full call.
    assert runtime.calls["video_patch_proj"]["chunks"] == 4
    assert torch.equal(layer.pre_quant_scale, scale)


def test_unknown_row_coupled_module_and_hook_semantics_delegate_once():
    root = nn.Module()
    root.video_patch_proj = initialize(ops.manual_cast.Linear(12, 8))
    selected = root.video_patch_proj.forward
    seen = []

    def foreign(self, x):
        seen.append(x.shape[0])
        return selected(x + x.mean(0))

    method = MethodType(foreign, root.video_patch_proj)
    root.video_patch_proj.forward = method
    x = block_inputs()[0]
    expected = root.video_patch_proj(x)
    with chunks.projection_scope(root, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = root.video_patch_proj(x)
    assert seen == [19, 19]
    assert root.video_patch_proj.forward is method
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.report()["partial"] is True


def test_nested_shared_network_plans_and_disabled_phase_do_not_mix_counters():
    layer, values = block(), block_inputs()
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, None):
        with chunks.projection_scope(
            layer, chunks.ProjectionPlan(rows=3), phase="LOW"
        ) as low:
            run_block(layer, values)
            before = deepcopy(low.calls)
            with chunks.projection_scope(
                layer, chunks.ProjectionPlan(rows=7), phase="HIGH"
            ) as high:
                torch.testing.assert_close(
                    run_block(layer, values), expected, rtol=3e-5, atol=3e-6
                )
            with chunks.projection_scope(layer, None) as off:
                torch.testing.assert_close(
                    run_block(layer, values), expected, rtol=0, atol=0
                )
            assert low.calls == before and not off.calls
            run_block(layer, values)
    assert low.calls["norm_qkv.fused"]["rows"] == 38
    assert high.calls["norm_qkv.fused"]["rows"] == 19
    assert not any("forward" in module.__dict__ for module in layer.modules())


def test_actual_model_patcher_clone_shares_network_and_uses_same_nested_lease():
    base = nn.Module()
    base.diffusion_model = block()
    base.device = torch.device("cpu")
    source = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    clone = source.clone()
    shared = source.get_model_object("diffusion_model")
    assert clone.get_model_object("diffusion_model") is shared
    values = block_inputs()
    expected = run_block(shared, values)
    with chunks.projection_scope(shared, chunks.ProjectionPlan(rows=3)) as outer:
        with chunks.projection_scope(
            clone.get_model_object("diffusion_model"), chunks.ProjectionPlan(rows=7)
        ) as inner:
            actual = run_block(shared, values)
        assert not outer.calls
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert inner.max_live_norm_rows <= 7
    assert not any("forward" in module.__dict__ for module in shared.modules())


def test_true_threads_plain_request_delegates_and_selected_requests_serialize():
    layer, values = block(), block_inputs()
    expected = run_block(layer, values)
    started, entered = threading.Event(), threading.Event()

    def selected():
        started.set()
        with (
            torch.no_grad(),
            chunks.projection_scope(layer, chunks.ProjectionPlan(rows=7)) as runtime,
        ):
            entered.set()
            return run_block(layer, values), runtime

    def ordinary():
        with torch.no_grad():
            return run_block(layer, values)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as low:
            blocked = pool.submit(selected)
            assert started.wait(2) and not entered.wait(0.1)
            torch.testing.assert_close(
                pool.submit(ordinary).result(timeout=10), expected, rtol=0, atol=0
            )
            assert not low.calls
            torch.testing.assert_close(
                run_block(layer, values), expected, rtol=3e-5, atol=3e-6
            )
        actual, high = blocked.result(timeout=10)
    assert entered.is_set() and high.max_live_norm_rows <= 7
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert not any("forward" in module.__dict__ for module in layer.modules())


def test_real_projection_exception_restores_methods_then_next_ordinary_request():
    layer, values = block(), block_inputs()
    expected = run_block(layer, values)
    calls = []

    def fail(value):
        calls.append(True)
        raise RuntimeError("owned projection failure")

    layer.attn.qkv_proj.weight_function = [fail]
    with pytest.raises(RuntimeError, match="owned projection failure"):
        with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)):
            run_block(layer, values)
    assert calls == [True]  # Preserve the unknown callback's actual error, no retry.
    assert not any("forward" in module.__dict__ for module in layer.modules())
    layer.attn.qkv_proj.weight_function = []
    torch.testing.assert_close(run_block(layer, values), expected, rtol=0, atol=0)


def test_later_selected_forward_is_not_overwritten_during_cleanup():
    layer = block()
    selected = layer.norm1.forward
    with chunks.projection_scope(layer, chunks.ProjectionPlan()) as runtime:
        layer.norm1.forward = selected
    assert layer.norm1.forward is selected
    assert "scope:later_owner_preserved" in runtime.delegated


def test_existing_row_coupled_norm_hook_keeps_its_full_sequence_call():
    layer, values = block(), block_inputs()
    calls = []

    def hook(_module, inputs, output):
        calls.append(inputs[0].shape[0])
        return output + inputs[0].mean(0)

    handle = layer.norm1.register_forward_hook(hook)
    try:
        expected = run_block(layer, values)
        with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = run_block(layer, values)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert calls == [19, 19]
    assert runtime.max_live_norm_rows == 0
    assert "block:fused_norm_qkv_delegate" in runtime.delegated


def test_real_dispatcher_observes_bounded_actual_qkv_gemm_rows():
    from torch.utils._python_dispatch import TorchDispatchMode

    rows = []

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if str(function) == "aten.mm.default" and tuple(args[1].shape) == (12, 48):
                rows.append(args[0].shape[0])
            return function(*args, **(kwargs or {}))

    layer, values = block(), block_inputs()
    with Observe(), chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)):
        run_block(layer, values)
    assert max(rows) <= 3 and sum(rows) == 19


def test_later_core_class_forward_is_not_misidentified_by_dynamic_class_comparison(
    monkeypatch,
):
    native = core.DiTBlock.forward
    calls = []

    def foreign(self, *args, **kwargs):
        calls.append(True)
        return native(self, *args, **kwargs) * 1.1

    monkeypatch.setattr(core.DiTBlock, "forward", foreign)
    layer, values = block(), block_inputs()
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert calls == [True, True] and runtime.max_live_norm_rows == 0
    assert runtime.report()["partial"] is True


def test_unhashable_foreign_network_keeps_execution_without_temporary_mutation():
    class Foreign(nn.Module):
        __hash__ = None

        def forward(self, x):
            return x + 2

    network = Foreign()
    with chunks.projection_scope(network, chunks.ProjectionPlan()) as runtime:
        assert network(torch.tensor(3)).item() == 5
    assert runtime.report()["partial"] is True
    assert "forward" not in network.__dict__


def test_core_accepted_noncontiguous_modulation_is_delegated_not_newly_banned():
    layer = block()
    x, t, _, rope = block_inputs()
    values = x, t, [(1, 3, 0), (6, 10, 2), (11, 19, -1)], rope
    expected = run_block(layer, values)
    with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3)) as runtime:
        actual = run_block(layer, values)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert "block:nonstandard_mod_segments" in runtime.delegated


@pytest.mark.parametrize("masked", [False, True])
def test_genuine_hyperflow_seven_parameter_final_delegate_remaps_before_chunking(
    monkeypatch, masked
):
    from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
    from h3_audio_t8_pkg.hyperflow_weights_advanced import DEFAULT_RAW_GRID
    from h3_audio_t8_pkg.hyperflow_runtime_advanced import (
        HyperFlowStep,
        install_hyperflow,
        pop_step,
        push_step,
    )
    from h3_audio_t8_pkg.hyperflow_sampling_advanced import shifted_grid

    bare, network = _tiny_native_model(monkeypatch)
    patched, binding, _ = install_hyperflow(bare, _tiny_weights(network))
    sigmas = shifted_grid(DEFAULT_RAW_GRID, 12.0)
    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
    context = torch.randn(1, 2, 8)
    kwargs = {}
    if masked:
        mask = torch.ones(1, 1, 2, 4, 4)
        mask[:, :, 0, :2] = 0.5
        mask[:, :, 1, :2] = 0
        kwargs = {
            "denoise_mask": mask,
            "audio_denoise_mask": torch.tensor([[[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]]]),
        }

    def invoke():
        options = {
            "sample_sigmas": sigmas,
            "wrappers": patched.wrappers,
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
        }
        token = push_step(
            HyperFlowStep(binding.owner, 0, float(sigmas[0]), float(sigmas[1]))
        )
        try:
            return network(
                [item.clone() for item in x],
                torch.tensor([float(sigmas[0]) * 1000]),
                context,
                transformer_options=options,
                minimax_payload={"layout": core.PackedLayout(2, 2, 4, 4, 3)},
                **kwargs,
            )
        finally:
            pop_step(token)

    patched.patch_model()
    original = network.final_layer.forward
    try:
        expected = invoke()
        with chunks.projection_scope(network, chunks.ProjectionPlan(rows=3)) as runtime:
            actual = invoke()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-6)
        assert network.final_layer.forward is original
        assert runtime.calls["final.delegate"]["max_rows"] <= 3
        assert "block:fused_norm_qkv_delegate" in runtime.delegated
    finally:
        patched.unpatch_model()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan")])
def test_real_invalid_chunk_fields_rejected_without_model_name_gate(value):
    with pytest.raises(ValueError, match="positive integer"):
        chunks.ProjectionPlan(rows=value)
    assert chunks.ProjectionPlan(rows=10**12).rows == 10**12
