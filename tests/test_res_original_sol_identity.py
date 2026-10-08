"""Original Sol structure, not CUDA/kernel or portable recovery qualification."""
import torch

from h3_audio_t8_pkg import sol_attn_minimax_v2 as sol
from h3_audio_t8_pkg import h3_memory_advanced as t8
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from h3_audio_t8_pkg.res_sol_identity import inspect_original_sol
from test_res_model_identity import model, source_latent, prepared
from test_relay_kj_memory import memory_nodes  # noqa: F401 -- real installed KJ source fixture
from test_relay_kj_backend import kj  # noqa: F401 -- dependent installed source fixture


def original_sol(request, tmp_path, *, tau=1.3, dense_blocks=""):
    from comfy.ldm.modules import attention
    # Match the actual GPU graph: memory-efficient Sage is a direct forward,
    # unlike KJ's separate low-memory forward which already uses the selector.
    sage = request.getfixturevalue("memory_nodes")[1]
    selected = t8.configure_chunk_feed_forward(model(), 2, 256)[0]
    selected = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(selected).result[0]
    # Invoke ModelAttentionBackend's actual Core setter/selected function. Its
    # unrelated module-wide `import nodes` collides with the test package's path.
    selected = selected.clone()
    selected.set_model_optimized_attention(attention.get_attention_function("pytorch", None))
    selected = prepared(selected, source_latent(), tmp_path, mode="disabled")[0]
    return sol.SolAttnMiniMax.execute(selected, tau=tau, start_percent=.2, end_percent=.9,
        min_tokens=12288, sink_conditioning="exact_kv_and_rows", morton=False,
        morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
        reuse_qkv_memory=False, verbose=False, dense_blocks=dense_blocks).result[0]


def snapshot(model):
    return (dict(model.object_patches), model.model_options["transformer_options"]["optimized_attention_override"],
            [(module, dict(module._forward_pre_hooks), dict(module._forward_hooks), vars(module).get("forward"))
             for module in model.model.modules()], core_layout())


def core_layout():
    from comfy.ldm.minimax.model import PackedLayout
    return PackedLayout.__init__


def assert_untouched(model, before):
    methods, selector, modules, layout = before
    assert model.object_patches == methods
    assert model.model_options["transformer_options"]["optimized_attention_override"] is selector
    for module, pre, post, forward in modules:
        assert dict(module._forward_pre_hooks) == pre and dict(module._forward_hooks) == post
        assert vars(module).get("forward") is forward
    assert core_layout() is layout and not torch.cuda.is_initialized()


def test_actual_original_composition_is_content_bound_without_unpatch_or_kernel_claim(request, tmp_path):
    selected = original_sol(request, tmp_path)
    before = snapshot(selected)
    contract, delegates = inspect_original_sol(selected)
    independent = original_sol(request, tmp_path)
    assert inspect_original_sol(independent)[0] == contract
    assert contract["python_composition_verified"]
    # This fixture's direct CPU Sage is deliberately not an actual quantized
    # calculation. Structural composition must not certify its raw delegate.
    assert not contract["unwrapped_attention_delegates_authenticated"]
    assert all(not item["calculation_content_verified"] for item in contract["raw_KJ_calculation_functions"].values())
    assert not contract["portable_cache_reuse"] and not contract["kernel_dispatch_binary_and_layout_cache_verified"]
    assert len(delegates) == len(selected.model.diffusion_model.blocks)
    assert contract["settings"]["tau"] == 1.3
    assert inspect_original_sol(original_sol(request, tmp_path, tau=1.4))[0]["sha256"] != contract["sha256"]
    # Core installing selected methods must not change structural identity.
    for path, method in selected.object_patches.items():
        if not path.endswith(".forward"):
            continue
        owner = selected.get_model_object(path[:-8])
        owner.forward = method
    installed = snapshot(selected)
    assert inspect_original_sol(selected)[0] == contract
    loaded = loaded_model_identity(selected)
    assert loaded["original_sol_structure"] == contract and not loaded["portable_cache_reuse"]
    assert_untouched(selected, installed)
    for module, _, _, forward in before[2]:
        if forward is None:
            vars(module).pop("forward", None)
        else:
            module.forward = forward
    assert_untouched(selected, before)


def test_actual_depth_gate_hook_order_and_coordinates_are_bound(request, tmp_path):
    selected = original_sol(request, tmp_path, dense_blocks="0")
    before = snapshot(selected)
    contract, _ = inspect_original_sol(selected)
    assert contract["settings"]["dense_blocks"] == [0]
    assert [item["role"] for item in contract["ordered_block_hooks"][0]] == ["morton", "index", "compose", "restore"]
    assert_untouched(selected, before)


def test_unknown_hook_and_changed_layout_constructor_are_not_authenticated_or_removed(request, tmp_path, monkeypatch):
    selected = original_sol(request, tmp_path)
    owner = selected.model.diffusion_model.blocks[0]
    handle = owner.register_forward_pre_hook(lambda module, args: None)
    before = snapshot(selected)
    loaded = loaded_model_identity(selected)
    assert not loaded["portable_cache_reuse"] and not loaded["original_sol_structure"]["python_composition_verified"]
    assert_untouched(selected, before)
    handle.remove()
    from comfy.ldm.minimax.model import PackedLayout
    constructor = PackedLayout.__init__
    def foreign(*args, **kwargs):
        raise AssertionError("inspection must not execute a foreign constructor")
    monkeypatch.setattr(PackedLayout, "__init__", foreign)
    changed = snapshot(selected)
    loaded = loaded_model_identity(selected)
    assert not loaded["portable_cache_reuse"] and not loaded["original_sol_structure"]["python_composition_verified"]
    assert_untouched(selected, changed)
    monkeypatch.setattr(PackedLayout, "__init__", constructor)
