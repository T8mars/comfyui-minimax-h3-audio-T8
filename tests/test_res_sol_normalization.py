"""Actual source-backed Sol projection; no CUDA math or recovery approval."""
from types import MethodType

import torch
from comfy.ldm.modules import attention
from comfy.weight_adapter.lora import LoRAAdapter

from h3_audio_t8_pkg import h3_memory_advanced as t8
from h3_audio_t8_pkg import sol_attn_minimax_v2 as sol
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from h3_audio_t8_pkg.res_sol_normalization import normalized_original_sol_identity
from test_res_model_identity import model, prepared, source_latent
from test_res_original_sol_identity import snapshot, assert_untouched, original_sol
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- actual code/ops/JIT fixture
from test_relay_kj_memory import memory_nodes  # noqa: F401 -- opaque negative fixture
from test_relay_kj_backend import kj  # noqa: F401 -- dependent fixture


def selected(module, tmp_path, *, tau=1.3, chunks=2, dense_blocks="", lora=False):
    branch = model()
    if lora:
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        rows, columns = branch.model_state_dict()[key].shape
        adapter = LoRAAdapter({"up", "down"}, (
            torch.ones(rows, 1) * .01, torch.ones(1, columns) * .02, 1., None, None, None))
        assert branch.add_patches({key: adapter}, .25) == [key]
    branch = t8.configure_chunk_feed_forward(branch, chunks, 256)[0]
    for index, block in enumerate(branch.model.diffusion_model.blocks):
        branch.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward",
                                MethodType(module.minimax_sageattn_forward, block.attn))
    branch.set_model_optimized_attention(attention.get_attention_function("pytorch", None))
    branch = prepared(branch, source_latent(), tmp_path, mode="disabled")[0]
    return sol.SolAttnMiniMax.execute(branch, tau=tau, start_percent=.2, end_percent=.9,
        min_tokens=12288, sink_conditioning="exact_kv_and_rows", morton=False,
        morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
        reuse_qkv_memory=False, verbose=False, dense_blocks=dense_blocks).result[0]


def test_actual_source_chain_readonly_weights_projection_is_equal_before_and_after_install(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path, dense_blocks="0")
    before = snapshot(branch)
    first = loaded_model_identity(branch)
    assert first["automatic_loaded_weights_verified"]
    assert first["original_sol_normalization"]["readonly_normalization_verified"]
    assert first["weights"]["mixed_memory"]["settings"] == {"chunks": 2, "seq_threshold": 256}
    assert first["original_sol_structure"]["unwrapped_attention_delegates_authenticated"]
    assert not first["portable_cache_reuse"]
    assert loaded_model_identity(selected(module, tmp_path, dense_blocks="0")) == first
    assert_untouched(branch, before)
    for path, method in branch.object_patches.items():
        if path.endswith(".forward"):
            branch.get_model_object(path[:-8]).forward = method
    installed = snapshot(branch)
    assert loaded_model_identity(branch) == first
    assert_untouched(branch, installed)


def test_actual_raw_weights_LoRA_order_and_Sol_FFN_settings_are_bound(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path, lora=True)
    first = loaded_model_identity(branch)
    assert first["automatic_loaded_weights_verified"] and first["weights"]["lora_target_count"] == 1
    assert loaded_model_identity(selected(module, tmp_path, lora=True, tau=1.4)) != first
    assert loaded_model_identity(selected(module, tmp_path, lora=True, chunks=3)) != first
    before = snapshot(branch)
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    branch.patches[key][0][1].weights[0][0, 0] += .01
    changed = loaded_model_identity(branch)
    assert changed != first and changed["automatic_loaded_weights_verified"]
    branch.patches[key][0] = (.75, *branch.patches[key][0][1:])
    strength = loaded_model_identity(branch)
    assert strength != changed
    assert branch.add_patches({key: ("diff", (torch.ones_like(branch.model_state_dict()[key]) * .001,))}, .5) == [key]
    ordered = loaded_model_identity(branch)
    branch.patches[key].reverse()
    reversed_order = loaded_model_identity(branch)
    assert reversed_order != ordered
    with torch.no_grad():
        next(branch.model.parameters()).reshape(-1)[0].add_(.001)
    assert loaded_model_identity(branch) != reversed_order
    assert_untouched(branch, before)


def test_unknown_calculation_hook_and_invocation_owner_are_never_removed(request, actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path)
    assert loaded_model_identity(branch)["automatic_loaded_weights_verified"]
    block = branch.model.diffusion_model.blocks[0]
    handle = block.register_forward_pre_hook(lambda owner, args: None)
    before = snapshot(branch)
    assert not loaded_model_identity(branch)["portable_cache_reuse"]
    assert not loaded_model_identity(branch)["automatic_loaded_weights_verified"]
    assert_untouched(branch, before)
    handle.remove()
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("inspection must not invoke unknown calculation")
    with monkeypatch.context() as changed:
        changed.setattr(module, "_per_thread_int8_i64", foreign)
        before = snapshot(branch)
        loaded = loaded_model_identity(branch)
        assert not loaded["automatic_loaded_weights_verified"]
        assert not loaded["original_sol_normalization"]["readonly_normalization_verified"]
        assert_untouched(branch, before)
    with monkeypatch.context() as changed:
        changed.setitem(block._forward_pre_hooks_with_kwargs, -12345, True)
        loaded = loaded_model_identity(branch)
        assert not loaded["automatic_loaded_weights_verified"]
        assert "invocation flags" in loaded["original_sol_normalization"]["reason"]
        assert -12345 in block._forward_pre_hooks_with_kwargs
    # The old ordinary-CPU Sage fixture intentionally installs a fake package.
    # Request it only after the actual native binary baseline/negatives; a fake
    # package must not falsely invalidate that real baseline or receive approval.
    opaque = original_sol(request, tmp_path)
    before = snapshot(opaque)
    weight, _, report = normalized_original_sol_identity(opaque.clone())
    assert weight is None and not report["readonly_normalization_verified"]
    assert "raw calculation" in report["reason"]
    assert_untouched(opaque, before)
    assert not calls and not torch.cuda.is_initialized()


def test_unknown_weight_owner_survives_strict_projection_validation(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path)
    assert loaded_model_identity(branch)["automatic_loaded_weights_verified"]
    branch.set_attachments("private_unknown_owner", object())
    before = snapshot(branch)
    loaded = loaded_model_identity(branch)
    assert not loaded["portable_cache_reuse"] and not loaded["automatic_loaded_weights_verified"]
    assert not loaded["original_sol_normalization"]["readonly_normalization_verified"]
    assert branch.get_attachment("private_unknown_owner") is not None
    assert_untouched(branch, before)
