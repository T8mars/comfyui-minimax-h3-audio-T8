"""Real installed KJ low-memory and T8 FFN factories; no GPU claims."""
from types import MethodType

import torch

from h3_audio_t8_pkg import h3_memory_advanced as t8
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from test_res_model_identity import model, source_latent, prepared
from test_relay_kj_memory import memory_nodes  # noqa: F401,F811 -- actual installed source fixture
from test_relay_kj_backend import kj  # noqa: F401 -- required by installed source fixture


def mixed(kj_sources, tmp_path, chunks=2):
    lowmem = kj_sources[0]
    bare = model()
    selected = t8.configure_chunk_feed_forward(bare, chunks, 256)[0]
    selected = lowmem.MiniMaxLowVRAMAttention.execute(selected, 2).result[0]
    return bare, prepared(selected, source_latent(), tmp_path, mode="disabled")[0]


def test_actual_mixed_factories_are_portable_content_bound_and_inspection_does_not_unpatch(memory_nodes, tmp_path):  # noqa: F811 -- pytest fixture
    bare, selected = mixed(memory_nodes, tmp_path)
    _, independent = mixed(memory_nodes, tmp_path)
    from h3_audio_t8_pkg.res_memory_identity import mixed_memory_model_identity
    inspection = selected.clone()
    inspection.object_patches.pop("model_sampling")
    assert mixed_memory_model_identity(inspection)["portable_cache_reuse"]
    first = loaded_model_identity(selected)
    assert first["portable_cache_reuse"] and first["automatic_loaded_weights_verified"]
    assert first == loaded_model_identity(independent)
    assert first["weights"]["mixed_memory"]["settings"] == {"chunks": 2, "seq_threshold": 256}
    before = dict(selected.object_patches)
    for path, method in before.items():
        if path.endswith(".forward"):
            method.__self__.forward = method
    try:
        assert loaded_model_identity(selected) == first
        assert selected.object_patches == before
        assert all(method.__self__.forward is method for path, method in before.items() if path.endswith(".forward"))
        _, changed = mixed(memory_nodes, tmp_path, chunks=3)
        assert loaded_model_identity(changed) != first
    finally:
        for path, method in before.items():
            if path.endswith(".forward"):
                method.__self__.__dict__.pop("forward", None)
    assert bare.object_patches == {} and not bare.wrappers and not torch.cuda.is_initialized()


def test_mixed_adapter_retains_unknown_forward_and_hook_without_portable_claim(memory_nodes, tmp_path):  # noqa: F811 -- pytest fixture
    _, selected = mixed(memory_nodes, tmp_path)
    owner = selected.model.diffusion_model.blocks[0].mlp
    original = owner.forward
    def foreign(self, x):
        return original(x)
    owner.forward = MethodType(foreign, owner)
    assert loaded_model_identity(selected)["portable_cache_reuse"] is False
    assert owner.forward.__func__ is foreign
    owner.__dict__.pop("forward")
    handle = owner.register_forward_pre_hook(lambda module, args: None)
    try:
        assert loaded_model_identity(selected)["portable_cache_reuse"] is False
        assert handle.id in owner._forward_pre_hooks
    finally:
        handle.remove()
    assert not torch.cuda.is_initialized()


def test_real_original_Sol_composer_and_hooks_are_retained_without_premature_portable_claim(request, tmp_path):
    from h3_audio_t8_pkg.sol_attn_minimax_v2 import SolAttnMiniMax
    kj_sources = request.getfixturevalue("memory_nodes")
    _, selected = mixed(kj_sources, tmp_path)
    selected = SolAttnMiniMax.execute(selected, tau=1.3, start_percent=.2, end_percent=.9,
        min_tokens=12288, sink_conditioning="exact_kv_and_rows", morton=False,
        morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
        reuse_qkv_memory=False, verbose=False, dense_blocks="").result[0]
    methods = dict(selected.object_patches)
    selector = selected.model_options["transformer_options"]["optimized_attention_override"]
    hooks = [(module, tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
             for module in selected.model.modules()]
    assert any(pre or post for _, pre, post in hooks)
    assert loaded_model_identity(selected)["portable_cache_reuse"] is False
    assert selected.model_options["transformer_options"]["optimized_attention_override"] is selector
    assert selected.object_patches == methods
    for module, pre, post in hooks:
        assert tuple(module._forward_pre_hooks.items()) == pre
        assert tuple(module._forward_hooks.items()) == post
    assert not torch.cuda.is_initialized()
