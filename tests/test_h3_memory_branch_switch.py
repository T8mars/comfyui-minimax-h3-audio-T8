"""A selected bare branch must not inherit another active clone's methods."""
import torch

from h3_audio_t8_pkg.h3_memory_advanced import configure_chunk_feed_forward, configure_low_vram_attention
from test_h3_memory_advanced import _small_model


def test_configuring_bare_ffn_while_other_clone_active_does_not_nest_token_chunks():
    bare = _small_model(1)
    mlp = bare.model.diffusion_model.blocks[0].mlp
    values = torch.randn(1024, 24)
    expected = mlp(values)
    active, _ = configure_chunk_feed_forward(bare, 2, 256)
    active.patch_model(load_weights=False)
    calls = []
    hook = mlp.fc1.register_forward_hook(lambda _module, args, _output: calls.append(len(args[0])))
    try:
        selected, _ = configure_chunk_feed_forward(bare, 2, 256)
        method = selected.object_patches['diffusion_model.blocks.0.mlp.forward']
        result = method(values)
        assert result.shape == (1024, 24)
        assert calls == [512, 512]
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    finally:
        hook.remove()
        active.unpatch_model(unpatch_weights=False)


def test_configuring_bare_attention_while_other_clone_active_installs_its_own_grouping():
    bare = _small_model(2)
    values = torch.randn(17, 24)
    expected = bare.model.diffusion_model.blocks[0].attn(values)
    active, _ = configure_low_vram_attention(bare, 2)
    active.patch_model(load_weights=False)
    try:
        selected, _ = configure_low_vram_attention(bare, 3)
        paths = [name for name in selected.object_patches if name.endswith('.attn.forward')]
        assert len(paths) == 2
        assert all(selected.object_patches[name].__func__._t8_h3_head_chunks == 3 for name in paths)
    finally:
        active.unpatch_model(unpatch_weights=False)
    selected.patch_model(load_weights=False)
    try:
        actual = bare.model.diffusion_model.blocks[0].attn(values)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    finally:
        selected.unpatch_model(unpatch_weights=False)
