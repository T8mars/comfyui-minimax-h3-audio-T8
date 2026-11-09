"""Prefix KV identity is separate from AV Stage identity; no CUDA/model download."""
import pytest
import torch

from h3_audio_t8_pkg import qwen_prefix_cache_advanced as cache
from test_qwen_prefix_cache_advanced import _NativeLookingClip


@pytest.mark.parametrize('change', ['tokenizer', 'encoder', 'device', 'dtype', 'weight_update'])
def test_same_wrapper_rekeys_execution_inputs_even_with_equal_patch_uuid(change):
    clip = _NativeLookingClip()
    clip.tokenizer = object()
    clip.patcher.load_device = torch.device('cpu')
    clip.cond_stage_model.qwen3vl_32b.transformer.model = torch.nn.Linear(2, 2)
    clip.cond_stage_model.dtypes = {torch.float32}
    first = cache._model_identity(clip, {'hashes': {}})
    encoder = clip.cond_stage_model.qwen3vl_32b.transformer
    if change == 'tokenizer':
        clip.tokenizer = object()
    elif change == 'encoder':
        clip.cond_stage_model.qwen3vl_32b.transformer = type(encoder)()
    elif change == 'device':
        clip.patcher.load_device = torch.device('cpu:1')  # Identity only; not a GPU run.
    elif change == 'dtype':
        encoder.model.to(dtype=torch.float64)
        clip.cond_stage_model.dtypes = {torch.float64}
    else:
        with torch.no_grad():
            encoder.model.weight.add_(.1)
    assert clip.patcher.patches_uuid == 'base'
    assert first != cache._model_identity(clip, {'hashes': {}})


def test_reference_bytes_and_order_rekey_but_prompt_text_reuses_prefix():
    clip = _NativeLookingClip()
    wrapped = cache.MiniMaxH3CachedClip(clip, cache.QwenReferencePrefixCache(4, 64), {'hashes': {}})
    a, b = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)
    initial = wrapped.tokenize('first prompt', images=[a, b])
    assert wrapped.tokenize('second prompt', images=[a, b]).prefix_fingerprint == initial.prefix_fingerprint
    assert wrapped.tokenize('first prompt', images=[b, a]).prefix_fingerprint != initial.prefix_fingerprint
    a.add_(.125)
    assert wrapped.tokenize('first prompt', images=[a, b]).prefix_fingerprint != initial.prefix_fingerprint


@pytest.mark.parametrize('relocate', [False, True])
def test_actual_core_prefix_and_suffix_rebuild_after_encoder_weight_update(relocate):
    """Real Core KV/layers; tiny deterministic visual embeddings are a fixture."""
    import comfy.ops
    from comfy.text_encoders.llama import Llama2_, Llama2Config
    config = Llama2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64)
    config.head_dim, config.qkv_bias = 8, True
    llama = Llama2_(config, device=torch.device('cpu'), dtype=torch.float32, ops=comfy.ops.disable_weight_init)
    rng = torch.Generator().manual_seed(710)
    with torch.no_grad():
        for parameter in llama.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=rng) * .03)
    clip = _NativeLookingClip()
    clip.patcher.load_device = torch.device('cpu')
    clip.tokenizer = object()
    encoder = clip.cond_stage_model.qwen3vl_32b
    encoder.transformer.model = llama
    encoder.execution_device = torch.device('cpu')
    embedding_table = torch.randn(128, 16, generator=rng)

    def process(rows, device):
        vectors, info = [], []
        for token in rows[0]:
            if isinstance(token, dict):
                index = len(vectors)
                vectors.extend([embedding_table[0], embedding_table[1] + token['image'].mean(), embedding_table[2]])
                info.append({'type': 'image', 'index': index+1, 'size': 1})
            else:
                vectors.append(embedding_table[token])
        embeds = torch.stack(vectors).unsqueeze(0).to(device)
        return embeds, torch.ones(1, len(vectors), device=device), len(vectors), info

    encoder.process_tokens = process
    encoder.transformer.build_image_inputs = lambda embeds, info: (
        torch.arange(embeds.shape[1]).unsqueeze(0), torch.zeros(1, embeds.shape[1], dtype=torch.bool), [])
    def load_model(*args):
        if relocate:
            # Core low-VRAM loading replaces parameter objects without a
            # numerical update. Use real Core Llama layers and actual tensors.
            for module in llama.modules():
                for name, value in list(module._parameters.items()):
                    if value is not None:
                        module.register_parameter(name, torch.nn.Parameter(value.detach().clone(), requires_grad=False))
    clip.load_model = load_model
    clip.cond_stage_model.reset_clip_options = lambda: None
    clip.cond_stage_model.set_clip_options = lambda options: setattr(encoder, 'execution_device', options['execution_device'])
    clip.add_hooks_to_dict = lambda *_: None
    store = cache.QwenReferencePrefixCache(5, 64)
    wrapped = cache.MiniMaxH3CachedClip(clip, store, {'hashes': {}})
    tokens = wrapped.tokenize('abc', images=[torch.ones(1, 2, 2, 3)])
    first = wrapped.encode_from_tokens_scheduled(tokens)[0][0]
    same = wrapped.encode_from_tokens_scheduled(tokens)[0][0]
    torch.testing.assert_close(first, same, rtol=0, atol=0)
    assert (store.misses, store.hits) == (1, 1)
    other = wrapped.tokenize('abc', images=[torch.zeros(1, 2, 2, 3)])
    wrapped.encode_from_tokens_scheduled(other)
    restored = wrapped.encode_from_tokens_scheduled(tokens)[0][0]
    torch.testing.assert_close(first, restored, rtol=0, atol=0)
    assert (store.misses, store.hits) == (2, 2)
    # No new patch UUID/model ID/tokenization: the existing wrapper must see
    # a real in-place encoder update before consulting the old prefix entry.
    with torch.no_grad():
        llama.layers[0].self_attn.v_proj.weight.add_(.1)
    changed = wrapped.encode_from_tokens_scheduled(tokens)[0][0]
    assert (store.misses, store.hits) == (3, 2)
    assert not torch.equal(first, changed)
    fresh = cache.MiniMaxH3CachedClip(clip, cache.QwenReferencePrefixCache(5, 64), {'hashes': {}})
    expected = fresh.encode_from_tokens_scheduled(tokens)[0][0]
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
