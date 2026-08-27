from __future__ import annotations

import json

from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg.qwen_prefix_cache_advanced as prefix_cache
from h3_audio_t8_pkg.nodes_qwen_prefix_cache_advanced import (
    MiniMaxH3QwenPrefixCacheStatsT8Advanced,
    MiniMaxH3QwenReferencePrefixCacheT8Advanced,
)
from h3_audio_t8_pkg.qwen_prefix_cache_advanced import (
    H3PrefixTokens,
    PrefixCacheEntry,
    QWEN_PREFIX_CACHE_SCHEMA,
    MiniMaxH3CachedClip,
    QwenReferencePrefixCache,
    build_prefix_cache_clip,
    encode_suffix_from_entry,
    prefix_split_matches_full,
)


def _entry(key: str, size: int = 32) -> PrefixCacheEntry:
    scalar = torch.zeros(1)
    return PrefixCacheEntry(
        key=key,
        hidden=scalar,
        key_values=[(scalar, scalar, 1)],
        next_position=torch.ones(1, dtype=torch.long),
        token_tags=torch.ones(1, dtype=torch.long),
        bytes=size,
        prefix_tokens=1,
        created_unix=0.0,
    )


class _NativeLookingClip:
    def __init__(self):
        transformer_type = type("MiniMaxQwen3VL", (), {})
        self.cond_stage_model = SimpleNamespace(
            qwen3vl_32b=SimpleNamespace(transformer=transformer_type())
        )
        self.patcher = SimpleNamespace(model=object(), patches_uuid="base")
        self.use_clip_schedule = False

    @staticmethod
    def _row(text, kwargs):
        row = []
        images = list(kwargs.get("images") or [])
        refs = list(kwargs.get("minimax_ref_items") or [])
        for image in images:
            row.append(({"type": "image", "image": image}, 1.0))
        for item in refs:
            if item.get("kind") in {"image", "video"}:
                row.append(({"type": "image", "image": item["value"]}, 1.0))
            else:
                row.append((41, 1.0))
        row.extend((ord(character) % 127, 1.0) for character in text)
        return {"qwen3vl_32b": [row]}

    def tokenize(self, text, return_word_ids=False, **kwargs):
        del return_word_ids
        return self._row(text, kwargs)

    def clone(self, disable_dynamic=False):
        del disable_dynamic
        return self


def test_current_comfy_qwen_contract_is_known():
    contract = prefix_cache.core_contract()
    assert contract["supported"] is True
    assert set(contract["hashes"]) == {
        "minimax_tokenizer",
        "minimax_clip_encode",
        "minimax_qwen_forward",
        "qwen_image_inputs",
        "llama_forward",
        "attention_forward",
        "transformer_block_forward",
    }
    assert contract["source_hash_policy"] == "diagnostic_only_not_a_compatibility_gate"
    assert contract["semantic_contract"]["status"] == "semantic_contract_validated"


def test_equivalent_qwen_source_text_change_is_not_a_compatibility_gate(monkeypatch):
    monkeypatch.setattr(prefix_cache, "_source_hash", lambda _value: "unknown-source")
    contract = prefix_cache.core_contract()
    assert contract["supported"] is True
    assert set(contract["hashes"].values()) == {"unknown-source"}


def test_prefix_split_requires_exact_full_token_sequence_including_visual_entries():
    image = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    visual = ({"type": "image", "image": image}, 1.0)
    full = {"qwen3vl_32b": [[(10, 1.0), visual, (20, 1.0), (30, 1.0)]]}
    prefix = {"qwen3vl_32b": [[(10, 1.0), visual]]}
    suffix = {"qwen3vl_32b": [[(20, 1.0), (30, 1.0)]]}
    assert prefix_split_matches_full(full, prefix, suffix) is True
    duplicated_template = {"qwen3vl_32b": [[(999, 1.0), (20, 1.0), (30, 1.0)]]}
    assert prefix_split_matches_full(full, prefix, duplicated_template) is False

    class _BadTemplateClip(_NativeLookingClip):
        def tokenize(self, text, return_word_ids=False, **kwargs):
            result = super().tokenize(text, return_word_ids, **kwargs)
            if text and not kwargs.get("images") and not kwargs.get("minimax_ref_items"):
                result["qwen3vl_32b"][0].insert(0, (999, 1.0))
            return result

    wrapped = MiniMaxH3CachedClip(_BadTemplateClip(), QwenReferencePrefixCache(1, 64), {"hashes": {}})
    tokens = wrapped.tokenize("prompt", images=[image])
    assert tokens.prefix_tokens is None
    assert tokens.prefix_fingerprint is None


def test_visual_prefix_fingerprint_is_prompt_independent_and_reference_sensitive():
    base = _NativeLookingClip()
    cache = QwenReferencePrefixCache(2, 64)
    wrapped = MiniMaxH3CachedClip(base, cache, {"hashes": {"probe": "ok"}})
    image_a = torch.zeros(1, 2, 2, 3)
    image_b = torch.ones(1, 2, 2, 3)
    first = wrapped.tokenize("prompt A", images=[image_a])
    second = wrapped.tokenize("prompt B", images=[image_a])
    changed = wrapped.tokenize("prompt A", images=[image_b])
    assert isinstance(first, H3PrefixTokens)
    assert first.prefix_fingerprint == second.prefix_fingerprint
    assert first.prefix_fingerprint != changed.prefix_fingerprint
    assert first.prefix_tokens != first.suffix_tokens


def test_audio_only_reference_bypasses_visual_prefix_cache():
    base = _NativeLookingClip()
    wrapped = MiniMaxH3CachedClip(
        base,
        QwenReferencePrefixCache(1, 64),
        {"hashes": {"probe": "ok"}},
    )
    tokens = wrapped.tokenize(
        "speech",
        minimax_ref_items=[{"kind": "audio", "value": torch.zeros(1)}],
    )
    assert tokens.prefix_tokens is None
    assert tokens.prefix_fingerprint is None
    assert dict(tokens) == base.tokenize(
        "speech",
        minimax_ref_items=[{"kind": "audio", "value": torch.zeros(1)}],
    )


def test_bounded_lru_evicts_oldest_and_rejects_oversize():
    cache = QwenReferencePrefixCache(2, 64)
    assert cache.put(_entry("a")) is True
    assert cache.put(_entry("b")) is True
    assert cache.get("a").key == "a"
    assert cache.put(_entry("c")) is True
    assert list(cache.entries) == ["a", "c"]
    assert cache.get("missing") is None
    assert cache.put(_entry("huge", size=65 * 1024**2)) is False
    report = cache.report()
    assert report["hits"] == 1
    assert report["misses"] == 1
    assert report["rejected_oversize"] == 1
    assert report["storage"] == "bounded_cpu_memory_lru"
    assert report["disk_writes"] is False
    assert report["memory_safe_claim"] is False


def test_report_only_is_identity_and_memory_mode_wraps_without_mutation(monkeypatch):
    monkeypatch.setattr(
        prefix_cache,
        "core_contract",
        lambda: {"supported": True, "hashes": {"probe": "ok"}},
    )
    base = _NativeLookingClip()
    returned, cache, report = build_prefix_cache_clip(base, "report_only", 1, 64, 0)
    assert returned is base
    assert report["status"] == "report_only"
    assert report["applied"] is False
    wrapped, wrapped_cache, report = build_prefix_cache_clip(
        base, "memory_lru_exp", 1, 64, 1
    )
    assert isinstance(wrapped, MiniMaxH3CachedClip)
    assert wrapped._clip is base
    assert wrapped_cache is not cache
    assert report["status"] == "applied_exp"
    assert report["applied"] is True


def test_unknown_core_fails_closed_only_when_apply_is_requested(monkeypatch):
    monkeypatch.setattr(
        prefix_cache,
        "core_contract",
        lambda: {"supported": False, "hashes": {"probe": "unknown"}},
    )
    base = object()
    returned, _cache, report = build_prefix_cache_clip(base, "report_only", 1, 64, 0)
    assert returned is base
    assert report["status"] == "unsupported_core"
    with pytest.raises(RuntimeError, match="unknown ComfyUI source contract"):
        build_prefix_cache_clip(base, "memory_lru_exp", 1, 64, 0)


def test_real_comfy_llama_prefix_kv_matches_full_causal_forward_on_cpu():
    import comfy.ops
    from comfy.text_encoders.llama import Llama2_, Llama2Config

    config = Llama2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.head_dim = 8
    config.qkv_bias = True
    model = Llama2_(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        ops=comfy.ops.disable_weight_init,
    )
    generator = torch.Generator().manual_seed(91)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)

    prefix = torch.randn(1, 5, 16, generator=generator)
    suffix = torch.randn(1, 4, 16, generator=generator)
    full_embeds = torch.cat((prefix, suffix), dim=1)
    with torch.no_grad():
        full_output = model.forward(
            None,
            embeds=full_embeds,
            position_ids=torch.arange(9).unsqueeze(0),
            dtype=torch.float32,
        )[0]

    past = [
        (
            torch.empty(1, config.num_key_value_heads, 5, config.head_dim),
            torch.empty(1, config.num_key_value_heads, 5, config.head_dim),
            0,
        )
        for _ in model.layers
    ]
    with torch.no_grad():
        prefix_output = model.forward(
            None,
            embeds=prefix,
            position_ids=torch.arange(5).unsqueeze(0),
            past_key_values=past,
            dtype=torch.float32,
        )
    entry = PrefixCacheEntry(
        key="tiny",
        hidden=prefix_output[0],
        key_values=prefix_output[2],
        next_position=torch.tensor([5]),
        token_tags=torch.zeros(5, dtype=torch.long),
        bytes=1,
        prefix_tokens=5,
        created_unix=0.0,
    )

    class _TinyClipModel:
        execution_device = torch.device("cpu")
        transformer = SimpleNamespace(model=model)

        @staticmethod
        def process_tokens(token_rows, device):
            assert device == torch.device("cpu")
            assert token_rows == [[11, 12, 13, 14]]
            return suffix, torch.ones(1, 4), 4, []

    suffix_tokens = {
        "qwen3vl_32b": [[(11, 1.0), (12, 1.0), (13, 1.0), (14, 1.0)]]
    }
    cached_output, tags = encode_suffix_from_entry(
        _TinyClipModel(), suffix_tokens, entry
    )
    torch.testing.assert_close(cached_output, full_output.float(), rtol=2e-6, atol=2e-6)
    assert tags.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_node_schemas_are_explicitly_experimental_and_report_only_by_default():
    main = MiniMaxH3QwenReferencePrefixCacheT8Advanced.define_schema()
    stats = MiniMaxH3QwenPrefixCacheStatsT8Advanced.define_schema()
    assert main.node_id == "MiniMaxH3QwenReferencePrefixCacheT8Advanced"
    assert stats.node_id == "MiniMaxH3QwenPrefixCacheStatsT8Advanced"
    assert main.category.endswith("/Experimental")
    assert main.is_experimental is True
    assert stats.is_output_node is True
    mode = next(value for value in main.inputs if value.id == "mode")
    assert mode.default == "report_only"
    cache = QwenReferencePrefixCache(1, 64)
    assert cache.report()["schema"] == QWEN_PREFIX_CACHE_SCHEMA
    assert cache.report()["bit_exact_claim"] is False
    output = MiniMaxH3QwenPrefixCacheStatsT8Advanced.execute(cache)
    assert json.loads(output[0])["schema"] == QWEN_PREFIX_CACHE_SCHEMA
    assert json.loads(output.ui["text"][0])["schema"] == QWEN_PREFIX_CACHE_SCHEMA
