from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.prompt_relay_token_bytes import token_byte_offsets, supports_byte_tokens


@pytest.fixture(scope="module")
def pair():
    from transformers import Qwen2TokenizerFast
    import comfy.text_encoders.qwen3vl as qwen
    slow = MiniMaxH3Tokenizer()
    fast = Qwen2TokenizerFast.from_pretrained(
        str(Path(qwen.__file__).parent / "qwen25_tokenizer"), local_files_only=True)
    native_hf = slow.qwen3vl_32b.tokenizer
    extras = getattr(native_hf, "additional_special_tokens", None)
    if extras is None:
        extras = native_hf.extra_special_tokens
    assert isinstance(extras, list) and extras
    fast.add_special_tokens({"additional_special_tokens": extras})
    assert {str(token): fast.convert_tokens_to_ids(str(token)) for token in extras} == {
        str(token): native_hf.convert_tokens_to_ids(str(token)) for token in extras}
    assert not hasattr(fast, "byte_decoder")
    return slow, fast


@pytest.mark.parametrize("text", [
    "夜晚的女人抬手，然后快速转身", "<d>[Korean] 아침 햇살 문을 열면</d>",
    "<Picture 1> 🐇 café é\n\t repeated  spaces", "<d>中文</d> <scenetrans>。",
    "a\r\nb\t👩‍🎤\u00a0𠀀", "<|lyrics_start|>春天<|lyrics_end|>",
])
def test_real_slow_and_fast_ids_and_lossless_offsets(pair, text):
    native, fast = pair
    slow = native.qwen3vl_32b.tokenizer
    ids = slow.encode(text, add_special_tokens=False)
    assert fast.encode(text, add_special_tokens=False) == ids
    assert supports_byte_tokens(fast)
    a, b = token_byte_offsets(text, ids, slow), token_byte_offsets(text, ids, fast)
    assert a == b
    assert a[0][0] == 0 and a[-1][1] == len(text.encode("utf-8"))


def test_native_binding_with_fast_backend_and_media_prefix(pair):
    native, fast = pair
    # Swap only the tokenizer implementation; native SD wrapper remains authority.
    import copy
    outer = copy.copy(native)
    outer.qwen3vl_32b = copy.copy(native.qwen3vl_32b)
    outer.qwen3vl_32b.tokenizer = fast
    plan = relay.build_prompt_relay_plan("静态场景", "<d>你好</d>\n그녀가 손을 든다 🐇",
        73, "auto_equal", "", "paper_v1", .1, False, False)[0]
    text = plan["compiled_prompt"]
    clip = SimpleNamespace(tokenizer=outer, tokenize=outer.tokenize_with_weights)
    tokens = outer.tokenize_with_weights(text)
    entries = tokens["qwen3vl_32b"][0]
    prefix = [(999, 1.), ({"type": "image"}, 1.)]
    tokens = {"qwen3vl_32b": [prefix + entries]}
    tags = torch.tensor([0, 0] + [1] * len(entries))
    condition = [[torch.zeros(1, len(tags), 5120), {"minimax_token_tags": tags}]]
    bound = relay.build_prompt_relay_binding(clip, plan, text, condition, tokens)
    assert len(bound["events"]) == 2
    assert bound["prompt_token_count"] == len(entries)
    broken = {"qwen3vl_32b": [prefix + [(1, 1.)] + entries[1:]]}
    with pytest.raises(RuntimeError):
        relay.build_prompt_relay_binding(clip, plan, text, condition, broken)


def test_unknown_backend_and_changed_text_are_rejected(pair):
    _, fast = pair
    assert not supports_byte_tokens(SimpleNamespace(convert_ids_to_tokens=lambda _: "a"))
    from tokenizers import decoders
    bad = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=decoders.WordPiece()),
                          convert_ids_to_tokens=lambda _: "a")
    assert not supports_byte_tokens(bad)
    ids = fast.encode("café", add_special_tokens=False)
    with pytest.raises(RuntimeError, match="does not match"):
        token_byte_offsets("cafe", ids, fast)


def test_lossy_unicode_normalization_is_not_silently_routed(pair):
    native, fast = pair
    text = "e" + chr(0x301)
    for tokenizer in (native.qwen3vl_32b.tokenizer, fast):
        ids = tokenizer.encode(text, add_special_tokens=False)
        with pytest.raises(RuntimeError, match="does not match"):
            token_byte_offsets(text, ids, tokenizer)
