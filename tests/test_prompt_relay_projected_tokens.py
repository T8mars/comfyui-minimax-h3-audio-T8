"""Selected actual Core raw tokenizer contracts; no projection/model weights.

The CLIP public wrapper and conditioning are explicit fixtures, independent of
the optional ClipProj package. Actual provider token tests live separately.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from comfy.text_encoders.boogu import BooguTokenizer
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer

from h3_audio_t8_pkg import prompt_relay_advanced as relay


def _fixture(family, local="她抬手 🐇\n她转身 café", *, references=False, scheduled=False):
    outer = BooguTokenizer() if family == "qwen3vl_8b" else Qwen3VLTokenizer(model_type=family)
    hf = getattr(outer, family).tokenizer

    class RawPublicCLIPFixture:
        tokenizer = outer

        @staticmethod
        def tokenize(text):
            return {family: [[(token, 1.) for token in hf.encode(text, add_special_tokens=False)]]}

    clip = RawPublicCLIPFixture()
    plan, text, *_ = relay.build_prompt_relay_plan("One continuous scene.", local,
        124, "auto_equal", "", "paper_v1", .1, False, False)
    tokens = clip.tokenize(text)
    prefix = [(999999, 1.), ({"type": "image", "data": torch.zeros(1, 16, 16, 3)}, 1.)] if references else []
    tokens[family][0][:0] = prefix
    count = len(tokens[family][0])
    condition = [[torch.zeros(1, count, 5120),
        {"minimax_token_tags": torch.tensor([0] * len(prefix) + [1] * (count - len(prefix)))}]]
    if scheduled:
        condition.append(deepcopy(condition[0]))
    return clip, plan, text, tokens, condition


def _unchanged(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _unchanged(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for item, expected in zip(left, right):
            _unchanged(item, expected)
    else:
        assert left == right


@pytest.mark.parametrize("family", ("qwen3vl_4b", "qwen3vl_8b"))
@pytest.mark.parametrize("references", (False, True))
@pytest.mark.parametrize("local", ("她抬手 🐇\n她转身 café", "<d>你好</d>\n<d>[Korean] 안녕</d>"))
def test_actual_selected_raw_tokenizer_unicode_and_reference_tail(family, references, local):
    clip, plan, text, tokens, condition = _fixture(family, local, references=references)
    before = deepcopy((tokens, condition))
    binding = relay.build_prompt_relay_binding(clip, plan, text, condition, tokens)
    assert binding["tokenizer_route"] == {"key": family,
        "verification": "connected_public_equals_selected_lossless_raw"}
    assert len(binding["events"]) == 2
    assert binding["events"][0]["text_key_end"] <= binding["events"][1]["text_key_start"]
    assert binding["binding_hash"] == relay._sha256_json({k: v for k, v in binding.items() if k != "binding_hash"})
    _unchanged((tokens, condition), before)


@pytest.mark.parametrize("family", ("qwen3vl_4b", "qwen3vl_8b"))
def test_every_projected_scheduled_item_retains_native_shape_and_tags(family):
    clip, plan, text, tokens, condition = _fixture(family, scheduled=True)
    binding = relay.build_prompt_relay_binding(clip, plan, text, condition, tokens)
    assert binding == relay.build_prompt_relay_binding(clip, plan, text, condition[:1], tokens)


FAULTS = ("unknown_key", "multiple_keys", "multiple_batches", "wrong_tail", "unprojected", "wrong_batch",
    "missing_tags", "visual_prompt_overlap", "public_family_changed", "public_chat_template",
    "public_suffix", "public_ids_changed", "hidden_hf", "unverifiable_hf_decoder", "public_boolean_ids",
    "schedule_unprojected", "schedule_lost_tags", "schedule_length", "schedule_visual_prompt_overlap")


@pytest.mark.parametrize("family", ("qwen3vl_4b", "qwen3vl_8b"))
@pytest.mark.parametrize("fault", FAULTS)
def test_wrong_selected_public_tokens_projection_or_scheduled_provenance_refuses(family, fault):
    clip, plan, text, tokens, condition = _fixture(family, scheduled=True)
    if fault == "unknown_key":
        tokens = {"unknown": tokens[family]}
    elif fault == "multiple_keys":
        tokens["qwen3vl_32b"] = tokens[family]
    elif fault == "multiple_batches":
        tokens[family].append(tokens[family][0])
    elif fault == "wrong_tail":
        tokens[family][0][-1] = (1, 1.)
    elif fault == "unprojected":
        condition[0][0] = condition[0][0][:, :, :4096]
    elif fault == "wrong_batch":
        condition[0][0] = condition[0][0].repeat(2, 1, 1)
    elif fault == "missing_tags":
        condition[0][1].clear()
    elif fault == "visual_prompt_overlap":
        condition[0][1]["minimax_token_tags"][-1] = 0
    elif fault == "hidden_hf":
        clip.tokenizer = SimpleNamespace()
    elif fault == "unverifiable_hf_decoder":
        hf = getattr(clip.tokenizer, family).tokenizer
        clip.tokenizer = SimpleNamespace(**{family: SimpleNamespace(tokenizer=SimpleNamespace(
            encode=hf.encode, convert_ids_to_tokens=hf.convert_ids_to_tokens))})
    elif fault == "public_chat_template":
        clip.tokenize = clip.tokenizer.tokenize_with_weights
    elif fault.startswith("public_"):
        tokenize = clip.tokenize

        def wrong(prompt):
            result = tokenize(prompt)
            if fault == "public_family_changed":
                return {"qwen3vl_32b": result[family]}
            if fault == "public_suffix":
                result[family][0].append((1, 1.))
            else:
                result[family][0][-1] = (True if fault == "public_boolean_ids" else 1, 1.)
            return result

        clip.tokenize = wrong
    elif fault == "schedule_unprojected":
        condition[1][0] = condition[1][0][:, :, :4096]
    elif fault == "schedule_lost_tags":
        condition[1][1].clear()
    elif fault == "schedule_length":
        condition[1][0] = condition[1][0][:, :-1]
        condition[1][1]["minimax_token_tags"] = condition[1][1]["minimax_token_tags"][:-1]
    elif fault == "schedule_visual_prompt_overlap":
        condition[1][1]["minimax_token_tags"][-1] = 0
    else:
        raise AssertionError("Test fault inventory incomplete")
    with pytest.raises(RuntimeError):
        relay.build_prompt_relay_binding(clip, plan, text, condition, tokens)
