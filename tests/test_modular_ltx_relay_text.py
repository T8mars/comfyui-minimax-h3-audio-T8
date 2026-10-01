"""Exact selected-CLIP segment binding; no invented LTX token positions."""
import hashlib
import json

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling.ltx_relay_plan import build_ltx_relay_plan
from h3_audio_t8_pkg.modular_sampling.ltx_relay_text import (
    encode_ltx_relay_conditioning,
    tensor_digest,
    validate_ltx_relay_binding,
)
from h3_audio_t8_pkg.modular_sampling.ltx_relay_nodes import MiniMaxH3LTXPromptRelayEncodeEXPT8


class SelectedClip:
    def __init__(self, *, scheduled=False, masked=False):
        self.calls = []
        self.scheduled = scheduled
        self.masked = masked

    def tokenize(self, text):
        self.calls.append(text)
        return text

    def encode_from_tokens_scheduled(self, tokens):
        length = len(tokens.split()) + 2
        rows = []
        for step in range(2 if self.scheduled else 1):
            base = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4)
            meta = {"unprocessed_ltxav_embeds": True,
                    "pooled_output": torch.tensor([step], dtype=torch.float32)}
            if self.scheduled:
                meta["clip_start_percent"] = step * .5
                meta["clip_end_percent"] = (step + 1) * .5
            if self.masked:
                meta["attention_mask"] = torch.ones(1, length, dtype=torch.int64)
            rows.append([base + step + length, meta])
        return rows


def plan():
    return build_ltx_relay_plan(
        {"samples": torch.zeros(1, 128, 15, 2, 2)}, "A continuous scene",
        "A first event\nA second event", "auto_equal", "", 24., .1, False, False,
    )[0]


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_selected_clip_encodes_real_segments_and_binds_exact_spans(scheduled, masked):
    clip = SelectedClip(scheduled=scheduled, masked=masked)
    positive, binding, raw = encode_ltx_relay_conditioning(clip, plan())
    assert len(clip.calls) == 3 and clip.calls[0].startswith("Global scene:")
    assert clip.calls[1].startswith("Event 1:") and clip.calls[2].startswith("Event 2:")
    assert len(positive) == (2 if scheduled else 1)
    assert binding.token_spans[0][0] == 0
    assert binding.token_spans[-1][1] == positive[0][0].shape[1]
    assert binding.token_count == positive[0][0].shape[1]
    assert json.loads(raw)["attention_applied"] is False
    assert json.loads(raw)["selected_clip_used"] is True
    assert binding.unprocessed_ltxav_embeds is True
    if masked:
        assert positive[0][1]["attention_mask"].shape == (1, binding.token_count)
    assert validate_ltx_relay_binding(binding, positive) is binding
    assert MiniMaxH3LTXPromptRelayEncodeEXPT8.execute(clip, plan()).result[1].token_spans == binding.token_spans


def test_binding_survives_frame_rate_metadata_but_rejects_wrong_content_or_latent():
    positive, binding, _ = encode_ltx_relay_conditioning(SelectedClip(), plan())
    forwarded = [[positive[0][0], {**positive[0][1], "frame_rate": 24.}]]
    assert validate_ltx_relay_binding(binding, forwarded) is binding
    positive[0][0][0, 0, 0] += 1
    with pytest.raises(ValueError, match="not the encoded binding"):
        validate_ltx_relay_binding(binding, positive)
    another, valid, _ = encode_ltx_relay_conditioning(SelectedClip(), plan())
    with pytest.raises(ValueError, match="connected LTX latent"):
        validate_ltx_relay_binding(valid, another, {"samples": torch.zeros(1, 128, 16, 2, 2)})


def test_text_budget_and_incompatible_schedules_fail_without_fallback():
    with pytest.raises(ValueError, match="max_text_tokens"):
        encode_ltx_relay_conditioning(SelectedClip(), plan(), 3)

    class Mismatch(SelectedClip):
        def encode_from_tokens_scheduled(self, tokens):
            rows = super().encode_from_tokens_scheduled(tokens)
            if tokens.startswith("Event 1:"):
                rows[0][1]["unprocessed_ltxav_embeds"] = False
            return rows
    with pytest.raises(ValueError, match="projection"):
        encode_ltx_relay_conditioning(Mismatch(), plan())


def test_tensor_digest_preserves_logical_bytes_across_noncontiguous_large_slices():
    value = torch.arange(1_300_000, dtype=torch.float32).reshape(1000, 1300).transpose(0, 1)
    expected = hashlib.sha256()
    expected.update(str(tuple(value.shape)).encode("ascii"))
    expected.update(str(value.dtype).encode("ascii"))
    expected.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    assert value.numel() * value.element_size() > 4 * 1024 * 1024
    assert tensor_digest(value) == expected.hexdigest()
