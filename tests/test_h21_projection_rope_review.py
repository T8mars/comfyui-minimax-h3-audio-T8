"""Actual native RoPE owner preservation, not a trained/GPU qualification."""

import comfy.model_management as mm
import comfy.quant_ops
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
import pytest
import torch

from h3_audio_t8_pkg import h3_projection_chunks as chunks
from test_h21_projection_chunks import block, block_inputs, run_block


@pytest.mark.parametrize("training", [False, True])
def test_actual_foreign_row_coupled_RoPE_keeps_full_native_call(monkeypatch, training):
    monkeypatch.setattr(mm, "in_training", training)
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    kitchen = comfy.quant_ops.ck
    name = "rms_rope_split_half" if training else "rms_rope_split_half_"
    original, calls = getattr(kitchen, name), []

    def foreign(q, k, *args, **kwargs):
        calls.append(q.shape[1])
        q.add_(q.mean(dim=1, keepdim=True) * .5)
        return original(q, k, *args, **kwargs)

    monkeypatch.setattr(kitchen, name, foreign)
    with torch.no_grad():
        layer, values = block(), block_inputs(vector=True, rope=True)
        expected = run_block(layer, values)
        assert calls == [19]
        calls.clear()
        with chunks.projection_scope(layer, chunks.ProjectionPlan(rows=3, query_rows=4)) as runtime:
            actual = run_block(layer, values)
        # Existing Core projection contract allows native GEMM row-scheduling
        # rounding. Neither that tolerance nor partial identity permits changing
        # the foreign RoPE's actual full nineteen-token input/call count.
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert calls == [19]
        assert "norm_qkv.fused" not in runtime.calls
        assert runtime.report()["partial"]
        assert not runtime.report()["portable_cache_identity"]
        calls.clear()
        torch.testing.assert_close(run_block(layer, values), expected, rtol=0, atol=0)
        assert calls == [19]
    assert not torch.cuda.is_initialized()
