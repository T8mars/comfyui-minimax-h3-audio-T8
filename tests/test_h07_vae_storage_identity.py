"""Logical-byte identity only; no encoder/sampler or pretrained weights."""
import hashlib

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_identity import value_identity


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_ndhwc_logical_bytes_equal_legacy_identity_with_bounded_copies(dtype, monkeypatch):
    original = (torch.arange(128*512*27, dtype=torch.float32).reshape(128,512,3,3,3) % 137).to(dtype)
    expected = dict(tensor_sha256=hashlib.sha256(original.view(torch.uint8).numpy().tobytes()).hexdigest(),
        dtype=str(dtype), shape=list(original.shape))
    ndhwc = original.contiguous(memory_format=torch.channels_last_3d)
    assert not ndhwc.is_contiguous()
    pointer, strides = ndhwc.data_ptr(), ndhwc.stride()
    actual_contiguous = torch.Tensor.contiguous
    copies, interrupts = [], []

    def bounded(value, *args, **kwargs):
        copies.append(value.numel()*value.element_size())
        assert copies[-1] <= 4*1024**2
        return actual_contiguous(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", bounded)
    assert value_identity(original) == expected
    assert value_identity(ndhwc, interrupt_check=lambda: interrupts.append(True)) == expected
    assert len(interrupts) >= 1 and copies
    assert ndhwc.data_ptr() == pointer and ndhwc.stride() == strides
    ndhwc[57,13,2,1,1] += 1
    assert value_identity(ndhwc) != expected
    assert value_identity(original) == expected


def test_native_ndhwc_nan_and_unrelated_materialization_still_rejected():
    bad = torch.zeros(3,4,3,3,3).contiguous(memory_format=torch.channels_last_3d)
    bad[1,2,1,1,1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        value_identity(bad)
    for value in (torch.ones(3,4).T, torch.empty(3,4,3,3,3, device="meta"),
                  torch.ones(3,4,3,3,3).transpose(1,2)):
        with pytest.raises(ValueError, match="unsupported materializing"):
            value_identity(value)
    huge_row = torch.zeros(2,512,16,16,16).contiguous(memory_format=torch.channels_last_3d)
    with pytest.raises(ValueError, match="bounded materialization"):
        value_identity(huge_row)


def test_native_ndhwc_interrupt_is_not_swallowed():
    value = torch.zeros(3,4,3,3,3).contiguous(memory_format=torch.channels_last_3d)

    def interrupted():
        raise RuntimeError("explicit interrupt")

    with pytest.raises(RuntimeError, match="explicit interrupt"):
        value_identity(value, interrupt_check=interrupted)
