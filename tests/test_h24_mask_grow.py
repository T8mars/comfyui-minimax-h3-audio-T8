"""Compare actual prepare_mask output/report bytes with its original 2D grow."""
import pytest
import torch
import torch.nn.functional as F

from h3_audio_t8_pkg import person_removal as removal


def original_square(value, radius):
    return F.max_pool2d(value, 2 * radius + 1, stride=1, padding=radius)


def raw(value):
    return value.contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 32, 64)))
@pytest.mark.parametrize("kind", ("soft_noncontiguous", "signed_zero", "ones", "edges"))
@pytest.mark.parametrize("grow", (0, 1, 12, 64))
@pytest.mark.parametrize("feather", (0, 2))
def test_actual_mask_and_report_are_byte_identical(dtype, shape, kind, grow, feather, monkeypatch):
    generator = torch.Generator().manual_seed(241024)
    frames = torch.rand((*shape, 3), generator=generator)
    if kind == "soft_noncontiguous":
        mask = torch.rand((shape[0], shape[1], shape[2] * 2), generator=generator).to(dtype)[..., ::2]
        assert not mask.is_contiguous()
    elif kind == "signed_zero":
        mask = torch.zeros(shape, dtype=dtype)
        mask[..., ::2] = -0.
    elif kind == "ones":
        mask = torch.ones(shape, dtype=dtype)
    else:
        mask = torch.zeros(shape, dtype=dtype)
        mask[:, 0, 0], mask[:, -1, -1] = 1., .5
    before = raw(mask), raw(frames)
    actual = removal.prepare_mask(frames, mask, grow, feather)
    monkeypatch.setattr(removal, "_grow_square", original_square)
    expected = removal.prepare_mask(frames, mask, grow, feather)
    assert raw(actual[0]) == raw(expected[0])
    assert raw(actual[1]) == raw(expected[1])
    assert actual[2] == expected[2]
    assert (raw(mask), raw(frames)) == before
    assert not torch.cuda.is_initialized()
