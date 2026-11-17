"""Compare the current grid-based FinalLayer to actual Core, preserving old ABIs."""
import inspect

import pytest
import torch
import comfy.ops
from comfy.ldm.minimax import model as core

from h3_audio_t8_pkg.h3_core_compat import call_h3_final_layer


@pytest.mark.parametrize("heads", (1, 4))
@pytest.mark.parametrize("shifts", ((12., 3.), (5., 1.)))
def test_actual_grid_head_bank_matches_native_exactly(heads, shifts):
    if "grids" not in inspect.signature(core.FinalLayer.forward).parameters:
        pytest.skip("This Core predates explicit FinalLayer grids")
    layer = core.FinalLayer(8, 4, 24, 32, 1e-5, pdd_steps=heads,
                            dtype=torch.float32, device="cpu",
                            operations=comfy.ops.disable_weight_init)
    generator = torch.Generator().manual_seed(17045)
    for parameter in layer.parameters():
        parameter.data.copy_(torch.randn(parameter.shape, generator=generator) * .05)
    args = (torch.randn(4, 8, generator=generator), torch.randn(1, 4, generator=generator),
            (0, 2, 0), (2, 4, 0))
    sigma = torch.tensor(.7)
    schedule = torch.tensor([1., .7, .3, 0.])
    grids = tuple(core.pdd_grid(heads, shift) for shift in shifts)
    expected = layer(*args, sigma=sigma, sample_sigmas=schedule, grids=grids)
    actual = call_h3_final_layer(layer, *args, sigma=sigma, sample_sigmas=schedule, shifts=shifts)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_explicit_grids_and_user_delegate_are_preserved_once():
    calls = []
    grids = ([1., .8, 0.], [1., .6, 0.])
    sigma, schedule = object(), object()

    class UserLayer:
        def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
            calls.append((sigma, sample_sigmas, grids))
            return x, t_emb

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    assert call_h3_final_layer(UserLayer(), 2, 3, None, None, sigma=sigma,
                               sample_sigmas=schedule, shifts=(12., 3.), grids=grids) == (2, 3)
    assert len(calls) == 1 and calls[0][0] is sigma and calls[0][1] is schedule
    assert calls[0][2] is grids


def test_grid_delegate_error_is_not_retried_or_hidden():
    calls = []

    class BrokenLayer:
        def __call__(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
            calls.append(grids)
            raise TypeError("actual grid consumer failed")

    with pytest.raises(TypeError, match="actual grid consumer failed"):
        call_h3_final_layer(BrokenLayer(), 2, 3, None, None, sigma=.5,
                             sample_sigmas=None, shifts=(12., 3.), grids=([1., 0.],))
    assert len(calls) == 1


def test_optional_grid_keyword_does_not_reject_a_legacy_user_delegate():
    calls = []

    class UserLayer:
        def __call__(self, x, t_emb, video_seg, audio_seg, grids=None):
            calls.append(grids)
            return x, t_emb

    assert call_h3_final_layer(UserLayer(), 2, 3, None, None, sigma=.5,
                               sample_sigmas=None, shifts=(12., 3.)) == (2, 3)
    assert calls == [None]
