"""New operator, actual Core-coordinate and AV ownership checks; no CUDA/weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import comfy.nested_tensor
from comfy.ldm.minimax.model import _frame_grid
from h3_audio_t8_pkg import dense_transfer_exp as dense
from h3_audio_t8_pkg import learned_latent_upscale_advanced as old


def _physical_coordinates(h, w):
    centers = _frame_grid(h, w)[0].reshape(h // 2, w // 2, 2).movedim(-1, 0)
    coordinates = centers.repeat_interleave(2, -2).repeat_interleave(2, -1)
    quarter_step = 16.0 / (h * w) ** .5
    signs_y = torch.tensor([-1., 1.]).repeat(h // 2)
    signs_x = torch.tensor([-1., 1.]).repeat(w // 2)
    coordinates[0] += quarter_step * signs_y[:, None]
    coordinates[1] += quarter_step * signs_x[None, :]
    return coordinates.float()[None, :, None]


def test_dense_coordinates_match_actual_core_patch_centers_not_image_half_pixels():
    for source, target in [((36, 54), (50, 76)), ((54, 36), (76, 50))]:
        value = _physical_coordinates(*source)
        output = dense.resize_dense_field(value, *target)
        exact = _physical_coordinates(*target)
        torch.testing.assert_close(output[..., 5:-5, 5:-5], exact[..., 5:-5, 5:-5], rtol=0, atol=2e-5)
        native = _frame_grid(*target)[0].reshape(target[0] // 2, target[1] // 2, 2).movedim(-1, 0)
        pooled = F.avg_pool2d(output[0, :, 0], 2).double()
        torch.testing.assert_close(pooled[:, 3:-3, 3:-3], native[:, 3:-3, 3:-3], rtol=0, atol=2e-5)
        half = F.interpolate(value, size=(1, *target), mode="trilinear", align_corners=False)
        assert float((half[..., 5:-5, 5:-5] - exact[..., 5:-5, 5:-5]).abs().max()) > .1


def test_both_axes_and_phases_have_one_contiguous_unimodal_edge():
    for axis in (0, 1):
        for phase in (0, 1):
            value = torch.zeros(1, 1, 1, 36, 54)
            if axis == 0:
                value[..., 18 + phase, :] = 1
            else:
                value[..., :, 26 + phase] = 1
            mapped = dense.resize_dense_field(value, 50, 76)[0, 0, 0]
            profile = mapped[:, 38] if axis == 0 else mapped[25]
            indices = (profile > 1e-5).nonzero().flatten()
            assert len(indices) and int(indices[-1] - indices[0]) + 1 == len(indices)
            peak = int(profile.argmax())
            assert bool((profile[:peak + 1].diff() >= -1e-5).all())
            assert bool((profile[peak:].diff() <= 1e-5).all())


def test_noncontiguous_dtype_batch_time_and_invalid_inputs():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        value = torch.arange(24, dtype=dtype).reshape(2, 3, 4, 1, 1).expand(2, 3, 4, 10, 8).transpose(-1, -2)
        before = value.clone()
        output = dense.resize_dense_field(value, 12, 16)
        assert output.dtype == dtype
        # Native grid_sample uses weighted FP32 sums; constant fields are not
        # promised bit-exact, unlike the unresized tensor and owned audio.
        torch.testing.assert_close(output, value[..., :1, :1].expand(2, 3, 4, 12, 16),
                                   rtol=0, atol=1e-5)
        assert torch.equal(value, before)
        assert dense.resize_dense_field(value, 8, 10) is value
    for h in (3, 0, 4.0, True):
        with pytest.raises(ValueError, match="even"):
            dense.resize_dense_field(value, h, 16)
    bad = torch.full((1, 1, 1, 8, 10), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        dense.resize_dense_field(bad, 12, 16)


class TinyNetwork(old.MiniMaxH3LearnedResizer3D):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv_in = nn.Conv3d(24, 32, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = nn.ModuleList([old._ResBlockEmb3D(32), old._TemporalConv(32)])
        self.out_blocks = nn.ModuleList([old._ResBlockEmb3D(32), old._TemporalConv(32)])
        self.norm_out = old._group_norm(32)
        self.conv_out = nn.Conv3d(32, 24, 3, padding=1)


def _setup(monkeypatch):
    network = TinyNetwork().eval()
    entry = old._CachedModel(SimpleNamespace(model=network, load_device=torch.device("cpu")),
                             Path("tiny.safetensors"), "test-only", "fp32", {})
    calls = []
    monkeypatch.setattr(old, "_load_cached_model", lambda *_: (entry, bool(calls)))
    monkeypatch.setattr(dense.management, "load_models_gpu", lambda *a, **k: calls.append(("load", k)))
    monkeypatch.setattr(dense.management, "unload_model_and_clones", lambda *_: calls.append(("unload",)))
    monkeypatch.setattr(dense.management, "soft_empty_cache", lambda: calls.append(("empty",)))
    monkeypatch.setattr(old, "_drop_cached_entry", lambda *_: calls.append(("drop",)))
    return network, calls


def test_real_tiny_layers_encoder_decoder_routing_and_ordinary_forward_unchanged(monkeypatch):
    torch.manual_seed(2081)
    network, calls = _setup(monkeypatch)
    encoded, decoded = [], []
    a = network.in_blocks[-1].register_forward_hook(lambda _m, _i, o: encoded.append(o.detach().clone()))
    b = network.out_blocks[0].register_forward_pre_hook(lambda _m, i: decoded.append(i[0].detach().clone()))
    value = torch.randn(1, 24, 3, 8, 12)
    before = value.clone()
    weights = {key: tensor.clone() for key, tensor in network.state_dict().items()}
    provider = dense.DenseTransferProvider("tiny.safetensors", "fp32")
    try:
        physical = provider.upscale_clean_video_h3_patch_lattice(value, target_h=12, target_w=18)
        ordinary = provider.upscale_clean_video(value, target_h=12, target_w=18)
        repeated = provider.upscale_clean_video_h3_patch_lattice(value, target_h=12, target_w=18)
        portrait = provider.upscale_clean_video_h3_patch_lattice(value.transpose(-1, -2), target_h=18, target_w=12)
    finally:
        a.remove()
        b.remove()
    assert physical.shape == ordinary.shape == repeated.shape == (1, 24, 3, 12, 18)
    assert portrait.shape == (1, 24, 3, 18, 12)
    assert len(encoded) == len(decoded) == 4
    torch.testing.assert_close(decoded[0], dense.resize_dense_field(encoded[0], 12, 18), rtol=0, atol=0)
    torch.testing.assert_close(decoded[1], F.interpolate(encoded[1], size=(3, 12, 18), mode="trilinear", align_corners=False), rtol=0, atol=0)
    torch.testing.assert_close(physical, repeated, rtol=0, atol=0)
    assert not torch.equal(physical, ordinary)
    assert torch.equal(value, before)
    assert all(torch.equal(weights[k], v) for k, v in network.state_dict().items())
    assert sum(row[0] == "load" for row in calls) == sum(row[0] == "unload" for row in calls) == 4


def _latent():
    video = torch.randn(1, 24, 3, 8, 12)
    audio = torch.randn(1, 32, 2, 40)
    mask = torch.linspace(0, 1, 12)[None, None, None, None].expand(1, 1, 3, 8, 12)
    audio_mask = torch.ones_like(audio)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mask, audio_mask)), "custom": {"unchanged": True}}


def test_av_audio_and_mask_ownership_actual_physical_mask_and_unknown_delegate(monkeypatch):
    _setup(monkeypatch)
    latent = _latent()
    original_video, original_audio = latent["samples"].unbind()
    original_mask, original_audio_mask = latent["noise_mask"].unbind()
    output, width, height, raw = dense.upscale_dense_av(av_latent=latent,
                                                      provider=dense.DenseTransferProvider("tiny", "fp32"), scale_by=1.5)
    v, audio = output["samples"].unbind()
    vm, am = output["noise_mask"].unbind()
    assert audio is original_audio and am is original_audio_mask
    assert width == v.shape[-1] * 16 and height == v.shape[-2] * 16
    assert output["custom"] is latent["custom"]
    torch.testing.assert_close(vm, dense.resize_dense_field(original_mask, *v.shape[-2:]), rtol=0, atol=0)
    report = json.loads(raw)
    assert report["complete_temporal_network"] and report["extra_DiT_NFE"] == 0
    delegate = SimpleNamespace(h3_patch_lattice_api=2,
                              upscale_clean_video_h3_patch_lattice=lambda value, **kw: dense.resize_dense_field(value, **kw))
    _, _, _, raw = dense.upscale_dense_av(av_latent=latent, provider=delegate, scale_by=1.5)
    assert not json.loads(raw)["complete_temporal_network"]
    assert torch.equal(original_video, latent["samples"].unbind()[0])


def test_protocol_and_bad_mask_fail_before_provider_and_provider_errors_propagate():
    calls = []
    provider = SimpleNamespace(h3_patch_lattice_api=1, upscale_clean_video_h3_patch_lattice=lambda *_a, **_k: calls.append(1))
    with pytest.raises(ValueError, match="api=2"):
        dense.upscale_dense_av(av_latent=_latent(), provider=provider)
    provider.h3_patch_lattice_api = 2
    latent = _latent()
    latent["noise_mask"].unbind()[0][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        dense.upscale_dense_av(av_latent=latent, provider=provider, scale_by=1.5)
    assert not calls
    def fail(*_a, **_k):
        raise RuntimeError("actual delegate failure")
    provider.upscale_clean_video_h3_patch_lattice = fail
    with pytest.raises(RuntimeError, match="actual delegate failure"):
        dense.upscale_dense_av(av_latent=_latent(), provider=provider, scale_by=1.5)


def test_broadcast_mask_noop_and_invalid_foreign_results(monkeypatch):
    _setup(monkeypatch)
    latent = _latent()
    vm = torch.ones(1, 1, 3, 1, 1)
    am = latent["noise_mask"].unbind()[1]
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
    output, _, _, raw = dense.upscale_dense_av(av_latent=latent, provider=dense.DenseTransferProvider("tiny", "fp32"), scale_by=1.)
    assert output["samples"].unbind()[0] is latent["samples"].unbind()[0]
    assert output["noise_mask"].unbind()[0] is vm
    assert json.loads(raw)["noise_mask"] == "broadcast_preserved"
    provider = SimpleNamespace(h3_patch_lattice_api=2,
                              upscale_clean_video_h3_patch_lattice=lambda v, **k: torch.zeros(1))
    with pytest.raises(RuntimeError, match="invalid video"):
        dense.upscale_dense_av(av_latent=latent, provider=provider, scale_by=1.5)
    # Do not weaken the existing geometry guard to accommodate the tiny fixture.
    with pytest.raises(ValueError, match="anisotropic scale"):
        dense.upscale_dense_av(av_latent=latent, provider=provider, scale_by=1.2)


def test_failure_releases_weights_drops_only_own_cached_entry(monkeypatch):
    network, calls = _setup(monkeypatch)
    def fail(*_args, **_kwargs):
        raise RuntimeError("encoder failure")
    monkeypatch.setattr(network.conv_in, "forward", fail)
    with pytest.raises(RuntimeError, match="encoder failure"):
        dense.DenseTransferProvider("tiny", "fp32").upscale_clean_video_h3_patch_lattice(torch.zeros(1, 24, 3, 8, 12), target_h=12, target_w=18)
    assert [row[0] for row in calls] == ["load", "unload", "drop", "empty"]
