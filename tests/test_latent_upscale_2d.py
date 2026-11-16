from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
import comfy.model_patcher
import comfy.nested_tensor

from h3_audio_t8_pkg import latent_upscale_2d as resize
from h3_audio_t8_pkg.vendor.lbh_upscale2d import VideoLatentResizer
from h3_audio_t8_pkg.nodes_latent_upscale_2d import MiniMaxH3LatentUpscale2DT8
import h3_audio_t8_pkg


def tiny_network():
    torch.manual_seed(123)
    model = VideoLatentResizer(
        in_channels=24,
        in_blocks=3,
        out_blocks=3,
        channels=32,
        embed_dim=16,
        tier_dim=32,
        use_coord=False,
        use_se=False,
        use_large_kernel=False,
        use_post_refine=False,
        use_multiscale_temporal=False,
        temporal_every=2,
        temporal_kernel=3,
        dropout=0.0,
    )
    # Nonzero trained-like residuals exercise tier and temporal paths.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.025)
    return model.eval()


def latent(frames=13):
    v = torch.randn(1, 24, frames, 4, 6)
    a = torch.randn(1, 32, 2, 40)
    am = torch.rand_like(a)
    return {
        "samples": comfy.nested_tensor.NestedTensor((v, a)),
        "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(v), am)),
        "metadata": {"source": "retained"},
    }


def entry(monkeypatch, network):
    patcher = comfy.model_patcher.ModelPatcher(
        network, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    e = resize.shared._CachedModel(
        patcher, Path("tiny.safetensors"), "tiny-fixture", "fp32", {}
    )
    monkeypatch.setattr(resize, "_load", lambda *_: (e, False))
    return e


def test_strict_structure_load_and_mismatch():
    model = tiny_network()
    restored, config = resize.build_resizer(model.state_dict())
    assert config["tier_dim"] == 32 and config["channels"] == 32
    x = torch.randn(1, 24, 7, 4, 6)
    with torch.inference_mode():
        assert torch.equal(
            model._forward_full(x, (8, 12), 3), restored._forward_full(x, (8, 12), 3)
        )
    bad = dict(model.state_dict())
    del bad["resizer.conv_out.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        resize.build_resizer(bad)


def test_chunked_output_preserves_every_frame_audio_mask_and_metadata(monkeypatch):
    model = tiny_network()
    entry(monkeypatch, model)
    source = latent()
    full, w, h, _ = resize.upscale_2d(source, "tiny", precision="fp32", chunk_frames=0)
    chunked, _, _, report = resize.upscale_2d(
        source, "tiny", precision="fp32", chunk_frames=2
    )
    fv, fa = tuple(full["samples"].unbind())
    cv, ca = tuple(chunked["samples"].unbind())
    sv, sa = tuple(source["samples"].unbind())
    assert (w, h) == (192, 128)
    assert cv.shape == (1, 24, 13, 8, 12)
    torch.testing.assert_close(cv, fv, rtol=2e-5, atol=2e-6)
    assert fa is sa and ca is sa
    assert chunked["metadata"] is source["metadata"]
    mask, audio_mask = tuple(chunked["noise_mask"].unbind())
    assert mask.shape == cv.shape
    assert audio_mask is tuple(source["noise_mask"].unbind())[1]
    assert sv.shape[-2:] == (4, 6)
    assert json.loads(report)["temporal_halo"] == 4
    assert json.loads(report)["actual_network_calls"] == 7


def test_author_tier_actually_reaches_the_network(monkeypatch):
    model = tiny_network()
    entry(monkeypatch, model)
    source = latent(5)
    one = resize.upscale_2d(source, "tiny", precision="fp32", sharpness=1)[0]
    six = resize.upscale_2d(source, "tiny", precision="fp32", sharpness=6)[0]
    assert not torch.equal(one["samples"].unbind()[0], six["samples"].unbind()[0])


def test_no_resize_is_exact_noop_without_loading_weights(monkeypatch):
    monkeypatch.setattr(
        resize, "_load", lambda *_: pytest.fail("noop loaded a network")
    )
    source = latent()
    result, _, _, report = resize.upscale_2d(source, "absent", scale_by=1.0)
    assert result is source and json.loads(report)["model_loaded"] is False


def test_exception_releases_owned_network_and_never_returns_partial_output(monkeypatch):
    model = tiny_network()
    e = entry(monkeypatch, model)
    unloaded = []
    monkeypatch.setattr(
        resize.mm, "unload_model_and_clones", lambda p: unloaded.append(p)
    )
    monkeypatch.setattr(
        model,
        "_forward_full",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("actual failure")),
    )
    with pytest.raises(RuntimeError, match="actual failure"):
        resize.upscale_2d(latent(), "tiny", precision="fp32")
    assert unloaded == [e.patcher]


def test_nonfinite_input_fails_before_weight_loading(monkeypatch):
    monkeypatch.setattr(
        resize, "_load", lambda *_: pytest.fail("invalid input loaded weights")
    )
    source = latent()
    source["samples"].unbind()[0][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        resize.upscale_2d(source, "tiny")


def test_missing_mask_and_normalized_geometry(monkeypatch):
    entry(monkeypatch, tiny_network())
    source = latent(3)
    source.pop("noise_mask")
    result, _, _, report = resize.upscale_2d(source, "tiny", precision="fp32")
    assert "noise_mask" not in result
    assert json.loads(report)["noise_mask"] == "absent"


def test_node_is_appended_to_existing_registration_prefix():
    nodes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert nodes[788] is MiniMaxH3LatentUpscale2DT8
    schema = MiniMaxH3LatentUpscale2DT8.define_schema()
    assert schema.node_id == "MiniMaxH3LatentUpscale2DT8"


def test_2D_does_not_borrow_3D_recipe_scale_cap_and_delegates_real_shapes(monkeypatch):
    entry(monkeypatch, tiny_network())
    source = latent(3)
    result, w, h, report = resize.upscale_2d(
        source, "tiny", precision="fp32", scale_by=5.0, chunk_frames=0
    )
    assert (w, h) == (480, 320)
    assert result["samples"].unbind()[0].shape[-2:] == (20, 30)
    assert result["samples"].unbind()[1] is source["samples"].unbind()[1]
    assert json.loads(report)["geometry"]["outside_legacy_3d_geometry"]
    small, w, h, _ = resize.upscale_2d(
        source,
        "tiny",
        precision="fp32",
        size_mode="target_dimensions",
        target_width=64,
        target_height=32,
        aspect_policy="honor_dimensions_exp",
        max_anisotropy=2.0,
    )
    assert (w, h) == (64, 32) and small["samples"].unbind()[0].shape[-2:] == (2, 4)
    with pytest.raises(ValueError, match="4x"):
        resize.shared.learned_upscale_geometry(
            6, 4, "target_dimensions", 5.0, 0.6, 480, 320, "preserve_source", 1.05
        )
