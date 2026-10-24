import json
import asyncio

import pytest
import torch

from h3_audio_t8_pkg import hyper_vae_2x as subject
from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.nodes_hyper_vae_2x import MiniMaxH3HyperVAE2xLoaderEXPT8


class _Slice:
    def __init__(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape


class _Weights:
    def __init__(self, *, packed=12, factor=2):
        self.shapes = {
            "decoder.proj_out.weight": (packed * 1024, 2048),
            "decoder.proj_out.bias": (packed * 1024,),
            "latents_mean": (24,),
            "latents_std": (24,),
            "encoder.down.5.block.0.conv1.weight": (1,),
        }
        self.info = {
            "minimax_h3_x2_adapter": json.dumps({
                "version": 1,
                "packed_output_channels": packed,
                "pixel_shuffle_factor": factor,
                "encoder_unchanged": True,
            }),
            "minimax_h3_video_vae": json.dumps({
                "packed_decoder_output_channels": packed,
                "decoder_pixel_shuffle_factor": factor,
            }),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return self.shapes.keys()

    def get_slice(self, key):
        return _Slice(self.shapes[key])

    def metadata(self):
        return self.info


def test_header_accepts_only_packed_h3_2x(tmp_path, monkeypatch):
    path = tmp_path / "hyper.safetensors"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(subject, "safe_open", lambda *_args, **_kwargs: _Weights())
    report = subject.inspect_hyper_vae_2x(path)
    assert report["decode_scale_from_latent"] == 32
    assert report["encode_scale"] == 16
    monkeypatch.setattr(subject, "safe_open", lambda *_args, **_kwargs: _Weights(packed=3))
    with pytest.raises(ValueError, match="形状不匹配"):
        subject.inspect_hyper_vae_2x(path)
    monkeypatch.setattr(subject, "safe_open", lambda *_args, **_kwargs: _Weights(factor=1))
    with pytest.raises(ValueError, match="元数据"):
        subject.inspect_hyper_vae_2x(path)


def test_pixel_shuffle_keeps_phase_and_time_order():
    packed = torch.arange(2 * 12, dtype=torch.float32).reshape(1, 2, 1, 1, 12)
    images = subject.pixel_shuffle_h3_images(packed)
    assert images.shape == (1, 2, 2, 2, 3)
    assert torch.equal(images[0, 0, :, :, 0], torch.tensor([[0., 1.], [2., 3.]]))
    assert torch.equal(images[0, 0, :, :, 1], torch.tensor([[4., 5.], [6., 7.]]))
    assert torch.equal(images[0, 1, :, :, 2], torch.tensor([[20., 21.], [22., 23.]]))
    with pytest.raises(ValueError, match="12"):
        subject.pixel_shuffle_h3_images(torch.zeros(1, 2, 1, 1, 3))


def test_packed_finalize_does_not_change_encoder_rgb_statistics():
    class Inner:
        pixel_mean = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([0.5, 0.6, 0.7]).reshape(1, 3, 1, 1, 1)

    inner = Inner()
    pixels = subject._finalize_hyper_pixels(inner, torch.zeros(1, 12, 1, 1, 1))
    assert torch.allclose(pixels.flatten(), torch.tensor([0.1] * 4 + [0.2] * 4 + [0.3] * 4))
    assert inner.pixel_mean.shape == (1, 3, 1, 1, 1)
    with pytest.raises(ValueError, match="12通道"):
        subject._finalize_hyper_pixels(inner, torch.zeros(1, 3, 1, 1, 1))


def test_vae_decode_wrapper_and_public_schema(monkeypatch):
    monkeypatch.setattr(subject.comfy.sd.VAE, "decode", lambda _self, *_args, **_kwargs: torch.zeros(1, 1, 2, 3, 12))
    monkeypatch.setattr(subject.comfy.sd.VAE, "decode_tiled", lambda _self, *_args, **_kwargs: torch.zeros(1, 1, 2, 3, 12))
    vae = object.__new__(subject.HyperVAE2x)
    latent = torch.empty(1, 24, 1, 1, 1)
    assert vae.spacial_compression_decode() == 32
    assert vae.decode(latent).shape == (1, 1, 4, 6, 3)
    assert vae.decode_tiled(latent).shape == (1, 1, 4, 6, 3)
    with pytest.raises(ValueError, match="仅接受"):
        vae.decode(torch.empty(1, 32, 1, 1, 1))
    info = MiniMaxH3HyperVAE2xLoaderEXPT8.define_schema().get_v1_info(MiniMaxH3HyperVAE2xLoaderEXPT8)
    assert info.output == ["VAE", "STRING"]


def test_absolute_path_override_does_not_require_models_folder(monkeypatch, tmp_path):
    path = tmp_path / "hyper.safetensors"
    path.write_bytes(b"fixture")
    captured = []
    monkeypatch.setattr(subject, "load_hyper_vae_2x", lambda value: (captured.append(value) or object(), {"status": "ok"}))
    result = MiniMaxH3HyperVAE2xLoaderEXPT8.execute("填写绝对路径", str(path))
    assert captured == [path]
    assert json.loads(result.result[1])["status"] == "ok"
    with pytest.raises(ValueError, match="绝对路径"):
        MiniMaxH3HyperVAE2xLoaderEXPT8.execute("填写绝对路径", "relative.safetensors")


def test_vae_name_resolves_from_models_vae(monkeypatch, tmp_path):
    import folder_paths

    name = "hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors"
    path = tmp_path / "models" / "vae" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda category: [name] if category == "vae" else [])
    calls = []

    def resolve(category, selected):
        calls.append((category, selected))
        return str(path)

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", resolve)
    loaded = []
    monkeypatch.setattr(subject, "load_hyper_vae_2x", lambda value: (loaded.append(value) or object(), {"status": "ok"}))
    schema = MiniMaxH3HyperVAE2xLoaderEXPT8.define_schema().get_v1_info(MiniMaxH3HyperVAE2xLoaderEXPT8)
    assert name in str(schema.input)
    result = MiniMaxH3HyperVAE2xLoaderEXPT8.execute(name, "")
    assert calls == [("vae", name)]
    assert loaded == [path]
    assert json.loads(result.result[1])["status"] == "ok"


def test_node_is_appended_after_existing_extension_nodes():
    import h3_audio_t8_pkg
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    prefix = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
              *h3_audio_t8_pkg._modular_node_classes()]
    assert classes[:len(prefix)] == prefix
    assert classes[len(prefix)] is MiniMaxH3HyperVAE2xLoaderEXPT8
    assert classes.count(MiniMaxH3HyperVAE2xLoaderEXPT8) == 1
