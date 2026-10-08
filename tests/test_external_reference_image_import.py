"""New importer unit boundaries only; synthetic VAE is NOT model qualification."""
import json

import pytest
from safetensors.torch import save_file
import torch

from h3_audio_t8_pkg import external_reference_image_import as runtime
from h3_audio_t8_pkg.reference_package import file_sha, load_package, route_packages, save_package


class TinyEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, rgb):
        self.calls += 1
        return (torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2) / 97
                + rgb.mean())


@pytest.fixture
def source(monkeypatch):
    rgb = torch.full((1, 32, 32, 3), 0.25)
    encoder = TinyEncoder()
    native = encoder.encode(rgb)
    encoder.calls = 0
    producer = {"role": "video_vae", "sha256": "1" * 64,
                "scope": "actual_native_weights_tokenizer_configuration_and_implementation"}
    monkeypatch.setattr(runtime, "portable_producer", lambda component, role: dict(producer))
    return rgb, encoder, native


def external(path, latent, *, version=4, mode="encode", hybrid=False, video=False):
    member = {"_format_version": 4, "kind": "video" if video else "image", "mode": mode,
              "latent_t": int(latent.shape[2]), "latent_h": 2, "latent_w": 2,
              "refmod_config": '{"execute":"must never run"}', "source": "Z:/not-followed.png"}
    if version == 4:
        tensors, meta = {"latent": latent}, member
    else:
        audio = {"_format_version": 4, "kind": "audio", "mode": "encode", "sample_rate": 32000}
        tensors = {"ref_0": torch.full((1, 32, 2, 3), float("nan")), "ref_1": latent}
        meta = {"_format_version": 5, "kind": "bundle", "members": [audio, member]}
    metadata = {"refmod_meta": json.dumps(meta)}
    if hybrid:
        tensors["lora.test.weight"] = torch.zeros(2, 2)
        metadata["h3_hybrid"] = json.dumps({"version": 1, "lora": {"keys": 1}, "refmod_count": 2})
    save_file(tensors, str(path), metadata=metadata)
    return file_sha(path)


def run(path, source, **kwargs):
    rgb, encoder, _ = source
    return runtime.import_image(path, rgb, encoder, expected_sha256=file_sha(path),
                                confirm_import=True, **kwargs)


def test_native_exact_preserves_source_payload_and_routes_and_save_is_create_only(tmp_path, source):
    path = tmp_path / "external.safetensors"
    original = external(path, source[2])
    rgb_before = source[0].clone()
    package, report = run(path, source)
    assert source[1].calls == 1 and torch.equal(source[0], rgb_before)
    assert report["native_encode_bytes_exact"] and report["external_latent_bytes_preserved"]
    assert not report["configuration_executed"] and not report["external_writer_or_encoder_authenticated"]
    assert file_sha(path) == original
    routed = route_packages([package], [{"role_id": "A", "visual": True, "voice": False}])
    assert torch.equal(routed["refs"][0]["latent"], source[2])
    assert torch.equal(routed["qwen_ref_items"][0]["data"], source[0])
    destination = tmp_path / "new.safetensors"
    saved = save_package(package, destination, confirmed=True)
    loaded = load_package(destination, expected_sha256=saved["sha256"])
    assert loaded.verify() == package.verify()
    with pytest.raises(FileExistsError):
        save_package(package, destination, confirmed=True)
    assert file_sha(path) == original


def test_v5_hybrid_loads_only_explicit_selected_image_and_author_cast_is_visible(tmp_path, source, monkeypatch):
    path = tmp_path / "hybrid.safetensors"
    original = external(path, source[2].half(), version=5, hybrid=True)
    calls = []
    actual_open = runtime.safe_open

    class SelectedReader:
        def __enter__(self):
            self.context = actual_open(path, framework="pt", device="cpu")
            self.file = self.context.__enter__()
            return self

        def get_tensor(self, key):
            calls.append(key)
            return self.file.get_tensor(key)

        def __exit__(self, *args):
            return self.context.__exit__(*args)

    monkeypatch.setattr(runtime, "safe_open", lambda *args, **kwargs: SelectedReader())
    package, report = run(path, source, member_ordinal=2, precision_profile="author_encode_fp16")
    assert calls == ["ref_1"] and report["tensor_loads"] == 1
    assert not report["native_encode_bytes_exact"] and not report["LoRA_weights_loaded_or_applied"]
    assert torch.equal(package.tensors["visual_latent"], source[2].half())
    assert file_sha(path) == original


def test_default_does_not_silently_cast_and_wrong_RGB_is_not_adopted(tmp_path, source):
    path = tmp_path / "fp16.safetensors"
    external(path, source[2].half())
    with pytest.raises(ValueError, match="exactly match"):
        run(path, source)
    changed = (source[0] + 0.1, source[1], source[2])
    with pytest.raises(ValueError, match="exactly match"):
        run(path, changed, precision_profile="author_encode_fp16")


def test_confirmation_hash_geometry_and_member_fail_before_encode(tmp_path, source):
    path = tmp_path / "external.safetensors"
    digest = external(path, source[2])
    for kwargs, message in [({"confirm_import": False, "expected_sha256": digest}, "confirmation"),
                            ({"confirm_import": True, "expected_sha256": "0" * 64}, "SHA256")]:
        with pytest.raises(ValueError, match=message):
            runtime.import_image(path, source[0], source[1], **kwargs)
    with pytest.raises(ValueError, match="ordinal"):
        run(path, source, member_ordinal=2)
    with pytest.raises(ValueError, match="geometry"):
        run(path, (source[0][:, :16], source[1], source[2]))
    assert source[1].calls == 0


def test_training_and_nonimage_members_are_not_qualified(tmp_path, source):
    path = tmp_path / "training.safetensors"
    external(path, source[2], mode="training")
    with pytest.raises(ValueError, match="Training/pooled"):
        run(path, source)
    path = tmp_path / "video.safetensors"
    external(path, source[2].repeat(1, 1, 2, 1, 1), video=True)
    with pytest.raises(ValueError, match="Only one image"):
        run(path, source)
    assert source[1].calls == 0


def test_selected_nonfinite_payload_rejected_before_encode(tmp_path, source):
    path = tmp_path / "bad.safetensors"
    values = source[2].clone()
    values[0, 0, 0, 0, 0] = float("nan")
    external(path, values)
    with pytest.raises(ValueError, match="finite"):
        run(path, source)
    assert source[1].calls == 0


def test_actual_producer_change_and_real_encode_exception_propagate(tmp_path, source, monkeypatch):
    path = tmp_path / "external.safetensors"
    external(path, source[2])
    checks = []

    def changing(component, role):
        checks.append(role)
        return {"role": role, "sha256": str(len(checks)) * 64, "scope": "actual"}

    monkeypatch.setattr(runtime, "portable_producer", changing)
    with pytest.raises(ValueError, match="changed during"):
        run(path, source)

    def broken(rgb):
        raise RuntimeError("native encode failed")

    monkeypatch.setattr(source[1], "encode", broken)
    with pytest.raises(RuntimeError, match="native encode failed"):
        run(path, source)


def test_file_change_during_encode_never_publishes(tmp_path, source, monkeypatch):
    path = tmp_path / "external.safetensors"
    external(path, source[2])
    actual_encode = source[1].encode

    def changing(rgb):
        value = actual_encode(rgb)
        with path.open("ab") as stream:
            stream.write(b"\0")
        return value

    monkeypatch.setattr(source[1], "encode", changing)
    with pytest.raises(ValueError, match="changed during import"):
        run(path, source)
    assert list(tmp_path.iterdir()) == [path]
