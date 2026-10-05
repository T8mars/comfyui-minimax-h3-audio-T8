"""Synthetic legal CPU fixtures; no pretrained weights or quality acceptance."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest
from safetensors.torch import save_file
import torch

from h3_audio_t8_pkg.reference_package import (
    META_KEY, NORMALIZATION, ReferencePackage, canonical, capture_package,
    installed_package_names, load_package, resolve_installed_package, route_packages,
    save_package, tensor_record,
)


def fixture(role="A", *, voice=True, grounding=True):
    rgb = torch.full((1, 32, 32, 3), .25)
    tensors = {"visual": torch.full((1, 24, 1, 2, 2), .125)}
    if grounding:
        tensors["rgb"] = rgb
    source_sha = tensor_record(rgb)["sha256"]
    producer = {"role": "video_vae", "sha256": "1" * 64,
                "scope": "actual_native_weights_tokenizer_configuration_and_implementation"}
    members = [{"id": "picture", "role_id": role, "kind": "image", "latent": "visual",
                "grounding": "rgb" if grounding else None,
                "source": {"sha256": source_sha, "scope": "actual_decoded_rgb_tensor"},
                "producer": producer, "normalization": NORMALIZATION}]
    if voice:
        tensors["voice"] = torch.full((1, 32, 2, 4), .75)
        members.append({"id": "voice", "role_id": role, "kind": "audio", "latent": "voice", "grounding": None,
            "source": {"sha256": "2" * 64, "scope": "actual_decoded_pcm_tensor"},
            "producer": {**producer, "role": "audio_vae"}, "normalization": NORMALIZATION})
    return members, tensors


def reseal(manifest):
    manifest = deepcopy(manifest)
    manifest.pop("sha256", None)
    manifest["sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    return manifest


def test_atomic_cpu_roundtrip_keeps_values_and_never_overwrites_or_saves_unconfirmed(tmp_path):
    members, tensors = fixture()
    package = capture_package(members, tensors)
    tensors["visual"].zero_()
    assert package.tensors["visual"].mean().item() == .125
    path = tmp_path / "A.safetensors"
    with pytest.raises(ValueError, match="confirmation"):
        save_package(package, path)
    assert not path.exists()
    receipt = save_package(package, path, confirmed=True)
    restored = load_package(path, expected_sha256=receipt["sha256"])
    assert restored.manifest_json == package.manifest_json
    assert restored.file_sha256 == receipt["sha256"]
    assert all(torch.equal(restored.tensors[name], value) for name, value in package.tensors.items())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_package(package, path, confirmed=True)
    assert path.read_bytes() == before
    restored.tensors["voice"][0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match="content changed"):
        restored.verify()


def test_missing_grounding_only_optin_does_not_bypass_producer_shape_or_sha(tmp_path):
    members, tensors = fixture(grounding=False)
    with pytest.raises(ValueError, match="Missing Qwen"):
        capture_package(members, tensors)
    package = capture_package(members, tensors, allow_missing_grounding=True)
    receipt = save_package(package, tmp_path / "latent_only.safetensors", confirmed=True, allow_missing_grounding=True)
    with pytest.raises(ValueError, match="Missing Qwen"):
        load_package(receipt["path"])
    assert load_package(receipt["path"], allow_missing_grounding=True).verify(allow_missing_grounding=True)
    wrong = deepcopy(members)
    wrong[0]["producer"]["scope"] = "filename_inference"
    with pytest.raises(ValueError, match="producer"):
        capture_package(wrong, tensors, allow_missing_grounding=True)
    bad = dict(tensors, visual=torch.ones((1, 23, 1, 2, 2)))
    with pytest.raises(ValueError, match="Visual reference"):
        capture_package(members, bad, allow_missing_grounding=True)
    wrong = deepcopy(members)
    wrong[0]["kind"] = "video"
    with pytest.raises(ValueError, match="Visual reference"):
        capture_package(wrong, tensors, allow_missing_grounding=True)
    bad = dict(tensors, voice=torch.full((1, 32, 2, 4), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        capture_package(members, bad, allow_missing_grounding=True)
    with pytest.raises(ValueError, match="SHA changed"):
        load_package(receipt["path"], expected_sha256="0" * 64, allow_missing_grounding=True)


def test_explicit_ordered_offscreen_voice_removes_both_visual_paths_not_global_strength():
    a = capture_package(*fixture("A"))
    b = capture_package(*fixture("B", voice=False))
    before = a.manifest_json
    routed = route_packages([a, b], [{"role_id": "A", "visual": False, "voice": True},
                                   {"role_id": "B", "visual": True, "voice": False}])
    assert [block["kind"] for block in routed["refs"]] == ["audio", "image"]
    assert routed["refs"][0]["audio_latent"] is a.tensors["voice"]
    assert routed["refs"][1]["latent"] is b.tensors["visual"]
    assert routed["qwen_ref_items"][0] == {"type": "audio"}
    assert routed["qwen_ref_items"][1]["data"] is b.tensors["rgb"]
    assert all(item.get("data") is not a.tensors["rgb"] for item in routed["qwen_ref_items"])
    assert routed["mapping"] == [
        {"role_id": "A", "member_id": "voice", "kind": "audio", "ordinal": 1, "packed_reference_rows": 8},
        {"role_id": "B", "member_id": "picture", "kind": "image", "ordinal": 1, "packed_reference_rows": 1}]
    assert routed["packed_reference_rows"] == 9
    assert routed["applied_to_conditioning"] is False and routed["latent_only_exp"] is False
    assert a.manifest_json == before
    with pytest.raises(ValueError, match="multiple packages"):
        route_packages([a, a], [{"role_id": "A", "visual": True, "voice": True}])
    with pytest.raises(ValueError, match="every package role"):
        route_packages([a, b], [{"role_id": "A", "visual": True, "voice": True}])


def test_untrusted_header_unknown_version_duplicate_metadata_and_allocation_descriptors_rejected(tmp_path):
    package = capture_package(*fixture())
    manifest = json.loads(package.manifest_json)
    manifest["schema"] = "community-format-5"
    wrong = ReferencePackage(canonical(reseal(manifest)), package.tensors)
    with pytest.raises(ValueError, match="Unsupported reference"):
        wrong.verify()
    duplicate = tmp_path / "duplicate.safetensors"
    raw = b'{"__metadata__":{},"__metadata__":{}}'
    duplicate.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ValueError, match="Duplicate"):
        load_package(duplicate)
    path = tmp_path / "descriptor.safetensors"
    manifest = json.loads(package.manifest_json)
    manifest["tensors"]["visual"]["shape"] = [1, 24, 16384, 16384, 16384]
    save_file(package.tensors, path, metadata={META_KEY: canonical(reseal(manifest))})
    with pytest.raises(ValueError, match="byte count"):
        load_package(path)
    manifest = json.loads(package.manifest_json)
    manifest["members"][0]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="RGB grounding"):
        ReferencePackage(canonical(reseal(manifest)), package.tensors).verify()


def test_models_refmods_registration_preserves_extra_dirs_and_exact_selection(tmp_path, monkeypatch):
    import folder_paths
    models, extra = tmp_path / "models", tmp_path / "extra"
    extra.mkdir()
    models.mkdir()
    monkeypatch.setattr(folder_paths, "models_dir", str(models))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {"refmods": ([str(extra)], {".safetensors"})})
    path = extra / "A.safetensors"
    save_package(capture_package(*fixture()), path, confirmed=True)
    assert "A.safetensors" in installed_package_names()
    assert str(extra) in folder_paths.get_folder_paths("refmods")
    assert str(models / "refmods") in folder_paths.get_folder_paths("refmods")
    assert resolve_installed_package("A.safetensors") == path
    for wrong in (str(path), "../A.safetensors", "..\\A.safetensors", "C:A.safetensors", "./A.safetensors"):
        with pytest.raises(ValueError, match="exact installed"):
            resolve_installed_package(wrong)
    with pytest.raises(FileNotFoundError):
        resolve_installed_package("Missing.safetensors")
