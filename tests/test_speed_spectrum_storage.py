from __future__ import annotations

import json
import math

import pytest
import torch

from h3_audio_t8_pkg.speed_advanced import accumulate_spectrum_dataset
from h3_audio_t8_pkg.speed_spectrum_storage import (
    load_spectrum_dataset_file,
    save_spectrum_dataset_file,
    sha256_file,
    spectrum_dataset_file_fingerprint,
)
from h3_audio_t8_pkg.nodes_speed_advanced import (
    MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced,
)
import h3_audio_t8_pkg.nodes_speed_advanced as nodes_speed_advanced


def _dataset():
    generator = torch.Generator().manual_seed(79)
    latent = torch.randn(2, 24, 4, 16, 24, generator=generator)
    flattened = latent.reshape(-1, 1, 16, 24)
    flattened = torch.nn.functional.avg_pool2d(
        flattened, kernel_size=3, stride=1, padding=1
    )
    latent = flattened.reshape(2, 24, 4, 16, 24)
    return accumulate_spectrum_dataset(
        latent,
        batch_id="batch-a",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model-a",
        vae_fingerprint="sha256:vae-a",
        max_temporal_samples=4,
    )[0]


def test_spectrum_dataset_file_round_trip_is_single_atomic_safetensors(tmp_path):
    dataset = _dataset()
    saved, path, save_report = save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="t2va-v1",
        overwrite=False,
    )
    loaded, loaded_path, load_report = load_spectrum_dataset_file(
        root=tmp_path, dataset_name="t2va-v1"
    )
    assert path == loaded_path
    assert saved["power_sum"].data_ptr() == dataset["power_sum"].data_ptr()
    assert loaded["power_sum"].device.type == "cpu"
    assert loaded["power_sum"].dtype == torch.float64
    assert torch.equal(loaded["power_sum"], dataset["power_sum"])
    assert loaded["power_sum_sha256"] == dataset["power_sum_sha256"]
    assert loaded["clip_fingerprints"] == dataset["clip_fingerprints"]
    assert json.loads(save_report)["source_latents_saved"] is False
    assert json.loads(save_report)["overwrote_existing"] is False
    assert json.loads(load_report)["source_latents_loaded"] is False
    assert len(list(tmp_path.iterdir())) == 1
    assert not list(tmp_path.glob("*.lock"))
    assert not list(tmp_path.glob("*.tmp"))


def test_spectrum_dataset_file_refuses_overwrite_and_unsafe_names(tmp_path):
    dataset = _dataset()
    save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="safe-name",
        overwrite=False,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        save_spectrum_dataset_file(
            dataset,
            root=tmp_path,
            dataset_name="safe-name",
            overwrite=False,
        )
    _, _, overwrite_report = save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="safe-name",
        overwrite=True,
    )
    assert json.loads(overwrite_report)["overwrote_existing"] is True

    _, _, new_file_report = save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="new-name",
        overwrite=True,
    )
    assert json.loads(new_file_report)["overwrote_existing"] is False
    for unsafe in ("../escape", "sub/path", "..", " space", ""):
        with pytest.raises(ValueError, match="dataset_name"):
            load_spectrum_dataset_file(root=tmp_path, dataset_name=unsafe)


def test_spectrum_dataset_file_detects_metadata_or_tensor_tampering(tmp_path):
    dataset = _dataset()
    save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="valid",
        overwrite=False,
    )
    path = tmp_path / "valid.safetensors"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_spectrum_dataset_file(root=tmp_path, dataset_name="valid")


def test_full_file_fingerprint_is_content_bound_and_rejects_symlink(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"same-size-A")
    second.write_bytes(b"same-size-B")
    assert sha256_file(first).startswith("sha256:")
    assert sha256_file(first) != sha256_file(second)
    symlink = tmp_path / "link.safetensors"
    try:
        symlink.symlink_to(first)
    except OSError:
        return
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(symlink)


def test_dataset_file_fingerprint_tracks_atomic_overwrite(tmp_path):
    missing = spectrum_dataset_file_fingerprint(root=tmp_path, dataset_name="tracked")
    assert missing.startswith("missing:")
    dataset = _dataset()
    save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="tracked",
        overwrite=False,
    )
    first = spectrum_dataset_file_fingerprint(root=tmp_path, dataset_name="tracked")
    changed = dict(dataset)
    changed["batch_ids"] = ["batch-b"]
    save_spectrum_dataset_file(
        changed,
        root=tmp_path,
        dataset_name="tracked",
        overwrite=True,
    )
    second = spectrum_dataset_file_fingerprint(root=tmp_path, dataset_name="tracked")
    assert first != second


def test_dataset_file_node_never_caches_save_and_content_keys_load(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        nodes_speed_advanced.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    save_key = MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced.fingerprint_inputs(
        "save", "node-cache-test"
    )
    assert math.isnan(save_key)
    missing_key = MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced.fingerprint_inputs(
        "load", "node-cache-test"
    )
    root = tmp_path / "h3_speed_spectrum_datasets"
    save_spectrum_dataset_file(
        _dataset(),
        root=root,
        dataset_name="node-cache-test",
        overwrite=False,
    )
    present_key = MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced.fingerprint_inputs(
        "load", "node-cache-test"
    )
    assert missing_key != present_key
