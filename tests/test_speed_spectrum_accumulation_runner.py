from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.tools.run_h3_speed_spectrum_accumulation import (
    MANIFEST_SCHEMA,
    build_accumulation_prompt,
    build_preprocess_command,
    existing_dataset_batch_ids,
    validate_manifest,
)


def _manifest():
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_name": "research-v1",
        "task_family": "I2VA",
        "video_vae_name": "h3-video-vae.safetensors",
        "checkpoint_fingerprint": "sha256:model",
        "vae_fingerprint": "sha256:vae",
        "entries": [{"file": "folder/a.mp4", "batch_id": "a"}],
    }


def test_first_prompt_creates_without_load_or_overwrite():
    prompt = build_accumulation_prompt(
        input_file="folder/a.mp4",
        batch_id="a",
        task_family="I2VA",
        dataset_name="research-v1",
        video_vae_name="h3-video-vae.safetensors",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        append_existing=False,
    )
    assert "6" not in prompt
    assert "previous_dataset" not in prompt["7"]["inputs"]
    assert prompt["8"]["inputs"]["confirm_write"] is True
    assert prompt["8"]["inputs"]["overwrite"] is False
    assert prompt["3"]["inputs"]["frames"] == ["2", 0]
    assert prompt["3"]["inputs"]["source_fps"] == ["2", 2]
    assert prompt["7"]["inputs"]["dataset_provenance_json"] == ""
    assert prompt["7"]["inputs"]["source_entry_json"] == ""


def test_append_prompt_loads_exact_dataset_before_atomic_overwrite():
    prompt = build_accumulation_prompt(
        input_file="folder/b.mp4",
        batch_id="b",
        task_family="I2VA",
        dataset_name="research-v1",
        video_vae_name="h3-video-vae.safetensors",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        append_existing=True,
    )
    assert prompt["6"]["inputs"] == {
        "mode": "load",
        "dataset_name": "research-v1",
        "overwrite": False,
        "confirm_write": False,
    }
    assert prompt["7"]["inputs"]["previous_dataset"] == ["6", 0]
    assert prompt["8"]["inputs"]["overwrite"] is True


def test_manifest_requires_unique_entries_and_complete_provenance():
    assert validate_manifest(_manifest())["task_family"] == "I2VA"
    duplicate = _manifest()
    duplicate["entries"].append({"file": "folder/b.mp4", "batch_id": "a"})
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(duplicate)
    missing = _manifest()
    missing.pop("vae_fingerprint")
    with pytest.raises(ValueError, match="vae_fingerprint"):
        validate_manifest(missing)


def test_prompt_and_manifest_bind_dataset_and_source_provenance():
    provenance = {"schema": "minimax_h3_speed_dataset_provenance_t8_v1"}
    source_entry = {
        "batch_id": "a",
        "source_file_sha256": "A" * 64,
        "decoded_window_sha256": "B" * 64,
    }
    prompt = build_accumulation_prompt(
        input_file="folder/a.mp4",
        batch_id="a",
        task_family="T2VA",
        dataset_name="formal-v1",
        video_vae_name="h3-video-vae.safetensors",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        append_existing=False,
        dataset_provenance=provenance,
        source_entry=source_entry,
    )
    assert json.loads(prompt["7"]["inputs"]["dataset_provenance_json"]) == provenance
    assert json.loads(prompt["7"]["inputs"]["source_entry_json"]) == source_entry

    manifest = _manifest()
    manifest["provenance"] = {"formal_dataset_authorized": True}
    manifest["dataset_provenance"] = provenance
    manifest["entries"][0]["source_entry"] = source_entry
    assert validate_manifest(manifest)["dataset_provenance"] == provenance
    manifest["entries"][0].pop("source_entry")
    with pytest.raises(ValueError, match="needs source_entry"):
        validate_manifest(manifest)


def test_preprocess_command_is_bounded_aspect_preserving_and_lossless(tmp_path: Path):
    command = build_preprocess_command(
        ffmpeg="ffmpeg",
        source=tmp_path / "source.mp4",
        temporary=tmp_path / "window.tmp.mkv",
        width=736,
        height=416,
        length=124,
    )
    filter_value = command[command.index("-vf") + 1]
    assert "fps=24" in filter_value
    assert "force_original_aspect_ratio=increase" in filter_value
    assert "crop=736:416" in filter_value
    assert "end_frame=124" in filter_value
    assert command[command.index("-frames:v") + 1] == "124"
    assert command[command.index("-c:v") + 1] == "ffv1"


def test_existing_dataset_batch_ids_reads_only_public_metadata(tmp_path: Path):
    path = tmp_path / "dataset.safetensors"
    dataset = {"batch_ids": ["a", "b"]}
    save_file(
        {"power_sum": torch.ones(2, 2)},
        path,
        metadata={
            "storage_schema": "minimax_h3_speed_spectrum_dataset_file_t8_v1",
            "dataset_json": json.dumps(dataset),
        },
    )
    assert existing_dataset_batch_ids(path) == {"a", "b"}
