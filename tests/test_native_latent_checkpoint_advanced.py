from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg import native_latent_checkpoint_advanced as checkpoint
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import (
    CHECKPOINT_EXTENSION,
    CHECKPOINT_METADATA_KEY,
    fingerprint_native_h3_checkpoint_file,
    load_native_h3_av_checkpoint,
    resolve_native_h3_checkpoint_path,
    save_native_h3_av_checkpoint,
)
from h3_audio_t8_pkg.native_latent_timeline_advanced import (
    audit_native_h3_av_latent_resume_manifest,
)
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)


def _latent(*, frames=22, mask=True):
    video_t = ((frames - 5) // 17) * 5 + 2
    audio_t = round(frames / 24 * 40)
    video = torch.arange(24 * video_t * 3 * 4, dtype=torch.float32).reshape(
        1, 24, video_t, 3, 4
    ).to(torch.float16)
    audio = torch.linspace(-1, 1, 32 * 2 * audio_t, dtype=torch.float32).reshape(
        1, 32, 2, audio_t
    ).to(torch.bfloat16)
    result = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "source_metadata": {
            "shot": 7,
            "enabled": True,
            "gain": 0.25,
            "name": "镜头A",
            "blob": b"\x00H3\xff",
            "list": [1, None, "x"],
            "tuple": (2, False),
        },
        "metadata_tensor": torch.tensor([3, 5, 8], dtype=torch.int32),
        "metadata_nested": comfy.nested_tensor.NestedTensor(
            (
                torch.tensor([1.0, 2.0], dtype=torch.float16),
                torch.tensor([3.0], dtype=torch.float32),
            )
        ),
    }
    if mask:
        result["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (
                torch.linspace(0, 1, video.numel(), dtype=torch.float32)
                .reshape(video.shape)
                .to(torch.float16),
                torch.linspace(1, 0, audio.numel(), dtype=torch.float32)
                .reshape(audio.shape)
                .to(torch.bfloat16),
            )
        )
    return result


def _parts(latent):
    return tuple(latent["samples"].unbind())


def test_confirm_false_computes_manifest_without_creating_store(tmp_path):
    store = tmp_path / "not-created"
    latent = _latent()
    result = save_native_h3_av_checkpoint(
        latent,
        store,
        filename_prefix="shots/shot_a",
        checkpoint_id=" shot_a ",
        confirm_save=False,
    )
    passthrough, status, path, file_sha, manifest_json, report_json = result
    assert passthrough is latent
    assert status == "NOT_SAVED" and path == "" and file_sha == ""
    assert json.loads(manifest_json)["checkpoint_id"] == "shot_a"
    assert json.loads(report_json)["checkpoint_id"] == "shot_a"
    assert json.loads(report_json)["files_written"] is False
    assert not store.exists()


def test_exact_roundtrip_preserves_av_masks_dtypes_and_metadata(tmp_path):
    store = tmp_path / "store"
    source = _latent()
    original_video, original_audio = _parts(source)
    result = save_native_h3_av_checkpoint(
        source,
        store,
        filename_prefix="shots/shot_a",
        checkpoint_id=" shot_a ",
        confirm_save=True,
        verify_after_write=True,
        hash_chunk_megabytes=1,
    )
    passthrough, status, relative, file_sha, manifest_json, report_json = result
    assert passthrough is source
    assert status == "SAVED_VERIFIED"
    assert relative.startswith("shots/shot_a_") and relative.endswith(CHECKPOINT_EXTENSION)
    assert len(file_sha) == 64 and file_sha == file_sha.upper()
    assert json.loads(report_json)["directory_fsync"] is False
    assert json.loads(report_json)["checkpoint_id"] == "shot_a"
    assert len(list(store.rglob("*.tmp"))) == 0

    loaded = load_native_h3_av_checkpoint(
        store,
        relative,
        expected_manifest_json=manifest_json,
        expected_file_sha256=file_sha,
        hash_chunk_megabytes=8,
    )
    latent, load_status, verified, checkpoint_id, digest, loaded_file_sha, authoritative, report = (
        loaded
    )
    assert load_status == "MATCH_EXTERNAL" and verified is True
    assert checkpoint_id == "shot_a"
    assert loaded_file_sha == file_sha
    assert digest == json.loads(manifest_json)["content_sha256"]
    assert json.loads(authoritative)["checkpoint_id"] == "shot_a"
    assert json.loads(report)["loaded_device"] == "cpu"
    video, audio = _parts(latent)
    assert video.dtype == torch.float16 and audio.dtype == torch.bfloat16
    assert torch.equal(video, original_video.cpu())
    assert torch.equal(audio, original_audio.cpu())
    source_masks = tuple(source["noise_mask"].unbind())
    loaded_masks = tuple(latent["noise_mask"].unbind())
    assert all(torch.equal(left.cpu(), right) for left, right in zip(source_masks, loaded_masks))
    assert latent["source_metadata"] == source["source_metadata"]
    assert torch.equal(latent["metadata_tensor"], source["metadata_tensor"])
    nested_items = tuple(latent["metadata_nested"].unbind())
    assert nested_items[0].dtype == torch.float16
    assert nested_items[1].dtype == torch.float32
    assert "t8_native_latent_checkpoint" in latent
    # The load report is explicitly volatile and cannot change checkpoint content identity.
    matched = audit_native_h3_av_latent_resume_manifest(
        latent,
        checkpoint_id="shot_a",
        expected_manifest_json=manifest_json,
        hash_chunk_megabytes=1,
    )
    assert matched[0] == "MATCH" and matched[1] is True
    assert fingerprint_native_h3_checkpoint_file(store, relative) == file_sha


def test_self_verified_load_and_external_sha_or_manifest_mismatch_fail_closed(tmp_path):
    store = tmp_path / "store"
    result = save_native_h3_av_checkpoint(
        _latent(mask=False),
        store,
        checkpoint_id="self_only",
        confirm_save=True,
    )
    relative, file_sha, manifest_json = result[2], result[3], result[4]
    loaded = load_native_h3_av_checkpoint(store, relative)
    assert loaded[1:4] == ("SELF_VERIFIED", True, "self_only")

    wrong_sha = ("0" if file_sha[0] != "0" else "1") + file_sha[1:]
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_native_h3_av_checkpoint(store, relative, expected_file_sha256=wrong_sha)

    wrong_manifest = json.loads(manifest_json)
    wrong_manifest["checkpoint_id"] = "another_checkpoint"
    with pytest.raises(ValueError, match="checkpoint_id"):
        load_native_h3_av_checkpoint(
            store,
            relative,
            expected_manifest_json=json.dumps(wrong_manifest),
        )


def test_valid_safetensors_payload_tamper_is_detected_by_embedded_manifest(tmp_path):
    store = tmp_path / "store"
    saved = save_native_h3_av_checkpoint(
        _latent(mask=False), store, checkpoint_id="tamper", confirm_save=True
    )
    original, _relative = resolve_native_h3_checkpoint_path(store, saved[2])
    with safe_open(str(original), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
        metadata = dict(handle.metadata() or {})
    tensors["samples_video"][0, 0, 0, 0, 0] += 1
    tampered = store / f"tampered{CHECKPOINT_EXTENSION}"
    save_file(tensors, str(tampered), metadata=metadata)
    assert CHECKPOINT_METADATA_KEY in metadata
    with pytest.raises(ValueError, match="content_sha256"):
        load_native_h3_av_checkpoint(store, tampered.name)


def test_paths_are_bounded_and_symlinks_are_rejected(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        resolve_native_h3_checkpoint_path(store, "../escape.h3latent.safetensors")
    with pytest.raises(ValueError, match="relative"):
        resolve_native_h3_checkpoint_path(store, str((tmp_path / "absolute").resolve()))
    with pytest.raises(ValueError, match="reserved Windows filename"):
        save_native_h3_av_checkpoint(
            _latent(mask=False),
            store,
            filename_prefix="CON",
            confirm_save=True,
        )
    with pytest.raises(ValueError, match="traversal"):
        save_native_h3_av_checkpoint(
            _latent(mask=False),
            store,
            filename_prefix="../escape",
            confirm_save=True,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    link = store / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this platform: {exc}")
    with pytest.raises(ValueError, match="symbolic links"):
        save_native_h3_av_checkpoint(
            _latent(mask=False),
            store,
            filename_prefix="linked/escape",
            confirm_save=True,
        )


def test_write_failure_removes_temporary_and_never_exposes_checkpoint(tmp_path, monkeypatch):
    store = tmp_path / "store"

    def fail_write(*_args, **_kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(checkpoint, "save_file", fail_write)
    with pytest.raises(OSError, match="injected write failure"):
        save_native_h3_av_checkpoint(
            _latent(mask=False),
            store,
            filename_prefix="fault/shot",
            confirm_save=True,
        )
    assert list(store.rglob(f"*{CHECKPOINT_EXTENSION}")) == []
    assert list(store.rglob("*.tmp")) == []


def test_node_schemas_append_safe_defaults_and_load_fingerprint_tracks_file(tmp_path, monkeypatch):
    save_schema = MiniMaxH3NativeLatentCheckpointSaveT8Advanced.define_schema()
    load_schema = MiniMaxH3NativeLatentCheckpointLoadT8Advanced.define_schema()
    save_inputs = {item.id: item for item in save_schema.inputs}
    assert save_schema.is_experimental is True and save_schema.is_output_node is True
    assert save_inputs["confirm_save"].default is False
    assert save_inputs["verify_after_write"].default is True
    assert load_schema.is_experimental is True

    import h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced as nodes_checkpoint

    monkeypatch.setattr(
        nodes_checkpoint.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    saved = save_native_h3_av_checkpoint(
        _latent(mask=False),
        tmp_path / "MiniMaxH3" / "latent_checkpoints",
        filename_prefix="fingerprint",
        checkpoint_id="fingerprint",
        confirm_save=True,
    )
    relative = saved[2]
    fingerprint = MiniMaxH3NativeLatentCheckpointLoadT8Advanced.fingerprint_inputs(
        relative, "", "", 8
    )
    assert saved[3] in fingerprint


def test_frontend_checkpoint_workflow_has_both_nodes_and_multiple_notes():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-23_H3_Native_Latent_Checkpoint_Save_Load_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert "MiniMaxH3NativeLatentCheckpointSaveT8Advanced" in by_type
    assert "MiniMaxH3NativeLatentCheckpointLoadT8Advanced" in by_type
    save_node = by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"]
    assert save_node["widgets_values"][2] is False
    assert save_node["widgets_values"][3] is True
    notes = [node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]
    assert len(notes) >= 5
    combined = "\n".join(str(node["widgets_values"][0]) for node in notes)
    assert "不会恢复某个NFE中间步" in combined
    assert "output/MiniMaxH3/latent_checkpoints" in combined


def test_completed_save_reloads_exactly_in_a_new_python_process(tmp_path):
    worker = Path(__file__).with_name("multiprocess_native_latent_checkpoint_worker.py")
    store = tmp_path / "store"
    handoff = tmp_path / "handoff.json"
    env = os.environ.copy()
    comfy_root = Path(__file__).resolve().parents[3]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(comfy_root) + (os.pathsep + existing if existing else "")

    save_process = subprocess.run(
        [sys.executable, str(worker), "save", str(store), str(handoff)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    load_process = subprocess.run(
        [sys.executable, str(worker), "load", str(store), str(handoff)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    saved = json.loads(save_process.stdout.strip().splitlines()[-1])
    loaded = json.loads(load_process.stdout.strip().splitlines()[-1])
    assert saved["status"] == "SAVED_VERIFIED"
    assert loaded["status"] == "MATCH_EXTERNAL"
    assert loaded["resume_verified"] is True
    assert loaded["exact_tensor_metadata_match"] is True
    assert loaded["save_pid"] == saved["save_pid"]
    assert loaded["load_pid"] != loaded["save_pid"]
    assert loaded["file_sha256"] == saved["file_sha256"]
