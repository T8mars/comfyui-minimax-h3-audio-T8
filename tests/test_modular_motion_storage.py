"""Motion cold pass-2 source must be explicit and leave old graphs untouched."""
import hashlib
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import motion_storage_nodes
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import save_native_h3_av_checkpoint
from tools.build_modular_motion_storage_workflows import (
    LOAD, SAVE, SOURCE, SOURCE_SHA, freeze_graph, resume_graph,
)
from tools.run_modular_motion_storage_gpu import CONTINUOUS_RGB_PCM, probe_graphs


@pytest.mark.parametrize("variant", ["Fullclip", "Windowed"])
def test_freeze_and_cold_graphs_keep_exact_source_and_only_one_sampling_stage(variant):
    path = SOURCE / f"Motion_Recovery_{variant}_Relay_EAV_Separate_Pass2_EXP.json"
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_SHA[variant]
    frozen, freeze_api = freeze_graph(variant)
    cold, cold_api = resume_graph(variant)
    assert path.read_bytes() == original
    assert len(frozen["nodes"]) == len(freeze_api) == 10
    assert sum(node["class_type"] == SAVE for node in freeze_api.values()) == 1
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in freeze_api.values()) == 1
    assert not any(node["class_type"] == "MiniMaxH3StageSamplerEXPT8"
                   for node in freeze_api.values())
    assert "9" not in cold_api
    assert not any(node["class_type"] == "SamplerCustomAdvanced" for node in cold_api.values())
    assert sum(node["class_type"] == LOAD for node in cold_api.values()) == 1
    assert sum(node["class_type"] == "MiniMaxH3StageSamplerEXPT8" for node in cold_api.values()) == 1
    assert sum(node["class_type"] == "MiniMaxH3StageEAVApplyEXPT8" for node in cold_api.values()) == 1
    assert sum(node["class_type"] == "MiniMaxH3MotionRelayBindEXPT8" for node in cold_api.values()) == 1
    assert sum(node["class_type"] == "MiniMaxH3MotionStageAuditEXPT8" for node in cold_api.values()) == 1
    load = next(node for node in cold_api.values() if node["class_type"] == LOAD)
    source = json.loads(
        (SOURCE / f"Motion_Recovery_{variant}_Relay_EAV_Separate_Pass2_EXP.api.json").read_bytes())["5"]["inputs"]
    assert load["inputs"]["checkpoint_path"] == ""
    assert load["inputs"]["expected_manifest_json"] == ""
    assert load["inputs"]["expected_file_sha256"] == ""
    assert (load["inputs"]["expected_frame_count"], load["inputs"]["expected_width"],
            load["inputs"]["expected_height"]) == (
                source["length"], source["width"], source["height"])
    assert len(cold["nodes"]) == len(cold_api)


def _av():
    return {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 7, 4, 8), torch.zeros(1, 32, 2, 37),
    ))}


@pytest.mark.parametrize("variant", ["Fullclip", "Windowed"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_cold_probe_effect_mode_preserves_exact_frozen_firstpass(variant, mode):
    freeze, cold = probe_graphs(variant, eav_mode=mode)
    eav_id = "36" if variant == "Windowed" else "34"
    assert freeze["5"]["inputs"]["length"] == 22
    assert freeze["39" if variant == "Windowed" else "37"]["inputs"]["confirm_save"] is True
    assert cold[eav_id]["inputs"]["mode"] == mode
    assert cold[eav_id]["inputs"]["tau"] == (10. if mode == "apply_exp" else .2)
    assert "9" not in cold
    assert len(CONTINUOUS_RGB_PCM[variant][mode]) == 2


def test_cold_load_demands_external_receipt_identity_and_geometry(tmp_path, monkeypatch):
    store = tmp_path / "MiniMaxH3" / "latent_checkpoints"
    monkeypatch.setattr(motion_storage_nodes, "native_h3_checkpoint_storage_root", lambda: store)
    saved = save_native_h3_av_checkpoint(
        _av(), store, filename_prefix="motion/frozen", checkpoint_id="motion_recovery_firstpass",
        confirm_save=True, verify_after_write=True)
    args = (saved[2], saved[4], saved[3], 22, 128, 64, 8)
    result = motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*args).result
    assert result[0]["samples"].unbind()[0].shape == (1, 24, 7, 4, 8)
    assert json.loads(result[1])["status"] == "MATCH_EXTERNAL"
    assert saved[3] in motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.fingerprint_inputs(*args)
    bad = list(args)
    bad[2] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*bad)
    bad = list(args)
    bad[1] = ""
    with pytest.raises(ValueError, match="external first-pass manifest"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*bad)
    bad = list(args)
    wrong_manifest = json.loads(bad[1])
    wrong_manifest["checkpoint_id"] = "another_motion_source"
    bad[1] = json.dumps(wrong_manifest)
    with pytest.raises(ValueError, match="manifest|mismatch"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*bad)
    bad = list(args)
    bad[3] = 39
    with pytest.raises(ValueError, match="expected AV geometry"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*bad)
    bad = list(args)
    bad[4] = 160
    with pytest.raises(ValueError, match="expected AV geometry"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*bad)
    wrong = save_native_h3_av_checkpoint(
        _av(), store, filename_prefix="motion/wrong", checkpoint_id="audio_refine_firstpass",
        confirm_save=True, verify_after_write=True)
    with pytest.raises(ValueError, match="receipt and ID"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(
            wrong[2], wrong[4], wrong[3], 22, 128, 64, 8)


def test_cold_load_rejects_missing_traversal_and_truncated_checkpoint(tmp_path, monkeypatch):
    store = tmp_path / "MiniMaxH3" / "latent_checkpoints"
    monkeypatch.setattr(motion_storage_nodes, "native_h3_checkpoint_storage_root", lambda: store)
    saved = save_native_h3_av_checkpoint(
        _av(), store, filename_prefix="motion/frozen", checkpoint_id="motion_recovery_firstpass",
        confirm_save=True, verify_after_write=True)
    args = [saved[2], saved[4], saved[3], 22, 128, 64, 8]
    missing = args.copy()
    missing[0] = "motion/missing.h3latent.safetensors"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*missing)
    traversal = args.copy()
    traversal[0] = "../outside.h3latent.safetensors"
    with pytest.raises(ValueError, match="traversal"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*traversal)
    source = store / saved[2]
    truncated = source.with_name("truncated.h3latent.safetensors")
    truncated.write_bytes(source.read_bytes()[:-1])
    damaged = args.copy()
    damaged[0] = truncated.relative_to(store).as_posix()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        motion_storage_nodes.MiniMaxH3MotionFrozenFirstPassLoadEXPT8.execute(*damaged)
