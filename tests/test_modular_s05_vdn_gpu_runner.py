"""Guard the saved real-asset S05 probe graphs without starting a Core."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

from tools import run_modular_s05_vdn_gpu as full
from tools import run_modular_s05_vdn_resume_gpu as resume


def test_full_and_cold_graph_keep_independent_relay_and_high_only_boundary():
    graph, full_sha = full.build_probe_graph(width=128, height=64, frames=22)
    receipt = {"path": "VDNRelay/stage_dmd_8nfe/native4/LOW-fixed/manifest.json",
               "sha256": "a" * 64}
    cold, cold_sha = resume.build_resume_graph(receipt)
    assert len(full_sha) == len(cold_sha) == 64
    assert graph["40"]["inputs"]["length"] == graph["140"]["inputs"]["length"] == 22
    assert cold["140"]["inputs"]["length"] == 22
    assert graph["41"]["inputs"] == graph["42"]["inputs"]
    assert cold["42"]["inputs"] == graph["42"]["inputs"]
    assert graph["110"]["inputs"]["mode"] == "apply_exp"
    assert graph["23"]["inputs"]["av_latent"] == ["112", 0]
    assert cold["23"]["inputs"]["av_latent"] == ["60", 0]
    assert cold["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert cold["60"]["inputs"]["artifact_sha256"] == receipt["sha256"]
    assert graph["22"]["class_type"] == cold["22"]["class_type"] == "UNETLoader"
    assert "13" not in cold and "50" not in cold and "110" not in cold


def _phase(order, progress):
    return {"terminal": {"type": "execution_success"},
            "events": ([{"type": "executing", "node": node} for node in order] +
                       [{"type": "progress", "node": node} for node in progress])}


def test_stage_audits_require_vdn8_then_native4_and_no_low_in_cold():
    complete = _phase(["13", "50", "112", "23", "22", "29", "51"],
                      ["13"] * 8 + ["29"] * 4)
    assert all(full.stage_checks(complete).values())
    cold = _phase(["60", "23", "22", "29", "51"], ["29"] * 4)
    assert all(resume.stage_checks(cold).values())
    bad_order = deepcopy(complete)
    bad_order["events"][1], bad_order["events"][2] = bad_order["events"][2], bad_order["events"][1]
    assert full.stage_checks(bad_order)["relay_audit_handoff_and_independent_model_order"] is False
    cold["events"].insert(3, {"type": "executing", "node": "13"})
    assert resume.stage_checks(cold)["no_vdn_low_execution"] is False


@pytest.mark.parametrize("refine", (3, 4, 5))
def test_saved_dmd8_native_refine_pair_keeps_its_exact_table_and_cold_boundary(refine):
    graph, full_sha = full.build_probe_graph(width=128, height=64, frames=22,
                                              refine=refine)
    receipt = {"path": f"VDNRelay/stage_dmd_8nfe/native{refine}/LOW-fixed/manifest.json",
               "sha256": "a" * 64}
    cold, cold_sha = resume.build_resume_graph(receipt, refine)
    assert len(full_sha) == len(cold_sha) == 64
    assert graph["93"]["inputs"]["refine_steps"] == cold["93"]["inputs"]["refine_steps"] == refine
    assert graph["50"]["inputs"]["prefix"].endswith(f"native{refine}/LOW")
    assert graph["51"]["inputs"]["prefix"].endswith(f"native{refine}/HIGH")
    assert graph["40"]["inputs"]["length"] == graph["140"]["inputs"]["length"] == 22
    assert cold["140"]["inputs"]["length"] == 22
    assert graph["23"]["inputs"]["av_latent"] == ["112", 0]
    assert cold["23"]["inputs"]["av_latent"] == ["60", 0]
    assert cold["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert "13" not in cold and "50" not in cold and "110" not in cold
    complete = _phase(["13", "50", "112", "23", "22", "29", "51"],
                      ["13"] * 8 + ["29"] * refine)
    cold_phase = _phase(["60", "23", "22", "29", "51"], ["29"] * refine)
    assert all(full.stage_checks(complete, refine).values())
    assert all(resume.stage_checks(cold_phase, refine).values())


def test_unknown_native_refine_is_rejected_before_loading_a_candidate():
    with pytest.raises(ValueError, match="3, 4 or 5"):
        full.candidate(6)


@pytest.mark.parametrize("refine", (3, 4, 5))
def test_saved_b50_native_refine_pair_keeps_complete_fifty_step_low(refine):
    training = "stage_b_50nfe"
    graph, full_sha = full.build_probe_graph(width=128, height=64, frames=22,
                                              refine=refine, training=training)
    receipt = {"path": f"VDNRelay/{training}/native{refine}/LOW-fixed/manifest.json",
               "sha256": "a" * 64}
    resumed, cold_sha = resume.build_resume_graph(receipt, refine, training)
    assert len(full_sha) == len(cold_sha) == 64
    assert graph["70"]["inputs"]["stage"] == training
    assert graph["50"]["inputs"]["prefix"] == f"VDNRelay/{training}/native{refine}/LOW"
    assert graph["51"]["inputs"]["prefix"] == f"VDNRelay/{training}/native{refine}/HIGH"
    assert graph["93"]["inputs"]["refine_steps"] == \
        resumed["93"]["inputs"]["refine_steps"] == refine
    assert graph["40"]["inputs"]["length"] == graph["140"]["inputs"]["length"] == 22
    assert resumed["23"]["inputs"]["av_latent"] == ["60", 0]
    assert resumed["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert "13" not in resumed and "50" not in resumed and "70" not in resumed
    assert full.schema(refine, training).startswith("t8.modular-sampling.s03-vdn-b50-")
    assert resume.schema(refine, training).startswith("t8.modular-sampling.s03-vdn-b50-")
    complete_phase = _phase(["13", "50", "112", "23", "22", "29", "51"],
                            ["13"] * 50 + ["29"] * refine)
    assert all(full.stage_checks(complete_phase, refine, training).values())
    missing = deepcopy(complete_phase)
    missing["events"].pop(7)
    assert not full.stage_checks(missing, refine, training)["b50_then_native_high_steps"]
    cold_phase = _phase(["60", "23", "22", "29", "51"], ["29"] * refine)
    assert all(resume.stage_checks(cold_phase, refine).values())


def test_unknown_vdn_training_is_rejected_before_loading_a_candidate():
    with pytest.raises(ValueError, match="DMD8 or B50"):
        full.candidate(4, "unknown")


@pytest.mark.parametrize("training", full.TRAINING)
def test_cold_high_preflight_needs_high_assets_but_not_vdn_low_weights(
        tmp_path, monkeypatch, training):
    args = Namespace(comfy_root=tmp_path, port=8869, min_free_vram_mib=1,
                     min_free_ram_mib=0, refine=4, training=training)
    monkeypatch.setattr(full.shared, "gpu_memory_mib",
                        lambda: {"available": True, "free_mib": 16000})
    monkeypatch.setattr(full.shared, "port_is_listening", lambda *_: False)
    high = full.preflight(args, "a" * 64, include_low_assets=False)
    assert "vdn_linear_branch" not in high["assets"]
    assert "vdn_default_adapter" not in high["assets"]
    for asset in high["assets"].values():
        path = Path(asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert full.preflight(args, "a" * 64, include_low_assets=False)["ready"]
    complete = full.preflight(args, "a" * 64)
    assert complete["checks"]["all_assets_installed"] is False
    assert complete["ready"] is False


@pytest.mark.parametrize("training", full.TRAINING)
def test_resume_graph_rejects_other_training_low_receipt(training):
    other = next(item for item in full.TRAINING if item != training)
    receipt = {"path": f"VDNRelay/{other}/native4/LOW-fixed/manifest.json",
               "sha256": "a" * 64}
    with pytest.raises(ValueError, match="does not match"):
        resume.build_resume_graph(receipt, 4, training)
    receipt["path"] = f"VDNRelay/{training}/native3/LOW-fixed/manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        resume.build_resume_graph(receipt, 4, training)
    receipt["path"] = f"VDNRelay/{training}/native4/LOW-fixed/manifest.json"
    receipt["sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="does not match"):
        resume.build_resume_graph(receipt, 4, training)
