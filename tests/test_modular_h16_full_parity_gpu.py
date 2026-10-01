"""The private H16 full-vs-cold pair must not silently cross the resume boundary."""

from copy import deepcopy

import pytest

from tools.run_modular_h16_full_parity_gpu import build_graphs, _shared_nodes
from tools.run_modular_h16_media_gpu import build_probe_graphs


def test_h16_full_parity_has_live_handoff_and_seven_external_effect_windows():
    full, freeze, cold = build_graphs()
    assert len(full) == 93
    assert len(freeze) == 48
    assert len(cold) == 58
    assert "36" not in full
    assert full["15"]["inputs"]["learned_latent"] == ["313", 0]
    assert full["41"]["inputs"]["previous_result"] == ["338", 1]
    assert full["56"]["inputs"]["previous_result"] == ["338", 1]
    assert full["70"]["inputs"]["previous_result"] == ["338", 1]
    assert full["338"]["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
    assert full["388"]["class_type"] == "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
    assert full["389"]["class_type"] == "MiniMaxH3H16WindowSaveEXPT8"
    assert full["250"]["inputs"]["av_latent"] == ["80", 0]
    assert cold["250"] == full["250"]
    assert [full[key]["inputs"]["mode"] for key in
            ("360", "363", "366", "69", "72", "75", "78")] == [
                "report_only"] * 6 + ["apply_exp"]
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in full.values()) == 7
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in cold.values()) == 4
    assert sum(node["class_type"] == "SamplerCustomAdvanced"
               for node in full.values()) == 1
    assert not any(node["class_type"] == "SamplerCustomAdvanced"
                   for node in cold.values())
    assert not any(node["class_type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
                   for node in full.values())
    assert cold["88"]["class_type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
    assert cold["89"]["class_type"] == "MiniMaxH3H16WindowLoadEXPT8"


def test_h16_full_parity_shares_only_identical_dependency_closure():
    freeze, resume = build_probe_graphs("effects")
    shared = _shared_nodes(freeze, resume)
    assert {"1", "2", "3", "4", "5", "6", "18", "27", "51"} == shared
    changed = deepcopy(resume)
    changed["1"]["inputs"]["vae_name"] = "changed.safetensors"
    assert "1" not in _shared_nodes(freeze, changed)


def test_h16_full_parity_pinned_source_rejects_drift(monkeypatch):
    monkeypatch.setattr("tools.run_modular_h16_full_parity_gpu.SOURCE_SHA",
                        {"01_freeze_after_window_2_relay_eav.api.json": "bad"})
    with pytest.raises(ValueError, match="candidate changed"):
        build_graphs()
