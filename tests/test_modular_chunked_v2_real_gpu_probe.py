"""CPU contracts for the owned real-asset Chunked v2 stage probe."""
from copy import deepcopy

import pytest

from tools import run_modular_chunked_v2_gpu as probe


def test_test_copies_keep_source_hashes_and_real_stage_edges():
    before = {path: probe._sha(path) for path in probe.HASHES}
    freeze = probe.build_graph("freeze", width=128, height=128)
    resume = probe.build_graph("resume", width=128, height=128)
    assert before == probe.HASHES == {path: probe._sha(path) for path in probe.HASHES}
    assert freeze["7"]["inputs"]["length"] == resume["7"]["inputs"]["length"] == 22
    assert freeze["7"]["inputs"]["width"] == 128
    assert freeze["6"]["inputs"]["width"] == 256
    assert freeze["35"]["inputs"]["confirm_save"] is True
    assert resume["14"]["inputs"]["target_width"] == 256
    assert resume["14"]["inputs"]["spatial_strategy"] == "full_frame_safe"
    assert resume["30"]["inputs"]["width"] == 256
    assert "12" not in resume
    assert resume["26"]["inputs"]["model"] == ["33", 0]
    assert "204" not in resume  # Relay+EAV uses the composed EAV audit.
    assert resume["35"]["inputs"]["checkpoint_path"] == ""


@pytest.mark.parametrize("kind,width,height", [
    ("other", 128, 128), ("freeze", 96, 63), ("resume", 130, 128),
])
def test_invalid_probe_geometry_or_stage_rejected(kind, width, height):
    with pytest.raises(ValueError):
        probe.build_graph(kind, width=width, height=height)


def test_checkpoint_receipt_population_does_not_restore_low_stage():
    resume = probe.build_graph("resume", width=128, height=128)
    original = deepcopy(resume)
    resume["35"]["inputs"].update(
        checkpoint_path="owned-first.h3latent.safetensors",
        expected_manifest_json='{"schema":"owned"}',
        expected_file_sha256="A" * 64,
    )
    assert "12" not in resume
    assert resume["26"] == original["26"]
    assert resume["35"]["inputs"]["expected_file_sha256"] == "A" * 64
    assert original["35"]["inputs"]["checkpoint_path"] == ""


def test_combined_effect_gate_requires_real_eav_and_relay_calls():
    report = {
        "status": "observed_report_only", "completed_forwards": 4,
        "config": {"mode": "report_only"},
        "relay_required": True, "selector_calls": 200,
        "relay_attention_calls": 200, "clock_match": True,
        "quality_accepted": False, "cache_reuse_authorized": False,
        "forward_plan": {"forwards": [{"attention_blocks": 50}] * 4},
        "feta": {"model_forward_count": 4, "attention_measurement_count": 100,
                 "g_max": 1.3},
    }
    assert all(probe._combined_effect_checks(report).values())
    for field, value in (("completed_forwards", 0), ("relay_attention_calls", 0),
                         ("relay_required", False), ("clock_match", False),
                         ("quality_accepted", True)):
        invalid = deepcopy(report)
        invalid[field] = value
        assert not all(probe._combined_effect_checks(invalid).values())
    applied = deepcopy(report)
    applied["status"] = "observed_apply_exp"
    applied["config"]["mode"] = "apply_exp"
    assert all(probe._combined_effect_checks(applied, mode="apply_exp").values())
    applied["feta"]["g_max"] = 1.0
    assert not all(probe._combined_effect_checks(applied, mode="apply_exp").values())
