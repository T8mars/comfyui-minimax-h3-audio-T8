"""Read-only contracts for SHA-pinned Chunked v3/v4 real-asset probes."""

from copy import deepcopy

import pytest

from tools import run_modular_chunked_v34_gpu as probe


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_test_copies_keep_source_and_separate_full8_from_high3(variant):
    source_hashes = tuple(probe._sha(probe._path(variant, kind))
                          for kind in ("freeze", "resume"))
    assert source_hashes == probe.CONFIG[variant]["hashes"]

    freeze = probe.build_graph(variant, "freeze", width=128, height=128)
    resume = probe.build_graph(variant, "resume", width=128, height=128)
    spec = probe.CONFIG[variant]
    assert tuple(probe._sha(probe._path(variant, kind))
                 for kind in ("freeze", "resume")) == source_hashes
    assert freeze["9"]["inputs"]["steps"] == 8
    assert freeze["12"]["inputs"]["sampler"] == ["9", 1]
    assert freeze["8"]["inputs"]["length"] == 22
    assert freeze["8"]["inputs"]["width"] == 128
    assert freeze["6"]["inputs"]["width"] == 256
    assert freeze[spec["save"]]["inputs"]["confirm_save"] is True
    assert freeze["200"]["inputs"]["source"] == [spec["save"], 4]
    assert freeze["201"]["inputs"]["source"] == [spec["save"], 5]

    assert spec["low"] not in resume
    scheduler = "30" if variant == "v3" else "40"
    assert resume[scheduler]["inputs"]["steps"] == 3
    assert resume[spec["high"]]["inputs"]["sigmas"] == [scheduler, 0]
    assert resume["14"]["inputs"]["target_width"] == 256
    assert resume["14"]["inputs"]["tile_width"] == 256
    assert resume[spec["relay_conditioning"]]["inputs"]["width"] == 256
    assert resume[spec["relay_plan"]]["inputs"]["length"] == 22
    eav_apply = "35" if variant == "v3" else "45"
    assert resume[spec["high"]]["inputs"]["model"] == [eav_apply, 0]
    assert resume[spec["eav_audit"]]["inputs"]["runtime"] == [eav_apply, 1]
    assert resume[spec["load"]]["inputs"]["checkpoint_path"] == ""
    assert resume["202"]["inputs"]["source"] == [spec["load"], 7]
    assert resume["203"]["inputs"]["source"] == [spec["eav_audit"], 1]


def test_v4_keeps_inherited_mask_and_scales_it_to_test_canvas():
    freeze = probe.build_graph("v4", "freeze", width=128, height=128)
    resume = probe.build_graph("v4", "resume", width=128, height=128)
    for graph in (freeze, resume):
        assert graph["26"]["class_type"] == "LoadImageMask"
        assert graph["26"]["inputs"]["image"] == probe.MASK
        assert graph["24"]["class_type"] == "RepeatImageBatch"
        assert graph["24"]["inputs"]["amount"] == 22
        assert graph["30"]["inputs"]["width"] == 128
        assert graph["30"]["inputs"]["height"] == 128
        assert graph["27"]["class_type"] == "SetLatentNoiseMask"
    assert resume["14"]["class_type"] == (
        "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced")


@pytest.mark.parametrize("variant,kind,width,height", [
    ("v2", "freeze", 128, 128), ("v3", "other", 128, 128),
    ("v4", "resume", 130, 128), ("v4", "freeze", 64, 32),
])
def test_invalid_variant_stage_or_canvas_rejected(variant, kind, width, height):
    with pytest.raises(ValueError):
        probe.build_graph(variant, kind, width=width, height=height)


def test_composed_effect_gate_rejects_missing_real_calls():
    report = {
        "status": "observed_report_only", "completed_forwards": 3,
        "config": {"mode": "report_only"}, "relay_required": True,
        "selector_calls": 150, "relay_attention_calls": 150,
        "clock_match": True, "quality_accepted": False,
        "cache_reuse_authorized": False,
        "forward_plan": {"forwards": [{"attention_blocks": 50}] * 3},
        "feta": {"model_forward_count": 3, "attention_measurement_count": 50},
    }
    assert all(probe._combined_effect_checks(report).values())
    for field, bad in (("completed_forwards", 2), ("relay_attention_calls", 0),
                       ("relay_required", False), ("clock_match", False),
                       ("quality_accepted", True)):
        changed = deepcopy(report)
        changed[field] = bad
        assert not all(probe._combined_effect_checks(changed).values())
    applied = deepcopy(report)
    applied["status"] = "observed_apply_exp"
    applied["config"]["mode"] = "apply_exp"
    applied["feta"]["g_max"] = 1.3
    assert all(probe._combined_effect_checks(applied, mode="apply_exp").values())
    applied["feta"]["g_max"] = 1.0
    assert not all(probe._combined_effect_checks(applied, mode="apply_exp").values())


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_apply_graph_changes_only_high_eav_mode_and_owned_media_name(variant):
    control = probe.build_graph(variant, "resume", width=128, height=128)
    spec = probe.CONFIG[variant]
    receipt = {"path": "owned.safetensors", "manifest_json": '{"x":1}',
               "file_sha256": "A" * 64}
    control[spec["load"]]["inputs"].update(
        checkpoint_path=receipt["path"],
        expected_manifest_json=receipt["manifest_json"],
        expected_file_sha256=receipt["file_sha256"],
    )
    original = deepcopy(control)
    applied = probe.build_apply_graph(variant, control, receipt,
                                      width=128, height=128)
    assert control == original
    assert spec["low"] not in applied
    expected = deepcopy(original)
    expected[spec["eav_config"]]["inputs"]["mode"] = "apply_exp"
    expected[spec["video"]]["inputs"]["filename_prefix"] = (
        f"MiniMaxH3/Chunked_{variant}_Real/eav_apply")
    assert applied == expected
    with pytest.raises(ValueError, match="control graph"):
        probe.build_apply_graph(variant, control, {**receipt, "file_sha256": "B" * 64},
                                width=128, height=128)
    tampered = deepcopy(control)
    tampered[spec["high"]]["inputs"]["cfg"] = 2.0
    with pytest.raises(ValueError, match="control graph"):
        probe.build_apply_graph(variant, tampered, receipt, width=128, height=128)
