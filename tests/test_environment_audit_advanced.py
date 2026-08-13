from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.environment_audit import (
    ENVIRONMENT_AUDIT_SCHEMA,
    audit_h3_environment,
    blocking_summary,
    collect_environment_snapshot,
)
from h3_audio_t8_pkg.nodes_environment_audit_advanced import (
    MiniMaxH3EnvironmentAuditT8Advanced,
)


def _capabilities(**overrides):
    values = {
        "native_model_sampling_av": {"state": "supported"},
        "diffusion_model_wrapper": {"state": "supported"},
        "dit_double_block_replace": {"state": "supported"},
        "video_vae_internal_temporal_chunking": {"state": "supported"},
        "video_vae_generic_chunked_io": {"state": "supported"},
        "tiled_decode_nested_tensor_fix": {"state": "supported"},
        "tiled_decode_global_coordinates": {"state": "supported"},
        "audio_vae_full_offload_fix": {"state": "supported"},
        "attention_peak_clone_fix": {"state": "supported"},
        "native_arbitrary_guides": {"state": "supported"},
        "per_token_h3_latent_masks": {"state": "supported"},
        "h3_attention_patch_hooks": {"state": "supported"},
    }
    for key, state in overrides.items():
        values[key] = {"state": state}
    return values


def _snapshot(*, capabilities=None, free_mib=4096.0, packed_owner=True):
    return {
        "git": {"available": True, "head": "abc", "known_commit_ancestry": {}},
        "capabilities": capabilities or _capabilities(),
        "runtime": {
            "gpu": {"whole_device_free_mib": free_mib},
            "comfy": {"dynamic_vram_enabled": True},
            "host": {"ram_available_gib": 64.0, "commit_headroom_gib": 64.0},
        },
        "model_patch_stack": {
            "connected": True,
            "packed_layout_owner": "comfy.ldm.minimax.model.PackedLayout.__init__",
            "packed_layout_expected_owner": packed_owner,
            "patch_replace_groups": {},
        },
        "conditioning": {"connected": False, "reference_kinds": []},
        "loaded_models": {
            "available": True,
            "count": 1,
            "currently_used_count": 1,
            "total_model_mib": 2048.0,
            "total_loaded_mib": 1024.0,
            "models": [],
        },
    }


def _audit(snapshot, **overrides):
    values = {
        "workload_profile": "t2va",
        "width": 736,
        "height": 416,
        "length": 124,
        "model_family": "fl2va",
        "model_precision": "int8_convrot",
        "attention_backend": "stock",
        "cache_backend": "none",
        "decode_mode": "regular",
        "dynamic_vram_mode": "auto_detect",
        "reference_media_count": 0,
        "middle_keyframe_count": 0,
        "minimum_current_headroom_mib": 512.0,
        "snapshot": snapshot,
    }
    values.update(overrides)
    return audit_h3_environment(**values)


def test_environment_audit_passes_a_known_small_contract_without_claiming_safety():
    report = _audit(_snapshot())
    assert report["schema"] == ENVIRONMENT_AUDIT_SCHEMA
    assert report["status"] == "pass"
    assert report["no_known_blocker"] is True
    assert report["memory_safe_claim"] is False
    assert report["quality_safe_claim"] is False
    assert report["estimated_packed_rows"]["target_video_rows"] > 0


def test_environment_audit_keeps_unknown_separate_from_supported():
    report = _audit(
        _snapshot(),
        model_family="auto_unknown",
        model_precision="auto_unknown",
    )
    assert report["status"] == "unknown"
    assert report["no_known_blocker"] is True
    codes = {item["code"] for item in report["issues"]["unknown"]}
    assert {"model_family_unknown", "model_precision_unknown"} <= codes


def test_environment_audit_blocks_invalid_grid_and_foreign_global_layout_patch():
    report = _audit(_snapshot(packed_owner=False), width=750, length=125)
    assert report["status"] == "blocked"
    assert report["no_known_blocker"] is False
    codes = {item["code"] for item in report["issues"]["hard"]}
    assert {
        "canvas_not_multiple_of_32",
        "frame_count_off_h3_grid",
        "global_packed_layout_patch_detected",
    } <= codes


def test_environment_audit_flags_unfixed_high_resolution_tiled_decode():
    capabilities = _capabilities(
        tiled_decode_global_coordinates="unsupported",
        tiled_decode_nested_tensor_fix="unsupported",
    )
    report = _audit(
        _snapshot(capabilities=capabilities),
        width=1920,
        height=1088,
        decode_mode="tiled",
    )
    assert report["status"] == "high_risk"
    assert report["no_known_blocker"] is False
    assert any(
        item["code"] == "h3_spatial_tiling_global_coordinates_missing"
        for item in report["issues"]["high_risk"]
    )


def test_environment_audit_flags_regular_h3_decode_above_internal_tile_boundary():
    capabilities = _capabilities(tiled_decode_global_coordinates="unsupported")
    report = _audit(
        _snapshot(capabilities=capabilities),
        width=736,
        height=416,
        decode_mode="regular",
    )
    assert report["status"] == "high_risk"
    assert report["requested"]["h3_internal_spatial_tiling_expected"] is True
    assert "h3_spatial_tiling_global_coordinates_missing" in {
        item["code"] for item in report["issues"]["high_risk"]
    }


def test_environment_audit_flags_known_fp8_ref2va_sage_dynamic_high_token_profile():
    report = _audit(
        _snapshot(),
        width=1920,
        height=1088,
        model_family="ref2va",
        model_precision="fp8",
        attention_backend="sage_attention",
        dynamic_vram_mode="enabled",
        reference_media_count=1,
    )
    assert report["status"] == "high_risk"
    assert any(
        item["code"] == "fp8_ref2va_sage_dynamic_high_token_risk"
        for item in report["issues"]["high_risk"]
    )


def test_environment_audit_flags_host_resource_thrashing_without_predicting_peak():
    snapshot = _snapshot()
    snapshot["runtime"]["host"] = {
        "ram_available_gib": 6.0,
        "commit_headroom_gib": 12.0,
    }
    snapshot["loaded_models"] = {
        "available": True,
        "count": 4,
        "currently_used_count": 1,
        "total_model_mib": 64.0 * 1024.0,
        "total_loaded_mib": 14.0 * 1024.0,
        "models": [],
    }
    report = _audit(snapshot)
    assert report["status"] == "high_risk"
    codes = {item["code"] for item in report["issues"]["high_risk"]}
    assert {
        "host_commit_headroom_low",
        "host_ram_available_low",
        "loaded_model_commit_thrashing_risk",
    } <= codes
    assert report["memory_safe_claim"] is False


def test_environment_audit_keeps_unavailable_host_and_model_state_unknown():
    snapshot = _snapshot()
    snapshot["runtime"]["host"] = {}
    snapshot["loaded_models"] = {"available": False}
    report = _audit(snapshot)
    assert report["status"] == "unknown"
    codes = {item["code"] for item in report["issues"]["unknown"]}
    assert {
        "host_commit_headroom_unknown",
        "host_ram_available_unknown",
        "loaded_model_state_unknown",
    } <= codes


def test_environment_audit_reports_current_snapshot_without_claiming_thrashing():
    snapshot = _snapshot()
    snapshot["runtime"]["process"] = {
        "available": True,
        "read_bytes": 270 * 1024**3,
        "page_faults": 123456,
    }
    report = _audit(snapshot)
    assert report["resource_fit_classification"] == (
        "fits_current_snapshot_thrashing_unmeasured"
    )
    assert any("before/after" in item for item in report["scientific_boundaries"])


def test_environment_audit_flags_fast_disk_and_active_thermal_throttling():
    snapshot = _snapshot()
    snapshot["runtime"]["comfy"]["fast_disk_enabled"] = True
    snapshot["runtime"]["gpu_health"] = {
        "available": True,
        "temperature_c": 89,
        "thermal_throttling": True,
        "throttle_reasons_raw": 64,
    }
    report = _audit(snapshot)
    assert report["status"] == "high_risk"
    assert report["resource_fit_classification"] == "unsafe_current_state"
    assert {item["code"] for item in report["issues"]["warnings"]} >= {
        "disk_backed_dynamic_loading_enabled"
    }
    assert {item["code"] for item in report["issues"]["high_risk"]} >= {
        "gpu_thermal_throttling_observed"
    }

def test_environment_audit_does_not_mutate_connected_objects():
    model = object()
    positive = [[object(), {"minimax_refs": [{"kind": "image"}]}]]
    model_id = id(model)
    positive_id = id(positive)
    snapshot = collect_environment_snapshot(model=None, positive=positive)
    report = audit_h3_environment(
        "ref2va",
        736,
        416,
        124,
        "ref2va",
        "int8_convrot",
        "stock",
        "none",
        "regular",
        "auto_detect",
        1,
        0,
        0.0,
        model,
        positive,
        snapshot=snapshot,
    )
    assert id(model) == model_id
    assert id(positive) == positive_id
    assert report["observed_conditioning_reference_kinds"] == ["image"]


def test_environment_audit_node_strict_mode_blocks_known_risk(monkeypatch):
    report = _audit(_snapshot(free_mib=128.0))
    monkeypatch.setattr(
        "h3_audio_t8_pkg.nodes_environment_audit_advanced.audit_h3_environment",
        lambda *_args, **_kwargs: report,
    )
    with pytest.raises(ValueError, match="current_vram_headroom_below_gate"):
        MiniMaxH3EnvironmentAuditT8Advanced.execute(
            "t2va",
            736,
            416,
            124,
            "fl2va",
            "int8_convrot",
            "stock",
            "none",
            "regular",
            "enabled",
            0,
            0,
            512.0,
            "block_known_unsafe",
        )
    assert "current_vram_headroom_below_gate" in blocking_summary(report)


def test_environment_audit_node_report_only_returns_machine_readable_json(monkeypatch):
    report = _audit(_snapshot())
    monkeypatch.setattr(
        "h3_audio_t8_pkg.nodes_environment_audit_advanced.audit_h3_environment",
        lambda *_args, **_kwargs: report,
    )
    output = MiniMaxH3EnvironmentAuditT8Advanced.execute(
        "t2va",
        736,
        416,
        124,
        "fl2va",
        "int8_convrot",
        "stock",
        "none",
        "regular",
        "enabled",
        0,
        0,
        512.0,
        "report_only",
    )
    parsed = json.loads(output[2])
    assert output[0] is True
    assert output[1] == "pass"
    assert parsed["schema"] == ENVIRONMENT_AUDIT_SCHEMA
