from __future__ import annotations

import json

from h3_audio_t8_pkg.community_diagnostics_advanced import (
    diagnose_official_h3_risks,
    probe_generic_loop_capability,
    probe_taeh3_preview_capability,
)


def test_generic_loop_probe_is_read_only_and_current_core_is_not_claimed_ready():
    available, status, report_json = probe_generic_loop_capability()
    report = json.loads(report_json)
    assert available is False
    assert status == "UNAVAILABLE_CURRENT_CORE"
    assert report["side_effects"] is False
    assert report["workflow_switched"] is False
    assert report["source_pr_state_at_2026_08_28"] == "draft"


def test_official_risk_diagnostic_keeps_unknowns_and_never_hard_gates():
    status, count, report_json = diagnose_official_h3_risks(
        736,
        416,
        124,
        2,
        2,
        1,
        "stock",
        runtime_report_json='{"minimum_free_vram_mib":300,"v_copy_peak_mib":700}',
        audio_report_json='{"peak_abs":1.0,"clipped_sample_count":12}',
        frame_report_json='{"dark_frame_indices":[17,34,51,68]}',
    )
    report = json.loads(report_json)
    assert status == "RISKS_OBSERVED"
    assert count >= 5
    assert report["hard_gates"] is False
    assert report["model_fingerprint_checked"] is False
    assert "MULTISPEAKER_REFERENCE_UNDERSPECIFIED" in {
        item["code"] for item in report["risks"]
    }
    assert report["dark_flash_periodicity"]["state"] == "suspected"


def test_official_risk_diagnostic_does_not_invent_evidence():
    status, count, report_json = diagnose_official_h3_risks(
        736, 416, 124, 0, 1, 1, "unknown"
    )
    report = json.loads(report_json)
    assert status == "INSUFFICIENT_EVIDENCE"
    assert count == 0
    assert report["unknowns"]


def test_taeh3_preview_probe_is_read_only_and_reports_native_boundary():
    available, active, status, report_json = probe_taeh3_preview_capability()
    report = json.loads(report_json)
    assert report["schema"] == "t8.minimax_h3.taeh3_preview_capability.v1"
    assert report["side_effects"] is False
    assert report["preview_method_changed"] is False
    assert report["model_loaded"] is False
    assert report["sampling_changed"] is False
    assert report["hard_gates"] is False
    assert report["model_hash_gate"] is False
    assert report["core"]["decoder_name"] == "taeh3"
    assert report["core"]["nested_av_video_stream_only_observed"] is True
    assert report["core"]["first_video_frame_per_step_observed"] is True
    assert available is bool(report["available"])
    assert active is bool(report["active"])
    assert status == report["status"]
