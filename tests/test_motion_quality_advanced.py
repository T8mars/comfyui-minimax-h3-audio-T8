from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.motion_quality_advanced import (
    TURBO_DUAL_CLOCK_TEST_STEPS,
    audit_motion_quality,
    build_av_sigma_tail_schedule,
)
from h3_audio_t8_pkg.nodes_motion_quality_advanced import (
    MiniMaxH3AVSigmaTailSubdivisionT8Advanced,
    MiniMaxH3MotionQualityAuditT8Advanced,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas, time_shift_sigma


def _build_schedule(sigmas, **overrides):
    values = {
        "mode": "report_only",
        "extra_substeps": 0,
        "range_mode": "tail_intervals",
        "tail_intervals": 1,
        "start_progress": 0.75,
        "end_progress": 1.0,
        "spacing": "base_time_linear",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "profile": "turbo_standard8",
        "sampling_route": "dual_clock_euler",
        "accept_turbo_schedule_ood": False,
    }
    values.update(overrides)
    return build_av_sigma_tail_schedule(sigmas, **values)


def test_turbo_dual_clock_test_standard_is_eight_steps_and_report_only_is_identity():
    assert TURBO_DUAL_CLOCK_TEST_STEPS == 8
    sigmas = native_flow_sigmas(TURBO_DUAL_CLOCK_TEST_STEPS, 12.0)
    output, actual_nfe, report_json = _build_schedule(sigmas)
    report = json.loads(report_json)

    assert output is sigmas
    assert actual_nfe == 8
    assert report["turbo_dual_clock_test_standard_steps"] == 8
    assert report["applied"] is False
    assert report["noop_reason"] == "report_only"
    assert report["input_schedule_sha256"] == report["output_schedule_sha256"]


def test_apply_preserves_all_original_knots_and_inserts_in_base_flow_time():
    sigmas = native_flow_sigmas(8, 12.0)
    output, actual_nfe, report_json = _build_schedule(
        sigmas,
        mode="apply_exp",
        extra_substeps=1,
        accept_turbo_schedule_ood=True,
    )
    report = json.loads(report_json)

    assert actual_nfe == 9
    assert output.device == sigmas.device
    assert output.dtype == sigmas.dtype
    assert output.shape == (10,)
    assert torch.all(output[:-1] > output[1:])
    assert output[-1].item() == 0.0

    cursor = 0
    for value in output:
        if cursor < sigmas.numel() and value.item() == sigmas[cursor].item():
            cursor += 1
    assert cursor == sigmas.numel()

    inserted = report["inserted_points"]
    assert len(inserted) == 1
    assert inserted[0]["base_sigma"] == pytest.approx(0.0625)
    assert inserted[0]["video_sigma"] == pytest.approx(4 / 9)
    assert inserted[0]["audio_sigma"] == pytest.approx(1 / 6)
    assert report["estimated_sampler_time_increase_percent"] == pytest.approx(12.5)
    assert report["all_original_knots_preserved"] is True
    assert report["quality_validated"] is False
    assert report["memory_safe_claim"] is False


def test_two_tail_insertions_keep_audio_clock_monotonic_and_report_exact_nfe():
    sigmas = native_flow_sigmas(8, 12.0).to(dtype=torch.float64)
    output, actual_nfe, report_json = _build_schedule(
        sigmas,
        mode="apply_exp",
        extra_substeps=2,
        tail_intervals=2,
        spacing="base_time_cosine",
        accept_turbo_schedule_ood=True,
    )
    report = json.loads(report_json)
    audio_sigmas = time_shift_sigma(output, 12.0, 3.0)

    assert actual_nfe == 10
    assert report["inserted_substeps"] == 2
    assert len(report["video_sigmas"]) == 11
    assert len(report["audio_sigmas"]) == 11
    assert torch.all(audio_sigmas[:-1] > audio_sigmas[1:])
    assert audio_sigmas[-1].item() == 0.0


def test_turbo_apply_requires_explicit_ood_acceptance_and_exact_eight_step_input():
    sigmas = native_flow_sigmas(8, 12.0)
    with pytest.raises(ValueError, match="8-step test baseline"):
        _build_schedule(sigmas, mode="apply_exp", extra_substeps=1)
    with pytest.raises(ValueError, match="requires 8 steps"):
        _build_schedule(native_flow_sigmas(4, 12.0))


def test_unverified_sampling_routes_fail_closed_for_apply_but_can_be_reported():
    sigmas = native_flow_sigmas(8, 12.0)
    output, actual_nfe, report_json = _build_schedule(
        sigmas,
        sampling_route="multirate_exp_unsupported",
    )
    assert output is sigmas
    assert actual_nfe == 8
    assert json.loads(report_json)["apply_blockers"]
    with pytest.raises(ValueError, match="supports only"):
        _build_schedule(
            sigmas,
            mode="apply_exp",
            extra_substeps=1,
            sampling_route="multirate_exp_unsupported",
            accept_turbo_schedule_ood=True,
        )


@pytest.mark.parametrize(
    "sigmas,match",
    [
        (torch.tensor([1.0, 0.5, 0.6, 0.0]), "strictly descending"),
        (torch.tensor([1.0, float("nan"), 0.0]), "NaN or Inf"),
        (torch.tensor([1.0, 0.5, 0.1]), "end at exactly zero"),
        (torch.tensor([1.2, 0.5, 0.0]), r"normalized \[0, 1\]"),
    ],
)
def test_invalid_schedules_fail_closed(sigmas, match):
    with pytest.raises(ValueError, match=match):
        _build_schedule(sigmas, profile="custom_strict")


def _audit(frames, **overrides):
    values = {
        "fps": 24.0,
        "roi_mode": "full_frame",
        "roi_x": 0.25,
        "roi_y": 0.05,
        "roi_width": 0.5,
        "roi_height": 0.5,
        "sharpness_ratio_floor": 0.55,
        "temporal_instability_multiplier": 2.5,
        "high_motion_delta_floor": 0.03,
        "freeze_delta_ceiling": 0.002,
        "repair_context_frames": 4,
        "face_mask": None,
    }
    values.update(overrides)
    return audit_motion_quality(frames, **values)


def test_motion_audit_flags_a_high_motion_blurred_frame_and_returns_legal_window():
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((xx + yy) % 2).float()
    frames = []
    for index in range(22):
        image = torch.roll(checker, shifts=index % 2, dims=1)
        if index == 10:
            image = torch.full_like(image, 0.5)
        frames.append(image.unsqueeze(-1).repeat(1, 1, 3))
    frames = torch.stack(frames)

    risk, range_count, ranges_json, report_json = _audit(frames)
    ranges = json.loads(ranges_json)
    report = json.loads(report_json)

    assert risk is True
    assert range_count >= 1
    assert any(
        item["raw_start_frame"] <= 10 <= item["raw_end_frame"] for item in ranges
    )
    assert all(item["suggested_length"] in {5, 22} for item in ranges)
    assert report["identity_metric_valid"] is False
    assert report["face_detection_valid"] is False
    assert report["quality_guarantee"] is False


def test_motion_audit_can_report_a_stable_sequence_without_claiming_identity():
    ramp = torch.linspace(0.0, 0.8, 32).view(1, 32).expand(32, 32)
    frames = torch.stack(
        [
            (ramp + index * 0.001).clamp(0.0, 1.0).unsqueeze(-1).repeat(1, 1, 3)
            for index in range(22)
        ]
    )
    risk, range_count, ranges_json, report_json = _audit(
        frames,
        freeze_delta_ceiling=0.0001,
    )
    report = json.loads(report_json)

    assert risk is False
    assert range_count == 0
    assert json.loads(ranges_json) == []
    assert report["status"] == "no_proxy_risk_detected"
    assert report["identity_metric_valid"] is False


def test_motion_audit_manual_roi_and_connected_mask_are_strict():
    frames = torch.zeros((5, 16, 16, 3))
    with pytest.raises(ValueError, match="inside normalized"):
        _audit(
            frames,
            roi_mode="manual_static_roi",
            roi_x=0.8,
            roi_width=0.5,
        )
    with pytest.raises(ValueError, match="requires face_mask"):
        _audit(frames, roi_mode="connected_mask")
    with pytest.raises(ValueError, match="empty frame"):
        _audit(
            frames,
            roi_mode="connected_mask",
            face_mask=torch.zeros((1, 16, 16)),
        )


def test_new_node_schemas_are_default_off_and_appendable():
    sigma_schema = MiniMaxH3AVSigmaTailSubdivisionT8Advanced.define_schema()
    sigma_inputs = {item.id: item for item in sigma_schema.inputs}
    assert sigma_schema.node_id.endswith("Advanced")
    assert sigma_schema.is_experimental is True
    assert sigma_inputs["mode"].default == "report_only"
    assert sigma_inputs["extra_substeps"].default == 0
    assert sigma_inputs["profile"].default == "turbo_standard8"
    assert sigma_inputs["accept_turbo_schedule_ood"].default is False

    audit_schema = MiniMaxH3MotionQualityAuditT8Advanced.define_schema()
    assert audit_schema.node_id.endswith("Advanced")
    assert audit_schema.is_experimental is True
    assert audit_schema.is_output_node is True
    assert audit_schema.category == "T8/MiniMax H3/Quality/Experimental"


def test_motion_quality_examples_use_eight_step_baseline_and_safe_defaults():
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (root / "examples" / "motion_quality_advanced_8step_api.json").read_text(
            encoding="utf-8"
        )
    )
    sampler = next(
        node
        for node in api.values()
        if node["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    sigma = next(
        node
        for node in api.values()
        if node["class_type"] == "MiniMaxH3AVSigmaTailSubdivisionT8Advanced"
    )
    sampler_custom = next(
        node for node in api.values() if node["class_type"] == "SamplerCustomAdvanced"
    )
    sigma_id = next(key for key, value in api.items() if value is sigma)
    assert sampler["inputs"]["steps"] == 8
    assert sigma["inputs"]["mode"] == "report_only"
    assert sigma["inputs"]["extra_substeps"] == 0
    assert sigma["inputs"]["profile"] == "turbo_standard8"
    assert sigma["inputs"]["accept_turbo_schedule_ood"] is False
    assert sampler_custom["inputs"]["sigmas"] == [sigma_id, 0]
    assert any(
        node["class_type"] == "MiniMaxH3MotionQualityAuditT8Advanced"
        for node in api.values()
    )

    frontend = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "H3_Motion_Quality_Advanced_8Step_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in frontend["nodes"]}
    assert frontend["version"] == 0.4
    assert frontend["last_node_id"] == max(nodes)
    assert frontend["last_link_id"] == max(link[0] for link in frontend["links"])
    dual = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    sigma = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3AVSigmaTailSubdivisionT8Advanced"
    )
    assert dual["widgets_values"][:3] == [8, 12.0, 3.0]
    assert sigma["widgets_values"][:2] == ["report_only", 0]
    assert "MiniMaxH3MotionQualityAuditT8Advanced" in {
        node["type"] for node in nodes.values()
    }
    for link_id, source, output_slot, target, input_slot, link_type in frontend[
        "links"
    ]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
