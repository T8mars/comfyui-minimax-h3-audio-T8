from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import ays_schedule_advanced as ays


class FakeModel:
    pass


def test_manual_schedule_requires_exact_monotone_endpoints():
    parsed = ays._parse_manual_base_sigmas("[1, 0.7, 0.2, 0]", 3)
    assert torch.equal(parsed, torch.tensor([1.0, 0.7, 0.2, 0.0]))
    with pytest.raises(ValueError, match=r"steps \+ 1"):
        ays._parse_manual_base_sigmas("[1, 0]", 3)
    with pytest.raises(ValueError, match="strictly descending"):
        ays._parse_manual_base_sigmas("[1, 0.5, 0.5, 0]", 3)
    with pytest.raises(ValueError, match="start at exactly 1.0"):
        ays._parse_manual_base_sigmas("[0.9, 0.5, 0.1, 0]", 3)


def test_schedule_contract_maps_one_base_grid_to_both_h3_clocks(monkeypatch):
    patched = object()
    sampler = object()

    def fake_setup(*args):
        assert args[-1] == "native_flow"
        return patched, sampler, torch.linspace(1.0, 0.0, 4)

    monkeypatch.setattr(ays, "setup_dual_clock_sampling", fake_setup)
    model_out, sampler_out, video, report_json = ays.build_dual_clock_schedule_contract(
        FakeModel(),
        {"samples": object()},
        3,
        12.0,
        3.0,
        "manual_h3_calibrated",
        "[1.0, 0.7, 0.2, 0.0]",
        "local probe",
    )
    report = json.loads(report_json)
    assert model_out is patched
    assert sampler_out is sampler
    assert torch.allclose(video, ays.shift_sigma(torch.tensor([1.0, 0.7, 0.2, 0.0]), 12.0))
    assert report["audio_sigmas"] == pytest.approx(
        [float(v) for v in ays.shift_sigma(torch.tensor([1.0, 0.7, 0.2, 0.0]), 3.0)]
    )
    assert report["audio_and_video_share_base_knots"] is True
    assert report["ays_klub_optimized_for_minimax_h3"] is False


def test_native_profile_does_not_parse_unused_manual_text(monkeypatch):
    monkeypatch.setattr(
        ays,
        "setup_dual_clock_sampling",
        lambda *_args: (object(), object(), torch.linspace(1.0, 0.0, 3)),
    )
    *_, report_json = ays.build_dual_clock_schedule_contract(
        FakeModel(),
        {},
        2,
        12.0,
        3.0,
        "native_flow_baseline",
        "not json",
        "ignored",
    )
    report = json.loads(report_json)
    assert report["base_sigmas"] == pytest.approx([1.0, 0.5, 0.0])


def test_frontend_workflow_is_ui_format_and_uses_the_new_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-28_H3_Dual_Clock_AYS_Schedule_Contract_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    target = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockAYSScheduleT8Advanced"
    )
    assert workflow["last_node_id"] == max(nodes)
    assert "nodes" in workflow and "links" in workflow
    assert target["widgets_values"][0:4] == [8, 12.0, 3.0, "native_flow_baseline"]
    assert any(node["type"] == "MarkdownNote" for node in nodes.values())
