from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.nodes import comfy_entrypoint
from h3_audio_t8_pkg.nodes_subject_safe_rgb_composite_advanced import (
    MiniMaxH3SubjectSafeRGBCompositeT8Advanced,
)
from h3_audio_t8_pkg.subject_safe_rgb_composite_advanced import (
    REPORT_SCHEMA,
    compose_subject_safe_rgb,
)


def _inputs():
    base = torch.zeros((3, 8, 10, 3), dtype=torch.float32)
    refined = torch.ones_like(base)
    alpha = torch.zeros((3, 8, 10), dtype=torch.float32)
    alpha[:, 2:6, 3:8] = 0.5
    return base, refined, alpha


def test_composite_keeps_zero_alpha_pixels_exact_and_audio_same_object():
    base, refined, alpha = _inputs()
    audio = {"waveform": torch.zeros((1, 2, 32)), "sample_rate": 32000}

    selected, candidate, source, used, audio_out, report_json = compose_subject_safe_rgb(
        base,
        refined,
        alpha,
        accept_candidate=True,
        minimum_subject_area=0.0,
        maximum_subject_area=1.0,
        audio=audio,
    )

    assert selected is candidate
    assert source is base
    assert audio_out is audio
    assert torch.equal(candidate[alpha == 0], base[alpha == 0])
    assert torch.allclose(candidate[alpha > 0], torch.full_like(candidate[alpha > 0], 0.5))
    assert torch.equal(used, alpha)
    report = json.loads(report_json)
    assert report["schema"] == REPORT_SCHEMA
    assert report["outside_zero_alpha_exact_by_construction"] is True
    assert report["automatic_quality_selection"] is False


def test_protect_mask_removes_refined_ownership_exactly():
    base, refined, alpha = _inputs()
    protect = torch.zeros_like(alpha)
    protect[:, 3:5, 4:7] = 1.0

    _selected, candidate, _source, used, _audio, _report = compose_subject_safe_rgb(
        base,
        refined,
        alpha,
        accept_candidate=True,
        minimum_subject_area=0.0,
        maximum_subject_area=1.0,
        protect_mask=protect,
    )

    assert torch.equal(used[protect == 1], torch.zeros_like(used[protect == 1]))
    protected_rgb = protect.unsqueeze(-1).expand_as(base) == 1
    assert torch.equal(candidate[protected_rgb], base[protected_rgb])


def test_contract_failure_returns_complete_source():
    base, refined, alpha = _inputs()

    selected, candidate, source, used, _audio, report_json = compose_subject_safe_rgb(
        base,
        refined,
        alpha,
        accept_candidate=True,
        minimum_subject_area=0.90,
        maximum_subject_area=1.0,
    )

    assert selected is base
    assert candidate is base
    assert source is base
    assert torch.count_nonzero(used) == 0
    report = json.loads(report_json)
    assert report["fallback_applied"] is True
    assert report["status"] == "ABSTAIN_CONTRACT_FAILURE_SOURCE_RETURNED"


def test_strict_mask_frames_rejects_accidental_static_broadcast():
    base, refined, alpha = _inputs()
    with pytest.raises(ValueError, match="provide a tracked per-frame mask"):
        compose_subject_safe_rgb(base, refined, alpha[:1])


def test_single_mask_broadcast_is_explicit_and_reported():
    base, refined, alpha = _inputs()
    _selected, _candidate, _source, used, _audio, report_json = compose_subject_safe_rgb(
        base,
        refined,
        alpha[:1],
        mask_frame_policy="allow_single_broadcast_exp",
        minimum_subject_area=0.0,
        maximum_subject_area=1.0,
    )
    assert used.shape[0] == base.shape[0]
    assert json.loads(report_json)["subject_alpha_single_frame_broadcast"] is True


def test_node_is_registered_at_append_only_tail():
    schema = MiniMaxH3SubjectSafeRGBCompositeT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3SubjectSafeRGBCompositeT8Advanced"
    assert schema.is_experimental is True
    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert node_ids[267] == schema.node_id


def test_frontend_workflow_is_comfyui_graph_and_explains_manual_alpha():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "13-latent-upscale"
        / "2026-09-01_H3_Subject_Safe_RGB_Composite_v8_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert types.count("MiniMaxH3SubjectSafeRGBCompositeT8Advanced") == 1
    assert types.count("LoadVideo") == 3
    assert types.count("GetVideoComponents") == 3
    assert types.count("ImageToMask") == 1
    assert types.count("MarkdownNote") == 2
    composite = next(
        node for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SubjectSafeRGBCompositeT8Advanced"
    )
    assert composite["widgets_values"][:3] == [
        True,
        "input_alpha_exact",
        "strict_exact",
    ]
    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "逐帧灰度无损视频" in notes
    assert "不会自动识别人" in notes
