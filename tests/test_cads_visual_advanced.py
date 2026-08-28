from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import cads_visual_advanced as cads


class FakeModel:
    def __init__(self):
        self.wrappers = {}
        self.attachments = {}

    def clone(self):
        cloned = FakeModel()
        cloned.wrappers = dict(self.wrappers)
        cloned.attachments = dict(self.attachments)
        return cloned

    def add_wrapper_with_key(self, wrapper_type, key, callback):
        self.wrappers[(wrapper_type, key)] = callback

    def set_attachments(self, key, value):
        self.attachments[key] = value


class CaptureExecutor:
    def __init__(self):
        self.kwargs = None

    def __call__(self, x, timestep, context, transformer_options, **kwargs):
        self.kwargs = kwargs
        return x


def _wrapper(model):
    assert len(model.wrappers) == 1
    return next(iter(model.wrappers.values()))


def test_cads_piecewise_gamma_matches_paper_equation():
    values = cads.cads_gamma(torch.tensor([0.0, 0.6, 0.75, 0.9, 1.0]), 0.6, 0.9)
    assert values.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="tau1 < tau2"):
        cads.cads_gamma(torch.tensor(0.5), 0.9, 0.6)


def test_anneal_visual_condition_has_exact_clean_and_noise_endpoints():
    clean = torch.tensor([1.0, 2.0, 3.0])
    noise = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(
        cads.anneal_visual_condition(clean, noise, 1.0, 0.25, 0.0), clean
    )
    assert torch.equal(
        cads.anneal_visual_condition(clean, noise, 0.0, 0.25, 0.0),
        noise * 0.25,
    )


def test_runtime_changes_only_visual_condition_payload_and_preserves_original():
    source = FakeModel()
    patched, report_json = cads.build_cads_visual_reference_model(
        source,
        noise_scale=0.2,
        tau1=0.6,
        tau2=0.9,
        rescale_mix=0.0,
        noise_mode="stable_fixed_path",
        seed=42,
    )
    report = json.loads(report_json)
    assert report["audio_conditioning_changed"] is False
    assert not source.wrappers

    visual = torch.ones(1, 2, 2)
    audio = torch.full((1, 2), 7.0)
    payload = {
        "cond_video_latents": [visual],
        "cond_audio_latents": [audio],
        "visual_cond_noise_aug": 0.99,
    }
    executor = CaptureExecutor()
    result = _wrapper(patched)(
        executor,
        [torch.zeros(1), torch.zeros(1)],
        torch.tensor([1000.0]),
        torch.zeros(1),
        {},
        minimax_payload=payload,
    )
    runtime_payload = executor.kwargs["minimax_payload"]
    assert result[0].shape == (1,)
    assert runtime_payload is not payload
    assert runtime_payload["cond_audio_latents"][0] is audio
    assert runtime_payload["visual_cond_noise_aug"] == 1.0
    assert not torch.equal(runtime_payload["cond_video_latents"][0], visual)
    assert payload["cond_video_latents"][0] is visual
    assert payload["visual_cond_noise_aug"] == 0.99


def test_runtime_is_exact_passthrough_without_visual_conditions():
    patched, _ = cads.build_cads_visual_reference_model(
        FakeModel(), 0.1, 0.6, 0.9, 1.0, "paper_independent", 0
    )
    payload = {"cond_audio_latents": [torch.ones(1)]}
    executor = CaptureExecutor()
    _wrapper(patched)(
        executor,
        [torch.zeros(1), torch.zeros(1)],
        torch.tensor([500.0]),
        torch.zeros(1),
        {},
        minimax_payload=payload,
    )
    assert executor.kwargs["minimax_payload"] is payload


def test_frontend_workflow_routes_model_through_cads_and_has_note():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "03-image-video-edit"
        / "2026-08-28_H3_CADS_Visual_Reference_Annealing_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    cads_node = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3CADSVisualReferenceT8Advanced"
    )
    dual_clock = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    link_by_id = {link[0]: link for link in workflow["links"]}
    assert workflow["last_node_id"] == max(nodes)
    assert link_by_id[cads_node["outputs"][0]["links"][0]][3] == dual_clock["id"]
    assert dual_clock["inputs"][0]["link"] == cads_node["outputs"][0]["links"][0]
    assert not any(
        node["type"] == "MiniMaxH3VisualReferenceStrengthEXPT8"
        for node in nodes.values()
    )
    assert any(node["type"] == "MarkdownNote" for node in nodes.values())
