from __future__ import annotations

import asyncio
import json

import pytest
import torch

from comfy.ldm.minimax.model import MiniMaxH3Model as CoreMiniMaxH3Model

from h3_audio_t8_pkg.forward_sync_advanced import (
    FORWARD_PATCH_PATH,
    _compile_rewritten_forward,
    _rewrite_forward_source,
    build_forward_sync_optimization,
    probe_native_forward_sync,
)


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


class MiniMaxH3Model:
    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0

    def _forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        payload = minimax_payload or {}
        shift_v = float(
            transformer_options.get(
                "minimax_h3_sigma_shift_video", self.sigma_shift_video
            )
        )
        shift_a = float(
            transformer_options.get(
                "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
            )
        )
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
        layout = payload["layout"]
        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = 0
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append(
                            (a + run_start, a + i, row_base + int(tags[run_start]))
                        )
                        run_start = i
        return t_v, t_a, sigma_v, mod_segments


class _Layout:
    segments = [(0, 3, "text")]


class _NoView:
    def view(self, *_args):
        raise AssertionError("cached text tags should not be copied again")


class _FakeModelPatcher:
    def __init__(self, diffusion, object_patches=None):
        self.diffusion = diffusion
        self.object_patches = dict(object_patches or {})
        self.attachments = {}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def clone(self):
        return _FakeModelPatcher(self.diffusion, self.object_patches)

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def set_attachments(self, key, value):
        self.attachments[key] = value


def test_current_core_forward_can_be_rewritten_without_copying_the_function():
    rewritten, report = _rewrite_forward_source(CoreMiniMaxH3Model._forward)
    assert report["sigma_rewritten"] is True
    assert report["text_tags_rewritten"] is True
    assert "sigma_v_scalar = float(sigma_v)" in rewritten
    assert "_text_token_tags_list" in rewritten
    assert "sigma_v.to(m.device)" in rewritten


def test_rewritten_forward_preserves_values_and_caches_tags():
    replacement, report = _compile_rewritten_forward(MiniMaxH3Model._forward)
    assert report["sigma_rewritten"] is True
    model = MiniMaxH3Model()
    payload = {
        "layout": _Layout(),
        "text_token_tags": torch.tensor([1, 1, 0]),
    }
    baseline = model._forward(None, torch.tensor([500.0]), None, minimax_payload=payload)
    payload.pop("_text_token_tags_list", None)
    optimized = replacement(
        model,
        None,
        torch.tensor([500.0]),
        None,
        minimax_payload=payload,
    )
    assert optimized[0] == pytest.approx(baseline[0])
    assert optimized[1] == pytest.approx(baseline[1])
    assert torch.equal(optimized[2], baseline[2])
    assert optimized[3] == baseline[3]
    assert payload["_text_token_tags_list"] == [1, 1, 0]

    payload["text_token_tags"] = _NoView()
    repeated = replacement(
        model,
        None,
        torch.tensor([500.0]),
        None,
        minimax_payload=payload,
    )
    assert repeated[3] == baseline[3]


def test_builder_uses_clone_only_object_patch_and_reports_contract():
    source = _FakeModelPatcher(MiniMaxH3Model())
    patched, report_json = build_forward_sync_optimization(source)
    report = json.loads(report_json)
    assert patched is not source
    assert source.object_patches == {}
    assert FORWARD_PATCH_PATH in patched.object_patches
    assert report["status"] == "compatibility_patch_ready"
    assert report["expected_host_syncs_per_step"] == 1


def test_builder_composes_a_structurally_compatible_existing_forward_patch():
    diffusion = MiniMaxH3Model()
    existing = diffusion._forward
    source = _FakeModelPatcher(diffusion, {FORWARD_PATCH_PATH: existing})
    patched, report_json = build_forward_sync_optimization(source)
    report = json.loads(report_json)
    assert patched.object_patches[FORWARD_PATCH_PATH] is not existing
    assert report["composed_existing_forward_patch"] is True


def test_native_probe_requires_both_optimizations():
    assert probe_native_forward_sync(MiniMaxH3Model._forward)["available"] is False
    replacement, _ = _compile_rewritten_forward(MiniMaxH3Model._forward)
    assert probe_native_forward_sync(replacement)["available"] is True


def test_registration_is_append_only():
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension

    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    ]
    assert node_ids.index("MiniMaxH3ForwardSyncOptimizationT8Advanced") > node_ids.index(
        "MiniMaxH3AttentionHooksT8Advanced"
    )
