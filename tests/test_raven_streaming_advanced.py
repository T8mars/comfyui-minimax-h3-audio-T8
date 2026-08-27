from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.raven_streaming_advanced import (
    audit_raven_streaming_request,
    build_raven_streaming_profile,
    load_raven_model_guarded,
)


class _FeatureReport:
    ok = True

    def to_dict(self):
        return {"ok": True, "required_features": "present"}


class _Attachment:
    rank = 128
    alpha = 128.0
    strength = 1.0
    detached = False

    def __len__(self):
        return 266


def _hardware(gpu=24.0, total=192.0, available=160.0):
    return {
        "cuda_available": True,
        "bf16_supported": True,
        "gpu_name": "fake",
        "gpu_total_gib": gpu,
        "system_total_ram_gib": total,
        "system_available_ram_gib": available,
    }


def _runtime(model=None, *, attachment=None, calls=None):
    model = model or SimpleNamespace(object_patches={})
    attachment = attachment if attachment is not None else _Attachment()
    calls = calls if calls is not None else []

    class LoaderNode:
        def load_model(self, unet_name, lora_name, weight_dtype):
            calls.append((unet_name, lora_name, weight_dtype))
            return (model,)

    request = SimpleNamespace(
        frames=90,
        width=768,
        height=448,
        latent_t=27,
        audio_t=150,
    )
    resolved = SimpleNamespace(
        patcher=model,
        diffusion_model=SimpleNamespace(),
        num_layers=50,
    )
    return {
        "package": SimpleNamespace(__version__="0.1.0", __file__=__file__),
        "nodes": SimpleNamespace(RAVENModelLoader=LoaderNode),
        "compat": SimpleNamespace(check_features=lambda: _FeatureReport()),
        "contracts": SimpleNamespace(
            parse_conditioning=lambda value: SimpleNamespace(
                cross_attn=torch.zeros((1, 64, 8))
            ),
            parse_latent=lambda value: request,
            resolve_model=lambda value: resolved,
        ),
        "loader": SimpleNamespace(get_raven_attachment=lambda value: attachment),
    }


def _resolved_files():
    return {
        "unet": ("C:/models/minimax_h3_full_bf16.safetensors", 55_000_000_000),
        "lora": ("C:/models/minimax_h3_raven.safetensors", 5_000_000_000),
    }


def test_published_profile_overrides_manual_values_exactly():
    values = build_raven_streaming_profile(
        "published_preview_4nfe", 9, 6.0, 1.5, 8, 9, "gpu"
    )
    assert values[:6] == (4, 12.0, 3.0, 2, 2, "cpu_pinned")
    report = json.loads(values[6])
    assert report["exact_published_profile"] is True


def test_manual_profile_is_forwarded_and_marked_experimental():
    values = build_raven_streaming_profile(
        "manual_experimental", 6, 10.0, 2.5, 3, 4, "cpu"
    )
    assert values[:6] == (6, 10.0, 2.5, 3, 4, "cpu")
    assert json.loads(values[6])["exact_published_profile"] is False


def test_guarded_loader_blocks_outside_reviewed_envelope_before_delegate():
    calls = []
    runtime = _runtime(calls=calls)
    with pytest.raises(RuntimeError, match="GPU_OUTSIDE_REVIEWED_ENVELOPE"):
        load_raven_model_guarded(
            "minimax_h3_full_bf16.safetensors",
            "minimax_h3_raven.safetensors",
            "bf16",
            "block_outside_reviewed_envelope",
            runtime=runtime,
            hardware=_hardware(gpu=16.0, total=128.0, available=100.0),
            installations=["C:/ComfyUI/custom_nodes/RAVEN"],
            resolved_files=_resolved_files(),
        )
    assert calls == []


def test_guarded_loader_delegates_exact_arguments_after_preflight():
    calls = []
    model = SimpleNamespace(object_patches={})
    returned, report_json = load_raven_model_guarded(
        "minimax_h3_full_bf16.safetensors",
        "minimax_h3_raven.safetensors",
        "bf16",
        "block_outside_reviewed_envelope",
        runtime=_runtime(model, calls=calls),
        hardware=_hardware(),
        installations=["C:/ComfyUI/custom_nodes/RAVEN"],
        resolved_files=_resolved_files(),
    )
    assert returned is model
    assert calls == [
        (
            "minimax_h3_full_bf16.safetensors",
            "minimax_h3_raven.safetensors",
            "bf16",
        )
    ]
    report = json.loads(report_json)
    assert report["decision"] == "LOADED"
    assert report["mechanically_compatible"] is True


def test_guarded_loader_reports_quantized_base_without_blocking():
    calls = []
    _model, report_json = load_raven_model_guarded(
        "minimax_h3_int8_convrot.safetensors",
        "minimax_h3_raven.safetensors",
        "default",
        "block_mechanical_conflicts",
        runtime=_runtime(calls=calls),
        hardware=_hardware(),
        installations=["C:/ComfyUI/custom_nodes/RAVEN"],
        resolved_files=_resolved_files(),
    )
    assert calls
    report = json.loads(report_json)
    assert any(
        item["code"] == "QUANTIZED_OR_PRUNED_BASE"
        for item in report["model_and_runtime_diagnostics"]
    )


def test_request_audit_passes_through_exact_objects_for_published_t2va():
    model = SimpleNamespace(object_patches={})
    positive = object()
    latent = object()
    result = audit_raven_streaming_request(
        model,
        positive,
        latent,
        4,
        12.0,
        3.0,
        2,
        2,
        "cpu_pinned",
        False,
        "block_outside_reviewed_envelope",
        runtime=_runtime(model),
    )
    assert result[:3] == (model, positive, latent)
    assert result[3:5] == (True, "PASS")
    report = json.loads(result[5])
    assert report["attachment"] == {
        "alpha": 128.0,
        "detached": False,
        "modules": 266,
        "present": True,
        "rank": 128,
        "strength": 1.0,
    }


def test_request_audit_blocks_foreign_object_patch():
    model = SimpleNamespace(object_patches={"diffusion_model": object()})
    with pytest.raises(ValueError, match="OBJECT_PATCH_CONFLICT"):
        audit_raven_streaming_request(
            model,
            object(),
            object(),
            4,
            12.0,
            3.0,
            2,
            2,
            "cpu_pinned",
            False,
            "block_mechanical_conflicts",
            runtime=_runtime(model),
        )


def test_request_audit_reports_profile_deviation_without_claiming_pass():
    model = SimpleNamespace(object_patches={})
    result = audit_raven_streaming_request(
        model,
        object(),
        object(),
        6,
        12.0,
        3.0,
        2,
        2,
        "cpu",
        False,
        "report_only",
        runtime=_runtime(model),
    )
    assert result[3] is True
    assert result[4] == "ABSTAIN"
    report = json.loads(result[5])
    assert {item["code"] for item in report["reviewed_envelope_findings"]} == {
        "PROFILE_DEVIATION"
    }


def test_request_audit_requires_explicit_ack_for_more_than_192_frames():
    runtime = _runtime()
    runtime["contracts"].parse_latent = lambda value: SimpleNamespace(
        frames=209,
        width=768,
        height=448,
        latent_t=62,
        audio_t=349,
    )
    with pytest.raises(ValueError, match="LONG_REQUEST_ACK_REQUIRED"):
        audit_raven_streaming_request(
            SimpleNamespace(object_patches={}),
            object(),
            object(),
            4,
            12.0,
            3.0,
            2,
            2,
            "cpu_pinned",
            False,
            "block_outside_reviewed_envelope",
            runtime=runtime,
        )


def test_raven_frontend_workflow_uses_one_profile_and_external_sampler():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "16-raven-streaming"
        / "2026-08-23_H3_RAVEN_Streaming_T2VA_Guarded_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3RavenStreamingProfileT8Advanced",
        "MiniMaxH3RavenGuardedLoaderT8Advanced",
        "MiniMaxH3RavenRequestAuditT8Advanced",
        "RAVENStreamingSampler",
        "MiniMaxH3ImageToVideo",
    } <= types
    assert "RAVENModelLoader" not in types
    assert "KSampler" not in types
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 4

    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3ImageToVideo"
    )
    assert conditioning["widgets_values"][1:] == [768, 448, 90]
    assert (
        next(
            value for value in conditioning["inputs"] if value["name"] == "first_frame"
        )["link"]
        is None
    )
    assert (
        next(
            value for value in conditioning["inputs"] if value["name"] == "last_frame"
        )["link"]
        is None
    )

    profile = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3RavenStreamingProfileT8Advanced"
    )
    audit = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3RavenRequestAuditT8Advanced"
    )
    sampler = next(
        node for node in nodes.values() if node["type"] == "RAVENStreamingSampler"
    )
    assert profile["widgets_values"] == [
        "published_preview_4nfe",
        4,
        12.0,
        3.0,
        2,
        2,
        "cpu_pinned",
    ]
    assert audit["widgets_values"][-2:] == [False, "block_outside_reviewed_envelope"]
    assert sampler["widgets_values"] == [0, "fixed", 4, 12.0, 3.0, 2, 2, "cpu_pinned"]

    for name, profile_slot in {
        "steps": 0,
        "video_shift": 1,
        "audio_shift": 2,
        "sink": 3,
        "window": 4,
        "kv_cache_storage": 5,
    }.items():
        for target in (audit, sampler):
            target_input = next(
                value for value in target["inputs"] if value["name"] == name
            )
            link = links[target_input["link"]]
            assert link[1:3] == [profile["id"], profile_slot]

    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, source, output_slot, target, input_slot, link_type in workflow[
        "links"
    ]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
