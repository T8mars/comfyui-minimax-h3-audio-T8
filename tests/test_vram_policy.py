from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import hybrid_model as hybrid
from h3_audio_t8_pkg import vram_policy as vram
from h3_audio_t8_pkg.speech_reliability import vram_preflight
from h3_audio_t8_pkg.vram_policy import process_resource_snapshot


def snapshot(*, free_mib=12288.0, total_mib=16384.0, commit_gib=96.0):
    return {
        "captured_unix": 1.0,
        "gpu": {
            "device": "cuda:0",
            "whole_device_free_mib": free_mib,
            "whole_device_total_mib": total_mib,
            "whole_device_used_mib": total_mib - free_mib,
            "torch_allocated_mib": 128.0,
            "torch_reserved_mib": 256.0,
        },
        "comfy": {
            "dynamic_vram_enabled": True,
            "extra_reserved_vram_gib": 0.683594,
            "startup_reserve_vram_gib": None,
            "startup_vram_headroom_gib": 2.0,
        },
        "aimdo": {
            "package_version": "0.4.13",
            "library_loaded": True,
            "initialized_device_count": 1,
        },
        "host": {
            "commit_headroom_gib": commit_gib,
            "commit_limit_gib": 256.0,
        },
    }


def build_policy(monkeypatch, **overrides):
    monkeypatch.setattr(vram, "runtime_snapshot", lambda: snapshot())
    values = {
        "mode": "fixed_total_reserved_exp",
        "fixed_total_reserved_gib": 2.0,
        "external_margin_gib": 1.0,
        "maximum_reserved_gib": 8.0,
        "clean_before_load": False,
        "require_dynamic_vram": True,
        "minimum_current_headroom_mib": 512.0,
        "minimum_commit_headroom_gib": 16.0,
        "block_when_commit_below_gate": True,
        "policy_epoch": 0,
    }
    values.update(overrides)
    policy, report = vram.build_vram_policy(**values)
    return policy, report


def fake_runtime(events, *, dynamic=True):
    model_management = SimpleNamespace(
        EXTRA_RESERVED_VRAM=700 * vram.MIB,
        unload_all_models=lambda: events.append("unload_all_models"),
        soft_empty_cache=lambda: events.append("soft_empty_cache"),
    )
    memory_management = SimpleNamespace(aimdo_enabled=dynamic)
    library = SimpleNamespace(
        set_simple_vram_headroom=lambda value: events.append(
            ("set_simple_vram_headroom", value)
        )
    )
    control = SimpleNamespace(lib=library, devctxs=[object()])
    return model_management, memory_management, control


def test_planner_is_side_effect_free_and_fingerprint_is_tamper_evident(monkeypatch):
    policy, report = build_policy(monkeypatch)
    assert policy["schema"] == vram.VRAM_POLICY_SCHEMA
    assert report["current_gate_pass"] is True
    assert report["memory_safe_claim"] is False
    assert vram.policy_descriptor_fingerprint(policy) == policy["fingerprint"]

    tampered = dict(policy)
    tampered["fixed_total_reserved_gib"] = 3.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        vram.validate_vram_policy(tampered)


def test_external_usage_mode_requires_explicit_global_cleanup(monkeypatch):
    with pytest.raises(ValueError, match="requires clean_before_load=true"):
        build_policy(
            monkeypatch,
            mode="external_usage_plus_margin_exp",
            clean_before_load=False,
        )


def test_fixed_policy_sets_comfy_and_direct_aimdo_headroom_without_init(monkeypatch):
    policy, _report = build_policy(monkeypatch)
    events = []
    model_management, memory_management, control = fake_runtime(events)
    monkeypatch.setattr(
        vram,
        "_runtime_modules",
        lambda: (model_management, memory_management),
    )
    monkeypatch.setattr(vram, "_aimdo_control", lambda: control)
    monkeypatch.setattr(vram, "runtime_snapshot", lambda: snapshot())

    report = vram.apply_vram_policy(policy)
    expected = 2 * vram.GIB
    assert model_management.EXTRA_RESERVED_VRAM == expected
    assert events == [("set_simple_vram_headroom", expected)]
    assert report["model_management_route"] == "EXTRA_RESERVED_VRAM_bytes"
    assert report["dynamic_vram_route"] == (
        "direct_lib.set_simple_vram_headroom"
    )
    assert report["cleanup_performed"] is False
    assert report["memory_safe_claim"] is False


def test_auto_policy_cleans_before_measurement_and_caps_external_usage(monkeypatch):
    events = []
    model_management, memory_management, control = fake_runtime(events)
    monkeypatch.setattr(
        vram,
        "_runtime_modules",
        lambda: (model_management, memory_management),
    )
    monkeypatch.setattr(vram, "_aimdo_control", lambda: control)

    def current_snapshot():
        if "unload_all_models" in events:
            return snapshot(free_mib=14848.0)
        return snapshot(free_mib=12288.0)

    monkeypatch.setattr(vram, "runtime_snapshot", current_snapshot)
    policy, _report = vram.build_vram_policy(
        mode="external_usage_plus_margin_exp",
        fixed_total_reserved_gib=2.0,
        external_margin_gib=1.0,
        maximum_reserved_gib=2.0,
        clean_before_load=True,
        require_dynamic_vram=True,
        minimum_current_headroom_mib=512.0,
        minimum_commit_headroom_gib=16.0,
        block_when_commit_below_gate=True,
        policy_epoch=0,
    )
    report = vram.apply_vram_policy(policy)
    assert events[:2] == ["unload_all_models", "soft_empty_cache"]
    assert events[2] == ("set_simple_vram_headroom", 2 * vram.GIB)
    assert report["raw_target_reserved_gib"] == pytest.approx(2.5)
    assert report["target_reserved_gib"] == pytest.approx(2.0)
    assert report["target_capped"] is True
    assert report["cleanup_performed"] is True


def test_required_dynamic_vram_fails_before_cleanup_or_mutation(monkeypatch):
    policy, _report = build_policy(
        monkeypatch,
        mode="external_usage_plus_margin_exp",
        clean_before_load=True,
    )
    events = []
    model_management, memory_management, control = fake_runtime(
        events,
        dynamic=False,
    )
    before = model_management.EXTRA_RESERVED_VRAM
    monkeypatch.setattr(
        vram,
        "_runtime_modules",
        lambda: (model_management, memory_management),
    )
    monkeypatch.setattr(vram, "_aimdo_control", lambda: control)
    monkeypatch.setattr(vram, "runtime_snapshot", lambda: snapshot())
    with pytest.raises(RuntimeError, match="requires DynamicVRAM"):
        vram.apply_vram_policy(policy)
    assert events == []
    assert model_management.EXTRA_RESERVED_VRAM == before


def test_commit_gate_fails_before_global_cleanup(monkeypatch):
    policy, _report = build_policy(
        monkeypatch,
        clean_before_load=True,
        minimum_commit_headroom_gib=16.0,
    )
    events = []
    model_management, memory_management, control = fake_runtime(events)
    monkeypatch.setattr(
        vram,
        "_runtime_modules",
        lambda: (model_management, memory_management),
    )
    monkeypatch.setattr(vram, "_aimdo_control", lambda: control)
    monkeypatch.setattr(
        vram,
        "runtime_snapshot",
        lambda: snapshot(commit_gib=8.0),
    )
    with pytest.raises(RuntimeError, match="host commit headroom"):
        vram.apply_vram_policy(policy)
    assert events == []


def test_report_only_policy_never_cleans_or_mutates(monkeypatch):
    policy, _report = build_policy(
        monkeypatch,
        mode="report_only",
        clean_before_load=True,
    )
    monkeypatch.setattr(vram, "runtime_snapshot", lambda: snapshot())
    report = vram.apply_vram_policy(policy)
    assert report["applied"] is False
    assert report["cleanup_performed"] is False


def test_hybrid_loader_applies_policy_before_stock_model_load(tmp_path, monkeypatch):
    import comfy.sd

    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    events = []
    sentinel = object()
    policy = {"schema": "test-policy"}

    def apply_policy(value):
        events.append(("policy", value))
        return {"applied": True}

    def load_model(path, model_options):
        events.append(("load", path, model_options))
        return sentinel

    monkeypatch.setattr(hybrid, "apply_vram_policy", apply_policy)
    monkeypatch.setattr(comfy.sd, "load_diffusion_model", load_model)
    model, report = hybrid.load_hybrid_model(
        base,
        "base_only",
        "default",
        vram_policy=policy,
    )
    assert model is sentinel
    assert events == [
        ("policy", policy),
        ("load", str(base.resolve()), {}),
    ]
    assert report["vram_policy"] == {"applied": True}


def test_speech_preflight_uses_shared_dynamic_vram_snapshot(monkeypatch):
    monkeypatch.setattr(vram, "runtime_snapshot", lambda: snapshot())
    report = vram_preflight(512.0)
    assert report["dynamic_vram_enabled"] is True
    assert report["effective_extra_reserved_vram_gib"] == pytest.approx(0.683594)
    assert report["current_gate_pass"] is True
    assert report["memory_safe_claim"] is False


def test_vram_policy_api_example_orders_policy_before_loader():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (root / "examples" / "hybrid_model_vbar_headroom_api.json").read_text(
            encoding="utf-8"
        )
    )
    by_type = {
        node["class_type"]: (node_id, node)
        for node_id, node in workflow.items()
    }
    policy_id, policy = by_type["MiniMaxH3VRAMPolicyT8Advanced"]
    _loader_id, loader = by_type["MiniMaxH3HybridModelLoaderT8Advanced"]
    assert policy["inputs"]["mode"] == "fixed_total_reserved_exp"
    assert policy["inputs"]["fixed_total_reserved_gib"] == 4.0
    assert policy["inputs"]["clean_before_load"] is False
    assert policy["inputs"]["require_dynamic_vram"] is True
    assert loader["inputs"]["vram_policy"] == [policy_id, 0]


def test_process_resource_snapshot_exposes_cumulative_counters_without_mutation():
    resource = process_resource_snapshot()
    assert resource["pid"] == os.getpid()
    if resource["available"]:
        assert resource["rss_mib"] > 0
        assert resource["read_bytes"] is None or resource["read_bytes"] >= 0
        assert resource["page_faults"] is None or resource["page_faults"] >= 0
    else:
        assert "inspection_error" in resource


def test_vram_policy_frontend_workflow_is_link_consistent_and_isolated():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "H3_Hybrid_Model_VBAR_Headroom_Stock20_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    policy = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3VRAMPolicyT8Advanced"
    )
    loader = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3HybridModelLoaderT8Advanced"
    )
    assert policy["widgets_values"] == [
        "fixed_total_reserved_exp",
        4.0,
        1.0,
        8.0,
        False,
        True,
        512.0,
        16.0,
        True,
        0,
    ]
    policy_input = next(
        item for item in loader["inputs"] if item["name"] == "vram_policy"
    )
    assert links[policy_input["link"]][1:5] == [policy["id"], 0, loader["id"], 4]
    for node in nodes.values():
        for input_value in node.get("inputs", []):
            link_id = input_value.get("link")
            if link_id is not None:
                assert link_id in links
                assert links[link_id][3] == node["id"]
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert link_id in links
                assert links[link_id][1] == node["id"]
