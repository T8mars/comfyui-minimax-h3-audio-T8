from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import hybrid_compatibility as compatibility
from h3_audio_t8_pkg import hybrid_model as hybrid
from h3_audio_t8_pkg.nodes_hybrid_compatibility_advanced import (
    MiniMaxH3HybridCompatibilityAuditT8Advanced,
)


def runtime_snapshot(*, free_mib=4096.0, commit_gib=128.0, dynamic=True):
    return {
        "gpu": {
            "whole_device_free_mib": free_mib,
            "whole_device_total_mib": 16384.0,
        },
        "host": {"commit_headroom_gib": commit_gib},
        "comfy": {"dynamic_vram_enabled": dynamic},
        "aimdo": {},
    }


def canonical_identity(profile="blocks_25_49_video_audio_exp"):
    return {
        "schema": hybrid.ARTIFACT_SCHEMA,
        "algorithm": hybrid.ALGORITHM,
        "base_sha256": hybrid.KNOWN_QUALITY_BASE_SHA256,
        "overlay_sha256": hybrid.KNOWN_REFERENCE_OVERLAY_SHA256,
        "base_curve_sha256": hybrid.KNOWN_QUALITY_CURVE_SHA256,
        "overlay_curve_sha256": hybrid.KNOWN_REFERENCE_CURVE_SHA256,
        "base_file_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "overlay_file_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "recipe": hybrid.recipe_spec(profile),
    }


class FakePatcher:
    def __init__(self, profile="blocks_25_49_video_audio_exp"):
        identity = canonical_identity(profile)
        operations = hybrid._expected_operations(identity)
        payload_bytes = sum(2 * torch.tensor(op["shape"]).prod().item() for op in operations)
        self.attachments = {
            hybrid.ATTACHMENT_KEY: {
                "schema": hybrid.ARTIFACT_SCHEMA,
                "artifact_sha256": "a" * 64,
                "fingerprint": hybrid.sha256_bytes(
                    hybrid.canonical_json(identity).encode("utf-8")
                ),
                "identity": identity,
                "operation_count": len(operations),
                "payload_bytes": int(payload_bytes),
            }
        }
        self.patches = {}
        for operation in operations:
            key = operation["model_key"]
            entry = (
                1.0,
                ("set", (torch.zeros(operation["shape"], dtype=torch.float16),)),
                1.0,
                tuple(operation["offset"]),
                None,
            )
            self.patches.setdefault(key, []).append(entry)
        blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(50)]
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(blocks=blocks))
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def get_attachment(self, key):
        return self.attachments.get(key)

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name == "diffusion_model":
            return self.model.diffusion_model
        raise KeyError(name)

    def is_dynamic(self):
        return True


@pytest.fixture(autouse=True)
def stable_runtime(monkeypatch):
    monkeypatch.setattr(compatibility, "runtime_snapshot", runtime_snapshot)


def error_codes(report):
    return {value["code"] for value in report["hard_errors"]}


def test_exact_hybrid_patch_stack_passes_and_node_returns_same_model():
    model = FakePatcher()
    report = compatibility.audit_hybrid_compatibility(model)
    assert report["compatible"] is True
    assert report["status"] == "conditional"
    assert report["memory_safe_claim"] is False
    assert report["weight_patches"]["hybrid_entries_found"] == 100

    result = MiniMaxH3HybridCompatibilityAuditT8Advanced.execute(
        model,
        "report_only",
        False,
        512.0,
        16.0,
    )
    assert result[0] is model
    assert result[1] is True
    assert json.loads(result[2])["model_passthrough"] is True


def test_nonselected_lora_patch_passes_but_adaln_overlap_after_hybrid_fails():
    model = FakePatcher()
    model.patches["diffusion_model.blocks.0.attn.qkv_proj.weight"] = [
        (1.0, ("lora", (torch.zeros(1),)), 1.0, None, None)
    ]
    assert compatibility.audit_hybrid_compatibility(model)["compatible"] is True

    selected_key = next(iter(model.patches))
    model.patches[selected_key].append(
        (1.0, ("lora", (torch.zeros(1),)), 1.0, None, None)
    )
    report = compatibility.audit_hybrid_compatibility(model)
    assert report["compatible"] is False
    assert "adaln_patch_overlaps_hybrid" in error_codes(report)
    with pytest.raises(ValueError, match="adaln_patch_overlaps_hybrid"):
        MiniMaxH3HybridCompatibilityAuditT8Advanced.execute(
            model,
            "block_hard_conflicts",
            False,
            512.0,
            16.0,
        )


def test_patch_before_hybrid_and_missing_or_duplicate_set_fail_closed():
    before = FakePatcher()
    selected_key = next(iter(before.patches))
    before.patches[selected_key].insert(
        0, (1.0, ("lora", (torch.zeros(1),)), 1.0, None, None)
    )
    assert "patch_precedes_hybrid_set" in error_codes(
        compatibility.audit_hybrid_compatibility(before)
    )

    missing = FakePatcher()
    missing.patches[selected_key].pop(0)
    assert "hybrid_set_patch_missing" in error_codes(
        compatibility.audit_hybrid_compatibility(missing)
    )

    duplicate = FakePatcher()
    duplicate.patches[selected_key].append(duplicate.patches[selected_key][0])
    assert "hybrid_set_patch_duplicate" in error_codes(
        compatibility.audit_hybrid_compatibility(duplicate)
    )


def test_block_cache_complete_passes_and_incomplete_contract_fails():
    model = FakePatcher()
    cache = SimpleNamespace(
        config=SimpleNamespace(
            residual_diff_threshold=0.05,
            start_percent=0.1,
            end_percent=0.9,
            max_consecutive_hits=3,
            cache_device="cpu",
            metric_stride=1,
        )
    )
    transformer = model.model_options["transformer_options"]
    transformer[compatibility.BLOCK_CACHE_KEY] = cache
    transformer["patches_replace"] = {
        "dit": {("double_block", 0): object(), ("double_block", 49): object()}
    }
    model.wrappers = {
        "outer_sample": {compatibility.BLOCK_CACHE_WRAPPER_KEY: [object()]},
        "diffusion_model": {compatibility.BLOCK_CACHE_WRAPPER_KEY: [object()]},
    }
    report = compatibility.audit_hybrid_compatibility(model)
    assert report["compatible"] is True
    assert report["components"]["block_cache"]["complete"] is True

    del model.wrappers["diffusion_model"]
    report = compatibility.audit_hybrid_compatibility(model)
    assert "block_cache_contract_incomplete" in error_codes(report)


def _named(name):
    def value():
        pass

    value.__name__ = name
    return value


def test_sage_all_blocks_passes_partial_and_unknown_fail():
    model = FakePatcher()
    for index in range(50):
        model.object_patches[
            f"diffusion_model.blocks.{index}.attn.forward"
        ] = _named("minimax_sageattn_forward")
    assert compatibility.audit_hybrid_compatibility(model)["compatible"] is True

    model.object_patches.pop("diffusion_model.blocks.49.attn.forward")
    assert "sage_attention_contract_incomplete" in error_codes(
        compatibility.audit_hybrid_compatibility(model)
    )
    model.object_patches[
        "diffusion_model.blocks.49.attn.forward"
    ] = _named("foreign_attention_forward")
    report = compatibility.audit_hybrid_compatibility(model)
    assert "unknown_attention_forward_patch" in error_codes(report)


def conditioning(**metadata):
    return [[torch.zeros((1, 1, 1)), metadata]]


def test_long_video_and_multikeyframe_contracts_are_paired_and_mutually_exclusive():
    model = FakePatcher()

    def long_extra():
        pass

    long_extra._t8_long_video_patch_version = 1
    model.object_patches["extra_conds"] = long_extra
    positive = conditioning(t8_long_video_schema={})
    assert compatibility.audit_hybrid_compatibility(model, positive)["compatible"] is True

    mismatched = compatibility.audit_hybrid_compatibility(
        FakePatcher(), positive
    )
    assert "long_video_conditioning_model_mismatch" in error_codes(mismatched)

    def multi_extra():
        pass

    def multi_forward():
        pass

    multi_extra._t8_multikeyframe_patch_version = 1
    multi_forward._t8_multikeyframe_patch_version = 1
    model.object_patches["extra_conds"] = multi_extra
    model.object_patches["diffusion_model._forward"] = multi_forward
    multi_positive = conditioning(t8_multikeyframe_schema={})
    assert compatibility.audit_hybrid_compatibility(model, multi_positive)["compatible"] is True

    multi_extra._t8_long_video_patch_version = 1
    conflict = compatibility.audit_hybrid_compatibility(model, multi_positive)
    assert "long_video_multikeyframe_conflict" in error_codes(conflict)


def test_vram_policy_provenance_and_current_memory_gates(monkeypatch):
    model = FakePatcher()
    missing = compatibility.audit_hybrid_compatibility(
        model, require_applied_vram_policy=True
    )
    assert "vram_policy_required_missing" in error_codes(missing)

    model.attachments[hybrid.VRAM_POLICY_ATTACHMENT_KEY] = {
        "schema": "t8.minimax_h3.vram_policy_apply_report.v1",
        "applied": False,
        "mode": "report_only",
        "memory_safe_claim": False,
    }
    report_only = compatibility.audit_hybrid_compatibility(
        model, require_applied_vram_policy=True
    )
    assert "vram_policy_required_not_applied" in error_codes(report_only)

    model.attachments[hybrid.VRAM_POLICY_ATTACHMENT_KEY]["applied"] = True
    assert compatibility.audit_hybrid_compatibility(
        model, require_applied_vram_policy=True
    )["compatible"] is True

    monkeypatch.setattr(
        compatibility,
        "runtime_snapshot",
        lambda: runtime_snapshot(free_mib=500.0, commit_gib=15.0),
    )
    low = compatibility.audit_hybrid_compatibility(model)
    assert {
        "current_gpu_headroom_below_gate",
        "host_commit_headroom_below_gate",
    } <= error_codes(low)


def test_sampling_routes_are_identified_without_requiring_a_sampler_patch():
    model = FakePatcher()
    assert compatibility.audit_hybrid_compatibility(model)["components"]["sampling"][
        "route"
    ] == "stock_or_unpatched"

    Stable = type("MiniMaxH3FlowSampling", (), {})
    model.object_patches["model_sampling"] = Stable()
    assert compatibility.audit_hybrid_compatibility(model)["components"]["sampling"][
        "route"
    ] == "stable_dual_clock_or_native_av"

    Experimental = type("MiniMaxH3MultiRateFlowSamplingEXP", (), {})
    model.object_patches["model_sampling"] = Experimental()
    assert compatibility.audit_hybrid_compatibility(model)["components"]["sampling"][
        "route"
    ] == "experimental_multirate"


def test_loader_writes_slim_vram_provenance_before_return(tmp_path, monkeypatch):
    import comfy.sd

    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    model = FakePatcher()
    model.attachments = {}
    monkeypatch.setattr(
        hybrid,
        "apply_vram_policy",
        lambda _value: {
            "schema": "t8.minimax_h3.vram_policy_apply_report.v1",
            "policy_fingerprint": "policy",
            "mode": "fixed_total_reserved_exp",
            "applied": True,
            "cleanup_performed": False,
            "target_reserved_gib": 4.0,
            "dynamic_vram_route": "direct_lib.set_simple_vram_headroom",
            "memory_safe_claim": False,
            "before": {"huge": True},
        },
    )
    monkeypatch.setattr(comfy.sd, "load_diffusion_model", lambda *_args, **_kwargs: model)
    returned, report = hybrid.load_hybrid_model(
        base,
        "base_only",
        "default",
        vram_policy={"policy": True},
    )
    assert returned is model
    provenance = model.attachments[hybrid.VRAM_POLICY_ATTACHMENT_KEY]
    assert provenance["applied"] is True
    assert provenance["target_reserved_gib"] == 4.0
    assert "before" not in provenance
    assert report["vram_policy_attachment_written"] is True


def test_node_schema_is_appended_and_defaults_to_report_only():
    schema = MiniMaxH3HybridCompatibilityAuditT8Advanced.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert schema.node_id.endswith("Advanced")
    assert schema.is_experimental is True
    assert inputs["enforcement"].default == "report_only"
    assert inputs["positive"].optional is True
    assert MiniMaxH3HybridCompatibilityAuditT8Advanced.fingerprint_inputs() != (
        MiniMaxH3HybridCompatibilityAuditT8Advanced.fingerprint_inputs()
    )


def test_api_example_routes_final_sampler_model_through_audit():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (root / "tests" / "fixtures" / "api" / "hybrid_compatibility_audit_api.json").read_text(
            encoding="utf-8"
        )
    )
    by_type = {
        value["class_type"]: (key, value) for key, value in workflow.items()
    }
    sampler_id, _sampler = by_type["MiniMaxH3DualClockSamplerT8"]
    conditioning_id, _conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    audit_id, audit = by_type["MiniMaxH3HybridCompatibilityAuditT8Advanced"]
    _guider_id, guider = by_type["BasicGuider"]
    assert audit["inputs"]["model"] == [sampler_id, 0]
    assert audit["inputs"]["positive"] == [conditioning_id, 0]
    assert audit["inputs"]["enforcement"] == "report_only"
    assert guider["inputs"]["model"] == [audit_id, 0]


def test_frontend_example_routes_audited_model_and_has_consistent_links():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "H3_Hybrid_Compatibility_Audit_Stock20_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    assert len(nodes) == 16

    by_type = {node["type"]: node for node in nodes.values()}
    sampler = by_type["MiniMaxH3DualClockSamplerT8"]
    conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    audit = by_type["MiniMaxH3HybridCompatibilityAuditT8Advanced"]
    guider = by_type["BasicGuider"]
    policy = by_type["MiniMaxH3VRAMPolicyT8Advanced"]
    assert audit["widgets_values"] == ["report_only", True, 512.0, 16.0]
    assert policy["widgets_values"][:6] == [
        "fixed_total_reserved_exp",
        4.0,
        1.0,
        8.0,
        False,
        True,
    ]
    assert links[audit["inputs"][0]["link"]][1:5] == [
        sampler["id"], 0, audit["id"], 0,
    ]
    assert links[audit["inputs"][5]["link"]][1:5] == [
        conditioning["id"], 0, audit["id"], 5,
    ]
    assert links[guider["inputs"][0]["link"]][1:5] == [
        audit["id"], 0, guider["id"], 0,
    ]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
