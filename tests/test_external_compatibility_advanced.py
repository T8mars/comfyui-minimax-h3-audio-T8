from __future__ import annotations

import json
from pathlib import Path
import struct

import torch

from tools import build_clipproj_4b_workflow as clipproj_4b

from h3_audio_t8_pkg.external_compatibility_advanced import (
    _read_plugin_version,
    audit_clipproj_compatibility,
    audit_sol_attn_compatibility,
)


def _plugin_root(tmp_path: Path, kind: str, version: str) -> Path:
    root = tmp_path / "custom_nodes"
    plugin = root / kind
    plugin.mkdir(parents=True)
    if kind == "ComfyUI-ClipProj":
        (plugin / "clipproj_nodes.py").write_text("class ClipProjApply: pass\n", encoding="utf-8")
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname="comfyui-clipproj"\nversion="{version}"\n',
            encoding="utf-8",
        )
    else:
        (plugin / "minimax.py").write_text(
            "class MiniMaxH3ScheduledSolAttentionPatch: pass\n", encoding="utf-8"
        )
        (plugin / "README.md").write_text(f"Version: v{version}\n", encoding="utf-8")
    return root


def _projection_file(path: Path, source_dim: int, output_dim: int) -> Path:
    header = {
        "mean_in": {"dtype": "F32", "shape": [source_dim], "data_offsets": [0, 0]},
        "mean_out": {"dtype": "F32", "shape": [output_dim], "data_offsets": [0, 0]},
        "W": {
            "dtype": "F16",
            "shape": [source_dim, output_dim],
            "data_offsets": [0, 0],
        },
    }
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


def test_plugin_version_parser_accepts_upstream_markdown_bold_headers(tmp_path):
    plugin = tmp_path / "ComfyUI-sol-attn"
    plugin.mkdir()
    (plugin / "README.md").write_text("**Version: v0.6.2**\n", encoding="utf-8")
    assert _read_plugin_version(plugin) == "0.6.2"

    (plugin / "README.md").unlink()
    (plugin / "README_ZH.md").write_text("**版本: v0.6.2**\n", encoding="utf-8")
    assert _read_plugin_version(plugin) == "0.6.2"


def test_release_metadata_declares_optional_external_plugin_boundaries():
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    dependencies = "\n".join(meta["optional_runtime_dependencies"])
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "ComfyUI-ClipProj 0.1.13" in dependencies
    assert "c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c" in dependencies
    assert "ComfyUI-sol-attn 0.6.2" in dependencies
    assert "930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf" in dependencies
    assert "ComfyUI-ClipProj interoperability" in notices
    assert "MIT License" in notices
    assert "ComfyUI-sol-attn interoperability" in notices
    assert "Apache License 2.0" in notices


class ProjectedCLIP:
    def __init__(self, source_dim=2560, output_dim=5120):
        self._proj_name = "mmh3-4b-ClipProj-v3.1.safetensors"
        self._proj = {
            "mean_in": torch.zeros(source_dim),
            "mean_out": torch.zeros(output_dim),
        }


def test_clipproj_audit_accepts_reviewed_4b_contract_without_mutating_clip(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-ClipProj", "0.1.13")
    projection = _projection_file(tmp_path / "matrix.safetensors", 2560, 5120)
    clip = ProjectedCLIP()
    result = audit_clipproj_compatibility(
        clip,
        "4B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        str(projection),
        True,
        False,
        root,
    )
    report = json.loads(result[-1])
    assert result[0] is clip
    assert result[1:6] == (True, "PASS", "0.1.13", 2560, 5120)
    assert report["clip_passthrough_identity"] is True
    assert report["projection"]["has_linear_matrix"] is True


def test_clipproj_rejects_dimension_text_only_and_unreviewed_int8_dynamic(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-ClipProj", "0.1.13")
    projection = _projection_file(tmp_path / "matrix.safetensors", 4096, 5120)
    result = audit_clipproj_compatibility(
        ProjectedCLIP(4096),
        "4B",
        "text_only_qwen3",
        "int8_convrot",
        "clipproj_dynamic",
        str(projection),
        False,
        True,
        root,
    )
    codes = {item["code"] for item in json.loads(result[-1])["hard_findings"]}
    assert result[1] is False and result[2] == "ABSTAIN"
    assert "encoder_projection_dimension_mismatch" in codes
    assert "text_only_encoder_rejected" in codes
    assert "int8_dynamic_loader_unsupported" in codes


class _Attention:
    head_dim = 128


class _Block:
    def __init__(self):
        self.attn = _Attention()


class _Diffusion:
    def __init__(self):
        self.blocks = [_Block() for _ in range(50)]


class _Model:
    def __init__(self):
        self.diffusion = _Diffusion()
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion


def _sol_forward():
    def forward(*_args, **_kwargs):
        return None

    forward._minimax_h3_sol_fallback = object()
    return forward


def _hardware(capability=(8, 9)):
    return {
        "cuda_available": True,
        "compute_capability": list(capability),
        "bf16_supported": True,
    }


def _complete_sol_model():
    model = _Model()
    for index in range(50):
        model.object_patches[f"diffusion_model.blocks.{index}.attn.forward"] = _sol_forward()
    return model


def test_sol_audit_accepts_complete_h3_owner_and_supported_hardware(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-sol-attn", "0.6.2")
    model = _complete_sol_model()
    result = audit_sol_attn_compatibility(
        model, "h3_scheduled", "", False, root, _hardware()
    )
    report = json.loads(result[-1])
    assert result[0] is model
    assert result[1:5] == (True, "PASS", "0.6.2", 50)
    assert report["model_passthrough_identity"] is True


def test_sol_audit_abstains_on_shadowing_unreviewed_composition_and_sm75(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-sol-attn", "0.6.2")
    model = _complete_sol_model()
    model.object_patches["diffusion_model.blocks.49.attn.forward"] = lambda *_a, **_k: None
    model.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): object()}
    }
    result = audit_sol_attn_compatibility(
        model, "h3_memory_efficient", "", False, root, _hardware((7, 5))
    )
    codes = {item["code"] for item in json.loads(result[-1])["hard_findings"]}
    assert result[1] is False and result[2] == "ABSTAIN"
    assert "compute_capability_unsupported" in codes
    assert "h3_sol_patch_incomplete_or_shadowed" in codes
    assert "unreviewed_model_composition" in codes


def test_allow_unreviewed_composition_downgrades_only_that_risk(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-sol-attn", "0.6.2")
    model = _complete_sol_model()
    model.model_options["model_function_wrapper"] = lambda *_a, **_k: None
    result = audit_sol_attn_compatibility(
        model, "h3_scheduled", "", True, root, _hardware()
    )
    report = json.loads(result[-1])
    assert result[1] is True
    assert "unreviewed_model_composition" in {item["code"] for item in report["warnings"]}


def test_sol_audit_accepts_explicit_dense_first_last_blocks(tmp_path):
    root = _plugin_root(tmp_path, "ComfyUI-sol-attn", "0.6.2")
    model = _complete_sol_model()
    for index in (0, 1, 2, 49):
        model.object_patches.pop(f"diffusion_model.blocks.{index}.attn.forward")
    result = audit_sol_attn_compatibility(
        model, "h3_scheduled", "0-2,-1", False, root, _hardware()
    )
    assert result[1] is True
    assert json.loads(result[-1])["expected_dense_blocks"] == "0-2,-1"


def test_external_bridge_frontend_workflows_are_documented_and_wired():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "12-system-memory"
    )
    expected = {
        "2026-08-22_H3_ClipProj_Compatibility_Audit_Advanced_EXP.json": (
            "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
            ["8B", "qwen3_vl", "fp8", "stock_pageable"],
        ),
        "2026-08-22_H3_Sol_Attn_Compatibility_Audit_Advanced_EXP.json": (
            "MiniMaxH3SolAttnCompatibilityAuditT8Advanced",
            ["h3_scheduled", "", False, "report_only"],
        ),
    }
    for filename, (audit_type, widget_prefix) in expected.items():
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        audit = next(node for node in nodes.values() if node["type"] == audit_type)
        assert audit["widgets_values"][: len(widget_prefix)] == widget_prefix
        assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) >= 2
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_external_bridge_full_t2va_workflows_use_reviewed_assets_and_order():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "12-system-memory"
    )

    clip_workflow = json.loads(
        (root / "2026-08-22_H3_ClipProj_8B_T2VA_Bridge_Advanced_EXP.json").read_text(
            encoding="utf-8"
        )
    )
    clip_nodes = {node["id"]: node for node in clip_workflow["nodes"]}
    loader = next(node for node in clip_nodes.values() if node["type"] == "CLIPLoader")
    apply = next(node for node in clip_nodes.values() if node["type"] == "ClipProjApply")
    audit = next(
        node
        for node in clip_nodes.values()
        if node["type"] == "MiniMaxH3ClipProjCompatibilityAuditT8Advanced"
    )
    conditioning = next(
        node for node in clip_nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    assert loader["widgets_values"] == [
        "qwen3vl_8b_fp8_scaled.safetensors",
        "boogu",
        "default",
    ]
    assert apply["widgets_values"] == ["mmh3-8b-ClipProj-v3.1.safetensors"]
    assert audit["widgets_values"] == [
        "8B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-8b-ClipProj-v3.1.safetensors",
        False,
        False,
        "block_hard_conflicts",
    ]
    assert conditioning["widgets_values"][1:4] == [736, 416, 124]
    assert conditioning["inputs"][0]["link"] in (audit["outputs"][0]["links"] or [])

    clip4_workflow = json.loads(clipproj_4b.TARGET.read_text(encoding="utf-8"))
    assert clip4_workflow == clipproj_4b.build()
    clip4_nodes = {node["id"]: node for node in clip4_workflow["nodes"]}
    loader4 = next(node for node in clip4_nodes.values() if node["type"] == "CLIPLoader")
    apply4 = next(node for node in clip4_nodes.values() if node["type"] == "ClipProjApply")
    audit4 = next(
        node
        for node in clip4_nodes.values()
        if node["type"] == "MiniMaxH3ClipProjCompatibilityAuditT8Advanced"
    )
    conditioning4 = next(
        node for node in clip4_nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    assert loader4["widgets_values"] == [
        "qwen3vl_4b_fp8_scaled.safetensors",
        "krea2",
        "default",
    ]
    assert apply4["widgets_values"] == ["mmh3-4b-ClipProj-v3.1.safetensors"]
    assert audit4["widgets_values"] == [
        "4B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-4b-ClipProj-v3.1.safetensors",
        False,
        False,
        "block_hard_conflicts",
    ]
    assert conditioning4["widgets_values"][1:4] == [256, 256, 22]
    assert conditioning4["inputs"][0]["link"] in (audit4["outputs"][0]["links"] or [])
    notes4 = "\n".join(
        str(node.get("widgets_values", [""])[0])
        for node in clip4_nodes.values()
        if node.get("type") == "MarkdownNote" and node.get("widgets_values")
    )
    assert "qwen_3_4b.safetensors" in notes4
    assert clipproj_4b.ENCODER_SHA256 in notes4
    assert clipproj_4b.PROJECTION_SHA256 in notes4
    assert "单条真实运行已通过" in notes4
    assert "15,015MiB" in notes4
    assert clipproj_4b.RUNTIME_MEDIA_SHA256 in notes4
    assert clip4_workflow["extra"]["t8_clipproj_4b"]["status"] == (
        "ASSET_AND_SINGLE_T2VA_RUNTIME_PASS"
    )
    runtime_evidence = clip4_workflow["extra"]["t8_clipproj_4b"]["runtime_evidence"]
    assert runtime_evidence["run_id"] == clipproj_4b.RUNTIME_RUN_ID
    assert runtime_evidence["seed"] == 123456789
    assert runtime_evidence["peak_used_mib"] == 15015
    assert runtime_evidence["minimum_free_mib"] == 1095
    assert runtime_evidence["media_sha256"] == clipproj_4b.RUNTIME_MEDIA_SHA256

    sol_workflow = json.loads(
        (root / "2026-08-22_H3_Sol_Attn_T2VA_Conservative_Advanced_EXP.json").read_text(
            encoding="utf-8"
        )
    )
    sol_nodes = {node["id"]: node for node in sol_workflow["nodes"]}
    sol = next(
        node
        for node in sol_nodes.values()
        if node["type"] == "MiniMaxH3ScheduledSolAttentionPatch"
    )
    audit = next(
        node
        for node in sol_nodes.values()
        if node["type"] == "MiniMaxH3SolAttnCompatibilityAuditT8Advanced"
    )
    sampler = next(
        node for node in sol_nodes.values() if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sol["widgets_values"] == [
        True,
        1.3,
        0.8,
        "linear",
        4096,
        True,
        0.0,
        "diag",
        False,
        False,
        "exact_kv",
        "0-2,-1",
    ]
    assert audit["widgets_values"] == [
        "h3_scheduled",
        "0-2,-1",
        False,
        "block_hard_conflicts",
    ]
    assert sampler["inputs"][0]["link"] in (audit["outputs"][0]["links"] or [])
    notes = "\n".join(
        str(node.get("widgets_values", [""])[0])
        for node in sol_nodes.values()
        if node.get("type") == "MarkdownNote" and node.get("widgets_values")
    )
    assert "4步Turbo默认dense_percent=0.0" in notes
    assert "547-token" in notes
    assert "5139 tokens" in notes


def test_clipproj_i2va_bridge_enables_the_visual_contract_and_first_frame():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "12-system-memory"
    )
    workflow = json.loads(
        (root / "2026-08-22_H3_ClipProj_8B_I2VA_Bridge_Advanced_EXP.json").read_text(
            encoding="utf-8"
        )
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    audit = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3ClipProjCompatibilityAuditT8Advanced"
    )
    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    first_frame = next(node for node in nodes.values() if node["type"] == "LoadImage")

    assert audit["widgets_values"] == [
        "8B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-8b-ClipProj-v3.1.safetensors",
        True,
        False,
        "block_hard_conflicts",
    ]
    assert conditioning["widgets_values"][1:6] == [256, 256, 22, "I2VA", "native"]
    conditioning_inputs = {item["name"]: item for item in conditioning["inputs"]}
    assert conditioning_inputs["clip"]["link"] in (audit["outputs"][0]["links"] or [])
    assert conditioning_inputs["first_frame"]["link"] in (
        first_frame["outputs"][0]["links"] or []
    )
    assert "<Picture 1>" in conditioning["widgets_values"][0]
    notes = "\n".join(
        str(node.get("widgets_values", [""])[0])
        for node in nodes.values()
        if node.get("type") == "MarkdownNote" and node.get("widgets_values")
    )
    assert "has_reference_images" in notes
    assert "36.125" in notes
    assert "不等于32B质量无损替换" in notes


def test_clipproj_fl2va_bridge_enables_both_visual_keyframes():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "12-system-memory"
    )
    workflow = json.loads(
        (root / "2026-08-22_H3_ClipProj_8B_FL2VA_Bridge_Advanced_EXP.json").read_text(
            encoding="utf-8"
        )
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    audit = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3ClipProjCompatibilityAuditT8Advanced"
    )
    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    load_images = [node for node in nodes.values() if node["type"] == "LoadImage"]

    assert audit["widgets_values"] == [
        "8B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-8b-ClipProj-v3.1.safetensors",
        True,
        False,
        "block_hard_conflicts",
    ]
    assert conditioning["widgets_values"][1:6] == [256, 256, 22, "FL2VA", "native"]
    assert len(load_images) == 2
    conditioning_inputs = {item["name"]: item for item in conditioning["inputs"]}
    assert conditioning_inputs["clip"]["link"] in (audit["outputs"][0]["links"] or [])
    assert conditioning_inputs["first_frame"]["link"] in (
        load_images[0]["outputs"][0]["links"] or []
    )
    assert conditioning_inputs["last_frame"]["link"] in (
        load_images[1]["outputs"][0]["links"] or []
    )
    assert "<Picture 1>" in conditioning["widgets_values"][0]
    assert "<Picture 2>" in conditioning["widgets_values"][0]
    notes = "\n".join(
        str(node.get("widgets_values", [""])[0])
        for node in nodes.values()
        if node.get("type") == "MarkdownNote" and node.get("widgets_values")
    )
    assert "has_reference_images" in notes
    assert "23.953" in notes
    assert "只证明短双关键帧链可执行" in notes


def test_clipproj_ref2va_bridge_uses_reference_media_and_stock20():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "12-system-memory"
    )
    workflow = json.loads(
        (root / "2026-08-22_H3_ClipProj_8B_Ref2VA_Bridge_Advanced_EXP.json").read_text(
            encoding="utf-8"
        )
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    audit = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3ClipProjCompatibilityAuditT8Advanced"
    )
    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    sampler = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    reference = next(node for node in nodes.values() if node["type"] == "LoadImage")

    assert audit["widgets_values"] == [
        "8B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-8b-ClipProj-v3.1.safetensors",
        True,
        False,
        "block_hard_conflicts",
    ]
    assert conditioning["widgets_values"][1:6] == [256, 256, 22, "Ref2VA", "native"]
    assert sampler["widgets_values"][:3] == [20, 12.0, 3.0]
    conditioning_inputs = {item["name"]: item for item in conditioning["inputs"]}
    assert conditioning_inputs["clip"]["link"] in (audit["outputs"][0]["links"] or [])
    assert conditioning_inputs["ref_images.ref_image_0"]["link"] in (
        reference["outputs"][0]["links"] or []
    )
    assert "<Picture 1>" in conditioning["widgets_values"][0]
    assert not any(node["type"] == "LoraLoaderModelOnly" for node in nodes.values())
    notes = "\n".join(
        str(node.get("widgets_values", [""])[0])
        for node in nodes.values()
        if node.get("type") == "MarkdownNote" and node.get("widgets_values")
    )
    assert "Ref2VA参考token会在每个采样步重复读取" in notes
    assert "22.907" in notes and "27.812" in notes
    assert "memory_safe=false" in notes
