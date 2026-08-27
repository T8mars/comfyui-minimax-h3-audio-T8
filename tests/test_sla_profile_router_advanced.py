from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import h3_audio_t8_pkg.sla_attention_advanced as sla
from h3_audio_t8_pkg.nodes_sla_profile_router_advanced import (
    MiniMaxH3TurboSLAProfileRouterT8Advanced,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from h3_audio_t8_pkg.sla_profile_router_advanced import (
    CONSUMER_TURBO_PROFILE,
    CORRECTED_TURBO_LORA_FILENAME,
    SLA_EXACT_PROFILE,
    SLA_INT8_BYPASS_END_PERCENT,
    SLA_INT8_BYPASS_PROFILE,
    SLA_INT8_BYPASS_START_PERCENT,
    _classify_int8_sla_bypass_base,
    _classify_published_sla_base,
    _validate_consumer_turbo_lora_header,
    _validate_profile_schedule,
)


def test_consumer_profile_requires_validated_turbo8_12_3_schedule():
    report = _validate_profile_schedule(
        native_flow_sigmas(8, 12.0),
        profile=CONSUMER_TURBO_PROFILE,
        shift_video=12.0,
        shift_audio=3.0,
    )
    assert report["nfe"] == 8
    assert report["profile_schedule_status"] == "consumer_turbo8_12v_3a_validated"

    with pytest.raises(RuntimeError, match="8 NFE.*12/3"):
        _validate_profile_schedule(
            native_flow_sigmas(4, 6.0),
            profile=CONSUMER_TURBO_PROFILE,
            shift_video=6.0,
            shift_audio=3.0,
        )


def test_sla_profile_requires_published_4step_6_3_schedule():
    report = _validate_profile_schedule(
        native_flow_sigmas(4, 6.0),
        profile=SLA_EXACT_PROFILE,
        shift_video=6.0,
        shift_audio=3.0,
    )
    assert report["nfe"] == 4
    assert report["profile_schedule_status"] == "sla_upstream_4nfe_6v_3a_exact_exp"

    with pytest.raises(RuntimeError, match="4 NFE.*6/3"):
        _validate_profile_schedule(
            native_flow_sigmas(8, 12.0),
            profile=SLA_EXACT_PROFILE,
            shift_video=12.0,
            shift_audio=3.0,
        )


def test_int8_bypass_profile_requires_same_published_schedule():
    report = _validate_profile_schedule(
        native_flow_sigmas(4, 6.0),
        profile=SLA_INT8_BYPASS_PROFILE,
        shift_video=6.0,
        shift_audio=3.0,
    )
    assert report["profile_schedule_status"] == "sla_int8_bypass_4nfe_6v_3a_exp"


def test_int8_bypass_profile_reports_non_reference_bases_without_blocking():
    assert (
        _classify_int8_sla_bypass_base(
            {
                "observed_quant_format": "int8_tensorwise",
                "observed_weight_type": "QuantizedTensor",
                "observed_convrot": True,
                "lora_target_quantization": {
                    "main_target_count": 200,
                    "main_int8_convrot_count": 200,
                    "main_unquantized_count": 0,
                    "token_refiner_target_count": 8,
                    "token_refiner_int8_convrot_count": 0,
                    "token_refiner_unquantized_count": 8,
                },
            }
        )
        == "comfyui_int8_convrot_bypass_experiment"
    )
    for contract in (
        {
            "observed_quant_format": "int8_tensorwise",
            "observed_weight_type": "QuantizedTensor",
            "observed_convrot": False,
            "lora_target_quantization": {},
        },
        {
            "observed_quant_format": "fp8-sgl",
            "observed_weight_type": "QuantizedTensor",
            "observed_convrot": True,
            "lora_target_quantization": {},
        },
    ):
        result = _classify_int8_sla_bypass_base(contract)
        assert result.startswith("user_selected_unvalidated_bypass_base:")


def test_sla_bypass_application_never_calls_standard_weight_patch(monkeypatch, tmp_path):
    adapter = sla.comfy.weight_adapter.WeightAdapterBase()

    class FakeBase:
        def state_dict(self):
            return {"linear.weight": torch.zeros(1)}

    class FakePatcher:
        def __init__(self):
            self.model = FakeBase()
            self.injections = None
            self.attachments = None

        def clone(self):
            return FakePatcher()

        def add_patches(self, *_args, **_kwargs):
            raise AssertionError("standard weight patch must not run in bypass mode")

        def set_injections(self, key, injections):
            self.injections = (key, injections)

        def set_attachments(self, key, value):
            self.attachments = (key, value)

    class FakeManager:
        def __init__(self):
            self.adapters = []

        def add_adapter(self, key, value, strength):
            self.adapters.append((key, value, strength))

        def create_injections(self, _model):
            return ["dynamic-bypass-injection"]

        def get_hook_count(self):
            return len(self.adapters)

    monkeypatch.setattr(
        sla, "_validate_lora_header", lambda _path: {"patch_count": 1}
    )
    monkeypatch.setattr(
        sla.comfy.utils,
        "load_torch_file",
        lambda *_args, **_kwargs: ({"raw": torch.zeros(1)}, {"source": "test"}),
    )
    monkeypatch.setattr(sla.comfy.lora_convert, "convert_lora", lambda state: state)
    monkeypatch.setattr(
        sla.comfy.lora,
        "model_lora_keys_unet",
        lambda _model, _keys: {"linear.weight": "linear.weight"},
    )
    monkeypatch.setattr(
        sla.comfy.lora,
        "load_lora",
        lambda *_args, **_kwargs: {"linear.weight": adapter},
    )
    monkeypatch.setattr(
        sla.comfy.weight_adapter, "BypassInjectionManager", FakeManager
    )

    patched, contract = sla._apply_authenticated_lora(
        FakePatcher(),
        tmp_path / "structurally-validated.safetensors",
        application_policy="bypass_model_only",
    )

    assert patched.injections == (
        "bypass_lora",
        ["dynamic-bypass-injection"],
    )
    assert contract["application_mode"] == "comfyui_bypass_model_only"
    assert contract["base_weight_mutation"] is False
    assert contract["bypass_hook_count"] == 1


def test_quality_sla_profile_reports_released_and_user_selected_base_families():
    assert (
        _classify_published_sla_base(
            {
                "official_bf16_base_observed": True,
                "observed_quant_format": None,
                "observed_weight_type": "Tensor",
            }
        )
        == "bf16_checkpoint_family"
    )
    assert (
        _classify_published_sla_base(
            {
                "official_bf16_base_observed": False,
                "observed_quant_format": "fp8-sgl",
                "observed_weight_type": "QuantizedTensor",
            }
        )
        == "lightx2v_fp8_recipe_family"
    )
    assert _classify_published_sla_base(
        {
            "official_bf16_base_observed": False,
            "observed_quant_format": "int8_tensorwise",
            "observed_weight_type": "QuantizedTensor",
        }
    ).startswith("user_selected_unvalidated_base:")


def test_consumer_lora_header_reports_corrected_alpha8_reference_without_blocking(tmp_path):
    path = tmp_path / "community_corrected_alpha8.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 8),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(8, 4),
        },
        str(path),
        metadata={
            "base_model": "MiniMax-H3",
            "peft_lora_alpha": "8",
            "effective_lora_scale": "0.0625",
            "comfyui_loader": "Load LoRA (Bypass, Model Only) (for debugging)",
            "sampler_steps": "4",
        },
    )
    report = _validate_consumer_turbo_lora_header(path)
    assert report["patch_count"] == 1
    assert report["application_mode"] == "comfyui_bypass_model_only"
    assert report["file_sha256_enforced"] is False

    bad = tmp_path / "sla_not_consumer_fallback.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 8),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(8, 4),
        },
        str(bad),
        metadata={
            "base_model": "MiniMax-H3",
            "training_alpha": "128.0",
            "training_scale": "1.0",
        },
    )
    bad_report = _validate_consumer_turbo_lora_header(bad)
    assert bad_report["corrected_alpha8_reference_match"] is False
    assert bad_report["model_identity_policy"] == "diagnostic_only_not_a_load_gate"


def test_existing_runtime_audit_accepts_consumer_profile_without_fake_sla_calls():
    runtime = sla.SLARuntime(
        {
            "mode": CONSUMER_TURBO_PROFILE,
            "sigma_contract": {"nfe": 8},
            "profile_contract": {"attention_backend": "native_dense"},
        }
    )
    for _ in range(8):
        runtime.begin_forward(
            {
                "task": "FL2VA",
                "seq_len": 12_587,
                "pixel_frames": 124,
                "latent_t": 37,
                "latent_h": 26,
                "latent_w": 46,
            }
        )
    latent = {"samples": torch.zeros(1)}
    returned, report_json = sla.finalize_sla_runtime(latent, runtime)
    assert returned is latent
    report = json.loads(report_json)
    assert report["status"] == "consumer_turbo8_profile_mechanically_verified"
    assert report["model_forward_count"] == 8
    assert report["main_attention_calls_per_forward"] == [0] * 8


def test_profile_router_schema_defaults_to_consumer_turbo_and_is_append_only():
    schema = MiniMaxH3TurboSLAProfileRouterT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3TurboSLAProfileRouterT8Advanced"
    assert schema.is_experimental is True
    assert inputs["profile"].default == CONSUMER_TURBO_PROFILE
    assert inputs["turbo_lora_name"].default == CORRECTED_TURBO_LORA_FILENAME
    assert inputs["sla_lora_name"].default == sla.SLA_LORA_FILENAME
    assert inputs["profile"].options[-1] == SLA_INT8_BYPASS_PROFILE
    assert inputs["sla_start_percent"].default == SLA_INT8_BYPASS_START_PERCENT
    assert inputs["sla_end_percent"].default == SLA_INT8_BYPASS_END_PERCENT


def test_profile_router_frontend_workflow_is_importable_and_defaults_to_turbo8():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "15-sla-attention"
        / "2026-08-26_H3_Turbo_SLA_Profile_Router_FL2VA_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    router = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3TurboSLAProfileRouterT8Advanced"
    )
    dual_clock = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    load_images = [
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "LoadImage"
    ]
    notes = [
        node["widgets_values"]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    ]
    assert dual_clock["widgets_values"][:3] == [8, 12.0, 3.0]
    assert router["widgets_values"] == [
        CORRECTED_TURBO_LORA_FILENAME,
        sla.SLA_LORA_FILENAME,
        CONSUMER_TURBO_PROFILE,
        "auto_detect_exp",
        512,
        0.15,
        0.9,
    ]
    assert len(load_images) == 2
    assert load_images == [
        "codex_prompt_relay_fl2va_first.png",
        "codex_prompt_relay_fl2va_first.png",
    ]
    assert len(notes) >= 3
    assert any("4/6/3" in note and "8 NFE" in note for note in notes)
    assert any("同一张近景图" in note and "完整人审否决" in note for note in notes)
    assert any("1344×768×362" in note and "INT8短视频" in note for note in notes)
    assert any("15%～90%" in note and "dense/sparse/sparse/sparse" in note for note in notes)
    node_ids = set(nodes)
    for link in workflow["links"]:
        assert link[1] in node_ids
        assert link[3] in node_ids
