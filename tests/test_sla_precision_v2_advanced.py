from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import h3_audio_t8_pkg.sla_precision_v2_advanced as precision
from h3_audio_t8_pkg.nodes_sla_precision_v2_advanced import (
    MiniMaxH3SLADynamicLoRABypassV2T8Advanced,
    MiniMaxH3SLAPrecisionV2AuditT8Advanced,
    MiniMaxH3SLAPrecisionV2T8Advanced,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas


class FakeModel:
    def __init__(self, *, shift_video=12.0, shift_audio=3.0):
        self.model_options = {
            "transformer_options": {
                "minimax_h3_sigma_shift_video": shift_video,
                "minimax_h3_sigma_shift_audio": shift_audio,
            }
        }
        self.attachments = {}
        self.patches = {}

    def clone(self):
        return FakeModel(
            shift_video=self.model_options["transformer_options"][
                "minimax_h3_sigma_shift_video"
            ],
            shift_audio=self.model_options["transformer_options"][
                "minimax_h3_sigma_shift_audio"
            ],
        )

    def set_attachments(self, key, value):
        self.attachments[key] = value


def test_recommended_schedule_requires_8nfe_12v_3a():
    report = precision._validate_schedule(
        FakeModel(), native_flow_sigmas(8, 12.0), precision.RECOMMENDED_SCHEDULE
    )
    assert report["nfe"] == 8
    assert report["status"] == "recommended_8nfe_12v_3a_valid"

    with pytest.raises(RuntimeError, match="8 NFE.*12/3"):
        precision._validate_schedule(
            FakeModel(shift_video=6.0),
            native_flow_sigmas(4, 6.0),
            precision.RECOMMENDED_SCHEDULE,
        )


def test_user_selected_schedule_is_explicit_and_bounded():
    report = precision._validate_schedule(
        FakeModel(shift_video=6.0),
        native_flow_sigmas(4, 6.0),
        precision.USER_SELECTED_SCHEDULE,
    )
    assert report["nfe"] == 4
    assert report["status"] == "user_selected_schedule_exp"


def test_dense_step_parser_is_strict_and_clamps_to_schedule():
    assert precision._parse_dense_steps("0,2-4,99,-1", 8) == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="Invalid dense_steps"):
        precision._parse_dense_steps("0,bad", 8)


def test_dynamic_lora_bypass_requires_one_hook_per_mapped_target(monkeypatch):
    patched = FakeModel()
    monkeypatch.setattr(
        precision,
        "_apply_authenticated_lora",
        lambda *_args, **_kwargs: (
            patched,
            {
                "mapped_patch_count": 208,
                "bypass_hook_count": 208,
                "base_weight_mutation": False,
            },
        ),
    )
    returned, report_json = precision.apply_sla_dynamic_lora_bypass(
        FakeModel(), "sla.safetensors"
    )
    assert returned is patched
    report = json.loads(report_json)
    assert report["status"] == "sla_dynamic_lora_bypass_applied"
    assert report["base_weight_mutation"] is False

    monkeypatch.setattr(
        precision,
        "_apply_authenticated_lora",
        lambda *_args, **_kwargs: (
            FakeModel(),
            {"mapped_patch_count": 208, "bypass_hook_count": 207},
        ),
    )
    with pytest.raises(RuntimeError, match="mapped=208, hooks=207"):
        precision.apply_sla_dynamic_lora_bypass(FakeModel(), "sla.safetensors")


def test_precision_patch_uses_current_plaguekind_defaults_and_keeps_state(
    monkeypatch,
):
    import h3_audio_t8_pkg.sla_precision_v2_vendor.patch as vendor_patch

    captured = {}
    state = {
        "calls": 0,
        "dense": 0,
        "n_steps": 0,
        "last_step_index": None,
        "summarized": False,
        "seq": 0,
        "kept": 0,
        "blocks": 0,
        "pinned": 0,
        "failed": None,
    }

    def fake_patch(model, **kwargs):
        captured.update(kwargs)
        return model.clone(), state

    monkeypatch.setattr(vendor_patch, "patch_h3_sla", fake_patch)
    patched, runtime, report_json = precision.patch_sla_precision_v2(
        FakeModel(), native_flow_sigmas(8, 12.0)
    )
    report = json.loads(report_json)
    assert captured["sparsity_ratio"] == 0.90
    assert captured["block_size"] == 32
    assert captured["min_seq_len"] == 8192
    assert captured["dense_last_steps"] == 1
    assert captured["dense_steps"] == "0"
    assert captured["protect_audio"] is True
    assert captured["disable_fp16_accum"] is True
    assert captured["return_state"] is True
    assert runtime.state is state
    assert patched.attachments[precision.RUNTIME_ATTACHMENT_KEY] is runtime
    assert report["attention"]["dense_step_indices"] == [0, 7]
    assert report["attention"]["sparse_step_indices"] == [1, 2, 3, 4, 5, 6]
    assert report["upstream"]["commit"] == precision.UPSTREAM_COMMIT


def _passing_runtime():
    config = {
        "schema": precision.SCHEMA,
        "schedule": {"nfe": 8},
        "attention": {
            "dense_step_indices": [0, 7],
            "sparse_step_indices": [1, 2, 3, 4, 5, 6],
            "min_seq_len": 8192,
            "protect_audio": True,
        },
    }
    state = {
        "calls": 300,
        "dense": 100,
        "n_steps": 8,
        "last_step_index": 7,
        "summarized": True,
        "seq": 12_785,
        "kept": 43,
        "blocks": 400,
        "pinned": 7,
        "failed": None,
        "step": 8,
        "dense_backend": "attention_comfy_kitchen_int8",
        "backend": "attention_xformers",
        "step_records": {
            index: {
                "wrapper_calls": 1,
                "expected_dense": index in {0, 7},
                "sparse_calls": 0 if index in {0, 7} else 50,
                "dense_calls": 50 if index in {0, 7} else 0,
                "kernel_fallbacks": 0,
                "max_sequence_tokens": 12_785,
                "selected_key_blocks": 43,
                "total_key_blocks": 400,
                "pinned_key_blocks": 7,
            }
            for index in range(8)
        },
    }
    return precision.SLAPrecisionV2Runtime(config=config, state=state)


def test_runtime_audit_requires_exact_sparse_calls_and_zero_fallback():
    latent = {"samples": object()}
    runtime = _passing_runtime()
    returned, report_json = precision.finalize_sla_precision_v2_runtime(
        latent, runtime
    )
    assert returned is latent
    report = json.loads(report_json)
    assert report["status"] == "precision_v2_mechanically_verified"
    assert all(report["checks"].values())
    assert report["observed"]["per_logical_step"]["0"]["dense_calls"] == 50
    assert report["observed"]["per_logical_step"]["1"]["sparse_calls"] == 50

    failed = _passing_runtime()
    failed.state["failed"] = "OutOfResources: test"
    with pytest.raises(RuntimeError, match="no_sparse_kernel_failure"):
        precision.finalize_sla_precision_v2_runtime(latent, failed)

    missing_step = _passing_runtime()
    del missing_step.state["step_records"][4]
    with pytest.raises(RuntimeError, match="all_logical_steps_observed"):
        precision.finalize_sla_precision_v2_runtime(latent, missing_step)


def test_precision_v2_nodes_are_append_only_and_default_to_reviewed_route():
    loader = MiniMaxH3SLADynamicLoRABypassV2T8Advanced.define_schema()
    attention = MiniMaxH3SLAPrecisionV2T8Advanced.define_schema()
    audit = MiniMaxH3SLAPrecisionV2AuditT8Advanced.define_schema()
    inputs = {item.id: item for item in attention.inputs}
    assert loader.node_id == "MiniMaxH3SLADynamicLoRABypassV2T8Advanced"
    assert attention.node_id == "MiniMaxH3SLAPrecisionV2T8Advanced"
    assert audit.node_id == "MiniMaxH3SLAPrecisionV2AuditT8Advanced"
    assert inputs["schedule_policy"].default == precision.RECOMMENDED_SCHEDULE
    assert inputs["sparsity_ratio"].default == 0.90
    assert inputs["block_size"].default == "32"
    assert inputs["dense_last_steps"].default == 1
    assert inputs["protect_audio"].default is True
    assert inputs["dense_steps"].default == "0"
    assert inputs["disable_fp16_accum"].default is True


def test_precision_v2_nodes_append_after_complete_v164_prefix():
    import h3_audio_t8_pkg

    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in classes]
    assert len(ids) == 284
    assert ids[276:279] == [
        "MiniMaxH3SLADynamicLoRABypassV2T8Advanced",
        "MiniMaxH3SLAPrecisionV2T8Advanced",
        "MiniMaxH3SLAPrecisionV2AuditT8Advanced",
    ]
    assert ids[279:281] == [
        "MiniMaxH3NativeMaskedVideoContextT8Advanced",
        "MiniMaxH3LongVideoColorMatchT8Advanced",
    ]
    features = json.loads(
        (Path(__file__).resolve().parents[1] / "features.json").read_text(
            encoding="utf-8"
        )
    )
    assert features["nodes"] == ids


def test_precision_v2_frontend_workflow_is_pinned_wired_and_reproducible():
    from tools.build_sla_precision_v2_workflow import build

    root = Path(__file__).resolve().parents[1]
    source_path = (
        root
        / "examples"
        / "workflows"
        / "15-sla-attention"
        / "2026-08-26_H3_Turbo_SLA_Profile_Router_FL2VA_Advanced_EXP.json"
    )
    path = (
        root
        / "examples"
        / "workflows"
        / "15-sla-attention"
        / "2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert build(source) == workflow
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert by_type["UNETLoader"]["widgets_values"] == [
        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "default",
    ]
    loader = by_type["MiniMaxH3SLADynamicLoRABypassV2T8Advanced"]
    attention = by_type["MiniMaxH3SLAPrecisionV2T8Advanced"]
    audit = by_type["MiniMaxH3SLAPrecisionV2AuditT8Advanced"]
    assert loader["widgets_values"] == [
        "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
    ]
    assert attention["widgets_values"] == [
        "recommended_8nfe_12v_3a",
        0.9,
        "32",
        8192,
        1,
        True,
        "0",
        "comfy_kitchen",
        True,
        False,
        False,
    ]
    links = {link[0]: link for link in workflow["links"]}
    model_to_attention = links[attention["inputs"][0]["link"]]
    sigmas_to_attention = links[attention["inputs"][1]["link"]]
    runtime_to_audit = links[audit["inputs"][1]["link"]]
    assert model_to_attention[1] == loader["id"]
    assert sigmas_to_attention[1] == by_type["MiniMaxH3DualClockSamplerT8"]["id"]
    assert runtime_to_audit[1] == attention["id"]
    assert runtime_to_audit[-1] == precision.RUNTIME_TYPE
    notes = "\n".join(
        node["widgets_values"]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "066ada9",
        "FP32",
        "首步与末步Dense",
        "300次Sparse",
        "211MiB",
        "Advanced EXP",
    ):
        assert required in notes


def test_real_validation_separates_mechanical_av_from_resource_failure():
    from tools.run_sla_precision_v2_validation import _gate_result

    mechanical_only = _gate_result(
        {"route": True},
        {"strict_decode": True},
        {"minimum_free_vram_at_least_512_mib": False},
    )
    assert mechanical_only == {
        "status": "MECHANICAL_AV_PASS_RESOURCE_GATE_FAIL_HUMAN_REVIEW_PENDING",
        "mechanical_av_pass": True,
        "resource_pass": False,
        "human_review_pass": False,
        "human_review_pending": True,
        "exit_success": False,
    }

    complete_mechanics_and_resource = _gate_result(
        {"route": True}, {"strict_decode": True}, {"vram": True}
    )
    assert complete_mechanics_and_resource["status"] == (
        "MECHANICAL_AV_AND_RESOURCE_PASS_HUMAN_REVIEW_PENDING"
    )
    assert complete_mechanics_and_resource["exit_success"] is True

    failed_mechanics = _gate_result(
        {"route": False}, {"strict_decode": True}, {"vram": True}
    )
    assert failed_mechanics["status"] == "FAIL_MECHANICAL_AV"
    assert failed_mechanics["human_review_pending"] is False
