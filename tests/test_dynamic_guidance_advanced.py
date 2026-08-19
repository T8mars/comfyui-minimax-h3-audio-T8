from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.dynamic_guidance_advanced import (
    DynamicGuidanceRuntime,
    build_dynamic_guidance_guider,
    conditioning_layout_contract,
    dynamic_guidance_scale,
    finalize_dynamic_guidance_report,
)
from h3_audio_t8_pkg.nodes_dynamic_guidance_advanced import (
    DYNAMIC_GUIDANCE_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas


class DummyModel:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def is_dynamic(self):
        return False


def make_conditioning(*, tokens=4, frame_count=124, value=0.0):
    return [
        [
            torch.full((1, tokens, 8), value),
            {
                "minimax_frame_count": frame_count,
                "minimax_keyframes": [
                    {
                        "resolved_frame_index": 0,
                        "latent": torch.zeros((1, 24, 1, 8, 8)),
                    }
                ],
            },
        ]
    ]


def build_kwargs(**overrides):
    values = {
        "model": DummyModel(),
        "positive": make_conditioning(),
        "sigmas": native_flow_sigmas(8, 12.0),
        "mode": "passthrough_basic",
        "early_scale": 1.0,
        "late_scale": 1.0,
        "start_progress": 0.0,
        "end_progress": 1.0,
        "curve": "linear",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "profile": "turbo_standard8",
        "accept_true_cfg_cost": False,
        "accept_turbo_guidance_ood": False,
        "negative": None,
    }
    values.update(overrides)
    return values


def test_device_side_guidance_curve_has_expected_endpoints_and_direction():
    sigma = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    scale = dynamic_guidance_scale(
        sigma,
        early_scale=0.9,
        late_scale=1.1,
        start_progress=0.0,
        end_progress=1.0,
        curve="linear",
        shift_video=12.0,
    )
    assert scale.dtype == sigma.dtype
    assert scale.device == sigma.device
    assert scale[0].item() == pytest.approx(0.9)
    assert scale[-1].item() == pytest.approx(1.1)
    assert torch.all(scale[:-1] < scale[1:])
    assert ".cpu(" not in inspect.getsource(DynamicGuidanceRuntime.cfg_function)
    assert ".item(" not in inspect.getsource(DynamicGuidanceRuntime.cfg_function)


def test_passthrough_and_identity_curve_do_not_install_any_sampler_wrapper():
    for mode in ("passthrough_basic", "single_condition_gain_exp", "true_cfg_exp"):
        guider, runtime, report_json = build_dynamic_guidance_guider(
            **build_kwargs(mode=mode)
        )
        report = json.loads(report_json)
        assert report["effective_mode"] == "passthrough_basic"
        assert report["is_exact_basic_passthrough"] is True
        assert "sampler_cfg_function" not in guider.model_options
        assert "model_function_wrapper" not in guider.model_options
        assert runtime.cfg_callback_calls == 0


def test_single_condition_gain_installs_device_callback_and_reports_schedule():
    guider, runtime, report_json = build_dynamic_guidance_guider(
        **build_kwargs(
            mode="single_condition_gain_exp",
            early_scale=0.9,
            late_scale=1.1,
            accept_turbo_guidance_ood=True,
        )
    )
    report = json.loads(report_json)
    assert report["effective_mode"] == "single_condition_gain_exp"
    assert report["single_condition_gain_not_true_cfg"] is True
    assert report["expected_nfe"] == 8
    assert report["expected_scales"][0] == pytest.approx(0.9)
    assert report["expected_scales"][-1] == pytest.approx(1.1)
    assert guider.model_options["sampler_cfg_function"] == runtime.cfg_function
    assert guider.model_options["model_function_wrapper"] == runtime.model_function_wrapper
    assert "disable_cfg1_optimization" not in guider.model_options

    cond = torch.tensor([[[[2.0]]]])
    uncond = torch.tensor([[[[1.0]]]])
    guided = runtime.cfg_function(
        {"cond": cond, "uncond": uncond, "sigma": torch.tensor([1.0])}
    )
    assert torch.equal(guided, torch.tensor([[[[1.9]]]]))


def test_dynamic_modes_fail_closed_on_ood_consent_and_existing_wrappers():
    with pytest.raises(ValueError, match="accept_turbo_guidance_ood"):
        build_dynamic_guidance_guider(
            **build_kwargs(
                mode="single_condition_gain_exp", early_scale=0.9, late_scale=1.1
            )
        )
    with pytest.raises(ValueError, match="existing sampler/model wrappers"):
        build_dynamic_guidance_guider(
            **build_kwargs(
                model=DummyModel({"sampler_cfg_function": lambda args: args["cond"]}),
                mode="single_condition_gain_exp",
                early_scale=0.9,
                late_scale=1.1,
                accept_turbo_guidance_ood=True,
            )
        )


def test_true_cfg_requires_cost_consent_and_identical_h3_layout():
    negative = make_conditioning(value=1.0)
    with pytest.raises(ValueError, match="accept_true_cfg_cost"):
        build_dynamic_guidance_guider(
            **build_kwargs(
                mode="true_cfg_exp",
                early_scale=0.9,
                late_scale=1.1,
                negative=negative,
                accept_turbo_guidance_ood=True,
            )
        )
    with pytest.raises(ValueError, match="identical H3 embedding shape"):
        build_dynamic_guidance_guider(
            **build_kwargs(
                mode="true_cfg_exp",
                early_scale=0.9,
                late_scale=1.1,
                negative=make_conditioning(tokens=5),
                accept_true_cfg_cost=True,
                accept_turbo_guidance_ood=True,
            )
        )

    guider, _, report_json = build_dynamic_guidance_guider(
        **build_kwargs(
            mode="true_cfg_exp",
            early_scale=0.9,
            late_scale=1.1,
            negative=negative,
            accept_true_cfg_cost=True,
            accept_turbo_guidance_ood=True,
        )
    )
    report = json.loads(report_json)
    assert report["effective_mode"] == "true_cfg_exp"
    assert report["expected_condition_branches_per_step"] == 2
    assert report["true_cfg_validated"] is False
    assert guider.model_options["disable_cfg1_optimization"] is True


def test_runtime_audit_counts_physical_forward_batches_without_changing_latent():
    _, runtime, _ = build_dynamic_guidance_guider(
        **build_kwargs(
            mode="single_condition_gain_exp",
            early_scale=0.9,
            late_scale=1.1,
            accept_turbo_guidance_ood=True,
        )
    )
    source = torch.zeros((2, 3))

    def apply_model(input_tensor, timestep, **kwargs):
        assert kwargs["marker"] == "kept"
        return input_tensor + timestep

    output = runtime.model_function_wrapper(
        apply_model,
        {
            "input": source,
            "timestep": torch.tensor(2.0),
            "c": {"marker": "kept"},
            "cond_or_uncond": [0, 1],
        },
    )
    assert torch.equal(output, source + 2.0)
    runtime.record_predict_noise()
    latent = {"samples": torch.zeros((1, 4, 2, 2))}
    returned, report_json = finalize_dynamic_guidance_report(latent, runtime)
    report = json.loads(report_json)
    assert returned is latent
    assert report["actual_predict_noise_calls"] == 1
    assert report["actual_physical_model_forward_calls"] == 1
    assert report["actual_cond_branch_evaluations"] == 1
    assert report["actual_uncond_branch_evaluations"] == 1
    assert report["actual_forward_branch_batches"] == [[0, 1]]


def test_conditioning_layout_fingerprint_ignores_text_values_but_not_layout():
    first = conditioning_layout_contract(make_conditioning(value=0.0))
    second = conditioning_layout_contract(make_conditioning(value=1.0))
    different = conditioning_layout_contract(make_conditioning(frame_count=90))
    assert first["sha256"] == second["sha256"]
    assert first["sha256"] != different["sha256"]


def test_node_schemas_are_append_only_advanced_and_safe_by_default():
    assert len(DYNAMIC_GUIDANCE_ADVANCED_NODE_CLASSES) == 2
    schemas = [node.define_schema() for node in DYNAMIC_GUIDANCE_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3DynamicCFGGuiderT8Advanced",
        "MiniMaxH3DynamicGuidanceAuditT8Advanced",
    ]
    assert all(schema.node_id.endswith("Advanced") for schema in schemas)
    assert all(schema.is_experimental for schema in schemas)
    assert schemas[1].is_output_node is True
    inputs = {item.id: item for item in schemas[0].inputs}
    assert inputs["mode"].default == "passthrough_basic"
    assert inputs["early_scale"].default == 1.0
    assert inputs["late_scale"].default == 1.0
    assert inputs["accept_true_cfg_cost"].default is False
    assert inputs["accept_turbo_guidance_ood"].default is False


def _assert_frontend_links_are_bidirectional(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    for link_id, source, output_slot, target, input_slot, link_type in workflow[
        "links"
    ]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_dynamic_guidance_api_and_frontend_examples_are_safe_and_complete():
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (
            root
            / "tests"
            / "fixtures"
            / "api"
            / "motion_quality_dynamic_guidance_8step_api.json"
        ).read_text(encoding="utf-8")
    )
    dynamic_id, dynamic = next(
        (key, node)
        for key, node in api.items()
        if node["class_type"] == "MiniMaxH3DynamicCFGGuiderT8Advanced"
    )
    audit_id, audit = next(
        (key, node)
        for key, node in api.items()
        if node["class_type"] == "MiniMaxH3DynamicGuidanceAuditT8Advanced"
    )
    sampler = next(
        node for node in api.values() if node["class_type"] == "SamplerCustomAdvanced"
    )
    decode = next(
        node for node in api.values() if node["class_type"] == "MiniMaxH3AVDecodeT8"
    )
    assert dynamic["inputs"]["mode"] == "passthrough_basic"
    assert dynamic["inputs"]["early_scale"] == 1.0
    assert dynamic["inputs"]["late_scale"] == 1.0
    assert dynamic["inputs"]["accept_true_cfg_cost"] is False
    assert dynamic["inputs"]["accept_turbo_guidance_ood"] is False
    assert sampler["inputs"]["guider"] == [dynamic_id, 0]
    assert audit["inputs"]["runtime"] == [dynamic_id, 1]
    assert decode["inputs"]["av_latent"] == [audit_id, 0]

    frontend = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "07-motion-detail"
            / "2026-08-09_H3_Motion_Quality_Dynamic_Guidance_8Step_EXP.json"
        ).read_text(encoding="utf-8")
    )
    _assert_frontend_links_are_bidirectional(frontend)
    nodes = {node["id"]: node for node in frontend["nodes"]}
    dynamic_frontend = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3DynamicCFGGuiderT8Advanced"
    )
    audit_frontend = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3DynamicGuidanceAuditT8Advanced"
    )
    note = next(node for node in nodes.values() if node["type"] == "MarkdownNote")
    assert dynamic_frontend["widgets_values"][:3] == ["passthrough_basic", 1.0, 1.0]
    assert "不是真正双分支CFG" in note["widgets_values"][0]
    decode_frontend = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AVDecodeT8"
    )
    assert decode_frontend["inputs"][0]["link"] in (
        audit_frontend["outputs"][0]["links"] or []
    )


def test_extra_tail_api_and_frontend_examples_are_default_off_and_explain_cost():
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (
            root
            / "tests"
            / "fixtures"
            / "api"
            / "motion_quality_extra_tail_nfe_8step_api.json"
        ).read_text(encoding="utf-8")
    )
    tail = next(
        node
        for node in api.values()
        if node["class_type"] == "MiniMaxH3AVSigmaTailSubdivisionT8Advanced"
    )
    assert tail["inputs"]["mode"] == "report_only"
    assert tail["inputs"]["extra_substeps"] == 0
    assert tail["inputs"]["tail_intervals"] == 2
    assert tail["inputs"]["accept_turbo_schedule_ood"] is False

    frontend = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "07-motion-detail"
            / "2026-08-09_H3_Motion_Quality_Extra_Tail_NFE_8Step_EXP.json"
        ).read_text(encoding="utf-8")
    )
    _assert_frontend_links_are_bidirectional(frontend)
    nodes = {node["id"]: node for node in frontend["nodes"]}
    tail_frontend = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3AVSigmaTailSubdivisionT8Advanced"
    )
    note = next(node for node in nodes.values() if node["type"] == "MarkdownNote")
    assert tail_frontend["widgets_values"][:4] == [
        "report_only",
        0,
        "tail_intervals",
        2,
    ]
    assert "8次增加到10次真实联合A/V DiT前向" in note["widgets_values"][0]
