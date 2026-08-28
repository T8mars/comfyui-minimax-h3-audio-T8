from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "examples" / "workflows" / "18-audio-refine"
WORKFLOWS = {
    "turbo8": "2026-08-29_H3_Audio_Refine_Turbo8_Plus_Refine4_Advanced_EXP.json",
    "learned_latent_twopass_final8": "2026-08-29_H3_Audio_Refine_Learned_TwoPass_Final8_Advanced_EXP.json",
    "pdd8": "2026-08-29_H3_Audio_Refine_PDD_Ref2VA8_Advanced_EXP.json",
    "pdd4_plus4": "2026-08-29_H3_Audio_Refine_PDD_Ref2VA_4Plus4_Advanced_EXP.json",
    "eav_turbo8": "2026-08-29_H3_Audio_Refine_EAV_Turbo8_Advanced_EXP.json",
    "prompt_relay_turbo8": "2026-08-29_H3_Audio_Refine_Prompt_Relay_Turbo8_Advanced_EXP.json",
    "long_video_prompt_relay_turbo8": "2026-08-29_H3_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_Advanced_EXP.json",
}


def load(profile: str) -> dict:
    return json.loads((WORKFLOW_ROOT / WORKFLOWS[profile]).read_text(encoding="utf-8"))


def nodes_by_type(workflow: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for node in workflow["nodes"]:
        result.setdefault(node["type"], []).append(node)
    return result


@pytest.mark.parametrize("profile", sorted(WORKFLOWS))
def test_audio_refine_compat_workflows_are_frontend_graphs(profile):
    workflow = load(profile)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = nodes_by_type(workflow)
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len({link[0] for link in workflow["links"]}) == len(workflow["links"])
    assert len(types["MiniMaxH3AudioRefineCompatibilityRouteT8Advanced"]) == 1
    assert len(types["MiniMaxH3AudioRefineCompatibilityPlanT8Advanced"]) == 1
    assert len(types["MiniMaxH3AudioRefineCompatibilitySetupT8Advanced"]) == 1
    assert len(types["MiniMaxH3AudioRefineQualityGateT8Advanced"]) == 1
    assert len(types["MarkdownNote"]) >= 1
    route = types["MiniMaxH3AudioRefineCompatibilityRouteT8Advanced"][0]
    gate = types["MiniMaxH3AudioRefineQualityGateT8Advanced"][0]
    assert route["widgets_values"] == [profile, 4 if profile == "turbo4" else 8]
    assert gate["widgets_values"][0] is False
    for link_id, origin, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[origin]["outputs"][output_slot].get("links") or [])
        assert nodes[origin]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_learned_two_pass_refines_only_sampler_pass2_output():
    workflow = load("learned_latent_twopass_final8")
    types = nodes_by_type(workflow)
    audit = types["MiniMaxH3AudioRefineAuditT8Advanced"][0]
    audit_link = audit["inputs"][2]["link"]
    link = next(item for item in workflow["links"] if item[0] == audit_link)
    assert link[1:3] == [19, 0]
    assert len(types["MiniMaxH3LearnedLatentUpscaleT8Advanced"]) == 1
    assert len(types["MiniMaxH3TwoPassDetailMixerT8Advanced"]) == 1


def test_pdd_and_eav_are_not_reapplied_in_refine_sidecar():
    cases = {
        "pdd8": ("MiniMaxH3PDD8StepSetupT8Advanced", 1),
        "pdd4_plus4": ("MiniMaxH3PDD8StepSetupT8Advanced", 4),
        "eav_turbo8": ("MiniMaxH3EnhanceAVideoT8Advanced", 1),
    }
    for profile, (generation_type, base_model_id) in cases.items():
        workflow = load(profile)
        types = nodes_by_type(workflow)
        assert len(types[generation_type]) == 1
        route = types["MiniMaxH3AudioRefineCompatibilityRouteT8Advanced"][0]
        model_link = next(
            link for link in workflow["links"] if link[0] == route["inputs"][1]["link"]
        )
        assert model_link[1:3] == [base_model_id, 0]


def test_pdd_four_plus_four_refines_only_after_second_pass():
    workflow = load("pdd4_plus4")
    types = nodes_by_type(workflow)
    audit = types["MiniMaxH3AudioRefineAuditT8Advanced"][0]
    audit_link = audit["inputs"][2]["link"]
    link = next(item for item in workflow["links"] if item[0] == audit_link)
    assert link[1:3] == [19, 0]
    assert len(types["SamplerCustomAdvanced"]) == 3
    assert len(types["MiniMaxH3PDD8StepSetupT8Advanced"]) == 1


def test_prompt_relay_refine_reuses_authenticated_conditioning_model():
    for profile, conditioning_id in (
        ("prompt_relay_turbo8", 6),
        ("long_video_prompt_relay_turbo8", 9),
    ):
        workflow = load(profile)
        route = nodes_by_type(workflow)[
            "MiniMaxH3AudioRefineCompatibilityRouteT8Advanced"
        ][0]
        model_link = next(
            link for link in workflow["links"] if link[0] == route["inputs"][1]["link"]
        )
        positive_link = next(
            link for link in workflow["links"] if link[0] == route["inputs"][2]["link"]
        )
        assert model_link[1:3] == [conditioning_id, 0]
        assert positive_link[1:3] == [conditioning_id, 1]


def test_long_video_split_keeps_original_context_and_separate_delivery():
    workflow = load("long_video_prompt_relay_turbo8")
    types = nodes_by_type(workflow)
    split = types["MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced"][0]
    context_save = types["MiniMaxH3LongVideoContextSaveT8"][0]
    selected_decode = max(types["MiniMaxH3AVDecodeT8"], key=lambda item: item["id"])
    context_link = next(
        link for link in workflow["links"] if link[0] == context_save["inputs"][0]["link"]
    )
    delivery_link = next(
        link for link in workflow["links"] if link[0] == selected_decode["inputs"][0]["link"]
    )
    assert context_link[1:3] == [split["id"], 0]
    assert delivery_link[1:3] == [split["id"], 1]
    assert len(types["MiniMaxH3OutputTrimT8"]) == 2
