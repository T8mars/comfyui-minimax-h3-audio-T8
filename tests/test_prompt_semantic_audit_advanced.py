from __future__ import annotations

import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.nodes_prompt_semantic_audit_advanced import (
    MiniMaxH3PromptSemanticContractAuditT8Advanced,
)
from h3_audio_t8_pkg.prompt_semantic_audit_advanced import audit_prompt_semantics


SOURCE = (
    "A woman in a red Hanfu turns beneath moonlight and says "
    "<d>[Chinese] 你在干嘛呢，我在这里呀。</d>"
)
BAD_CANDIDATE = (
    "integrated_multimodal_description: [Shot 1] A woman in a red Hanfu "
    "stands still beneath moonlight and says "
    "<d>[Chinese] 你在干嘛呢，我在这里呀。</d>\n"
    "overall_soundscape: Quiet wind and soft cloth movement.\n"
    "non_diegetic_music: N/A"
)
GOOD_CANDIDATE = BAD_CANDIDATE.replace("stands still", "rotates slowly")


def _contract(*, required=True, forbidden=True):
    payload = {
        "required_groups": (
            [
                {
                    "id": "turn_motion",
                    "any_of": ["turns", "turning", "rotates", "spins", "转身", "旋转"],
                    "scope": "integrated",
                }
            ]
            if required
            else []
        ),
        "forbidden_groups": (
            [
                {
                    "id": "stillness",
                    "any_of": ["stands still", "motionless", "静止不动"],
                    "scope": "integrated",
                }
            ]
            if forbidden
            else []
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _run(candidate=GOOD_CANDIDATE, contract=None, accept=False, **overrides):
    values = {
        "original_prompt": SOURCE,
        "candidate_prompt": candidate,
        "semantic_contract_json": _contract() if contract is None else contract,
        "accept_candidate_after_review": accept,
        "preserve_exact_dialogue": True,
        "preserve_source_media_tags": True,
        "allow_new_media_tags": True,
    }
    values.update(overrides)
    return audit_prompt_semantics(**values)


def test_known_real_provider_semantic_regression_is_rejected():
    safe, candidate, passed, decision, report_json = _run(candidate=BAD_CANDIDATE)
    report = json.loads(report_json)
    assert safe == SOURCE
    assert candidate == BAD_CANDIDATE
    assert passed is False
    assert decision == "REJECT"
    assert {finding["code"] for finding in report["findings"]} >= {
        "required_group_missing",
        "forbidden_group_present",
    }
    assert report["safe_prompt_source"] == "original"


def test_mechanical_pass_still_requires_explicit_human_acceptance():
    review = _run()
    assert review[:4] == (SOURCE, GOOD_CANDIDATE, True, "REVIEW_REQUIRED")
    review_report = json.loads(review[-1])
    assert review_report["mechanical_pass"] is True
    assert review_report["safe_prompt_source"] == "original"

    accepted = _run(accept=True)
    assert accepted[:4] == (GOOD_CANDIDATE, GOOD_CANDIDATE, True, "ACCEPTED")
    assert json.loads(accepted[-1])["safe_prompt_source"] == "candidate"


def test_empty_anchor_contract_abstains_even_when_accept_is_enabled():
    result = _run(
        contract=json.dumps({"required_groups": [], "forbidden_groups": []}),
        accept=True,
    )
    assert result[:4] == (
        SOURCE,
        GOOD_CANDIDATE,
        False,
        "ABSTAIN_NO_SEMANTIC_ANCHORS",
    )


def test_exact_dialogue_and_source_media_tags_are_fail_closed():
    source = "Use <Picture 1> and <Video 2>; she says <d>[English] Go.</d>"
    candidate = (
        "integrated_multimodal_description: [Shot 1] Use <Picture 1>; she says "
        "<d>[English] Stop.</d>\n"
        "overall_soundscape: N/A\n"
        "non_diegetic_music: N/A"
    )
    result = audit_prompt_semantics(
        original_prompt=source,
        candidate_prompt=candidate,
        semantic_contract_json=json.dumps(
            {
                "required_groups": [
                    {"id": "picture", "any_of": ["Picture 1"], "scope": "full"}
                ],
                "forbidden_groups": [],
            }
        ),
        accept_candidate_after_review=True,
        preserve_exact_dialogue=True,
        preserve_source_media_tags=True,
        allow_new_media_tags=True,
    )
    report = json.loads(result[-1])
    assert result[0] == source
    assert result[2:4] == (False, "REJECT")
    assert {finding["code"] for finding in report["findings"]} >= {
        "exact_dialogue_mismatch",
        "source_media_tags_missing",
    }


def test_new_media_tags_are_allowed_by_default_and_optionally_rejected():
    source = "A dancer turns."
    candidate = (
        "integrated_multimodal_description: [Shot 1] A dancer turns beside <Picture 1>.\n"
        "overall_soundscape: N/A\n"
        "non_diegetic_music: N/A"
    )
    contract = json.dumps(
        {
            "required_groups": [
                {"id": "turn", "any_of": ["turns"], "scope": "integrated"}
            ],
            "forbidden_groups": [],
        }
    )
    allowed = audit_prompt_semantics(
        original_prompt=source,
        candidate_prompt=candidate,
        semantic_contract_json=contract,
        accept_candidate_after_review=True,
        allow_new_media_tags=True,
    )
    assert allowed[2:4] == (True, "ACCEPTED")

    rejected = audit_prompt_semantics(
        original_prompt=source,
        candidate_prompt=candidate,
        semantic_contract_json=contract,
        accept_candidate_after_review=True,
        allow_new_media_tags=False,
    )
    assert rejected[0] == source
    assert rejected[2:4] == (False, "REJECT")
    assert "new_media_tags_present" in {
        finding["code"] for finding in json.loads(rejected[-1])["findings"]
    }


def test_unicode_normalization_cjk_substring_and_latin_boundaries():
    fullwidth = GOOD_CANDIDATE.replace("rotates slowly", "ＴＵＲＮＳ并旋转")
    contract = json.dumps(
        {
            "required_groups": [
                {"id": "latin", "any_of": ["turns"], "scope": "integrated"},
                {"id": "cjk", "any_of": ["旋转"], "scope": "integrated"},
            ],
            "forbidden_groups": [],
        },
        ensure_ascii=False,
    )
    assert _run(candidate=fullwidth, contract=contract, accept=True)[2:4] == (
        True,
        "ACCEPTED",
    )

    return_only = GOOD_CANDIDATE.replace("rotates slowly", "returns slowly")
    boundary_contract = json.dumps(
        {
            "required_groups": [
                {"id": "turn", "any_of": ["turn"], "scope": "integrated"}
            ],
            "forbidden_groups": [],
        }
    )
    assert _run(candidate=return_only, contract=boundary_contract)[2:4] == (
        False,
        "REJECT",
    )


@pytest.mark.parametrize(
    "contract",
    [
        "{not-json",
        json.dumps({"required_groups": [], "unknown": []}),
        json.dumps(
            {
                "required_groups": [
                    {"id": "same", "any_of": ["turns"], "scope": "full"}
                ],
                "forbidden_groups": [
                    {"id": "same", "any_of": ["still"], "scope": "full"}
                ],
            }
        ),
        json.dumps(
            {
                "required_groups": [
                    {"id": "bad", "any_of": [], "scope": "integrated"}
                ]
            }
        ),
    ],
)
def test_invalid_contracts_reject_without_forwarding_candidate(contract):
    result = _run(contract=contract, accept=True)
    report = json.loads(result[-1])
    assert result[0] == SOURCE
    assert result[2:4] == (False, "REJECT")
    assert report["contract_valid"] is False
    assert report["findings"][0]["code"] == "invalid_semantic_contract"


def test_non_full_scope_requires_structured_candidate_fields():
    result = _run(candidate="The dancer rotates slowly.")
    report = json.loads(result[-1])
    assert result[2:4] == (False, "REJECT")
    assert "candidate_structure_warning" in {
        finding["code"] for finding in report["findings"]
    }


def test_node_schema_is_safe_and_frontend_workflow_is_documented():
    schema = MiniMaxH3PromptSemanticContractAuditT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["accept_candidate_after_review"].default is False
    assert inputs["preserve_exact_dialogue"].default is True
    assert inputs["preserve_source_media_tags"].default is True
    assert inputs["allow_new_media_tags"].default is True

    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "14-prompt-relay"
        / "2026-08-23_H3_Prompt_Semantic_Contract_Audit_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = [node["type"] for node in workflow["nodes"]]
    assert "PrimitiveStringMultiline" in types
    assert "MiniMaxH3PromptProviderRouterT8Advanced" in types
    assert "MiniMaxH3PromptSemanticContractAuditT8Advanced" in types
    assert types.count("MarkdownNote") >= 3
    audit = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3PromptSemanticContractAuditT8Advanced"
    )
    assert audit["widgets_values"][-4:] == [False, True, True, True]
