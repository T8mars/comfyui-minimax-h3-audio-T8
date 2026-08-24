from __future__ import annotations

import json
from pathlib import Path

from h3_audio_t8_pkg.prompt_budget_advanced import compile_prompt_budget


def _args(**overrides):
    values = {
        "prompt": "<Picture 1>中的主角听见<Audio 1>后转身。",
        "character_limit": 7000,
        "token_limit": 0,
        "picture_count": 1,
        "video_count": 0,
        "audio_count": 1,
        "media_assignments_json": json.dumps(
            {
                "subjects": [
                    {
                        "subject_id": "lead",
                        "picture_ordinal": 1,
                        "audio_ordinal": 1,
                        "role": "primary_character",
                    }
                ]
            }
        ),
        "append_role_bindings": True,
        "allow_shared_audio": False,
        "require_exact_token_count": False,
        "clip": None,
    }
    values.update(overrides)
    return values


class _Clip:
    def tokenize(self, _text):
        return {"qwen": [[(index, 1.0) for index in range(37)]]}


def test_prompt_budget_compiles_roles_and_never_truncates():
    compiled, passed, decision, chars, estimate, exact, media_json, report_json = (
        compile_prompt_budget(**_args())
    )
    report = json.loads(report_json)
    media = json.loads(media_json)
    assert passed is True and decision == "PASS"
    assert "Subject lead" in compiled
    assert "<Picture 1>" in compiled and "<Audio 1>" in compiled
    assert chars == len(compiled) and estimate > 0 and exact == -1
    assert report["prompt_truncated"] is False
    assert media["role_bindings_appended"] is True


def test_connected_clip_count_is_labelled_exact():
    result = compile_prompt_budget(**_args(clip=_Clip(), require_exact_token_count=True))
    report = json.loads(result[-1])
    assert result[1] is True
    assert result[5] == 37
    assert report["token_count_method_for_limit"] == "connected_clip_tokenizer"
    assert report["token_count_scope"] == "compiled_text_only_without_connected_visual_embeddings"


def test_budget_overflow_abstains_without_shortening_prompt():
    prompt = "这是一段不会被静默截断的提示词" * 10
    result = compile_prompt_budget(
        **_args(
            prompt=prompt,
            character_limit=20,
            media_assignments_json='{"subjects":[]}',
            append_role_bindings=False,
        )
    )
    compiled, passed, decision = result[:3]
    codes = {item["code"] for item in json.loads(result[-1])["findings"]}
    assert compiled == prompt
    assert passed is False and decision == "ABSTAIN"
    assert "character_budget_exceeded" in codes


def test_character_guard_accepts_7000_and_abstains_at_7001_without_mutation():
    at_limit = "界" * 7000
    over_limit = at_limit + "外"
    common = {
        "media_assignments_json": '{"subjects":[]}',
        "append_role_bindings": False,
        "picture_count": 0,
        "audio_count": 0,
    }
    passed = compile_prompt_budget(**_args(prompt=at_limit, **common))
    failed = compile_prompt_budget(**_args(prompt=over_limit, **common))
    assert passed[0] == at_limit and passed[1:3] == (True, "PASS")
    assert failed[0] == over_limit and failed[1:3] == (False, "ABSTAIN")
    report = json.loads(failed[-1])
    assert report["character_count_unicode_codepoints"] == 7001
    assert report["prompt_truncated"] is False
    assert report["character_limit_semantics"] == "official_h3_submission_compatibility_ceiling"
    assert report["official_h3_submission_character_limit"] == 7000
    assert report["official_h3_submission_compatible"] is False
    assert report["local_tokenizer_runtime_boundary_probed"] is False


def test_character_limit_override_is_explicit_and_does_not_claim_official_compatibility():
    prompt = "界" * 7001
    result = compile_prompt_budget(
        **_args(
            prompt=prompt,
            character_limit=8000,
            media_assignments_json='{"subjects":[]}',
            append_role_bindings=False,
            picture_count=0,
            audio_count=0,
        )
    )
    report = json.loads(result[-1])
    warning = next(
        item
        for item in report["warnings"]
        if item["code"] == "configured_character_limit_exceeds_official_h3_submission_limit"
    )
    assert result[0] == prompt and result[1:3] == (True, "PASS")
    assert report["character_limit_semantics"] == "user_configured_audit_ceiling"
    assert report["official_h3_submission_compatible"] is False
    assert warning["configured_limit"] == 8000
    assert warning["official_submission_limit"] == 7000


def test_source_whitespace_is_preserved_while_role_bindings_are_appended():
    prompt = "  <Picture 1>中的人物保持安静。\n"
    result = compile_prompt_budget(**_args(prompt=prompt))
    assert result[0].startswith(prompt)
    assert "Reference role bindings:" in result[0]
    assert result[1] is True


def test_three_subject_multimedia_contract_has_complete_coverage():
    assignments = {
        "subjects": [
            {
                "subject_id": "lead",
                "picture_ordinal": 1,
                "video_ordinal": 1,
                "audio_ordinal": 1,
                "role": "lead_actor",
            },
            {
                "subject_id": "partner",
                "picture_ordinal": 2,
                "video_ordinal": 2,
                "audio_ordinal": 2,
                "role": "dialogue_partner",
            },
            {
                "subject_id": "narrator",
                "picture_ordinal": 3,
                "audio_ordinal": 3,
                "role": "narrator",
            },
        ]
    }
    prompt = (
        "<Picture 1>与<Picture 2>沿用<Video 1>和<Video 2>的动作；"
        "<Picture 3>只负责旁白。对白依次参考<Audio 1>、<Audio 2>与<Audio 3>。"
    )
    result = compile_prompt_budget(
        **_args(
            prompt=prompt,
            picture_count=3,
            video_count=2,
            audio_count=3,
            media_assignments_json=json.dumps(assignments),
        )
    )
    media = json.loads(result[-2])
    report = json.loads(result[-1])
    assert result[1:3] == (True, "PASS")
    assert media["assigned_ordinals"] == {
        "Picture": [1, 2, 3],
        "Video": [1, 2],
        "Audio": [1, 2, 3],
    }
    assert media["unassigned_connected_ordinals"] == {
        "Picture": [],
        "Video": [],
        "Audio": [],
    }
    coverage_codes = {
        item["code"]
        for item in report["warnings"]
        if "assignment" in item["code"]
    }
    assert coverage_codes == set()


def test_unassigned_connected_media_and_nonappended_roles_are_explicit_warnings():
    assignments = {
        "subjects": [
            {
                "subject_id": "lead",
                "picture_ordinal": 1,
                "audio_ordinal": 1,
                "role": "   ",
            }
        ]
    }
    result = compile_prompt_budget(
        **_args(
            picture_count=2,
            audio_count=2,
            media_assignments_json=json.dumps(assignments),
            append_role_bindings=False,
        )
    )
    media = json.loads(result[-2])
    warnings = json.loads(result[-1])["warnings"]
    codes = [item["code"] for item in warnings]
    assert result[1:3] == (True, "PASS")
    assert media["assignments"][0]["role"] == "subject_reference"
    assert media["unassigned_connected_ordinals"]["Picture"] == [2]
    assert media["unassigned_connected_ordinals"]["Audio"] == [2]
    assert "role_bindings_not_appended_to_prompt" in codes
    assert codes.count("connected_media_without_subject_assignment") == 2


def test_explicit_shared_audio_can_pass_but_remains_visible_in_assignments():
    assignments = {
        "subjects": [
            {"subject_id": "a", "picture_ordinal": 1, "audio_ordinal": 1},
            {"subject_id": "b", "picture_ordinal": 2, "audio_ordinal": 1},
        ]
    }
    result = compile_prompt_budget(
        **_args(
            prompt="<Picture 1>和<Picture 2>共同参考<Audio 1>。",
            picture_count=2,
            audio_count=1,
            media_assignments_json=json.dumps(assignments),
            allow_shared_audio=True,
        )
    )
    media = json.loads(result[-2])
    assert result[1:3] == (True, "PASS")
    assert [item["audio_ordinal"] for item in media["assignments"]] == [1, 1]


def test_out_of_range_and_shared_audio_assignments_abstain():
    assignments = {
        "subjects": [
            {"subject_id": "a", "picture_ordinal": 2, "audio_ordinal": 1},
            {"subject_id": "b", "picture_ordinal": 1, "audio_ordinal": 1},
        ]
    }
    result = compile_prompt_budget(
        **_args(media_assignments_json=json.dumps(assignments), append_role_bindings=True)
    )
    codes = {item["code"] for item in json.loads(result[-1])["findings"]}
    assert result[1] is False and result[2] == "ABSTAIN"
    assert "assignment_media_ordinal_out_of_range" in codes
    assert "audio_reference_assigned_to_multiple_subjects" in codes


def test_dangling_prompt_media_tag_is_reported_not_silently_removed():
    result = compile_prompt_budget(
        **_args(
            prompt="Use <Picture 2> as the character.",
            media_assignments_json='{"subjects":[]}',
            append_role_bindings=False,
        )
    )
    codes = {item["code"] for item in json.loads(result[-1])["findings"]}
    assert result[1] is False
    assert "prompt_media_tag_out_of_range" in codes
    assert "Picture 2" in result[0]


def test_prompt_budget_frontend_workflow_is_importable_and_documented():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "14-prompt-relay"
        / "2026-08-22_H3_Prompt_Budget_Role_Compiler_Advanced.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    compiler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3PromptBudgetCompilerT8Advanced"
    )
    assert compiler["widgets_values"][1:6] == [7000, 0, 1, 1, 1]
    assert compiler["widgets_values"][-3:] == [True, False, False]
    assert "MarkdownNote" in {node["type"] for node in nodes.values()}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == 0
    assert workflow["links"] == []
