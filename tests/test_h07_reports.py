import copy
import json

import pytest
import torch

from h3_t8.h07_reports import reference_summary, continuity_lint
from h3_t8.prompt_tags import media_map_json


def test_actual_numbering_with_video_audio_and_same_role_multiview():
    media = media_map_json(["ref_image_0", "ref_image_1"], ["ref_video_0"],
                           ["ref_video_0 soundtrack", "ref_audio_0"], 0)
    roles = [{"role_id": "A", "tags": ["Picture 1", "Picture 2", "Audio 2"],
              "expected_labels": {"Audio 2": "ref_audio_0"}}]
    raw = json.dumps(roles)
    sigmas = torch.tensor([1., .8, .2, 0.])
    before = sigmas.clone()
    report = reference_summary(media, raw, sigmas=sigmas)
    entries = {row["tag"]: row for row in report["references"]}
    assert entries["Audio 1"]["role_bindings"] == []
    assert entries["Audio 2"]["role_bindings"][0]["role_id"] == "A"
    assert entries["Picture 1"]["Subject"] == "unknown"
    assert report["SIGMAS"]["interval_count"] == 3
    assert report["completion"]["actual_NFE"] == "unknown"
    assert torch.equal(sigmas, before) and json.loads(raw) == roles
    wrong = reference_summary(media, json.dumps([{**roles[0], "expected_labels": {"Audio 2": "wrong"}}]))
    assert wrong["warnings"][0]["issue"] == "expected_source_label_differs"


def test_prop_identity_kind_hands_and_explicit_transfer():
    states = [{"t": 0, "props": [dict(id="cup_a", kind="cup", holder="A", hand="left"),
                                  dict(id="cup_b", kind="cup", holder="A", hand="right")]},
              {"t": 1, "props": [dict(id="cup_a", kind="cup", holder="B", hand="left"),
                                  dict(id="cup_b", kind="cup", holder="A", hand="right")]}]
    before = copy.deepcopy(states)
    allowed = [{"t": 1, "id": "cup_a", "from": "A", "to": "B"}]
    assert continuity_lint(json.dumps(states), json.dumps(allowed))["warnings"] == []
    assert states == before
    assert continuity_lint(json.dumps(states))["warnings"][0]["issue"] == "holder_transfer_not_explicitly_allowed"
    states[1]["props"].append(dict(id="bottle", kind="bottle", holder="B", hand="left"))
    states[1]["props"][0]["kind"] = "glass"
    issues = {item["issue"] for item in continuity_lint(json.dumps(states), json.dumps(allowed))["warnings"]}
    assert issues == {"same_hand_multiple_props", "kind_changed_for_stable_ID"}


def test_nonfinite_or_ambiguous_declarations_are_not_silently_rewritten():
    with pytest.raises(ValueError, match="Nonfinite"):
        continuity_lint('[{"t":NaN,"props":[]}]')
    with pytest.raises(ValueError, match="contiguous"):
        reference_summary('{"audios":{"2":"ref_audio_0"}}')
    with pytest.raises(ValueError, match="strictly increasing"):
        continuity_lint('[{"t":0,"props":[]},{"t":0,"props":[]}]')
