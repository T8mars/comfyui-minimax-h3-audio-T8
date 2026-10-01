"""Fail-closed source selection for S14/S15 native-browser QA."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.run_modular_hyperflow_fresh_browser_matrix import (
    CANDIDATE,
    EDIT_NAMES,
    _selected_edits,
)


def test_matrix_sources_are_exactly_two_existing_fresh_routes():
    sources = tuple(CANDIDATE.glob("HyperFlow_*_EXP.json"))
    assert len(sources) == 62
    assert len({source.name for source in sources}) == len(sources)
    assert sum("full8plus4" in source.name for source in sources) == 31
    assert sum("partial4plus4" in source.name for source in sources) == 31
    assert EDIT_NAMES <= {source.name for source in sources}


@pytest.mark.parametrize("name", sorted(EDIT_NAMES))
def test_four_paired_cases_require_independent_stage_edits(name):
    graph = json.loads((CANDIDATE / name).read_bytes())
    actions, edits = _selected_edits(graph, name)
    count = 1 if "resume_high" in name else 2
    assert len([action for action in actions if action["kind"] == "relay"]) == count
    assert len([action for action in actions if action["kind"] == "eav"]) == 1
    assert len(edits) == count + 1
    assert len({action["node_id"] for action in actions}) == len(actions)
    assert set(edits.values()) & {"apply_exp"} == {"apply_exp"}
    if count == 2:
        assert "Independent low stage prompt." in next(
            action["before"] for action in actions if "low stage" in action["before"])
        assert "Independent high stage prompt." in next(
            action["before"] for action in actions if "high stage" in action["before"])


def test_qa_rejects_missing_or_preapplied_high_effect():
    name = "HyperFlow_full8plus4_combined_both_save_EXP.json"
    original = json.loads((CANDIDATE / name).read_bytes())
    wrong_mode = deepcopy(original)
    for node in wrong_mode["nodes"]:
        if node["type"] == "MiniMaxH3StageEAVConfigEXPT8" and node["id"] == 25:
            node["widgets_values"][0] = "apply_exp"
    with pytest.raises(ValueError, match="original HIGH EAV mode"):
        _selected_edits(wrong_mode, name)

    missing_plan = deepcopy(original)
    missing_plan["nodes"] = [node for node in missing_plan["nodes"] if node["id"] != 23]
    with pytest.raises(ValueError, match="independent Relay/EAV node count"):
        _selected_edits(missing_plan, name)
