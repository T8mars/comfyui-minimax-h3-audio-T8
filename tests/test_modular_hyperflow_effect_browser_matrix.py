"""Fail-closed source selection for S13 native-browser QA."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.run_modular_hyperflow_effect_browser_matrix import (
    CANDIDATE,
    EDIT_NAMES,
    _selected_edits,
)


def test_matrix_has_all_three_continuous_splits():
    sources = tuple(CANDIDATE.glob("HyperFlow_*_EXP.json"))
    assert len(sources) == 81
    assert len({source.name for source in sources}) == len(sources)
    for split in ("1plus7", "4plus4", "7plus1"):
        assert sum(split in source.name for source in sources) == 27
    assert EDIT_NAMES <= {source.name for source in sources}


@pytest.mark.parametrize("name", sorted(EDIT_NAMES))
def test_six_paired_cases_require_independent_relay_and_eav(name):
    graph = json.loads((CANDIDATE / name).read_bytes())
    actions, edits = _selected_edits(graph, name)
    expected = 1 if "resume_tail" in name else 2
    assert len([action for action in actions if action["kind"] == "relay"]) == expected
    assert len([action for action in actions if action["kind"] == "eav"]) == 1
    assert len(edits) == expected + 1
    assert len({action["node_id"] for action in actions}) == len(actions)
    assert list(edits.values()).count("apply_exp") == 1


def test_qa_rejects_missing_tail_effect_or_plan():
    name = "HyperFlow_4plus4_combined_both_save_EXP.json"
    original = json.loads((CANDIDATE / name).read_bytes())
    wrong_mode = deepcopy(original)
    for node in wrong_mode["nodes"]:
        if node["id"] == 25:
            node["widgets_values"][0] = "apply_exp"
    with pytest.raises(ValueError, match="original TAIL EAV mode"):
        _selected_edits(wrong_mode, name)

    missing_plan = deepcopy(original)
    missing_plan["nodes"] = [node for node in missing_plan["nodes"] if node["id"] != 23]
    with pytest.raises(ValueError, match="independent Relay/EAV node count"):
        _selected_edits(missing_plan, name)
