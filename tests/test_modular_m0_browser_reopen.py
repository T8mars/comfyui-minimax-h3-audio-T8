"""Reopen proof compares actual widget values, including VHS named storage."""

from copy import deepcopy

import pytest

from tools.run_modular_m0_legacy_browser import _audit_reopened


def _graph():
    values = {"crf": 19, "save_metadata": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}}
    return {"nodes": [{"id": 1, "type": "VHS_VideoCombine", "inputs": [], "outputs": [],
                       "widgets_values": values, "widgets_values_named": deepcopy(values)}],
            "links": []}


def test_vhs_named_reopen_is_exact_and_does_not_mutate_inputs():
    graph = _graph()
    before = deepcopy(graph)
    assert _audit_reopened(graph, deepcopy(graph))["widgets"] == 3
    assert graph == before


@pytest.mark.parametrize("damage", ["value", "bool", "name", "extra", "type"])
def test_vhs_reopen_refuses_damaged_values_or_names(damage):
    graph = _graph()
    latest = deepcopy(graph)
    node = latest["nodes"][0]
    if damage == "value":
        node["widgets_values"]["crf"] = 20
        node["widgets_values_named"]["crf"] = 20
    elif damage == "bool":
        node["widgets_values"]["save_metadata"] = 1
        node["widgets_values_named"]["save_metadata"] = 1
    elif damage == "name":
        node["widgets_values_named"]["wrong"] = node["widgets_values_named"].pop("crf")
    elif damage == "extra":
        node["widgets_values"]["extra"] = ""
    else:
        node["type"] = "Other"
    with pytest.raises(ValueError):
        _audit_reopened(graph, latest)
