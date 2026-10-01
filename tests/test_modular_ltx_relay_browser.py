"""The new canvas audit targets exactly the additive S27 Relay controls."""

from copy import deepcopy

import pytest

from tools import build_modular_ltx_relay_workflows as relay
from tools import run_modular_ltx_relay_browser as browser


@pytest.mark.parametrize("name", sorted(relay.generated()))
def test_each_relay_graph_has_visible_plan_and_effect_controls(name):
    graph = relay.generated()[name]
    actions, changes = browser._edits(graph, "S27R_" + name + ".json")
    assert len(actions) == (3 if name.endswith("_external_eav_relay") else 2)
    assert len(changes) == len(actions)
    assert actions[0]["kind"] == "relay"
    assert all(action["kind"] == "mode" for action in actions[1:])
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[actions[0]["node_id"]]["type"] == relay.PLAN
    assert nodes[actions[1]["node_id"]]["type"] == relay.APPLY
    assert actions[0]["marker"] in changes[(actions[0]["node_id"], 0)]
    if len(actions) == 3:
        assert nodes[actions[2]["node_id"]]["type"] == "MiniMaxH3StageEAVConfigEXPT8"


def test_browser_audit_rejects_changed_default_mode():
    name, graph = next(iter(relay.generated().items()))
    node = next(node for node in graph["nodes"] if node["type"] == relay.APPLY)
    node["widgets_values"][0] = "apply_exp"
    with pytest.raises(ValueError, match="controls changed"):
        browser._edits(graph, "S27R_" + name + ".json")


def test_browser_linked_report_placeholder_allowance_is_exact():
    graph = next(iter(relay.generated().values()))
    saved = deepcopy(graph)
    node = next(node for node in saved["nodes"]
                if node["type"] == "MiniMaxH3LTXRGBStageBindEXPT8")
    node["widgets_values"] = ["", ""]
    node["widgets_values_named"] = {"prep_report_json": "", "setup_report_json": ""}
    schema = {node["type"]: {"info": {"input": {"required": {
        "prep_report_json": ["STRING"], "setup_report_json": ["STRING"]}}}}}
    assert browser._schema_default_appends(graph, saved, schema) == {node["id"]: ["", ""]}
    node["widgets_values"][1] = "not an empty display default"
    with pytest.raises(ValueError, match="display widgets changed"):
        browser._schema_default_appends(graph, saved, schema)
