"""Inner graph edits must never hide behind an unchanged outer UUID node."""
from copy import deepcopy

import pytest

from tools.audit_modular_embedded_browser import (
    _audit_level, _level, audit_definitions, audit_reopened,
    normalize_boundary_link_ids, normalize_node_collisions, normalize_proxy_promotions,
)
from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.serve_modular_m0_embedded_browser import selected
from tools.serve_modular_m0_legacy_browser import selected as flat_selected
from tools.serve_modular_m0_markdown_browser import selected as markdown_selected


def _graph():
    return {"nodes": [{"id": 1, "type": "bundle", "inputs": [], "outputs": [],
                       "properties": {"proxyWidgets": [["2", "value"]]}}], "links": [],
            "definitions": {"subgraphs": [{
                "id": "bundle", "name": "test", "widgets": [], "config": {},
                "inputNode": {"id": -10}, "outputNode": {"id": -20},
                "inputs": [{"id": "x", "name": "x", "type": "INT", "linkIds": [1]}],
                "outputs": [{"id": "y", "name": "y", "type": "INT", "linkIds": [2]}],
                "nodes": [{"id": 2, "type": "Example", "mode": 0,
                           "inputs": [{"name": "value", "type": "INT", "link": 1}],
                           "outputs": [{"name": "result", "type": "INT", "links": [2]}],
                           "widgets_values": [4], "widgets_values_named": {"value": 4}}],
                "links": [{"id": 1, "origin_id": -10, "origin_slot": 0,
                           "target_id": 2, "target_slot": 0, "type": "INT"},
                          {"id": 2, "origin_id": 2, "origin_slot": 0,
                           "target_id": -20, "target_slot": 0, "type": "INT"}],
            }]}}


def test_embedded_scope_is_eight_additional_frozen_graphs():
    embedded = selected()
    assert len(embedded) == 8
    old = {case["relative"] for case in flat_selected() + markdown_selected()}
    assert not old & {case["relative"] for case in embedded}
    assert all(case["copy_name"].startswith("M0E_") and
               case["saved_name"].startswith("QA_M0E_") for case in embedded)


def test_nested_level_checks_boundary_links_without_changing_the_graph():
    graph = _graph()
    before = deepcopy(graph)
    result = audit_definitions(graph, deepcopy(graph))
    assert result["bundle"]["nodes"] == 3  # Includes the two audit-only boundaries.
    assert result["bundle"]["named_edges"] == 2
    assert result["bundle"]["widgets"] == 1
    assert audit_reopened(graph, deepcopy(graph))["embedded_semantic_audit"] == result
    assert graph == before


@pytest.mark.parametrize("damage", ["widget", "mode", "type", "wire", "pin", "proxy",
                                     "definition", "extra_definition", "contract", "identity"])
def test_saved_or_reopened_subgraph_semantic_drift_is_rejected(damage):
    graph = _graph()
    changed = deepcopy(graph)
    nested = changed["definitions"]["subgraphs"][0]
    node = nested["nodes"][0]
    if damage == "widget":
        node["widgets_values"] = [5]
        node["widgets_values_named"]["value"] = 5
    elif damage == "mode":
        node["mode"] = 4
    elif damage == "type":
        node["type"] = "Other"
    elif damage == "wire":
        nested["links"][0]["target_id"] = -20
    elif damage == "pin":
        nested["outputs"][0]["linkIds"] = [1]
    elif damage == "proxy":
        changed["nodes"][0]["properties"]["proxyWidgets"] = [["2", "wrong"]]
    elif damage == "definition":
        changed["definitions"]["subgraphs"] = []
    elif damage == "extra_definition":
        extra = deepcopy(nested)
        extra["id"] = "other"
        changed["definitions"]["subgraphs"].append(extra)
    elif damage == "contract":
        nested["config"]["different"] = True
    else:
        nested["inputNode"]["id"] = -11
    for check in (audit_reopened, audit_definitions):
        with pytest.raises((ValueError, KeyError)):
            check(graph, changed)


def test_layout_only_changes_do_not_mask_execution_state():
    graph = _graph()
    changed = deepcopy(graph)
    definition = changed["definitions"]["subgraphs"][0]
    definition["nodes"][0]["pos"] = [123, 456]
    definition["inputs"][0]["pos"] = [5, 6]
    definition["inputNode"]["bounding"] = [0, 0, 100, 100]
    assert audit_reopened(graph, changed)["embedded_semantic_audit"]["bundle"]["named_edges"] == 2


def test_reopen_null_presentation_fields_are_the_same_as_omitted_but_values_are_not():
    graph = _graph()
    changed = deepcopy(graph)
    pin = changed["definitions"]["subgraphs"][0]["inputs"][0]
    pin.update(localized_name=None, label=None, dir=None, shape=None,
               color_off=None, color_on=None)
    assert audit_reopened(graph, changed)["embedded_semantic_audit"]
    pin["label"] = "Changed semantic label"
    with pytest.raises(ValueError, match="boundary changed"):
        audit_reopened(graph, changed)


@pytest.mark.parametrize("damage", ["duplicate_pin", "bool_link_id", "two_output_sources"])
def test_boundary_ambiguity_is_rejected_even_when_identical_on_both_sides(damage):
    graph = _graph()
    definition = graph["definitions"]["subgraphs"][0]
    if damage == "duplicate_pin":
        definition["inputs"].append(deepcopy(definition["inputs"][0]))
    elif damage == "bool_link_id":
        definition["links"][0]["id"] = True
    else:
        definition["outputs"][0]["linkIds"] = [1, 2]
    with pytest.raises(ValueError):
        audit_definitions(graph, deepcopy(graph))


def _promoted():
    original = _graph()
    top = original["nodes"][0]
    top["properties"]["proxyWidgets"] = [["2", "noise_seed"]]
    top["widgets_values"] = []
    top["inputs"] = [{"name": "x", "type": "INT", "link": None, "widget": {"name": "x"}}]
    inner = original["definitions"]["subgraphs"][0]["nodes"][0]
    inner["type"] = "RandomNoise"
    inner["inputs"][0].update(name="noise_seed", widget={"name": "noise_seed"})
    inner["widgets_values"] = [4, "fixed"]
    inner["widgets_values_named"] = {"noise_seed": 4, "control_after_generate": "fixed"}
    saved = deepcopy(original)
    actual = saved["nodes"][0]
    actual["properties"].pop("proxyWidgets")
    actual["widgets_values"] = [4]
    actual["widgets_values_named"] = {"x": 4}
    return original, saved


def test_promoted_values_are_derived_from_inner_layout_without_mutation():
    original, saved = _promoted()
    frozen = deepcopy(original)
    normalized, record = normalize_proxy_promotions(original, saved)
    assert original == frozen
    assert normalized["nodes"][0]["widgets_values"] == [4]
    assert record == {"1": [{"name": "x", "target": ["2", "noise_seed"], "widget_index": 0}]}
    assert audit_roundtrip(normalized, saved)["widgets"] == 1
    assert audit_definitions(normalized, saved)


@pytest.mark.parametrize("damage", ["value", "bool_value", "name", "target", "count",
                                     "widget", "retained_proxy", "quarantine", "source_values"])
def test_proxy_migration_never_accepts_value_or_binding_loss(damage):
    original, saved = _promoted()
    top = saved["nodes"][0]
    if damage in ("value", "bool_value"):
        top["widgets_values"] = [True if damage == "bool_value" else 5]
        top["widgets_values_named"]["x"] = top["widgets_values"][0]
    elif damage == "name":
        top["widgets_values_named"] = {"wrong": 4}
    elif damage == "target":
        original["nodes"][0]["properties"]["proxyWidgets"][0][1] = "wrong"
    elif damage == "count":
        original["nodes"][0]["properties"]["proxyWidgets"].append(["2", "noise_seed"])
    elif damage == "widget":
        top["inputs"][0]["widget"]["name"] = "wrong"
    elif damage == "retained_proxy":
        top["properties"]["proxyWidgets"] = []
    elif damage == "quarantine":
        top["properties"]["proxyWidgetErrorQuarantine"] = [{"reason": "unresolved"}]
    else:
        original["nodes"][0]["widgets_values"] = [5]
    with pytest.raises(ValueError):
        normalize_proxy_promotions(original, saved)


def _renumber_link(definition, old_id, new_id):
    for link in definition["links"]:
        if link["id"] == old_id:
            link["id"] = new_id
    for direction in ("inputs", "outputs"):
        for pin in definition[direction]:
            pin["linkIds"] = [new_id if value == old_id else value for value in pin["linkIds"]]
    for node in definition["nodes"]:
        for pin in node.get("inputs", []):
            if pin.get("link") == old_id:
                pin["link"] = new_id
        for pin in node.get("outputs", []):
            pin["links"] = [new_id if value == old_id else value for value in pin["links"]]


def test_boundary_collision_repair_preserves_all_edge_ownership():
    old = _graph()["definitions"]["subgraphs"][0]
    new = deepcopy(old)
    _renumber_link(new, 1, 3)
    normalized, record = normalize_boundary_link_ids(old, new, {1})
    assert record == {"1": 3}
    assert normalized == old
    assert audit_roundtrip(_level(old), _level(normalized))["named_edges"] == 2
    assert new["links"][0]["id"] == 3


@pytest.mark.parametrize("damage", ["noncollision", "internal", "wrong_target", "ownership"])
def test_arbitrary_link_id_or_edge_changes_are_not_collision_repairs(damage):
    old = _graph()["definitions"]["subgraphs"][0]
    new = deepcopy(old)
    _renumber_link(new, 2 if damage == "internal" else 1, 3)
    if damage == "wrong_target":
        new["links"][0]["target_id"] = -20
    if damage == "ownership":
        new["nodes"][0]["inputs"][0]["link"] = 1
    with pytest.raises((ValueError, IndexError)):
        normalize_boundary_link_ids(old, new, {9} if damage == "noncollision" else {1, 2})


def test_node_collision_mapping_cannot_mask_changed_widgets_or_edges():
    old = _graph()["definitions"]["subgraphs"][0]
    new = deepcopy(old)
    new["nodes"][0]["id"] = 3
    new["links"][0]["target_id"] = 3
    new["links"][1]["origin_id"] = 3
    normalized, record = normalize_node_collisions(old, new, {2})
    assert normalized == old and record == {"2": 3}
    with pytest.raises(ValueError, match="allocation"):
        normalize_node_collisions(old, new, {1})
    new["nodes"][0]["widgets_values"] = [5]
    normalized, _ = normalize_node_collisions(old, new, {2})
    with pytest.raises(ValueError, match="widgets"):
        audit_roundtrip(_level(old), _level(normalized))
    new["links"][0]["target_id"] = -20
    normalized, _ = normalize_node_collisions(old, new, {2})
    with pytest.raises(ValueError):
        audit_roundtrip(_level(old), _level(normalized))


@pytest.mark.parametrize("kind,name,old_type,new_type,spec,value", [
    ("CreateVideo", "bit_depth", "INT", "COMBO", [[8, 10], {}], 8),
    ("SaveVideo", "format", "COMBO", "COMFY_DYNAMICCOMBO_V3", ["COMFY_DYNAMICCOMBO_V3", {}], "mp4"),
])
def test_core_socket_migration_requires_unconnected_same_widget_and_schema(
        kind, name, old_type, new_type, spec, value):
    old = {"nodes": [{"id": 1, "type": kind, "outputs": [],
                      "inputs": [{"name": name, "type": old_type, "link": None,
                                  "widget": {"name": name}}],
                      "widgets_values": [value], "widgets_values_named": {name: value}}], "links": []}
    new = deepcopy(old)
    new["nodes"][0]["inputs"][0]["type"] = new_type
    current = {kind: {"info": {"input": {"required": {name: spec}}}}}
    assert _audit_level(old, new, current)["disconnected_core_widget_socket_changes"]
    new["nodes"][0]["inputs"][0]["link"] = 5
    with pytest.raises(ValueError, match="socket migration"):
        _audit_level(old, new, current)
